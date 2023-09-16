import bisect
import itertools
import json
from typing import List, Tuple, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.components.entity_manager import prepare_document_containers, BertCandidateManager
from src.evaluation.second_stage_clustering import HierarchicalTwoStageClustering
from src.utilities.utils import get_context_entities, preprocess_and_split_text, transform_item
from src.evaluation.evaluation_tools import calculate_final_results
from src.model.embedding_model import BertDocumentEmbedder
from src.model.mention_comparison import MentionComparator
from src.utilities.utilities import (
    update_current_representations,
    calculate_similarities, get_nested_candidates, )
from src.utilities.various_dataclasses import Result, Example, CandidateContainerWrapper


class Evaluator:
    def __init__(
        self,
        document_tokenizer,
        candidate_manager,
        device,
        kg_connector,
        args: dict,
        num_types: int = 1,
        filter_set=None,
    ):
        self.candidate_manager: BertCandidateManager = candidate_manager
        self.document_tokenizer = document_tokenizer
        self.device = device
        self.num_types = num_types
        self.filter_set = filter_set
        self.kg_connector = kg_connector
        self.args = args

    def pipeline(self, texts: List[dict], ranking_model, mention_comparator):
        texts = self.transform_to_examples(texts)
        results = sum(self.link(texts, ranking_model, mention_comparator, 1, 1, 50, 10), [])
        clusterer = HierarchicalTwoStageClustering(self.args, ranking_model, mention_comparator, self.candidate_manager, self.device)
        final_results = clusterer.cluster(results, [0.8])
        return final_results

    def transform_to_examples(self, texts: List[dict]) -> List[Example]:
        texts = [transform_item(x) for x in texts]
        examples = preprocess_and_split_text(texts, self.document_tokenizer, 512, 0, self.kg_connector,
                                             self.candidate_manager.transe_mappings,
                                             self.candidate_manager.transe_embeddings, 0)
        return examples

    def link(self, examples: List[Example], ranking_model, mention_comparator, mention_split: int, mentions_to_add_if_full: int,
             number_candidates: int, num_beams: int):
        return self.predict_examples(examples, ranking_model, mention_comparator,
                                        mention_split, mentions_to_add_if_full, number_candidates,
                                        add_correct_candidate=False,
                                        spoof_perfect_linking=False, num_beams=num_beams)

    def evaluate_model(
        self,
        dataset: DataLoader,
        document_model: BertDocumentEmbedder,
        entity_model,
        ranking_model,
        mention_comparator_model,
        mention_split, mentions_to_add_if_full,
        second_stage_clustering=False,
        detailed_info=False,
        spoof_perfect_linking=False,
        number_candidates=None,
        add_correct_candidate = False,
        return_results=False,
        threshold=None,
    ) -> List[Result]:
        if self.args.get("_10sec"):
            dataset.dataset.examples = dataset.dataset.examples[:10]
        print("Evaluate")
        document_model.eval()
        entity_model.eval()
        ranking_model.eval()
        self.candidate_manager.eval()
        self.candidate_manager.entity_model = entity_model
        self.candidate_manager.document_embedder = document_model

        all_results = self.\
            predict(dataset, ranking_model, mention_comparator_model, mention_split, mentions_to_add_if_full, number_candidates=number_candidates,
                    add_correct_candidate=add_correct_candidate, spoof_perfect_linking=spoof_perfect_linking,
                    threshold=threshold)

        if second_stage_clustering:
            print("Dumping pickle file.")
            test = [x.to_dict() for x in all_results]
            json.dump(test, open(f"all_results{self.args['suffix']}.p", "w"))
        if return_results:
            return all_results
        return calculate_final_results(all_results, mention_comparator_model, ranking_model,
                                       self.candidate_manager, self.args, self.device, detailed_info)

    def predict_examples(self, examples: List[Example], ranking_model, mention_comparator,
                                      mention_split, mentions_to_add_if_full, number_candidates, add_correct_candidate,
                                      spoof_perfect_linking, num_beams,
                                         threshold=None,
                         results_to_reuse: Dict[int, Result]=None):
        if results_to_reuse is not None:
            cls_embedding = []
            embedded_mentions = []
            for example in examples:
                embedded_mentions_of_example = []
                for x in example.mentions:
                    embedded_mentions_of_example.append(results_to_reuse[x.mention_counter].mention)
                embedded_mentions.append(embedded_mentions_of_example)
                cls_embedding.append(None)
        else:
            cls_embedding, embedded_mentions = self.candidate_manager.embed_documents(
                self.device,
                examples,
                mention_comparator
            )

        maximum_num_mentions = max(len(x.mentions) for x in examples)
        current_entity_representations: List[List] = [[] for _ in range(len(examples))]
        current_mentions: List[List] = [[] for _ in range(len(examples))]

        results: List[List[Result]] = [[] for _ in range(len(examples))]

        if self.args.get("spoof_linking_and_detection", False):
            number_candidates = 0

        all_nested_candidates, num_candidates_with_mentions = get_nested_candidates(maximum_num_mentions,
                                                                                    examples,
                                                                                    embedded_mentions,
                                                                                    mention_split,
                                                                                    mentions_to_add_if_full,
                                                                                    self.candidate_manager,
                                                                                    number_candidates,
                                                                                    add_correct_candidate)
        beam_list = [[(0, [], [])] for _ in range(len(examples))]
        for nested_candidates, sub_sampled_embedded_mentions, mention_idx in all_nested_candidates:
            entity_representations = self.candidate_manager.get_candidate_representations(nested_candidates,
                                                                                          sub_sampled_embedded_mentions,
                                                                                          mention_comparator)

            update_current_representations(entity_representations, sub_sampled_embedded_mentions,
                                           current_entity_representations, current_mentions,
                                           mention_split)

            if not entity_representations:
                continue

            document_containers = prepare_document_containers(
                cls_embedding,
                sub_sampled_embedded_mentions,
                entity_representations,
            )

            calculate_similarities(document_containers, ranking_model)

            for idx, (
                    embedded_document,
                    mentions_,
            ) in enumerate(zip(
                document_containers,
                sub_sampled_embedded_mentions,
            )):
                for mention in embedded_document.mentions:
                    beams = beam_list[idx]
                    new_beams = []
                    for beam in beams:
                        cumulative_score, already_embedded_example, all_decisions = beam
                        non_normalized_scores, concatenated = ranking_model(
                            mention, already_embedded_example)

                        max_tuple = torch.sort(non_normalized_scores, dim=0, descending=True)

                        if len(already_embedded_example) >= mention_split:
                            already_embedded_example.pop(0)

                        for score, max_index in list(zip(*max_tuple))[:num_beams]:
                            tmp_already_embedded = list(already_embedded_example)
                            if spoof_perfect_linking:
                                try:
                                    already_embedded_index = [x.complex_candidate.qid for x in
                                                              mention.candidate_representations].index(
                                        mention.embedded_mention.mention_container.label_qid)
                                except ValueError:
                                    already_embedded_index = max_index
                            else:
                                already_embedded_index = max_index

                            is_ookg = True
                            link = ""
                            if max_index < len(mention.candidate_representations):
                                is_ookg = False
                                link = mention.candidate_representations[max_index]
                            non_normalized_score_decision = non_normalized_scores[max_index]
                            if already_embedded_index < len(mention.candidate_representations):
                                non_normalized_score = non_normalized_scores[already_embedded_index]
                                tmp_candidate = mention.candidate_representations[already_embedded_index]
                                tmp_already_embedded.append((tmp_candidate, None, non_normalized_score.squeeze(1)))
                            tmp_decisions = list(all_decisions)
                            tmp_decisions.append(
                                (link, is_ookg, mention, non_normalized_score_decision, non_normalized_scores))
                            bisect_index = bisect.bisect([x[0] for x in new_beams],
                                                         cumulative_score - non_normalized_score_decision)
                            new_beams = new_beams[:bisect_index] + [(cumulative_score - non_normalized_score_decision,
                                                                     tmp_already_embedded, tmp_decisions)] + new_beams[
                                                                                                             bisect_index:]

                    beam_list[idx] = new_beams[:num_beams]

                for beams in beam_list:
                    cumulative_score, already_embedded, decisions = beams[0]
                    for idx_, (link, is_ookg, mention, non_normalized_score, non_normalized_scores) in enumerate(decisions):
                        if is_ookg:
                            other_entities_before = [x for idx__, x in enumerate(decisions) if
                                                     not x[1] and idx__ < idx_]
                            other_entities_after = [x for idx__, x in enumerate(decisions) if
                                                    not x[1] and idx__ > idx_]

                            context_entities: List[tuple] = get_context_entities(other_entities_before, other_entities_after,
                                                                                 self.args.get("window_size", 6))
                            context_transe_embeddings = [x[0].kg_embedding for x in context_entities]
                            context_mention_embeddings = [x[2].embedded_mention.processed_mention.mention_embedding for x in
                                                          context_entities]

                            if self.args.get("alternate_mention_embedding", False):
                                post_mention_embedding = mention_comparator.combine_mentions_with_mention_embeddings(
                                    mention.embedded_mention.processed_mention.mention_embedding, context_mention_embeddings,
                                    context_transe_embeddings).to("cpu")
                            else:
                                post_mention_embedding = None
                        else:
                            context_transe_embeddings = []
                            post_mention_embedding = None
                        if self.args.get("spoof_linking_and_detection", False):
                            is_ookg = mention.embedded_mention.mention_container.label_out_of_kg
                        results[idx].append(Result(link, mention.embedded_mention, is_ookg,
                                                   non_normalized_scores.tolist(),
                                                   float(non_normalized_score),
                                                   float(non_normalized_score),
                                                   context_transe_embeddings,
                                                   post_mention_embedding,
                                                   mention.candidate_representations))
        return results

    def predict(self, dataset, ranking_model, mention_comparator: MentionComparator,
                mention_split: int, mentions_to_add_if_full: int, number_candidates=None, add_correct_candidate=False,
                spoof_perfect_linking=False,
                num_beams=10, threshold=None, results_to_reuse: Dict[int, Result]=None):

        if not (self.args.get("use_contextual_types") or self.args.get("use_transe")):
            num_beams = 1
        all_results = []
        with torch.inference_mode():
            for idx_test_set, examples in enumerate(tqdm(dataset)):
                results = self.predict_examples(examples, ranking_model, mention_comparator,
                                      mention_split, mentions_to_add_if_full, number_candidates, add_correct_candidate,
                                      spoof_perfect_linking, num_beams, threshold=threshold, results_to_reuse=results_to_reuse)


                all_results += list(itertools.chain.from_iterable(results))

                if self.args.get("_10sec") and idx_test_set > 10:
                    break

        return all_results