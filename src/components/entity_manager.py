import json
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Optional, Tuple, Union, Any

import numpy
import torch
import torch.nn.functional
from elasticsearch import Elasticsearch
from rapidfuzz import fuzz

from src.candidate_retrieval.candidate_retrieval import get_candidates_alt
from src.components.kg_connector import KnowledgeGraphConnector
from src.utilities.utils import get_context_entities
from src.model.embedding_model import BertDocumentEmbedder
from src.model.mention_comparison import MentionComparator
from src.utilities.special_tokens import SEP
from src.utilities.constants import AdditionalFeatures
from src.utilities.utilities import bert_embed_document
from src.utilities.various_dataclasses import (
    CandidateContainerForProcessing,
    DocumentContainerForProcessing,
    MentionContainerForProcessing, ClusterRepresentation, KGCandidateEntity, ComplexCandidateEntity, Example,
    EmbeddedMention, MentionInClusterToBeEmbedded, ProcessedMention, )


class CandidateManager:
    def __init__(
        self,
        device,
        kg_connector: KnowledgeGraphConnector,
        type_list: List[str],
        two_hop_type_list: List[str],
        args: dict,
        mention_document_index: dict = None,
        document_embedder: Tuple[BertDocumentEmbedder, Any] = None,
    ):
        self.args = args
        self.device = device
        self.entity_model = None
        self.kg_connector = kg_connector
        self.additional_entity_features = self.args.get("additional_entity_features")
        self.type_list = type_list
        self.two_hop_type_list = two_hop_type_list
        mention_dictionary = json.load(self.args.get("mention_dictionary_file").open())

        normalized_mention_dictionary = {}
        self.special_keys = False
        if isinstance(mention_dictionary, list):
            self.special_keys = True
            for item in mention_dictionary:
                item = deepcopy(item)
                item["identifier"]["mention"] = item["identifier"]["mention"].lower()
                if frozenset(item["identifier"].items()) not in mention_dictionary:
                    normalized_mention_dictionary[frozenset(item["identifier"].items())] = item[
                        "candidates"
                    ]
        else:
            normalized_mention_dictionary = {key.lower(): value for key, value in mention_dictionary.items() if key.lower() not in mention_dictionary}
        self.mention_dictionary = mention_dictionary
        self.normalized_mention_dictionary = normalized_mention_dictionary
        self.kg_connector = kg_connector
        self.add_correct_candidate_during_training = self.args.get("add_correct_candidate_during_training")

        self.number_candidates = self.args.get("number_candidates")
        self.num_mentions_in_cluster = self.args.get("num_mentions_in_cluster")


        self.num_cluster_candidates = self.args.get("num_cluster_candidates")
        self.use_cosine_for_filtering = self.args.get("use_cosine_for_filtering", False)

        self.ookg_representations: Dict[int, ClusterRepresentation] = {}
        self.inkg_representations: Dict[str, ClusterRepresentation] = {}

        if self.args.get("transe_embeddings_file") is not None and self.args.get("transe_mappings_file") is not None:
            self.transe_embeddings = numpy.load(self.args.get("transe_embeddings_file"))
            self.transe_mappings = json.load(open(self.args.get("transe_mappings_file")))
            self.transe_mappings = {qid: idx for idx, qid in enumerate(self.transe_mappings)}
            self.transe_embeddings = torch.from_numpy(self.transe_embeddings)
            self.transe_embeddings.requires_grad = False
        else:
            self.transe_embeddings = None
            self.transe_mappings = {}

        self.train = False
        self.mention_document_index = mention_document_index
        self.pre_sampled_tuples = {}
        self.document_embedder = document_embedder

        if kg_connector is None:
            Warning("No kg_connector. Return only qid.")

    def pre_sampled_from_mention_document_index(self, mention_document_index: dict):
        pre_sampled_tuples = defaultdict(list)
        if mention_document_index is not None:
            for key, value in mention_document_index.items():
                tmp_list = list(value)
                random.shuffle(tmp_list)
                chunk_size = self.num_mentions_in_cluster + 1
                for i in range(0, len(tmp_list), chunk_size):
                    chunk = tmp_list[i:i + chunk_size]
                    pre_sampled_tuples[(chunk[0][0].identifier, chunk[0][1])] = chunk[1:]
        return pre_sampled_tuples

    def reset_representations(self):
        self.ookg_representations = {}
        self.inkg_representations = {}
        self.pre_sampled_tuples = self.pre_sampled_from_mention_document_index(self.mention_document_index)

    def eval(self):
        self.train = False
        self.reset_representations()

    @staticmethod
    def calculate_popularity(candidate_content: KGCandidateEntity) -> float:
        return candidate_content.degree


class BertCandidateManager(CandidateManager):
    def __init__(
        self,
        device,
        entity_tokenizer,
        kg_connector,
        args: dict,
        type_list: List[str] = None,
        two_hop_type_list: List[str] = None,
        mention_document_index=None,
        document_embedder: Tuple[BertDocumentEmbedder, Any] = None,
        mask_token_id = None
    ):
        super().__init__(
            device,
            kg_connector,
            type_list,
            two_hop_type_list,
            args,
            mention_document_index,
            document_embedder,
        )
        self.entity_tokenizer = entity_tokenizer
        self.mask_token_id = mask_token_id
        self.alternative_description=args.get("alternative_description")
        self.index_name = "wikinews_2019_extended"  # old_dump
        self.dedicated_es = None #Elasticsearch()

    def compute_additional_mention_features(
        self,
        mention_list: List[List[EmbeddedMention]],
        candidates: List[List[List[ComplexCandidateEntity]]],
        device=None,
    ) -> Optional[list]:
        additional_features = []
        for (
            mentions,
            candidates_per_mentions_list,
        ) in zip(mention_list, candidates):
            additional_features_sub = []
            for mention, candidates_list in zip(
                mentions, candidates_per_mentions_list
            ):
                popularities = []
                for candidate in candidates_list:
                    popularity = 0
                    if candidate.is_kg_candidate:
                        popularity = self.calculate_popularity(candidate.candidate)

                    popularities.append(popularity)
                # popularities = [x/sum(popularities) for x in popularities]
                features_scores = []
                if AdditionalFeatures.POPULARITY in self.additional_entity_features:
                    features_scores.append(popularities)
                combined_features = [
                    torch.tensor(x, device=self.device if device is None else device)
                    for x in zip(*features_scores)
                ]
                additional_features_sub.append(
                    torch.stack(combined_features, dim=0)
                    if combined_features
                    else torch.empty((len(candidates_list), 0), device=self.device if device is None else device)
                )
            additional_features.append(additional_features_sub)
        return additional_features


    def get_spoofed_complex_candidate_entity(self, text, qid, kg_candidate: Optional[KGCandidateEntity], mention_idx, current_qid_is_gold_qid) -> Tuple[List[ComplexCandidateEntity], bool]:
        sampled_texts = None
        if current_qid_is_gold_qid:
            sampled_texts = self.pre_sampled_tuples[(text.identifier, mention_idx)]

        if sampled_texts is None:
            mention_texts_with_same_qid = self.mention_document_index.get(qid, [])
            mention_texts_with_same_qid = [(example, mention_idx) for example, mention_idx in mention_texts_with_same_qid]
            texts_of_other_mentions_same_qid = []

            for x, idx in mention_texts_with_same_qid:
                if x.identifier == text.identifier:
                    entity = x.mentions[idx]
                    if entity.label_qid == qid and idx != mention_idx:
                        texts_of_other_mentions_same_qid.append((x, idx))
                else:
                    entity = x.mentions[idx]
                    if entity.label_qid == qid:
                        texts_of_other_mentions_same_qid.append((x, idx))

            if texts_of_other_mentions_same_qid:
                num_mentions = self.num_mentions_in_cluster
                mentions_to_sample = num_mentions

                # TODO: Confusing variable name change
                indices_to_sample = list(range(len(texts_of_other_mentions_same_qid)))
                sampled_indices_of_other_mention_indices = random.sample(indices_to_sample,
                                                                         min(len(indices_to_sample),
                                                                             mentions_to_sample))
                if sampled_indices_of_other_mention_indices:
                    sampled_texts = []
                    for idx in sampled_indices_of_other_mention_indices:
                        sampled_texts.append(texts_of_other_mentions_same_qid[idx])

        final_candidates = []
        spoofed = False
        if sampled_texts:
            spoofed = True
            for (x, ment_idx) in sampled_texts:
                global_type_assignment = torch.zeros(len(self.two_hop_type_list), device=self.device)

                representation = MentionInClusterToBeEmbedded(qid, x, ment_idx, x.mentions[ment_idx], global_type_assignment)

                final_candidates.append(ComplexCandidateEntity(representation))
        if kg_candidate is not None:
            final_candidates.append(ComplexCandidateEntity(kg_candidate))
        return final_candidates, spoofed

    def enrich_with_spoofed_cluster_candidates(self, text: Example, kg_candidates: List[KGCandidateEntity], mention,
                                               mention_idx: int) -> Tuple[List[ComplexCandidateEntity], bool]:
        assert self.mention_document_index is not None
        assert self.document_embedder is not None

        final_candidates = []
        spoofed = False
        for kg_candidate in kg_candidates:
            complex_candidate_entities, tmp_spoofed = self.get_spoofed_complex_candidate_entity(text, kg_candidate.qid, kg_candidate,
                                                                                          mention_idx, kg_candidate.qid == mention.label_qid
                                                                                        )
            if kg_candidate.qid == mention.label_qid:
                spoofed = spoofed or tmp_spoofed
            final_candidates += complex_candidate_entities
        if mention.label_out_of_kg or mention.label_qid not in kg_candidates:
            complex_candidate_entities, tmp_spoofed = self.get_spoofed_complex_candidate_entity(text, mention.label_qid,
                                                                                            None,
                                                                                            mention_idx,
                                                                                            mention.label_qid == mention.label_qid
                                                                                            )
            spoofed = spoofed or tmp_spoofed

            final_candidates += complex_candidate_entities
        return final_candidates, spoofed

    def enrich_with_cluster_candidates(self, kg_candidates: List[KGCandidateEntity], mention, mention_embedding: torch.Tensor, threshold=0.9):
        final_candidates: List[ComplexCandidateEntity] = [ComplexCandidateEntity(x) for x in kg_candidates]

        representations = list(self.inkg_representations.values()) + list(
            self.ookg_representations.values()
        )
        similarities = []
        for rep in representations:
            if rep.get_bounded_list():
                if not self.use_cosine_for_filtering:
                    similarity = max(
                        fuzz.partial_ratio(mention.mention.lower(), surface_form.lower(), score_cutoff=threshold * 100)
                        for surface_form in rep.surface_forms
                    )
                else:
                    similarity = max(
                        torch.cosine_similarity(mention_embedding, elem.embedded_mention_container.mention_embedding)
                        for elem in rep.get_bounded_list()
                    )
            else:
                similarity = 0.0
            similarities.append(torch.tensor(similarity))

        if not similarities:
            return final_candidates
        max_scores, indices = torch.sort(
            torch.stack(similarities), dim=0, descending=True
        )
        indices = [
            index
            for max_score, index in zip(max_scores.tolist(), indices.tolist())
            if max_score > threshold
        ]
        indices_range = list(
            indices[0: min(len(indices), self.num_cluster_candidates)]
        )
        cluster_candidates = [representations[idx] for idx in indices_range if representations[idx].identifier not in final_candidates]

        for idx, candidate in enumerate(list(final_candidates)):
            if candidate.qid in self.inkg_representations:
                representation = self.inkg_representations[candidate.qid]
                for sub_representation in representation.get_mentions_in_cluster():
                    final_candidates.append(sub_representation)

        for item in cluster_candidates:
            for sub_representation in item.get_mentions_in_cluster():
                final_candidates.append(ComplexCandidateEntity(candidate=sub_representation,
                                                               identified_in_cluster_comparison=True))
            if isinstance(item.identifier, str):
                in_set = item.identifier in kg_candidates
                if not in_set:
                    kg_candidate_entity = self.kg_connector.get_entity(
                        item.identifier, labelized_neighborhood=True
                    )
                    final_candidates.append(ComplexCandidateEntity(candidate=kg_candidate_entity,
                                                                   identified_in_cluster_comparison=True))

        return final_candidates

    def assign_candidates(
            self,
            candidates,
            mention,
            number_candidates,
            add_correct_candidate,
    ):
        candidates = [candidate if isinstance(candidate, dict) else {"qid": candidate} for candidate in candidates]
        if number_candidates >= 0:
            new_candidates = []
            for candidate  in candidates:
                if candidate["qid"] in self.inkg_representations:
                    new_candidates.append(candidate)
                elif len(new_candidates) < number_candidates:
                    new_candidates.append(candidate)
            candidates = new_candidates
        if not mention.label_out_of_kg and add_correct_candidate:
            if mention.label_qid not in {
                candidate if isinstance(candidate, str) else candidate["qid"]
                for candidate in candidates
            }:
                if len(candidates) == number_candidates:
                    candidates = candidates[:-1]
                candidates.append({"qid": mention.label_qid, "prior": 0.0})
        elif mention.label_out_of_kg:
            for idx, candidate in reversed(list(enumerate(candidates))):
                if (
                        candidate if isinstance(candidate, str) else candidate["qid"]
                ) == mention.label_qid:
                    del candidates[idx]
        return candidates


    def normalize_and_get_candidates(self, mention_container, super_idx, idx):
        if mention_container.mention in self.mention_dictionary:
            raw_candidates =  self.mention_dictionary.get(mention_container.mention, [])
        else:
            if self.special_keys and mention_container.other_info is not None:
                mention_key = frozenset(
                    {
                        "mention": mention_container.mention.lower(),
                        "docid": mention_container.other_info[super_idx][idx]["docid"],
                    }.items()
                )
            else:
                mention_key = mention_container.mention.lower()
            if mention_key in self.mention_dictionary:
                raw_candidates = self.mention_dictionary[mention_key]
            else:
                raw_candidates = self.normalized_mention_dictionary.get(mention_key, [])
        summation = sum([x["prior"] for x in raw_candidates])
        if summation > 1.0:
            raw_candidates = [{**x, "prior": x["prior"] / summation} for x in raw_candidates]
        return raw_candidates

    def normalize_and_get_candidates_es(self, mention_container):
        raw_candidates = get_candidates_alt([mention_container.mention], dedicated_es=self.dedicated_es)[mention_container.mention]
        raw_candidates = [{"qid": x[0], "prior": 1 / len(raw_candidates)} for x in raw_candidates]
        return raw_candidates

    def get_candidates(
        self,
        texts: list,
        processed_mentions: list,
        number_candidates: int = None,
        add_correct_candidate: bool = False,
        mention_idx_offset: int = -1,
    ):
        if number_candidates is None:
            number_candidates = self.number_candidates
        if not add_correct_candidate:
            if self.train:
                add_correct_candidate = self.add_correct_candidate_during_training
        num_candidates_with_mentions = 0
        candidates_list = []
        for super_idx, (processed_mentions_, text) in enumerate(
            zip(processed_mentions, texts)
        ):
            candidates = []
            for idx in reversed(range(len(processed_mentions_))):
                mention = processed_mentions_[idx].mention_container
                processed_mention = processed_mentions_[idx]
                raw_candidates = self.normalize_and_get_candidates(mention, super_idx, idx)

                assigned_candidates = self.assign_candidates(
                    raw_candidates,
                    mention,
                    number_candidates,
                    add_correct_candidate,
                )

                kg_candidates = [
                    self.kg_connector.get_entity(candidate)
                    for candidate in assigned_candidates
                ]

                actual_kg_candidates = []
                for x in kg_candidates:
                    if isinstance(x, str):
                        warnings.warn(f"{x} not found in KG.")
                    else:
                        actual_kg_candidates.append(x)

                kg_candidates = actual_kg_candidates

                if self.train:
                    final_candidates, spoofed = self.enrich_with_spoofed_cluster_candidates(text, kg_candidates, mention, idx + mention_idx_offset)
                    num_candidates_with_mentions += spoofed
                else:
                    final_candidates = self.enrich_with_cluster_candidates(kg_candidates, mention, processed_mention)

                candidates.append(final_candidates)
            candidates_list.append(list(reversed(candidates)))

        return candidates_list, num_candidates_with_mentions

    def embed_documents(self, device,
                        examples: List[Example],
                        mention_comparator: MentionComparator,
                        num_context_transe_embeddings: int = 6) -> Tuple[torch.Tensor, List[List[EmbeddedMention]]]:

        cls_embeddings, embedded_mentions_examples = bert_embed_document(device, self.document_embedder, examples)

        embedded_mentions_examples_final = []
        for example, embedded_mentions in zip(examples, embedded_mentions_examples):
            assert len(example.mentions) == len(embedded_mentions)
            embedded_mentions_final = []
            for idx, (mention, embedded_mention) in enumerate(zip(example.mentions, embedded_mentions)):
                other_entities_before = [(x, embedded_mentions[idx_]) for idx_, x in enumerate(example.mentions) if
                                         x.label_qid != mention.label_qid and not mention.label_out_of_kg and idx_ < idx]
                other_entities_after = [(x, embedded_mentions[idx_]) for idx_, x in enumerate(example.mentions) if
                                        x.label_qid != mention.label_qid and not mention.label_out_of_kg and idx_ > idx]

                context_entities = get_context_entities(other_entities_before, other_entities_after,
                                                        num_context_transe_embeddings)
                transe_embeddings_of_mention = []
                other_entity_embeddings_of_mention = []
                for other_processed_mention, other_entity in [(x[1], self.kg_connector.get_entity(x[0].label_qid)) for x
                                                              in
                                                              context_entities]:
                    if not isinstance(other_entity, str) and other_entity.qid in self.transe_mappings:
                        transe_embeddings_of_mention.append(self.transe_embeddings[self.transe_mappings[other_entity.qid]])
                        other_entity_embeddings_of_mention.append(other_processed_mention.mention_embedding)
                if self.args.get("alternate_mention_embedding", False):
                    post_mention_embedding = mention_comparator.combine_mentions_with_mention_embeddings(
                        embedded_mention.mention_embedding,
                        other_entity_embeddings_of_mention,
                        transe_embeddings_of_mention
                        )
                else:
                    post_mention_embedding = None
                embedded_mentions_final.append(EmbeddedMention(mention, embedded_mention, post_mention_embedding))
            embedded_mentions_examples_final.append(embedded_mentions_final)

        return cls_embeddings, embedded_mentions_examples_final

    def get_candidate_representations(
        self,
        nested_candidates,
        mention_list: List[List[EmbeddedMention]],
        mention_comparator: MentionComparator,
    ) -> List[List[List[CandidateContainerForProcessing]]]:

        flat_candidate_entities_dict = {}
        flat_mentions_dict = {}
        flat_candidate_entities = []
        flat_texts = []
        for x in nested_candidates:
            for y in x:
                for z in y:
                    if z.is_kg_candidate and z.candidate not in flat_candidate_entities_dict:
                        flat_candidate_entities_dict[z.candidate] = len(flat_candidate_entities)
                        flat_candidate_entities.append(z.candidate)
                    elif z.is_mention_cluster_to_be_embedded and z.candidate.example.identifier not in flat_mentions_dict:
                        flat_mentions_dict[z.candidate.example.identifier] = len(flat_texts)
                        flat_texts.append(z.candidate.example)
        entity_embeddings = self.embed_entities(flat_candidate_entities)
        spoofed_mention_embeddings = self.embed_spoofed_mentions(flat_texts, mention_comparator)
        additional_mention_features = self.compute_additional_mention_features(
            mention_list, nested_candidates, self.device
        )
        document_mentions = []
        for candidates_mention_example, features_mention_example in zip(
            nested_candidates, additional_mention_features
        ):
            mentions_per_document = []
            for candidates, features_candidates in zip(
                candidates_mention_example, features_mention_example
            ):
                candidate_representations = []
                for complex_candidate, features in zip(candidates, features_candidates):
                    kg_one_hop_neighborhood = None
                    kg_embedding = None
                    post_mention_embedding = None
                    if complex_candidate.is_kg_candidate:
                        candidate_list_idx = flat_candidate_entities_dict[complex_candidate.qid]
                        kg_one_hop_neighborhood = complex_candidate.candidate.one_hop_types_tensor.to(self.device).to_dense()
                        two_hop_neighborhood = complex_candidate.candidate.two_hop_types_tensor.to(self.device)
                        if complex_candidate.qid in self.transe_mappings:
                            kg_embedding = self.transe_embeddings[self.transe_mappings[complex_candidate.qid], :].to(self.device)

                        entity_embedding = entity_embeddings[candidate_list_idx, :]
                    elif complex_candidate.is_mention_cluster_to_be_embedded:
                        doc_list_idx = flat_mentions_dict[complex_candidate.candidate.example.identifier]
                        two_hop_neighborhood = None
                        entity_embedding = spoofed_mention_embeddings[doc_list_idx][complex_candidate.candidate.mention_identifier].processed_mention.mention_embedding
                        post_mention_embedding = spoofed_mention_embeddings[doc_list_idx][complex_candidate.candidate.mention_identifier].post_mention_embedding
                    else:
                        two_hop_neighborhood = None
                        entity_embedding = complex_candidate.candidate.embedded_mention_container.mention_embedding

                    # if self.args["use_types"]:
                    #     entity_embedding = self.entity_model.type_enrichment(entity_embedding, kg_one_hop_neighborhood)

                    candidate_representations.append(
                        CandidateContainerForProcessing(
                            complex_candidate,
                            entity_embedding,
                            kg_embedding,
                            two_hop_neighborhood,
                            features,
                            kg_one_hop_neighborhood,
                            post_mention_embedding=post_mention_embedding
                        )
                    )
                mentions_per_document.append(candidate_representations)
            document_mentions.append(mentions_per_document)
        return document_mentions


    @staticmethod
    def create_head_mask(label_spans, description_spans, triple_spans, dimensionality):
        head_mask = torch.zeros(size=(dimensionality, dimensionality))
        head_mask[0, 0] = 1

        for indices in label_spans:
            indices = torch.tensor(list(indices), dtype=torch.int64)
            head_mask[0, indices] = 1
            head_mask[indices, 0] = 1
            combs = torch.combinations(indices, 2, with_replacement=True)
            head_mask[combs] = 1

        for indices in description_spans:
            indices = torch.tensor(list(indices), dtype=torch.int64)
            head_mask[0, indices] = 1
            head_mask[indices, 0] = 1
            combs = torch.combinations(indices, 2, with_replacement=True)
            head_mask[combs] = 1

        for indices in triple_spans:
            indices = torch.tensor(list(indices), dtype=torch.int64)
            head_mask[0, indices] = 1
            head_mask[indices, 0] = 1
            combs = torch.combinations(indices, 2, with_replacement=True)
            head_mask[combs] = 1

        return head_mask

    @staticmethod
    def transform_entity_partially_connected(entity: KGCandidateEntity):
        spans = []
        concatenated_string = "<s> "

        current_idx = len(concatenated_string)
        for label in [entity.label] + entity.aliases:
            concatenated_string += label + " "
            spans.append((0, (current_idx, current_idx + len(label))))
            current_idx = len(concatenated_string)

        if entity.description:
            concatenated_string += entity.description + " "
            spans.append((1, (current_idx, current_idx + len(entity.description))))

        current_idx = len(concatenated_string)

        if entity.labelized_out_claims is not None:
            for claim in entity.labelized_out_claims:
                concatenated_string += f"{claim[0]} {claim[1]}" + " "
                spans.append(
                    (2, (current_idx, current_idx + len(claim[0]) + len(claim[1]) + 1))
                )
                current_idx = len(concatenated_string)

        concatenated_string += SEP

        return concatenated_string, spans

    def transform_entity(self, entity: KGCandidateEntity, lower_text: bool = False):
        concatenated_suffix = ""
        if entity.labelized_out_claims is not None:
            sorted_labelized_claims = random.sample(entity.labelized_out_claims, min(15, len(entity.labelized_out_claims)))
            sorted_labelized_claims = sorted(sorted_labelized_claims, key=lambda x:x[0][1])
            concatenated_suffix += f" {SEP} ".join(
                [f"{x[0][1]} {x[1][1]}" for x in sorted_labelized_claims]
            )

        if entity.labelized_in_claims is not None:
            pass

        description = entity.description.lower() if lower_text else entity.description
        if self.alternative_description is not None:
            alt_description = entity.other_info.get(self.alternative_description, None)
            if alt_description is not None:
                description = alt_description
            else:
                warnings.warn(f"{self.alternative_description} not found.")

        return (
            f"{entity.label.lower() if lower_text else entity.label}"
            + f" {SEP} {description}"
        )

    def encode_fully_connected(self, candidate_entities):
        combined_strings = [self.transform_entity(x) for x in candidate_entities]

        tokenized = self.entity_tokenizer(
            combined_strings,
            padding=True,
            return_tensors="pt",
            max_length=self.args.get("max_length_description_encoder", 32),
            truncation=True,
            return_offsets_mapping=True,
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def encode_partially_connected(self, candidate_entities):
        combined_strings, spans = zip(
            *[
                self.transform_entity_partially_connected(
                    self.kg_connector.get_entity(x)
                )
                for x in candidate_entities
            ]
        )

        tokenized = self.entity_tokenizer(
            list(combined_strings),
            padding=True,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offset_mappings = tokenized["offset_mapping"]
        head_masks = []
        all_position_ids = []
        all_type_ids = []
        for batch, spans_ in zip(offset_mappings, spans):
            offset = 0
            batch_as_list = list(batch)

            label_spans = []
            triple_spans = []
            description_spans = []
            position_ids = []
            type_ids = []
            position_id = 0
            for span in spans_:
                current_indices = set()
                for idx in range(len(batch_as_list)):
                    tokenized_span = batch_as_list[idx]
                    if current_indices:
                        current_indices.add(offset + idx)
                    if tokenized_span[0] == span[1][0]:
                        current_indices.add(offset + idx)
                    position_ids.append(position_id)
                    type_ids.append(span[0])
                    position_id += 1
                    if tokenized_span[1] == span[1][1]:
                        batch_as_list = batch_as_list[idx + 1 :]
                        offset += idx + 1
                        position_id = 1
                        break
                else:
                    batch_as_list = []

                if span[0] == 0:
                    label_spans.append(current_indices)
                elif span[0] == 1:
                    description_spans.append(current_indices)
                elif span[0] == 2:
                    triple_spans.append(current_indices)
            head_mask = self.create_head_mask(
                label_spans, description_spans, triple_spans, batch.size()[0]
            )
            head_masks.append(head_mask)
            for _ in batch_as_list:
                position_ids.append(1)
                type_ids.append(0)
            all_position_ids.append(torch.tensor(position_ids))
            all_type_ids.append(torch.tensor(type_ids))

        batch_head_masks = torch.stack(head_masks)
        batch_position_ids = torch.stack(all_position_ids)
        batch_type_ids = torch.stack(all_type_ids)
        batch_head_masks = batch_head_masks.repeat((12, 12, 1, 1, 1))
        batch_head_masks = torch.transpose(batch_head_masks, 1, 2)
        return (
            tokenized["input_ids"],
            tokenized["attention_mask"],
            batch_head_masks,
            batch_position_ids,
            batch_type_ids,
        )

    def encode_entities(self, candidate_entities: list):

        if not candidate_entities:
            return None, None, None, None, None

        input_ids, attention_mask = self.encode_fully_connected(candidate_entities)
        position_ids = None
        head_mask = None
        token_type_ids = None
        return input_ids, attention_mask, position_ids, head_mask, token_type_ids

    def embed_entities_alt(
        self, candidate_entities: list
    ) -> Tuple[Optional[list], Optional[torch.Tensor]]:
        (
            input_ids,
            attention_masks,
            position_ids,
            head_mask,
            token_type_ids,
        ) = self.encode_entities(candidate_entities)

        if not candidate_entities:
            return [], None
        input_ids = input_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)
        if head_mask is not None:
            head_mask = head_mask.to(self.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        entity_embeddings = self.entity_model(
            input_ids,
            attention_masks,
            position_ids=position_ids,
            head_mask=torch.transpose(head_mask, 0, 1),
            token_type_ids=token_type_ids,
        )[
            0
        ]  # A tuple is returned to support probabilistic embeddings, but we are not using them, so use the first element
        return candidate_entities, entity_embeddings

    def embed_spoofed_mentions(self, flat_texts: list, mention_comparator: MentionComparator) -> List[List[EmbeddedMention]]:
        if not flat_texts:
            return []
        _, mention_embeddings = self.embed_texts(flat_texts, mention_comparator)
        return mention_embeddings



    def embed_entities(self, candidate_entities: list) -> torch.Tensor:
        if not candidate_entities:
            return torch.empty((0,0), device=self.device)

        position_ids = None
        head_mask = None
        token_type_ids = None
        input_ids = torch.stack([entity.input_ids for entity in candidate_entities])
        attention_masks = torch.stack([entity.attention_mask for entity in candidate_entities])

        input_ids = input_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        entity_embeddings = self.entity_model(
            input_ids=input_ids,
            attention_masks=attention_masks,
            position_ids=position_ids,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
        )[0]
        return entity_embeddings

    def embed_texts(self, texts: list, mention_comparator) -> Tuple[torch.Tensor, List[List[EmbeddedMention]]]:
        doc_embeddings, mention_embeddings = self.embed_documents(self.device, texts, mention_comparator,
                                                                 )
        return doc_embeddings, mention_embeddings


def get_original_texts_plus_mention_spans(texts: List[str]):
    original_texts = []
    mention_spans = []
    for _, text in texts:
        # starts = [m for m in re.finditer(f'{re.escape(ENTITY_START)}', text)]
        # ends =[m for m in re.finditer(f'{re.escape(ENTITY_END)}', text)]
        # assert len(starts) == len(ends)
        # tmp_mention_spans = []
        # tmp_text = text
        # offset = 0
        # for start, end in zip(starts, ends):
        #     tmp_text = tmp_text[:end.start() - 1 - offset] + tmp_text[end.end() + 1 - offset:]
        #     tmp_text = tmp_text[:start.start() - 1 - offset] + tmp_text[start.end() + 1 - offset:]
        #     mention_span = (start.end() - 1 - len(ENTITY_START) - offset, end.start() - 3 - len(ENTITY_START) - offset)
        #     offset += len(ENTITY_START) + len(ENTITY_END) + 4
        #     tmp_mention_spans.append(mention_span)
        original_texts.append(text["text"])
        mention_spans.append(
            [(x["offset"], x["offset"] + x["length"]) for x in text["entities"]]
        )
    return original_texts, mention_spans


def prepare_document_containers(
    cls_embeddings,
    embedded_mentions: List[List[EmbeddedMention]],
    entity_representations_documents: List[List[List[CandidateContainerForProcessing]]],
) -> List[DocumentContainerForProcessing]:
    embedded_documents = []
    for embedded_mentions_, cls_embedding, entity_representations_document in zip(
        embedded_mentions, cls_embeddings, entity_representations_documents
    ):
        mentions_per_document = []
        for embedded_mention, entity_representations in zip(
            embedded_mentions_, entity_representations_document
        ):

            mentions_per_document.append(
                MentionContainerForProcessing(
                    embedded_mention, entity_representations
                )
            )
        embedded_documents.append(
            DocumentContainerForProcessing(cls_embedding, mentions_per_document)
        )
    return embedded_documents


