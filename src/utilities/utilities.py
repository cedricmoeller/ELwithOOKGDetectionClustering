import json
from collections import defaultdict
from pathlib import Path
from random import random, sample
from typing import Tuple, List

import jsonlines
import torch
from rapidfuzz import fuzz
from tqdm import tqdm

from src.model.ranking_models import SupervisedRankingModel
from src.utilities.special_tokens import SEP
from src.utilities.various_dataclasses import ClusterAssignment, Example, CandidateContainerWrapper, \
    DocumentContainerForProcessing, CandidateContainerForProcessing, ComplexCandidateEntity, MentionInCluster, \
    EmbeddedMention, ProcessedMention


def count_correctly_classified_out_of_kg(
        is_out_of_kg: torch.Tensor, entities: list, links: list
) -> Tuple[int, int, int, int, int, int, torch.Tensor]:
    out_of_kg_entities = torch.tensor([entity[1] for entity in entities])
    # Interpret links to integers also as out_of_kg
    all_correct = torch.logical_not(
        torch.logical_xor(out_of_kg_entities,
                          torch.logical_or(is_out_of_kg, torch.tensor([isinstance(link, int) for link in links])))
    )
    in_kg_linked_ookg_cluster =  torch.logical_and(torch.logical_not(out_of_kg_entities),torch.tensor([isinstance(link, int) for link in links]))
    newly_out_of_kg_in_kg = torch.logical_and(torch.logical_not(out_of_kg_entities),
                                                  is_out_of_kg)
    all_incorrect = torch.logical_not(all_correct)
    non_out_of_kg_entities = torch.logical_not(out_of_kg_entities)
    correctly_classified_non_EE = torch.logical_and(
        all_correct, non_out_of_kg_entities
    )

    return (
        int(torch.count_nonzero(torch.logical_and(all_correct, out_of_kg_entities))),
        int(
            torch.count_nonzero(torch.logical_and(all_incorrect, out_of_kg_entities))
        ),
        int(torch.count_nonzero(correctly_classified_non_EE)),
        int(
            torch.count_nonzero(
                torch.logical_and(all_incorrect, non_out_of_kg_entities)
            )
        ),
        int(torch.count_nonzero(in_kg_linked_ookg_cluster)),
        int(torch.count_nonzero(newly_out_of_kg_in_kg)),
        correctly_classified_non_EE,

    )


def count_correctly_classified_in_kg(
    links: list, entities: list, correctly_classified_non_EE: torch.Tensor
) -> Tuple[int, int]:
    ground_truths = [entity[0] for entity in entities]
    equal_clusters = torch.tensor(
        [link == ground_truth for link, ground_truth in zip(links, ground_truths)]
    )

    return int(torch.count_nonzero(equal_clusters)), int(
        torch.count_nonzero(
            torch.logical_and(correctly_classified_non_EE, equal_clusters)
        )
    )


def calculate_fmeasure(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall)
        if precision > 0 or recall > 0
        else 0
    )


def mask_example(example: Example, mask_ratio: float, mask_token_id, fast_masking=True) -> Example:
    input_ids_to_use, attention_masks_to_use, start_embedding_positions_to_use, end_embedding_positions_to_use = (
    example.input_ids, example.attention_masks, example.start_embedding_positions, example.end_embedding_positions)
    mentions_to_mask = set()
    for position in range(start_embedding_positions_to_use.size(0)):
        if random() < mask_ratio:
            mentions_to_mask.add(position)
    offset_ = 0
    if mentions_to_mask:
        if fast_masking:
            input_ids_to_use = torch.clone(input_ids_to_use)
            fill_cells = []
            for mention_idx, (start_position, end_position) in enumerate(
                    zip(start_embedding_positions_to_use, end_embedding_positions_to_use)):
                if mention_idx in mentions_to_mask:
                    fill_cells.append(torch.arange(int(start_position), int(end_position) + 1))
            fill_cells = torch.cat(fill_cells, dim=0)
            input_ids_to_use.index_fill_(1, fill_cells, mask_token_id)

            example = Example(input_ids_to_use, attention_masks_to_use, start_embedding_positions_to_use,
                              end_embedding_positions_to_use, example.identifier,
                              example.mentions)

        else:
            input_ids_to_use = input_ids_to_use.tolist()[0]
            attention_masks_to_use = attention_masks_to_use.tolist()[0]
            new_input_ids = []
            new_start_positions = []
            new_end_positions = []
            new_attention_masks_to_use = []
            last_end_position = 0
            for mention_idx, (start_position, end_position) in enumerate(
                    zip(start_embedding_positions_to_use, end_embedding_positions_to_use)):
                new_start_positions.append(start_position - offset_)
                segment = input_ids_to_use[last_end_position: start_position]
                attention_segment = attention_masks_to_use[last_end_position: start_position]
                if mention_idx in mentions_to_mask:
                    entity_length = end_position - start_position + 1
                    if entity_length > 0:
                        offset_ += entity_length - 1
                        segment += [mask_token_id]
                    else:
                        segment += []
                    attention_segment += [1]
                else:
                    segment += input_ids_to_use[start_position: end_position + 1]
                    attention_segment += attention_masks_to_use[start_position: end_position + 1]
                last_end_position = end_position + 1
                new_end_positions.append(end_position - offset_)
                new_input_ids += segment
                new_attention_masks_to_use += attention_segment

            new_input_ids += input_ids_to_use[last_end_position:]

            new_input_ids = torch.tensor(new_input_ids).unsqueeze(0)
            new_attention_masks_to_use += attention_masks_to_use[last_end_position:]
            new_attention_masks_to_use = torch.tensor(new_attention_masks_to_use).unsqueeze(0)
            new_start_positions = torch.tensor(new_start_positions).unsqueeze(1)
            new_end_positions = torch.tensor(new_end_positions).unsqueeze(1)
            start_embedding_positions_to_use = new_start_positions
            end_embedding_positions_to_use = new_end_positions
            assert new_input_ids.size() == new_attention_masks_to_use.size()
            input_ids_to_use = new_input_ids
            attention_masks_to_use = new_attention_masks_to_use
            example = Example(input_ids_to_use, attention_masks_to_use, start_embedding_positions_to_use,
                              end_embedding_positions_to_use, example.identifier,
                              example.mentions)
    return example

def create_candidate_from_mention(current_embedded_mention: EmbeddedMention, device):
    return CandidateContainerForProcessing(post_mention_embedding=current_embedded_mention.post_mention_embedding,additional_features=torch.tensor((),
                                                                                                                       device=device),
                                                                                      candidate_embedding=current_embedded_mention.processed_mention.mention_embedding,
                                                                                      complex_candidate=ComplexCandidateEntity(candidate=MentionInCluster(
                                                                                          embedded_mention_container=current_embedded_mention,
                                                                                          non_normalized_score = 0.0,
                                                                                          cluster_identifier= current_embedded_mention.mention_container.label_qid,
                                                                                        example= None,
                                                                                        mention_identifier=0,
                                                                                        global_type= None,
                                                                                          mention_container=current_embedded_mention.mention_container
                                                                                      )),
                                                                                      kg_embedding=None,
                                                                                      kg_one_hop=None,
                                                                                      two_hop=None)
def get_nested_candidates(maximum_num_mentions, examples, processed_mentions,
                          mention_split, mentions_to_add_if_full, candidate_manager,
                          number_candidates=None, add_correct_candidate=None):
    mention_idx = 0
    all_nested_candidates = []
    num_candidates_with_mentions = 0
    while mention_idx < maximum_num_mentions:
        old_mention_idx = mention_idx
        sub_sampled_processed_mentions, mention_idx = get_new_mentions_to_link(
            mention_idx,
            examples,
            processed_mentions,
            mentions_to_add_if_full if mention_idx > 0  else mention_split)
        candidates, tmp_num_candidates_with_mentions = candidate_manager.get_candidates(
            examples, sub_sampled_processed_mentions, mention_idx_offset=old_mention_idx,
            number_candidates=number_candidates, add_correct_candidate=add_correct_candidate
        )
        num_candidates_with_mentions += tmp_num_candidates_with_mentions
        all_nested_candidates.append((candidates, sub_sampled_processed_mentions, mention_idx))
    return all_nested_candidates, num_candidates_with_mentions

def get_new_mentions_to_link(mention_idx: int, examples, mention_embeddings, mentions_to_add: int = 1):
    new_mention_embeddings = []
    new_mention_idx = mention_idx + mentions_to_add
    for idx, (example, mention_embeddings_example) in enumerate(zip(examples, mention_embeddings)):
        new_mention_embeddings.append(mention_embeddings_example[mention_idx:new_mention_idx] if example.mentions else [])
    return new_mention_embeddings, new_mention_idx

def update_current_representations(entity_representations, new_embedded_mentions,
                                   current_entity_representations, current_mentions,
                                   mention_split: int):
    for entity_representations_example, new_embedded_mentions_example, current_entity_representations_example, current_mentions_example in zip(entity_representations, new_embedded_mentions, current_entity_representations
                                                                                                                                              , current_mentions):
        for entity_representation, embedded_mention in zip(entity_representations_example, new_embedded_mentions_example):
            if len(current_entity_representations_example) >= mention_split:
                current_entity_representations_example.pop(0)
                current_mentions_example.pop(0)
            current_entity_representations_example.append(list(entity_representation))
            current_mentions_example.append(embedded_mention)

def calculate_stats(linking_decisions: List[dict], ookg_entity_cluster_assignments:List[ClusterAssignment]=None):
    links = []
    is_out_of_kg = []
    labels = []
    count_in_kg = 0
    count_ookg = 0
    count_all = 0
    unique_ookg = set()
    unique_inkg = set()
    no_candidates_counter_in_kg = 0
    no_candidates_counter_ookg = 0
    in_candidate_set = 0
    avg_type_distance = 0
    for example in linking_decisions:
        for mention in example["links"]:
            links.append(mention["link"])
            is_out_of_kg.append(mention["is_out_of_kg"])
            labels.append((mention["qid"], mention["out_of_kg"]))
            if mention["in_candidate_set"]:
                in_candidate_set += 1
            if mention["out_of_kg"]:
                unique_ookg.add(mention["qid"])
                count_ookg += 1
                if mention["no_candidates"]:
                    no_candidates_counter_ookg += 1
            else:
                if "type_distance" in mention:
                    avg_type_distance += mention["type_distance"]

                unique_inkg.add(mention["qid"])
                count_in_kg += 1
                if mention["no_candidates"]:
                    no_candidates_counter_in_kg += 1
            count_all += 1

    is_out_of_kg = torch.tensor(is_out_of_kg)
    (
        correct_ookg,
        incorrect_ookg,
        correct_non_ookg,
        incorrect_non_ookg,
        in_kg_linked_ookg_cluster,
        newly_out_of_kg_in_kg,
        correctly_classified_non_EE,
    ) = count_correctly_classified_out_of_kg(is_out_of_kg, labels, links)

    (
        correctly_linked_in_kg,
        correctly_linked_in_kg_non_ookg_classified,
    ) = count_correctly_classified_in_kg(
        links, labels, correctly_classified_non_EE
    )

    precision_in_kg_non_ookg = (
        correctly_linked_in_kg_non_ookg_classified / (correct_non_ookg + incorrect_ookg)
        if correctly_linked_in_kg_non_ookg_classified
        else 0
    )
    recall_in_kg_non_ookg = correctly_linked_in_kg_non_ookg_classified / count_in_kg
    f_measure_in_kg_non_ookg = calculate_fmeasure(
        precision_in_kg_non_ookg, recall_in_kg_non_ookg
    )

    # Ignores all which are classified as non-out_of_kg but incorrectly linked (not really useful)
    precision_in_kg = (
        correctly_linked_in_kg / (correctly_linked_in_kg + incorrect_ookg)
        if correctly_linked_in_kg
        else 0
    )
    recall_in_kg = correctly_linked_in_kg / count_in_kg
    f_measure_in_kg = calculate_fmeasure(precision_in_kg, recall_in_kg)

    in_kg_identification_accuracy = (
        correct_non_ookg / count_in_kg if count_in_kg else 0
    )
    ookg_identification_accuracy = correct_ookg / count_ookg if count_ookg else 0
    ookg_identification_precision = correct_ookg / (correct_ookg + count_in_kg - correct_non_ookg) if (correct_ookg + count_in_kg - correct_non_ookg) else 0
    identification_harmonic_mean = calculate_fmeasure(
        in_kg_identification_accuracy, ookg_identification_accuracy
    )

    in_kg_linked_to_incorrect_mention_init_by_ookg = 0
    in_kg_linked_to_correct_mention_init_by_inkg = 0
    in_kg_linked_to_incorrect_mention_init_by_inkg = 0
    if ookg_entity_cluster_assignments is not None:
        clusters_created_by_ookg_entity = defaultdict(list)
        clusters_created_by_inkg_entity = defaultdict(list)
        clusters_encountered = set()
        for ookg_entity_cluster_assignment in ookg_entity_cluster_assignments:
            if isinstance(ookg_entity_cluster_assignment.assignment, int):
                if ookg_entity_cluster_assignment.assignment not in clusters_encountered:
                    clusters_encountered.add(ookg_entity_cluster_assignment.assignment)
                    if ookg_entity_cluster_assignment.ground_truth in unique_ookg:
                        clusters_created_by_ookg_entity[ookg_entity_cluster_assignment.assignment].append(ookg_entity_cluster_assignment.ground_truth)
                    else:
                        clusters_created_by_inkg_entity[ookg_entity_cluster_assignment.assignment].append(ookg_entity_cluster_assignment.ground_truth)

                if ookg_entity_cluster_assignment.ground_truth in unique_inkg:
                    if ookg_entity_cluster_assignment.assignment in clusters_created_by_inkg_entity:
                        if ookg_entity_cluster_assignment.ground_truth in clusters_created_by_inkg_entity[ookg_entity_cluster_assignment.assignment]:
                            in_kg_linked_to_correct_mention_init_by_inkg += 1
                        else:
                            in_kg_linked_to_incorrect_mention_init_by_inkg += 1
                    else:
                        in_kg_linked_to_incorrect_mention_init_by_ookg += 1
    return {
        "correct_ookg": correct_ookg,  # All out_of_kg entities correctly classified as out_of_kg
        "incorrect_ookg": incorrect_ookg,  # All out_of_kg entities incorrectly classified as non-out_of_kg
        "correct_non_ookg": correct_non_ookg,  # All non-out_of_kg entities correctly classified as non-out_of_kg
        "incorrect_non_ookg": incorrect_non_ookg,  # All non-out_of_kg entities incorrectly classified as out_of_kg
        "in_kg_linked_ookg_cluster": in_kg_linked_ookg_cluster,
        "in_kg_classified_as_ookg": newly_out_of_kg_in_kg,
        "in_kg_linked_to_correct_mention_init_by_inkg": in_kg_linked_to_correct_mention_init_by_inkg,
        "in_kg_linked_to_incorrect_mention_init_by_inkg": in_kg_linked_to_incorrect_mention_init_by_inkg,
        "in_kg_linked_to_incorrect_mention_init_by_ookg": in_kg_linked_to_incorrect_mention_init_by_ookg,
        "correctly_linked_in_kg": correctly_linked_in_kg,
        # "precision_in_kg": precision_in_kg, # precision not affected by non-ee classification
        # "recall_in_kg": recall_in_kg, # recall not affected by non-ee classification
        # "f_measure_in_kg": f_measure_in_kg,
        "precision_in_kg_non_ookg": precision_in_kg_non_ookg,
        # Regular precision considering in-KG entities marked as out_of_kg or not (Correct interpretation)
        "recall_in_kg_non_ookg": recall_in_kg_non_ookg,
        # Regular recall considering in-KG entities marked as out_of_kg or not
        "f_measure_in_kg_non_ookg": f_measure_in_kg_non_ookg,  # F1 of both above
        "count_in_kg": count_in_kg,
        "count_ookg": count_ookg,
        "count_all": count_all,
        "count_unique_in_kg": len(unique_inkg),
        "count_unique_ookg": len(unique_ookg),
        "in_kg_identification_accuracy": in_kg_identification_accuracy,
        "ookg_identification_accuracy": ookg_identification_accuracy,
        "ookg_identification_precision": ookg_identification_precision,
        "identification_harmonic_mean": identification_harmonic_mean,
        "no_candidates_counter_in_kg": no_candidates_counter_in_kg,
        "no_candidates_counter_ookg": no_candidates_counter_ookg,
        "gold_recall": in_candidate_set / count_in_kg,
        "gold_recall_all": in_candidate_set / count_all,
        "avg_type_distance": avg_type_distance/count_in_kg
    }


def read_dump(filepath: Path, fields_to_be_printed: set = None):
    print(f"Reading {filepath}")

    if fields_to_be_printed is None:
        fields_to_be_printed = {"evaluation_results"}

    content = torch.load(filepath)

    for field in fields_to_be_printed:
        content_of_field = content[field]
        if isinstance(content_of_field, dict):
            print(json.dumps(content_of_field, indent=4))
        else:
            print(content_of_field)


def get_entity_overlap(dataset_1_path: Path, dataset_2_path: Path):
    dataset_1 = jsonlines.open(dataset_1_path)
    dataset_2 = jsonlines.open(dataset_2_path)

    entities_dataset_1 = set()
    entities_dataset_2 = set()

    for item in dataset_1:
        for x in item["target_page_ids"]:
            entities_dataset_1.add(x)

    for item in dataset_2:
        for x in item["target_page_ids"]:
            entities_dataset_2.add(x)

    return entities_dataset_1.intersection(entities_dataset_2)


# def calculate_entity_overlap(dataset_1_path: Path, dataset_2_path: Path):
#     intersection = get_entity_overlap(dataset_1_path,dataset_2_path)
#
#     intersection_ratio_1 = len(intersection) / len(entities_dataset_1)
#     intersection_ratio_2 = len(intersection) / len(entities_dataset_2)
#
#     dataset_2 = jsonlines.open(dataset_2_path)
#     counter = 0
#     for item in dataset_2:
#         for x in item["target_page_ids"]:
#             if x in intersection:
#                 counter += 1
#
#     print(len(intersection))
#     print(intersection_ratio_2)
#     print(counter)

def transform_entity(label: str, description: str, lower_text: bool = False):
    description = description.lower() if lower_text else description

    return (
        f"{label.lower() if lower_text else label}"
        + f" {SEP} {description}"
    )

def encode_fully_connected(combined_strings, entity_tokenizer, max_length_description_encoder: int = 32):
    tokenized = entity_tokenizer(
        combined_strings,
        padding='max_length',
        return_tensors="pt",
        max_length=max_length_description_encoder,
        truncation=True,
        return_offsets_mapping=True,
    )
    return tokenized["input_ids"], tokenized["attention_mask"]


def calculate_similarities(document_containers: List[DocumentContainerForProcessing], ranking_model: SupervisedRankingModel):
    embedding_a = []
    embedding_b = []
    edit_distances = []
    type_distances = []
    for idx, embedded_document in enumerate(
            document_containers
    ):
        for mention in embedded_document.mentions:
            for idx, candidate in enumerate(mention.candidate_representations):
                embedding_a.append(mention.embedded_mention.processed_mention.mention_embedding)
                embedding_b.append(candidate.candidate_embedding)
                if not candidate.complex_candidate.is_kg_candidate:
                    edit_distances.append(calculate_syntactical_similarity(mention.embedded_mention.mention_container.mention,
                                                                           candidate.complex_candidate.candidate.mention_container.mention))
                else:
                    edit_distances.append(None)
                if candidate.complex_candidate.is_kg_candidate:
                    type_distances.append(torch.cdist(mention.embedded_mention.processed_mention.type_prediction.unsqueeze(0), candidate.kg_one_hop.unsqueeze(0)))
                else:
                    type_distances.append(None)

    if embedding_a:
        v1 = torch.stack(embedding_a)
        v2 = torch.stack(embedding_b)
        entity_similarities = ranking_model.calculate_pairwise_scores(v1, v2)
    else:
        entity_similarities = torch.zeros((0,1))


    offset = 0
    for idx, embedded_document in enumerate(
            document_containers):

        for mention in embedded_document.mentions:
            l = len(mention.candidate_representations)
            new_candidate_representations = []
            for candidate, entity_similarity, type_distance, ed in zip(mention.candidate_representations,
                                                    entity_similarities[offset: l + offset],
                                                        type_distances[offset: l + offset],
                                                    edit_distances[offset: l + offset]
                                                                            ):
                new_candidate_representations.append(CandidateContainerWrapper(candidate, entity_similarity,
                                                                               ed, type_distance))
            mention.candidate_representations = new_candidate_representations
            offset += l


def pairwise_loss(x_1: torch.Tensor, x_2: torch.Tensor, y: torch.Tensor, margin=1.0):
    distances = torch.cdist(x_1.unsqueeze(0), x_2)
    loss = 0.5 * (y * torch.square(distances) + (1 - y) * torch.square(torch.clamp(margin - distances, min=0.0)))
    return torch.mean(loss)

def pairwise_loss_weighted(x_1: torch.Tensor, x_2: torch.Tensor, y: torch.Tensor, margin=1.0):
    num_positive_pairs = torch.sum(y)
    num_negative_pairs = y.size(0) - num_positive_pairs
    weight_positive = (1 / ( 2 * num_positive_pairs)) if num_positive_pairs else 0.0
    weight_negative = (1 / ( 2 * num_negative_pairs)) if num_negative_pairs else 0.0
    weights = torch.ones(y.size(), device=x_1.device) * torch.tensor(weight_positive)
    weights[y == 0] = weight_negative
    weights = weights.to(x_1.device)
    distances = torch.cdist(x_1.unsqueeze(0), x_2)
    loss = 0.5 * (y * torch.square(distances) + (1 - y) * torch.square(torch.clamp(margin - distances, min=0.0)))
    loss = weights * loss
    return torch.sum(loss)

def create_type_index_list(decisive_type_dict: dict, type_dicts: List[dict]):
    type_indices = defaultdict(int)
    for type_dict in type_dicts:
        for key, value in type_dict.items():
            if key in decisive_type_dict:
                type_indices[decisive_type_dict[key]] += value
    return list(type_indices.items())

def get_labels_only():
    items = jsonlines.open("items.jsonl")
    labels = jsonlines.open("labels.json", "w")

    for item in tqdm(items, total=81966486):
        del item["instance_of"]
        del item["subclass_of"]
        labels.write(item)


def split_candidates(candidate_entities, kg_connector):
    labels = []
    descriptions = []
    triples = []

    for candidate in candidate_entities:
        entity_info = kg_connector.get_entity(candidate)
        if "labelized_out_claims" in entity_info:
            triples.append([f'{x[0]} {x[1]}' for x in entity_info["labelized_out_claims"]])
        labels.append([entity_info["labels"]] + entity_info["aliases"])
        descriptions.append(entity_info['descriptions'])

    return labels, descriptions, triples


def calculate_syntactical_similarity(x: str, y: str) -> float:
    return fuzz.ratio(x, y) / 100 # 1 - edit_distance(x, y) / max(len(x), len(y))

def calculate_partial_syntactical_similarity(x: str, y: str) -> float:
    return fuzz.partial_ratio(x, y) / 100 # 1 - edit_distance(x, y) / max(len(x), len(y))

def is_true(x):
    return str(x).lower() in {"true", "1"}

def bert_embed_document(device, document_model, examples) -> Tuple[torch.Tensor, List[List[ProcessedMention]]]:
    input_ids = []
    attention_masks = []
    start_embedding_positions = []
    end_embedding_positions = []
    for example in examples:
        input_ids.append(example.input_ids)
        attention_masks.append(example.attention_masks)
        start_embedding_positions.append(example.start_embedding_positions)
        end_embedding_positions.append(example.end_embedding_positions)
    max_len = max(x.size(1) for x in input_ids)
    input_ids = [torch.cat((x, torch.ones(1, max_len - x.size(1), dtype=torch.int64)), dim=1) for x in input_ids]
    attention_masks = [torch.cat((x, torch.zeros(1, max_len - x.size(1), dtype=torch.int64)), dim=1) for x in attention_masks]
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    number_of_mentions  = [y.size(0) for y in start_embedding_positions]
    max_num_mentions = max(number_of_mentions)
    start_embedding_positions_tensor = torch.stack(
        [
            torch.cat(
                (x, -torch.ones((max_num_mentions - x.size(0), 1), dtype=torch.int64)),
                dim=0,
            )
            for x in start_embedding_positions
        ],
        dim=0,
    )
    end_embedding_positions_tensor = torch.stack(
        [
            torch.cat(
                (x, -torch.ones((max_num_mentions - x.size(0), 1), dtype=torch.int64)),
                dim=0,
            )
            for x in end_embedding_positions
        ],
        dim=0,
    )
    embedded_documents = document_model(
        input_ids, attention_masks, start_embedding_positions=start_embedding_positions_tensor,
        end_embedding_positions=end_embedding_positions_tensor
    )
    cls_embeddings, mention_embeddings, type_predictions  = embedded_documents
    nested_processed_mentions = []
    offset = 0
    for num in number_of_mentions:
        processed_mentions = [ProcessedMention(mention_embeddings[idx, :], type_predictions[idx, :]) for idx in range(offset, offset + num)]
        nested_processed_mentions.append(processed_mentions)
        offset += num
    return cls_embeddings, nested_processed_mentions