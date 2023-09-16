from collections import defaultdict, Counter
from typing import List, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from src.evaluation.clustering_tools import compute_cluster_assignments_from_results
from src.utilities.utilities import calculate_stats, create_candidate_from_mention, pairwise_loss, calculate_similarities
from src.utilities.various_dataclasses import Result, MentionContainerForProcessing, DocumentContainerForProcessing, \
    CandidateContainerWrapper

ranking_loss = CrossEntropyLoss()


def calculate_loss(mention_comparator_model, results: List[Result], ranking_model, include_ookg_score: bool,
                   alternate_mention_embedding: bool,
                   device):
    ce_labels = []
    valid_probs = []
    ookg_entities = defaultdict(list)
    for result in results:
        if result.mention.mention_container.label_qid in result.candidates:
            valid_probs.append(torch.tensor(result.non_normalized_scores))
            idx = [x.complex_candidate.qid for x in result.candidates].index(result.mention.mention_container.label_qid)
            ce_labels.append(torch.tensor([idx]))
        elif include_ookg_score:
            valid_probs.append(torch.tensor(result.non_normalized_scores))
            ce_labels.append(torch.tensor([len(result.non_normalized_scores) - 1]))
        if result.mention.mention_container.label_out_of_kg:
            ookg_entities[result.mention.mention_container.label_qid].append(result)

    mention_loss = torch.zeros((), device=device)
    counter = 0
    correct = 0
    accuracy_counter = 0
    if alternate_mention_embedding:
        ookg_entities = list(ookg_entities.items())
        for idx, (qid, mentions) in tqdm(enumerate(ookg_entities)):
            if len(mentions) > 1:
                main_mention = mentions[0]
                other_correct_mentions = mentions[1:]
                other_mentions = ookg_entities[:idx] + ookg_entities[idx + 1:]
                mention_candidates = [x[0] for _, x in other_mentions]
                mention_candidates += other_correct_mentions
                mention_candidates = [create_candidate_from_mention(x.mention, device) for x in
                                      mention_candidates]
                x_2 = []
                y = []
                ground_truth_indices = []
                for idx_, candidate in enumerate(mention_candidates):
                    x_2.append(candidate.post_mention_embedding)
                    y.append(torch.tensor(0,
                                          device=device) if candidate.complex_candidate.qid != main_mention.mention.mention_container.label_qid else torch.tensor(
                        1, device=device))
                    if candidate.complex_candidate.qid == main_mention.mention.mention_container.label_qid:
                        ground_truth_indices.append(idx_)
                x_2 = torch.stack(x_2)
                distances = torch.cdist(main_mention.mention.post_mention_embedding.unsqueeze(0), x_2)
                if ground_truth_indices:
                    sorted_indices = torch.argsort(distances).squeeze()
                    for idx_ in ground_truth_indices:
                        if idx_ in sorted_indices[:len(ground_truth_indices)]:
                            correct += 1
                        accuracy_counter += 1
                y = torch.stack(y)
                mention_loss = pairwise_loss(main_mention.mention.post_mention_embedding, x_2, y)
                counter += 1
    else:
        ookg_entities = list(ookg_entities.items())
        mention_containers = []
        ground_truth_indices_all = []

        for idx, (qid, mentions) in enumerate(ookg_entities):
            if len(mentions) > 1:
                main_mention = mentions[0]
                other_correct_mentions = mentions[1:]
                other_mentions = ookg_entities[:idx] + ookg_entities[idx + 1:]
                mention_candidates = [x[0] for _, x in other_mentions]
                ground_truth_indices = list(
                    range(len(mention_candidates), len(mention_candidates) + len(other_correct_mentions)))
                mention_candidates += other_correct_mentions
                mention_candidates = [create_candidate_from_mention(x.mention, device) for x in
                                      mention_candidates]
                mention_container_for_processing = MentionContainerForProcessing(
                    main_mention.mention, mention_candidates
                )
                mention_containers.append(mention_container_for_processing)
                ground_truth_indices_all.append(ground_truth_indices)
        document_container = DocumentContainerForProcessing(None, mention_containers)
        calculate_similarities([document_container], ranking_model)
        for mention, ground_truth_indices in zip(document_container.mentions, ground_truth_indices_all):
            scores = mention_comparator_model(mention, mention.candidate_representations)
            for g_i in ground_truth_indices:
                mention_loss += ranking_loss(scores, torch.tensor(g_i, device=device)) / len(
                    ground_truth_indices)
            if ground_truth_indices:
                sorted_indices = torch.argsort(scores, descending=True).squeeze()
                for idx_ in ground_truth_indices:
                    if idx_ in sorted_indices[:len(ground_truth_indices)]:
                        correct += 1
                    accuracy_counter += 1
            counter += 1

    if counter > 0:
        mention_loss /= counter

    accuracy = 0.0
    if accuracy_counter > 0:
        accuracy = correct / accuracy_counter
    loss = torch.zeros((), device=device)
    for probs_, ce_label in zip(valid_probs, ce_labels):
        loss += ranking_loss(probs_.T, ce_label)
    if len(valid_probs) > 0:
        loss /= len(valid_probs)

    return loss, mention_loss, accuracy


def _compute_mention_cluster_assignment(ookg_detected_mentions: List[Result], all_ookg_mentions: list):
    # compute count of actual linked entities per cluster
    mention_cluster_entity_counts = []
    mention_clusters = defaultdict(list)
    for result in ookg_detected_mentions:
        mention_clusters[result.link].append(result)

    cluster_mentions = []
    for cluster_id, mentions in mention_clusters.items():
        cluster_mentions.append(mentions)
    for mentions in cluster_mentions:
        unknown_entity_counts = Counter([ent.mention.mention_container.label_qid for ent in mentions if ent.mention.mention_container.label_out_of_kg])
        mention_cluster_entity_counts.append(unknown_entity_counts)

    unknown_entities = list({ent.mention.mention_container.label_qid for ent in all_ookg_mentions if ent.mention.mention_container.label_out_of_kg})
    # create cost matrix for every cluster based on entity counts (use negatives as we want to maximize entity hit
    # s)
    unknown_entity_indices = {ent: idx for idx, ent in enumerate(unknown_entities)}
    mention_cluster_costs = np.zeros((len(cluster_mentions), len(unknown_entities)))
    for cluster_idx, ent_counts in enumerate(mention_cluster_entity_counts):
        for ent_id, cnt in ent_counts.items():
            mention_cluster_costs[cluster_idx, unknown_entity_indices[ent_id]] = -cnt

    # find optimal assignment of entities to clusters
    cluster_entities = [None] * len(cluster_mentions)
    for cluster_idx, entity_idx in zip(*linear_sum_assignment(mention_cluster_costs)):
        ent_id = unknown_entities[entity_idx]
        if mention_cluster_entity_counts[cluster_idx][ent_id] == 0:
            # discard assignment of entity to cluster if no mention in the cluster is linked to the entity
            continue
        cluster_entities[cluster_idx] = ent_id
    return cluster_mentions, cluster_entities


def calculate_prf1_ookg(all_results: List[Result]):
    all_ookg_mentions = [result for result in all_results if result.mention.mention_container.label_out_of_kg]
    ookg_detected_mentions = [result for result in all_results if result.is_ookg]
    inkg_detected_mentions_being_ookg = [result for result in all_results if not result.is_ookg and result.mention.mention_container.label_out_of_kg]
    mention_clusters, mention_cluster_assignment = _compute_mention_cluster_assignment(ookg_detected_mentions, all_ookg_mentions)
    tp = 0
    fp = 0

    for mentions, ent in zip(mention_clusters, mention_cluster_assignment):
        for mention in mentions:
            if mention.mention.mention_container.label_qid == ent:
                tp += 1
            else:
                fp += 1
    fn = len(inkg_detected_mentions_being_ookg)
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    if tp + fp > 0:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1



def calculate_final_results(all_results: List[Result], mention_comparator_model, ranking_model, candidate_manager,
                            args: dict,
                            device,
                            detailed_info: bool, skip_loss_calculation: bool = False,
                            skip_clustering_calculation: bool = True):
    linking_decisions = []

    results_to_use = all_results
    ookg_precision, ookg_recall, ookg_f1 = calculate_prf1_ookg(all_results)
    cluster_stats, ookg_entity_cluster_assignments = compute_cluster_assignments_from_results(
        results_to_use, skip_clustering_calculation)

    linking_decisions_sub = []
    for result in all_results:
        type_distance = 0.0
        non_out_of_kg_but_out_of_kg_classified = not result.mention.mention_container.label_out_of_kg and bool(
            result.is_ookg)
        out_of_kg_but_non_out_of_kg_classified = result.mention.mention_container.label_out_of_kg and not bool(
            result.is_ookg)
        if isinstance(result.link, CandidateContainerWrapper):
            link = result.link.complex_candidate
        else:
            link = result.link
        wrongly_linked = result.mention.mention_container.label_qid != result.link

        linking_decisions_sub.append({
            "qid": result.mention.mention_container.label_qid,
            "out_of_kg": result.mention.mention_container.label_out_of_kg,
            "is_out_of_kg": bool(result.is_ookg),
            "link": link.qid if not isinstance(link, (str, int)) else link,
            "prob": result.action_prob,
            "candidates": [(candidate.complex_candidate.qid, candidate.complex_candidate.is_kg_candidate,
                            not candidate.complex_candidate.is_kg_candidate) for candidate in result.candidates],
            "non_normalized_scores": result.non_normalized_scores,
            "non_normalized_score": result.non_normalized_score,
            "type_distance": float(type_distance),
            "mention": result.mention.mention_container.mention,
            "no_candidates": not result.candidates,
            "in_candidate_set": result.mention.mention_container.label_qid in result.candidates,
            "non_out_of_kg_but_out_of_kg_classified": non_out_of_kg_but_out_of_kg_classified,
            "out_of_kg_but_non_out_of_kg_classified": out_of_kg_but_non_out_of_kg_classified,
            "wrongly_linked": wrongly_linked,
        })
    linking_decisions.append({
        "text": "",
        "links": linking_decisions_sub
    })

    if skip_loss_calculation:
        loss = 0.0
        mention_loss = 0.0
        accuracy = 0.0
    else:
        loss, mention_loss, accuracy = calculate_loss(mention_comparator_model, all_results, ranking_model, args.get("include_ookg_score"),
                                                      args.get("alternate_mention_embedding"), device)
    stats = calculate_stats(linking_decisions, ookg_entity_cluster_assignments)
    stats = {**stats,
             **cluster_stats,
                "ookg_precision": ookg_precision,
                "ookg_recall": ookg_recall,
                "ookg_f1": ookg_f1,
             "combined_f1": (stats["f_measure_in_kg_non_ookg"] + ookg_f1) / 2,
             "number_ookg_entities": len(candidate_manager.ookg_representations),
             "loss": float(loss),
             "mention_loss": float(mention_loss),
             "mention_accuracy": float(accuracy),
             "args": {key: str(value) for key, value in args.items()}
             }

    if detailed_info:
        linking_decisions = [{"text": "", "links": decision["links"]} for decision in linking_decisions]

        detailed_info_dict = {
            "ookg_entities": [x.to_dict() for x in candidate_manager.ookg_representations.values()],
            "linking_decisions": linking_decisions}
        return stats, detailed_info_dict
    else:
        return stats, {}