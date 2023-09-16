import csv
import math
from collections import defaultdict
from typing import Tuple, List, Union, Optional

import networkx
import torch
from neleval.evaluate import Evaluate
from networkx import DiGraph
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, pair_confusion_matrix
from tqdm import tqdm
from src.model.mention_comparison import MentionComparator
from src.model.ranking_models import SupervisedRankingModel
from src.utilities.utilities import calculate_syntactical_similarity
from src.utilities.various_dataclasses import MentionInCluster, Result, EmbeddedMention, ClusterAssignment, \
    CandidateContainerWrapper


def build_graph(pairwise_scores: torch.Tensor, mention_pairs: List[Tuple[Union[int, str], Union[int, str]]],
                threshold: float,
                mention_mention_threshold: float,
                add_main_mention: bool = True):
    graph = DiGraph()
    all_entities = set()
    all_mentions = set()
    if add_main_mention:
        graph.add_node(-1)
        all_mentions.add(-1)
    for score, pair in zip(pairwise_scores, mention_pairs):
        if isinstance(pair[0], str) or isinstance(pair[1], str):
            threshold_to_use = threshold
        else:
            threshold_to_use = mention_mention_threshold

        second_node = None
        first_node = None
        both_are_mentions = True
        if isinstance(pair[0], (int, MentionInCluster)):
            all_mentions.add(pair[0])
            second_node = pair[0]
        else:
            all_entities.add(pair[0])
            first_node = pair[0]
            both_are_mentions = False
        if isinstance(pair[1], (int, MentionInCluster)):
            all_mentions.add(pair[1])
            if second_node is None:
                second_node = pair[1]
            else:
                first_node = pair[1]
        else:
            all_entities.add(pair[1])
            first_node = pair[1]
            both_are_mentions = False

        graph.add_node(pair[0])
        graph.add_node(pair[1])

        assert first_node is not None
        if score >= threshold_to_use:
            graph.add_edge(first_node, second_node, score=score)
            if both_are_mentions:
                graph.add_edge(second_node, first_node, score=score)

    return graph, all_entities, all_mentions

def create_cluster_template():
    return {"ookg": False,
            "link": "",
            "mentions": list()}




def graph_to_clusters(all_mentions, graph: DiGraph, original_graph: DiGraph):
    cluster_counter = 0
    mentions_already_handled = set()
    clusters = defaultdict(create_cluster_template)
    for mention in all_mentions:
        if mention in mentions_already_handled:
            continue
        mentions_found, entities_found = extract_subgraph(graph, mention)
        mentions_already_handled.update(mentions_found)
        assert len(entities_found) <= 1
        out_of_kg = len(entities_found) == 0
        entity_found = entities_found.pop() if not out_of_kg else cluster_counter
        for mention_ in mentions_found:
            score = original_graph.get_edge_data(entity_found, mention_, {"score": 0})["score"]
            clusters[entity_found]["mentions"].append((mention_, score))
            clusters[entity_found]["link"] = entity_found
            clusters[entity_found]["ookg"] = out_of_kg
        if out_of_kg:
            cluster_counter += 1
    assert len({mention for cluster in clusters.values() for mention in cluster["mentions"] }) == len(all_mentions)
    return list(clusters.values())


def calculate_pairwise_scores_comparator(a: List[EmbeddedMention], b: List[EmbeddedMention],
                                         ranking_model: SupervisedRankingModel,
                                         mention_comparator_model: MentionComparator, device,
                                         batch_size=10000):
    print("Calculate pairwise scores")
    print("Calculate context scores")
    a_ = [x.processed_mention.mention_embedding for x in a]
    b_ = [x.processed_mention.mention_embedding for x in b]
    n_a = len(a_)
    pairwise_scores_embeddings = torch.zeros(n_a)
    for i in tqdm(range(0, len(a), batch_size)):
        v1_batch = torch.stack(a_[i:i + batch_size]).to(device)
        v2_batch = torch.stack(b_[i:i + batch_size]).to(device)

        i_batch_size = v1_batch.shape[0]
        pairwise_scores_embeddings_batch = ranking_model.calculate_pairwise_scores(v1_batch, v2_batch).to("cpu")
        pairwise_scores_embeddings[i:i + i_batch_size] = pairwise_scores_embeddings_batch

    print("Calculate edit scores")
    # Set cosine scores to 0 if deactivated
    a_ = [x.mention_container.mention for x in a]
    b_ = [x.mention_container.mention for x in b]
    pairwise_scores_edit = torch.stack([torch.tensor(calculate_syntactical_similarity(x, y)) for x, y in zip(a_, b_)])
    pairwise_scores_edit = pairwise_scores_edit.to(device)

    print("Calculate final scores")
    pairwise_scores = []
    num_instances = len(a)

    for i in range(0, num_instances, batch_size):
        a_batch = a[i:i + batch_size]
        b_batch = b[i:i + batch_size]
        pairwise_scores_embeddings_batch = pairwise_scores_embeddings[i:i + batch_size]
        pairwise_scores_edit_batch = pairwise_scores_edit[i:i + batch_size]

        batch_pairwise_scores = mention_comparator_model.calculate_pairwise_scores(a_batch, b_batch,
                                                                                   pairwise_scores_embeddings_batch.to(device),
                                                                                   pairwise_scores_edit_batch.to(device))
        pairwise_scores.extend(batch_pairwise_scores.to("cpu"))

    return torch.tensor(pairwise_scores)

def identify_best_threshold(a: List[EmbeddedMention], b:List[EmbeddedMention], pairwise_scores) -> torch.Tensor:
    neg_infty = lambda: -math.inf
    pos_infty = lambda: math.inf
    max_negative = defaultdict(neg_infty)
    min_positive = defaultdict(pos_infty)
    for a_i, b_i, score in zip(a, b, pairwise_scores):
        if not torch.any(torch.isnan(score)):
            if a_i.mention_container.label_qid == b_i.mention_container.label_qid:
                if score < min_positive[a_i.mention_container.label_qid]:
                    min_positive[a_i.mention_container.label_qid] = score
            else:
                if score > max_negative[a_i.mention_container.label_qid]:
                    max_negative[a_i.mention_container.label_qid] = score
                if score > max_negative[b_i.mention_container.label_qid]:
                    max_negative[b_i.mention_container.label_qid] = score
    overall_average = 0
    counter  = 0
    for key, value in min_positive.items():
        if key in max_negative:
            local_average = (value + max_negative[key]) / 2
            counter += 1

            overall_average += local_average
    return overall_average/counter

def clustering(threshold: torch.Tensor, pairwise_scores: torch.Tensor, ookg_entities: List[Result],
               clustering_type:str, device, alternate_mention_embedding:bool, use_dbscan: bool = True,
               ):
    pairwise_scores = pairwise_scores.to("cpu")
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to("cpu")

    maximum_score = torch.max(pairwise_scores[torch.logical_not(torch.isnan(pairwise_scores))])
    minimum_score = torch.min(pairwise_scores[torch.logical_not(torch.isnan(pairwise_scores))])
    pairwise_scores[torch.isnan(pairwise_scores)] = -10000
    new_pairwise_scores = pairwise_scores - minimum_score
    new_pairwise_scores = (maximum_score - minimum_score) - new_pairwise_scores
    if isinstance(threshold, torch.Tensor):
        threshold = torch.clone(threshold)
    threshold -= minimum_score
    threshold = max((maximum_score - minimum_score) - threshold, 0.00001)
    new_pairwise_scores = new_pairwise_scores.to(device)
    A = torch.zeros(len(ookg_entities), len(ookg_entities), device=device)
    i, j = torch.triu_indices(len(ookg_entities), len(ookg_entities))
    A[i, j] = new_pairwise_scores
    A.T[i, j] = new_pairwise_scores
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to("cpu")
        threshold = threshold.detach().numpy()
    if use_dbscan:
        clustering_method = DBSCAN(metric="precomputed", eps=float(threshold), min_samples=1)
    else:
        clustering_method = AgglomerativeClustering(affinity="precomputed", linkage=clustering_type, distance_threshold=threshold, n_clusters=None)
    A = A.to("cpu")
    A = A.detach().numpy()
    clustered = clustering_method.fit_predict(A)

    clusters = defaultdict(create_cluster_template)
    for idx, (mention, cluster_id) in enumerate(zip(ookg_entities, clustered)):
        clusters[cluster_id]["mentions"].append((idx, mention.non_normalized_score))
        clusters[cluster_id]["link"] = cluster_id
        clusters[cluster_id]["ookg"] = True
    return list(clusters.values())

def create_clustering_outputs(ookg_entity_cluster_assignments: list) -> Tuple[list, list]:
    ground_truth = []
    predicted = []
    for mention_idx, item in enumerate(ookg_entity_cluster_assignments):
        ground_truth.append([0, mention_idx * 2, mention_idx * 2 + 1, item.ground_truth, 1.0, "MISC"])
        predicted.append([0, mention_idx * 2, mention_idx * 2 + 1, str(item.assignment), 1.0, "MISC"])
    return ground_truth, predicted

def ookg_clustering(pairwise_scores, ookg_entities: List[Result], threshold: float = None,
                    device=None, clustering_type="average", alt_clustering=False, alternate_mention_embedding=False):

    if len(ookg_entities) == 1:
        content = ookg_entities[0]
        return [Result(0, content.mention, content.is_ookg, content.non_normalized_scores, content.non_normalized_score, content.action_prob, [], None, content.candidates)], threshold
    a = []
    b = []
    mention_pairs: List[Tuple[int, int]] = []
    candidate_mapping = {}
    for idx, candidate in enumerate(ookg_entities):
        candidate_mapping[idx] = candidate
        for idx_, candidate_ in enumerate(ookg_entities[idx:]):
            a_id = idx
            b_id = idx + idx_
            mention_pairs.append((a_id, b_id))
            a.append(candidate.mention)
            b.append(candidate_.mention)

    assert len(a) == len(b)

    if threshold is None:
        print("Determine threshold")
        threshold = identify_best_threshold(a, b, pairwise_scores)

    if alt_clustering:
        clusters = clustering(threshold, pairwise_scores, ookg_entities, clustering_type, device, alternate_mention_embedding)
    else:
        graph, all_entities, all_mentions = build_graph(pairwise_scores,
                                                             mention_pairs,
                                                             threshold, threshold, add_main_mention=False)

        original_graph = graph.copy()

        sorted_edges = list(sorted(graph.edges(data="score"), key=lambda x: x[2]))

        for edge in sorted_edges:
            entity_is_in_cluster = entity_in_cluster(graph, edge[1], all_entities)
            if entity_is_in_cluster is not None:
                remove_edge(graph, edge[0], edge[1])
                if entity_in_cluster(graph, edge[1], all_entities) is None:
                    graph.add_edge(edge[0], edge[1], score=edge[2])

        clusters = graph_to_clusters(all_mentions, graph, original_graph)

    new_results = []
    for idx, cluster in enumerate(clusters):
        for mention_id, _ in cluster["mentions"]:
            content = candidate_mapping[mention_id]
            new_results.append((mention_id, Result(idx, content.mention, content.is_ookg, content.non_normalized_scores, content.non_normalized_score, content.action_prob,[], None, content.candidates)))
    new_results = sorted(new_results, key= lambda x: x[0])
    new_results = [x[1] for x in new_results]
    return new_results, threshold

def create_tab_delimited_clustering_file(results ,filename: str):
    with open(filename, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerows(results)


def calculate_clustering_scores(ookg_entity_cluster_assignments, suffix: str = "",
                                skip_computation_of_extended_clustering_metrics: bool = False):
    ground_truth_same_cluster_assignments = defaultdict(list)
    predicted_same_cluster_assignments = defaultdict(list)
    for cluster_assignment in ookg_entity_cluster_assignments:
        ground_truth_same_cluster_assignments[
            cluster_assignment.ground_truth
        ].append(cluster_assignment.assignment)
        predicted_same_cluster_assignments[cluster_assignment.assignment].append(
            cluster_assignment.ground_truth
        )
    labels = []
    int_labels_dict = {}
    for x in ookg_entity_cluster_assignments:
        if x.ground_truth not in int_labels_dict:
            int_labels_dict[x.ground_truth] = len(int_labels_dict)
        labels.append(int_labels_dict[x.ground_truth])
    predicted = [x.assignment for x in ookg_entity_cluster_assignments]
    adjusted_rand_index = adjusted_rand_score(labels, predicted)
    pair_confusion_m = pair_confusion_matrix(labels, predicted)
    tn = pair_confusion_m[0,0]
    fn = pair_confusion_m[1, 0]
    tp = pair_confusion_m[1, 1]
    fp = pair_confusion_m[0, 1]
    nmi = normalized_mutual_info_score(labels, predicted)
    f1 = 2 * (adjusted_rand_index * nmi) / (adjusted_rand_index + nmi) if (adjusted_rand_index + nmi) > 0  else 0

    if not skip_computation_of_extended_clustering_metrics:
        ground_truth_clustering, predicted_clustering = create_clustering_outputs(ookg_entity_cluster_assignments)

        create_tab_delimited_clustering_file(ground_truth_clustering, "ground_truth.tsv")
        create_tab_delimited_clustering_file(predicted_clustering, "predicted.tsv")

        evaluation = Evaluate("predicted.tsv", "ground_truth.tsv", {"b_cubed", "mention_ceaf", "entity_ceaf", "muc"})
        clustering_results = evaluation()
        b_cubed = clustering_results["b_cubed"]["fscore"]
        mention_ceaf = clustering_results["mention_ceaf"]["fscore"]
        entity_ceaf = clustering_results["entity_ceaf"]["fscore"]
        muc = clustering_results["muc"]["fscore"]
    else:
        b_cubed = 0.0
        mention_ceaf = 0.0
        entity_ceaf = 0.0
        muc = 0.0

    return {
        f"adjusted_rand_index_{suffix}": adjusted_rand_index,
        f"nmi_{suffix}": nmi,
        f"ari_nmi_f1_{suffix}": f1,
        f"b_cubed_{suffix}": b_cubed,
        f"mention_ceaf_{suffix}": mention_ceaf,
        f"entity_ceaf_{suffix}": entity_ceaf,
        f"muc_{suffix}": muc,
        f"tn_{suffix}": int(tn),
        f"fn_{suffix}": int(fn),
        f"tp_{suffix}": int(tp),
        f"fp_{suffix}": int(fp)
    }


def has_path(graph: DiGraph, node, nodes: set, already_found = None):
    if already_found is None:
        already_found = set()
    if not nodes:
        return None
    already_found.update(nodes)
    new_nodes = set()
    for entity in nodes:
        for neighbor in networkx.neighbors(graph, entity):
            if neighbor not in already_found:
                if neighbor == node:
                    return entity
                new_nodes.add(neighbor)
    return has_path(graph, node, new_nodes, already_found)

def has_path_alternative(graph: DiGraph, node, entities: set):
    already_found = set()
    neighbors = [node]

    while neighbors:
        node = neighbors.pop()
        all_neighbors = networkx.all_neighbors(graph, node)
        for x in all_neighbors:
            if x not in already_found:
                if x in entities:
                    return x
                already_found.add(x)
                neighbors.append(x)
    return None

def entity_in_cluster(graph, node, all_entities) -> Optional[str]:
    return has_path(graph, node, all_entities)

def remove_edge(graph, source: str, target: str):
    graph.remove_edge(source, target)

def extract_subgraph(graph: DiGraph, mention: str, mentions_found = None, entities_found=None):
    if mentions_found is None:
        mentions_found = set()
    if entities_found is None:
        entities_found = set()

    if not isinstance(mention, str):
        mentions_found.add(mention)
    neighbors = networkx.all_neighbors(graph, mention)

    extend_with = []
    for neighbor in neighbors:
        if neighbor not in mentions_found and neighbor not in entities_found:
            if isinstance(neighbor, str):
                entities_found.add(neighbor)
            else:
                mentions_found.add(neighbor)
            extend_with.append(neighbor)

    if extend_with:
        for neighbor in extend_with:
            extract_subgraph(graph, neighbor, mentions_found, entities_found)

    return mentions_found, entities_found


def compute_cluster_assignments_from_results(results: List[Result],
                                             skip_computation_of_extended_clustering_metrics: bool,
                                             meta_suffix: str = ""):
    ookg_entity_cluster_assignments = []
    ookg_entity_cluster_assignments_only_ookg_identified = []
    ookg_entity_cluster_assignments_only_inkg_identified = []
    cluster_assignments_ookg = []
    cluster_assignments_inkg = []

    cluster_assignments_all = []

    for result in results:
        ookg_detected = isinstance(result.link, (str, int))
        if isinstance(result.link, (int, str)):
            link = result.link
        else:
            if isinstance(result.link, CandidateContainerWrapper):
                link = result.link.complex_candidate.qid
            else:
                link = result.link.qid

        if isinstance(link, int):
            if result.mention.mention_container.label_out_of_kg and ookg_detected:
                ookg_entity_cluster_assignments_only_ookg_identified.append(
                    ClusterAssignment(link, result.mention.mention_container.label_qid)
                )
            if not result.mention.mention_container.label_out_of_kg and ookg_detected:
                ookg_entity_cluster_assignments_only_inkg_identified.append(
                    ClusterAssignment(link, result.mention.mention_container.label_qid)
                )
            ookg_entity_cluster_assignments.append(
                ClusterAssignment(link, result.mention.mention_container.label_qid)
            )
        if result.mention.mention_container.label_out_of_kg:
            cluster_assignments_ookg.append(
                ClusterAssignment(link,
                                  result.mention.mention_container.label_qid)
            )
        else:
            cluster_assignments_inkg.append(
                ClusterAssignment(link,
                                  result.mention.mention_container.label_qid)
            )
        cluster_assignments_all.append(
            ClusterAssignment(link,
                              result.mention.mention_container.label_qid))
    clusters_stats = {}
    suffices = ["all_ookg_detected", "all_ookg_detected_only_ookg", "all_ookg_detected_only_inkg","all", "inkg", "ookg"]
    all_assignments = [ookg_entity_cluster_assignments, ookg_entity_cluster_assignments_only_ookg_identified,
                   ookg_entity_cluster_assignments_only_inkg_identified, cluster_assignments_all, cluster_assignments_inkg, cluster_assignments_ookg]
    for suffix, assignments in zip(suffices, all_assignments):
        clusters_stats.update(calculate_clustering_scores(assignments, suffix + meta_suffix,
                                                          skip_computation_of_extended_clustering_metrics))

    return clusters_stats, ookg_entity_cluster_assignments