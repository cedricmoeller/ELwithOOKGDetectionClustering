import copy
import json
import math
from argparse import ArgumentParser
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Tuple, Optional, Set, Iterable

import networkx as nx
import numpy as np
import optuna
import torch
import wandb
from torch.nn.functional import sigmoid
from tqdm import tqdm

from src.evaluation.clustering_tools import ookg_clustering, \
    calculate_pairwise_scores_comparator
from src.evaluation.evaluation_tools import calculate_final_results
from src.model.initialization import init_for_evaluation
import faiss
from src.utilities.utilities import is_true, calculate_partial_syntactical_similarity, \
    calculate_syntactical_similarity
from src.utilities.various_dataclasses import Result, ComplexCandidateEntity, EmbeddedMention, KGCandidateEntity, \
    CandidateContainerWrapper


class Clustering:
    def __init__(self, args, ranking_model, mention_comparator_model, candidate_manager, device,
                 default_metric):
        self.device = device
        self.ranking_model = copy.deepcopy(ranking_model).to(self.device)
        self.mention_comparator = copy.deepcopy(mention_comparator_model).to(self.device)
        self.mention_comparator_model = copy.deepcopy(mention_comparator_model).to(self.device)
        self.candidate_manager = candidate_manager
        self.args = args
        self.pairwise_scores = None
        self.mention_pairs = None
        self.inverted = False
        self.default_metric = default_metric

    def clear_pairwise_scores(self):
        self.pairwise_scores = None


    def calculate_result(self, all_results, skip_clustering_calculation: bool):
        if not self.args["skip_stats_calculation"]:
            stats, _ = calculate_final_results(all_results, self.mention_comparator_model, self.ranking_model,
                                               self.candidate_manager, self.args, self.device, True,
                                               skip_loss_calculation=self.args.get("skip_loss_calculation"),
                                               skip_clustering_calculation=skip_clustering_calculation)
        else:
            stats = None
        return stats

    def calculate_nearest_neighbors_via_faiss_index(self, results: List[Result], threshold: float):
        if self.pairwise_scores is not None:
            return self.pairwise_scores, self.mention_pairs

        index = faiss.IndexFlatIP(300)
        for idx, candidate in enumerate(results):
            vector = candidate.mention.processed_mention.mention_embedding.unsqueeze(0).numpy()
            vector /= np.linalg.norm(vector)
            index.add(vector)

        pairwise_scores = []
        mention_pairs = []
        for idx, candidate in enumerate(results):
            vector = candidate.mention.processed_mention.mention_embedding.unsqueeze(0).numpy()
            vector /= np.linalg.norm(vector)
            result = index.range_search(vector,threshold)
            for score, idx_ in zip(result[1], result[2]):
                if idx != idx_:
                    mention_pairs.append((idx, idx_))
                    pairwise_scores.append(score)
        self.mention_pairs = mention_pairs
        self.pairwise_scores = pairwise_scores
        return pairwise_scores, mention_pairs


    def calculate_pairwise_scores(self, results: List[Result], use_embedding_comparison = False):
        if self.pairwise_scores is not None:
            return self.pairwise_scores, self.mention_pairs
        a = []
        b = []
        mention_pairs: List[Tuple[int, int]] = []
        candidate_mapping = {}
        for idx, candidate in enumerate(results):
            candidate_mapping[idx] = candidate
            for idx_, candidate_ in enumerate(results[idx:]):
                a_id = idx
                b_id = idx + idx_
                mention_pairs.append((a_id, b_id))
                a.append(candidate)
                b.append(candidate_)

        assert len(a) == len(b)
        if not a:
            pairwise_scores = torch.zeros((0, 1))
        elif self.args.get("use_embedding_comparison", False) or use_embedding_comparison:
            a_emb = [x.mention.processed_mention.mention_embedding for x in a]
            b_emb = [x.mention.processed_mention.mention_embedding for x in b]
            v1 = torch.stack(a_emb)
            v2 = torch.stack(b_emb)
            pairwise_scores = self.ranking_model.calculate_pairwise_scores(v1, v2)
        elif self.args.get("use_syntactical_comparison", False):
            if self.args["use_regular_edit_distance"]:
                pairwise_scores = [calculate_syntactical_similarity(x, y) for x, y in
                                   zip([x.mention.mention_container.mention for x in a],
                                       [x.mention.mention_container.mention for x in b])]
            else:
                pairwise_scores = [calculate_partial_syntactical_similarity(x, y) for x, y in
                                   zip([x.mention.mention_container.mention for x in a],
                                       [x.mention.mention_container.mention for x in b])]
            pairwise_scores = torch.tensor(pairwise_scores)
        elif self.args.get("alternate_mention_embedding", False):
            a_emb = [x.post_mention_embedding for x in a]
            b_emb = [x.post_mention_embedding for x in b]
            v1 = torch.stack(a_emb).to(self.device)
            v2 = torch.stack(b_emb).to(self.device)
            pairwise_scores = torch.norm(v1 - v2, dim=1)
        else:
            pairwise_scores = calculate_pairwise_scores_comparator([x.mention for x in a], [x.mention for x in b],
                                                                   self.ranking_model, self.mention_comparator_model, self.device,
                                                                   )

        if self.args.get("pre_filtering", False):
            pairwise_scores_two_stage = torch.tensor([calculate_syntactical_similarity(x, y) for x, y in
                                                      zip([x.mention.mention_container.mention for x in a],
                                                          [x.mention.mention_container.mention for x in b])])
            pairwise_scores[pairwise_scores_two_stage < 0.75] = torch.nan
        self.mention_pairs = mention_pairs
        self.pairwise_scores = pairwise_scores
        return pairwise_scores, mention_pairs

    def determine_threshold_on_dev_set(self, all_results: List[Result]) -> List[float]:


        result, thresholds_first_guess, max_differences = self.cluster_and_create_results(all_results, [])
        if isinstance(thresholds_first_guess, torch.Tensor):
            thresholds_first_guess = thresholds_first_guess.to("cpu")

        def objective_method(trial):
            suggested_thresholds = []
            for idx, (threshold, max_difference) in enumerate(zip(thresholds_first_guess, max_differences)):
                lower_boundary = (threshold - max_difference / 2)
                upper_boundary = (threshold + max_difference / 2)
                t = trial.suggest_float(f"t{idx}",lower_boundary , upper_boundary)
                suggested_thresholds.append(t)
            wandb.log({"thresholds": suggested_thresholds})
            result_optuna, _, _ = self.cluster_and_create_results(all_results, thresholds=suggested_thresholds)
            return result_optuna[self.default_metric]
        study = optuna.create_study(direction="maximize")
        study.enqueue_trial({f"t{idx}": threshold for idx, threshold in enumerate(thresholds_first_guess)})
        study.optimize(objective_method, n_trials=self.args["n_trials"], show_progress_bar=False)

        best_params = study.best_params
        self.clear_pairwise_scores()
        return [best_params[f"t{idx}"] for idx, _ in enumerate(thresholds_first_guess)]

    def cluster_and_create_results(self, all_results: List[Result], thresholds: List[float], skip_clustering_calculation=True) -> Tuple[Result, List[float], List[float]]:
        new_results, thresholds, max_difference = self.cluster(all_results, thresholds)
        return self.calculate_result(new_results, skip_clustering_calculation), thresholds, max_difference
    def cluster(self, all_results: List[Result], thresholds: List[float]) -> Tuple[list, list, list]:
        raise NotImplementedError


class Cluster:
    mentions: set
    entity: Optional[int]

    def __init__(self, mentions: set, entity: Optional[int]):
        self.mentions = mentions
        self.entity = entity


# Reused and modified code from the following source:
# Author: Nicolas Heist
# Title: CaLiGraph
# Repository URL: https://github.com/nheist/CaLiGraph
# License: GPL-3.0

# The original source code is licensed under the GNU General Public License v3.0.
# A copy of the license can be found in the 'LICENSE' file or at
# https://www.gnu.org/licenses/gpl-3.0.html


class GreedyClustering(Clustering):
    def __init__(self, args, ranking_model, mention_comparator_model, candidate_manager, device,
                 me_threshold, mm_threshold, default_metric):

        super().__init__(args, ranking_model, mention_comparator_model, candidate_manager, device, default_metric)
        self.me_threshold = me_threshold
        self.mm_threshold = mm_threshold

    def _get_alignment_graph(self, all_results, add_entities: bool, thresholds: list) -> nx.Graph:
        me_threshold, mm_threshold = thresholds
        ag = nx.Graph()
        for m_id, result in enumerate(all_results):
            ag.add_node(m_id, is_ent=False)
            for candidate in result.candidates:
                e_id, score = candidate.complex_candidate.candidate.qid, candidate.similarity
                if add_entities and score > me_threshold:
                    ag.add_node(e_id, is_ent=True)
                    ag.add_edge(m_id, e_id, weight=max(min(score, 1.0), 0.000001),)
        pairwise_scores, mention_pairs = self.calculate_nearest_neighbors_via_faiss_index(all_results, mm_threshold)
        edges = []
        for (m_one, m_two), score in zip(mention_pairs, pairwise_scores):
            if score > mm_threshold:
                edges.append(((m_one, m_two), score))

        ag.add_weighted_edges_from(
            [(u, v, min(score, 1)) for (u, v), score in edges])
        return ag

    def _get_subgraphs(self, ag: nx.Graph) -> Iterable[nx.Graph]:
        for nodes in nx.connected_components(ag):
            yield ag.subgraph(nodes)

    def _get_top_entities_for_mentions(self, all_results, me_threshold):
        mention_ents = defaultdict(dict)
        for m_id, result in enumerate(all_results):
            for candidate in result.candidates:
                if candidate.similarity <= me_threshold:
                    continue
                mention_ents[m_id][candidate.complex_candidate.qid] = candidate.similarity
        mention_ents = {m_id: max(ent_scores.items(), key=lambda x: x[1])[0] for m_id, ent_scores in
                        mention_ents.items()}
        return mention_ents

    @classmethod
    def _get_mention_nodes(cls, g: nx.Graph):
        return {node for node, is_ent in g.nodes(data='is_ent') if not is_ent}

    @staticmethod
    def create_dummy_result(all_results, ent, mention, ookg_counter):
        if ent is None:
            ent_to_use = ookg_counter
        else:
            ent_to_use = ComplexCandidateEntity(
                KGCandidateEntity(ent, None, None, None, None, None, None, None, None, None, None, None, None, None))
        mention = all_results[mention].mention
        return Result(ent_to_use, mention, isinstance(ent_to_use, int), [], 0.0, 0.0, [], None, [])


class EdinClustering(GreedyClustering):
    def __init__(self, args, ranking_model, mention_comparator_model, candidate_manager, device, me_threshold,
                 mm_threshold, me_cluster_threshold, default_metric):

        super().__init__(args, ranking_model, mention_comparator_model, candidate_manager, device, me_threshold,
                         mm_threshold, default_metric)
        self.me_threshold = me_threshold
        self.mm_threshold = mm_threshold
        self.me_cluster_threshold = me_cluster_threshold

    def cluster(self, all_results, thresholds) -> Tuple[list, list, list]:
        if not thresholds:
            thresholds = [self.me_threshold, self.mm_threshold, self.me_cluster_threshold]

        me_threshold, mm_threshold, me_cluster_threshold = thresholds
        mention_graph = self._get_alignment_graph(all_results, False, thresholds[:2])
        mention_ents = self._get_top_entities_for_mentions(all_results, me_threshold)
        new_results = []
        ookg_counter = 0
        for mention_cluster in self._get_subgraphs(mention_graph):
            mentions = self._get_mention_nodes(mention_cluster)
            ent = None
            ent_counts = Counter([mention_ents[m_id] for m_id in mentions if m_id in mention_ents])
            if ent_counts:  # assign entity to cluster only if it is closest entity for >= X% of mentions
                top_ent, top_ent_count = ent_counts.most_common(1)[0]
                top_ent_score = top_ent_count / len(mentions)
                if top_ent_score >= me_cluster_threshold:
                    ent = top_ent
            for mention in mentions:
                new_results.append(self.create_dummy_result(all_results, ent, mention, ookg_counter))
            if ent is None:
                ookg_counter += 1
        return new_results, thresholds, [2, 2, 2]

    @classmethod
    def _get_mention_nodes(cls, g: nx.Graph):
        return {node for node, is_ent in g.nodes(data='is_ent') if not is_ent}


def _to_dijkstra_node_weight(u, v, attrs: dict) -> float:
    return -math.log2(attrs['weight'])


def _from_dijkstra_node_weight(weight: float) -> float:
    return 2**(-weight)


class NastyLinker(GreedyClustering):
    def __init__(self, args, ranking_model, mention_comparator_model, candidate_manager, device, me_threshold,
                 mm_threshold, path_threshold, default_metric):
        super().__init__(args, ranking_model, mention_comparator_model, candidate_manager, device, me_threshold,
                         mm_threshold, default_metric)
        self.path_threshold = path_threshold

    def cluster(self, all_results: List[Result], thresholds: list) -> Tuple[list, list, list]:
        if not thresholds:
            thresholds = [self.me_threshold, self.mm_threshold, self.path_threshold]
        ag = self._get_alignment_graph(all_results, True, thresholds[:2])
        valid_subgraphs = self._compute_valid_subgraphs(ag, thresholds)
        new_results = []
        ookg_counter = 0
        for g in valid_subgraphs:
            mentions = self._get_mention_nodes(g)
            ent = self._get_entity_node(g)
            for mention in mentions:
                new_results.append(self.create_dummy_result(all_results, ent, mention, ookg_counter))
            if ent is None:
                ookg_counter += 1

        return new_results, thresholds, [1, 1, 1]

    def _compute_valid_subgraphs(self, ag: nx.Graph, thresholds) -> List[nx.Graph]:
        valid_subgraphs = []

        for sg in self._get_subgraphs(ag):
            if self._is_valid_graph(sg):
                valid_subgraphs.append(sg)
            else:
                valid_subgraphs.extend(self._split_into_valid_subgraphs(sg, thresholds))
        return valid_subgraphs

    def _is_valid_graph(self, ag: nx.Graph) -> bool:
        return len(self._get_entity_nodes(ag)) <= 1

    @classmethod
    def _get_entity_node(cls, g: nx.Graph) -> Optional[int]:
        ent_nodes = cls._get_entity_nodes(g)
        return ent_nodes[0] if ent_nodes else None

    @classmethod
    def _get_entity_nodes(cls, g: nx.Graph) -> List[int]:
        return [node for node, is_ent in g.nodes(data='is_ent') if is_ent]

    def _split_into_valid_subgraphs(self, ag: nx.Graph, thresholds) -> List[nx.Graph]:
        ent_groups = defaultdict(set)
        unassigned_mentions = set()

        distances, paths = nx.multi_source_dijkstra(ag, self._get_entity_nodes(ag), weight=_to_dijkstra_node_weight)
        for node, path in paths.items():
            score = _from_dijkstra_node_weight(distances[node])
            if score > thresholds[2]:
                ent_node = path[0]
                ent_groups[ent_node].add(node)
            else:
                unassigned_mentions.add(node)
        return [ag.subgraph(nodes) for nodes in ent_groups.values()] + list(self._get_subgraphs(ag.subgraph(unassigned_mentions)))


class BottomUpClustering(Clustering):
    def __init__(self, args, ranking_model, mention_comparator_model, candidate_manager, device,
                 me_threshold, mm_threshold, default_metric):

        super().__init__(args, ranking_model, mention_comparator_model, candidate_manager, device, default_metric)
        self.me_threshold = me_threshold
        self.mm_threshold = mm_threshold

    def cluster(self, all_results: List[Result], thresholds: List[float]) -> Tuple[list, list, list]:
        if not thresholds:
            thresholds = [self.me_threshold, self.mm_threshold]
        me_threshold, mm_threshold = thresholds
        clusters_by_mid, edges = self._init_clusters_and_edges(all_results, me_threshold, mm_threshold)
        print("Processing edges")
        for u, v in tqdm(edges):
            if isinstance(v, str):  # ME edge
                c = clusters_by_mid[u]
                if c.entity is None:
                    c.entity = v
            else:  # MM edge
                c_one, c_two = clusters_by_mid[u], clusters_by_mid[v]
                if c_one.entity is not None and c_two.entity is not None:
                    continue
                if len(c_one.mentions) < len(c_two.mentions):
                    c_one, c_two = c_two, c_one  # merge smaller cluster into bigger one
                c_one.mentions = c_one.mentions | c_two.mentions
                if c_one.entity is None:
                    c_one.entity = c_two.entity
                for m_id in c_two.mentions:
                    clusters_by_mid[m_id] = c_one
        clusters = self._collapse_clusters(set(clusters_by_mid.values()))
        new_results = []
        ookg_counter = 0
        for c in clusters:
            for mention in c.mentions:
                if c.entity is None:
                    ent_to_use = ookg_counter
                else:
                    ent_to_use = ComplexCandidateEntity(KGCandidateEntity(c.entity, None, None, None, None, None, None, None, None, None, None, None, None, None))
                mention = all_results[mention].mention
                new_results.append(Result(ent_to_use, mention, isinstance(ent_to_use, int), [], 0.0, 0.0, [], None, []))
        return new_results, thresholds, [1, 1]

    def _init_clusters_and_edges(self, all_results: List[Result], me_threshold: float, mm_threshold: float):
        clusters_by_mid = {}
        # find best entity match per mention
        me_edges = defaultdict(dict)
        for m_id, result in enumerate(all_results):
            if m_id not in clusters_by_mid:
                clusters_by_mid[m_id] = Cluster({m_id}, None)
            for candidate in result.candidates:
                e_id, score = candidate.complex_candidate.candidate.qid, candidate.similarity
                if score > me_threshold:
                    me_edges[m_id][e_id] = score
        # collect all potential edges
        edges = [(m_id, *max(ent_scores.items(), key=lambda x: x[1])) for m_id, ent_scores in me_edges.items()]
        pairwise_scores, mention_pairs = self.calculate_nearest_neighbors_via_faiss_index(all_results, mm_threshold)
        for (m_one, m_two), score in zip(mention_pairs, pairwise_scores):
            if score > mm_threshold:
                edges.append((m_one, m_two, score))
        ordered_edges = [(u, v) for u, v, _ in sorted(edges, key=lambda x: x[0], reverse=True)]
        return clusters_by_mid, ordered_edges

    @classmethod
    def _collapse_clusters(cls, clusters: Set[Cluster]) -> Set[Cluster]:
        cluster_by_ent = defaultdict(set)
        for c in clusters:
            cluster_by_ent[c.entity].add(c)
        collapsed_clusters = set()
        for ent, clusters in cluster_by_ent.items():
            if ent is None:
                collapsed_clusters.update(clusters)
            else:
                collapsed_clusters.add(Cluster({m for c in clusters for m in c.mentions}, ent))
        return collapsed_clusters


class HierarchicalTwoStageClustering(Clustering):
    def __init__(self, args, ranking_model, mention_comparator_model, candidate_manager, device, default_metric):
        super().__init__(args, ranking_model, mention_comparator_model, candidate_manager, device, default_metric)
        self.inverted = True

    def split_for_clustering(self, all_results: List[Result]):
        ookg_entities = []
        results_to_use = []
        for result in all_results:
            if bool(result.is_ookg):
                ookg_entities.append(result)
            else:
                results_to_use.append(result)
        return ookg_entities, results_to_use
    def agglomerative_clustering(self, results, pairwise_scores, threshold = None):
        if not results:
            return [], 0.0

        pairwise_scores = pairwise_scores.to("cpu")

        if self.args.get("alternate_mention_embedding"):
            pairwise_scores = -pairwise_scores

        two_stage_clustering_results, threshold = ookg_clustering(pairwise_scores, results,
                                                                  threshold=threshold,
                                                                  device=self.device,
                                                                  alternate_mention_embedding=self.args.get(
                                                                      "alternate_mention_embedding"),
                                                                  alt_clustering=True)
        return two_stage_clustering_results, threshold
    def cluster(self, all_results: List[Result], thresholds: List[float]) -> Tuple[list, list, list]:
        if not thresholds:
            thresholds = [0.0]
        ookg_entities, results = self.split_for_clustering(all_results)
        pairwise_scores, mention_pairs = self.calculate_pairwise_scores(ookg_entities)
        if ookg_entities:
            max_value = torch.max(pairwise_scores[torch.logical_not(torch.isnan(pairwise_scores))])
            min_value = torch.min(pairwise_scores[torch.logical_not(torch.isnan(pairwise_scores))])
            max_difference = max_value - min_value

            ookg_results, threshold = self.agglomerative_clustering(ookg_entities, pairwise_scores, thresholds[0])
        else:
            ookg_results = []
            threshold = 0.0
            max_difference = 0.0
        results += ookg_results

        for item in results:
            if isinstance(item.link, CandidateContainerWrapper):
                item.link = item.link.complex_candidate
        return results, [threshold], [max_difference]
    def cluster_and_create_results(self, all_results: List[Result], thresholds: List[float],
                                   skip_clustering_calculation=True) -> Tuple[Result, List[float], List[float]]:
        new_results, thresholds, max_difference = self.cluster(all_results, thresholds)
        return self.calculate_result(new_results, skip_clustering_calculation), thresholds, max_difference

class TwoStageGreedyClustering(HierarchicalTwoStageClustering):
    def __init__(self, args, ranking_model, mention_comparator_model, candidate_manager, device, default_metric, second_stage_clustering_method):

        super().__init__(args, ranking_model, mention_comparator_model, candidate_manager, device, default_metric)
        self.second_stage_clustering_method = second_stage_clustering_method

    def clear_pairwise_scores(self):
        self.pairwise_scores = None
        self.second_stage_clustering_method.clear_pairwise_scores()

    def cluster(self, all_results: List[Result], thresholds: List[float]) -> Tuple[list, list, list]:
        ookg_entities, other_results = self.split_for_clustering(all_results)
        pairwise_scores, mention_pairs = self.calculate_pairwise_scores(ookg_entities)
        max_value = torch.max(pairwise_scores[torch.logical_not(torch.isnan(pairwise_scores))])
        min_value = torch.min(pairwise_scores[torch.logical_not(torch.isnan(pairwise_scores))])
        max_difference = max_value - min_value

        results, thresholds, max_differences = self.second_stage_clustering_method.cluster(ookg_entities, thresholds)
        results += other_results
        for item in results:
            if isinstance(item.link, CandidateContainerWrapper):
                item.link = item.link.complex_candidate
        return results, thresholds, [max_difference] + max_differences


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

clustering_method = {
    "H" : HierarchicalTwoStageClustering,
    "E" : EdinClustering,
    "N" : NastyLinker,
    "B" : BottomUpClustering}

def fix_results(results: List):
    for result in results:
        if "emerging" in result["mention"]["mention_container"]["other_info"]:
            is_emerging = result["mention"]["mention_container"]["other_info"]["emerging"]
            result["mention"]["mention_container"]["_label_out_of_kg"] = is_emerging

def main(args):
    wandb.init(project="ookg", config=args, group="second_stage", name=args["name"])
    args, document_tokenizer, entity_tokenizer, kg_connector, type_list, document_model, entity_model, ranking_model, mention_comparator_model, candidate_manager, device  = init_for_evaluation(args, skip_unnecessary=True)
    print("Loading results file.")
    all_results = json.load(args["results_file"].open())
    all_results_dev = json.load(args["results_file_dev"].open())
    fix_results(all_results)
    fix_results(all_results_dev)
    print("Loaded results file.\nUnpack.")
    all_results_dev = [Result.from_dict(x) for x in all_results_dev]
    print("Loaded results file.\nUnpack.")
    all_results = [Result.from_dict(x) for x in all_results]
    print("Unpacked.")

    if args["clustering_method"] == GreedyClustering:
        additional_params = {"me_threshold": 0.5,
                             "mm_threshold": 0.5,
                             "me_cluster_threshold": 0.5}
    elif args["clustering_method"] == NastyLinker:
        additional_params = {"me_threshold": 0.5,
                             "mm_threshold": 0.5,
                             "path_threshold": 0.5}
    elif args["clustering_method"] == BottomUpClustering:
        additional_params = {"me_threshold": 0.5,
                             "mm_threshold": 0.5}
    elif args["clustering_method"] == EdinClustering:
        additional_params = {"me_threshold": 0.5,
                             "mm_threshold": 0.5,
                             "me_cluster_threshold": 0.5}
    else:
        additional_params = {}

    clustering_method = args["clustering_method"](args, ranking_model, mention_comparator_model, candidate_manager,
                                                  device, default_metric= args["default_metric"],
                                                  **additional_params)

    if args["two_stage"]:
        clustering_method = TwoStageGreedyClustering(args, ranking_model, mention_comparator_model, candidate_manager,
                                                      device, default_metric=args["default_metric"],
                                                      second_stage_clustering_method=clustering_method)

    with torch.no_grad():
        if not args.get("clustering_general_threshold"):
            dev_thresholds = clustering_method.determine_threshold_on_dev_set(all_results_dev)
        else:
            dev_thresholds = args.get("clustering_general_threshold")
        print(f"Best thresholds: {dev_thresholds}")
        wandb.summary["best_thresholds"] = dev_thresholds
        average_results = {}
        for i in tqdm(range(1, args.get("partial_clustering_parts", 1) + 1)):
            average_result = {}
            size_of_chunks  = len(all_results) // i
            all_chunks = list(chunks(all_results, size_of_chunks))
            for chunk in all_chunks:
                ookg_entities = []
                results_to_use = []
                for x in chunk:
                    if bool(x.is_ookg):
                        ookg_entities.append(x)
                    else:
                        results_to_use.append(x)

                chunk_result, thresholds, _ = clustering_method.cluster_and_create_results(all_results, thresholds=dev_thresholds,
                                                                                           skip_clustering_calculation=False)


                if "detailed_info" in chunk_result:
                    del chunk_result["detailed_info"]
                if not average_result:
                    average_result.update(chunk_result)
                else:
                    for key, value in chunk_result.items():
                        if isinstance(value, (float, int)):
                            average_result[key] += value
            for key, value in average_result.items():
                if isinstance(value, (float, int)):
                    average_result[key] /= len(all_chunks)
            average_results[i] = average_result
        if len(average_results) == 1:
            average_results = average_results[1]
            wandb.summary.update(average_results)


        json.dump(average_results, open(f"detailed_results{args['suffix']}.json", "w"), indent=4)

        print(json.dumps(average_results, indent=4))

if __name__ == '__main__':
    argparser = ArgumentParser()

    argparser.add_argument("--results_file", type=Path, default="all_results.p", help="Results file as created by standalone_evaluate.py")
    argparser.add_argument("--results_file_dev", type=Path, default="all_results_dev.p", help="Development results file as created by standalone_evaluate.py")
    argparser.add_argument("--clustering_method", type=lambda x: clustering_method[x], default=HierarchicalTwoStageClustering)
    argparser.add_argument("--model", type=Path, help="Path to model file to evaluate")
    argparser.add_argument("--models", type=Path, nargs='+', help="Path to multiple model files to evaluate")

    argparser.add_argument("--im_kg_path", type=Path,
                           help="In memory KG file. Expected to be a jsonl file parsed as detailed in the README.md")
    argparser.add_argument("--mention_dictionary_file", type=Path, required=False,
                           help="A JSON file containing a mapping from each mention to a set of candidates.")
    argparser.add_argument("--type_list", type=Path, required=False,
                           help="A JSON file with including all types that should be used for the type feature as a list.")
    argparser.add_argument("--two_hop_type_list", type=Path, required=False,
                           help="A JSON file with including all types that should be used for the two-hop type feature as a list. Currently not in use.")
    argparser.add_argument("--transe_embeddings_file", type=Path, required=False,
                           help="A numpy file containing all TransE embeddings necessary for the in memory KG.")
    argparser.add_argument("--transe_mappings_file", type=Path, required=False,
                           help="A JSON file mapping each QID to a TransE embedding index as provided via argument transe_embeddings_file.")
    argparser.add_argument("--suffix", type=str, default="")


    argparser.add_argument("--use_embedding_comparison", type=is_true, default=False, help="Only use the mention encoder embeddings for pairwise scoring..")
    argparser.add_argument("--use_syntactical_comparison", type=is_true, default=False, help="Only use the edit distance as pairwise score.")
    argparser.add_argument("--use_regular_edit_distance", type=is_true, default=True, help="Whether to use regular edit distance or partial.")
    argparser.add_argument("--debug", type=is_true, default=False)

    argparser.add_argument("--clustering_general_threshold", action='store', type=float, nargs="*", required=False, help="If not set, the threshold will be determined by using the validation set.")
    argparser.add_argument("--alt_clustering", type=is_true, default=True, help="Should stay true. Other method is currently not tested.")
    argparser.add_argument("--partial_clustering_parts", type=int, default=1, help="Splits the dataset in parts and evaluates them separately")
    argparser.add_argument("--n_trials", type=int, default=100, help="Number of trials to determine the best threshold on the development set.")
    argparser.add_argument("--pre_filtering", type=is_true, default=False)
    argparser.add_argument("--skip_stats_calculation", type=is_true, default=False)
    argparser.add_argument("--skip_loss_calculation", type=is_true, default=True)
    argparser.add_argument("--default_metric", type=str, default="ari_nmi_f1_all")
    argparser.add_argument("--name", type=str, default="evaluation")
    argparser.add_argument("--two_stage", type=is_true, default=False)



    main(vars(argparser.parse_args()))
