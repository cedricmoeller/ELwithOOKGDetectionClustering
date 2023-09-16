import json
from typing import List, Tuple

from tqdm import tqdm

from src.evaluation.clustering_tools import calculate_clustering_scores
from src.utilities.utilities import calculate_syntactical_similarity
from src.utilities.various_dataclasses import ClusterAssignment


def ookg_mention_matching(ookg_entities: List[Tuple[Tuple[str, bool], str]], threshold: float = 0.9, plus=False):
    clusters = []
    for entity in tqdm(ookg_entities):
        max_sim = threshold
        closest_cluster = -1
        for cluster_idx, cluster in enumerate(clusters):
            mentions, _ = cluster
            similarities = [calculate_syntactical_similarity(mention, entity[1]) for mention in mentions]
            contained = [entity[1] in mention or mention in entity[1] for mention in mentions]
            current_sim = max(similarities)
            if current_sim > max_sim or (any(contained) and plus):
                max_sim = current_sim
                closest_cluster = cluster_idx
        if closest_cluster < 0:
            clusters.append(([entity[1]], [entity[0]]))
        else:
            clusters[closest_cluster][0].append(entity[1])
            clusters[closest_cluster][1].append(entity[0])
    cluster_assignments = []
    for idx, cluster in enumerate(clusters):
        for label in cluster[1]:
            cluster_assignments.append(ClusterAssignment(idx, label[0]))
    return cluster_assignments, len(clusters)


def analyse_dataset(examples: List[dict]):
    ookg_entities_mentions = []
    ookg_entity_mentions = 0
    entity_mentions = 0
    ookg_entities = set()
    for example in examples:
        for entity in example["entities"]:
            if entity["out_of_kg"]:
                ookg_entities_mentions.append(((entity["qid"], entity["out_of_kg"]), entity["mention"], example["text"]))
                ookg_entities.add(entity["qid"])
                ookg_entity_mentions += 1
            entity_mentions += 1
    print(len(ookg_entities))
    print(ookg_entity_mentions)
    print(entity_mentions)
    cluster_assignments, cluster_num = ookg_mention_matching(ookg_entities_mentions)
    print(calculate_clustering_scores(cluster_assignments))
    print(cluster_num)



# content = json.load(open("aida_transformed_testb_filtered.json"))
# analyse_dataset(content)
content = json.load(open("data/execution_relevant_data/events_trainvaltest.json"))
content = json.load(open("gw.json"))
# analyse_dataset(content["train"][0])
analyse_dataset(content)
# analyse_dataset(content["test"])