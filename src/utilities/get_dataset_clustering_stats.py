import copy
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tabulate import SEPARATING_LINE, tabulate


def create_table(filename: Path, dataset_paths: List[Path], dataset_labels: list):
    header = ["Dataset", "Entity type","1", "2", "3", "4", "5", "6-10", "11-20", "21-50", "50-"]
    list_of_lists = []
    for dataset_path, dataset_label in zip(dataset_paths, dataset_labels):
        inkg_clusters, okkg_clusters = compute_clusters(dataset_path)
        inkg_stats = calculate_clustering_stats(inkg_clusters)
        list_of_lists.append([dataset_label,"inkg", *inkg_stats])
        ookg_stats = calculate_clustering_stats(okkg_clusters)
        list_of_lists.append(["","ookg", *ookg_stats])
        list_of_lists.append(SEPARATING_LINE)

    table = tabulate(list_of_lists, header, floatfmt=".1f", tablefmt="latex_booktabs")
    print(table)
    filename.open("w").writelines(table)

def sum_according_to_range(in_dict: dict, start: int, end: int = None):
    sum_ = 0
    if end is None:
        in_dict = copy.deepcopy(in_dict)
        for i in range(0, start):
            if i in in_dict:
                del(in_dict[i])
        for value in in_dict.values():
            sum_ += value
    else:
        for i in range(start, end):
            if i in in_dict:
                sum_ += in_dict[i]
    return sum_

def calculate_clustering_stats(clusters: dict):

    results = defaultdict(int)
    for key, value in clusters.items():
        results[len(value)] += 1

    normalizer = sum(results.values())
    if normalizer == 0:
        normalizer = 1

    results = [results[1]/normalizer,
               results[2]/normalizer,
               results[3]/normalizer,
               results[4]/normalizer,
               results[5]/normalizer,
               sum_according_to_range(results, 6, 11)/normalizer,
               sum_according_to_range(results, 11, 21)/normalizer,
               sum_according_to_range(results, 21, 51)/normalizer,
               sum_according_to_range(results, 50)/normalizer]
    results = [x * 100 for x in results]
    return results


def compute_clusters(dataset: Path):
    dataset = json.load(dataset.open())

    inkg_clusters = defaultdict(list)
    okkg_clusters = defaultdict(list)
    num_ookg_entities = 0
    num_ikg_entities = 0

    for example in dataset:
        for entity in example["entities"]:
            if entity["out_of_kg"]:
                okkg_clusters[entity["qid"]].append(entity["mention"])
                num_ookg_entities += 1
            else:
                inkg_clusters[entity["qid"]].append(entity["mention"])
                num_ikg_entities += 1
    return inkg_clusters, okkg_clusters

def main(dataset_path: Path):
    inkg_clusters, okkg_clusters = compute_clusters(dataset_path)
    inkg_clusters_sorted = np.array([len(x) for x in inkg_clusters.values()])
    ookg_clusters_sorted = np.array([len(x) for x in okkg_clusters.values()])

    print(json.dumps(calculate_clustering_stats(inkg_clusters), indent=4))
    print(json.dumps(calculate_clustering_stats(okkg_clusters), indent=4))

    d_inkg = np.sort(inkg_clusters_sorted)
    d_ookg = np.sort(ookg_clusters_sorted)

    # Percentile values
    p = np.array(list(range(101)))

    perc = np.percentile(d_inkg, q=p)

    plt.plot(d_inkg)
    # Place red dots on the percentiles
    plt.plot((len(d_inkg) - 1) * p / 100., perc, 'b')

    # Set tick locations and labels
    blah = np.array([0, 0.25, 0.5, 0.75])
    plt.xticks((len(d_inkg) - 1) * np.array([0, 0.25, 0.5, 0.75]), map(str, blah))
    plt.xlabel("Percentile")
    plt.ylabel("Number of mentions in cluster")
    plt.yscale("log")

    plt.show()



if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=Path, required=False)
    args = argparser.parse_args()
    if args.dataset is None:
        dataset_paths = [
            "/data1/anonym/wikinews_extended/wikievents_2000-2022_train.json",
            "/data1/anonym/wikinews_extended/wikievents_2000-2022_dev.json",
            "/data1/anonym/wikinews_extended/wikievents_2000-2022_test.json",
        ]
        dataset_paths = [Path(x) for x in dataset_paths]
        create_table(Path("out.tex"), dataset_paths, ["train", "dev", "test"])

        dataset_paths = [
            "/data1/anonym/aida_titov/aida_train_ookg_art_2019.json",
            "/data1/anonym/aida_titov/aida_testa_ookg_art_2019.json",
            "/data1/anonym/aida_titov/aida_testb_ookg_art_2019.json",
        ]
        dataset_paths = [Path(x) for x in dataset_paths]
        create_table(Path("out.tex"), dataset_paths, ["train", "dev", "test"])
    else:
        main(args.dataset)

