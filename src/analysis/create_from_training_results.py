import dataclasses
import json
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import List, Tuple

import tabulate

REPLACEMENT_DICT = {
    "precision_in_kg_non_ookg": "P",
    "recall_in_kg_non_ookg": "R",
    "f_measure_in_kg_non_ookg": "F1",
    "in_kg_identification_accuracy": "IKGA",
    "ookg_identification_accuracy": "IOOKGA",
    "ookg_identification_precision": "IOOKGP",
    "adjusted_rand_index_all": "ARI_A",
    "nmi_all": "NMI_A",
    "ari_nmi_f1_all": "F1_A",
    "adjusted_rand_index_inkg": "ARI_IKG",
    "nmi_inkg": "NMI_IKG",
    "ari_nmi_f1_inkg": "F1_IKG",
    "adjusted_rand_index_ookg": "ARI_OOKG",
    "nmi_ookg": "NMI_OOKG",
    "ari_nmi_f1_ookg": "F1_OOKG",
    "adjusted_rand_index_": "ARI_OOKGI",
    "nmi_": "NMI_OOKGI",
    "ari_nmi_f1_": "F1_OOKGI",
    "clustering_general_threshold": "T",
    "use_cosine_for_filtering": "c"
}

@dataclasses.dataclass
class ResultsContainer:
    evaluation: dict
    args: dict

def retrieve_results(directory_path: Path) -> ResultsContainer:
    result = json.load(directory_path.joinpath("detailed_results.json").open())
    args =  result.get("args", {})
    del result["detailed_info"]
    if "args" in result:
        del result["args"]
    return ResultsContainer(result, args)

def create_clustering_table(results: List[ResultsContainer], differing_args: set):
    relevant_result_identifiers = ["adjusted_rand_index_all", "nmi_all", "ari_nmi_f1_all"
                                   , "adjusted_rand_index_ookg", "nmi_ookg",
                                   "ari_nmi_f1_ookg", "adjusted_rand_index_", "nmi_", "ari_nmi_f1_"]
    sort_by = "ari_nmi_f1_all"
    return create_table(relevant_result_identifiers, results, differing_args, sort_by)

def create_linking_detection_table(results: List[ResultsContainer], differing_args: set):
    relevant_result_identifiers = ["precision_in_kg_non_ookg", "recall_in_kg_non_ookg", "f_measure_in_kg_non_ookg",
                           "in_kg_identification_accuracy", "ookg_identification_accuracy", "ookg_identification_precision"]
    sort_by = "f_measure_in_kg_non_ookg"
    return create_table(relevant_result_identifiers, results, differing_args, sort_by)


def create_table(relevant_result_identifiers: List[str], results: List[ResultsContainer], differing_args: set, sort_by: str = None) -> tabulate.TableFormat:
    relevant_result_identifiers_as_set = set(relevant_result_identifiers)
    reduced_results = []
    for result in results:
        evaluation = {}
        for key, value in result.evaluation.items():
            if key in relevant_result_identifiers_as_set:
                if isinstance(value, (int, float)):
                    value = round(value, 2)
                evaluation[key] = value

        args = {}
        for key, value in result.args.items():
            if key in differing_args:
                args[key] = value
        reduced_results.append(ResultsContainer(evaluation, args))

    sorted_args = sorted(list(differing_args))
    if sort_by is not None:
        sorted_results = sorted(reduced_results, key= lambda x: x.evaluation[sort_by])
    else:
        sorted_results = sorted(reduced_results, key= lambda x: tuple(x.args[item] for item in sorted_args))

    header = [REPLACEMENT_DICT.get(x, x) for x in chain(sorted_args, relevant_result_identifiers)]
    list_of_lists = []
    for result in sorted_results:
        row = []
        for arg in sorted_args:
            row.append(result.args[arg])
        for identifier in relevant_result_identifiers:
            row.append(result.evaluation[identifier])
        list_of_lists.append(row)

    return tabulate.tabulate(list_of_lists, header, tablefmt="latex_booktabs")



def create_tables(results: List[ResultsContainer], output_path: Path, differing_args: set, differing_eval: set):
    clustering_table_filename = Path("clustering.tex")
    linking_table_filename = Path("linking.tex")

    clustering_table = create_clustering_table(results, differing_args)
    linking_table = create_linking_detection_table(results, differing_args)

    output_path.joinpath(clustering_table_filename).open("w").writelines(clustering_table)
    output_path.joinpath(linking_table_filename).open("w").writelines(linking_table)

def create_plots(results: List[ResultsContainer], output_path: Path, differing_args: set, differing_eval: set):
    pass

def get_differing_keys(results: List[ResultsContainer]) -> Tuple[set, set]:
    args_value_dict = defaultdict(set)
    eval_value_dict = defaultdict(set)
    for result in results:
        for key, value in result.args.items():
            args_value_dict[key].add(value)
        for key, value in result.evaluation.items():
            eval_value_dict[key].add(value)
    differing_args = {key for key, value in args_value_dict.items() if len(value) > 1}
    differing_eval = {key for key, value in eval_value_dict.items() if len(value) > 1}
    return differing_args, differing_eval


def main():
    argparser = ArgumentParser()
    argparser.add_argument('-r', '--run_directories', nargs='+', required=True, type=Path)
    argparser.add_argument('-o', '--output_path', required=True, type=Path)
    argparser.add_argument('--force', action='store_true', default=False)

    args = argparser.parse_args()

    output_path: Path = args.output_path

    if output_path.exists():
        if not args.force:
            answer = input(f"{output_path} already exists. Overwrite? y/yes")
            if answer.lower() not in {"y", "yes"}:
                return
    else:
        output_path.mkdir()

    results = [retrieve_results(item) for item in args.run_directories]

    differing_args, differing_eval = get_differing_keys(results)
    create_tables(results, output_path, differing_args, differing_eval)
    create_plots(results, output_path, differing_args, differing_eval)

if __name__ == '__main__':
    main()


