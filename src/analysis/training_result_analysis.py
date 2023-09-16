import json
from argparse import ArgumentParser
from pathlib import Path

import distinctipy
import numpy as np
from matplotlib import pyplot as plt

from src.utilities.utilities import calculate_stats

weight = "bold"
font_size = 11
dpi = 300

def create_precision_recall_curves(linking_decisions: list, label: str, eeacc_harm_plot: plt.Figure,
                                   eeacc_ookgprec_plot: plt.Figure, eeacc_ookgacc_plot: plt.Figure,
                                   color: str):
    actual_stats = calculate_stats(linking_decisions)
    thresholds = np.linspace(0,1,21)

    num_no_candidates_inkg = 0
    num_no_candidates_ookg = 0
    all_linking_decisions = [link for example in linking_decisions for link in example["links"]]
    all_linking_decisions_final = []
    for link in all_linking_decisions:
        if len(link["candidates"]) < len(link["non_normalized_scores"]):
            if not link["candidates"]:
                if link["out_of_kg"]:
                    num_no_candidates_ookg += 1
                else:
                    num_no_candidates_inkg += 1
                continue
            non_normalized_scores = [x[0] for x in link["non_normalized_scores"]][:-1]
            max_score = max(non_normalized_scores)
            link["non_normalized_score"] = max_score
        all_linking_decisions_final.append(link)
    all_linking_decisions = all_linking_decisions_final
    all_linking_decisions = sorted(all_linking_decisions, key=lambda x: x["non_normalized_score"])
    identification_accuracies_inkg = []
    identification_accuracies_ookg = []
    identification_precisions_ookg = []
    harmonic_means = []
    num_out_of_kg = len([link for link in all_linking_decisions if link["out_of_kg"]]) + num_no_candidates_ookg
    num_inkg = len(all_linking_decisions) - num_out_of_kg + num_no_candidates_inkg
    num_out_of_kg_under_threshold = num_no_candidates_ookg
    num_in_kg_over_threshold = num_inkg
    for link in all_linking_decisions:
        identification_accuracy_inkg = num_in_kg_over_threshold / num_inkg
        identification_accuracy_ookg = num_out_of_kg_under_threshold / num_out_of_kg
        identification_precision_ookg = (num_out_of_kg_under_threshold / (num_inkg - num_in_kg_over_threshold + num_out_of_kg_under_threshold)) if (num_inkg - num_in_kg_over_threshold + num_out_of_kg_under_threshold) > 0 else 1

        identification_accuracies_inkg.append(identification_accuracy_inkg)
        identification_accuracies_ookg.append(identification_accuracy_ookg)
        identification_precisions_ookg.append(identification_precision_ookg)
        harmonic_means.append(2 * (identification_accuracy_inkg * identification_accuracy_ookg) /
                              (identification_accuracy_inkg + identification_accuracy_ookg))

        if link["out_of_kg"]:
            num_out_of_kg_under_threshold += 1
        else:
            num_in_kg_over_threshold -= 1



    # precisions = []
    # recalls = []
    # f_measures = []
    # ookg_identification_precisions = []
    # ookg_accuracies = []
    # harmonic_means = []
    #
    #
    # for key, value in all_stats.items():
    #     precisions.append(value["precision_in_kg_non_ookg"])
    #     recalls.append(value["recall_in_kg_non_ookg"])
    #     f_measures.append(value["f_measure_in_kg_non_ookg"])
    #     ookg_identification_precisions.append(value["ookg_identification_precision"])
    #     ookg_accuracies.append(value["ookg_identification_accuracy"])
    #     harmonic_means.append(value["identification_harmonic_mean"])

    # precisions, recalls, f_measures = zip(*sorted(zip(precisions, recalls, f_measures), key=lambda x: x[1]))
    # ookg_identification_precisions, ookg_accuracies, harmonic_means = zip(*sorted(zip(ookg_identification_precisions, ookg_accuracies, harmonic_means), key=lambda x: x[1]))

    # precisions = np.array(precisions)
    # recalls = np.array(recalls)
    # f_measures = np.array(f_measures)
    # ookg_identification_precisions = np.array(ookg_identification_precisions)
    # ookg_accuracies = np.array(ookg_accuracies)
    identification_accuracies_inkg = np.array(identification_accuracies_inkg)
    identification_accuracies_ookg = np.array(identification_accuracies_ookg)
    identification_precisions_ookg = np.array(identification_precisions_ookg)

    harmonic_means = np.array(harmonic_means)
    # plt.plot(recalls, precisions, label=label)
    # plt.plot([actual_stats["recall_in_kg_non_ookg"]], [actual_stats["precision_in_kg_non_ookg"]], marker='o', markersize=3, color="red", label=label)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.savefig("rec_prec.png")
    # plt.close()
    # plt.plot(recalls, f_measures, label=label)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xlabel("Recall")
    # plt.ylabel("F1")
    # plt.plot([actual_stats["recall_in_kg_non_ookg"]], [actual_stats["f_measure_in_kg_non_ookg"]], marker='o', markersize=3,
    #          color="red", label=label)
    # plt.savefig("rec_f.png")
    # plt.close()
    eeacc_ookgprec_plot.plot(identification_accuracies_ookg, identification_precisions_ookg, label=label, color=color, zorder=0)
    eeacc_ookgprec_plot.set_xlim(0, 1)
    eeacc_ookgprec_plot.set_ylim(0, 1)
    eeacc_ookgprec_plot.set_xlabel("OOKG Identification Accuracy", fontsize=font_size, fontweight=weight)
    eeacc_ookgprec_plot.set_ylabel("OOKG Identification Precision", fontsize=font_size, fontweight=weight)
    eeacc_ookgprec_plot.legend(loc="upper left")
    eeacc_ookgprec_plot.plot([actual_stats["ookg_identification_accuracy"]], [actual_stats["ookg_identification_precision"]],
                         marker='o', markersize=8, color=color, markeredgecolor="black", zorder=10)

    eeacc_ookgacc_plot.plot(identification_accuracies_inkg, identification_accuracies_ookg, label=label, color=color, zorder=0)
    eeacc_ookgacc_plot.set_xlim(0, 1)
    eeacc_ookgacc_plot.set_ylim(0, 1)
    eeacc_ookgacc_plot.set_xlabel("INKG Identification Accuracy", fontsize=font_size, fontweight=weight)
    eeacc_ookgacc_plot.set_ylabel("OOKG Identification Accuracy", fontsize=font_size, fontweight=weight)
    eeacc_ookgacc_plot.legend(loc="lower left")
    eeacc_ookgacc_plot.plot([actual_stats["in_kg_identification_accuracy"]], [actual_stats["ookg_identification_accuracy"]], marker='o', markersize=8,
                            color=color, markeredgecolor="black", zorder=10)

    eeacc_harm_plot.plot(identification_accuracies_inkg, harmonic_means, label=label, color=color, zorder=0)
    eeacc_harm_plot.set_xlim(0, 1)
    eeacc_harm_plot.set_ylim(0, 1)
    eeacc_harm_plot.set_xlabel("OOKG Identification Accuracy", fontsize=font_size, fontweight=weight)
    eeacc_harm_plot.set_ylabel("Identification Harmonic Mean", fontsize=font_size, fontweight=weight)
    eeacc_harm_plot.legend(loc="upper left")
    eeacc_harm_plot.plot([actual_stats["ookg_identification_accuracy"]], [actual_stats["identification_harmonic_mean"]], marker='o',
             markersize=8,
                         color=color, markeredgecolor="black", zorder=10)




def main(labels, results_files):
    font = {
            'weight': weight,
            'size': font_size}

    plt.rc('font', **font)


    assert len(results_files) == len(labels)

    eeacc_harm_figure = plt.figure()
    eeacc_ookgprec_figure = plt.figure()
    eeacc_ookgacc_figure = plt.figure()
    eeacc_harm_plot = eeacc_harm_figure.add_subplot(111)
    eeacc_ookgprec_plot = eeacc_ookgprec_figure.add_subplot(111)
    eeacc_ookgacc_plot = eeacc_ookgacc_figure.add_subplot(111)
    colors = distinctipy.get_colors(len(labels))
    accepted_labels = {'0.0', '0.05', '0.1', '0.3', '0.5', '1.0'}
    for label, results_file, color in zip(labels, results_files, colors):
        if label in accepted_labels:
            results = json.load(results_file.open())
            linking_decisions = results["linking_decisions"]
            create_precision_recall_curves(linking_decisions, label, eeacc_harm_plot, eeacc_ookgprec_plot,
                                           eeacc_ookgacc_plot, color)

    eeacc_ookgprec_figure.savefig("eeacc_ookgprec.png", dpi=dpi)
    eeacc_ookgacc_figure.savefig("eeacc_ookgacc.png", dpi=dpi)

    eeacc_harm_figure.savefig("eeacc_harm.png", dpi=dpi)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--results_files",  nargs='+', type=Path)
    arg_parser.add_argument("--labels", nargs='+', required=True, type=str)

    args = arg_parser.parse_args()
    results_files = args.results_files
    if args.results_files is None:
        current_dictionary = Path(".")
        all_files = []
        for file in current_dictionary.glob("detailed_results*.json"):
            all_files.append(file)
        all_files = sorted(all_files)
        results_files = all_files

    print(args.labels)
    print(results_files)
    main(args.labels, results_files)