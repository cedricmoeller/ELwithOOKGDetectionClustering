import csv
import dataclasses
from collections import defaultdict
from typing import Dict, List

import numpy as np

@dataclasses.dataclass
class ModelRuns:
    model_name: str
    metrics: Dict[str, List[float]]


metric_replacements = {
    "mention_ceaf_inkg": "CEAF_inkg",
    "muc_inkg": "MUC_inkg",
    "b_cubed_inkg": "B3_inkg",
    "mention_ceaf_ookg": "CEAF_ookg",
    "muc_ookg": "MUC_ookg",
    "b_cubed_ookg": "B3_ookg",
    "f_measure_in_kg_non_ookg": "F1_inkg",
    "ookg_f1": "F1_ookg",
    "combined_f1": "F1_combined"
}

model_name_replacements = {
 "N": "Nasty",
    "E": "Edin",
    "H": "Sequential",
"Hn": "Sequential w/o transe",
    "B": "Bottom-up",}

def create_several_model_runs_from_csv(csv_file: str, acceptable_metrics: set =  None) -> List[ModelRuns]:
    if acceptable_metrics is None:
        acceptable_metrics = {"mention_ceaf_inkg", "muc_inkg", "b_cubed_inkg", "mention_ceaf_ookg", "muc_ookg", "b_cubed_ookg", "f_measure_in_kg_non_ookg", "ookg_f1", "combined_f1"}
    all_model_runs = []
    # CSV File has header
    different_runs = defaultdict(lambda: defaultdict(list))
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["Name"]
            model_type = model_name.split("_")[1]
            model_type = model_name_replacements[model_type]
            for metric, value in row.items():
                if metric in acceptable_metrics:
                    different_runs[model_type][metric_replacements[metric]].append(float(value))
        # headers = f.readline().strip().split(",")
        # for line in f:
        #     line = line.strip().split(",")
        #     model_name = line[0]
        #     model_type = model_name.split("_")[1]
        #     for i, metric in enumerate(headers[1:]):
        #         if metric in acceptable_metrics:
        #             different_runs[model_type][metric].append(float(line[i + 1]))
    for model_type, metrics in different_runs.items():
        all_model_runs.append(ModelRuns(model_type, metrics))
    return all_model_runs

def create_table(model_runs: List[ModelRuns]):
    all_metrics_used_as_sorted_in_model_runs = []
    for model_run in model_runs:
        all_metrics_used_as_sorted_in_model_runs.extend(model_run.metrics.keys())
    # Keep unique ones in correct order
    all_metrics_used = list(dict.fromkeys(all_metrics_used_as_sorted_in_model_runs))
    headers = [""]
    table = []
    for model_run in model_runs:
        headers.append(model_run.model_name)
    for metric in all_metrics_used:
        row = []
        row.append(metric)
        for model_run in model_runs:
            # Calculate mean and std
            if metric in model_run.metrics:
                row.append(f"{np.mean(model_run.metrics[metric]):.3f} ± {np.std(model_run.metrics[metric]):.3f}")
            else:
                row.append("")
        table.append(row)

    return table, headers

# Recreate create_table function but make the biggest value in each row bold if parsed to latex
# Additionally replace each metric of format "X_Y" with "X\textsubscript{Y}$" to make it more readable
def create_table_with_bold(model_runs: List[ModelRuns]
                           ):
    all_metrics_used_as_sorted_in_model_runs = []
    for model_run in model_runs:
        all_metrics_used_as_sorted_in_model_runs.extend(model_run.metrics.keys())
    # Keep unique ones in correct order
    all_metrics_used = list(dict.fromkeys(all_metrics_used_as_sorted_in_model_runs))
    headers = [""]
    table = []
    for model_run in model_runs:
        headers.append(model_run.model_name)
    for metric in all_metrics_used:
        row = []
        row.append(metric)
        for model_run in model_runs:
            # Calculate mean and std
            if metric in model_run.metrics:
                if len(model_run.metrics[metric]) > 1:
                    row.append(f"{model_run.metrics[metric][0]:.3f}")
                    row.append(f"{np.mean(model_run.metrics[metric]):.3f} ± {np.std(model_run.metrics[metric]):.3f}")
                else:
                    row.append(f"{np.mean(model_run.metrics[metric]):.3f}")
            else:
                row.append("")
        table.append(row)
    # Make the biggest value in each row bold and the second biggest value underlined
    for row in table:
        values = [(float(value.split(" ± ")[0]), idx) for idx, value in enumerate(row[1:])]
        sorted_values = sorted(values, reverse=True)
        max_value, max_idx = sorted_values[0]
        second_best_value, second_best_idx = sorted_values[1]
        row[max_idx + 1] = f"\\textbf{{{row[max_idx + 1]}}}"
        row[second_best_idx + 1] = f"\\underline{{{row[second_best_idx + 1]}}}"
    # Replace each metric of format "X_Y" with "X\textsubscript{Y}$" to make it more readable
    for row in table:
        for i, value in enumerate(row):
            if "_" in value:
                row[i] = value.replace("_", "\\textsubscript{") + "}"
    return table, headers



# Print the output of create_table in latex format without using tabulate in booktabs format
# That means include \toprule, \midrule, \bottomrule
# Add latex tabular environment as well
def print_table_latex(model_runs: List[ModelRuns]):
    table, headers = create_table_with_bold(model_runs)
    print("\\begin{tabular}{l" + "r" * len(headers[1:]) + "}")
    print("\\toprule")
    print(" & ".join(headers) + " \\\\")
    print("\\midrule")
    for row in table:
        print(" & ".join(row) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")



print_table_latex(create_several_model_runs_from_csv("/Users/anonymized/Downloads/wandb_export_2023-04-25T17_18_21.239+02_00.csv"))

print_table_latex(create_several_model_runs_from_csv("/Users/anonymized/Downloads/wandb_export_2023-04-26T14_46_10.221+02_00.csv"))
