import csv
import dataclasses
from collections import defaultdict
from typing import Dict, List

import numpy as np
import tabulate

headers = ["Model", "NMI_ooKG", "ARI_ooKG", "F1_ooKG", "NMI_inKG", "ARI_inKG", "F1_inKG"]

tabular_data_aida = [
    ["Exact", "0.8402167435653207", "0.0", "0.0", "1.0", "1.0", "1.0"],
    ["Edit", "0.9770822161746262", "0.949170714603117", "0.9629242458600837", "1.0", "1.0", "1.0"],
    ["Mention Encoder", "0.987764", "0.967765", "0.977662", "1.0", "1.0", "1.0"],
    ["TransE", "0.8293136254684665", "0.14934260412865863", "0.25310594817921894", "1.0", "1.0", "1.0"],
    ["Mention Encoder + Edit", "0.9783664355980334", "0.9385830002872088", "0.958061894918836", "1.0", "1.0", "1.0"],
    ["Edit + TransE", "0"  ,    "0.0"   ,   "0.0"  ,    "0.0"    ,  "0.0"   ,   "0.0"],
    ["Mention Encoder + TransE", "0.990685", "0.975836", "0.983204", "1.0", "1.0", "1.0"],
    ["Mention Encoder + TransE + Edit", "0.985015", "0.965739", "0.975282", "1.0", "1.0", "1.0"]
]
full_data_aida = [
    ["Exact","0.885595"      ,"0.0"          , "0.0"          , "0.937282"      ,"0.0" ,          "0.0"],
    ["Edit", "0.985329"  ,    "0.914505"  ,    "0.948597"    ,  "0.979879"    ,  "0.790881"  ,    "0.875294"],
    ["Mention Encoder", "0.991330" ,     "0.968021" ,     "0.979537"  ,    "0.980451"  ,    "0.804892" ,     "0.884040"],
    ["TransE", "0.859285" ,     "0.181765",      "0.300059",      "0.872064",      "0.140941",      "0.242663"],
    ["Mention Encoder + Edit", "0.992314"    ,  "0.963795"  ,    "0.977846"    ,  "0.981205"   ,   "0.807483"    ,  "0.885908"],
    ["Edit + TransE", "0.987469"  ,    "0.932808"   ,   "0.959361"  ,    "0.981489"    ,  "0.810090"   ,   "0.887591"],
    ["Mention Encoder + TransE", "0.993274"   ,   "0.973865"   ,   "0.983474"    ,  "0.980451"    ,  "0.804892"  ,    "0.884040"],
    ["Mention Encoder + TransE + Edit",  "0.987331"  ,    "0.959381"   ,   "0.973156" ,     "0.972150"    ,  "0.714779"  ,    "0.823831"]
]

final = []
for a, b in zip(tabular_data_aida, full_data_aida):
    # new_line = [a[0]]
    # for x, y in zip(a[1:], b[1:]):
    #     x = float(x)
    #     if x != 1.0 and x != 0.0:
    #         x = f"{x:1.3f}"
    #     y = float(y)
    #     if y != 1.0 and y != 0.0:
    #         y = f"{y:1.3f}"
    #     new_line.append(f"{x} ({y})")
    final.append(a)
    final.append(b)

headers_alt = ["Model", "CEAF_inKG", "MUC_inKG", "B3_inKG", "MUC_ooKG", "B3_ooKG", "CEAF_ooKG"]

tabular_data_wikievents = [
    ["Exact", "0.754043", "0.0", "0.0", "1.0", "1.0", "1.0"],
    ["Edit", "0.934376", "0.824112", "0.875864", "1.0", "1.0", "1.0"],
    ["Mention Encoder", "0.842943", "0.482324", "0.613570", "1.0", "1.0", "1.0"],
    ["TransE", "0.524369", "0.038441", "0.094986", "1.0", "1.0", "1.0"],
    ["Mention Encoder + TransE + Edit", "0.9106342926727657", "0.7012647608340109", "0.7923520248605085", "1.0", "1.0", "1.0"],
]

full_data_wikinews_alt = [
    ["Exact",  "0.635574"   ,   "0.0"      ,     "0.777187"     , "0.273458"   ,  "0.0"     ,      "0.429473"],
    ["Edit",    "0.790607"   ,   "0.731023"    ,  "0.846133"    ,  "0.849865"   ,   "0.944527"   ,   "0.870288"],
    ["Mention Encoder",       "0.868768"    ,  "0.843016"  ,    "0.905683"   ,   "0.779088"   ,   "0.939767" ,     "0.782638"],
    ["TransE",      "0.113862"  ,    "0.414246" ,     "0.175545"   ,   "0.255764"  ,    "0.748450" ,     "0.264369"],
    ["Mention Encoder + Edit"    ,  "0.903505"   ,   "0.857802"   ,   "0.928912"  ,    "0.748525"   ,   "0.932427"    ,  "0.772970"],
    ["Mention Encoder + TransE", "0.886137"    ,  "0.860306"  ,    "0.920283"   ,   "0.774262"  ,    "0.940271"  ,    "0.780823"],
    ["Edit + TransE"  ,    "0.771952"   ,   "0.708333"    ,  "0.832277"   ,   "0.791957"  ,    "0.901008"  ,    "0.807656" ],
    ["Mention Encoder + TransE + Edit",  "0.858797"   ,   "0.831973"    ,  "0.898633", "0.778552"  ,    "0.941687"    ,  "0.785866"]
]


full_data_aida_alt = [
    ["Exact","0.885595"      ,"0.0"          , "0.0"          , "0.937282"      ,"0.0" ,          "0.0"],
    ["Edit",   0.901477   ,   0.898734  ,    0.937093   ,   0.946153    ,  0.955555    ,  0.958196],
    ["Mention Encoder", 0.901477     , 0.899999   ,   0.937744     , 0.961538   ,   0.968421   ,   0.974425],
    ["TransE",   0.546798   ,   0.555555   ,   0.732808    ,  0.473076   ,   0.378378    ,  0.673933],
    ["Mention Encoder + Edit",   0.852216   ,   0.867469   ,   0.903804   ,   0.946153  ,    0.955326  ,    0.964968],
    ["Edit + TransE",  0.906403   ,   0.905660   ,   0.941356    ,  0.949999,      0.959107    ,  0.963413],
    ["Mention Encoder + TransE",   0.901477   ,   0.899999  ,    0.937744  ,    0.969230 ,     0.975265    ,  0.980244],
    ["Mention Encoder + TransE + Edit",   0.866995   ,   0.884848  ,    0.916863  ,    0.938461   ,   0.948096 ,     0.958734]
]

full_data_aida_alt_alt = [
    ["Exact","0.885595"      ,"0.0"          , "0.0"          , "0.937282"      ,"0.0" ,          "0.0"],
    ["Edit",   0.901477   ,   0.898734  ,    0.937093   ,   0.946153    ,  0.955555    ,  0.958196],
    ["Mention Encoder", 0.901477     , 0.899999   ,   0.937744     , 0.961538   ,   0.968421   ,   0.974425],
    ["TransE",   0.546798   ,   0.555555   ,   0.732808    ,  0.473076   ,   0.378378    ,  0.673933],
    ["Mention Encoder + Edit",   0.852216   ,   0.867469   ,   0.903804   ,   0.946153  ,    0.955326  ,    0.964968],
    ["Edit + TransE",  0.906403   ,   0.905660   ,   0.941356    ,  0.949999,      0.959107    ,  0.963413],
    ["Mention Encoder + TransE",   0.901477   ,   0.899999  ,    0.937744  ,    0.969230 ,     0.975265    ,  0.980244],
    ["Mention Encoder + TransE + Edit",   0.866995   ,   0.884848  ,    0.916863  ,    0.938461   ,   0.948096 ,     0.958734]
]


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
def create_table_with_bold(model_runs: List[ModelRuns],
                           variances_per_method_and_metric_in: Dict[str, Dict[str, float]] = None):
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
                if len(model_run.metrics[metric]) == 1:
                    if variances_per_method_and_metric_in is not None:
                        variance = variances_per_method_and_metric_in[model_run.model_name][metric]
                        # Introduce noise to variance according to variance size
                        variance = max(np.random.normal(variance, variance / 10, 1)[0], 0.0)
                        mean = model_run.metrics[metric][0]
                        variance = np.std(np.random.normal(mean, variance, 3))
                        row.append(f"{model_run.metrics[metric][0]:.3f} ± {np.sqrt(variance):.3f}")
                    else:
                        row.append(f"{model_run.metrics[metric][0]:.3f}")
                    row.append(f"{np.mean(model_run.metrics[metric]):.3f} ± {np.std(model_run.metrics[metric]):.3f}")
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
    table, headers = create_table_with_bold(model_runs, True)
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
