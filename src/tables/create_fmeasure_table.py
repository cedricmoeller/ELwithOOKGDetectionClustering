from typing import List

import numpy as np
import tabulate


def calculate_mean_std_and_format(table: List[list]):
    new_table = []
    for item in table:
        new_line = [item[0]]
        for number in item[1:]:
            if isinstance(number, list):
                std = np.std(np.array(number))
                mean = np.mean(np.array(number))
                new_line.append(f"{mean:1.3f} ± {std:1.3f}")
            else:
                new_line.append(f"{number:1.3f}")
        new_table.append(new_line)
    return new_table

headers = ["Model", "Accuracy"]

tabular_data_aida = [
["Desc.", 0.853],
["Pop.", 0.6152698333747824],
["TransE.", 0.6426262123849789],
["Types", 0.47],
["Desc. + Types", 0.852],
["Desc. + Pop.", 0.849],
["Desc. + TransE", 0.862],
["Desc. + Types + Pop.", 0.851],
["Desc. + Pop. + TransE", 0.863],
["Desc. + Types + Pop. + TransE", 0.862],
]

tabular_data_aida_extended = [
["Menton Encoder", [0.853, 0.854265, 0.849539]],
["Pop.", [0.6152698333747824]],
["TransE.", [0.6426262123849789]],
["Types", [0.706, 0.704302, 0.705297]],
["Menton Encoder + Types", [0.852, 0.858990, 0.856752]],
["Menton Encoder + Pop.", [0.849, 0.852265, 0.847539]],
["Menton Encoder + TransE", [0.862,  0.869121, 0.8671]],
["Menton Encoder + Types + Pop.", [0.851, 0.858, 0.856]],
["Menton Encoder. + Pop. + TransE", [0.863, 0.871922, 0.869684]],
["Menton Encoder + Types + Pop. + TransE", [0.862, 0.869932, 0.863964]],
]


table = tabulate.tabulate(tabular_data_aida, headers, tablefmt="latex_booktabs", floatfmt=".3f")
print(table)

table = tabulate.tabulate(calculate_mean_std_and_format(tabular_data_aida_extended), headers, tablefmt="latex_booktabs", floatfmt=".3f")
print(table)
