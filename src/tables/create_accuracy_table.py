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

headers = ["ooKG probability", "P", "R", "F1"]

tabular_data_aida = [
    [0.0,0.7952989502510269, 0.8666998259139518, 0.8294656670236822],
    [0.05,  0.860040,      0.843571 ,     0.851726],
    [0.1, 0.860134  ,    0.827406   ,   0.843452],
    [0.2, 0.875267   ,   0.814971     , 0.844043],
    [0.3, 0.875372527770252   ,   0.8035314598358617   ,   0.837914937759336],
    [0.4, 0.8718371837183718   ,   0.7883611042029346   ,   0.8280005223978059],
    [0.5, 0.880173   ,   0.756279  ,    0.813536],
    [1.0,  0.446540  ,    0.070629   ,   0.121966],
]

tabular_data_aida_avg = [
    [0.0,[0.7952989502510269, 0.798037], [0.8666998259139518,0.869684], [0.8294656670236822, 0.832321]],
    [0.05,  [0.850112,0.860040],      [0.842079,0.843571] ,     [0.851726, 0.846076]],
    [0.1, [0.860134, 0.871312]  ,    [0.827406, 0.830141]   ,   [0.843452, 0.850229]],
    [0.2, [0.866347,0.875267]   ,   [0.809251,0.814971]     , [0.844043, 0.836826]],
    [0.3, [0.878961,0.875372527770252]   ,   [0.800049,0.8035314598358617]   ,   [0.837914937759336, 0.837651]],
    [0.4, [0.884921,0.8718371837183718]   ,   [0.7883611042029346, 0.801293]   ,   [0.8280005223978059, 0.841033]],
    [0.5, [0.880173,0.879463]   ,   [0.765729,0.756279]  ,    [0.818665, 0.813536]],
    [1.0,  0.446540  ,    0.070629   ,   0.121966],
]

tabular_data_wikievents = [
    [0.0, 0.7711664160554822   ,   0.8337480508277761   ,   0.8012370871062365],
[0.05, 0.818571    ,  0.830629  ,    0.824556],
[0.1, 0.831878    ,  0.828539   ,   0.830205],
[0.2, 0.850952  ,    0.824159    ,  0.837341],
[0.3, 0.861062  ,    0.820410   ,   0.840245],
[0.4, 0.875355    ,  0.815964  ,    0.844617],
[0.5, 0.886932   ,   0.808865   ,   0.846101],
[1.0, 0.011278   ,   0.000398   ,   0.000769],
]

table = tabulate.tabulate(calculate_mean_std_and_format(tabular_data_aida_avg), headers, tablefmt="latex_booktabs", floatfmt=["", ".3f", ".3f",".3f"])
print(table)

table = tabulate.tabulate(tabular_data_wikievents, headers, tablefmt="latex_booktabs", floatfmt=["", ".3f", ".3f",".3f"])
print(table)