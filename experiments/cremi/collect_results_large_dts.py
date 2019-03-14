import long_range_compare

from segmfriends.algorithms.agglo.greedy_edge_contr import runGreedyGraphEdgeContraction
import numpy as np
from nifty.graph import UndirectedGraph

from long_range_compare.data_paths import get_trendytukan_drive_path
import os
import time
import vigra
from segmfriends.utils.various import check_dir_and_create
from segmfriends.utils.config_utils import recursive_dict_update
import json

DATA_DIR = "datasets/CREMI/paul_swoboda_graphs/fruit_fly_brain_segmentation_Pape/multicut_problems/large"

# for problem_name in ["large_problem_L1.txt",
# "large_problem_L0.txt",
#                      ]:
#     path_file = os.path.join(get_trendytukan_drive_path(), DATA_DIR, problem_name)
#     print("Loading data")
#     my_data = np.genfromtxt(path_file, delimiter=' ')
#     print("Writing...")
#     vigra.writeHDF5(my_data, path_file.replace(".txt", ".h5"), 'data')

saving_path = os.path.join(get_trendytukan_drive_path(), "projects/agglo_cluster_compare/bigFruitFlyGraphs/scores/large_problem_L4")

IDs, configs, json_files = [], [], []
for item in os.listdir(saving_path):
    if os.path.isfile(os.path.join(saving_path, item)):
        filename = item
        if not filename.endswith(".json") or filename.startswith("."):
            continue
        # outputs = filename.split("_")
        if len(filename.split("_")) != 2:
            continue
        agglo_type, CLC = filename.split("_")
        result_file = os.path.join(saving_path, filename)
        json_files.append(filename)
        with open(result_file, 'rb') as f:
            file_dict = json.load(f)
        configs.append(file_dict)

collected_results = []
for config in configs:
    for agglo_type in config:
        for CLC in config[agglo_type]:
            new_table_entrance = [agglo_type, str(CLC)]
            new_table_entrance.append('{:.0f}'.format(config[agglo_type][CLC]['MC_energy']))
            new_table_entrance.append('{:.0f}'.format(config[agglo_type][CLC]['runtime']))
            collected_results.append(new_table_entrance)

np.savetxt(os.path.join(saving_path, "collected.csv"), np.array(collected_results), delimiter=' & ', fmt='%s', newline=' \\\\\n')

