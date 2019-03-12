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


all_data = {}

for problem_name in ["large_problem_L4.txt",
                    "large_problem_L3.txt",
                     "large_problem_L2.txt",
                    "large_problem_L1.txt",
                    # "large_problem_L0.txt"
                     ]:
    path_file = os.path.join(get_trendytukan_drive_path(), DATA_DIR, problem_name)
    print("Loading data")
    # my_data = np.genfromtxt(path_file, delimiter=' ')
    # print("Writing...")
    # vigra.writeHDF5(my_data, path_file.replace(".txt", ".h5"), 'data')

    all_data[problem_name] = vigra.readHDF5(path_file.replace(".txt", ".h5"), 'data')



def run_agglo(problem_name=None, update_rule='mean', CLC=False):
    saving_path = os.path.join(get_trendytukan_drive_path(), "projects/agglo_cluster_compare/bigFruitFlyGraphs/scores")
    check_dir_and_create(saving_path)
    saving_path_2 = os.path.join(saving_path,
                                 problem_name.replace(".txt", ''))

    check_dir_and_create(saving_path_2)
    result_file_path = os.path.join(saving_path_2,
                                    "{}_{}.json".format(update_rule, CLC))
    if os.path.exists(result_file_path):
        print(result_file_path)
        print("Skip agglo ", update_rule, CLC, problem_name)
        return

    uvIds = all_data[problem_name][:,:2].astype('uint64')
    edge_weights = all_data[problem_name][:, 2].astype('float64')


    print("Building graph")
    graph = UndirectedGraph(uvIds.max())
    graph.insertEdges(uvIds)

    # Run agglo:
    tick = time.time()
    print("Start agglo ", update_rule, CLC, problem_name)
    try:
        nodeSeg, _ = runGreedyGraphEdgeContraction(graph, edge_weights, update_rule=update_rule, add_cannot_link_constraints=CLC)
    except RuntimeError:
        print("Nifty Exception on {} {} {}!".format(problem_name, update_rule, CLC))
    tock = time.time()

    edge_labels = graph.nodesLabelsToEdgeLabels(nodeSeg)
    MC_energy = (edge_weights * edge_labels).sum()

    print(tock-tick)
    print(MC_energy)

    new_results = {}
    new_results[update_rule] = {}
    new_results[update_rule][CLC] = {}

    new_results[update_rule][CLC]['MC_energy'] = MC_energy
    new_results[update_rule][CLC]['runtime'] = tock - tick


    with global_lock:
        # if os.path.exists(result_file_path):
        #     try:
        #         with open(result_file_path, 'r') as f:
        #             result_dict = json.load(f)
        #     except Exception:
        #         result_dict = {}
        #         print("Exception raised on {}, {}!!".format(problem_name, update_rule) )
        # else:
        #     result_dict = {}
        #
        # result_dict = recursive_dict_update(new_results, result_dict)

        with open(result_file_path, 'w') as f:
            try:
                json.dump(new_results, f, indent=4, sort_keys=True)
            except Exception:
                print("Exception again!")
        print("saved")


kwargs_iter = []
for prob_name in ["large_problem_L4.txt",
                    "large_problem_L3.txt",
                     "large_problem_L2.txt",
                    "large_problem_L1.txt",
                    # "large_problem_L0.txt"
                     ]:
    for agglo_type in ['mean', 'max', 'min', 'sum', 'MutexWatershed']:
        for CLC_var in [False, True]:
            if agglo_type == 'MutexWatershed' and CLC_var:
                continue
            kwargs_iter.append({"update_rule": agglo_type, "CLC": CLC_var, "problem_name": prob_name})


from segmfriends.utils.various import starmap_with_kwargs
from multiprocessing.pool import ThreadPool
from itertools import repeat

from multiprocessing import Lock
def get_lock():
    global global_lock
    global_lock = Lock()

pool = ThreadPool(processes=3, initializer=get_lock)
starmap_with_kwargs(pool, run_agglo, args_iter=repeat([]),
                    kwargs_iter=kwargs_iter)
pool.close()
pool.join()
