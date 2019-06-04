import long_range_compare # Add missing package-paths

import vigra
import numpy as np
import os
import argparse
from multiprocessing.pool import ThreadPool
from itertools import repeat

from long_range_compare.data_paths import get_hci_home_path, get_trendytukan_drive_path
from long_range_compare import cremi_utils as cremi_utils
from long_range_compare import cremi_experiments as cremi_experiments

from segmfriends.utils.various import starmap_with_kwargs
from skunkworks.metrics.cremi_score import cremi_score
import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="FullTestSamples")  #DebugExp
    parser.add_argument('--project_directory', default="projects/agglo_cluster_compare",  type=str)
    # TODO: option to pass some other fixed kwargs and overwrite it...?

    args = parser.parse_args()

    exp_name = args.exp_name
    proj_dir = os.path.join(get_trendytukan_drive_path(), args.project_directory)

    results_collected = {}

    for segm_key in ["long_range_problem/multicut","long_range_problem/lifted_multicut", "short_range_problem/lifted_multicut","short_range_problem/multicut",   "thresholded", "watershed"]:
        results_collected[segm_key] = {}
        for sample in ["A", "B", "C"]:

            data_path = os.path.join(get_hci_home_path(), "../cpape/Work/data/sample{}_results_constantin.h5".format(sample))
            GT_path = os.path.join(get_hci_home_path(), "datasets/cremi/tmp_cropped_train_data/sample_{}.h5".format(sample))
            save_dir = os.path.join(get_trendytukan_drive_path(), "projects/agglo_cluster_compare/FullTrainSamples/scores/")

            # Load:
            segm = vigra.readHDF5(data_path,segm_key)
            GT = vigra.readHDF5(GT_path, 'GT')

            print("Computing scores...")
            evals = cremi_score(GT, segm, border_threshold=None, return_all_scores=True)
            print(segm_key, sample, evals)

            results_collected[segm_key][sample] = evals

            json_file_path = os.path.join(save_dir, 'scores_{}_{}.json'.format(segm_key.replace("/","_"), sample))
            with open(json_file_path, 'w') as f:
                json.dump(evals, f, indent=4, sort_keys=True)

    with open(os.path.join(save_dir, 'scores_collected.json'), 'w') as f:
        json.dump(results_collected, f, indent=4, sort_keys=True)
