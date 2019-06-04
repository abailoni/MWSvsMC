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

    save_dir = os.path.join(get_trendytukan_drive_path(), "projects/agglo_cluster_compare/FullTrainSamples/scores/")
    with open(os.path.join(save_dir, 'scores_collected.json'), 'r') as f:
        collected_scores = json.load(f)

    table = []
    for postproc_type in collected_scores:
        arand = []
        for sample in collected_scores[postproc_type]:
            arand.append(collected_scores[postproc_type][sample]['cremi-score'])
        arand = np.array(arand)
        table.append([postproc_type, arand.mean(), arand.std()])

    table = np.array(table)
    table = table[table[:, 1].argsort()]
    np.savetxt(os.path.join(save_dir, "collected_cremi.csv"), table, delimiter=' & ',
               fmt='%s',
               newline=' \\\\\n')


