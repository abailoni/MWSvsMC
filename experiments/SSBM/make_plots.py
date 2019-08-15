import long_range_compare # Add missing package-paths

import vigra
import numpy as np
import os
import argparse
from multiprocessing.pool import ThreadPool
from itertools import repeat

from long_range_compare.data_paths import get_hci_home_path, get_trendytukan_drive_path
from long_range_compare import SSBM_utils as SSBM_utils
from long_range_compare import SSBM_experiments as SSBM_experiments

from segmfriends.utils.various import starmap_with_kwargs



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="SSBMExperiment")  #DebugExp
    parser.add_argument('--project_directory', default="projects/spectral_compare",  type=str)
    # parser.add_argument('--project_directory', default="projects/agglo_cluster_compare/quadxeon5_results",  type=str)
    # parser.add_argument('--project_directory', default="../quadxeon5_scratch/projects/agglo_cluster_compare",  type=str)

    args = parser.parse_args()

    exp_name = args.exp_name
    project_dir = os.path.join(get_trendytukan_drive_path(), args.project_directory)
    # project_dir = os.path.join(get_hci_home_path(), "../quadxeon5_scratch", args.project_directory)

    fixed_kargs = {
        "experiment_name": exp_name,
        "project_directory": project_dir,
    }

    # Select experiment and plot results:
    experiment = SSBM_experiments.get_experiment_by_name(exp_name)(fixed_kwargs=fixed_kargs)
    experiment.make_plots(project_dir)



