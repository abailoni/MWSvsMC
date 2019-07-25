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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="FullTestSamples")  #DebugExp
    parser.add_argument('--project_directory', default="projects/agglo_cluster_compare",  type=str)
    # TODO: option to pass some other fixed kwargs and overwrite it...?

    args = parser.parse_args()

    exp_name = args.exp_name

    fixed_kargs = {
        "experiment_name": exp_name,
        "project_directory": os.path.join(get_trendytukan_drive_path(), args.project_directory),
        "configs_dir_path": os.path.join(get_hci_home_path(), "pyCharm_projects/longRangeAgglo/experiments/cremi/configs")
    }

    # Select experiment and load data:
    experiment = cremi_experiments.get_experiment_by_name(exp_name)(fixed_kwargs=fixed_kargs)
    kwargs_iter, nb_threads_pool = experiment.get_data()
    print("Agglomarations to run: ", len(kwargs_iter))

    # for kwrg in kwargs_iter:
    #     path = os.path.join(get_hci_home_path(), "datasets/cremi/tmp_cropped_train_data/compressed/sample_{}.h5".format(kwrg["sample"]))
    #     print(path)
    #     from segmfriends.utils.various import readHDF5, writeHDF5
    #     writeHDF5(kwrg["GT"].astype('uint32'), path, "gt")
    #     writeHDF5(kwrg["affinities"].astype('float16'), path, "affinities")
    #     vigra.writeHDF5(kwrg["GT"], path, "GT")
    #     print("GT wrote")
    #     vigra.writeHDF5(kwrg["affinities"], path, "affs")
    #     print("affs wrote")

    #
    # Start pool:
    pool = ThreadPool(processes=nb_threads_pool)
    starmap_with_kwargs(pool, cremi_utils.run_clustering, args_iter=repeat([]),
                        kwargs_iter=kwargs_iter)
    pool.close()
    pool.join()
