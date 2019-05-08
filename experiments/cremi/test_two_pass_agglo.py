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
from long_range_compare.load_datasets import get_dataset_offsets

from functools import partial

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="BlockWiseExp")  #DebugExp
    parser.add_argument('--project_directory', default="projects/agglo_cluster_compare",  type=str)
    # TODO: option to pass some other fixed kwargs and overwrite it...?

    args = parser.parse_args()

    exp_name = args.exp_name

    fixed_kargs = {
        "experiment_name": exp_name,
        "project_directory": os.path.join(get_trendytukan_drive_path(), args.project_directory),
        "configs_dir_path": os.path.join(get_hci_home_path(), "pyCharm_projects/longRangeAgglo/experiments/cremi/configs")
    }

    # Select experiment and load kwargs:
    experiment = cremi_experiments.get_experiment_by_name(exp_name)(fixed_kwargs=fixed_kargs)
    kwargs_iter, nb_threads_pool = experiment.get_data()
    print("Agglomarations to run: ", len(kwargs_iter))

    # # Start pool:
    # pool = ThreadPool(processes=nb_threads_pool)
    # starmap_with_kwargs(pool, cremi_utils.run_clustering, args_iter=repeat([]),
    #                     kwargs_iter=kwargs_iter)

    from long_range_compare.two_pass_utls import two_pass_agglomeration, GUACA_agglomerator
    kwargs = kwargs_iter[0]
    affinities = kwargs.pop("affinities")
    offsets = get_dataset_offsets(kwargs.pop("dataset"))
    block_shape = [6, 300, 300]
    halo = [4, 30, 30]



    post_proc = cremi_utils.get_postproc_config(kwargs['agglo'],
                                    kwargs["configs_dir_path"],
                                    kwargs["edge_prob"],
                                    kwargs["local_attraction"],
                                    kwargs["save_UCM"],
                                    kwargs["from_superpixels"], kwargs["use_multicut"],
                                    kwargs["additional_model_keys"],
                                    kwargs["mask_used_edges"]
                                    )
    agglomerator = partial(GUACA_agglomerator, **post_proc["generalized_HC_kwargs"]["agglomeration_kwargs"])


    final_segm = two_pass_agglomeration(affinities, offsets, agglomerator, block_shape, halo, n_threads=6)

    vigra.writeHDF5(final_segm, "./test_two_pass.h5", 'data')


    # from segmfriends.vis import plot_segm, get_figure, save_plot
    # fig, ax = get_figure(1,1)
    # plot_segm(ax, final_segm, z_slice=26, alpha_labels=1.)
    # save_plot(fig, './', 'two_pass_test.pdf')



