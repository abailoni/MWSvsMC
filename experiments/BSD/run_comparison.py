import long_range_compare

from long_range_compare.data_paths import get_trendytukan_drive_path, get_abailoni_hci_home_path
import numpy as np
import os

from imageio import imread

dataset_path = os.path.join(get_abailoni_hci_home_path(), "projects/GASP_on_BSD/images/affinites.h5")
from segmfriends.utils import readHDF5

affinities = readHDF5(dataset_path, "affs")
numbers = readHDF5(dataset_path, "numbers")

from GASP.segmentation import GaspFromAffinities, WatershedOnDistanceTransformFromAffinities
from GASP.utils.various import parse_data_slice


offsets = [
    [0,1,0],
    [0,0,1]
]


def convert_to_weights(edges, bias=0.5):
    beta = np.log(bias/(1-bias))
    z = np.log(edges/(1-edges))
    return 1. / (1. + np.exp(-z-beta))

def run_GASP_on_affinities(affinities,
                           offsets,
                           linkage_criteria="average",
                           beta_bias=0.5,
                           add_cannot_link_constraints=True):
    # Prepare graph pre-processor or superpixel generator:
    # a WSDT segmentation is intersected with connected components
    boundary_kwargs = {
        'boundary_threshold': 0.5,
        # 'used_offsets': [0, 1, 2, 4, 5, 7, 8],
        # 'offset_weights': [1., 1., 1., 1., 1., 0.9, 0.9]
    }
    run_GASP_kwargs = {'linkage_criteria': linkage_criteria,
                       'add_cannot_link_constraints': add_cannot_link_constraints}

    gasp_instance = GaspFromAffinities(offsets,
                                       superpixel_generator=None,
                                       beta_bias=beta_bias,
                                       run_GASP_kwargs=run_GASP_kwargs)
    final_segmentation, runtime = gasp_instance(affinities)
    print("Clustering took {} s".format(runtime))

    return final_segmentation

segmentations_collected = []
for affs, num in zip(affinities[:5], numbers[:5]):
    affs = np.expand_dims(affs, axis=1)

    all_criteria = ["sum", "abs_max", "mean"]

    for criteria in all_criteria:
        segmentations_collected.append(run_GASP_on_affinities(1-affs, offsets, criteria, beta_bias=0.9))


        import segmfriends.vis as vis

        f, ax = vis.get_figure(1,1, figsize=(17,17))
        vis.plot_segm(ax, segmentations_collected[-1], alpha_labels=1, alpha_boundary=0.5)
        vis.save_plot(f, "./", f"BSD_{num}_{criteria}.pdf")
    # break
