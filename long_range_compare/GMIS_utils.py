import numpy as np

BACKGROUND_THRESHOLD = 0.3
from nifty.graph import rag as nrag
from .data_paths import get_hci_home_path, get_trendytukan_drive_path
import os
from segmfriends.io.load import parse_offsets

def get_offsets(strides):
    offset_file = os.path.join(get_hci_home_path(),
                               'pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/',
                               'GMIS_offsets.json')
    offsets = np.array(parse_offsets(offset_file))
    all_offsets = []
    for str in strides:
        all_offsets.append(offsets * str)
    return np.concatenate(all_offsets, axis=0)





def sigma_fun(x):
    return (1. / (1. + np.exp(-5 * x)) -0.5) * 2.

def get_foreground_mask(affinities):
    return np.max(np.max(affinities, axis=-1), axis=-1) > BACKGROUND_THRESHOLD

def combine_affs_with_class(affinities, class_affinities, refine_bike=False, class_mask=None):
    combined_affs = sigma_fun(class_affinities) * affinities
    if refine_bike:
        assert class_mask is not None
        # assert class_mask.max() <= 7, "Apparently there are more than 7 classes...!"

        # Restrict to the bike class:
        bike_mask = class_mask >= 7
        bike_mask = np.tile(bike_mask, reps=(affinities.shape[2], affinities.shape[3], 1, 1))
        bike_mask = np.transpose(bike_mask, (2, 3, 0, 1))

        # Restrict to long-range edges:
        long_range_mask = np.zeros_like(affinities, dtype='bool')
        long_range_mask[:,:,-1,:] = True

        # Apply another sigma function only to these edges:
        combined_mask = np.logical_and(bike_mask, long_range_mask)
        # if combined_mask.max():
        #     print("Bike refined!")
        combined_affs[combined_mask] = sigma_fun(combined_affs[combined_mask])

    return combined_affs


def get_confidence_scores(instance_labels, affinities, offsets, size_thresh = 256,
                                                  minimum_score=0.4):
    assert instance_labels.ndim == 3 and affinities.ndim == 4 and offsets.shape[1] == 3, "Expect 3D data here"
    rag = nrag.gridRag(instance_labels.astype('uint32'))

    confidence_scores, sizes, max_aff = nrag.accumulateAffinitiesMeanAndLengthOnNodes(
        rag,
        instance_labels.astype('int'),
        np.rollaxis(affinities, axis=0, start=4),
        offsets,
        np.ones(offsets.shape[0], dtype='float32'),
        numberOfThreads=1
    )


    # Set background confidence to zero:
    confidence_scores[0] = 0.
    assert confidence_scores.shape[0] == instance_labels.max() + 1
    if sizes.shape[0] > 1:
        if not all(sizes[1:] > 20):
            print(sizes)
            assert all(sizes[1:] > 20)

        # # Get rid of tiny instances:
        # node_sizes = nrag.accumulateMeanAndLength(rag, instance_labels*0.)[1][:, 1]
        # node_sizes[0] = 10000 # mod for ignoring background
        # size_mask = node_sizes < size_thresh
        # score_mask = confidence_scores[1:] > minimum_score
        # # Find instances to keep: (big instances or
        # np.argwhere(size_mask)
        # confidence_scores = np.delete(confidence_scores, size_mask)


    return confidence_scores


def get_affinities_representation(affinities, offsets):
    """
    :param affinities: shape (nb_offset, z, x, y) ---> 1 if merge, 0 if not merge
    :param offsets: shape (nb_offsets, 3)
    """
    out = np.zeros((3,) + affinities.shape[1:], dtype='float32')
    for nb_offs, offs in enumerate(offsets):
        for i in range(3):
            out[i] += affinities[nb_offs] * offsets[nb_offs, i]

    return out




