import matplotlib
matplotlib.use('Agg')
import nifty.graph.rag as nrag
import vigra
import numpy as np

try:
    from cremi.evaluation import NeuronIds
    from cremi import Volume
except ImportError:
    print("Warning: for the evaluation score the cremi repository should be installed")


# -----------------------
# HELPER FUNCTIONS:
# -----------------------
MAX_PLOTTED_LABEL = 1000000
matplotlib.rcParams.update({'font.size': 5})
rand_cm = matplotlib.colors.ListedColormap(np.random.rand(MAX_PLOTTED_LABEL, 3))

DEF_INTERP = 'none'
segm_plot_kwargs = {'vmax': MAX_PLOTTED_LABEL, 'vmin':0}

def mask_the_mask(mask, value_to_mask=0., interval=None):
    if interval is not None:
        return np.ma.masked_where(np.logical_and(mask < interval[1], mask > interval[0]), mask)
    else:
        return np.ma.masked_where(np.logical_and(mask < value_to_mask+1e-3, mask > value_to_mask-1e-3), mask)

def plot_segm(target, segm, z_slice=0, background=None, mask_value=None, highlight_boundaries=True, plot_label_colors=True):
    """Shape of expected background: (z,x,y)"""
    segm, _, _ = vigra.analysis.relabelConsecutive(segm.astype('uint32'))
    if background is not None:
        target.matshow(background[z_slice], cmap='gray', interpolation=DEF_INTERP)

    if mask_value is not None:
        segm = mask_the_mask(segm,value_to_mask=mask_value)
    if plot_label_colors:
        target.matshow(segm[z_slice], cmap=rand_cm, alpha=0.4, interpolation=DEF_INTERP, **segm_plot_kwargs)
    if highlight_boundaries:
        masked_bound = get_masked_boundary_mask(segm)
        target.matshow(masked_bound[z_slice], cmap='gray', alpha=0.6, interpolation=DEF_INTERP)

def map_features_to_label_array(label_array, features, ignore_label=-1,
                                fill_value=0., number_of_threads=1):
    """

    :param label_array:
            accepted nb. of dimensions 3 (check if 2 and 4 also work...)
            max label is N
    :param features: array with shape (N, M) where M is the number of features
    :param ignore_label: the label in label_array that should be ignored in the mapping
    :param fill_value: the fill value used in the mapped array to replace the ignore_label
    :return: array with shape: label_array.shape + (M, )
    """
    if label_array.ndim != 3:
        raise NotImplementedError("Bug in nifty function...! Atm only 3-dimensional arrays are accepted!")

    if ignore_label is None:
        ignore_label = -1

    # Using multi-threaded nifty version:
    return nrag.mapFeaturesToLabelArray(label_array.astype(np.int64),
                                        features.astype(np.float64),
                                        ignore_label,
                                        fill_value,
                                        numberOfThreads=number_of_threads)

def compute_mask_boundaries(label_image,
                                offsets,
                                compress_channels=False,
                                channel_affs=-1):
    """
    Faster than the nifty version, but does not check the actual connectivity of the segments (no rag is
    built). A non-local edge could be cut, but it could also connect not-neighboring segments.

    It returns a boundary mask (1 on boundaries, 0 otherwise). To get affinities reverse it.

    :param offsets: numpy array
        Example: [ [0,1,0], [0,0,1] ]

    :param return_boundary_affinities:
        if True, the output shape is (len(axes, z, x, y)
        if False, the shape is       (z, x, y)

    :param channel_affs: accepted options are 0 or -1
    """
    assert label_image.ndim == 3
    ndim = 3

    padding = [[0,0] for _ in range(3)]
    for ax in range(3):
        padding[ax][1] = offsets[:,ax].max()

    padded_label_image = np.pad(label_image, pad_width=padding, mode='edge')
    crop_slices = [slice(0, padded_label_image.shape[ax]-padding[ax][1]) for ax in range(3)]

    boundary_mask = []
    for offset in offsets:
        rolled_segm = padded_label_image
        for ax, offset_ax in enumerate(offset):
            if offset_ax!=0:
                rolled_segm = np.roll(rolled_segm, -offset_ax, axis=ax)
        boundary_mask.append((padded_label_image != rolled_segm)[crop_slices])

    boundary_affin = np.stack(boundary_mask)



    if compress_channels:
        compressed_mask = np.zeros(label_image.shape[:ndim], dtype=np.int8)
        for ch_nb in range(boundary_affin.shape[0]):
            compressed_mask = np.logical_or(compressed_mask, boundary_affin[ch_nb])
        return compressed_mask

    if channel_affs==0:
        return boundary_affin
    else:
        assert channel_affs == -1
        return np.transpose(boundary_affin, (1,2,3,0))


def get_bound_mask(segm):
    # print("B mask is expensive...")
    return compute_mask_boundaries(segm,
                                   np.array([[0,1,0], [0,0,1]]),
                                   compress_channels=True)

def get_masked_boundary_mask(segm):
    #     bound = np.logical_or(get_boundary_mask(segm)[0, 0],get_boundary_mask(segm)[1, 0])
    bound = get_bound_mask(segm)
    return mask_the_mask(bound)



def cremi_score(gt, seg, return_all_scores=False, border_threshold=None):
    # # the zeros must be kept in the gt since they are the ignore label
    gt = vigra.analysis.labelVolumeWithBackground(gt.astype(np.uint32))
    # seg = vigra.analysis.labelVolume(seg.astype(np.uint32))

    seg = np.array(seg)
    seg = np.require(seg, requirements=['C'])
    # Make sure that all labels are strictly positive:
    seg = seg.astype('uint32')
    # FIXME: it seems to have some trouble with label 0 in the segmentation:
    seg += 1

    gt = np.array(gt)
    gt = np.require(gt, requirements=['C'])
    gt = (gt - 1).astype('uint32')
    # assert gt.min() >= -1


    gt_ = Volume(gt)
    seg_ = Volume(seg)

    metrics = NeuronIds(gt_, border_threshold=border_threshold)
    arand = metrics.adapted_rand(seg_)

    vi_s, vi_m = metrics.voi(seg_)
    cs = np.sqrt(arand * (vi_s + vi_m))
    # cs = (vi_s + vi_m + arand) / 3.
    if return_all_scores:
        return {'cremi-score': cs, 'vi-merge': vi_m, 'vi-split': vi_s, 'adapted-rand': arand}
    else:
        return cs


def probs_to_costs(probs,
                   beta=.5):
    # Probs: prob. map (0: merge; 1: split)
    p_min = 0.001
    p_max = 1. - p_min
    # Costs: positive (merge), negative (split)
    costs = (p_max - p_min) * probs + p_min

    # probabilities to energies, second term is boundary bias
    costs = np.log((1. - costs) / costs) + np.log((1. - beta) / beta)

    return costs
