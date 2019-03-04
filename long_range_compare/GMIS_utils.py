import numpy as np

BACKGROUND_THRESHOLD = 0.3
from nifty.graph import rag as nrag
from .data_paths import get_hci_home_path, get_trendytukan_drive_path
import os
import h5py
from segmfriends.io.load import parse_offsets

import torch.nn as nn
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
import torch

from torch.autograd import Variable
from segmfriends.transform.segm_to_bound import compute_boundary_mask_from_label_image


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


def combine_affs_and_mask(instance_affs, semantic_affs, semantic_argmax, offsets,
                          combine_with_semantic=True, mask_background_affinities=True):
    if combine_with_semantic:
        combined_affs = combine_affs_with_class(instance_affs, semantic_affs, refine_bike=True,
                                            class_mask=semantic_argmax)
    else:
        combined_affs = instance_affs

    foreground_mask = get_foreground_mask(combined_affs)

    # Reshape affinities in the expected nifty-shape:
    affinities = np.expand_dims(combined_affs.reshape(combined_affs.shape[0], combined_affs.shape[1], -1), axis=0)
    affinities = np.rollaxis(affinities, axis=-1, start=0)

    foreground_mask_affs = compute_boundary_mask_from_label_image(np.expand_dims(foreground_mask, axis=0),
                                                                  offsets, channel_affs=0,
                                                                  pad_mode='constant',
                                                                  pad_constant_values=False,
                                                                  background_value=False,
                                                                  return_affinities=True)
    if mask_background_affinities:
        affinities *= foreground_mask_affs

    return affinities, foreground_mask_affs


# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        # TODO: add batchnorm? So far I used batch_size = 1
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.activation = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        l1 = self.activation(self.linear1(x))
        out = self.linear2(l1)
        return self.sigmoid(out)


class LogRegrModel:
    def __init__(self):
        # Hyper Parameters
        self.input_size = 48
        self.output_size = 48
        self.num_epochs = 2
        # self.learning_rate = 0.001
        self.learning_rate = 0.005

        # training_ratio = 1.
        #
        # all_images_paths = get_GMIS_dataset(partial=False, type="train")
        # print("Number of ROIs: ", len(all_images_paths))
        # nb_images_in_training = int(len(all_images_paths) * training_ratio)
        # print("Training ROIs: ", nb_images_in_training)

        model_path = os.path.join(get_trendytukan_drive_path(),
                                  "GMIS_predictions/logistic_regression_model/pyT_model_train_2.pkl")
        if os.path.exists(model_path):
            print("Model loaded from file!")
            model = torch.load(model_path)
        else:
            model = LogisticRegression(self.input_size, self.output_size)

        # Loss and Optimizer
        # Softmax is internally computed.
        # Set parameters to be updated.
        # criterion = nn.BCEWithLogitsLoss(reduction='none')

        criterion = SorensenDiceLoss()

        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=0.0005)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.to(self.device)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def get_data(self, image_path):
        # Load data:
        with h5py.File(image_path, 'r') as f:
            shape = f['shape'][:]
            strides = f['offset_ranges'][:]
            affs_prob = f['instance_affinities'][:]
            class_prob = f['semantic_affinities'][:]
            class_mask = f['semantic_argmax'][:]
            GT_instances = f['instance_gt'][:]

        offsets = get_offsets(strides)
        # -----------------------------------
        # Pre-process affinities:
        # -----------------------------------
        affinities, foreground_mask_affs = combine_affs_and_mask(affs_prob, class_prob, class_mask, offsets)

        # -----------------------------------
        # Get GT-instance-affinities:
        # -----------------------------------
        GT_instances = np.expand_dims(GT_instances, axis=0)
        # Pixels connected to the background should always predict "split":
        GT_affs = compute_boundary_mask_from_label_image(GT_instances, offsets,
                                                         channel_affs=0, pad_mode='constant',
                                                         pad_constant_values=0,
                                                         background_value=0,
                                                         return_affinities=True)

        # Find edge-mask (False if any of the two pixels includes a GT-background label):
        # we use it to mask the Dice Loss (so we focus the training only on boundaries between instances and not between
        # pixels connected to the background)
        foreground_GT_affs = compute_boundary_mask_from_label_image(GT_instances != 0, offsets,
                                                                    channel_affs=0, pad_mode='constant',
                                                                    pad_constant_values=False,
                                                                    background_value=False,
                                                                    return_affinities=True)

        # real_boundaries = np.logical_and(np.logical_not(GT_affs), foreground_GT_affs)
        # real_inner_parts = np.logical_and(GT_affs, foreground_GT_affs)

        # Pixels connected to the background estimated by GMIS according to the semantic output will be also
        # automatically zeroed. So it does not make sense to give loss on these affinities:
        foreground_GT_affs *= foreground_mask_affs

        # -----------------------------------
        # Reshape for logistic regression:
        # -----------------------------------

        affs_var = Variable(torch.from_numpy(np.rollaxis(affinities.astype('float32'), axis=0, start=4))).to(self.device)
        GT_var = Variable(torch.from_numpy(np.rollaxis(GT_affs.astype('float32'), axis=0, start=4))).to(self.device)
        GT_mask_var = Variable(torch.from_numpy(np.rollaxis(foreground_GT_affs.astype('float32'), axis=0, start=4))).to(self.device)
        is_only_background = foreground_GT_affs.sum() == 0

        return affs_var, GT_var, GT_mask_var, is_only_background

    def infer(self, image_path):
        affs_var, GT_var, GT_mask_var, is_only_background = self.get_data(image_path)

        outputs = self.model(affs_var)

        new_affs = outputs.cpu().data.numpy()

        # Reshape them in the original form:
        current_shape = new_affs.shape
        new_affs = new_affs[0].reshape(current_shape[1], current_shape[2], 6, 8)

        return new_affs

        # with h5py.File(image_path, 'r+') as f:
        #     if 'balanced_affs' in f:
        #         del f['balanced_affs']
        #     f['balanced_affs'] = new_affs.astype('float16')

