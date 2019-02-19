# Add missing package-paths
import long_range_compare


import vigra
import numpy as np
import os

import time
import h5py


from segmfriends.utils import yaml2dict, parse_data_slice

import matplotlib.pyplot as plt

from long_range_compare.load_datasets import get_dataset_data, get_dataset_offsets, CREMI_crop_slices, CREMI_sub_crops_slices, get_GMIS_dataset

from long_range_compare import GMIS_utils as GMIS_utils

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out




def get_training_data(image_path):
    # Load data:
    with h5py.File(image_path, 'r') as f:
        shape = f['shape'][:]
        strides = f['strides'][:]
        affs_prob = f['probs'][:]
        class_prob = f['class_probs'][:]
        class_mask = f['semantic_argmax'][:]
        GT_instances = f['gt_instances'][:]
        # affs_prob = f['real_affs'][:]


    # TODO: real_affs, combine, foreground
    # -----------------------------------
    # Pre-process affinities:
    # -----------------------------------
    strides = np.array([1, 2, 4, 8, 16, 32], dtype=np.int32)
    offsets = GMIS_utils.get_offsets(strides)

    # combined_affs = affs_prob
    combined_affs = GMIS_utils.combine_affs_with_class(affs_prob, class_prob, refine_bike=True, class_mask=class_mask)

    foreground_mask = GMIS_utils.get_foreground_mask(combined_affs)

    # Reshape affinities in the expected nifty-shape:
    affinities = np.expand_dims(combined_affs.reshape(combined_affs.shape[0], combined_affs.shape[1], -1), axis=0)
    affinities = np.rollaxis(affinities, axis=-1, start=0)

    foreground_mask_affs = GMIS_utils.compute_real_background_mask(np.expand_dims(foreground_mask, axis=0), offsets, channel_affs=0)
    affinities *= foreground_mask_affs

    # -----------------------------------
    # Get GT-instance-affinities:
    # -----------------------------------
    GT_instances = np.expand_dims(GT_instances, axis=0)
    foreground_GT_affs = GMIS_utils.compute_real_background_mask(GT_instances != 0, offsets,
                                                        channel_affs=0)
    from segmfriends.transform.segm_to_bound import compute_mask_boundaries
    GT_affs = np.logical_not(compute_mask_boundaries(GT_instances, offsets, channel_affs=0))

    real_bundaries = np.logical_and(np.logical_not(GT_affs), foreground_GT_affs)
    real_inner_parts = np.logical_and(GT_affs, foreground_GT_affs)

    # Consider also estimated background:
    foreground_GT_affs *= foreground_mask_affs

    # -----------------------------------
    # Reshape for logistic regression:
    # -----------------------------------

    affs_var = Variable(torch.from_numpy(np.rollaxis(affinities.astype('float32'), axis=0, start=4)))
    GT_var = Variable(torch.from_numpy(np.rollaxis(GT_affs.astype('float32'), axis=0, start=4)))
    GT_mask_var = Variable(torch.from_numpy(np.rollaxis(foreground_GT_affs.astype('float32'), axis=0, start=4)))
    is_only_background = foreground_GT_affs.sum() == 0


    #
    # # # # Debug plots:
    # # # #
    # from segmfriends import vis as vis
    # for off_stride in [0, 24]:
    #     # affs_repr = GMIS_utils.get_affinities_representation(affinities[:off_stride+8], offsets[:off_stride+8])
    #     # affs_repr = GMIS_utils.get_affinities_representation(affinities[16:32], offsets[16:32])
    #     # affs_repr = np.rollaxis(affs_repr, axis=0, start=4)[0]
    #     # if affs_repr.min() < 0:
    #     #     affs_repr += np.abs(affs_repr.min())
    #     # affs_repr /= affs_repr.max()
    #
    #
    #     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    #     for a in fig.get_axes():
    #         a.axis('off')
    #
    #
    #     # affs_repr = np.linalg.norm(affs_repr, axis=-1)
    #     # ax.imshow(affs_repr, interpolation="none")
    #
    #     vis.plot_output_affin(ax, affinities*foreground_GT_affs.astype('float32'), nb_offset=off_stride+3, z_slice=0)
    #     # vis.plot_output_affin(ax, np.zeros_like(real_inner_parts, dtype='float32'), nb_offset=off_stride+3, z_slice=0)
    #
    #     pdf_path = image_path.replace(
    #         '.input.h5', '.affs_{}.pdf'.format(off_stride))
    #
    #     pdf_path = "./GT_affs_{}_C.pdf".format(off_stride)
    #     # fig.savefig(pdf_path)
    #     vis.save_plot(fig, os.path.dirname(pdf_path), os.path.basename(pdf_path))

    return affs_var, GT_var, GT_mask_var, is_only_background










if __name__ == '__main__':
    # Hyper Parameters
    input_size = 1
    output_size = 1
    num_epochs = 2
    learning_rate = 0.005

    training_ratio = 0.6



    all_images_paths = get_GMIS_dataset(partial=True)
    print("Number of ROIs: ", len(all_images_paths))
    nb_images_in_training = int(len(all_images_paths) * training_ratio)
    print("Training ROIs: ", nb_images_in_training)

    model = LogisticRegression(input_size, output_size)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    # TODO: choose optimizer and learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Training the Model
    for epoch in range(num_epochs):
        for i, image_path in enumerate(all_images_paths[:nb_images_in_training]):

            affs_var, GT_var, GT_mask_var, is_only_background = get_training_data(image_path)


            if not is_only_background:
                # Forward + Backward + Optimize
                affs_var, GT_var, GT_mask_var = affs_var.to(device), GT_var.to(device), GT_mask_var.to(device)

                optimizer.zero_grad()
                outputs = model(affs_var.unsqueeze(4)).squeeze(4)
                loss = criterion(outputs, GT_var)
                loss = (loss * GT_mask_var).sum() / GT_mask_var.sum()
                loss.backward()
                optimizer.step()

                if (i + 1) % 5 == 0:
                    print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(all_images_paths), loss.cpu().data[0]))


    # Test the Model
    true_boundaries = None

    NEW_found_boundaries = None
    NEW_true_found_boundaries = None
    OLD_found_boundaries = None
    OLD_true_found_boundaries = None


    sigmoid = nn.Sigmoid().to(device)

    for i, image_path in enumerate(all_images_paths[nb_images_in_training:]):
        affs_var, GT_var, GT_mask_var, is_only_background = get_training_data(image_path)
        if not is_only_background:
            # Forward + Backward + Optimize
            affs_var, GT_var, GT_mask_var = affs_var.to(device), GT_var.to(device), GT_mask_var.to(device)

            # TODO: should I take care of something...?
            outputs = model(affs_var.unsqueeze(4)).squeeze(4)

            # # Compute loss:
            loss = criterion(outputs, GT_var)
            # loss = (loss * GT_mask_var).sum() / GT_mask_var.sum()
            loss = (loss * GT_mask_var)

            # Compute precision and recall:
            nb_offsets = affs_var.size()[-1]


            old_predictions = (((1. - affs_var) * GT_mask_var) > 0.5).view(-1, nb_offsets)
            new_predictions = (((1. - sigmoid(outputs)) * GT_mask_var) > 0.5).view(-1, nb_offsets)
            targets = ( ((1. - GT_var) * GT_mask_var) > 0.5).view(-1, nb_offsets)

            from segmfriends import vis as vis
            # np_predictions = sigmoid(outputs).cpu().data.numpy()
            np_predictions = loss.cpu().data.numpy()
            np_predictions = (affs_var * GT_mask_var).cpu().data.numpy()
            for off_stride in [16, 32]:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
                for a in fig.get_axes():
                    a.axis('off')
                vis.plot_output_affin(ax, np.rollaxis(np_predictions, axis=-1, start=0), nb_offset=off_stride+3, z_slice=0)
                pdf_path = "./GT_affs_{}.pdf".format(off_stride)
                vis.save_plot(fig, os.path.dirname(pdf_path), os.path.basename(pdf_path))

            np_predictions = (sigmoid(outputs)* GT_mask_var).cpu().data.numpy()
            for off_stride in [16, 32]:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
                for a in fig.get_axes():
                    a.axis('off')
                vis.plot_output_affin(ax, np.rollaxis(np_predictions, axis=-1, start=0), nb_offset=off_stride+3, z_slice=0)
                pdf_path = "./GT_affs_{}_B.pdf".format(off_stride)
                vis.save_plot(fig, os.path.dirname(pdf_path), os.path.basename(pdf_path))

            np_predictions = (GT_var * GT_mask_var).cpu().data.numpy()
            for off_stride in [16, 32]:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
                for a in fig.get_axes():
                    a.axis('off')
                vis.plot_output_affin(ax, np.rollaxis(np_predictions, axis=-1, start=0), nb_offset=off_stride + 3,
                                      z_slice=0)
                pdf_path = "./GT_affs_{}_C.pdf".format(off_stride)
                vis.save_plot(fig, os.path.dirname(pdf_path), os.path.basename(pdf_path))

            last_true_boundaries = targets.sum(dim=0)

            last_new_found_boundaries = new_predictions.sum(dim=0)
            last_new_true_found_boundaries = ( new_predictions * targets ).sum(dim=0)

            last_old_found_boundaries = old_predictions.sum(dim=0)
            last_old_true_found_boundaries = (old_predictions * targets).sum(dim=0)

            if true_boundaries is None:
                true_boundaries =  last_true_boundaries
                NEW_found_boundaries = last_new_found_boundaries
                NEW_true_found_boundaries = last_new_true_found_boundaries
                OLD_found_boundaries = last_old_found_boundaries
                OLD_true_found_boundaries = last_old_true_found_boundaries
            else:
                true_boundaries += last_true_boundaries
                NEW_found_boundaries += last_new_found_boundaries
                NEW_true_found_boundaries += last_new_true_found_boundaries
                OLD_found_boundaries += last_old_found_boundaries
                OLD_true_found_boundaries += last_old_true_found_boundaries

            break

    print("New Precision: {}".format(NEW_true_found_boundaries.data.float() / NEW_found_boundaries.data.float()))
    print("New Recall: {}".format(NEW_true_found_boundaries.data.float() / true_boundaries.data.float()))

    print("Old Precision: {}".format(OLD_true_found_boundaries.data.float() / OLD_found_boundaries.data.float()))
    print("Old Recall: {}".format(OLD_true_found_boundaries.data.float() / true_boundaries.data.float()))



