import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import vigra
import numpy as np
import json
import os

import matplotlib.animation as manimation

__MAX_LABEL__ = 1000000
rand_cm = matplotlib.colors.ListedColormap(np.random.rand(__MAX_LABEL__, 3))
segm_plot_kwargs = {'vmax': __MAX_LABEL__, 'vmin':0}

def save_UCM_video(filename, root_path, selected_offset = 1,
                    selected_slice = 0,
                    nb_frames = 100,
                   postfix="", final_segm=None):
    ucm_path = os.path.join(root_path, "UCM", filename + ".h5")
    config_path = os.path.join(root_path, filename + ".json")

    assert len(filename.split("_")) == 4

    ID, sample, agglo_type, _ = filename.split("_")
    with open(config_path, 'rb') as f:
        config_dict = json.load(f)

    UCM = vigra.readHDF5(ucm_path, 'merge_times')[:, selected_slice]

    mask_1 = UCM == -15
    nb_nodes = UCM.max()
    mask_2 = UCM == nb_nodes

    nb_iterations = (UCM * np.logical_not(mask_2)).max()
    nb_iterations_at_frame = np.linspace(0, nb_iterations, nb_frames)

    masked_UCM = UCM.copy()
    masked_UCM[mask_1] = 0
    masked_UCM = masked_UCM[1:].max(axis=0)

    print("Done")

    # Prepare video:
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie UCM', artist='alberto.bailoni@iwr.uni-heidelberg.de',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig, ax = plt.subplots(ncols=1, nrows=1,
                         figsize=(7, 7))

    data = np.ones_like(masked_UCM) * nb_iterations
    img = ax.matshow(data, cmap=plt.get_cmap('binary'), vmin=0, vmax=nb_iterations,
                         interpolation='none')
    segm_plt = ax.matshow(np.ma.masked_where(np.ones_like(data, dtype='bool'), data), cmap=rand_cm, alpha=0.4, interpolation='none', **segm_plot_kwargs)

    # vis.plot_output_affin(ax, affinities, nb_offset=16, z_slice=0)

    plt.subplots_adjust(wspace=0, hspace=0)


    with writer.saving(fig, "UCM_{}_{}.mp4".format(config_dict['agglo_type'], postfix), nb_frames):
        for i in range(nb_frames):
            frame_data = masked_UCM.copy()
            frame_data[frame_data > nb_iterations_at_frame[i]] = nb_iterations
            frame_data[frame_data < nb_iterations_at_frame[i]] = 0
            img.set_data(frame_data)

            if final_segm is not None and i == nb_frames-1:
                segm_plt.set_data(final_segm[selected_slice])

            writer.grab_frame()