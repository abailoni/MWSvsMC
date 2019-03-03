# Add missing package-paths
import long_range_compare


import os
import h5py
from long_range_compare.data_paths import get_hci_home_path, get_trendytukan_drive_path


from long_range_compare.load_datasets import get_GMIS_dataset

image_paths = get_GMIS_dataset(type="val", partial=False)

key_map = {
    'class_probs': 'semantic_affinities',
    'gt_instances': 'instance_gt',
    'gt_labels': 'semantic_gt',
    'probs': 'instance_affinities',
    'semantic_predictions': 'semantic_prediction',
    'strides': 'offset_ranges'
}

# old = ['class_probs', 'gt_instances', 'gt_labels', 'probs', 'semantic_argmax', 'semantic_predictions', 'shape', 'strides']
# new = ['instance_affinities', 'instance_gt', 'offset_ranges', 'semantic_affinities', 'semantic_argmax', 'semantic_gt',
#          'semantic_prediction', 'shape']

# [`instance_affinities`, `instance_gt`, `offset_ranges`, `semantic_affinities`, `semantic_argmax`, `semantic_gt`, `semantic_prediction`, `shape`]

for i, path in enumerate(image_paths):
    if i%50 == 0:
        print(i, end=' ', flush=True)
    with h5py.File(path, 'r+') as f:
        print([k for k in f])
        print(f["semantic_prediction"].shape)
        break
        # for old_key in key_map:
        #     f[key_map[old_key]] = f[old_key]
        #     del f[old_key]