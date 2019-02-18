import sys

sys.path += ["/home/abailoni_local/hci_home/python_libraries/nifty/python",
"/home/abailoni_local/hci_home/python_libraries/cremi_python",
"/home/abailoni_local/hci_home/python_libraries/affogato/python",
"/home/abailoni_local/hci_home/pyCharm_projects/inferno",
"/home/abailoni_local/hci_home/pyCharm_projects/MWSvsMC",
"/home/abailoni_local/hci_home/pyCharm_projects/constrained_mst",
"/home/abailoni_local/hci_home/pyCharm_projects/neuro-skunkworks",
"/home/abailoni_local/hci_home/pyCharm_projects/segmfriends",
"/home/abailoni_local/hci_home/pyCharm_projects/hc_segmentation",
"/home/abailoni_local/hci_home/pyCharm_projects/neurofire",]

sys.path += ["/home/abailoni/hci_home/python_libraries/nifty/python",
"/home/abailoni/hci_home/python_libraries/cremi_python",
"/home/abailoni/hci_home/python_libraries/affogato/python",
"/home/abailoni/hci_home/pyCharm_projects/inferno",
"/home/abailoni/hci_home/pyCharm_projects/MWSvsMC",
"/home/abailoni/hci_home/pyCharm_projects/constrained_mst",
"/home/abailoni/hci_home/pyCharm_projects/neuro-skunkworks",
"/home/abailoni/hci_home/pyCharm_projects/segmfriends",
"/home/abailoni/hci_home/pyCharm_projects/hc_segmentation",
"/home/abailoni/hci_home/pyCharm_projects/neurofire",]


sys.path += [
"/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation",
"/net/hciserver03/storage/abailoni/pyCharm_projects/MWSvsMC",
"/net/hciserver03/storage/abailoni/python_libraries/affogato/python",
]

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from h5py import highlevel
import vigra
import nifty as nf
import nifty.graph.agglo as nagglo
import numpy as np
import os

from skimage import io
import argparse
import time
import yaml
import json
import h5py

import getpass
from PIL import Image


from long_range_compare import utils as utils

from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update
from segmfriends.utils import yaml2dict, parse_data_slice

from segmfriends.io.save import get_hci_home_path, get_trendytukan_drive_path
from segmfriends.algorithms.agglo import GreedyEdgeContractionAgglomeraterFromSuperpixels
from segmfriends.algorithms.WS.WS_growing import SizeThreshAndGrowWithWS
from segmfriends.algorithms.blockwise import BlockWise

from skunkworks.metrics.cremi_score import cremi_score
from long_range_hc.postprocessing.pipelines import get_segmentation_pipeline

from long_range_compare.load_datasets import get_dataset_data, get_dataset_offsets, CREMI_crop_slices, CREMI_sub_crops_slices, get_GMIS_dataset

from long_range_compare import GMIS_utils as GMIS_utils


from shutil import copyfile

result_root_dir = os.path.join(get_trendytukan_drive_path(), "datasets/cityscape/data/gtFine_trainvaltest/out")

collected_result_dir = os.path.join(get_trendytukan_drive_path(), "datasets/cityscape/data/gtFine_trainvaltest/out/COLLECTED")

original_images_dir = os.path.join(get_trendytukan_drive_path(), "datasets/cityscape/data/leftImg8bit_trainvaltest/leftImg8bit/val")

image_name = 'munster/munster_000167_000019_leftImg8bit_combine.inst.jpg'
image_name = 'frankfurt/frankfurt_000001_020693_leftImg8bit_combine.inst.jpg'
image_name = "munster/munster_000167_000019_leftImg8bit_combine.inst.jpg"
image_name = 'frankfurt/frankfurt_000001_015768_leftImg8bit_combine.inst.jpg'

ignored = ["COLLECTED", "MAX_bk_mask", "MEAN_bk_mask", "MEAN_constr_bk_mask"]

for subdir, dirs, files in os.walk(result_root_dir):
    # Copy original image:
    original_image_path = os.path.join(original_images_dir, image_name.replace("_combine.inst.jpg", ".png"))
    copyfile(original_image_path, os.path.join(collected_result_dir, image_name.replace("_combine.inst.jpg", ".png")))

    for agglo_type in dirs:
        if agglo_type in ignored or ("clean" not in agglo_type and "ORIG" not in agglo_type):
            continue
        main_dir = os.path.join(subdir, agglo_type)
        agglo_image = os.path.join(main_dir, image_name)
        if not os.path.exists(agglo_image):
            print( "Image not found in {}".format(main_dir))
            continue

        # Copy it in collected folder:
        copyfile( agglo_image, os.path.join(collected_result_dir, image_name.replace(".jpg", "_{}.jpg".format(agglo_type))) )

