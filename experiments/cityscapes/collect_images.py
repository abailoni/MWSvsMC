# Add missing package-paths
import long_range_compare


import os
from long_range_compare.data_paths import get_hci_home_path, get_trendytukan_drive_path


from shutil import copyfile

result_root_dir = os.path.join(get_trendytukan_drive_path(), "datasets/cityscape/data/gtFine_trainvaltest/out")

collected_result_dir = os.path.join(get_trendytukan_drive_path(), "datasets/cityscape/data/gtFine_trainvaltest/out/COLLECTED")

original_images_dir = os.path.join(get_trendytukan_drive_path(), "datasets/cityscape/data/leftImg8bit_trainvaltest/leftImg8bit/val")

image_name = 'munster/munster_000167_000019_leftImg8bit_combine.inst.jpg'
image_name = 'frankfurt/frankfurt_000001_020693_leftImg8bit_combine.inst.jpg'
# image_name = "munster/munster_000167_000019_leftImg8bit_combine.inst.jpg"
# image_name = 'frankfurt/frankfurt_000001_015768_leftImg8bit_combine.inst.jpg'


ignored = ["COLLECTED", "MAX_bk_mask", "MEAN_bk_mask", "MEAN_constr_bk_mask"]

for subdir, dirs, files in os.walk(result_root_dir):
    # Copy original image:
    original_image_path = os.path.join(original_images_dir, image_name.replace("_combine.inst.jpg", ".png"))
    copyfile(original_image_path, os.path.join(collected_result_dir, image_name.replace("_combine.inst.jpg", ".png")))

    for agglo_type in dirs:
        # if agglo_type in ignored or ("clean" not in agglo_type and "ORIG" not in agglo_type):
        if agglo_type in ignored or ("finetuned_affs_avg_thresh0" not in agglo_type and "ORIG" not in agglo_type):
            continue
        main_dir = os.path.join(subdir, agglo_type)
        agglo_image = os.path.join(main_dir, image_name)
        if not os.path.exists(agglo_image):
            print( "Image not found in {}".format(main_dir))
            continue

        # Copy it in collected folder:
        copyfile( agglo_image, os.path.join(collected_result_dir, image_name.replace(".jpg", "_{}.jpg".format(agglo_type))) )

