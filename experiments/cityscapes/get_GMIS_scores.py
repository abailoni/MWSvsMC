# Add missing package-paths
import long_range_compare

from long_range_compare.data_paths import get_hci_home_path, get_trendytukan_drive_path
import numpy as np
import json
import os


if __name__ == '__main__':

    # Create dictionary:
    results_collected = {}

    results_dir = os.path.join(get_trendytukan_drive_path(), 'datasets/cityscape/data/gtFine_trainvaltest/evaluationResults/eval_out')

    result_matrix = []
    all_agglo_type = [' ']



    for subdir, dirs, files in os.walk(results_dir):
        for i, agglo_type in enumerate(dirs):
            json_file = os.path.join(subdir, agglo_type, 'resultInstanceLevelSemanticLabeling.json')
            if not os.path.exists(json_file):
                continue

            if 'clean' not in agglo_type and 'ORIG' not in agglo_type:
                continue
            with open(json_file, 'rb') as f:
                result_dict = json.load(f)

            scores = [agglo_type.replace('_', ' ')]
            scores.append('{:.4f}'.format(result_dict['averages']['allAp']))
            scores.append('{:.4f}'.format(result_dict['averages']['allAp50%']))

            # for cls in result_dict['instLabels']:
            #     scores.append( '{:.2f}'.format(result_dict['averages']['classes'][cls]['ap']))

            result_matrix.append(np.array(scores))

            labels = result_dict['instLabels']

    # labels = [' ', 'Overall mAP'] + labels
    labels = [' ', 'AP', 'AP50\% ']


    result_matrix = [labels] + result_matrix

    ndarray = np.array(result_matrix)

    np.savetxt(os.path.join(results_dir, "GMIS.csv"), ndarray, delimiter=' & ', fmt='%s', newline=' \\\\\n')


