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
    scores_collected = []
    all_agglo_type = [' ']


    def assign_color(value, good_thresh, bad_thresh, nb_flt, best="lowest"):
        if best == "lowest":
            if value < good_thresh:
                return '{{\color{{ForestGreen}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
            if value > good_thresh and value < bad_thresh:
                return '{{\color{{Orange}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
            if value > bad_thresh:
                return '{{\color{{Red}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
        elif best == "highest":
            if value > good_thresh:
                return '{{\color{{ForestGreen}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
            if value < good_thresh and value > bad_thresh:
                return '{{\color{{Orange}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
            if value < bad_thresh:
                return '{{\color{{Red}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
        else:
            raise ValueError
    for subdir, dirs, files in os.walk(results_dir):
        for i, agglo_type in enumerate(dirs):
            json_file = os.path.join(subdir, agglo_type, 'resultInstanceLevelSemanticLabeling.json')
            if not os.path.exists(json_file):
                continue

            if 'finetuned_affs_avg_thresh0' not in agglo_type and 'ORIG' not in agglo_type:
            # if 'orig_affs_thresh030' not in agglo_type and 'ORIG' not in agglo_type:
                continue

            with open(json_file, 'rb') as f:
                result_dict = json.load(f)

            scores_collected.append(result_dict['averages']['allAp'])
            parsed = agglo_type.split("_")
            if "dont_use_log_costs" in agglo_type:
                use_logs = "False"
            elif "use_log_costs" in agglo_type:
                use_logs = "True"
            else:
                use_logs = "-"
            thresh = [partial for partial in parsed if "thresh" in partial]
            if len(thresh) == 0 or len(parsed) == 1:
                thresh = "-"
                update_rule = parsed[0]
            else:
                update_rule = parsed[0] if parsed[1] != "constr" else parsed[0]+parsed[1]
                thresh = thresh[0]
            # scores = [agglo_type.replace('_', ' ')]
            scores = [update_rule, thresh, use_logs]
            scores.append(assign_color(result_dict['averages']['allAp'],0.34,0.3,4, "highest"))
            scores.append(assign_color(result_dict['averages']['allAp50%'],0.53,0.5,4, "highest"))
            # scores.append('{:.4f}'.format(result_dict['averages']['allAp50%']))

            # for cls in result_dict['instLabels']:
            #     scores.append( '{:.2f}'.format(result_dict['averages']['classes'][cls]['ap']))

            result_matrix.append(np.array(scores))

            labels = result_dict['instLabels']

    # labels = [' ', 'Overall mAP'] + labels
    labels = [' ', 'AP', 'AP50\% ']


    # result_matrix = [labels] + result_matrix

    result_matrix = np.array(result_matrix)
    result_matrix = result_matrix[np.array(scores_collected).argsort()[::-1]]

    np.savetxt(os.path.join(results_dir, "../GMIS_orig.csv"), result_matrix, delimiter=' & ', fmt='%s', newline=' \\\\\n')


