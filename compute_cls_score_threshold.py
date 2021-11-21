from math import e
import os, sys, copy, gc, yaml

import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

# Dataset evaluation
from detection_eval.detection_eval import DetectionEval
from detection_eval.dataset_filters import build_kitti_filters, build_cadc_filters, build_nuscenes_filters
from detection_eval.box_list import combine_box_lists

# Cluster predictions
from cluster import cluster_preds

from utils import *

dataset_types = ['KITTI', 'CADC', 'NuScenes']
selected_dataset = sys.argv[1] # dataset_types[1]
dataset_path = None
if selected_dataset == dataset_types[0]:
    dataset_path = '/root/kitti'
    logdir = '/root/logdir/output_pkls/kitti'
    gts_path = os.path.join(logdir, 'kitti_infos_val.pkl')
elif selected_dataset == dataset_types[1]:
    dataset_path = '/root/cadc'
    logdir = '/root/logdir/output_pkls/cadc'
    gts_path = os.path.join(logdir, 'cadc_infos_val.pkl')
elif selected_dataset == dataset_types[2]:
    dataset_path = '/root/nusc'
    logdir = '/root/logdir/output_pkls/nuscenes'
    gts_path = os.path.join(logdir, 'nuscenes_infos_10sweeps_val.pkl')

# Load arguments for clustering
ENSEMBLE_TYPE = int(sys.argv[2])
VOTING_STATEGY = int(sys.argv[3])
CLUSTER_MODE = str(sys.argv[4])

# Create path to the pickle file
PKL_FILE = sys.argv[5]
preds_path = os.path.join(logdir, PKL_FILE)

if CLUSTER_MODE == 'softmax':
    print('Cluster mode:', CLUSTER_MODE)
elif CLUSTER_MODE == 'temp_scaled':
    print('Cluster mode:', CLUSTER_MODE)
    yaml_path = os.path.join(logdir + '/tmp_output', PKL_FILE + '_step_two.yaml')
    # Read YAML file
    with open(yaml_path, 'r') as stream:
        CALIBRATION_VALS = yaml.safe_load(stream)
else:
    print('Incorrect cluster mode:', CLUSTER_MODE)
    exit()

if ENSEMBLE_TYPE == -1: # original model
    print('Ensemble type: Sigmoid model')
    MIN_CLUSTER_SIZE = 1
    model_name = 'baseline'
elif ENSEMBLE_TYPE == 0: # mc-dropout
    print('Ensemble type: mc-dropout')
    MIN_CLUSTER_SIZE = 4
    model_name = 'MC dropout'
elif ENSEMBLE_TYPE == 1: # ensemble
    print('Ensemble type: ensemble')
    MIN_CLUSTER_SIZE = 4
    model_name = 'ensemble'
elif ENSEMBLE_TYPE == 2: # mimo
    print('Ensemble type: mimo-ID')
    MIN_CLUSTER_SIZE = 2
    model_name = 'mimo-ID'
elif ENSEMBLE_TYPE == 3: # mimo
    print('Ensemble type: mimo-noID')
    MIN_CLUSTER_SIZE = 2
    model_name = 'mimo-noID'
elif ENSEMBLE_TYPE == 4: # mimo
    print('Ensemble type: mimo-BEV')
    MIN_CLUSTER_SIZE = 2
    model_name = 'mimo-BEV'
else:
    raise NotImplementedError

if VOTING_STATEGY == 0: # Affirmative: all objects kept
    MIN_CLUSTER_SIZE = 1
elif VOTING_STATEGY == 1: # Consensus: Majority must agree
    MIN_CLUSTER_SIZE = int(MIN_CLUSTER_SIZE/2 + 1)
elif VOTING_STATEGY == 2: # Unanimous: All must agree
    MIN_CLUSTER_SIZE = MIN_CLUSTER_SIZE
else:
    raise NotImplementedError

print('Minimum cluster size of', MIN_CLUSTER_SIZE)

def pcdet_get_labels(data_dict):
    if isinstance(data_dict, list):
        data_dict = data_dict[0]
    for label_key in ['labels', 'gt_labels', 'pred_labels']:
        if label_key in data_dict:
            return data_dict[label_key]
    if 'annos' in data_dict:
        data_dict = data_dict['annos']
    for label_key in ['name', 'gt_names']:
        if label_key in data_dict:
            if selected_dataset == 'KITTI':
                classes = dict(
                    Car=1, Pedestrian=2, Cyclist=3,
                    Truck=-1, Misc=-1, Van=-1, Tram=-1, Person_sitting=-1,
                    DontCare=-1
                )
            elif selected_dataset == 'CADC':
                classes = dict(
                    Car=1, Pedestrian=2, Pickup_Truck=3
                )
            elif selected_dataset == 'NuScenes':
                classes = dict(
                    car=1, truck=2, construction_vehicle=3, bus=4, trailer=5, \
                    barrier=6, motorcycle=7, bicycle=8, pedestrian=9, traffic_cone=10, \
                    ignore=-1
                )
            labels = []
            for name in data_dict[label_key]:
                if name == 'DontCare':
                    continue
                labels.append(classes[name])
            return labels
    raise ValueError()

def pcdet_get_scores(data_dict):
    if isinstance(data_dict, list):
        data_dict = data_dict[0]
    if 'score' in data_dict:
        return data_dict['score']

def pcdet_get_boxes(data_dict):
    if isinstance(data_dict, list):
        data_dict = data_dict[0]
    if 'annos' in data_dict:
        data_dict = data_dict['annos']
    for box_key in ['gt_boxes', 'gt_boxes_lidar', 'boxes_lidar', 'box3d_lidar']:
        if box_key in data_dict:
            if selected_dataset == 'NuScenes':
                return data_dict[box_key][...,:7]
            else:
                return data_dict[box_key]
    return ValueError()

def gt_processor(data_dict):
    return (
        pcdet_get_labels(data_dict),
        pcdet_get_boxes(data_dict)
    )

def pred_processor(data_dict):
    return (
        pcdet_get_labels(data_dict),
        pcdet_get_scores(data_dict),
        pcdet_get_boxes(data_dict)
    )

# No points required
def attach_data(sample_idx, gt_dict, pred_dict, gt_list, pred_list):
    for i in range(len(gt_list)):
        gt_list.data[i] = dict(gt_boxes=pcdet_get_boxes(gt_dict)[i])
    if 'score_all' not in pred_dict: # original model
        for i in range(len(pred_list)):
            pred_list.data[i] = dict(
                boxes_lidar=pred_dict['boxes_lidar'][i]
            )
    else:
        for i in range(len(pred_list)):
            pred_list.data[i] = dict(
                score_all=pred_dict['score_all'][i],
                boxes_lidar=pred_dict['boxes_lidar'][i],
                pred_vars=pred_dict['pred_vars'][i],
                shannon_entropy=pred_dict['shannon_entropy'][i],
                aleatoric_entropy=pred_dict['aleatoric_entropy'][i]
            )

def main():
    print("Load dictionaries...")
    gt_dicts, pred_dicts = load_dicts(gts_path, preds_path)

    print("Using FIRST half of val set as the recal set.")
    half_point = int(len(gt_dicts)/2)
    gt_dicts = gt_dicts[:half_point]
    pred_dicts = pred_dicts[:half_point]

    # KITTI data loader has the issue where they don't deep copy the variable
    # When converting from lidar to camera frame so the center of the cuboid
    # is at the bottom instead of the center of the bounding box
    if selected_dataset == dataset_types[0]:
        print('Fixing KITTI Z values by adding H/2')
        if isinstance(pred_dicts[0], dict):
            for i in range(len(pred_dicts)):
                for j in range(len(pred_dicts[i]['boxes_lidar'])):
                    pred_dicts[i]['boxes_lidar'][j][2] += pred_dicts[i]['boxes_lidar'][j][5] / 2
        else:
            for i in range(len(pred_dicts)):
                for fwd_pass in range(len(pred_dicts[i])):
                    for j in range(len(pred_dicts[i][fwd_pass]['boxes_lidar'])):
                        pred_dicts[i][fwd_pass]['boxes_lidar'][j][2] += \
                            pred_dicts[i][fwd_pass]['boxes_lidar'][j][5] / 2


    if CLUSTER_MODE == 'softmax':
        print("Clustering predictions by frame...")
        clustered_pred_dicts = cluster_preds(pred_dicts, selected_dataset, CLUSTER_MODE, MIN_CLUSTER_SIZE)

    # Threshold (list or dict) maps a label to a matching threshold
    if selected_dataset == dataset_types[0]:
        kitti_filter_list = build_kitti_filters(gts_path)
        filter_list = [
            kitti_filter_list[1], # Car moderate
            kitti_filter_list[4], # Ped moderate
            kitti_filter_list[7]  # Cyc moderate
        ]

        class_names = ['Car', 'Pedestrian', 'Cyclist']
        NUM_CLASSES = 3
        eval_criterion = 'iou_3d'
        num_recall_positions = 40
    elif selected_dataset == dataset_types[1]:
        cadc_filter_list = build_cadc_filters(gts_path)
        filter_list = [
            cadc_filter_list[1], # Car moderate
            cadc_filter_list[4], # Ped moderate
            cadc_filter_list[7]  # Pickup_Truck moderate
        ]

        class_names = ['Car', 'Pedestrian', 'Pickup_Truck']
        NUM_CLASSES = 3
        eval_criterion = 'iou_3d'
        num_recall_positions = 40
    elif selected_dataset == dataset_types[2]:
        nusc_threshold = 0.75
        thresholds = {
            -1: -1,     # Just to not output error
            0: -1,      # Just to not output error
            1: nusc_threshold,    # Car
            2: nusc_threshold,    # Truck
            3: nusc_threshold,    # Construction Vehicle
            4: nusc_threshold,     # Bus
            5: nusc_threshold,     # Trailer
            6: nusc_threshold,     # Barrier
            7: nusc_threshold,     # Motorcycle
            8: nusc_threshold,     # Bicycle
            9: nusc_threshold,     # Pedestrian
            10: nusc_threshold     # Traffic Cone
        }
        
        # Since there are no difficulty levels we can directly get the filter
        filter_list = build_nuscenes_filters(gts_path)

        class_names = ['Car', 'Truck', 'Construction Vehicle', 'Bus', 'Trailer', \
            'Barrier', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Traffic Cone']
        NUM_CLASSES = 10
        eval_criterion = 'euclidean_3d'
        num_recall_positions = 100

    # Dictionary with key as the class and a list for each threshold
    gt_output = {}
    pred_output = {}

    iou_thresholds = np.arange(0.5, 1.0, 0.05).round(2)

    for filter_idx in range(len(filter_list)):
        print("Performing evaluation for class " + filter_list[filter_idx].class_name + \
            " and difficulty " + str(filter_list[filter_idx].difficulty))

        # Init lists for this class
        gt_output[class_names[filter_idx]] = []
        pred_output[class_names[filter_idx]] = []

        for iou_threshold_idx in range(len(iou_thresholds)):
            # Evaluate over validation set
            print("Loop through and evaluate all samples at iou:", iou_thresholds[iou_threshold_idx])

            if CLUSTER_MODE == 'temp_scaled':
                print("Clustering predictions by frame...")
                t_vals = (CALIBRATION_VALS[class_names[filter_idx]]['cls'][iou_threshold_idx], \
                    CALIBRATION_VALS[class_names[filter_idx]]['reg'][iou_threshold_idx])
                clustered_pred_dicts = cluster_preds(pred_dicts, selected_dataset, CLUSTER_MODE, MIN_CLUSTER_SIZE, t_vals)

            gt_list, pred_list = DetectionEval.evaluate_all_samples(
                gt_dicts, clustered_pred_dicts,
                thresholds=create_thresholds(iou_thresholds[iou_threshold_idx], selected_dataset),
                criterion=eval_criterion,
                filta=filter_list[filter_idx],
                gt_processor=gt_processor,
                pred_processor=pred_processor,
                pbar=tqdm(total=len(gt_dicts)),
                callback=attach_data
            )

            gt_output[class_names[filter_idx]].append(gt_list)
            pred_output[class_names[filter_idx]].append(pred_list)

    # To calculate mean over all classes
    best_f1_score_thresholds = []

    # Clear memory, mostly needed for large datasets like NuScenes
    del gt_dicts
    del pred_dicts
    gc.collect()

    for idx in range(len(class_names)):
        # Store results for this class and iou threshold
        tmp_best_f1_score_thresholds = []

        for iou_threshold_idx in range(len(iou_thresholds)):
            print("Evaluating Uncertainty for " + class_names[idx] + " at iou " + str(iou_thresholds[iou_threshold_idx]))
            # Get current gt and pred list
            gt_list = gt_output[class_names[idx]][iou_threshold_idx]
            pred_list = pred_output[class_names[idx]][iou_threshold_idx]

            # A prediction box is either a TP, location error, duplicate or FP
            # TP: valid, localized and classified
            tp = pred_list.valid & pred_list.localized & pred_list.classified & ~pred_list.duplicated
            # Duplicate: is a TP but there is a better matched TP
            dup = pred_list.valid & pred_list.localized & pred_list.classified & pred_list.duplicated
            # Location Errors are a combination of:
            # Type 1: Correctly localized, misclassified, with valid GT class
            l_mc = pred_list.valid & (pred_list.localized) & (~pred_list.classified) & (pred_list.gt_labels > 0) & (pred_list.gt_labels <= NUM_CLASSES)
            # Type 2: Mislocalized, correctly classified
            ml_c = pred_list.valid & (~pred_list.localized) & (pred_list.classified)
            # FP Type 2: Mislocalized, misclassified, with valid GT class
            ml_mc = pred_list.valid & (~pred_list.localized) & (~pred_list.classified) & \
                        (~pred_list.bg) & (pred_list.gt_labels > 0) & (pred_list.gt_labels <= NUM_CLASSES)
            loc_err = l_mc | ml_c | ml_mc
            # FP: is valid and either not localized or not classified correctly
            # This is a broader definition of FP that we are not using
            # fp = pred_list.valid & ~(pred_list.localized & pred_list.classified)
            # FP: Completely missed
            fp = pred_list.valid & (pred_list.bg | (pred_list.gt_labels <= 0) )

            fn_bg = (~gt_list.ignored) & (gt_list.bg)


            print('Number of predictions', len(pred_list))
            print('Number of TPs', len(pred_list[tp]))
            print('Number of Duplicates', len(pred_list[dup]))
            print('Number of Localization Errors', len(pred_list[loc_err]))
            print('  LE Type 1: Number of Correctly localized, misclassified', len(pred_list[l_mc]))
            print('  LE Type 2: Number of Mislocalized, correctly classified', len(pred_list[ml_c]))
            print('  LE Type 3: Number of Mislocalized, misclassified', len(pred_list[ml_mc]))
            print('Number of FPs', len(pred_list[fp]))

            gt_tp = (~gt_list.ignored) & (gt_list.localized) & (gt_list.classified)
            # Correctly localized, misclassified
            gt_l_mc = (~gt_list.ignored) & (gt_list.localized) & (~gt_list.classified)
            # Mislocalized, correctly classified
            gt_ml_c = (~gt_list.ignored) & (~gt_list.localized) & (gt_list.classified)
            # Mislocalized, misclassified
            gt_ml_mc = (~gt_list.ignored) & (~gt_list.localized) & (~gt_list.classified) & (~gt_list.bg)
            # Completely missed
            gt_bg = (~gt_list.ignored) & (gt_list.bg)
            print('Number of GT TPs', len(gt_list[gt_tp]))
            print('Number of GT Loc errors', len(gt_list[gt_l_mc | gt_ml_c | gt_ml_mc]))
            print('Number of FN_BGs', len(gt_list[fn_bg]))

            stats_result = DetectionEval.compute_statistics(gt_list=gt_list, \
                                pred_list=pred_list, n_positions=num_recall_positions)
            print('Number of FNs', stats_result['fn'])
        
            curr_ap = (100 * stats_result['ap'])
            print('AP@40', curr_ap.round(2))

            max_f1_score, max_f1_score_threshold = stats_result['f1']
            # item converts to python number
            tmp_best_f1_score_thresholds.append(float(max_f1_score_threshold))

        best_f1_score_thresholds.append(tmp_best_f1_score_thresholds)

    table = PrettyTable()
    table.field_names = (['Class Name', 'Mean CLS score threshold'])

    for idx in range(len(class_names)):
        table.add_row([
            class_names[idx] + ":",
            '{:.2f}'.format(np.nanmean(best_f1_score_thresholds[idx]))
        ])

    table.add_row([
        "All:",
        '{:.2f}'.format(np.nanmean([
            np.nanmean(best_f1_score_thresholds[0]),
            np.nanmean(best_f1_score_thresholds[1]),
            np.nanmean(best_f1_score_thresholds[2])
        ]))
    ])

    print(table)
    print(best_f1_score_thresholds)

    # Create dictionary that holds score thresholds for each class
    output_dict = {}
    for i in range(len(best_f1_score_thresholds)):
        output_dict[class_names[i]] = best_f1_score_thresholds[i]

    if CLUSTER_MODE == 'softmax':
        yaml_path = os.path.join(logdir + '/tmp_output', PKL_FILE + '_step_one.yaml')
    elif CLUSTER_MODE == 'temp_scaled':
        yaml_path = os.path.join(logdir + '/tmp_output', PKL_FILE + '_step_three.yaml')

    with open(yaml_path, 'w') as file:
        yaml.dump(output_dict, file)

main()
