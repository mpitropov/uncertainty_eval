from math import e
import os, sys, copy, gc, csv

import numpy as np
from tqdm import tqdm

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

# Create path to the pickle file
PKL_FILE = sys.argv[4]
preds_path = os.path.join(logdir, PKL_FILE)

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
                # classes = dict(
                #     Car=1, Pedestrian=2, Cyclist=3,
                #     Person_sitting=4, Van=5, Truck=6, Tram=7, Misc=8,
                #     DontCare=-1
                # )
                classes = dict(
                    Car=1, Pedestrian=2, Cyclist=3,
                    Person_sitting=-1, Van=-1, Truck=-1, Tram=-1, Misc=-1,
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

def pcdet_get_occlusion(data_dict):
    if isinstance(data_dict, list):
        data_dict = data_dict[0]
    if 'annos' in data_dict:
        data_dict = data_dict['annos']
    if 'occluded' in data_dict:
        return data_dict['occluded']
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

def attach_data(sample_idx, gt_dict, pred_dict, gt_list, pred_list):
    for i in range(len(gt_list)):
        gt_list.data[i] = dict(
            gt_boxes=pcdet_get_boxes(gt_dict)[i], occluded=pcdet_get_occlusion(gt_dict)[i])
    if 'score_all' not in pred_dict: # original model
        for i in range(len(pred_list)):
            pred_list.data[i] = dict(
                boxes_lidar=pred_dict['boxes_lidar'][i]#,
                # num_pts_in_pred=pred_dict['num_pts_in_pred'][i]
            )
    else:
        for i in range(len(pred_list)):
            pred_list.data[i] = dict(
                score_all=pred_dict['score_all'][i],
                boxes_lidar=pred_dict['boxes_lidar'][i],
                pred_vars=pred_dict['pred_vars'][i],
                shannon_entropy=pred_dict['shannon_entropy'][i],
                aleatoric_entropy=pred_dict['aleatoric_entropy'][i],
                mutual_info=pred_dict['mutual_info'][i],
                epistemic_total_var=pred_dict['epistemic_total_var'][i],
                aleatoric_total_var=pred_dict['aleatoric_total_var'][i],
                # num_pts_in_pred=pred_dict['num_pts_in_pred'][i],
                dist=np.linalg.norm(pred_dict['boxes_lidar'][i][:2])
            )

def main():
    print("Load dictionaries...")
    gt_dicts, pred_dicts = load_dicts(gts_path, preds_path)
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
        
    print("Clustering predictions by frame...")
    CLUSTER_MODE = 'softmax'
    pred_dicts = cluster_preds(pred_dicts, selected_dataset, CLUSTER_MODE, MIN_CLUSTER_SIZE)

    # # Add predictions to the points
    # add_num_pts_full_dataset(selected_dataset, dataset_path, pred_dicts)

    # Threshold (list or dict) maps a label to a matching threshold
    if selected_dataset == dataset_types[0]:
        thresholds = {
            -1: 1,     # Just to not output error
            0: 1,      # Just to not output error
            1: 0.7,    # Car
            2: 0.5,    # Pedestrian
            3: 0.5     # Cyclist
        }
        # thresholds = [
        #     {
        #         -1: 1,     # Just to not output error (DontCare)
        #         0: 1,      # Just to not output error
        #         1: 0.7,    # Car
        #         2: 1.0,    # Pedestrian
        #         3: 1.0,    # Cyclist
        #         4: 0.1,    # Person_sitting
        #         5: 0.4,    # Van
        #         6: 0.2,    # Truck
        #         7: 0.1,    # Tram
        #         8: 0.5     # Misc
        #     },
        #     {
        #         -1: 1,     # Just to not output error (DontCare)
        #         0: 1,      # Just to not output error
        #         1: 1.0,    # Car
        #         2: 0.5,    # Pedestrian
        #         3: 1.0,    # Cyclist
        #         4: 0.5,    # Person_sitting
        #         5: 0.1,    # Van
        #         6: 0.1,    # Truck
        #         7: 0.1,    # Tram
        #         8: 0.1     # Misc
        #     },
        #     {
        #         -1: 1,     # Just to not output error (DontCare)
        #         0: 1,      # Just to not output error
        #         1: 1.0,    # Car
        #         2: 1.0,    # Pedestrian
        #         3: 0.5,    # Cyclist
        #         4: 0.3,    # Person_sitting
        #         5: 0.1,    # Van
        #         6: 0.1,    # Truck
        #         7: 0.1,    # Tram
        #         8: 0.1     # Misc
        #     }
        # ]
        VAN_CLASS = 5

        kitti_filter_list = build_kitti_filters(gts_path)
        filter_list = [
            kitti_filter_list[1], # Car moderate
            kitti_filter_list[4], # Ped moderate
            kitti_filter_list[7]  # Cyc moderate
        ]

        class_names = ['Car', 'Pedestrian', 'Cyclist', 'All']
        NUM_CLASSES = 3
        eval_criterion = 'iou_3d'
        num_recall_positions = 40
    elif selected_dataset == dataset_types[1]:
        thresholds = {
            -1: 1,     # Just to not output error
            0: 1,      # Just to not output error
            1: 0.7,    # Car
            2: 0.5,    # Pedestrian
            3: 0.7     # Pickup_Truck
        }
        
        cadc_filter_list = build_cadc_filters(gts_path)
        filter_list = [
            cadc_filter_list[1], # Car moderate
            cadc_filter_list[4], # Ped moderate
            cadc_filter_list[7]  # Pickup_Truck moderate
        ]

        class_names = ['Car', 'Pedestrian', 'Pickup_Truck', 'All']
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
            'Barrier', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Traffic Cone', 'All']
        NUM_CLASSES = 10
        eval_criterion = 'euclidean_3d'
        num_recall_positions = 100

    gt_list_list = []
    pred_list_list = []

    for filter_idx in range(len(filter_list)):
        print("Performing evaluation for class " + filter_list[filter_idx].class_name + \
            " and difficulty " + str(filter_list[filter_idx].difficulty))

        # Evaluate over validation set
        print("Loop through and evaluate all samples...")
        gt_list, pred_list = DetectionEval.evaluate_all_samples(
            gt_dicts, pred_dicts,
            thresholds=thresholds, #[filter_idx],
            criterion=eval_criterion,
            filta=filter_list[filter_idx],
            gt_processor=gt_processor,
            pred_processor=pred_processor,
            pbar=tqdm(total=len(gt_dicts)),
            callback=attach_data
        )

        gt_list_list.append(gt_list)
        pred_list_list.append(pred_list)

    # Create combined gt and pred_list
    gt_list_list.append(combine_box_lists(gt_list_list))
    pred_list_list.append(combine_box_lists(pred_list_list))

    # To calculate mean over all classes
    ap_list = []
    output_csv_row = []

    # Clear memory, mostly needed for large datasets like NuScenes
    del gt_dicts
    del pred_dicts
    gc.collect()

    for idx in range(len(class_names)):
        print("Evaluating Uncertainty for " + class_names[idx] + "...")

        # Get current gt and pred list
        gt_list = gt_list_list[idx]
        pred_list = pred_list_list[idx]

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
        fp = pred_list.valid & ( pred_list.bg | (pred_list.gt_labels <= 0) )

        fn_bg = (~gt_list.ignored) & (gt_list.bg)

        # # Detected objects that are not of the current eval class and not van
        # class_idx = idx + 1
        # ignored = pred_list.valid & pred_list.localized & (pred_list.gt_labels != class_idx) & (pred_list.gt_labels != VAN_CLASS)


        print('Number of predictions', len(pred_list))
        print('Number of TPs', len(pred_list[tp]))
        print('Number of Duplicates', len(pred_list[dup]))
        print('Number of Localization Errors', len(pred_list[loc_err]))
        print('  LE Type 1: Number of Correctly localized, misclassified', len(pred_list[l_mc]))
        print('  LE Type 2: Number of Mislocalized, correctly classified', len(pred_list[ml_c]))
        print('  LE Type 3: Number of Mislocalized, misclassified', len(pred_list[ml_mc]))
        print('Number of FP_BGs', len(pred_list[fp]))

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
        
        if idx == NUM_CLASSES: # Use mean of APs
            mAP = np.mean(ap_list).round(2)
            print('mAP@40', mAP)
            output_csv_row.append(mAP)
            output_csv_row.append(len(pred_list[tp]))
            output_csv_row.append(len(pred_list[ml_c]))
            output_csv_row.append(len(pred_list[fp]))
        else:
            curr_ap = (100 * stats_result['ap'])
            ap_list.append(curr_ap)
            output_csv_row.append(curr_ap.round(2))
            print('AP@40', curr_ap.round(2))

        plot_pr_curve(stats_result['prec'], stats_result['rec'], model_name, class_names[idx])

    csv_path = os.path.join(logdir + '/final_output', PKL_FILE + '_compute_ap.csv')

    # data to be written row-wise in csv fil
    data = [
        ['Car AP', 'Ped AP', 'Cyc AP', 'mAP', 'TP', 'FP_ML', 'FP_BG'],
        output_csv_row
    ]

    # opening the csv file in 'a+' mode
    file = open(csv_path, 'a+', newline ='')
    
    # writing the data into the file
    with file:    
        write = csv.writer(file)
        write.writerows(data)

main()
