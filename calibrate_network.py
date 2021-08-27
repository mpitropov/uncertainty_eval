from math import e
import os, sys, copy
from pprint import pprint

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from tqdm import tqdm

# Dataset evaluation
from detection_eval.detection_eval import DetectionEval
from detection_eval.filter import build_kitti_filters, build_cadc_filters
from detection_eval.box_list import combine_box_lists

# Cluster predictions
from cluster import cluster_preds

from scipy.special import entr

# Scoring Rules
from scoring.nll_clf import NLLCLF, NLLCLF_Calibration
from scoring.nll_reg import NLLREG, NLLREG_Calibration
from scoring.binary_brier_score import BINARYBRIERSCORE
from scoring.brier_score import BRIERSCORE
from scoring.dmm import DMM
from scoring.energy_score import ENERGYSCORE

# Calibration Error
# from ece import calculate_ece, plot_reliability, calculate_ece_reg
# import calibration as cal

dataset_types = ['KITTI', 'CADC', 'NuScenes']
selected_dataset = dataset_types[0]
dataset_path = None
if selected_dataset == dataset_types[0]:
    dataset_path = '/root/kitti'
    logdir = '/root/logdir/output_pkls'
    gts_path = os.path.join(logdir, 'kitti_infos_val.pkl')
elif selected_dataset == dataset_types[1]:
    dataset_path = '/root/cadc'
    logdir = '/root/logdir/output_pkls'
    gts_path = os.path.join(logdir, 'cadc_infos_train.pkl')

# Load arguments for clustering
ENSEMBLE_TYPE = int(sys.argv[1])
VOTING_STATEGY = int(sys.argv[2])

# Create path to the pickle file
PKL_FILE = sys.argv[3]
preds_path = os.path.join(logdir, PKL_FILE)

if ENSEMBLE_TYPE == -1: # original model
    print('Ensemble type: Sigmoid model')
    MIN_CLUSTER_SIZE = 1
elif ENSEMBLE_TYPE == 0: # mc-dropout
    print('Ensemble type: mc-dropout')
    MIN_CLUSTER_SIZE = 4
elif ENSEMBLE_TYPE == 1: # ensemble
    print('Ensemble type: ensemble')
    MIN_CLUSTER_SIZE = 4
elif ENSEMBLE_TYPE == 2: # mimo
    print('Ensemble type: mimo')
    MIN_CLUSTER_SIZE = 2
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

def load_dicts():
    # Load gt and prediction data dict
    with open(gts_path, 'rb') as f:
        gt_dicts = pickle.load(f)
    with open(preds_path, 'rb') as f:
        pred_dicts = pickle.load(f)
    return gt_dicts, pred_dicts

def pcdet_get_labels(data_dict):
    if isinstance(data_dict, list):
        data_dict = data_dict[0]
    for label_key in ['labels', 'gt_labels', 'pred_labels']:
        if label_key in data_dict:
            return data_dict[label_key]
    if 'annos' in data_dict:
        data_dict = data_dict['annos']
    if 'name' in data_dict:
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
        labels = []
        for name in data_dict['name']:
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

def pcdet_get_boxes(data_dict, gt_mode=True):
    if isinstance(data_dict, list):
        data_dict = data_dict[0]
    if 'annos' in data_dict:
        data_dict = data_dict['annos']
    for box_key in ['gt_boxes', 'gt_boxes_lidar', 'boxes_lidar', 'box3d_lidar']:
        if box_key in data_dict:
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
        pcdet_get_boxes(data_dict, gt_mode=False)
    )

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
    gt_dicts, pred_dicts = load_dicts()
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
    pred_dicts = cluster_preds(pred_dicts, MIN_CLUSTER_SIZE)

    # Threshold (list or dict) maps a label to a matching threshold
    if selected_dataset == dataset_types[0]:
        thresholds = {
            -1: 1,     # Just to not output error
            0: 1,      # Just to not output error
            1: 0.7,    # Car
            2: 0.5,    # Pedestrian
            3: 0.5     # Cyclist
        }
        
        kitti_filter_list = build_kitti_filters(gts_path)
        filter_list = [
            kitti_filter_list[1], # Car moderate
            kitti_filter_list[4], # Ped moderate
            kitti_filter_list[7]  # Cyc moderate
        ]

        class_names = ['Car', 'Pedestrian', 'Cyclist', 'All']
        NUM_CLASSES = 3
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

    gt_list_list = []
    pred_list_list = []

    for filter_idx in range(len(filter_list)):
        print("Performing evaluation for class " + filter_list[filter_idx].class_name + \
            " and difficulty " + str(filter_list[filter_idx].difficulty))

        # Evaluate over validation set
        print("Loop through and evaluate all samples...")
        gt_list, pred_list = DetectionEval.evaluate_all_samples(
            gt_dicts, pred_dicts,
            thresholds=thresholds,
            criterion='iou_3d',
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

    ap_list = []

    for idx in range(len(class_names)):
        print("Calibrating network for " + class_names[idx] + "...")
        # Get current gt and pred list
        gt_list = gt_list_list[idx]
        pred_list = pred_list_list[idx]

        # A prediction box is either a TP or FP
        # TP is valid, localized and classified
        tp = pred_list.valid & pred_list.localized & pred_list.classified
        # FP is valid and either not localized or not classified correctly
        fp = pred_list.valid & ~(pred_list.localized & pred_list.classified)
        # Correctly localized, misclassified
        l_mc = pred_list.valid & (pred_list.localized) & (~pred_list.classified)
        # Mislocalized, correctly classified
        ml_c = pred_list.valid & (~pred_list.localized) & (pred_list.classified)
        # Mislocalized, misclassified
        ml_mc = pred_list.valid & (~pred_list.localized) & (~pred_list.classified) & (~pred_list.bg)
        # Completely missed
        bg = pred_list.valid & (pred_list.bg)

        # Init Scoring Rules
        # nll_clf_obj = NLLCLF()
        nll_clf_calib_obj = NLLCLF_Calibration()
        # nll_reg_obj = NLLREG()
        nll_reg_calib_obj = NLLREG_Calibration()
        # binary_brier_obj = BINARYBRIERSCORE()
        # brier_obj = BRIERSCORE()
        # dmm_obj = DMM()
        # energy_obj = ENERGYSCORE()

        # Calibration library lists
        preds_score_all = []
        gt_score_all = []

        print('Number of predictions', len(pred_list))
        print('Number of TPs', len(pred_list[tp]))
        print('Number of FPs', len(pred_list[fp]))
        print('Number of Correctly localized, misclassified', len(pred_list[l_mc]))
        print('Number of Mislocalized, correctly classified', len(pred_list[ml_c]))
        print('Number of Mislocalized, misclassified', len(pred_list[ml_mc]))
        print('Number of Background detections', len(pred_list[bg]))
        stats_result = DetectionEval.compute_statistics(gt_list=gt_list, \
                            pred_list=pred_list, n_positions=40)
        print('Number of FNs', stats_result['fn'])
        if idx == NUM_CLASSES: # Use mean of APs
            print('mAP@40', np.mean(ap_list).round(2))
        else:
            curr_ap = (100 * stats_result['ap'])
            ap_list.append(curr_ap)
            print('AP@40', curr_ap.round(2))

        print('Starting calibration')

        # Scoring Rules
        if ENSEMBLE_TYPE == -1: # original model
            print("We don't calibrate original model")
            # # TP
            # nll_clf_obj.add_tp(pred_list[tp])
            # binary_brier_obj.add_tp(pred_list[tp])

            # # FP
            # nll_clf_obj.add_fp(pred_list[fp])
            # binary_brier_obj.add_fp(pred_list[fp])
        else:
            # # TP
            # nll_clf_obj.add_tp(pred_list[tp])
            # binary_brier_obj.add_tp(pred_list[tp])
            # brier_obj.add_tp(pred_list[tp])
            # dmm_obj.add_tp(gt_list, pred_list[tp])
            # nll_reg_obj.add_tp(gt_list, pred_list[tp])
            # energy_obj.add_tp(gt_list, pred_list[tp])

            # # FP 
            # nll_clf_obj.add_bg_tp(pred_list[fp])
            # binary_brier_obj.add_bg_tp(pred_list[fp])
            # brier_obj.add_fp(pred_list[fp])
            # energy_obj.add_fp(pred_list[fp])

            # nll_clf_calib_obj.calibrate(pred_list[tp], pred_list[fp])
            # nll_clf_calib_obj.calibrate_isotonic_reg(pred_list[tp], pred_list[fp])
            nll_clf_calib_obj.calibrate_multiclass_isotonic_reg(pred_list[tp], pred_list[fp])
            nll_reg_calib_obj.calibrate(gt_list, pred_list[tp])

        # print('NLL Classification mean', nll_clf_obj.mean().round(4))
        # print('NLL Classification mean TP', nll_clf_obj.mean_tp().round(4))
        # print('NLL Classification mean FP', nll_clf_obj.mean_fp().round(4))
        # print('Binary Brier Score Classification mean', binary_brier_obj.mean().round(4))
        # print('Binary Brier Score Classification mean TP', binary_brier_obj.mean_tp().round(4))
        # print('Binary Brier Score Classification mean FP', binary_brier_obj.mean_fp().round(4))
        # if ENSEMBLE_TYPE != -1:
        #     print('Brier Score Classification mean', brier_obj.mean().round(4))
        #     print('Brier Score Classification mean TP', brier_obj.mean_tp().round(4))
        #     print('Brier Score Classification mean FP', brier_obj.mean_fp().round(4))
        #     print('NLL Regression mean', nll_reg_obj.mean().round(4))
        #     print('DMM Regression mean', dmm_obj.mean().round(4))
        #     print('Energy Regression mean', energy_obj.mean().round(4))

main()
