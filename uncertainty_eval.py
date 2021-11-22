from math import e
import os, sys, copy, gc, yaml, csv
from pprint import pprint

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from tqdm import tqdm
from prettytable import PrettyTable

# Dataset evaluation
from detection_eval.detection_eval import DetectionEval
from detection_eval.dataset_filters import build_kitti_filters, build_cadc_filters, build_nuscenes_filters
from detection_eval.box_list import combine_box_lists

# Cluster predictions
from cluster import cluster_preds

from utils import load_dicts, create_thresholds, add_num_pts, filter_pred_dicts, plot_conf_pts, plot_pr_curve

from scipy.special import entr

# Scoring Rules
from scoring.nll_clf import NLLCLF
from scoring.nll_reg import NLLREG
from scoring.binary_brier_score import BINARYBRIERSCORE
from scoring.brier_score import BRIERSCORE
from scoring.dmm import DMM
from scoring.energy_score import ENERGYSCORE

# Uncertainty Metrics
from scoring.uncertainty_metrics import ShannonEntropy, AleatoricEntropy, MutualInfo, EpistemicTotalVar, AleatoricTotalVar
# from scoring.uncertainty_metrics import UncertaintyMetrics

# Calibration Error
from ece import calculate_ece, plot_reliability_clf, calculate_ece_reg, plot_reliability_reg
import calibration as cal

dataset_types = ['KITTI', 'CADC', 'NuScenes']
selected_dataset = sys.argv[1] # dataset_types[1]
dataset_path = None
if selected_dataset == dataset_types[0]:
    dataset_path = '/root/kitti'
    logdir = '/root/logdir/output_pkls/kitti'
    gts_path = os.path.join(logdir, 'kitti_infos_val.pkl') # 'gts.pkl' 'kitti_infos_val.pkl' 'kitti_infos_train.pkl'
    # gts_path = os.path.join(logdir, 'kitti_infos_train.pkl') # 'gts.pkl' 'kitti_infos_val.pkl' 'kitti_infos_train.pkl'
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

yaml_path = os.path.join(logdir + '/tmp_output', PKL_FILE + '_step_three.yaml')
# Read YAML file
with open(yaml_path, 'r') as stream:
    SCORE_THRESHOLDS = yaml.safe_load(stream)

yaml_path = os.path.join(logdir + '/tmp_output', PKL_FILE + '_step_two.yaml')
# Read YAML file
with open(yaml_path, 'r') as stream:
    CALIBRATION_VALS = yaml.safe_load(stream)

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

def pcdet_get_boxes(data_dict, gt_mode=True):
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
        pcdet_get_boxes(data_dict, gt_mode=False)
    )

def attach_data(sample_idx, gt_dict, pred_dict, gt_list, pred_list):
    for i in range(len(gt_list)):
        gt_list.data[i] = dict(
            gt_boxes=pcdet_get_boxes(gt_dict)[i], occluded=pcdet_get_occlusion(gt_dict)[i])
        # gt_list.data[i] = dict(gt_boxes=pcdet_get_boxes(gt_dict)[i])
    if 'score_all' not in pred_dict: # original model
        for i in range(len(pred_list)):
            pred_list.data[i] = dict(
                boxes_lidar=pred_dict['boxes_lidar'][i],
                num_pts_in_pred=pred_dict['num_pts_in_pred'][i]
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
                num_pts_in_pred=pred_dict['num_pts_in_pred'][i],
                dist=np.linalg.norm(pred_dict['boxes_lidar'][i][:2])
            )

def get_uncertainty_by_thres(pred_list_tp_ml_c):
    loc_score_thresholds = np.arange(0.0, 1.1, 0.1).round(2) # [0, 0.1, ..., 0.8, 1.0]
    loc_scores = np.array([obj.loc_score for obj in pred_list_tp_ml_c])
    tmp_se = np.array([obj.data['shannon_entropy'] for obj in pred_list_tp_ml_c])
    tmp_ae = np.array([obj.data['aleatoric_entropy'] for obj in pred_list_tp_ml_c])
    tmp_mi = np.array([obj.data['mutual_info'] for obj in pred_list_tp_ml_c])
    tmp_etv = np.array([obj.data['epistemic_total_var'] for obj in pred_list_tp_ml_c])
    tmp_atv = np.array([obj.data['aleatoric_total_var'] for obj in pred_list_tp_ml_c])
    csv_um_rows = [[],[],[],[],[],]
    cv_num_row = []
    for i in range(len(loc_score_thresholds) - 1):
        valid_loc_scores = (loc_scores >= loc_score_thresholds[i]) & (loc_scores < loc_score_thresholds[i+1])
        csv_um_rows[0].append(np.mean(tmp_se[valid_loc_scores]))
        csv_um_rows[1].append(np.mean(tmp_ae[valid_loc_scores]))
        csv_um_rows[2].append(np.mean(tmp_mi[valid_loc_scores]))
        csv_um_rows[3].append(np.mean(tmp_etv[valid_loc_scores]))
        csv_um_rows[4].append(np.mean(tmp_atv[valid_loc_scores]))
        cv_num_row.append(len(tmp_se[valid_loc_scores]))

    print('OUTPUT CAR')
    print(csv_um_rows)
    print(cv_num_row)
    exit()

def main():
    print("Load dictionaries...")
    gt_dicts, pred_dicts = load_dicts(gts_path, preds_path)

    print("Using SECOND half of val set as the eval set.")
    half_point = int(len(gt_dicts)/2)
    gt_dicts = gt_dicts[half_point:]
    pred_dicts = pred_dicts[half_point:]

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

            print("Clustering predictions by frame...")
            t_vals = (CALIBRATION_VALS[class_names[filter_idx]]['cls'][iou_threshold_idx], \
                CALIBRATION_VALS[class_names[filter_idx]]['reg'][iou_threshold_idx])
            clustered_pred_dicts = cluster_preds(pred_dicts, selected_dataset, 'temp_scaled', MIN_CLUSTER_SIZE, t_vals)

            curr_score_threshold = SCORE_THRESHOLDS[class_names[filter_idx]][iou_threshold_idx]
            print("Filtering predictions with score threhold:", curr_score_threshold)
            filtered_pred_dicts = filter_pred_dicts(clustered_pred_dicts, curr_score_threshold)

            # Add predictions to the points
            add_num_pts(selected_dataset, dataset_path, filtered_pred_dicts)

            gt_list, pred_list = DetectionEval.evaluate_all_samples(
                gt_dicts, filtered_pred_dicts,
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

            # tmp for testing
            # tp = pred_list.valid & pred_list.localized & pred_list.classified & ~pred_list.duplicated
            # ml_c = pred_list.valid & (~pred_list.localized) & (pred_list.classified)
            # get_uncertainty_by_thres(pred_list[tp | ml_c])
            # exit()

    # Data to store for each class
    nll_clf_list = []
    brier_list = []
    nll_reg_list = []
    energy_list = []
    mce = []
    reg_ce = []
    reg_maxce = []
    se_list = []
    ae_list = []
    mi_list = []
    etv_list = []
    atv_list = []

    individual_metrics = {
        'SE': [], 'AE': [], 'MI': [], 'ETV': [], 'ATV': []
    }

    uncertainty_metrics = {
        'tp_car': copy.deepcopy(individual_metrics),
        'tp_ped_cyc': copy.deepcopy(individual_metrics)
    }

    tp_car_uncertainty = {
        'SE': [], 'AE': [], 'MI': [], 'ETV': [], 'ATV': [],
        'dist': [], # Display Aleatoric Uncertainty by using distances
        'occluded': [] # Display Epistemic Uncertainty by using occlusion
    }

    # Clear memory, mostly needed for large datasets like NuScenes
    del gt_dicts
    del pred_dicts
    gc.collect()

    for idx in range(len(class_names)):
        # Store results for this class and iou threshold
        tmp_nll_clf_list = []
        tmp_brier_list = []
        tmp_nll_reg_list = []
        tmp_energy_list = []
        tmp_mce = []
        tmp_reg_ce = []
        tmp_reg_maxce = []
        tmp_se_list = []
        tmp_ae_list = []
        tmp_mi_list = []
        tmp_etv_list = []
        tmp_atv_list = []

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

            # Init Scoring Rules
            nll_clf_obj = NLLCLF()
            nll_reg_obj = NLLREG()
            binary_brier_obj = BINARYBRIERSCORE()
            brier_obj = BRIERSCORE()
            dmm_obj = DMM()
            energy_obj = ENERGYSCORE()

            # Init Uncertainty Metrics
            se_obj = ShannonEntropy()
            ae_obj = AleatoricEntropy()
            mi_obj = MutualInfo()
            etv_obj = EpistemicTotalVar()
            atv_obj = AleatoricTotalVar()

            # Our own ECE Implementation
            num_bins = 10
            bins = np.arange(0, 1.0+(1.0/num_bins), 1.0/num_bins)
            conf_mat = [{'TP': 0, 'FP': 0, 'confidence_list': []} for i in range(len(bins))]

            # Calibration library lists
            preds_score_all = []
            gt_score_all = []

            print('Number of predictions', len(pred_list))
            print('Number of TPs', len(pred_list[tp]))
            print('Number of Duplicates', len(pred_list[dup]))
            print('Number of Localization Errors', len(pred_list[loc_err]))
            print('  LE Type 1: Number of Correctly localized, misclassified', len(pred_list[l_mc]))
            print('  LE Type 2: Number of Mislocalized, correctly classified', len(pred_list[ml_c]))
            print('  LE Type 3: Number of Mislocalized, misclassified', len(pred_list[ml_mc]))
            print('Number of FPs', len(pred_list[fp]))

            stats_result = DetectionEval.compute_statistics(gt_list=gt_list, \
                                pred_list=pred_list, n_positions=num_recall_positions)
            print('Number of FNs', stats_result['fn'])

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

            curr_ap = (100 * stats_result['ap'])
            print('AP@40', curr_ap.round(2))

            # Scoring Rules

            if ENSEMBLE_TYPE == -1: # original model
                # TP
                nll_clf_obj.add_tp(pred_list[tp])
                binary_brier_obj.add_tp(pred_list[tp])

                # Loc Errors
                # nll_clf_obj.add_loc_err(pred_list[loc_err])
                # binary_brier_obj.add_loc_err(pred_list[loc_err])

                # FP
                nll_clf_obj.add_fp(pred_list[fp])
                binary_brier_obj.add_fp(pred_list[fp])
            else:
                # TP
                nll_clf_obj.add_tp(pred_list[tp])
                binary_brier_obj.add_tp(pred_list[tp])
                brier_obj.add_tp(pred_list[tp])
                dmm_obj.add_tp(gt_list, pred_list[tp])
                nll_reg_obj.add_tp(gt_list, pred_list[tp])
                energy_obj.add_tp(gt_list, pred_list[tp])
                se_obj.add_tp(pred_list[tp])
                ae_obj.add_tp(pred_list[tp])
                mi_obj.add_tp(pred_list[tp])
                etv_obj.add_tp(pred_list[tp])
                atv_obj.add_tp(pred_list[tp])

                # Duplicates
                nll_clf_obj.add_dup(pred_list[dup])
                binary_brier_obj.add_dup(pred_list[dup])
                brier_obj.add_dup(pred_list[dup])
                dmm_obj.add_dup(gt_list, pred_list[dup])
                nll_reg_obj.add_dup(gt_list, pred_list[dup])
                energy_obj.add_dup(gt_list, pred_list[dup])
                se_obj.add_dup(pred_list[dup])
                ae_obj.add_dup(pred_list[dup])
                mi_obj.add_dup(pred_list[dup])
                etv_obj.add_dup(pred_list[dup])
                atv_obj.add_dup(pred_list[dup])

                # Loc Errors
                nll_clf_obj.add_loc_err(pred_list[ml_c])
                binary_brier_obj.add_loc_err(pred_list[ml_c])
                brier_obj.add_loc_err(pred_list[ml_c])
                dmm_obj.add_loc_err(gt_list, pred_list[ml_c])
                nll_reg_obj.add_loc_err(gt_list, pred_list[ml_c])
                energy_obj.add_loc_err(gt_list, pred_list[ml_c])
                se_obj.add_loc_err(pred_list[ml_c])
                ae_obj.add_loc_err(pred_list[ml_c])
                mi_obj.add_loc_err(pred_list[ml_c])
                etv_obj.add_loc_err(pred_list[ml_c])
                atv_obj.add_loc_err(pred_list[ml_c])

                # FP 
                nll_clf_obj.add_bg_tp(pred_list[fp])
                binary_brier_obj.add_bg_tp(pred_list[fp])
                brier_obj.add_fp(pred_list[fp])
                energy_obj.add_fp(pred_list[fp])
                se_obj.add_fp(pred_list[fp])
                ae_obj.add_fp(pred_list[fp])
                mi_obj.add_fp(pred_list[fp])
                etv_obj.add_fp(pred_list[fp])
                atv_obj.add_fp(pred_list[fp])

                # FN
                brier_obj.add_fn(gt_list[fn_bg])

            # Only show this for sigmoid models
            if ENSEMBLE_TYPE == -1:
                print('Binary Brier Score Classification mean', binary_brier_obj.mean().round(4))
                print('Binary Brier Score Classification mean TP', binary_brier_obj.mean_tp().round(4))
                print('Binary Brier Score Classification mean FP', binary_brier_obj.mean_fp().round(4))

            tmp_nll_clf_list.append([nll_clf_obj.mean_tp().round(4), \
                                nll_clf_obj.mean_loc_err().round(4), \
                                nll_clf_obj.mean_fp().round(4)])
            # print('NLL Classification mean var', nll_clf_obj.mean().round(4), nll_clf_obj.var().round(4))
            # print('NLL Classification mean: TP DUP FP_ML FP', nll_clf_obj.mean_tp().round(4), \
            #     nll_clf_obj.mean_dup().round(4), nll_clf_obj.mean_loc_err().round(4), nll_clf_obj.mean_fp().round(4))
            # print('NLL Classification var: TP FP_ML FP', np.var(nll_clf_obj.tp_value_list).round(4), \
            #     np.var(nll_clf_obj.loc_err_value_list).round(4), np.var(nll_clf_obj.fp_value_list).round(4))

            # tp_scores = []
            # tp_loc_scores = []
            # tp_se = []
            # tp_ae = []
            # for obj in pred_list[tp]:
            #     tp_scores.append(obj.pred_score)
            #     tp_loc_scores.append(obj.loc_score)
            #     tp_se.append(obj.data['shannon_entropy'])
            #     tp_ae.append(obj.data['aleatoric_entropy'])
            # print('TP: score', np.mean(tp_scores).round(4))
            # print('TP: loc score', np.mean(tp_loc_scores).round(4))
            # print('TP: SE AE', np.mean(tp_se).round(4), np.mean(tp_ae).round(4))

            loc_err_scores = [obj.loc_score for obj in pred_list[loc_err]]
            print('LOC ERR: loc score', np.mean(loc_err_scores).round(4), np.var(loc_err_scores).round(4))

            plt.hist(loc_err_scores, bins=50, cumulative=True)
            plt.title('Localization error localization scores')
            plt.savefig('images/loc_err_loc_scores_cum_hist.png')
            plt.close()

            if ENSEMBLE_TYPE != -1:
                tmp_brier_list.append([brier_obj.mean_tp().round(4), \
                                    brier_obj.mean_loc_err().round(4), brier_obj.mean_fp().round(4)])
                tmp_nll_reg_list.append([nll_reg_obj.mean_tp().round(4), nll_reg_obj.mean_loc_err().round(4), \
                    nll_reg_obj.mean_fp().round(4)])
                tmp_energy_list.append([energy_obj.mean_tp().round(4), energy_obj.mean_loc_err().round(4), \
                    energy_obj.mean_fp().round(4)])
                tmp_se_list.append([se_obj.mean_tp().round(4), se_obj.mean_loc_err().round(4), \
                    se_obj.mean_fp().round(4)])
                tmp_ae_list.append([ae_obj.mean_tp().round(4), ae_obj.mean_loc_err().round(4), \
                    ae_obj.mean_fp().round(4)])
                tmp_mi_list.append([mi_obj.mean_tp().round(4), mi_obj.mean_loc_err().round(4), \
                    mi_obj.mean_fp().round(4)])
                tmp_etv_list.append([etv_obj.mean_tp().round(4), etv_obj.mean_loc_err().round(4), \
                    etv_obj.mean_fp().round(4)])
                tmp_atv_list.append([atv_obj.mean_tp().round(4), atv_obj.mean_loc_err().round(4), \
                    atv_obj.mean_fp().round(4)])

                # print('Brier Score Classification mean var', brier_obj.mean().round(4), brier_obj.var().round(4))
                # print('Brier Score Classification mean: TP DUP FP_ML FP FN', brier_obj.mean_tp().round(4), \
                #     brier_obj.mean_dup().round(4), brier_obj.mean_loc_err().round(4), brier_obj.mean_fp().round(4), brier_obj.mean_fn().round(4))
                # print('Brier Score Classification var: TP DUP FP_ML FP FN', np.var(brier_obj.tp_value_list).round(4), \
                #     np.var(brier_obj.dup_value_list).round(4), np.var(brier_obj.loc_err_value_list).round(4), np.var(brier_obj.fp_value_list).round(4), np.var(brier_obj.mean_fn()).round(4))

                # print('NLL Regression mean', nll_reg_obj.mean().round(4))
                # print('NLL Regression mean: TP DUP FP_ML', nll_reg_obj.mean_tp().round(4), nll_reg_obj.mean_dup().round(4), \
                #     nll_reg_obj.mean_loc_err().round(4))
                # print('NLL Regression var: TP DUP FP_ML', np.var(nll_reg_obj.tp_value_list).round(4), np.var(nll_reg_obj.dup_value_list).round(4), \
                #     np.var(nll_reg_obj.loc_err_value_list).round(4))

                # print('DMM Regression  mean', dmm_obj.mean().round(4))
                # print('DMM Regression mean: TP DUP FP_ML', dmm_obj.mean_tp().round(4), dmm_obj.mean_dup().round(4), \
                #     dmm_obj.mean_loc_err().round(4))
                # print('DMM Regression var: TP DUP FP_ML', np.var(dmm_obj.tp_value_list).round(4), np.var(dmm_obj.mean_dup()).round(4), \
                #     np.var(dmm_obj.loc_err_value_list).round(4))

                # print('Energy Regression mean', energy_obj.mean().round(4))
                # print('Energy Regression mean: TP DUP FP_ML', energy_obj.mean_tp().round(4), energy_obj.mean_dup().round(4), \
                #     energy_obj.mean_loc_err().round(4), energy_obj.mean_fp().round(4))
                # print('Energy Regression var: TP DUP FP_ML', np.var(energy_obj.tp_value_list).round(4), np.var(energy_obj.dup_value_list).round(4), \
                #     np.var(energy_obj.loc_err_value_list).round(4), np.var(energy_obj.fp_value_list).round(4))

            # Classification Calibration Error
            # TP and duplicates are grouped
            # FP and mislocalized but correctly classified are grouped
            if ENSEMBLE_TYPE == -1: # original model
                # TP and duplicates
                for obj in pred_list[tp | dup]:
                    bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
                    conf_mat[bin_num]['TP'] += 1
                    conf_mat[bin_num]['confidence_list'].append(obj.pred_score)
                    # Calibration library
                    preds_score_all.append(obj.pred_score)
                    # Add GT as 1
                    gt_score_all.append(1)

                # FP and localization errors
                for obj in pred_list[fp | loc_err]:
                    bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
                    conf_mat[bin_num]['FP'] += 1
                    conf_mat[bin_num]['confidence_list'].append(obj.pred_score)
                    # Calibration library
                    preds_score_all.append(obj.pred_score)
                    # Add GT as 0
                    gt_score_all.append(0)
            else:
                # TP and duplicates
                for obj in pred_list[tp | dup]:
                    bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
                    conf_mat[bin_num]['TP'] += 1
                    conf_mat[bin_num]['confidence_list'].append(obj.pred_score)
                    # Calibration library
                    preds_score_all.append(obj.data['score_all'])
                    # add class to GT
                    gt_label = obj.gt_label
                    gt_score_all.append(int(gt_label - 1))

                # FP and localization errors
                for obj in pred_list[fp | loc_err]:
                    bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
                    conf_mat[bin_num]['FP'] += 1
                    conf_mat[bin_num]['confidence_list'].append(obj.pred_score)
                    # Calibration library
                    preds_score_all.append(obj.data['score_all'])
                    # Append BG class
                    gt_score_all.append(NUM_CLASSES)

            accuracy, confidence, ece, max_ce, weight_per_bin, key = calculate_ece(conf_mat, num_bins, class_names[idx])
            # print("Calculated ECE", ece)
            # print("Calculated MaxCE", max_ce)
            # print("Calculated Accuracy bins", accuracy)
            # print("Calculated conf_mat", conf_mat)

            plot_reliability_clf(key, accuracy, confidence, ece, max_ce, weight_per_bin, save_path='calibration_images/ECE_CLF_'+key+'.png')

            # plot_conf_pts(pred_list[tp | dup], pred_list[fp], model_name, class_names[idx])

            # # ECE and ECE stable binning (seems the same)
            # cls_expected_calibration_error = cal.get_ece(
            #     preds_score_all, gt_score_all)
            # print("Classification Expected Calibration Error", cls_expected_calibration_error.round(4))
            # cls_expected_calibration_error = cal.get_top_calibration_error(
            #     preds_score_all, gt_score_all, p=1)
            # print("Classification Expected Calibration Error 'stable binning'", cls_expected_calibration_error.round(4))
            # Marginal CE
            try:
                cls_marginal_calibration_error = cal.get_calibration_error(
                    preds_score_all, gt_score_all)
                tmp_mce.append(cls_marginal_calibration_error.round(4))
            except:
                print('ERROR: in classification error library')
                tmp_mce.append(np.nan)

            # print("Classification Marginal Calibration Error", cls_marginal_calibration_error.round(4))

            if ENSEMBLE_TYPE != -1:
                # Group TP, Duplicates and localization errors together
                reg_maximum_calibration_error, reg_calibration_error, acc_list = \
                    calculate_ece_reg(gt_list, pred_list[tp | dup | loc_err])
                tmp_reg_ce.append(reg_calibration_error.round(4))
                tmp_reg_maxce.append(reg_maximum_calibration_error.round(4))
                # print("Regression Calibration Error", reg_calibration_error.round(4))
                # print("Regression Maximum Calibration Error", reg_maximum_calibration_error.round(4))
                plot_reliability_reg(key, acc_list, reg_calibration_error, \
                                    save_path='calibration_images/ECE_REG_'+key+'.png')

            # Uncertainty metrics (per class)
            MIN_POINTS = 50
            tp_pred_list = pred_list[tp]
            pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in tp_pred_list]) > MIN_POINTS
            tmp_se = np.array([obj.data['shannon_entropy'] for obj in tp_pred_list[pt_filter]])
            tmp_ae = np.array([obj.data['aleatoric_entropy'] for obj in tp_pred_list[pt_filter]])
            tmp_mi = np.array([obj.data['mutual_info'] for obj in tp_pred_list[pt_filter]])
            tmp_etv = np.array([obj.data['epistemic_total_var'] for obj in tp_pred_list[pt_filter]])
            tmp_atv = np.array([obj.data['aleatoric_total_var'] for obj in tp_pred_list[pt_filter]])
            dist_list = np.array([obj.data['dist'] for obj in tp_pred_list[pt_filter]])
            occ_list = np.array([gt_list[int(obj.matched_idx)].data['occluded'] for obj in tp_pred_list[pt_filter]])
            if idx == 0:
                tp_type = 'tp_car'

                tmp_dict = {
                    'SE': tmp_se,
                    'AE': tmp_ae,
                    'MI': tmp_mi,
                    'ETV': tmp_etv,
                    'ATV': tmp_atv
                }
                uncertainty_text = ['SE', 'AE', 'MI', 'ETV', 'ATV']
                for um in uncertainty_text:
                    dist_lvl_1 = 10
                    dist_lvl_2 = 20
                    dist_lvl_3 = 30
                    dist_close = tmp_dict[um][(dist_list <= dist_lvl_1) & (occ_list == 0)]
                    dist_med = tmp_dict[um][(dist_list > dist_lvl_1) & (dist_list <= dist_lvl_2) & (occ_list == 0)]
                    dist_far = tmp_dict[um][(dist_list > dist_lvl_2) & (dist_list <= dist_lvl_3) & (occ_list == 0)]
                    occ_low = tmp_dict[um][(occ_list == 0) & (dist_list <= dist_lvl_1)]
                    occ_med = tmp_dict[um][(occ_list == 1) & (dist_list <= dist_lvl_1)]
                    occ_high = tmp_dict[um][(occ_list == 2) & (dist_list <= dist_lvl_1)]
                    print("Distance amounts", len(dist_close), len(dist_med), len(dist_far))
                    print("occlusion amounts", len(occ_low), len(occ_med), len(occ_high))
                    tp_car_uncertainty[um].append([np.nanmean(dist_close).round(4), np.nanmean(dist_med).round(4), np.nanmean(dist_far).round(4), \
                        np.nanmean(occ_low).round(4), np.nanmean(occ_med).round(4), np.nanmean(occ_high).round(4)])

            elif idx == 1 or idx == 2:
                tp_type = 'tp_ped_cyc'
            uncertainty_metrics[tp_type]['SE'].append(np.nanmean(tmp_se))
            uncertainty_metrics[tp_type]['AE'].append(np.nanmean(tmp_ae))
            uncertainty_metrics[tp_type]['MI'].append(np.nanmean(tmp_mi))
            uncertainty_metrics[tp_type]['ETV'].append(np.nanmean(tmp_etv))
            uncertainty_metrics[tp_type]['ATV'].append(np.nanmean(tmp_atv))
            # print(uncertainty_metrics[tp_type])
            # print(tp_pred_list[pt_filter])
            # print(tp_pred_list[pt_filter].data)
            # exit()

            # ignored_pred_list = pred_list[ignored]
            # pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in ignored_pred_list]) > MIN_POINTS
            # fp_type = 'fp_ignore_minus_van'
            # pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in ignored_pred_list]) > MIN_POINTS
            # se_list = np.array([obj.data['shannon_entropy'] for obj in ignored_pred_list[pt_filter]])
            # ae_list = np.array([obj.data['aleatoric_entropy'] for obj in ignored_pred_list[pt_filter]])
            # mi_list = np.array([obj.data['mutual_info'] for obj in ignored_pred_list[pt_filter]])
            # etv_list = np.array([obj.data['epistemic_total_var'] for obj in ignored_pred_list[pt_filter]])
            # atv_list = np.array([obj.data['aleatoric_total_var'] for obj in ignored_pred_list[pt_filter]])
            # uncertainty_metrics[fp_type]['SE'] = np.concatenate((uncertainty_metrics[tp_type]['SE'], se_list))
            # uncertainty_metrics[fp_type]['AE'] = np.concatenate((uncertainty_metrics[tp_type]['AE'], ae_list))
            # uncertainty_metrics[fp_type]['MI'] = np.concatenate((uncertainty_metrics[tp_type]['MI'], mi_list))
            # uncertainty_metrics[fp_type]['ETV'] = np.concatenate((uncertainty_metrics[tp_type]['ETV'], etv_list))
            # uncertainty_metrics[fp_type]['ATV'] = np.concatenate((uncertainty_metrics[tp_type]['ATV'], atv_list))

            # Uncertainty by threshold
            # if class_names[idx] == 'Car' and iou_thresholds[iou_threshold_idx] == 0.5:
            #     get_uncertainty_by_thres(pred_list[tp | ml_c])
        
        # With all IoUs for this class completed we can add to the main lists
        tmp_nll_clf_list = np.array(tmp_nll_clf_list)
        tmp_brier_list = np.array(tmp_brier_list)
        tmp_nll_reg_list = np.array(tmp_nll_reg_list)
        tmp_energy_list = np.array(tmp_energy_list)
        tmp_mce = np.array(tmp_mce)
        tmp_reg_ce = np.array(tmp_reg_ce)
        tmp_reg_maxce = np.array(tmp_reg_maxce)
        tmp_se_list = np.array(tmp_se_list)
        tmp_ae_list = np.array(tmp_ae_list)
        tmp_mi_list = np.array(tmp_mi_list)
        tmp_etv_list = np.array(tmp_etv_list)
        tmp_atv_list = np.array(tmp_atv_list)

        print('class results')
        print(tmp_nll_clf_list)
        print(tmp_brier_list)
        print(tmp_nll_reg_list)
        print(tmp_energy_list)
        print(tmp_mce)
        print(tmp_reg_ce)
        print(tmp_reg_maxce)
        print(tmp_se_list)
        print(tmp_ae_list)
        print(tmp_mi_list)
        print(tmp_etv_list)
        print(tmp_atv_list)

        nll_clf_list.append(np.nanmean(tmp_nll_clf_list, axis=0))
        brier_list.append(np.nanmean(tmp_brier_list, axis=0))
        nll_reg_list.append(np.nanmean(tmp_nll_reg_list, axis=0))
        energy_list.append(np.nanmean(tmp_energy_list, axis=0))
        mce.append(np.nanmean(tmp_mce, axis=0))
        reg_ce.append(np.nanmean(tmp_reg_ce, axis=0))
        reg_maxce.append(np.nanmean(tmp_reg_maxce, axis=0))
        se_list.append(np.nanmean(tmp_se_list, axis=0))
        ae_list.append(np.nanmean(tmp_ae_list, axis=0))
        mi_list.append(np.nanmean(tmp_mi_list, axis=0))
        etv_list.append(np.nanmean(tmp_etv_list, axis=0))
        atv_list.append(np.nanmean(tmp_atv_list, axis=0))

    # Convert to numpy
    nll_clf_list = np.array(nll_clf_list)
    brier_list = np.array(brier_list)
    nll_reg_list = np.array(nll_reg_list)
    energy_list = np.array(energy_list)
    mce = np.array(mce)
    reg_ce = np.array(reg_ce)
    reg_maxce = np.array(reg_maxce)
    se_list = np.array(se_list)
    ae_list = np.array(ae_list)
    mi_list = np.array(mi_list)
    etv_list = np.array(etv_list)
    atv_list = np.array(atv_list)

    print('Final results')

    print('Raw results (row is class, columns are TP, FP_ML, FP)')
    print(nll_clf_list)
    print(brier_list)
    print(nll_reg_list)
    print(energy_list)
    print(mce)
    print(reg_ce)
    print(reg_maxce)
    print('se_list', se_list)
    print('ae_list', ae_list)
    print('mi_list', mi_list)
    print('etv_list', etv_list)
    print('atv_list', atv_list)

    table = PrettyTable()
    table.field_names = (['Output Type',
                          'Cls Ignorance Score',
                          'Cls Brier/Probability Score',
                          'Reg Ignorance Score',
                          'Reg Energy Score'])

    table.add_row([
        "True Positives:",
        '{:.4f}'.format(np.nanmean(nll_clf_list[:, 0])),
        '{:.4f}'.format(np.nanmean(brier_list[:, 0])),
        '{:.4f}'.format(np.nanmean(nll_reg_list[:, 0])),
        '{:.4f}'.format(np.nanmean(energy_list[:][0]))
    ])
    table.add_row([
        "Localization Errors:",
        '{:.4f}'.format(np.nanmean(nll_clf_list[:, 1])),
        '{:.4f}'.format(np.nanmean(brier_list[:, 1])),
        '{:.4f}'.format(np.nanmean(nll_reg_list[:, 1])),
        '{:.4f}'.format(np.nanmean(energy_list[:, 1]))
    ])
    table.add_row([
        "False Positives:",
        '{:.4f}'.format(np.nanmean(nll_clf_list[:, 2])),
        '{:.4f}'.format(np.nanmean(brier_list[:, 2])),
        '{:.4f}'.format(np.nanmean(nll_reg_list[:, 2])),
        '{:.4f}'.format(np.nanmean(energy_list[:, 2]))
    ])

    print(table)

    table = PrettyTable()
    table.field_names = (['Cls Marginal Calibration Error',
                          'Reg Expected Calibration Error',
                          'Reg Maximum Calibration Error'])

    table.add_row([
        '{:.4f}'.format(np.mean(mce[0:])),
        '{:.4f}'.format(np.mean(reg_ce[0:])),
        '{:.4f}'.format(np.mean(reg_maxce[0:]))
    ])

    print(table)

    csv_path = os.path.join(logdir + '/final_output', PKL_FILE + '_uncertainty_eval.csv')

    # data to be written row-wise in csv fil
    data = [
        ['CLS NLL TP', 'CLS NLL FP_ML', 'CLS NLL FP_BG', \
            'CLS Brier TP', 'CLS Brier FP_ML', 'CLS Brier FP_BG', \
            'REG NLL TP', 'REG NLL FP_ML', \
            'REG Energy TP', 'REG Energy FP_ML', \
            'MCE', 'REG ECE'
        ],
        # # Car
        # [round(nll_clf_list[0, 0], 4), round(nll_clf_list[0, 1], 4), round(nll_clf_list[0, 2], 4), \
        #     round(brier_list[0, 0], 4), round(brier_list[0, 1], 4), round(brier_list[0, 2], 4), \
        #     round(nll_reg_list[0, 0], 4), round(nll_reg_list[0, 1], 4), \
        #     round(energy_list[0, 0], 4), round(energy_list[0, 1], 4), \
        #     round(mce[0], 4), round(reg_ce[0], 4)
        # ],
        # ALL
        [np.nanmean(nll_clf_list[:, 0]).round(4), np.nanmean(nll_clf_list[:, 1]).round(4), np.nanmean(nll_clf_list[:, 2]).round(4), \
            np.nanmean(brier_list[:, 0]).round(4), np.nanmean(brier_list[:, 1]).round(4), np.nanmean(brier_list[:, 2]).round(4), \
            np.nanmean(nll_reg_list[:, 0]).round(4), np.nanmean(nll_reg_list[:, 1]).round(4), \
            np.nanmean(energy_list[:, 0]).round(4), np.nanmean(energy_list[:, 1]).round(4), \
            np.mean(mce).round(4), np.mean(reg_ce).round(4)
        ]
    ]

    # opening the csv file in 'a+' mode
    file = open(csv_path, 'a+', newline ='')
    
    # writing the data into the file
    with file:    
        write = csv.writer(file)
        write.writerows(data)

    csv_path = os.path.join(logdir + '/final_output', PKL_FILE + '_uncertainty_sum_count.csv')

    # data to be written row-wise in csv fil
    data = [
        ['SE TP', 'SE FP_ML', 'SE FP_BG', \
            'AE TP', 'AE FP_ML', 'AE FP_BG', \
            'MI TP', 'MI FP_ML', 'MI FP_BG', \
            'ETV TP', 'ETV FP_ML', 'ETV FP_BG', \
            'ATV TP', 'ATV FP_ML', 'ATV FP_BG'
        ],
        # # Car
        # [round(se_list[0, 0], 4), round(se_list[0, 1], 4), round(se_list[0, 2], 4), \
        #     round(ae_list[0, 0], 4), round(ae_list[0, 1], 4), round(ae_list[0, 2], 4), \
        #     round(mi_list[0, 0], 4), round(mi_list[0, 1], 4), round(mi_list[0, 2], 4), \
        #     round(etv_list[0, 0], 4), round(etv_list[0, 1], 4), round(etv_list[0, 2], 4), \
        #     round(atv_list[0, 0], 4), round(atv_list[0, 1], 4), round(atv_list[0, 2], 4)
        # ],
        # ALL
        [np.nanmean(se_list[:, 0]).round(4), np.nanmean(se_list[:, 1]).round(4), np.nanmean(se_list[:, 2]).round(4), \
            np.nanmean(ae_list[:, 0]).round(4), np.nanmean(ae_list[:, 1]).round(4), np.nanmean(ae_list[:, 2]).round(4), \
            np.nanmean(mi_list[:, 0]).round(4), np.nanmean(mi_list[:, 1]).round(4), np.nanmean(mi_list[:, 2]).round(4), \
            np.nanmean(etv_list[:, 0]).round(4), np.nanmean(etv_list[:, 1]).round(4), np.nanmean(etv_list[:, 2]).round(4), \
            np.nanmean(atv_list[:, 0]).round(4), np.nanmean(atv_list[:, 1]).round(4), np.nanmean(atv_list[:, 2]).round(4)
        ]
    ]

    # opening the csv file in 'a+' mode
    file = open(csv_path, 'a+', newline ='')
    
    # writing the data into the file
    with file:    
        write = csv.writer(file)
        write.writerows(data)

    temp = ['tp_car', 'tp_ped_cyc'] # , 'fp_ignore_minus_van'
    output_um_row = []
    for val in temp:
        # print(uncertainty_metrics[val]['SE'])
        # exit()
    
        se_val = np.nanmean(uncertainty_metrics[val]['SE']).round(4)
        ae_val = np.nanmean(uncertainty_metrics[val]['AE']).round(4)
        mi_val = np.nanmean(uncertainty_metrics[val]['MI']).round(4)
        etv_val = np.nanmean(uncertainty_metrics[val]['ETV']).round(4)
        atv_val = np.nanmean(uncertainty_metrics[val]['ATV']).round(4)
        print(val, ' SE', se_val)
        print(val, ' AE', ae_val)
        print(val, ' MI', mi_val)
        print(val, ' ETV', etv_val)
        print(val, ' ATV', atv_val)
        output_um_row.append([se_val, ae_val, mi_val, etv_val, atv_val])

    csv_path = os.path.join(logdir + '/final_output', PKL_FILE + '_compute_uncertainty_metrics.csv')

    # data to be written row-wise in csv fil
    data = [
        ['TP Car SE', 'TP Car AE', 'TP Car MI', 'TP Car ETV', 'TP Car ATV', \
        'TP Ped & Cyc SE', 'TP Ped & Cyc AE', 'TP Ped & Cyc MI', 'TP Ped & Cyc ETV', 'TP Ped & Cyc ATV'],
        [output_um_row[0][0], output_um_row[0][1], output_um_row[0][2], output_um_row[0][3], output_um_row[0][4],
        output_um_row[1][0], output_um_row[1][1], output_um_row[1][2], output_um_row[1][3], output_um_row[1][4],
        ]
    ]

    # opening the csv file in 'a+' mode
    file = open(csv_path, 'a+', newline ='')
    
    # writing the data into the file
    with file:    
        write = csv.writer(file)
        write.writerows(data)


    csv_path = os.path.join(logdir + '/final_output', PKL_FILE + '_compute_uncertainty_tp_car.csv')

    output_rows = []

    uncertainty_text = ['SE', 'AE', 'MI', 'ETV', 'ATV']
    for um in uncertainty_text:
        stats = np.array(tp_car_uncertainty[um])
        dist_close = stats[:,0]
        dist_med = stats[:,1]
        dist_far = stats[:,2]
        occ_low = stats[:,3]
        occ_med = stats[:,4]
        occ_high = stats[:,5]
        output_rows.append([um, \
            np.nanmean(dist_close).round(4), np.nanmean(dist_med).round(4), np.nanmean(dist_far).round(4), \
            np.nanmean(occ_low).round(4), np.nanmean(occ_med).round(4), np.nanmean(occ_high).round(4)])

    # data to be written row-wise in csv fil
    data = [
        ['Uncertainty', 'Dist <= 10', '10 < Dist <= 20', '20 < Dist <= 30', 'occ 0', 'occ 1', 'occ 2'],
        output_rows[0],
        output_rows[1],
        output_rows[2],
        output_rows[3],
        output_rows[4]
    ]

    # opening the csv file in 'a+' mode
    file = open(csv_path, 'a+', newline ='')
    
    # writing the data into the file
    with file:    
        write = csv.writer(file)
        write.writerows(data)

main()
