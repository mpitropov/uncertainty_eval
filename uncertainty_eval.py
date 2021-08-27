from math import e
import os, sys, copy, gc
from pprint import pprint

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from tqdm import tqdm

# Dataset evaluation
from detection_eval.detection_eval import DetectionEval
from detection_eval.filter import build_kitti_filters, build_cadc_filters, build_nuscenes_filters
from detection_eval.box_list import combine_box_lists

# Cluster predictions
from cluster import cluster_preds

from scipy.special import entr

# Scoring Rules
from scoring.nll_clf import NLLCLF
from scoring.nll_reg import NLLREG
from scoring.binary_brier_score import BINARYBRIERSCORE
from scoring.brier_score import BRIERSCORE
from scoring.dmm import DMM
from scoring.energy_score import ENERGYSCORE

# Calibration Error
from ece import calculate_ece, plot_reliability_clf, calculate_ece_reg, plot_reliability_reg
import calibration as cal

dataset_types = ['KITTI', 'CADC', 'NuScenes']
selected_dataset = dataset_types[1]
dataset_path = None
if selected_dataset == dataset_types[0]:
    dataset_path = '/root/kitti'
    logdir = '/root/logdir/output_pkls'
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
            thresholds=thresholds,
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
    ap_list = []

    # Clear memory, mostly needed for large datasets like NuScenes
    del gt_dicts
    del pred_dicts
    gc.collect()

    for idx in range(len(class_names)):
        print("Evaluate Uncertainty for " + class_names[idx] + "...")
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

        # import math
        # tp_list = pred_list[tp | l_mc | ml_c | ml_mc]
        # tp_var_sum = []
        # tp_var_sums = [[],[],[]]
        # tp_norm_var_sum = []
        # tp_vars = []
        # tp_scores = []
        # tp_dists = []
        # tp_shannon_entropys = []
        # tp_aleatoric_entropys = []
        # tp_mutual_info = []
        # tp_means = [[],[],[],[],[],[],[]]
        # for j in range(len(tp_list)):
        #     obj = tp_list[j]
        #     tp_var_sum.append(np.sum(obj.data['pred_vars']))
        #     tp_var_sums[0].append(np.sum(obj.data['pred_vars'][0:3]))
        #     tp_var_sums[1].append(np.sum(obj.data['pred_vars'][3:6]))
        #     tp_var_sums[2].append(np.sum(obj.data['pred_vars'][6]))
        #     norm_var = np.sqrt(np.array(obj.data['pred_vars']))/np.abs(np.array(obj.data['boxes_lidar']))
        #     tp_norm_var_sum.append(np.sum(norm_var))
        #     tp_vars.append(obj.data['pred_vars'])
        #     tp_scores.append(obj.pred_score)
        #     tp_dists.append(math.sqrt(obj.data['boxes_lidar'][0] ** 2 + obj.data['boxes_lidar'][1] ** 2) )
        #     tp_shannon_entropys.append(obj.data['shannon_entropy'])
        #     tp_aleatoric_entropys.append(obj.data['aleatoric_entropy'])
        #     tp_mutual_info.append(obj.data['shannon_entropy'] - obj.data['aleatoric_entropy'])
        #     for x in range(len(tp_means)):
        #         tp_means[x].append(obj.data['boxes_lidar'][x])

        # print('tp shannon entropy mean', np.mean(tp_shannon_entropys).round(2))
        # print('tp aleatoric entropy mean', np.mean(tp_aleatoric_entropys).round(2))
        # print('tp mutual info mean', np.mean(tp_mutual_info).round(2))
        # print('tp distance variance mean', np.mean(tp_var_sums[0]).round(2))
        # print('tp dimensions variance mean', np.mean(tp_var_sums[1]).round(2))
        # print('tp yaw variance mean', np.mean(tp_var_sums[2]).round(2))

        # fp_list = pred_list[fp]
        # fp_var_sum = []
        # fp_var_sums = [[],[],[]]
        # fp_norm_var_sum = []
        # fp_vars = []
        # fp_scores = []
        # fp_dists = []
        # fp_shannon_entropys = []
        # fp_aleatoric_entropys = []
        # fp_mutual_info = []
        # fp_means = [[],[],[],[],[],[],[]]
        # for j in range(len(fp_list)):
        #     obj = fp_list[j]
        #     fp_var_sum.append(np.sum(obj.data['pred_vars']))
        #     fp_var_sums[0].append(np.sum(obj.data['pred_vars'][0:3]))
        #     fp_var_sums[1].append(np.sum(obj.data['pred_vars'][3:6]))
        #     fp_var_sums[2].append(np.sum(obj.data['pred_vars'][6]))
        #     norm_var = np.sqrt(np.array(obj.data['pred_vars']))/np.abs(np.array(obj.data['boxes_lidar']))
        #     fp_norm_var_sum.append(np.sum(norm_var))
        #     fp_vars.append(obj.data['pred_vars'])
        #     fp_scores.append(obj.pred_score)
        #     fp_dists.append(math.sqrt(obj.data['boxes_lidar'][0] ** 2 + obj.data['boxes_lidar'][1] ** 2) )
        #     fp_shannon_entropys.append(obj.data['shannon_entropy'])
        #     fp_aleatoric_entropys.append(obj.data['aleatoric_entropy'])
        #     fp_mutual_info.append(obj.data['shannon_entropy'] - obj.data['aleatoric_entropy'])
        #     for x in range(len(fp_means)):
        #         fp_means[x].append(obj.data['boxes_lidar'][x])

        # print('fp shannon entropy mean', np.mean(fp_shannon_entropys).round(2))
        # print('fp aleatoric entropy mean', np.mean(fp_aleatoric_entropys).round(2))
        # print('fp mutual info mean', np.mean(fp_mutual_info).round(2))
        # print('fp distance variance mean', np.mean(fp_var_sums[0]).round(2))
        # print('fp dimensions variance mean', np.mean(fp_var_sums[1]).round(2))
        # print('fp yaw variance mean', np.mean(fp_var_sums[2]).round(2))

        # import matplotlib.pyplot as plt

        # plt.hist(tp_var_sum, bins =10)
        # plt.title(class_names[idx] + '_tp_var_sum_hist.png')
        # plt.savefig('images/' + class_names[idx] + '_tp_var_sum_hist.png')
        # plt.close()

        # plt.hist(fp_var_sum, bins =10)
        # plt.title(class_names[idx] + '_fp_var_sum_hist.png')
        # plt.savefig('images/' + class_names[idx] + '_fp_var_sum_hist.png')
        # plt.close()

        # plt.hist(tp_scores, bins =10)
        # plt.title(class_names[idx] + '_tp_scores_hist.png')
        # plt.savefig('images/' + class_names[idx] + '_tp_scores_hist.png')
        # plt.close()

        # plt.hist(fp_scores, bins =10)
        # plt.title(class_names[idx] + '_fp_scores_hist.png')
        # plt.savefig('images/' + class_names[idx] + '_fp_scores_hist.png')
        # plt.close()

        # plt.hist(tp_dists, bins =50)
        # plt.title(class_names[idx] + '_tp_dists_hist.png')
        # plt.savefig('images/' + class_names[idx] + '_tp_dists_hist.png')
        # plt.close()

        # plt.hist(fp_dists, bins =50)
        # plt.title(class_names[idx] + '_fp_dists_hist.png')
        # plt.savefig('images/' + class_names[idx] + '_fp_dists_hist.png')
        # plt.close()

        # plt.hist(tp_shannon_entropys, bins=10)
        # plt.title(class_names[idx] + '_tp_shannon_entropy_hist.png')
        # plt.savefig('images/' + class_names[idx] + '_tp_shannon_entropy_hist.png')
        # plt.close()
        # plt.hist(fp_shannon_entropys, bins=10)
        # plt.title(class_names[idx] + '_fp_shannon_entropy_hist.png')
        # plt.savefig('images/' + class_names[idx] + '_fp_shannon_entropy_hist.png')
        # plt.close()
        # plt.hist(tp_aleatoric_entropys, bins=10)
        # plt.title(class_names[idx] + '_tp_aleatoric_entropy_hist.png')
        # plt.savefig('images/' + class_names[idx] + '_tp_aleatoric_entropy_hist.png')
        # plt.close()
        # plt.hist(fp_aleatoric_entropys, bins=10)
        # plt.title(class_names[idx] + '_fp_aleatoric_entropy_hist.png')
        # plt.savefig('images/' + class_names[idx] + '_fp_aleatoric_entropy_hist.png')
        # plt.close()
        # plt.hist(tp_mutual_info, bins=10)
        # plt.title(class_names[idx] + '_tp_mutual_info_hist.png')
        # plt.savefig('images/' + class_names[idx] + '_tp_mutual_info_hist.png')
        # plt.close()
        # plt.hist(fp_mutual_info, bins=10)
        # plt.title(class_names[idx] + '_fp_mutual_info_hist.png')
        # plt.savefig('images/' + class_names[idx] + '_fp_mutual_info_hist.png')
        # plt.close()

        # plt.hist2d(x=tp_dists, y=tp_var_sum, bins=50, cmap='hot', cmin = 1)
        # # plt.scatter(x=tp_dists, y=tp_var_sum)
        # plt.title(class_names[idx] + '_tp_var_sum_vs_dists_hmap.png')
        # plt.colorbar()
        # plt.savefig('images/' + class_names[idx] + '_tp_var_sum_vs_dists_hmap.png')
        # plt.close()

        # plt.hist2d(x=fp_dists, y=fp_var_sum, bins=50, cmap='hot', cmin = 1)
        # # plt.scatter(x=fp_dists, y=fp_var_sum)
        # plt.title(class_names[idx] + '_fp_var_sum_vs_dists_hmap.png')
        # plt.colorbar()
        # plt.savefig('images/' + class_names[idx] + '_fp_var_sum_vs_dists_hmap.png')
        # plt.close()

        # # var sum
        # plot_names = ['distance', 'dimensions', 'yaw']
        # for i in range(len(tp_var_sums)):
        #     plt.hist2d(x=tp_dists, y=tp_var_sums[i], bins=50, cmap='hot', cmin = 1)
        #     plt.title(class_names[idx] + '_tp_var_sum_' + plot_names[i] + '_vs_dists_hmap.png')
        #     plt.colorbar()
        #     plt.savefig('images/' + class_names[idx] + '_tp_var_sum_' + plot_names[i] + '_vs_dists_hmap.png')
        #     plt.close()

        #     plt.hist2d(x=fp_dists, y=fp_var_sums[i], bins=50, cmap='hot', cmin = 1)
        #     plt.title(class_names[idx] + '_fp_var_sum_' + plot_names[i] + '_vs_dists_hmap.png')
        #     plt.colorbar()
        #     plt.savefig('images/' + class_names[idx] + '_fp_var_sum_' + plot_names[i] + '_vs_dists_hmap.png')
        #     plt.close()

        # plt.hist2d(x=tp_dists, y=tp_norm_var_sum, bins=100, cmap='hot', cmin = 1)
        # # plt.scatter(x=tp_dists, y=tp_norm_var_sum)
        # plt.yscale('log')
        # plt.title(class_names[idx] + '_tp_norm_var_sum_vs_dists_hmap.png')
        # plt.colorbar()
        # plt.savefig('images/' + class_names[idx] + '_tp_norm_var_sum_vs_dists_hmap.png')
        # plt.close()

        # plt.hist2d(x=fp_dists, y=fp_norm_var_sum, bins=100, cmap='hot', cmin = 1)
        # # plt.scatter(x=fp_dists, y=fp_norm_var_sum)
        # plt.yscale('log')
        # plt.title(class_names[idx] + '_fp_norm_var_sum_vs_dists_hmap.png')
        # plt.colorbar()
        # plt.savefig('images/' + class_names[idx] + '_fp_norm_var_sum_vs_dists_hmap.png')
        # plt.close()

        # plt.hist2d(x=tp_scores, y=tp_var_sum, bins=100, cmap='hot', cmin = 1)
        # plt.title(class_names[idx] + '_tp_var_sum_vs_scores_hmap.png')
        # plt.colorbar()
        # plt.savefig('images/' + class_names[idx] + '_tp_var_sum_vs_scores_hmap.png')
        # plt.close()

        # plt.hist2d(x=fp_scores, y=fp_var_sum, bins=100, cmap='hot', cmin = 1)
        # plt.title(class_names[idx] + '_fp_var_sum_vs_scores_hmap.png')
        # plt.colorbar()
        # plt.savefig('images/' + class_names[idx] + '_fp_var_sum_vs_scores_hmap.png')
        # plt.close()

        # # for x in range(len(tp_means)):
        # #     plt.hist(tp_means[x], bins =50)
        # #     plt.title(class_names[idx] + '_index' + str(x) + '_tp_means_hist.png')
        # #     plt.savefig('images/' + class_names[idx] + '_index' + str(x) + '_tp_means_hist.png')
        # #     plt.close()

        # #     plt.hist(fp_means[x], bins =50)
        # #     plt.title(class_names[idx] + '_index' + str(x) + '_fp_means_hist.png')
        # #     plt.savefig('images/' + class_names[idx] + '_index' + str(x) + '_fp_means_hist.png')
        # #     plt.close()

        # # Create DT data
        # # For DT
        # X = []
        # y = []
        # l_mc_list = pred_list[l_mc]
        # ml_c_list = pred_list[ml_c]
        # ml_mc_list = pred_list[ml_mc]
        # bg_list = pred_list[bg]
        # normal_mode = False
        # # obj.pred_score
        # for j in range(len(tp_list)):
        #     obj = tp_list[j]
        #     dist = math.sqrt(obj.data['boxes_lidar'][0] ** 2 + obj.data['boxes_lidar'][1] ** 2)
        #     obj_entropy = entr(obj.data['score_all']).sum()
        #     if normal_mode:
        #         test = np.concatenate(([obj.pred_label, obj.pred_score, dist], [obj_entropy]))
        #     else:
        #         test = np.concatenate(([obj.pred_label, obj.pred_score, dist, np.sum(obj.data['pred_vars'])], [obj_entropy]))
        #     X.append(test)
        #     y.append(1)
        # for j in range(len(l_mc_list)):
        #     obj = l_mc_list[j]
        #     dist = math.sqrt(obj.data['boxes_lidar'][0] ** 2 + obj.data['boxes_lidar'][1] ** 2)
        #     obj_entropy = entr(obj.data['score_all']).sum()
        #     if normal_mode:
        #         test = np.concatenate(([obj.pred_label, obj.pred_score, dist], [obj_entropy]))
        #     else:
        #         test = np.concatenate(([obj.pred_label, obj.pred_score, dist, np.sum(obj.data['pred_vars'])], [obj_entropy]))
        #     X.append(test)
        #     y.append(1)
        # for j in range(len(ml_c_list)):
        #     obj = ml_c_list[j]
        #     dist = math.sqrt(obj.data['boxes_lidar'][0] ** 2 + obj.data['boxes_lidar'][1] ** 2)
        #     obj_entropy = entr(obj.data['score_all']).sum()
        #     if normal_mode:
        #         test = np.concatenate(([obj.pred_label, obj.pred_score, dist], [obj_entropy]))
        #     else:
        #         test = np.concatenate(([obj.pred_label, obj.pred_score, dist, np.sum(obj.data['pred_vars'])], [obj_entropy]))
        #     X.append(test)
        #     y.append(0)
        # for j in range(len(ml_mc_list)):
        #     obj = ml_mc_list[j]
        #     dist = math.sqrt(obj.data['boxes_lidar'][0] ** 2 + obj.data['boxes_lidar'][1] ** 2)
        #     obj_entropy = entr(obj.data['score_all']).sum()
        #     if normal_mode:
        #         test = np.concatenate(([obj.pred_label, obj.pred_score, dist], [obj_entropy]))
        #     else:
        #         test = np.concatenate(([obj.pred_label, obj.pred_score, dist, np.sum(obj.data['pred_vars'])], [obj_entropy]))
        #     X.append(test)
        #     y.append(0)
        # for j in range(len(bg_list)):
        #     obj = bg_list[j]
        #     dist = math.sqrt(obj.data['boxes_lidar'][0] ** 2 + obj.data['boxes_lidar'][1] ** 2)
        #     obj_entropy = entr(obj.data['score_all']).sum()
        #     if normal_mode:
        #         test = np.concatenate(([obj.pred_label, obj.pred_score, dist], [obj_entropy]))
        #     else:
        #         test = np.concatenate(([obj.pred_label, obj.pred_score, dist, np.sum(obj.data['pred_vars'])], [obj_entropy]))
        #     X.append(test)
        #     y.append(0)

        # tp_list = pred_list[tp]
        # print('tp_list', len(tp_list))
        # pred_var_minus_actual_diff = []
        # pred_vars = []
        # for j in range(len(tp_list)):
        #     obj = tp_list[j]
        #     pred_box = obj.data['boxes_lidar']
        #     gt_box = gt_list[int(obj.matched_idx)].data['gt_boxes']
        #     pred_minus_gt = np.square(pred_box - gt_box)
        #     pred_vars.append( obj.data['pred_vars'] )
        #     pred_var_minus_actual_diff.append( np.abs(obj.data['pred_vars'] - pred_minus_gt) )
        # pred_vars = np.array(pred_vars)
        # pred_var_minus_actual_diff = np.array(pred_var_minus_actual_diff)
        # for x in range(len(pred_var_minus_actual_diff[0])):
        #     plt.hist(pred_var_minus_actual_diff[:,x], bins =50)
        #     plt.title(class_names[idx] + '_index' + str(x) + '_var_diff_hist.png')
        #     plt.savefig('images/' + class_names[idx] + '_index' + str(x) + '_var_diff_hist.png')
        #     plt.close()
        #     plt.hist(pred_vars[:,x], bins =50)
        #     plt.title(class_names[idx] + '_index' + str(x) + '_pred_vars_hist.png')
        #     plt.savefig('images/' + class_names[idx] + '_index' + str(x) + '_pred_vars_hist.png')
        #     plt.close()

        # Init Scoring Rules
        nll_clf_obj = NLLCLF()
        nll_reg_obj = NLLREG()
        binary_brier_obj = BINARYBRIERSCORE()
        brier_obj = BRIERSCORE()
        dmm_obj = DMM()
        energy_obj = ENERGYSCORE()

        # Our own ECE Implementation
        num_bins = 10
        bins = np.arange(0, 1.0+(1.0/num_bins), 1.0/num_bins)
        conf_mat = [{'TP': 0, 'FP': 0, 'confidence_list': []} for i in range(len(bins))]

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
                            pred_list=pred_list, n_positions=num_recall_positions)
        print('Number of FNs', stats_result['fn'])
        if idx == NUM_CLASSES: # Use mean of APs
            print('mAP@40', np.mean(ap_list).round(2))
        else:
            curr_ap = (100 * stats_result['ap'])
            ap_list.append(curr_ap)
            print('AP@40', curr_ap.round(2))

        # Scoring Rules

        if ENSEMBLE_TYPE == -1: # original model
            # TP
            nll_clf_obj.add_tp(pred_list[tp])
            binary_brier_obj.add_tp(pred_list[tp])

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
            # energy_obj.add_tp(gt_list, pred_list[tp])

            # FP 
            nll_clf_obj.add_bg_tp(pred_list[fp])
            binary_brier_obj.add_bg_tp(pred_list[fp])
            brier_obj.add_fp(pred_list[fp])
            # energy_obj.add_fp(pred_list[fp])

        # Skip
        # continue

        print('NLL Classification mean', nll_clf_obj.mean().round(4))
        print('NLL Classification mean TP', nll_clf_obj.mean_tp().round(4))
        print('NLL Classification mean FP', nll_clf_obj.mean_fp().round(4))
        print('Binary Brier Score Classification mean', binary_brier_obj.mean().round(4))
        print('Binary Brier Score Classification mean TP', binary_brier_obj.mean_tp().round(4))
        print('Binary Brier Score Classification mean FP', binary_brier_obj.mean_fp().round(4))
        if ENSEMBLE_TYPE != -1:
            print('Brier Score Classification mean', brier_obj.mean().round(4))
            print('Brier Score Classification mean TP', brier_obj.mean_tp().round(4))
            print('Brier Score Classification mean FP', brier_obj.mean_fp().round(4))
            print('NLL Regression mean', nll_reg_obj.mean().round(4))
            print('DMM Regression mean', dmm_obj.mean().round(4))
            # print('Energy Regression mean', energy_obj.mean().round(4))

        # Calibration Error
        if ENSEMBLE_TYPE == -1: # original model
            # TP loop Calibration Error
            for obj in pred_list[tp]:
                bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
                conf_mat[bin_num]['TP'] += 1
                conf_mat[bin_num]['confidence_list'].append(obj.pred_score)
                # Calibration library
                preds_score_all.append(obj.pred_score)
                # Add GT as 1
                gt_score_all.append(1)

            # FP loop Calibration Error
            for obj in pred_list[fp]:
                bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
                conf_mat[bin_num]['FP'] += 1
                conf_mat[bin_num]['confidence_list'].append(obj.pred_score)
                # Calibration library
                preds_score_all.append(obj.pred_score)
                # Add GT as 0
                gt_score_all.append(0)
        else:
            # TP loop Calibration Error
            for obj in pred_list[tp]:
                bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
                conf_mat[bin_num]['TP'] += 1
                conf_mat[bin_num]['confidence_list'].append(obj.pred_score)
                # # Calibration library
                # for score in obj.data['score_all']:
                #     preds_score_all.append(score)
                # # add one hot to GT
                # gt_label = obj.gt_label
                # one_hot_gt = np.eye(NUM_CLASSES+1, dtype=int)[int(gt_label - 1)]
                # for score in one_hot_gt:
                #     gt_score_all.append(score)
                # Calibration library
                preds_score_all.append(obj.data['score_all'])
                # add class to GT
                gt_label = obj.gt_label
                gt_score_all.append(int(gt_label - 1))
            # FP loop Calibration Error
            for obj in pred_list[fp]:
                bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
                conf_mat[bin_num]['FP'] += 1
                conf_mat[bin_num]['confidence_list'].append(obj.pred_score)
                # # Calibration library
                # for score in obj.data['score_all']:
                #     preds_score_all.append(score)
                # one_hot_gt = np.eye(NUM_CLASSES+1, dtype=int)[NUM_CLASSES] # [0,0,...,0,1]
                # for score in one_hot_gt:
                #     gt_score_all.append(score)
                # Calibration library
                preds_score_all.append(obj.data['score_all'])
                # Append BG class
                gt_score_all.append(NUM_CLASSES)

        accuracy, confidence, ece, max_ce, key = calculate_ece(conf_mat, num_bins, class_names[idx])
        print("Calculated ECE", ece)
        print("Calculated MaxCE", max_ce)
        print("Calculated Accuracy bins", accuracy)
        # print("Calculated conf_mat", conf_mat)

        plot_reliability_clf(key, accuracy, confidence, ece, max_ce, save_path='calibration_images/ECE_CLF_'+key+'.png')

        # ECE
        cls_expected_calibration_error = cal.get_ece(
            preds_score_all, gt_score_all)
        print("cls_expected_calibration_error", cls_expected_calibration_error.round(4))
        # ECE stable binning (seems the same)
        cls_expected_calibration_error = cal.get_top_calibration_error(
            preds_score_all, gt_score_all, p=1)
        print("cls_expected_calibration_error 'stable binning'", cls_expected_calibration_error.round(4))
        # Marginal CE
        cls_marginal_calibration_error = cal.get_calibration_error(
            preds_score_all, gt_score_all)
        print("cls_marginal_calibration_error", cls_marginal_calibration_error.round(4))

        if ENSEMBLE_TYPE != -1:
            reg_maximum_calibration_error, reg_expected_calibration_error, acc_list = calculate_ece_reg(gt_list, pred_list[tp])
            print("Regression Maximum Calibration Error", reg_maximum_calibration_error.round(4))
            print("Regression Expected Calibration Error", reg_expected_calibration_error.round(4))
            plot_reliability_reg(key, acc_list, reg_expected_calibration_error, \
                                save_path='calibration_images/ECE_REG_'+key+'.png')

main()
