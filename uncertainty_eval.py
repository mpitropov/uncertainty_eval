import os
from pprint import pprint

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from tqdm import tqdm

# Dataset evaluation
from detection_eval.detection_eval import DetectionEval
from detection_eval.filter import build_kitti_filters, ClassFilter
from detection_eval.box_list import BoxList

# Cluster predictions
from cluster import cluster_preds

# Scoring Rules
from nll_clf import NLLCLF
from nll_reg import NLLREG
from binary_brier_score import BINARYBRIERSCORE
from brier_score import BRIERSCORE
from dmm import DMM
from energy_score import ENERGYSCORE

# Calibration Error
from ece import calculate_ece, plot_reliability
import calibration as cal

dataset_path = '/home/matthew/git/cadc_testing/WISEOpenLidarPerceptron/data/kitti'
logdir = '/home/matthew/git/cadc_testing/uncertainty_eval'
gts_path = os.path.join(logdir, 'gts.pkl')

# Clustering
logdir = '/home/matthew/git/cadc_testing/uncertainty_eval'
ENSEMBLE_TYPE = 1
VOTING_STATEGY = 1

if ENSEMBLE_TYPE == -1: # original model
    print('Ensemble type: OpenPCDet PointPillars 80 epochs')
    MIN_CLUSTER_SIZE = 1
    preds_path = os.path.join(logdir, 'openpcdet_80_result.pkl')
elif ENSEMBLE_TYPE == 0: # mc-dropout
    print('Ensemble type: mc-dropout')
    MIN_CLUSTER_SIZE = 20
    preds_path = os.path.join(logdir, 'mcdropout_20_result.pkl')
    # MIN_CLUSTER_SIZE = 5
    # preds_path = os.path.join(logdir, 'mcdropout_5_result.pkl')
elif ENSEMBLE_TYPE == 1: # ensemble
    print('Ensemble type: ensemble')
    MIN_CLUSTER_SIZE = 4
    preds_path = os.path.join(logdir, 'ensemble_result.pkl')
elif ENSEMBLE_TYPE == 2: # mimo
    print('Ensemble type: mimo')
    MIN_CLUSTER_SIZE = 2
    preds_path = os.path.join(logdir, 'result_converted.pkl')
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

# Uncertainty Eval path to save the reliability diagrams
uncertainty_eval_path='/home/matthew/git/cadc_testing/uncertainty_eval'

def load_dicts():
    # Load gt and prediction data dict
    with open(gts_path, 'rb') as f:
        gt_dicts = pickle.load(f)
    with open(preds_path, 'rb') as f:
        pred_dicts = pickle.load(f)
    return gt_dicts, pred_dicts

def load_image(frame_id):
    img_path = os.path.join(dataset_path, 'training', 'image_2', f'{frame_id}.png')
    return plt.imread(img_path)

def load_lidar(frame_id, xlim, ylim):
    lidar_path = os.path.join(dataset_path, 'training', 'velodyne', f'{frame_id}.bin')
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    valid_mask = (points[:,0] > xlim[0]) & \
        (points[:,0] < xlim[1]) & \
        (points[:,1] > ylim[0]) & \
        (points[:,1] < ylim[1]) & \
        (points[:,2] < 4)
    points = points[valid_mask]
    return points

def add_box(ax, box, color=None):
    # box: [x, y, z, w, l, h, theta]
    w = box[3]
    h = box[4]
    xy = (box[0] - w/2, box[1] - h/2)
    angle = box[-1] * 180 / np.pi
    box_rect = Rectangle(
        xy, w, h, facecolor='none', edgecolor=color
    )
    t = Affine2D().rotate_around(box[0], box[1], box[-1]) + ax.transData
    box_rect.set_transform(t)
    ax.add_patch(box_rect)

def get_labels(data_dict):
    if 'gt_labels' in data_dict:
        return data_dict['gt_labels']
    if 'name' in data_dict:
        classes = ['Car', 'Pedestrian', 'Cyclist']
        return np.array([classes.index(name)+1 for name in data_dict['name']])
    raise ValueError()

def get_boxes(data_dict):
    if 'boxes_lidar' in data_dict:
        return data_dict['boxes_lidar']
    if 'gt_boxes' in data_dict:
        return data_dict['gt_boxes']
    raise ValueError()

def get_scores(data_dict):
    return data_dict['score']
    
def gt_processor(data_dict):
    return ( get_labels(data_dict), get_boxes(data_dict) )

def pred_processor(data_dict):
    return (get_labels(data_dict),
            get_scores(data_dict),
            get_boxes(data_dict))

def get_dist(data_dict):
    boxes = get_boxes(data_dict)
    coords = boxes[:,:2]
    dist = np.linalg.norm(coords, axis=1)
    return dist

def main():
    print("Load dictionaries...")
    gt_dicts, pred_dicts = load_dicts()
    print("Clustering predictions by frame...")
    pred_dicts = cluster_preds(pred_dicts, MIN_CLUSTER_SIZE)

    # Threshold (list or dict) maps a label to a matching threshold
    # thresholds[label] = threshold
    thresholds = {
        0: 0.5,    # Just to not output error
        1: 0.7,    # Car
        2: 0.5,    # Pedestrian
        3: 0.5     # Cyclist
    }
    filter_list = build_kitti_filters(dataset_path + '/kitti_infos_val.pkl')

    for filter_idx in range(len(filter_list)):
        print("Performing evaluation for class " + filter_list[filter_idx].class_name + \
            " and difficulty " + str(filter_list[filter_idx].difficulty))

        # Evaluate over validation set
        print("Loop through and evaluate all samples...")
        def attach_data(sample_idx, gt_dict, pred_dict, gt_list, pred_list):
            for i in range(len(gt_list)):
                gt_list.data[i] = dict(gt_boxes=gt_dict['gt_boxes'][i])
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
                        pred_vars=pred_dict['pred_vars'][i]
                    )

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

        print("Evaluate Uncertainty...")
        # A prediction box is either a TP or FP
        # TP is valid, localized and classified
        tp = pred_list.valid & pred_list.localized & pred_list.classified
        # FP is valid and either not localized or not classified correctly
        fp = pred_list.valid & ~(pred_list.localized & pred_list.classified)

        NUM_CLASSES = 3

        # Init Scoring Rules
        nll_clf_obj = NLLCLF()
        nll_reg_obj = NLLREG()
        binary_brier_obj = BINARYBRIERSCORE()
        brier_obj = BRIERSCORE()
        dmm_obj = DMM()
        energy_obj = ENERGYSCORE()

        # Our own ECE Implementation
        num_bins = 10
        bins = np.arange(0, 1+1/num_bins, 1/num_bins)
        conf_mat = [{'TP': 0, 'FP': 0} for i in range(len(bins))]

        # Calibration library lists
        preds_score_all = []
        gt_score_all = []

        print('Number of predictions', len(pred_list))
        print('Number of TPs', len(pred_list[tp]))
        print('Number of FPs', len(pred_list[fp]))

        # Scoring Rules
        if ENSEMBLE_TYPE == -1: # original model
            # TP loop
            for obj in pred_list[tp]:
                nll_clf_obj.add_tp(obj.pred_score)
                binary_brier_obj.add_tp(obj.pred_score)

            # FP loop
            for obj in pred_list[fp]:
                nll_clf_obj.add_fp(obj.pred_score)
                binary_brier_obj.add_fp(obj.pred_score)
        else:
            # TP loop
            for obj in pred_list[tp]:
                gt_box = gt_list[int(obj.matched_idx)].data['gt_boxes']
                nll_clf_obj.add_tp(obj.pred_score)
                nll_reg_obj.add_tp(gt_box, obj.data['boxes_lidar'], obj.data['pred_vars'])
                binary_brier_obj.add_tp(obj.pred_score)
                brier_obj.add_tp(obj.pred_label, obj.data['score_all'])
                dmm_obj.add_tp(gt_box, obj.data['boxes_lidar'], obj.data['pred_vars'])
                energy_obj.add_tp(gt_box, obj.data['boxes_lidar'], obj.data['pred_vars'])

            # FP loop
            for obj in pred_list[fp]:
                nll_clf_obj.add_bg_tp(obj.data['score_all'][NUM_CLASSES])
                binary_brier_obj.add_bg_tp(obj.data['score_all'][NUM_CLASSES])
                brier_obj.add_fp(NUM_CLASSES, obj.data['score_all'])
                energy_obj.add_fp(obj.data['boxes_lidar'], obj.data['pred_vars'])

        print('NLL Classification mean', nll_clf_obj.mean())
        print('NLL Classification mean TP', nll_clf_obj.mean_tp())
        print('NLL Classification mean FP', nll_clf_obj.mean_fp())
        print('Binary Brier Score Classification mean', binary_brier_obj.mean())
        print('Binary Brier Score Classification mean TP', binary_brier_obj.mean_tp())
        print('Binary Brier Score Classification mean FP', binary_brier_obj.mean_fp())
        if ENSEMBLE_TYPE != -1:
            print('Brier Score Classification mean', brier_obj.mean())
            print('Brier Score Classification mean TP', brier_obj.mean_tp())
            print('Brier Score Classification mean FP', brier_obj.mean_fp())
            print('NLL Regression mean', nll_reg_obj.mean())
            print('DMM Regression mean', dmm_obj.mean())
            print('Energy Regression mean', energy_obj.mean())

        # Calibration Error
        if ENSEMBLE_TYPE == -1: # original model
            # TP loop Calibration Error
            for obj in pred_list[tp]:
                bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
                conf_mat[bin_num]['TP'] += 1
                # Calibration library
                preds_score_all.append(obj.pred_score)
                # Add GT as 1
                gt_score_all.append(1)

            # FP loop Calibration Error
            for obj in pred_list[fp]:
                bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
                conf_mat[bin_num]['FP'] += 1
                # Calibration library
                preds_score_all.append(obj.pred_score)
                # Add GT as 0
                gt_score_all.append(0)
        else:
            # TP loop Calibration Error
            for obj in pred_list[tp]:
                bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
                conf_mat[bin_num]['TP'] += 1
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

        accuracy, ece, key = calculate_ece(conf_mat, num_bins, filter_list[filter_idx].name)

        print("Accuracy bins", accuracy)
        print("Calculated ECE", ece)

        plot_reliability(accuracy, ece, save_path=uncertainty_eval_path+'/ECE_'+key+'.png')

        # ECE
        cls_expected_calibration_error = cal.get_ece(
            preds_score_all, gt_score_all)
        print("cls_expected_calibration_error", cls_expected_calibration_error)
        # ECE stable binning (seems the same)
        cls_expected_calibration_error = cal.get_top_calibration_error(
            preds_score_all, gt_score_all, p=1)
        print("cls_expected_calibration_error 'stable binning'", cls_expected_calibration_error)
        # Marginal CE
        cls_marginal_calibration_error = cal.get_calibration_error(
            preds_score_all, gt_score_all)
        print("cls_marginal_calibration_error", cls_marginal_calibration_error)

main()
