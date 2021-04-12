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

# Calibration Error
from ece import calculate_ece, plot_reliability
import calibration as cal

dataset_path = '/home/matthew/git/cadc_testing/WISEOpenLidarPerceptron/data/kitti'
logdir = '/home/matthew/git/cadc_testing/al_output/pcdet_log_20_epoch/kittidatasetvar/random/eval_192e0b82f64b5c2207934a583985aa2e'
gts_path = os.path.join(logdir, 'gt.pkl')

# Clustering
logdir = '/home/matthew/git/cadc_testing/uncertainty_eval'
preds_path = os.path.join(logdir, 'result_converted.pkl')
# preds_path = os.path.join(logdir, 'dropout_s11_result.pkl')
# preds_path = os.path.join(logdir, 'dropout_s11_old_result.pkl')

# Uncertainty Eval path to save the reliability diagram
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
    pred_dicts = cluster_preds(pred_dicts)

    # Threshold (list or dict) maps a label to a matching threshold
    # thresholds[label] = threshold
    thresholds = {
        0: 0.5,    # Just to not output error
        1: 0.7,    # Car
        2: 0.5,    # Pedestrian
        3: 0.5     # Cyclist
    }
    car_filter = ClassFilter(name='Car', label=1,
                            gt_processor=gt_processor, pred_processor=pred_processor)
    ped_filter = ClassFilter(name='Pedestrian', label=2,
                            gt_processor=gt_processor, pred_processor=pred_processor)
    cyc_filter = ClassFilter(name='Cyclist', label=3,
                            gt_processor=gt_processor, pred_processor=pred_processor)
    filter_list = [car_filter, ped_filter, cyc_filter]
    # filter_list = build_kitti_filters(dataset_path + '/kitti_infos_val.pkl')

    for filter_idx in range(len(filter_list)):
        print("Performing evaluation for:")
        print(filter_list[filter_idx].name)
        # print(filter_list[filter_idx].class_name, filter_list[filter_idx].class_label, \
        #     filter_list[filter_idx].difficulty)

        # Evaluate over validation set
        print("Loop through all dictionaries for each sample...")
        def attach_data(sample_idx, gt_dict, pred_dict, gt_list, pred_list):
            for i in range(len(gt_list)):
                gt_list.data[i] = dict(gt_boxes=gt_dict['gt_boxes'][i])
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
        # TP is both localized and classified
        tp = (~pred_list.ignored) & (pred_list.localized) & (pred_list.classified)
        # FP is the negative of TPs
        fp = (~tp)

        NUM_CLASSES = 3

        # Init Scoring Rules
        nll_clf_obj = NLLCLF()
        nll_reg_obj = NLLREG()
        binary_brier_obj = BINARYBRIERSCORE()
        brier_obj = BRIERSCORE()
        dmm_obj = DMM()

        # Our own ECE Implementation
        num_bins = 10
        bins = np.arange(0, 1+1/num_bins, 1/num_bins)
        conf_mat = [{'TP': 0, 'FP': 0} for i in range(len(bins))]

        # Calibration library lists
        preds_score_all = []
        gt_score_all = []

        # Keep track of odd TPs/FPs
        tp_dontcare_count = 0
        fp_dontcare_count = 0

        print("Display variance of first TP", pred_list[tp][0].data['pred_vars'])
        # TP loop
        for obj in pred_list[tp]:
            gt_box = gt_list[int(obj.matched_idx)].data['gt_boxes']
            # print(obj.data['pred_vars'])
            # if np.sum(obj.data['pred_vars']) > 1:
            #     print(obj.data['pred_vars'])
            nll_clf_obj.add_tp(obj.pred_score)
            nll_reg_obj.add_tp(gt_box, obj.data['boxes_lidar'], obj.data['pred_vars'])
            binary_brier_obj.add_tp(obj.pred_score)
            brier_obj.add_tp(obj.pred_label, obj.data['score_all'])
            dmm_obj.add_tp(gt_box, obj.data['boxes_lidar'], obj.data['pred_vars'])

        # FP loop
        for obj in pred_list[fp]:
            nll_clf_obj.add_fp(obj.pred_score)
            binary_brier_obj.add_fp(obj.pred_score)

        print('NLL Classification mean', nll_clf_obj.mean())
        print('NLL Regression mean', nll_reg_obj.mean())
        print('Binary Brier Score Classification mean', binary_brier_obj.mean())
        print('Brier Score Classification mean', brier_obj.mean())
        print('DMM Regression mean', dmm_obj.mean())

        # TP loop Calibration Error
        for obj in pred_list[tp]:
            bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
            conf_mat[bin_num]['TP'] += 1
            # Calibration library
            preds_score_all.append(obj.data['score_all'][0])
            preds_score_all.append(obj.data['score_all'][1])
            preds_score_all.append(obj.data['score_all'][2])
            # add one hot to GT
            gt_label = obj.gt_label
            for one_hot in range(NUM_CLASSES):
                if one_hot == int(gt_label - 1):
                    gt_score_all.append(1)
                else:
                    gt_score_all.append(0)

            # These check for strange output
        #     if pred['target_labels'][idx] == -1:
        #         tp_dontcare_count += 1
        #     if pred['target_labels'][idx] != pred['pred_labels'][idx]:
        #         print('WOAH target label != pred label')
        #         print(pred['target_labels'][idx])
        #         print(pred['pred_labels'][idx])

        # FP loop Calibration Error
        for obj in pred_list[fp]:
            bin_num = np.ceil(obj.pred_score * num_bins).astype(int)
            conf_mat[bin_num]['FP'] += 1
            nll_clf_obj.add_fp(obj.pred_score)
            binary_brier_obj.add_fp(obj.pred_score)
            # Calibration library
            preds_score_all.append(obj.data['score_all'][0])
            preds_score_all.append(obj.data['score_all'][1])
            preds_score_all.append(obj.data['score_all'][2])
            gt_score_all.append(0)
            gt_score_all.append(0)
            gt_score_all.append(0)

            # These check for strange output
        #     if pred['target_labels'][idx] == -1:
        #         print("FP with target label Don't Care")
        #         print(ret)
        #         fp_dontcare_count += 1
        #     elif pred['target_labels'][idx] == 0:
        #         print("FP with target label BG")
        #     else:
        #         print("Interesting case FP with not BG target label")

        accuracy, ece, key = calculate_ece(conf_mat, num_bins, filter_list[filter_idx].name)

        # print('tp_dontcare_count', tp_dontcare_count)
        # print('fp_dontcare_count', fp_dontcare_count)
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
