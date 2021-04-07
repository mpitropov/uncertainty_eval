import os
from pprint import pprint

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from tqdm import tqdm

from detection import Detection
from filter import ClassFilter, RangeFilter, CombinedFilter

# Cluster predictions
from cluster import cluster_preds

# Scoring Rules
from nll_clf import NLLCLF
from nll_reg import NLLREG
from binary_brier_score import BINARYBRIERSCORE
from brier_score import BRIERSCORE
from dmm import DMM

dataset_path = '/home/matthew/git/cadc_testing/WISEOpenLidarPerceptron/data/kitti'
logdir = '/home/matthew/git/cadc_testing/al_output/pcdet_log_20_epoch/kittidatasetvar/random/eval_192e0b82f64b5c2207934a583985aa2e'
gts_path = os.path.join(logdir, 'gt.pkl')

# Clustering
logdir = '/home/matthew/git/cadc_testing/uncertainty_eval'
preds_path = os.path.join(logdir, 'result_converted.pkl')

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
        1: 0.7,    # Car
        2: 0.5,    # Pedestrian
        3: 0.5     # Cyclist
    }
    # Class filters
    car_filter = ClassFilter(name='Car', label=1,
                            gt_processor=gt_processor, pred_processor=pred_processor)
    ped_filter = ClassFilter(name='Pedestrian', label=2,
                            gt_processor=gt_processor, pred_processor=pred_processor)
    cyc_filter = ClassFilter(name='Cyclist', label=3,
                            gt_processor=gt_processor, pred_processor=pred_processor)
    filter_list = [car_filter, ped_filter, cyc_filter]

    for filter_idx in range(len(filter_list)):
        print("Performing evaluation for class", filter_list[filter_idx].name)
        # Evaluate over validation set
        # Construct empty list to store the results
        gt_list = Detection.BoxList()
        pred_list = Detection.BoxList()

        print("Loop through all dictionaries for each sample...")
        for gt_dict, pred_dict in tqdm(zip(gt_dicts, pred_dicts), total=len(gt_dicts)):
            # Get results for one sample
            gt_list_one, pred_list_one = Detection.evaluate_one_sample(
                gt_dict,
                pred_dict,
                thresholds,
                criterion='iou',
                epsilon=0.1,
                filta=filter_list[filter_idx],
                gt_processor=gt_processor,
                pred_processor=pred_processor
            )

            # Attach any extra data you need, e.g. frame_id, box coordinates, etc.
            # The data field needs to be an np array of dictinary/objects of the same size as other fields
            # i.e.  1) len( list.data ) = number of gt boxes
            #       2) list.data[i] = object or dict
            gt_list_one.data = np.array([gt_dict] * len(get_labels(gt_dict)), dtype=object)
            # Add scores, box coordinates and box variances
            data_list = []
            for i in range(len(pred_dict['score_all'])):
                data_list.append({
                    'score_all': pred_dict['score_all'][i],
                    'boxes_lidar': pred_dict['boxes_lidar'][i],
                    'pred_vars': pred_dict['pred_vars'][i]
                })
            pred_list_one.data = data_list

            # Aggregate the results by simple addition
            gt_list += gt_list_one
            pred_list += pred_list_one

        print("Evaluate Uncertainty...")
        # A prediction box is either a TP or FP
        # TP is both localized and classified
        tp = (~pred_list.ignored) & (pred_list.localized) & (pred_list.classified)
        # FP is the negative of TPs
        fp = (~tp)

        # Init Scoring Rules
        nll_clf_obj = NLLCLF()
        nll_reg_obj = NLLREG()
        binary_brier_obj = BINARYBRIERSCORE()
        brier_obj = BRIERSCORE()
        dmm_obj = DMM()

        # TODO: Need to get the gt box coordinates for NLL REG and DMM

        # TP loop
        for obj in pred_list[tp]:
            nll_clf_obj.add_tp(obj.pred_score)
        #     nll_reg_obj.add_tp(GT_TODO_HOW, obj.data['boxes_lidar'], obj.data['pred_vars'])
            binary_brier_obj.add_tp(obj.pred_score)
            brier_obj.add_tp(obj.pred_label, obj.data['score_all'])
        #     dmm_obj.add_tp(GT_TODO_HOW, obj.data['boxes_lidar'], obj.data['pred_vars'])

        # FP loop
        for obj in pred_list[fp]:
            nll_clf_obj.add_fp(obj.pred_score)
            binary_brier_obj.add_fp(obj.pred_score)

        print('NLL Classification mean', nll_clf_obj.mean())
        # print('NLL Regression mean', nll_reg_obj.mean())
        print('Binary Brier Score Classification mean', binary_brier_obj.mean())
        print('Brier Score Classification mean', brier_obj.mean())
        # print('DMM Regression mean', dmm_obj.mean())

main()
