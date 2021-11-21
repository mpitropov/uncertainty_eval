import logging
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

def load_dicts(gts_path, preds_path):
    # Load gt and prediction data dict
    with open(gts_path, 'rb') as f:
        gt_dicts = pickle.load(f)
    with open(preds_path, 'rb') as f:
        pred_dicts = pickle.load(f)
    return gt_dicts, pred_dicts


def create_thresholds(threshold, selected_dataset):
    if selected_dataset == "KITTI":
        return {
            -1: 1,     # Just to not output error
            0: 1,      # Just to not output error
            1: threshold,    # Car
            2: threshold,    # Pedestrian
            3: threshold     # Cyclist
        }
    elif selected_dataset == "CADC":
        return {
            -1: 1,     # Just to not output error
            0: 1,      # Just to not output error
            1: threshold,    # Car
            2: threshold,    # Pedestrian
            3: threshold     # Pickup_Truck
        }

# Add points to predictions
def add_num_pts_full_dataset(selected_dataset, dataset_path, pred_dicts):
    batch_size = 1
    workers = 4

    # Have to switch paths to load yaml files
    pcdet_tools_path = '/root/pcdet/tools/'
    if selected_dataset == 'KITTI':
        cfg_file = '/root/pcdet/tools/cfgs/kitti_models/pointpillar_mimo_var_c.yaml'
    elif selected_dataset == 'CADC':
        cfg_file = '/root/pcdet/tools/cfgs/cadc_models/second_mimo_var_c.yaml'
    old_dir = os.getcwd()
    os.chdir(pcdet_tools_path)
    cfg_from_yaml_file(cfg_file, cfg)
    # cfg.DATA_CONFIG.DATA_PATH = dataset_path


    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False, workers=workers, logger=logger, training=False
    )

    print('Adding points to pred_dicts')
    for i in tqdm(range(len(test_set))):
        pred_dict_idx = i
        if len(pred_dicts[pred_dict_idx]['boxes_lidar']) == 0:
            continue

        if selected_dataset == 'KITTI':
            sample_idx = test_set.kitti_infos[i]['point_cloud']['lidar_idx']
        elif selected_dataset == 'CADC':
            sample_idx = test_set.cadc_infos[i]['point_cloud']['lidar_idx']
        # print(sample_idx)
        points = test_set.get_lidar(sample_idx)
        cuda_points = torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda()
        cuda_boxes = torch.from_numpy(pred_dicts[pred_dict_idx]['boxes_lidar']).unsqueeze(dim=0).float().cuda()
        # print(cuda_points.size())
        # print(cuda_boxes.size())
        pred_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
            cuda_points, cuda_boxes
        ).long().squeeze(dim=0).cpu().numpy()
        # print(pred_box_idxs_of_pts)
        pred_pts_list = []
        for id in range(len(pred_dicts[pred_dict_idx]['boxes_lidar'])):
            box_pts = points[pred_box_idxs_of_pts == id]
            pred_pts_list.append(len(box_pts))
        # print(pred_pts_list)
        pred_dicts[pred_dict_idx]['num_pts_in_pred'] = pred_pts_list

    os.chdir(old_dir)

# Add points to predictions
def add_num_pts(selected_dataset, dataset_path, pred_dicts):
    batch_size = 1
    workers = 4

    # Have to switch paths to load yaml files
    pcdet_tools_path = '/root/pcdet/tools/'
    if selected_dataset == 'KITTI':
        cfg_file = '/root/pcdet/tools/cfgs/kitti_models/pointpillar_mimo_var_c.yaml'
    elif selected_dataset == 'CADC':
        cfg_file = '/root/pcdet/tools/cfgs/cadc_models/second_mimo_var_c.yaml'
    old_dir = os.getcwd()
    os.chdir(pcdet_tools_path)
    cfg_from_yaml_file(cfg_file, cfg)
    # cfg.DATA_CONFIG.DATA_PATH = dataset_path


    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False, workers=workers, logger=logger, training=False
    )

    print("Using SECOND half of val set as the eval set to add points")
    half_point = int(len(test_set)/2)

    print('Adding points to pred_dicts')
    for i in tqdm(range(len(test_set))):
        if i < half_point:
            continue
        pred_dict_idx = i-half_point
        if len(pred_dicts[pred_dict_idx]['boxes_lidar']) == 0:
            continue

        if selected_dataset == 'KITTI':
            sample_idx = test_set.kitti_infos[i]['point_cloud']['lidar_idx']
        elif selected_dataset == 'CADC':
            sample_idx = test_set.cadc_infos[i]['point_cloud']['lidar_idx']
        # print(sample_idx)
        points = test_set.get_lidar(sample_idx)
        cuda_points = torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda()
        cuda_boxes = torch.from_numpy(pred_dicts[pred_dict_idx]['boxes_lidar']).unsqueeze(dim=0).float().cuda()
        # print(cuda_points.size())
        # print(cuda_boxes.size())
        pred_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
            cuda_points, cuda_boxes
        ).long().squeeze(dim=0).cpu().numpy()
        # print(pred_box_idxs_of_pts)
        pred_pts_list = []
        for id in range(len(pred_dicts[pred_dict_idx]['boxes_lidar'])):
            box_pts = points[pred_box_idxs_of_pts == id]
            pred_pts_list.append(len(box_pts))
        # print(pred_pts_list)
        pred_dicts[pred_dict_idx]['num_pts_in_pred'] = pred_pts_list

    os.chdir(old_dir)

# Filter pred_dicts using score threshold
def filter_pred_dicts(pred_dicts, SCORE_THRESHOLD):
    new_pred_dicts = []

    for frame_dict in tqdm(pred_dicts):
        final_name_list = []
        final_label_list = []
        final_score_list = []
        final_score_all_list = []
        final_shannon_entropy_list = []
        final_aleatoric_entropy_list = []
        final_mutual_info_list = []
        final_epistemic_total_var_list = []
        final_aleatoric_total_var_list = []
        final_box_list = []
        final_var_list = []
        final_cluster_size_list = []

        for i in range(len(frame_dict['score'])):
            if frame_dict['score'][i] < SCORE_THRESHOLD:
                continue

            final_name_list.append(frame_dict['name'][i])
            final_label_list.append(frame_dict['pred_labels'][i])
            final_score_list.append(frame_dict['score'][i])
            final_score_all_list.append(frame_dict['score_all'][i])
            final_shannon_entropy_list.append(frame_dict['shannon_entropy'][i])
            final_aleatoric_entropy_list.append(frame_dict['aleatoric_entropy'][i])
            final_mutual_info_list.append(frame_dict['mutual_info'][i])
            final_epistemic_total_var_list.append(frame_dict['epistemic_total_var'][i])
            final_aleatoric_total_var_list.append(frame_dict['aleatoric_total_var'][i])
            final_box_list.append(frame_dict['boxes_lidar'][i])
            final_var_list.append(frame_dict['pred_vars'][i])
            final_cluster_size_list.append(frame_dict['cluster_size'][i])

        final_name_list = np.array(final_name_list)
        final_label_list = np.array(final_label_list)
        final_score_list = np.array(final_score_list)
        final_score_all_list = np.array(final_score_all_list)
        final_shannon_entropy_list = np.array(final_shannon_entropy_list)
        final_aleatoric_entropy_list = np.array(final_aleatoric_entropy_list)
        final_mutual_info_list = np.array(final_mutual_info_list)
        final_epistemic_total_var_list = np.array(final_epistemic_total_var_list)
        final_aleatoric_total_var_list = np.array(final_aleatoric_total_var_list)
        final_box_list = np.array(final_box_list)
        final_var_list = np.array(final_var_list)
        final_cluster_size_list = np.array(final_cluster_size_list)
        # if tracking_mode:
        #     final_bbox_list = np.array(final_bbox_list)
        #     final_location_list = np.array(final_location_list)
        #     final_dimensions_list = np.array(final_dimensions_list)
        #     final_rotation_y_list = np.array(final_rotation_y_list)
        #     final_alpha_list = np.array(final_alpha_list)

        # Add mean output for the frame
        seq_id = None
        # if tracking_mode:
        #     seq_id = frame_dict_list[0]['seq_id']

        new_pred_dict = {
            'frame_id': frame_dict['frame_id'],
            'seq_id': seq_id,
            'name': final_name_list,
            'pred_labels': final_label_list,
            'score': final_score_list,
            'score_all': final_score_all_list,
            'shannon_entropy': final_shannon_entropy_list,
            'aleatoric_entropy': final_aleatoric_entropy_list,
            'mutual_info': final_mutual_info_list,
            'epistemic_total_var': final_epistemic_total_var_list,
            'aleatoric_total_var': final_aleatoric_total_var_list,
            'boxes_lidar': final_box_list,
            'pred_vars': final_var_list,
            'cluster_size': final_cluster_size_list
        }
        # if tracking_mode:
        #     new_pred_dict['bbox'] = final_bbox_list
        #     new_pred_dict['location'] = final_location_list
        #     new_pred_dict['dimensions'] = final_dimensions_list
        #     new_pred_dict['rotation_y'] = final_rotation_y_list
        #     new_pred_dict['alpha'] = final_alpha_list
        new_pred_dicts.append(new_pred_dict)

    return new_pred_dicts

import matplotlib.pyplot as plt

# Plot confidence vs number of points in prediction
def plot_conf_pts(tp_pred_list, fp_pred_list, model_name, class_name):
    tp_confs = [obj.pred_score for obj in tp_pred_list]
    tp_pts = [obj.data['num_pts_in_pred'] for obj in tp_pred_list]
    fp_confs = [obj.pred_score for obj in fp_pred_list]
    fp_pts = [obj.data['num_pts_in_pred'] for obj in fp_pred_list]

    plt.scatter(tp_pts, tp_confs, c="green", alpha=1.0)
    plt.title(model_name + ' Class:' +  class_name + ' TP confidence vs points')
    plt.savefig('images/tp_pts_vs_conf_' + class_name + '.png')
    plt.close()

    plt.scatter(fp_pts, fp_confs, c="red", alpha=1.0)
    plt.title(model_name + ' Class:' +  class_name + ' FP_BG confidence vs points')
    plt.savefig('images/fp_bg_pts_vs_conf_' + class_name + '.png')
    plt.close()

# Plot PR curve
# https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248
def plot_pr_curve(precision_scores, recall_scores, model_name, class_name):
    print('precision_scores', precision_scores)
    print('recall_scores', recall_scores)

    plt.plot(recall_scores, precision_scores)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(model_name + ' Class:' +  class_name + ' PR Curve')
    plt.savefig('images/pr_curve_' + class_name + '.png')
    plt.close()
