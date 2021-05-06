import functools
from collections import defaultdict
from easydict import EasyDict
import numpy as np
from scipy import special

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# import wise_alf as alf
from pcdet.config import cfg, cfg_from_yaml_file

def parse_pcdet_cfg(cfg_file):
    """ This function will read and load PCDet yaml format configuration file
    and extend the model config by including necessary fields defined in the
    data config. In this way, the two configurations are decoupled and
    both self-contained.

    The following additional fields are included in the model config:
        * CLASS_NAMES
        * POINT_CLOUD_RANGE
        * NUM_POINT_FEATURES
        * VOXEL_SIZE
        * GRID_SIZE

    Args:
        cfg_file (str): configuration file path

    Returns:
        tuple: model config and data config, both in EasyDict format
    """
    config = cfg_from_yaml_file(cfg_file, cfg)

    dataset_cfg = config.DATA_CONFIG
    dataset_cfg.CLASS_NAMES = config.CLASS_NAMES

    model_cfg = EasyDict()
    model_cfg.CLASS_NAMES = config.CLASS_NAMES
    model_cfg.MODEL = config.MODEL
    model_cfg.OPTIMIZATION = config.OPTIMIZATION

    point_cloud_range = model_cfg.POINT_CLOUD_RANGE = np.array(dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
    model_cfg.NUM_POINT_FEATURES = len(dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list)
    voxel_size = None
    for processor in dataset_cfg.DATA_PROCESSOR:
        if processor.NAME == 'transform_points_to_voxels':
            voxel_size = processor.VOXEL_SIZE
            break
    if voxel_size is not None:
        model_cfg.VOXEL_SIZE = voxel_size
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
        model_cfg.GRID_SIZE = np.round(grid_size).astype(np.int64)
    return model_cfg, dataset_cfg

def collate_batch(batch_list, _unused=False):
    data_dict = defaultdict(list)
    for cur_sample in batch_list:
        for key, val in cur_sample.items():
            data_dict[key].append(val)
    batch_size = len(batch_list)
    ret = {}

    for key, val in data_dict.items():
        try:
            if key in ['voxels', 'voxel_num_points']:
                ret[key] = np.concatenate(val, axis=0)
            elif key in ['points', 'voxel_coords']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ['gt_boxes']:
                max_gt = max([len(x) for x in val])
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                ret[key] = batch_gt_boxes3d
            elif key in ['gt_names']:
                ret[key] = val
            else:
                ret[key] = np.stack(val, axis=0)
        except:
            print('Error in collate_batch: key=%s' % key)
            raise TypeError

    ret['batch_size'] = batch_size
    return ret

def process_gt_batch_dict(batch_dict):
    batch_size = batch_dict['batch_size']
    gt_dicts = []
    for index in range(batch_size):
        if batch_dict.get('batch_index', None) is not None:
            assert batch_dict['gt_boxes'].shape.__len__() == 2
            batch_mask = (batch_dict['batch_index'] == index)
        else:
            assert batch_dict['gt_boxes'].shape.__len__() == 3
            batch_mask = index

        gt_boxes = np.copy(batch_dict['gt_boxes'][batch_mask])
        if 'metadata' not in batch_dict:
            gt_boxes[:,2] -= gt_boxes[:,5]/2
        gt_labels = gt_boxes[:,-1].astype(int)
        valid_mask = gt_labels != 0
        gt_record = {
            'gt_boxes': gt_boxes[:,:7][valid_mask],
            'gt_labels': gt_labels[valid_mask],
        }
        if 'gt_names' in batch_dict:
            gt_names = batch_dict['gt_names'][batch_mask]
            if len(gt_record['gt_boxes']) == len(gt_names):
                gt_record['gt_names'] = gt_names
        if 'frame_id' in batch_dict:
            gt_record['frame_id'] = batch_dict['frame_id'][batch_mask]
        gt_dicts.append(gt_record)
    return gt_dicts


# def cluster_dist(x, cluster, scores, iou_table, *args, **kwargs):
#     rep_id = cluster[np.argmax(scores[cluster])]
#     ious = iou_table[x, rep_id]
#     return 1-ious

# def cluster_worker(samples, min_object_per_cluster):
#     clusters = []
#     for sample in tqdm(samples, desc=f'Cluster worker {mp.current_process().name}'):
#         scores = np.concatenate([ s['score'] for s in sample ])
#         boxes_lidar = np.concatenate([ s['boxes_lidar'] for s in sample ])

#         boxes_lidar = torch.from_numpy(boxes_lidar).float().cuda()
#         iou_table = alf.utils.bbox3d.boxes_iou3d_gpu(boxes_lidar, boxes_lidar).cpu().numpy()
# #         boxes_lidar = boxes_lidar.cpu().numpy()
#         sample_clusters = alf.utils.cluster.bsas(
#             range(boxes_lidar.shape[0]),
#             dist_threshold=0.95,
#             dist_func=functools.partial(cluster_dist, scores=scores, iou_table=iou_table)
#         )
#         clusters.append(list(filter(lambda c: len(c) >= min_object_per_cluster, sample_clusters)))
#     return clusters

# def post_nms_cluster(raw_preds, processes=4, min_object_per_cluster=10):
#     """

#     Args:
#         raw_preds (list): list of size (num_mcsamples, num_samples)
#     """
#     with mp.get_context('spawn').Pool(processes) as p:
#         # (num_samples, num_mcsamples)
#         samples_split = np.array_split(raw_preds, processes)
#         clusters = p.map(functools.partial(cluster_worker, min_object_per_cluster=min_object_per_cluster), samples_split)
#         return list(zip(raw_preds, sum(clusters, [])))


def extract_scores(clustered_preds):
    """

    Args:
        clustered_preds (zip):
    Returns:
        list: a list of size (num_samples, num_object, cluster_size, num_classes)
    """
    results = []
    for preds, clusters in clustered_preds:
        scores_all = np.concatenate([pred['score_all'] for pred in preds])
        scores_softmax = special.softmax(scores_all, axis=-1)
        results.append([scores_softmax[c] for c in clusters])
    return results


def extract_boxes(clustered_preds):
    """

    Args:
        clustered_preds (zip): [description]

    Returns:
        list: a list of size (num_samples, num_object, 7, cluster_size)
    """
    results = []
    for preds, clusters in clustered_preds:
        boxes = np.concatenate([pred['boxes_lidar'] for pred in preds])
        results.append([boxes[c].reshape(7,-1) for c in clusters])
    return results

def extract_vars(clustered_preds):
    """

    Args:
        clustered_preds (zip): [description]

    Returns:
        list: a list of size (num_samples, num_object, 7, cluster_size)
    """
    results = []
    for preds, clusters in clustered_preds:
        vars = np.concatenate([pred['pred_vars'] for pred in preds])
        results.append([vars[c].reshape(7,-1) for c in clusters])
    return results

def extract_scores_single(raw_preds):
    return [special.softmax(pred['score_all'], axis=-1) for pred in raw_preds]


def pcdet_get_labels(data_dict):
    """ Returns the labels (gt or pred) from a given data_dict

    Args:
        data_dict (dict): the data dict of a single frame
    
    Returns:
        np.ndarray: labels
    """
    if isinstance(data_dict, list):
        data_dict = data_dict[0]
    for label_key in ['labels', 'gt_labels', 'pred_labels']:
        if label_key in data_dict:
            return data_dict[label_key]
    if 'name' in data_dict:
        classes = ['Car', 'Pedestrian', 'Cyclist']
        return np.array([
            classes.index(c)+1 if c in classes else -1 for c in data_dict['name']
        ])
    raise ValueError()

def pcdet_get_scores(data_dict):
    """ Returns the predicted scores from a given data_dict.
    Note that this function does not return the full softmax distribution.

    Args:
        data_dict (dict): the data dict of a single frame
    
    Returns:
        np.ndarray: scores
    """
    if isinstance(data_dict, list):
        data_dict = data_dict[0]
    if 'score' in data_dict:
        return data_dict['score']

def pcdet_get_boxes(data_dict):
    """ Returns the boxes (gt or pred) from a given data_dict

    Args:
        data_dict (dict): the data dict of a single frame
    
    Returns:
        np.ndarray: boxes
    """
    if isinstance(data_dict, list):
        data_dict = data_dict[0]
    for box_key in ['gt_boxes', 'gt_lidar_boxes', 'boxes_lidar', 'box3d_lidar']:
        if box_key in data_dict:
            return data_dict[box_key]
    return ValueError()

def pcdet_gt_processor(data_dict):
    """ Returns a tuple of two arrays ( gt_labels, gt_boxes )

    Args:
        data_dict (dict): the data dict of a single frame
    
    Returns:
        tuple: ( gt_labels, gt_boxes )
    """
    return (
        pcdet_get_labels(data_dict),
        pcdet_get_boxes(data_dict)
    )

def pcdet_pred_processor(data_dict):
    """ Returns a tuple of three arrays ( labels, scores, boxes )

    Args:
        data_dict (dict): the data dict of a single frame
    
    Returns:
        tuple: ( labels, scores, boxes )
    """
    return (
        pcdet_get_labels(data_dict),
        pcdet_get_scores(data_dict),
        pcdet_get_boxes(data_dict)
    )

def pcdet_ensemble_collate_fn(outputs):
    ensemble_size = len(outputs)
    assert(ensemble_size > 1)
    n_samples = len(outputs[0][0])
    collated_preds = [ [None]*ensemble_size for _ in range(n_samples) ]
    for ensemble_idx, output in enumerate(outputs):
        gts, preds = output
        for sample_idx, pred in enumerate(preds):
            collated_preds[sample_idx][ensemble_idx] = pred
    return (gts, collated_preds)
