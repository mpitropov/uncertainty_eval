from pathlib import Path
from functools import partial
import pickle
from tqdm import tqdm
import numpy as np

# from ..metric import Metric
# from metric import Metric

# __all__ = ['KITTIObjectEval', 'NuScenesObjectEval', 'Detection']

# class KITTIObjectEval(Metric):
#     def __init__(self, class_names=['Car', 'Pedestrian', 'Cyclist'], annos=None, name='kitti_object', *args, **kwargs):
#         self.class_names = class_names
#         super().__init__(name=name, config={
#             'annos': annos,
#             'class_names': class_names
#         }, *args, **kwargs)

#     def keys(self):
#         metrics = ['3d', 'aos', 'bev', 'image']
#         difficultys = ['easy', 'moderate', 'hard']
#         keys = []
#         for c in self.class_names:
#             for m in metrics:
#                 for d in difficultys:
#                     keys.append(f'{c}_{m}/{d}_R40')
#         return keys

#     @staticmethod
#     def evaluate(preds, gt, name='kitti_object', config=None):
#         import copy
#         from .kitti_object_eval_python import eval as kitti_eval

#         if config is None or 'class_names' not in config:
#             class_names = ['Car', 'Pedestrian', 'Cyclist']
#         else:
#             class_names = config['class_names']

#         if config is None or 'annos' not in config:
#             annos = gt
#         else:
#             annos = [copy.deepcopy(info) for info in config['annos']]

#         _, results = kitti_eval.get_official_eval_result(annos, preds, class_names)
#         return results


# class NuScenesObjectEval(Metric):
#     nusc_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

#     def __init__(self, root_path, output_path='/tmp', version='v1.0-trainval', verbose=False, name='nuscenes_eval', *args, **kwargs):
#         super().__init__(name=name, config={
#             'root_path': root_path,
#             'output_path': output_path,
#             'version': version,
#             'verbose': verbose
#         }, *args, **kwargs)

#     def keys(self):
#         keys = ['eval_time']
#         for metric in ['ap', 'attr_err', 'orient_err', 'scale_err', 'trans_err', 'vel_err']:
#             keys += [ metric+'_'+c for c in NuScenesObjectEval.nusc_classes ]
#         return keys

#     @classmethod
#     def evaluate(cls, preds, gt, name='nuscenes_eval', config=None):
#         import json
#         from nuscenes.nuscenes import NuScenes
#         from nuscenes.eval.detection.config import config_factory
#         from nuscenes.eval.detection.evaluate import NuScenesEval as _NuScenesEval
#         from pcdet.datasets.nuscenes import nuscenes_utils
#         nusc = NuScenes(
#             version=config['version'],
#             dataroot=str(config['root_path']),
#             verbose=config['verbose'] if 'verbose' in config else False
#         )
#         nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(preds, nusc)
#         nusc_annos['meta'] = {
#             'use_camera': False,
#             'use_lidar': True,
#             'use_radar': False,
#             'use_map': False,
#             'use_external': False,
#         }

#         output_path = Path(config['output_path'])
#         output_path.mkdir(exist_ok=True, parents=True)
#         res_path = str(output_path / 'results_nusc.json')
#         with open(res_path, 'w') as f:
#             json.dump(nusc_annos, f)

#         eval_set_map = {
#             'v1.0-mini': 'mini_val',
#             'v1.0-trainval': 'val',
#             'v1.0-test': 'test'
#         }
#         try:
#             eval_version = 'detection_cvpr_2019'
#             eval_config = config_factory(eval_version)
#         except:
#             eval_version = 'cvpr_2019'
#             eval_config = config_factory(eval_version)

#         nusc_eval = _NuScenesEval(
#             nusc,
#             config=eval_config,
#             result_path=res_path,
#             eval_set=eval_set_map[config['version']],
#             output_dir=str(output_path),
#             verbose=config['verbose'] if 'verbose' in config else False,
#         )
#         metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

#         with open(output_path / 'metrics_summary.json', 'r') as f:
#             metrics = json.load(f)

#         results = {
#             'eval_time': metrics['eval_time']
#         }

#         for c in NuScenesObjectEval.nusc_classes:
#             results['ap_'+c] = metrics['mean_dist_aps'][c]
#             for err in ['attr_err', 'orient_err', 'scale_err', 'trans_err', 'vel_err']:
#                 metrics['label_tp_errors'][c][err]

#         return results

# def pcdet_get_labels(data_dict):
#     if 'gt_labels' in data_dict:
#         return data_dict['gt_labels']
#     if 'name' in data_dict:
#         classes = ['Car', 'Pedestrian', 'Cyclist']
#         return np.array([classes.index(name)+1 for name in data_dict['name']])
#     raise ValueError()

# def pcdet_get_boxes(data_dict):
#     if 'boxes_lidar' in data_dict:
#         return data_dict['boxes_lidar']
#     if 'gt_boxes' in data_dict:
#         return data_dict['gt_boxes']
#     raise ValueError()

# def pcdet_get_labels_boxes(data_dict):
#     return ( pcdet_get_labels(data_dict), pcdet_get_boxes(data_dict) )

# def pcdet_get_scores(data_dict):
#     return data_dict['score']

# def pcdet_get_dist(data_dict):
#     boxes = pcdet_get_boxes(data_dict)
#     coords = boxes[:,:2]
#     dist = np.linalg.norm(coords, axis=1)
#     return dist

# def pcdet_get_occ(infos, data_dict):
#     frame_id = data_dict['frame_id']
#     boxes = pcdet_get_boxes(data_dict)
#     if 'boxes_lidar' in data_dict:
#         return np.full(len(boxes), np.nan)

#     names = data_dict['gt_names']
#     occ = np.zeros(len(boxes)) - 1
#     for idx, name in enumerate(names):
#         if name.lower() == 'dontcare':
#             continue
#         annos = infos[frame_id][idx]
#         occ[idx] = annos['bbox_occlusion']
#     return occ


# def pcdet_get_npoints(infos, data_dict):
#     frame_id = data_dict['frame_id']
#     boxes = pcdet_get_boxes(data_dict)
#     if 'boxes_lidar' in data_dict:
#         return np.full(len(boxes), np.nan)

#     names = data_dict['gt_names']
#     npoints = np.zeros(len(boxes)) - 1
#     for idx, name in enumerate(names):
#         if name.lower() == 'dontcare':
#             continue
#         annos = infos[frame_id][idx]
#         npoints[idx] = annos['num_points_in_gt']
#     return npoints


class Detection():
    class BoxInfo():
        def __init__(self, box_list, idx):
            self.box_list = box_list
            self.idx = idx
        
        @staticmethod
        def keys():
            return [
                'ignored', 'bg',
                'localized', 'loc_score',
                'classified', 'gt_label',
                'pred_label', 'pred_score',
                'data'
            ]

        def __getattr__(self, name):
            if name == 'box_list' or name == 'idx':
                raise AttributeError
            attr = getattr(self.box_list, name, None)
            if isinstance(attr, np.ndarray):
                return attr[self.idx]
            attr = getattr(self.box_list, name+'s', None)
            if isinstance(attr, np.ndarray):
                return attr[self.idx]
            raise AttributeError
            
        def __setattr__(self, name, value):
            if name == 'box_list' or name == 'idx':
                super().__setattr__(name, value)
                return
            attr = getattr(self.box_list, name, None)
            if isinstance(attr, np.ndarray):
                attr[self.idx] = value
                return
            attr = getattr(self.box_list, name+'s', None)
            if isinstance(attr, np.ndarray):
                attr[self.idx] = value
                return
        
        def __repr__(self):
            return repr(
                {attr_name: getattr(self, attr_name) for attr_name in self.keys() if attr_name != 'data'}
            )

    class BoxList():
        def __init__(self, n_boxes=0):
            self.n_boxes = n_boxes

            self.ignored = np.zeros(n_boxes, dtype=bool)

            # This is an indicator list that determines if the box
            # does not match any GT/pred
            self.bg = np.zeros(n_boxes, dtype=bool)

            # This is an indicator list that determines if the box is
            #   * Correctly localized
            #   * Mislocalized
            self.localized = np.zeros(n_boxes, dtype=bool)
            self.loc_scores = np.full(n_boxes, np.nan, dtype=float)

            # This is an indicator list that determines if the box is
            #   * Correctly classified
            #   * Misclassified
            self.classified = np.zeros(n_boxes, dtype=bool)
            self.gt_labels = np.full(n_boxes, -1, dtype=int)
            self.pred_labels = np.full(n_boxes, -1, dtype=int)
            self.pred_scores = np.full(n_boxes, np.nan, dtype=float)

            self.data = np.full(n_boxes, None, dtype=object)
        
        @staticmethod
        def keys():
            return [
                'ignored', 'bg',
                'localized', 'loc_scores',
                'classified', 'gt_labels',
                'pred_labels', 'pred_scores',
                'data'
            ]

        def __add__(self, other):
            ret = Detection.BoxList(self.n_boxes + other.n_boxes)
            for attr_name in self.keys():
                attr = getattr(ret, attr_name)
                attr[:self.n_boxes,...] = getattr(self, attr_name)
                attr[self.n_boxes:,...] = getattr(other, attr_name)
            return ret
        
        def __len__(self):
            return self.n_boxes

        def __getitem__(self, idx):
            if isinstance(idx, int):
                if idx < 0 or idx >= len(self):
                    raise IndexError
                return Detection.BoxInfo(self, idx)
            elif isinstance(idx, (list, slice, np.ndarray)):
                if isinstance(idx, np.ndarray) and (len(idx.shape) != 1) and \
                    (idx.dtype == bool and idx.shape[0] != self.n_boxes):
                    raise IndexError
                n_boxes = len(self.ignored[idx])
                ret = Detection.BoxList(n_boxes)
                for attr_name in self.keys():
                    setattr(ret, attr_name, getattr(self, attr_name)[idx])
                return ret
            raise IndexError
        
        def __repr__(self):
            from pprint import pformat
            return pformat([ info for info in self ])


    def __init__(self, criterion, thresholds, class_names, get_labels, get_boxes, get_scores, class_sims=[], filters=[], *args, **kwargs):
        # kitti_dist_filters, kitti_infos = build_kitti_range_filters(
        #     name='dist', info_path='/root/datasets/kitti/kitti_infos_val.pkl',
        #     start=0, stop=45, step=10,
        #     gt_processor=pcdet_get_dist, pred_processor=pcdet_get_dist
        # )
        # filters.extend(kitti_dist_filters)
        # get_npoints = partial(pcdet_get_npoints, kitti_infos)
        # kitti_points_filters, kitti_infos = build_kitti_range_filters(
        #     name='npoints', info_path='/root/datasets/kitti/kitti_infos_val.pkl',
        #     start=0, stop=1001, step=100,
        #     gt_processor=get_npoints, pred_processor=get_npoints
        # )
        # filters.extend(kitti_points_filters)
        # front_vehicle_filter = CombinedFilter([
        #     FrontVehicleFilter(gt_processor=pcdet_get_labels_boxes, pred_processor=pcdet_get_labels_boxes),
        #     RangeFilter(name='dist', value_range=[0, 20], gt_processor=pcdet_get_dist, pred_processor=pcdet_get_dist)
        # ])
        # filters.append(front_vehicle_filter)
        # front_vehicle_filter = CombinedFilter([
        #     FrontVehicleFilter(gt_processor=pcdet_get_labels_boxes, pred_processor=pcdet_get_labels_boxes),
        #     RangeFilter(name='dist', value_range=[0, 10], gt_processor=pcdet_get_dist, pred_processor=pcdet_get_dist)
        # ])
        # filters.append(front_vehicle_filter)

        # kitti_filters, kitti_infos = build_kitti_filters('/root/datasets/kitti/kitti_infos_occlusion_val.pkl')
        # filters += kitti_filters

        # get_occ = partial(pcdet_get_occ, kitti_infos)
        # unocc_ped_filter = CombinedFilter([
        #     KITTIFilter(class_name='pedestrian', class_label=2, infos=kitti_infos),
        #     RangeFilter(name='dist', value_range=[0, 20], gt_processor=pcdet_get_dist, pred_processor=pcdet_get_dist),
        #     RangeFilter(name='occ', value_range=[0.0, 0.0], gt_processor=get_occ, pred_processor=get_occ)
        # ])
        # filters.append(unocc_ped_filter)

        # unocc_cyc_filter = CombinedFilter([
        #     KITTIFilter(class_name='cyclist', class_label=3, infos=kitti_infos),
        #     RangeFilter(name='dist', value_range=[0, 20], gt_processor=pcdet_get_dist, pred_processor=pcdet_get_dist),
        #     RangeFilter(name='occ', value_range=[0.0, 0.0], gt_processor=get_occ, pred_processor=get_occ)
        # ])
        # filters.append(unocc_cyc_filter)
        
        # front_vehicle_filter = CombinedFilter([
        #     FrontVehicleFilter(gt_processor=pcdet_get_labels_boxes, pred_processor=pcdet_get_labels_boxes),
        #     RangeFilter(name='dist', value_range=[0, 20], gt_processor=pcdet_get_dist, pred_processor=pcdet_get_dist),
        #     RangeFilter(name='occ', value_range=[0.0, 0.0], gt_processor=get_occ, pred_processor=get_occ)
        # ])
        # filters.append(front_vehicle_filter)

        super().__init__(name='detection_stats', config={
            'criterion': criterion,
            'thresholds': thresholds,
            'ignore_label': -1,
            'class_names': class_names,
            'class_sims': [],
            'filters': filters,
            'get_labels': get_labels,
            'get_boxes': get_boxes,
            'get_scores': get_scores
        }, *args, **kwargs)


    def keys(self):
        keys = []
        for metric in self.metrics:
            for filta in self.config['filters']:
                keys.append(f'{metric}_{filta.name}')
        print(keys)
        return keys

    @staticmethod
    def compute_ctable(boxes_a, boxes_b, criterion):
        """ Compute matching values based on criterion between each pair of boxes from boxes_a and boxes_b

        Args:
            boxes_a (np.ndarray): 2D boxes: [N_a, x, y, w, h, ...], 3D boxes: [N_a, x, y, z, w, l, h, ...]
            boxes_b (np.ndarray): 2D boxes: [N_b, x, y, w, h, ...], 3D boxes: [N_b, x, y, z, w, l, h, ...]
            criterion (str): one of ['iou', 'iou_2d', 'iou_3d', 'iou_bev'] or scipy.spatial.distance.cdist metrics

        Returns:
            np.ndarray: [N_a, N_b] matrix where each entry corresponds to a maching value
        """
        assert(boxes_a.shape[1] == boxes_b.shape[1])

        cdist_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule']

        if criterion == 'iou' or criterion == 'iou_2d':
            import torch
            from mmcv.ops import box_iou_rotated
            if boxes_a.shape[1] == 5:
                pass
            if boxes_a.shape[1] == 7:
                boxes_a = boxes_a[:,[0,1,3,4,6]]
                boxes_b = boxes_b[:,[0,1,3,4,6]]
            boxes_a_cuda = torch.from_numpy(boxes_a).float().cuda()
            boxes_b_cuda = torch.from_numpy(boxes_b).float().cuda()
            return box_iou_rotated(boxes_a_cuda, boxes_b_cuda).cpu().numpy()
        if criterion == 'iou_3d':
            import torch
            from wise_alf.utils.bbox3d import boxes_iou3d_gpu
            boxes_a_cuda = torch.from_numpy(boxes_a).float().cuda()
            boxes_b_cuda = torch.from_numpy(boxes_b).float().cuda()
            return boxes_iou3d_gpu(boxes_a_cuda, boxes_b_cuda).cpu().numpy()
        if criterion == 'iou_bev':
            raise NotImplementedError
        if criterion in cdist_metrics:
            from scipy.spatial.distance import cdist
            return cdist(boxes_a[:,:2], boxes_b[:,:2], metric=criterion)

    @classmethod
    def evaluate_one_sample(cls, gt, pred, thresholds, criterion='iou', epsilon=0.1,
        filta=None, ctable=None, gt_processor=None, pred_processor=None):
        """ Evaluate one sample

        Args:
            gt (any): ground truths for the current sample. If not in (labels, boxes) format, then `gt_processor` is required.
            pred (any): predictions for the current sample. If not in (labels, scores, boxes) format, then `pred_processor` is required.
            thresholds (list or dict): matching thresholds for different classes.
            criterion (str, optional): box matching criterion. Defaults to 'iou'.
            epsilon (float, optional): minimum matching threshold. Defaults to 0.1.
            filta (DetectionFilter, optional): filter object for detection evaluation. Defaults to None.
            ctable (ndarray, optional): pairwise distance table between the gt and pred boxes. Defaults to None.
            gt_processor (callable, optional): a function that transforms `gt`into (labels, boxes) format. Defaults to None.
            pred_processor (callable, optional): a function that transforms `pred` into (labels, scores, boxes) format. Defaults to None.

        Returns:
            tuple: (gt_list, pred_list)
        """
    

        if 'iou' in criterion or 'iof' in criterion:
            better = np.greater
        else:
            better = np.less

        ignored_gt, ignored_pred = filta(gt, pred)

        # Extract ground truth labels and boxes
        gt_labels, gt_boxes = gt_processor(gt) if callable(gt_processor) else gt
        assert(len(gt_labels) == len(gt_boxes) == len(ignored_gt))

        # Extract predicted labels and boxes
        pred_labels, pred_scores, pred_boxes = pred_processor(pred) if callable(pred_processor) else pred
        assert(len(pred_scores) == len(pred_labels) == len(pred_boxes) == len(ignored_pred))

        # Compute matching scores (iou, distance, etc.) between each pair
        # of the predicted and ground truth boxes
        if ctable is None:
            ctable = cls.compute_ctable(gt_boxes, pred_boxes, criterion)

        gt_list = cls.BoxList(len(gt_labels))
        pred_list = cls.BoxList(len(pred_labels))
        
        ################################################################################ 
        # False negative loop
        # For each ground truth label/box, find the best matching prediction and
        # evaluate matching score.
        ################################################################################ 
        for gt_idx, gt_label in enumerate(gt_labels):
            gt_info = gt_list[gt_idx]
            gt_info.gt_label = gt_label

            if gt_label < 0 or ignored_gt[gt_idx]:
                gt_info.ignored = True
                continue

            # Best matching score and pred idx for same class
            best_match_sc = -np.inf if better is np.greater else np.inf
            best_pred_idx_sc = np.nan
            # Best matching score and pred idx considering all classes
            best_match_ac = -np.inf if better is np.greater else np.inf
            best_pred_idx_ac = np.nan

            for pred_idx, pred_label in enumerate(pred_labels):
                # NOTE: do not skip discarded predictions when calculating FN
                match = ctable[gt_idx, pred_idx]
                # Record best match if better than existing one
                if better(match, best_match_sc) and gt_label == pred_label:
                    best_match_sc = match
                    best_pred_idx_sc = pred_idx
                if better(match, best_match_ac):
                    best_match_ac = match
                    best_pred_idx_ac = pred_idx

            if better(best_match_sc, thresholds[gt_label]):
                # TP case, correctly classified, localized
                gt_info.localized = True
                gt_info.loc_score = best_match_sc
                gt_info.classified = True
                gt_info.pred_label = gt_label
                gt_info.pred_score = pred_scores[best_pred_idx_sc]
                ignored_pred[best_pred_idx_sc] = False
            else:
                # Not TP, check bounding boxes for all classes
                gt_info.bg = not better(best_match_ac, epsilon)
                gt_info.loc_score = best_match_ac
                if not gt_info.bg:
                    gt_info.localized = better(best_match_ac, thresholds[gt_label])
                    gt_info.classified = gt_label == pred_labels[best_pred_idx_ac]
                    gt_info.pred_label = pred_labels[best_pred_idx_ac]
                    gt_info.pred_score = pred_scores[best_pred_idx_ac]


        ################################################################################ 
        # False positive loop
        # For each predicted label/box, find the best matching GT and
        # evaluate matching score.
        ################################################################################ 
        for pred_idx, (pred_label, pred_score) in enumerate(zip(pred_labels, pred_scores)):
            pred_info = pred_list[pred_idx]
            pred_info.pred_label = pred_label
            pred_info.pred_score = pred_score

            if ignored_pred[pred_idx]:
                pred_info.ignored = True
                continue

            # Best matching score and pred idx for same class
            best_match_sc = -np.inf if better is np.greater else np.inf
            best_gt_idx_sc = np.nan
            # Best matching score and pred idx considering all classes
            best_match_ac = -np.inf if better is np.greater else np.inf
            best_gt_idx_ac = np.nan

            for gt_idx, gt_label in enumerate(gt_labels):
                if gt_label < 0:
                    continue
                # NOTE: do not skip discarded GTs when calculating FN
                match = ctable[gt_idx, pred_idx]
                # Record best match if better than existing one
                if better(match, best_match_sc) and gt_label == pred_label:
                    best_match_sc = match
                    best_gt_idx_sc = gt_idx
                if better(match, best_match_ac):
                    best_match_ac = match
                    best_gt_idx_ac = gt_idx

            if better(best_match_sc, thresholds[pred_label]):
                # TP case, correctly classified, localized
                pred_info.localized = True
                pred_info.loc_score = best_match_sc
                pred_info.classified = True
                pred_info.gt_label = pred_label
                if gt_list[best_gt_idx_sc].ignored:
                    pred_info.ignored = True
            else:
                # Not TP, check bounding boxes for all classes
                pred_info.bg = not better(best_match_ac, epsilon)
                pred_info.loc_score = best_match_ac
                if not pred_info.bg:
                    gt_label = gt_labels[best_gt_idx_ac]
                    pred_info.localized = better(best_match_ac, thresholds[gt_label])
                    pred_info.classified = gt_label == pred_label 
                    pred_info.gt_label = gt_label

        return (gt_list, pred_list)
    
    @classmethod
    def compute_statistics(cls, gt_list, pred_list, num_positions=100):
        gt_list = gt_list[~gt_list.ignored]
        pred_list = pred_list[~pred_list.ignored]

        # Sort by prediction scores (confidence)
        sorted_idx = np.argsort(pred_list.pred_scores)[::-1]
        pred_list = pred_list[sorted_idx]

        # Compute binary indicators and cummulative sums for TP and FP
        tp_mask = pred_list.localized & pred_list.classified
        tp_cum = np.cumsum(tp_mask)
        fp_cum = np.cumsum(~tp_mask)
        tp_count = np.sum(tp_mask)
        fn_count = np.sum(~(gt_list.localized & gt_list.classified))
        gt_count = tp_count + fn_count

        # Interpolate recall positions
        rec = tp_cum / gt_count
        prec = tp_cum / ( tp_cum + fp_cum )
        rec_interp = np.linspace(0, 1, num_positions+1)
        prec_interp = np.interp(rec_interp, rec, prec, right=0)

        prob_fn = fn_count / gt_count

        sigma_99 = 2.58 * np.sqrt( ( prob_fn * (1-prob_fn) ) / gt_count )
        prob_fn_bound = prob_fn + sigma_99

        results = {
            'rec': rec_interp.tolist(),
            'prec': prec_interp.tolist(),
            'ap': np.mean(prec_interp),
            'tp': int(tp_count),
            'fn': int(fn_count),
            'prob_fn': prob_fn,
            'prob_fn_bound': prob_fn_bound
        }

        # ################################################################################
        # # Compute statistics for FP categorization
        # ################################################################################
        # for fp_type in BoxType:
        #     if fp_type == BoxType.IGN or fp_type == BoxType.TP:
        #         continue
                
        #     # Ignore FP of different type and recompute cummulative sum
        #     fp_type_mask = fp_mask.copy()
        #     fp_type_mask[fp >= fp_type] = False
        #     fp_type_cum = np.cumsum(fp_type_mask)

        #     # Compute precision and AP for the current FP type
        #     prec_fp = tp_cum / ( tp_cum + fp_type_cum )
        #     prec_fp_interp = np.interp(rec_interp, rec, prec_fp, right=0)
        #     ap = np.mean(prec_fp_interp)

        #     results[f'fp_{fp_type.name.lower()}'] = fp_type_cum.tolist()
        #     results[f'prec_fp_{fp_type.name.lower()}'] = prec_fp_interp.tolist()
        #     results[f'ap_fp_{fp_type.name.lower()}'] = ap

        
        # ################################################################################
        # # Compute statistics for FN categorization
        # ################################################################################
        # for fn_type in BoxType:
        #     if fn_type == BoxType.IGN or fn_type == BoxType.TP:
        #         continue

        #     fn_type_count = np.sum((fn > BoxType.TP) & (fn < fn_type))

        #     rec_fn = tp_cum / ( tp_count + fn_type_count )
        #     prec_fn_interp = np.interp(rec_interp, rec_fn, prec, right=0)
        #     ap = np.mean(prec_fn_interp)

        #     results[f'fn_{fn_type.name.lower()}'] = ((tp_count + fn_type_count) - tp_cum).tolist()
        #     results[f'prec_fn_{fn_type.name.lower()}'] = prec_fn_interp.tolist()
        #     results[f'ap_fn_{fn_type.name.lower()}'] = ap

        return results


    # @classmethod
    # def evaluate(cls, preds, gts, thresholds, criterion='iou', filters=None, n_positions=100):
    #     """ Evaluate all samples

    #     Args:
    #         gts (list): a list of 2-tuples of the format `(gt_labels, gt_boxes)`
    #             where `gt_lables` is a list of labels
    #             and `gt_boxes` is a list of ndarray with one of the formats:
    #                 * 2D bbox: [x_center, y_center, width, height, rotation]
    #                 * 3D bbox: [x, y, z, w, h, l, rotation]
    #         preds (list): a list of 3-tuples of the format `(pred_labels, pred_scores, pred_boxes)`
    #             `pred_labels` and `pred_boxes` follows the format for `gts`
    #             `pred_scores` is a list/ndarray 
    #         thresholds (list/dict): a list or dict that maps labels to their detection threshold
    #         criterion (str, optional): evaluation criterion. Defaults to 'iou'.
    #         filters (list, optional): a list of detection filters. Defaults to None.
    #         n_positions (int, optional): number of positions when calculating AP. Defaults to 100.

    #     Returns:
    #         dict: a dictionary containing evaluation results
    #     """

    #     assert(len(gts) == len(preds))


    #     pbar = tqdm(total=len(filters)*len(gts), desc='Evaluating detection')

    #     results = {}
    #     ctables = {}
    #     for filta in filters:
    #         scores = np.array([])
    #         fp = np.array([])
    #         fn = np.array([])

    #         # Calcuate tp, fp, fn for each sample
    #         for sample_idx, (pred, gt) in enumerate(zip(preds, gts)):

    #             pred_boxes = get_boxes(pred)
    #             gt_boxes = get_boxes(gt)
    #             if sample_idx not in ctables:
    #                 ctables[sample_idx] = cls.compute_ctable(pred_boxes, gt_boxes, criterion)

    #             ret = cls.evaluate_one_sample(
    #                 pred=pred, gt=gt, 
    #                 filta=filta,
    #                 criterion=criterion,
    #                 thresholds=thresholds,
    #                 get_labels=get_labels,
    #                 get_boxes=get_boxes,
    #                 ctable = ctables[sample_idx]
    #             )

    #             # Aggregate information from each sample
    #             scores = np.append(scores, get_scores(pred))
    #             fp = np.append(fp, ret['fp'])
    #             fn = np.append(fn, ret['fn'])

    #             pbar.update()
            
    #         stats = cls.compute_statistics( scores, fp, fn, num_positions )
    #         for metric in cls.metrics:
    #             key = f'{metric}_{filta.name}'
    #             results[key] = stats[metric]


    #     # cls.plot(results, config)
    #     return results