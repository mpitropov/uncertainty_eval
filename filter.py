import numpy as np

class DetectionFilter():
    def __init__(
        self,
        name,
        gt_processor=None,
        pred_processor=None
    ):
        self.name = name
        self.gt_processor = gt_processor
        self.pred_processor = pred_processor
        if gt_processor is not None and not callable(gt_processor):
            raise ValueError('gt_processor must be callable')
        if pred_processor is not None and not callable(pred_processor):
            raise ValueError('pred_processor must be callable')

    def __setattr__(self, name, value):
        if name in ['gt_processor', 'pred_processor'] and \
                value is not None and not callable(value):
            raise ValueError(f'{name} must be callable')
        return super().__setattr__(name, value)

    def __str__(self):
        return self.name
    
    def __call__(self, gt, pred):
        if callable(self.gt_processor):
            gt = self.gt_processor(gt)
        if callable(self.pred_processor):
            pred = self.pred_processor(pred)
        return (
            self.get_ignored_gt(gt),
            self.get_ignored_pred(pred)
        )

    def get_ignored_gt(self, gt):
        raise NotImplementedError

    def get_ignored_pred(self, pred):
        raise NotImplementedError


class ClassFilter(DetectionFilter):
    def __init__(self, name, label, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.label = label

    def get_ignored_gt(self, gt):
        labels, boxes = gt
        return labels != self.label

    def get_ignored_pred(self, pred):
        labels, scores, boxes = pred
        return labels != self.label

def build_class_filters(names, labels, gt_processor, pred_processor):
    assert(len(names) == len(labels))
    return [ClassFilter(
        name, label,
        gt_processor,
        pred_processor) for name, label in zip(names, labels)]

class RangeFilter(DetectionFilter):
    def __init__(self, name, value_range, *args, **kwargs):
        super().__init__(
            name=f'{name}:{value_range[0]}-{value_range[1]}', *args, **kwargs)
        self.range = value_range
    
    def get_ignored_gt(self, gt):
        ignore_mask = np.isnan(gt)
        ret = (gt < self.range[0]) | (gt > self.range[1])
        ret[ignore_mask] = False
        return ret

    def get_ignored_pred(self, pred):
        ignore_mask = np.isnan(pred)
        ret = (pred < self.range[0]) | (pred > self.range[1])
        ret[ignore_mask] = False
        return ret


# class FrontVehicleFilter(DetectionFilter):
#     def __init__(self, name='front_veh', threshold=0.4, class_label=1, *args, **kwargs):
#         super().__init__(name=name, *args, **kwargs)
#         self.threshold = threshold
#         self.class_label = class_label
    
#     def is_front_vehicle(self, boxes):
#         angle = np.arctan2(boxes[:,1], boxes[:,0]) - boxes[:,6]
#         ret = (np.abs(angle) < self.threshold)
#         return ret
    
#     def get_ignored_gt(self, gt):
#         # Ignore any vehicle that is not considered front vehicle
#         labels, boxes = gt
#         return (labels == self.class_label) & ( ~self.is_front_vehicle(boxes) )

#     def get_ignored_gt(self, gt):
#         # Discard objects from other classes
#         labels, boxes = gt
#         return labels != self.class_label
    
#     def get_ignored_pred(self, pred):
#         labels, boxes = pred
#         return labels != self.class_label


class CombinedFilter(DetectionFilter):
    def __init__(self, filters, *args, **kwargs):
        super().__init__(name='_'.join([f.name for f in filters]), *args, **kwargs)
        self.filters = filters
    
    def get_ignored_gt(self, gt):
        ret = None
        for f in self.filters:
            gt_proc = f.gt_processor(gt) if callable(f.gt_processor) else gt
            mask = f.get_ignored_gt(gt_proc)
            ret = mask if ret is None else (ret | mask)
        return ret
    
    def get_ignored_pred(self, pred):
        ret = None
        for f in self.filters:
            pred_proc = f.pred_processor(pred) if callable(f.pred_processor) else pred
            mask = f.get_ignored_pred(pred_proc)
            ret = mask if ret is None else (ret | mask)
        return ret


# class KITTIFilter(DetectionFilter):
#     DIFFICULTY = ['easy', 'moderate', 'hard']
#     MIN_HEIGHT = [40, 25, 25]
#     MAX_OCCLUSION = [0, 1, 2]
#     MAX_TRUNCATION = [0.15, 0.3, 0.5]

#     def __init__(self, class_name='car', class_label=1, difficulty='moderate', infos=None, info_path=None, *args, **kwargs):
#         """ Detection filter for KITTI dataset.
#         Results will be filtered by class and difficulty levels
#         """
#         if type(difficulty) is int and 0 <= difficulty < 3:
#             pass
#         elif type(difficulty) is str:
#             difficulty = self.DIFFICULTY.index(difficulty)
#         else:
#             raise ValueError(f'invalid KITTI difficulty level: {difficulty}')

#         class_name = class_name.lower()
#         name = f'kitti_{class_name}_{self.DIFFICULTY[difficulty]}'
#         super().__init__(name, *args, **kwargs)
        
#         self.class_name = class_name
#         self.class_label = class_label
#         self.difficulty = difficulty

#         if (infos is None and info_path is None) or (infos is not None and info_path is not None):
#             raise ValueError('must provide either infos or info_path')
#         elif infos is not None:        
#             self.infos = infos
#         else:
#             self.infos = self.build_kitti_infos(info_path)
    
#     @staticmethod
#     def build_kitti_infos(info_path):
#         with open(info_path, 'rb') as f:
#             raw_infos = pickle.load(f)
        
#         # Build metadata database for annotations
#         # This will be used to calculate difficulty levels later
#         kitti_infos = {}
#         for info in raw_infos:
#             frame_id = info['point_cloud']['lidar_idx']
#             kitti_infos[frame_id] = []
#             annos = info['annos']
#             names = annos['name']
#             boxes = annos['gt_boxes_lidar']
#             occluded = annos['occluded']
#             truncated = annos['truncated']
#             bboxes = annos['bbox']
#             heights = bboxes[:,3] - bboxes[:,1]
#             num_points = annos['num_points_in_gt']
#             bbox_occlusion = annos['occlusion_level'] if 'occlusion_level' in annos else np.full(len(names), np.nan)
#             for name, box, o, t, h, n, bo in zip(names, boxes, occluded, truncated, heights, num_points, bbox_occlusion):
#                 kitti_infos[frame_id].append({
#                     'name': name,
#                     'loc': box[:2],
#                     'occluded': o,
#                     'truncated': t,
#                     'height': h,
#                     'num_points_in_gt': n,
#                     'bbox_occlusion': bo
#                 })
#         return kitti_infos

#     @classmethod
#     def check_difficulty(cls, difficulty, occluded, truncated, height):
#         return (occluded <= cls.MAX_OCCLUSION[difficulty] and
#                 truncated <= cls.MAX_TRUNCATION[difficulty] and
#                 height > cls.MIN_HEIGHT[difficulty])

#     def get_ignored_gt(self, gt):
#         # Ignored boxes will not be counted towards FP if detected or FN if not detected
#         frame_id = gt['frame_id']
#         names = gt['gt_names']

#         ignored = np.zeros(len(names), dtype=bool)
#         for idx, name in enumerate(names):
#             name = name.lower()
#             if name == 'dontcare':
#                 ignored[idx] = True
#                 continue
#             annos = self.infos[frame_id][idx]
#             # Ignore don't care class
#             if name == 'dontcare':
#                 ignored[idx] = True
#             # Ignore similar classes
#             elif self.class_name == 'pedestrian' and name == 'person_sitting':
#                 ignored[idx] = True
#             elif self.class_name == 'car' and name == 'van':
#                 ignored[idx] = True
#             # If label is a different class, don't ignore
#             elif self.class_name != name:
#                 ignored[idx] = False
#             # Ignore same class but different difficulty
#             elif not self.check_difficulty(self.difficulty, annos['occluded'], annos['truncated'], annos['height']):
#                 ignored[idx] = True
#             else:
#                 ignored[idx] = False
#         return ignored
    
#     def get_ignored_gt(self, gt):
#         # Discarded boxes will not be counted towards FN if not detected,
#         # but will be counted as FP if detected
#         ignored = self.get_ignored_gt(gt)
#         labels = gt['gt_labels']
#         return (labels != self.class_label)

#     def get_ignored_pred(self, pred):
#         # Discarded prediction will not be regarded as positive
#         # i.e. will not be counted as either TP or FP
#         class_names = ['Car', 'Pedestrian', 'Cyclist']
#         labels = np.array([class_names.index(n)+1 for n in pred['name']])
#         return labels != self.class_label


# def build_kitti_filters(info_path, class_names=['Car', 'Pedestrian', 'Cyclist'], class_labels=[1, 2, 3]):
#     kitti_infos = KITTIFilter.build_kitti_infos(info_path)
#     filters = []
#     for name, label in zip(class_names, class_labels):
#         for difficulty in ['moderate']:
#             filters.append( KITTIFilter(name, label, difficulty, kitti_infos) )
#     return filters, kitti_infos

# def build_kitti_range_filters(name, info_path, start=0, stop=45, step=10, class_names=['Car', 'Pedestrian', 'Cyclist'], class_labels=[1, 2, 3], gt_processor=None, pred_processor=None):
#     kitti_filters, kitti_infos = build_kitti_filters(info_path, class_names, class_labels)
#     ranges = np.arange(start, stop, step)
#     filters = []
#     for kitti_filter in kitti_filters:
#         for i in range(len(ranges)-1):
#             range_filter = RangeFilter(name, ranges[i:i+2], gt_processor=gt_processor, pred_processor=pred_processor)
#             filters.append( CombinedFilter([kitti_filter, range_filter]) )
#     return filters, kitti_infos