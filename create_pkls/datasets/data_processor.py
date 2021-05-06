from pathlib import Path
import numpy as np

from pcdet.utils import common_utils
from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pcdet.datasets.processor.data_processor import DataProcessor as PCDetDataProcessor

class DataProcessor():
    def __init__(self, dataset_cfg, root_path=None):
        self.dataset_cfg = dataset_cfg
        self.class_names = dataset_cfg.CLASS_NAMES
        self.root_path = Path(self.dataset_cfg.DATA_PATH) if root_path is None else root_path

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)

        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path,
            self.dataset_cfg.DATA_AUGMENTOR,
            self.class_names
        )
        self.train_data_processor = PCDetDataProcessor(
            self.dataset_cfg.DATA_PROCESSOR,
            point_cloud_range=self.point_cloud_range,
            training=True
        )
        self.eval_data_processor = PCDetDataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, 
            point_cloud_range=self.point_cloud_range, 
            training=False
        )

    # Ported from PCDet/datasets/dataset.py, prepare_data()
    def __call__(self, data_dict, split):
        if split == 'train':
            self.training = True
            self.data_processor = self.train_data_processor
        elif split == 'val' or split == 'test':
            self.training = False
            self.data_processor = self.eval_data_processor
        else:
            raise ValueError('Split is expected to be one of train, val or test.')

        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                # new_index = np.random.randint(self.__len__())
                # return self.__getitem__(new_index)
                return data_dict

        if data_dict.get('gt_boxes', None) is not None:
            # NOTE: in evaluation, we keep all annotations so other classes will not be filtered out
            if self.training:
                selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            else:
                selected = np.ones(len(data_dict['gt_boxes']), dtype=bool)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            # NOTE: in evaluation, if a class is not specified in class_names, then assign label -1
            gt_classes = np.array([self.class_names.index(n) + 1 if n in self.class_names else -1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        # data_dict.pop('gt_names', None)

        return data_dict