import logging
from pathlib import Path
import os
import pickle
import numpy as np

# from wise_alf.data import DataSource
from data_loader import DataSource, DataPool
from .data_processor import DataProcessor
from pcdet.datasets.cadc import cadc_calibration
from pcdet.utils import box_utils

class CADCPool(DataPool):
    def __init__(self, indices, data_source, dbinfos_stem, dbinfos_logger):
        super().__init__(indices)
        self.data_source = data_source
        self.root_path = Path(data_source.config.DATA_PATH)
        self.dbinfos_stem = dbinfos_stem
        self.dbinfos_logger = dbinfos_logger
        self.dbinfos_logger.set_logger(logging.getLogger(__name__))

    def mark_as_labeled(self, indices, cycle=None):
        super().mark_as_labeled(indices, cycle)

        # Get labeled frame_ids
        labeled_indices = {}
        for idx in self.indices[self.labels == 1]:
            info = self.data_source.train_split[idx]
            labeled_indices[ info['point_cloud']['lidar_idx'] ] = 1
        new_indices = {}
        for idx in indices:
            info = self.data_source.train_split[idx]
            print(info['point_cloud']['lidar_idx'])
            new_indices[ info['point_cloud']['lidar_idx'] ] = 1

        # Filter dbinfos
        with open(str(self.root_path / 'cadc_dbinfos_train.pkl'), 'rb') as f:
            dbinfos_all = pickle.load(f)
        dbinfos_labeled = {}
        for class_name, samples in dbinfos_all.items():
            dbinfos_labeled[class_name] = []
            for sample in samples:
                sample_frame_id = sample['image_idx'][0] + '_' + sample['image_idx'][1] + '_' + sample['image_idx'][2]
                if sample_frame_id in new_indices:
                    sample['new'] = True
                if sample_frame_id in labeled_indices:
                    dbinfos_labeled[class_name].append(sample)
            self.dbinfos_logger.info(f'DB objects ({class_name}): {len(dbinfos_labeled[class_name])}')

        with open(self.root_path / self.dbinfos_stem, 'wb') as f:
            pickle.dump(dbinfos_labeled, f)

        with open(self.dbinfos_logger.get_log_path(), 'wb') as f:
            pickle.dump(dbinfos_labeled, f)

# This class is ported from PCDet/datasets/cadc/cadc_dataset.py
class CADC(DataSource):
    def __init__(self, dataset_cfg, processor=lambda x: x):
        super().__init__(name=dataset_cfg.DATASET)
        self.dataset_cfg = self.config = dataset_cfg
        self.data_processor = DataProcessor(self.config, Path(self.config.DATA_PATH))
        self.root_path = Path(self.dataset_cfg.DATA_PATH)
        self.train_split = self.include_cadc_data(mode='train')
        self.test_split = self.include_cadc_data(mode='test')

    # Ported from include_cadc_data()
    def include_cadc_data(self, mode):
        """ Load metadata from pkl files, called during class initialization
        Args:
            mode (str): 'train' or 'test'
        Returns:
            meta_data (numpy array)
        """
        print('Loading CADC dataset - {}'.format(mode))
        cadc_infos = []

        for info_path in self.config.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                raise NameError('File {} does not exist'.format(info_path))
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                cadc_infos.extend(infos)

        print('Total samples for CADC dataset: %d' % (len(cadc_infos)))
        return np.array(cadc_infos)
    
    # Directly (no change) ported from get_lidar
    def get_lidar(self, sample_idx):
        date, set_num, idx = sample_idx
        lidar_file = os.path.join(self.root_path, date, set_num, 'labeled', 'lidar_points', 'data', '%s.bin' % idx)
        assert os.path.exists(lidar_file)
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        return points

    # Directly (no change) ported from get_calib
    def get_calib(self, sample_idx):
        date, set_num, idx = sample_idx
        calib_path = os.path.join(self.root_path, date, 'calib')
        assert os.path.exists(calib_path)
        return cadc_calibration.Calibration(calib_path)
    
    # Directly (no change) ported from get_fov_flag
    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        '''
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param img_shape:
        :return:
        '''
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        return pts_valid_flag
        
    # Ported from __getitem()
    def prepare_data(self, info, split=None):
        """ This function involves two steps:
                Convert metadata to the actual data.
                Convert actual data to the format used by the model.
        Args: 
            info: a *single* datapoint as the metadata
        Returns:
            actual_data: in the format that the model needs, the actual_data's format
            should be independent of the dataset used
        """
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)

        img_shape = info['image']['image_shape']
        if self.config.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]
        
        input_dict = {
            'points': points,
            'sample_idx': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']

            # Create mask to filter annotations during training
            # if self.training and self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
            #     mask = (annos['num_points_in_gt'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            # else:
            # We are always testing in this code
            mask = None

            gt_names = annos['name'] if mask is None else annos['name'][mask]
            if 'gt_boxes_lidar' in annos:
                gt_boxes_lidar = annos['gt_boxes_lidar'] if mask is None else annos['gt_boxes_lidar'][mask]
            else:
                # This should not run, although the code should look somewhat like this
                raise NotImplementedError
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

            data_dict = self.data_processor(data_dict=input_dict, split=split)

            data_dict['image_shape'] = img_shape

        return data_dict       