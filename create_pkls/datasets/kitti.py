import logging
from pathlib import Path
import pickle
import numpy as np

# from wise_alf.data import DataSource, DataPool
from data_loader import DataSource, DataPool
from .data_processor import DataProcessor
from pcdet.utils import box_utils, calibration_kitti, common_utils

class KITTIPool(DataPool):
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
            new_indices[ info['point_cloud']['lidar_idx'] ] = 1

        # Filter dbinfos
        with open(str(self.root_path / 'kitti_dbinfos_train.pkl'), 'rb') as f:
            dbinfos_all = pickle.load(f)
        dbinfos_labeled = {}
        for class_name, samples in dbinfos_all.items():
            dbinfos_labeled[class_name] = []
            for sample in samples:
                if sample['image_idx'] in new_indices:
                    sample['new'] = True
                if sample['image_idx'] in labeled_indices:
                    dbinfos_labeled[class_name].append(sample)
            self.dbinfos_logger.info(f'DB objects ({class_name}): {len(dbinfos_labeled[class_name])}')

        with open(self.root_path / self.dbinfos_stem, 'wb') as f:
            pickle.dump(dbinfos_labeled, f)

        with open(self.dbinfos_logger.get_log_path(), 'wb') as f:
            pickle.dump(dbinfos_labeled, f)


# This class is ported from PCDet/datasets/kitti/kitti_dataset.py
class KITTI(DataSource):
    def __init__(self, config, use_val=False):
        super().__init__(name=config.DATASET.lower())
        print('Loading KITTI datasete')

        self.dataset_cfg = self.config = config
        self.root_path = Path(config.DATA_PATH)
        official_train_split = self.include_kitti_data(mode='train')
        if use_val:
            idx = np.arange(len(official_train_split), dtype=int)
            np.random.seed(0)
            np.random.shuffle(idx)
            self.train_split = [ official_train_split[i] for i in idx[:3000] ]
            self.val_split = [ official_train_split[i] for i in idx[3000:] ]
        else:
            self.train_split = official_train_split
        self.test_split = self.include_kitti_data(mode='test')

        print(f'Total samples for KITTI dataset: train {len(self.train_split)}, val {len(self.val_split)}, test {len(self.test_split)}')

    # Ported from include_kitti_data()
    def include_kitti_data(self, mode):
        """ Load metadata from pkl files, called during class initialization
        Args:
            mode (str): 'train', 'val' or 'test'
        Returns:
            meta_data (numpy array)
        """
        kitti_infos = []

        for info_path in self.config.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                raise NameError('File {} does not exist'.format(info_path))
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        return kitti_infos

    # Directly (no change) ported from get_lidar
    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    # Directly (no change) ported from get_calib
    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    # Directly (no change) ported from get_road_plane
    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    # Directly (no change) ported from get_fov_flag
    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def data(self, *args, **kwargs):
        self.data_processor = DataProcessor(self.config, Path(self.config.DATA_PATH))
        return super().data(*args, **kwargs)

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
        # self.root_split_path = self.root_path / ('training' if split != 'test' else 'testing')
        self.root_split_path = self.root_path / 'training'

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)

        img_shape = info['image']['image_shape']
        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            # NOTE: in evaluation, we keep all annotations so DontCare boxes will not be filtered out.
            if split == 'train':
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        data_dict = self.data_processor(data_dict=input_dict, split=split)

        data_dict['image_shape'] = img_shape
        return data_dict
