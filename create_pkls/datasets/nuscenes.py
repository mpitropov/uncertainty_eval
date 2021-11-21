from pathlib import Path
import pickle
import numpy as np

# NuScenes has not been tested

from data_loader import DataSource, DataPool
from .data_processor import DataProcessor

val_scenes = {
    'v1.0-mini': ['bebf5f5b2a674631ab5c88fd1aa9e87a', 'de7d80a1f5fb4c3e82ce8a4f213b450a'],
    'v1.0-trainval':
        ['2121cc12888e47ce856dc5391f202594', 'fe4fdd7a28754baeac7074ad78f55a52',
        'd90b94e8bfd446cd9407f48665122268', 'da783af4b5124333911684dbaf31d969',
        '4f18d9a7ed374a0fb93c026589dcf9a0', '1891b66d9f58463399aa242f3b521232',
        '4098aaf3c7074e7d87285e2fc95369e0', 'ab7f4909bc6c4841a12020355a7d505c',
        '88084e0150d64dd1949a9cf1493e5188', 'ff6af17f52c34e9c9958c64e41fb3786',
        '268099669c954f429087081530094337', 'c610d0ac2e2045a5972bda66ac42ede3',
        'a99120daccb24bcd941b33e6e03bf718', 'a2e8f126913b4f72909e06d881dc5a8b',
        'f68ef41fbd1142d3a74f71be605a2227', '13c6a31ab85547b08dab2544024553a8',
        'a2e3c0a763c04e56a5e26611eb69395b', '5c9bd7ead37e4aa9989f7909f3a78baa',
        'f8ef029224b84d14843db98a76a7f4a1', '4431d97ea17044ad9c09c13d16684054',
        'cfa36eca40364e5bb15f550a077a21c5', '8c45a7a9142f4f848d463cc46843db71',
        '567fcd99d0dc4fa088f52b047f4ebdcf', '865c607c8ef44f13b39744a0de110740',
        '94f224a2a36c4bbbab59120cc6c31f6e', 'ab5bc0504e2c4103ab306c5bd18f6791',
        'bb73edb93a0a46c4be997c576e9beb61', '827712f3e79340d8878de0202c5b3960',
        '49eb09ab4c4142268131125d6e619a0c', '26540bfbab79463cb1ba76b52ec6013b',
        '9ac6dff089a84a6d8f066122443a10db', '5e270cfc71714e2bb94bad445c6159a4',
        'ca6abd081eaf48689f06b5e8fcc9d369', '634e7fbfe29c4a72b1ceb692b1d2ab44',
        'bed8426a524d45afab05b19cf02386b2', '857922568e4e4fb89c3ee3010eb93c7c',
        '1d4db80d13f342aba4881b38099bc4b7', '5560a973257e407b8d5bf9fb92b1e0f3',
        'c77e6ecf108c4c8b80d35d84642c805f', 'add2303ca41d445c8339e05769745fc0',
        'a5003bf9af8545a3a4c7f1f5604e8d3b', '48ba943c3d19463a81281bf6a7078eac',
        '2c1f2e5096434cd78919b2ae442d4ba4', '881da81bb35a4bdb925e0a2884ee7f0f',
        'cb0cd06a1dd84271968466d7b65e48f8', 'f69f7b4cefb7493bb1a8185f6f01d137',
        '34a9823fac0d4b9db30898e00b5f9f9c', 'aa54277d163d419b9f952b014112643e',
        'a8bd4c1efe284a5bba46f59454764965', 'd3d94f2ce3dc4db4b3ba6f4aa81c3987',
        '9771da5dd7bb4657be34365619e39dab', '62696ecd66a3445da436ce2d87ee3501',
        '88abbe4f3b37466791b2ae58490044d9', '73d9a36f20594e658020ebfc5b0ba74a',
        '90d616aaefc4429380388205ebb23398', 'cf10376382b74af1975be86f7190df20',
        'c9c28de6e66442f7921858f1ebbec3ef', '8461c1125eaf46c7abaf4ee18e8c4ee6',
        '3d42cd90a8184d9ebd3bb85b6416314d', '1c89941a6935484182ca19eddcd3bc77',
        '7ad94b6e54f641208bf59cba6bb55220', 'c74a3a7265ae45f99f65628fbd7cd2a4',
        'a1ca2ba59ac9452fb3da60019bf32c71', '9047b53fd41540649dce014a128cbe1b',
        '1f23540a109243438a5e3bd47e70105d', '36b7b02f0f034f0595e3437a85554151',
        'fc9068b303e448a6bbcefd37204386a9', '6e81ee0f64274490a403bbd6482c2bf9',
        '295c6c85275e4376bc8446b4a76504cd', 'e02aee6f1fec4d13ad77044466da5fb4',
        '35c3bc100e4b4e5089820986d3f50fa3', '1ec0b9bc3eca4b76b2610c519f843762',
        'fc61f52dcc1b4def95a278665d23af00', 'd865cb18632040fdb4cb06b6e3331cdb',
        '5ab815cbcfee494499e41151890d6d8f', 'd25718445d89453381c659b9c8734939',
        'efe3681f556b43e6aa65bf7bfc61a2f8', '8c7dfcee70754286959b80c6cfe2d246',
        '19640fe4b6894f97b16e0faad51432b4', '4b5bf3f4668d44fea9a676e9c4a8a79e',
        'e5a3df5fe95149b5b974af1d14277ea7', 'c2d5a5e7f7dc4e12b967fff22a0e1bf1',
        '209e9e9c3a2e4a399c44b6aa8be659d6', '491322eec05043c9907c59aa35b046ea',
        'eba4b7c19b5f48289e105937d03e5222', '9a61a88ed9094334a73aa93c08222110',
        '66be189398d74681a8b8db4b7dd52259', '7540f31d12f1439db67a50be7726c70d',
        '0cc28fe2c1064fb9a51f1647c28ca564', 'da41ecbc644b4915b84bb732e35ebf8c',
        '4c5c675340d14cd88a65b5e22eb9c9c1', '3032b651c72e4b5c86d4dea1b4930689',
        '15e1fa06e30e438a98430cc1fd0e8a69', '2eb0dd074d8e4a328fd2283184c4412e',
        'deab903b0b644d7a984c552968026631', '38b3010304734913bb148d17ebbc352b',
        'ab8bf18dfce14db4b8213b88f5dc1429', '788c5502523f4d01b3a8de47ec3dadfb',
        'd4c560022dda448dabb0b60fb62dd6ba', 'f13bd38bf9b64d579a63d4e7d750f592',
        '8b93e49bac094cf0a1c3ea983912f89f', '25496f19ffd14bd088cb430bfc01a4d7',
        '6d4b2bd795ae4c66900ad98ccd2371a6', '359f9c029ae44e1d9d47c05bc7915561',
        '2ce2df158372461fa91ce77455656f81', '373bf99c103d4464a7b963a83523fbcb',
        '5c76a3728b8b4d02ae20bd69fb928aa1', '3e1f0257a4b8457486651de2d16b61ab',
        'c67b72f4a0bd4f2b96ecbd0e01c58232', '5557f4edc1464442812293e2ea90a586',
        '19284973bd0342998c37848e931a90d0', 'd7ebcbbd26d849b384c11bec8df28a9b',
        '1623e7fab5174f2a84cf1b539233e3a6', '89f20737ec344aa48b543a9e005a38ca',
        '55c853d5aab04850a8059dc95e62c2ad', 'e1e664292aa144bc8d0d5d6441df084c',
        '4ea72ee843914a6cbd755e39f2489156', '52bb785dcb5548a1836898f14fb3cf6d',
        '4f679c8f8f6d4f5d8466253dda4733ba', '4ed628299b1e45a3a73704e8cf8287a9',
        'e0a212aafd574781b122a6ba66599a1e', '01c8c59260db4a3682d7b4f8da65425e',
        '673ce9bdd9254fff82a524653a07b57f', '9c83e438973e4824853e6a38928ca4ff',
        '5fc7c5dfc56b4971b14005bc53f69908', 'bf753f22bc524274bc359467e0be149c',
        '9f5f2211d6f943a89dd87bbe9703539d', '7809330e0345423ab6212787edf3561b',
        '69e393da7cb54eb6bea7927a2af0d1ee', '0be1ff07a8f148ca9535fb7f0deaf828',
        'e9a94a93c36a4adb831eb67ec8bdf289', '210add02013a4dfa84b7c5e23058781f',
        '5dd272b76c3f4e2582040a91f2be2dde', 'ed97ee6a3b444ba7800641baab057556',
        '83ffc453d209404781ea4574d77adb22', '724957e51f464a9aa64a16458443786d',
        '6308d6d934074a028fc3145eedf3e65f', 'e323a27a0ba94ab5ab11010d778cd41f',
        '53e8446852bf488bb1b09ae032918bbd', '53d376c20b7146349a53554a4616c48c',
        '3bc4553925494890a21ef7a15c40eaed', '9c5dc664216e43a99d5da3f23d373e4d',
        'ae643c14ded14e6ea74f8c8605200ed1', 'f4395dd8e4004f12a8dec3079a93d804',
        '1ad821fcfc9f4c24bb879ec7d5ba0ec1', '5d709891c41d423687ae4ea0473cb9c4',
        'd2db9f5df62c4d338d3bed43f616b954', 'c3e0e9f6ee8d4170a3d22a6179f1ca3a',
        'b3f63d06b6c54ae6ae80d688f4966408', '52678e6091214431bf9f3b41e5eeaa24']
}

class NuScenes(DataSource):
    def __init__(self, config, use_val=True):
        super().__init__(name=config.DATASET.lower())
        print('Loading NuScenes dataset')

        self.data_processor = DataProcessor(config, Path(config.DATA_PATH)/config.VERSION)
        self.dataset_cfg = self.config = config
        self.class_names = self.dataset_cfg.CLASS_NAMES

        self.root_path = Path(self.dataset_cfg.DATA_PATH) / self.dataset_cfg.VERSION

        official_train_split = self.include_nuscenes_data(mode='train')
        if use_val:
            from nuscenes.nuscenes import NuScenes as _NuScenes
            from tqdm import tqdm
            nusc = _NuScenes(version=self.dataset_cfg.VERSION, dataroot=self.root_path, verbose=False)
            for info in tqdm(official_train_split, desc='Split val scenes'):
                scene_token = nusc.get('sample', info['token'])['scene_token']
                if scene_token in val_scenes[self.dataset_cfg.VERSION]:
                    self.val_split.append(info)
                else:
                    self.train_split.append(info)
        else:
            self.train_split = official_train_split
        self.test_split = self.include_nuscenes_data(mode='test')

        print(f'Total samples for NuScenes dataset: train {len(self.train_split)}, val {len(self.val_split)}, test {len(self.test_split)}')

    def include_nuscenes_data(self, mode):
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)

        return nuscenes_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, info, max_sweeps=1):
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def prepare_data(self, info, split=None):
        points = self.get_lidar_with_sweeps(info, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })

        data_dict = self.data_processor(data_dict=input_dict, split=split)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict
