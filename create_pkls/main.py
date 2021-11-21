import os
import argparse
from pathlib import Path
from functools import partial
import logging

import numpy as np
import torch

from data_loader import PyTorchDataLoader, DataPool
from logger import FileLogger, CustomLogger
from initial_sampling.random_initialization import RandomInitializer
import model
from active_learning import ActiveLearning

import pcdet.datasets
from pcdet.datasets import DistributedSampler

from pcdet_model import PCDet
from utils import (
    parse_pcdet_cfg,
    collate_batch,
    # post_nms_cluster,
    # extract_scores,
    # extract_boxes,
    # extract_vars,
    extract_scores_single,
    pcdet_gt_processor,
    pcdet_pred_processor,
    pcdet_ensemble_collate_fn
)
# from callbacks import QueriedFNCallback, LabeledFNCallback

def train_loader_cfg_builder(batch_size, num_workers, data):
    return {
        'batch_size': batch_size,
        'collate_fn': collate_batch,
        'num_workers': num_workers,
        'sampler': DistributedSampler(data, shuffle=False)
    }
def eval_loader_cfg_builder(batch_size, num_workers, data):
    return {
        'batch_size': batch_size,
        'collate_fn': collate_batch,
        'num_workers': num_workers,
        'sampler': DistributedSampler(data, shuffle=False)
    }

if __name__ == '__main__':
    logging.basicConfig(level=20)

    parser = argparse.ArgumentParser(description='PCDet Active Learning')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify PCDet configuration file')
    parser.add_argument('--logdir_root', type=str, default=os.environ.get('LOGDIR'), required=False, help='specify where the log files will be stored')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='pytorch')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')

    # Active learning arguments
    parser.add_argument('--query_method', type=str, default='random', help='acquisition function')
    parser.add_argument('--query_method_name', type=str, default=None, help='query method name')
    parser.add_argument('--init_data_size', type=int, default=0, help='initial number of samples')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--query_batch_size', type=int, default=0, help='query batch size')
    parser.add_argument('--num_cycles', type=int, default=0, help='number of active learning cycles')
    parser.add_argument('--early_stopping', type=bool, default=False, help='apply early stopping')
    parser.add_argument('--use_val', type=bool, default=False, help='use validation dataset')
    parser.add_argument('--query_with_gt', type=bool, default=False, help='enable cheats')
    parser.add_argument('--score_cfg', type=str, default=None, help='score config for weighted_fn acquisition')
    parser.add_argument('--continuous', type=bool, default=False, help='train continuously')
    parser.add_argument('--ensemble_type', type=str, default='none', help='ensemble type')
    parser.add_argument('--ensemble_size', type=int, default=5, help='ensemble size')
    args = parser.parse_args()

    model_cfg, dataset_cfg = parse_pcdet_cfg(args.cfg_file)
    model_name = model_cfg.MODEL.NAME.lower()
    dataset_name = dataset_cfg.DATASET.lower()
    query_method_name = args.query_method.lower()
    logdir_root = os.path.join(args.logdir_root, 'al_log')
    if args.query_method_name is None:
        args.query_method_name = args.query_method
    
    loggers = []
    callbacks = []

    ############################################################################
    #                           Initialize dataset
    ############################################################################
    dbinfos_logger = FileLogger(name='dbinfos', ext='pkl')
    loggers.append(dbinfos_logger)
    if 'kitti' in dataset_name:
        from datasets.kitti import KITTIPool, KITTI
        dbinfos_stem = f'kitti_dbinfos_train.pkl'
        for aug_config in dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST:
            if aug_config.NAME == 'gt_sampling':
                aug_config.DB_INFO_PATH = [dbinfos_stem]
        data_source = KITTI(dataset_cfg, use_val=args.use_val)
        data_pool = KITTIPool(
            len(data_source.train_split),
            data_source=data_source, 
            dbinfos_stem=dbinfos_stem,
            dbinfos_logger=dbinfos_logger
        )
        eval_config = {
            'thresholds': {1:0.7, 2:0.5, 3:0.5},
            'criterion': 'iou_3d',
            'epsilon': 0.1,
            'n_positions': 100,
            'metrics': ['ap', 'fn', 'prob_fn'],
            'gt_processor': pcdet_gt_processor,
            'pred_processor': pcdet_pred_processor if args.ensemble_type == 'none' else \
                lambda collated_dicts: pcdet_pred_processor(collated_dicts[0]),
            'verbose': True
        }

        if args.init_data_size == 0:
            args.init_data_size = len(data_source.train_data())

    # elif 'nuscenes' in dataset_name:
    #     from datasets.nuscenes import NuScenes
    #     data_source = NuScenes(dataset_cfg, use_val=args.use_val)
    #     data_pool = alf.data.DataPool(len(data_source.train_split))
    #     eval_config = {
    #         'criterion': 'euclidean',
    #         'thresholds': [2.0, 2.0, 2.0],
    #         'class_names': dataset_cfg.CLASS_NAMES
    #     }
    elif 'cadc' in dataset_name:
        from datasets.cadc import CADCPool, CADC
        dbinfos_stem = f'cadc_dbinfos_train.pkl'
        data_source = CADC(dataset_cfg)
        data_pool = CADCPool(
            len(data_source.train_split),
            data_source=data_source, 
            dbinfos_stem=dbinfos_stem,
            dbinfos_logger=dbinfos_logger
        )
        # data_pool = DataPool(len(data_source.test_split))
        eval_config = {
            'criterion': 'iou',
            'thresholds': [0.7, 0.5, 0.7],
            'class_names': dataset_cfg.CLASS_NAMES
        }
    else:
        raise NotImplementedError(f'Unknown dataset {dataset_cfg.DATASET}')


    ############################################################################
    #                         Initialize data loader
    ############################################################################
    data_loader = PyTorchDataLoader(
        data_source=data_source,
        train_cfg_builder=partial(train_loader_cfg_builder, args.batch_size, args.workers),
        val_cfg_builder=partial(eval_loader_cfg_builder, 2*args.batch_size, args.workers),
        test_cfg_builder=partial(eval_loader_cfg_builder, 2*args.batch_size, args.workers),
        init_loader=False
    )


    ############################################################################
    #                      Initialize data initializer
    ############################################################################
    data_initializer = RandomInitializer(args.seed, data_pool, data_source, args.init_data_size)


    ############################################################################
    #                        Initialize PCDet model
    ############################################################################
    pcdet_logger = CustomLogger(name='train_log')
    loggers.append(pcdet_logger)
    pcdet_model = PCDet(model_cfg,
                        logger=pcdet_logger,
                        generate_prediction_dicts=getattr(pcdet.datasets, dataset_cfg.DATASET).generate_prediction_dicts,
                        **vars(args))

    if args.ensemble_type == 'ensemble':
        pcdet_model = model.EnsembleModel(
            pcdet_model, 
            n_models=args.ensemble_size, 
            collate_fn=pcdet_ensemble_collate_fn
        )
    elif args.ensemble_type == 'mcdropout':
        pcdet_model = model.MCDropoutModel(
            pcdet_model, 
            n_forward_passes=args.ensemble_size,
            collate_fn=pcdet_ensemble_collate_fn
        )

    ActiveLearning(
        model=pcdet_model,
        data_loader=data_loader,
        data_pool=data_pool,
        data_initializer=data_initializer,
        query_method=None, # query_method,
        query_batch_size=None, # qbs,
        num_cycles=args.num_cycles,
        metrics=None, # metrics,
        query_with_gt=args.query_with_gt,
        logdir_root=logdir_root,
        loggers=loggers,
        callbacks=None, # callbacks,
        weights_only=not args.continuous,
        overwrite=False
    ).run()
