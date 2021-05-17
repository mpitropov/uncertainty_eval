import os
import argparse
from pathlib import Path
from functools import partial
import logging

import numpy as np
import torch

# import wise_alf as alf
# from wise_alf.data import PyTorchDataLoader
from data_loader import PyTorchDataLoader
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
    # elif 'cadc' in dataset_name:
    #     from datasets.cadc import CADC
    #     data_source = CADC(dataset_cfg)
    #     data_pool = alf.data.DataPool(len(data_source.train_split))
    #     eval_config = {
    #         'criterion': 'iou',
    #         'thresholds': [0.7, 0.5, 0.5],
    #         'class_names': dataset_cfg.CLASS_NAMES
    #     }
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
    

    # callbacks.append(QueriedFNCallback(data_source, eval_config))
    # callbacks.append(LabeledFNCallback(
    #     pcdet_model, data_loader, data_pool, eval_config=eval_config,
    #     class_names=dataset_cfg.CLASS_NAMES, dbinfos_path=os.path.join(dataset_cfg.DATA_PATH, dbinfos_stem),
    #     fn_dbinfos_path=os.path.join(dataset_cfg.DATA_PATH, dbinfos_stem.replace('train', 'labeled_fn'))))


    ############################################################################
    #                        Initialize constant QBS
    ############################################################################
    # qbs = alf.query.qbs.ConstantQBS(args.query_batch_size)


    ############################################################################
    #                         Initialize query method
    ############################################################################
    # if query_method_name == 'random':
    #     query_method = alf.query.Random(args.seed)
    # elif query_method_name == 'entropy':
    #     # Mean entropy acquisition
    #     entr_sample = alf.query.Entropy(aggregate=alf.query.aggregation.Mean())
    #     entr = alf.query.Map(entr_sample, pre_processor=extract_scores_single)
    #     query_method = entr
    # elif query_method_name == 'margin':
    #     # Sum margin acquisition
    #     # Calculate `Sum( 1-margin )` for each sample.
    #     # It's implemented as `num_preds-Sum(margin)`
    #     margin_sample = alf.query.Margin(
    #         aggregate=alf.query.aggregation.Sum(),
    #         post_processor=lambda agg_scores, preds: len(preds)-agg_scores)
    #     margin_sample.reverse = False    # sort in descending order (select largest Sum(1-margin))
    #     margin = alf.query.Map(margin_sample, pre_processor=extract_scores_single)
    #     query_method = margin
    # # elif query_method_name == 'mutual_information':
    # #     mi_obj = alf.query.MutualInformation()
    # #     # Calculate mutual information for each sample by averaging mi_obj
    # #     mi_sample = alf.query.Map(mi_obj, aggregate=alf.query.aggregation.Mean() )
    # #     # Calculate mutual information for entire dataset
    # #     pre_processor = alf.utils.processing.Sequential([partial(post_nms_cluster, processes=8), extract_scores])
    # #     query_method = alf.query.Map(mi_sample, pre_processor=pre_processor)
    # elif query_method_name == 'weighted_fn_gt':
    #     from query.weighted_fn_gt import WeightedFN
    #     # feature_logger = alf.logging.FileLogger(name='features', ext='pdf')
    #     # loggers.append(feature_logger)
    #     query_method = WeightedFN(
    #         name=args.query_method_name.lower(),
    #         model=pcdet_model,
    #         data_source=data_source,
    #         data_loader=data_loader,
    #         qbs=qbs,
    #         eval_config=eval_config,
    #         score_cfg=args.score_cfg,
    #         dbinfos_stem=dbinfos_stem,
    #         # feature_logger=feature_logger
    #     )
    # elif query_method_name == 'weighted_fn_reg':
    #     from query.weighted_fn_reg import WeightedFN
    #     # feature_logger = alf.logging.FileLogger(name='features', ext='pdf')
    #     # loggers.append(feature_logger)
    #     labeled_output_logger = alf.logging.CustomLogger(name='output/labeled')
    #     loggers.append(labeled_output_logger)
    #     fn_predictor_logger = alf.logging.CustomLogger(name='fn_predictor')
    #     loggers.append(fn_predictor_logger)
    #     query_method = WeightedFN(
    #         name=args.query_method_name.lower(),
    #         model=pcdet_model,
    #         data_source=data_source,
    #         data_loader=data_loader,
    #         qbs=qbs,
    #         eval_config=eval_config,
    #         score_cfg=args.score_cfg,
    #         dbinfos_stem=dbinfos_stem,
    #         labeled_output_logger=labeled_output_logger,
    #         predictor_logger=fn_predictor_logger,
    #         # feature_logger=feature_logger,
    #     )
    # elif query_method_name == 'weighted_fn_cls':
    #     from query.weighted_fn_cls import WeightedFN
    #     tmp_output_logger = alf.logging.CustomLogger(name='output/tmp')
    #     loggers.append(tmp_output_logger)
    #     fn_predictor_logger = alf.logging.CustomLogger(name='fn_predictor')
    #     loggers.append(fn_predictor_logger)
    #     query_method = WeightedFN(
    #         name=args.query_method_name.lower(),
    #         model=pcdet_model,
    #         data_source=data_source,
    #         data_loader=data_loader,
    #         qbs=qbs,
    #         eval_config=eval_config,
    #         score_cfg=args.score_cfg,
    #         tmp_output_logger=tmp_output_logger,
    #         predictor_logger=fn_predictor_logger,
    #     )
    # else:
    #     raise NotImplementedError(f'Unknown query method {query_method_name}')


    ############################################################################
    #                     Initialize evaluation metrics
    ############################################################################
    # metrics = []
    # for metric_name in model_cfg.MODEL.POST_PROCESSING.EVAL_METRIC:
    #     if metric_name == 'official_eval':
    #         if 'kitti' in dataset_name:
    #             official_eval = alf.metrics.KITTIObjectEval(
    #                 annos=[info['annos'] for info in data_source.test_split],
    #                 class_names=dataset_cfg.CLASS_NAMES, metrics=['3d'],
    #                 pred_processor=None if args.ensemble_type == 'none' else \
    #                     lambda collated_preds: [ collated_dicts[0] for collated_dicts in collated_preds ]
    #             )
    #         # elif 'nuscenes' in dataset_name:
    #         #     metric_filter = [f'ap_{c}' for c in dataset_cfg.CLASS_NAMES]
    #         #     official_eval = alf.metrics.NuScenesObjectEval(root_path=Path(dataset_cfg.DATA_PATH) / dataset_cfg.VERSION,
    #         #         version=dataset_cfg.VERSION, filter=metric_filter, pred_processor=pred_processor)
    #         else:
    #             raise NotImplementedError(f'No official evaluation supported for {dataset_name}')
    #         metrics.append(official_eval)

    #     elif metric_name == 'detection_eval':
    #         eval_filters = []
    #         if 'kitti' in dataset_name:
    #             from wise_alf.metrics import build_kitti_filters
    #             kitti_infos_path = os.path.join(dataset_cfg.DATA_PATH, 'kitti_infos_val.pkl')
    #             eval_filters += build_kitti_filters(
    #                 kitti_infos_path, 
    #                 pred_processor=None if args.ensemble_type == 'none' else \
    #                     lambda collated_dicts: collated_dicts[0]
    #             )
    #         metrics.append(alf.metrics.DetectionEval(filters=eval_filters, **eval_config))

    #     # if 'entropy' in metric_names:
    #     #     metrics.append(alf.metrics.QueryScoreStats( entr, pred_processor=pred_processor ))
    #     # if 'margin' in metric_names:
    #     #     metrics.append(alf.metrics.QueryScoreStats( margin, pred_processor=pred_processor ))
    #     # if 'mutual_information' in metric_names:
    #     #     # Calculate mutual information per object
    #     #     mi_obj = alf.query.MutualInformation()
    #     #     # Calculate total mutual information for each sample by summing mi_obj
    #     #     mi_sample = alf.query.Map(mi_obj, aggregate=alf.query.aggregation.Sum() )
    #     #     # Calculate total mutual information for entire dataset
    #     #     mi_all = alf.query.Map(mi_sample,
    #     #         pre_processor=alf.utils.processing.Sequential([partial(post_nms_cluster, processes=8), extract_scores]))
    #     #     metrics.append(alf.metrics.QueryScoreStats( mi_all ))
    #     # if 'aleatoric_entropy' in metric_names:
    #     #     # Calculate aleatoric entropy per object
    #     #     ae_obj = alf.query.AleatoricEntropy()
    #     #     # Calculate total aleatoric entropy for each sample by summing ae_obj
    #     #     ae_sample = alf.query.Map(ae_obj, aggregate=alf.query.aggregation.Sum() )
    #     #     # Calculate total aleatoric entropy for entire dataset
    #     #     ae_all = alf.query.Map(ae_sample,
    #     #         pre_processor=alf.utils.processing.Sequential([partial(post_nms_cluster, processes=8), extract_scores]))
    #     #     metrics.append(alf.metrics.QueryScoreStats( ae_all ))
    #     # if 'epistemic_variance' in metric_names:
    #     #     # Calculate covariance per object
    #     #     cov_obj = alf.query.Covariance(aggregate=alf.query.aggregation.Sum())
    #     #     # Calculate total covariance for each sample by summing cov_obj
    #     #     cov_sample = alf.query.Map(cov_obj, aggregate=alf.query.aggregation.Sum())
    #     #     # Calculate total covariance for entire dataset
    #     #     cov_all = alf.query.Map(cov_sample,
    #     #         pre_processor=alf.utils.processing.Sequential([partial(post_nms_cluster, processes=8), extract_boxes]))
    #     #     metrics.append(alf.metrics.QueryScoreStats( cov_all ))
    #     # if 'aleatoric_variance' in metric_names:
    #     #     # Calculate aleatoric variance per object
    #     #     av_obj = alf.query.Noop( name='aleatoric_variance', aggregate=alf.query.aggregation.Sum() )
    #     #     # Calculate total aleatoric variance for each sample by summing av_obj
    #     #     av_sample = alf.query.Map(av_obj, aggregate=alf.query.aggregation.Sum())
    #     #     # Calculate total aleatoric variance for entire dataset
    #     #     av_all = alf.query.Map(av_sample,
    #     #         pre_processor=alf.utils.processing.Sequential([partial(post_nms_cluster, processes=8), extract_vars]))
    #     #     metrics.append(alf.metrics.QueryScoreStats( av_all ))
    #     else:
    #         raise NotImplementedError(f'Unknown evaluation metric {metric_name}')


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
