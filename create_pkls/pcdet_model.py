import time
import datetime
import hashlib
import glob
import shutil
import os
import logging
from pathlib import Path
from easydict import EasyDict
import pickle
import tqdm
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from tensorboardX import SummaryWriter

# import wise_alf as alf
from model import Model
# from wise_alf.utils.train_utils import EarlyStoppingController
from utils import process_gt_batch_dict
from pcdet.config import log_config_to_file
from pcdet.models import build_network, model_fn_decorator, load_data_to_gpu
from pcdet.utils import common_utils
from tools.train_utils.optimization import build_optimizer, build_scheduler
from tools.train_utils.train_utils import train_one_epoch, checkpoint_state, save_checkpoint

__all__ = ['PCDet']

default_args = {
    'epochs': None,
    'logdir': None,
    'batch_size': 1,
    'ckpt': None,
    'pretrained_model': None,
    'launcher': 'pytorch',
    'tcp_port': 18888,
    'sync_bn': False,
    'ckpt_save_interval': 1,
    'max_ckpt_save_num': 30,
    'merge_all_iters_to_one_epoch': False,
    'generate_prediction_dicts': None,
    'num_mcsamples': 0,
    'early_stopping': False,
    'continuous': False
}

class PCDet(Model):
    """
    This class is a thin wrapper around PCDet's model and training functionalities.
    It is designed to be faithful to the original PCDet design and at the same time
    conform with the interface requirements of the active learning framework.

    Most of the code in this class is adapted from PCDet/pcdet/tools/train.py, main()

    The configuration file used in this wrapper is an extended version of the
    `MODEL` config in PCDet's config file. The following additional fields are required:
        * CLASS_NAMES
        * POINT_CLOUD_RANGE
        * NUM_POINT_FEATURES
        * VOXEL_SIZE
        * GRID_SIZE
    """
    def __init__(self, cfg, logger, **args):
        """ Initialize PCDet model, training configurations and load pretrained weights.

        Args:
            cfg ([type]): [description]
        """

        super().__init__(name=cfg.MODEL.NAME.lower(), platform='pytorch')

        # Fill in default values for args and convert to easy dict
        for name, value in default_args.items():
            args[name] = args[name] if name in args else value
        args = EasyDict(**args)

        self.logger = logger
        self.args = args
        self.cfg = cfg
        self.ckpt = args.pretrained_model

    def random_init(self, seed):
        self.ckpt = None

    def predict(self, test_loader, cycle=None, mcdropout=False, use_cache=False, n_times=1, log_level=logging.INFO, leave=True):
        print('in pcdet preditc')
        print(test_loader)
        print('after print')
        self.args.ckpt = self.ckpt
        print(self.logger)
        print(self.logger.get_log_path())
        self.args.logdir = Path(self.logger.get_log_path())

        md5 = hashlib.md5()
        md5.update( pickle.dumps(test_loader) )
        with open(self.ckpt, 'rb') as ckpt_file:
            ckpt_bin = ckpt_file.read()
            md5.update( ckpt_bin )
        eval_hash = md5.hexdigest()

        output_dir = self.args.logdir
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_output_dir = output_dir / ('eval_%s' % eval_hash)

        try:
            if not use_cache:
                raise
            preds = pickle.load(open(eval_output_dir / 'result.pkl', 'rb'))
            gt = pickle.load(open(eval_output_dir / 'gt.pkl', 'rb'))
            print(f'Loading existing prediction results from {eval_output_dir}')
        except:
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            mp.spawn(test_worker,
                    nprocs=torch.cuda.device_count(),
                    args=(self.args, self.cfg, test_loader, eval_output_dir, mcdropout, n_times, log_level, leave),
                    join=True)

            preds = pickle.load(open(eval_output_dir / 'result.pkl', 'rb'))
            gt = pickle.load(open(eval_output_dir / 'gt.pkl', 'rb'))
        if not use_cache:
            shutil.rmtree(eval_output_dir)
        return gt, preds


    def train(self, train_loader, train_size, val_loader, val_size, cycle):
        self.args.logdir = Path(self.logger.get_log_path())

        if cycle > 0:
            self.args.ckpt = self.ckpt if self.args.continuous else None
            self.args.pretrained_model = None if self.args.continuous else self.ckpt

        ckpt_dir = self.args.logdir / 'ckpt'
        if (ckpt_dir / 'checkpoint_best.pth').is_file():
            self.ckpt = ckpt_dir / 'checkpoint_best.pth'
            return

        mp.spawn(train_worker,
                 nprocs=torch.cuda.device_count(),
                 args=(self.args, self.cfg, train_loader, train_size, val_loader, val_size, cycle),
                 join=True)

        if (ckpt_dir / 'checkpoint_best.pth').is_file():
            self.ckpt = ckpt_dir / 'checkpoint_best.pth'
        else:
            ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
            if len(ckpt_list) > 0:
                ckpt_list.sort(key=os.path.getmtime)
                self.ckpt = ckpt_list[-1]


    def save(self, path, weights_only=True):
        if self.ckpt is None:
            # Build and save randomly initialized model
            train_set = EasyDict({
                'class_names': self.cfg.CLASS_NAMES,
                'grid_size': self.cfg.GRID_SIZE,
                'voxel_size': self.cfg.VOXEL_SIZE,
                'point_cloud_range': self.cfg.POINT_CLOUD_RANGE,
                'point_feature_encoder': {
                    'num_point_features': self.cfg.NUM_POINT_FEATURES
                }
            })
            model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=train_set)
            if self.args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            save_checkpoint(
                checkpoint_state(model, None, 0, 0), filename=path,
            )
            self.ckpt = f'{path}.pth'
        shutil.copyfile(self.ckpt, f'{path}')

    def load(self, path, weights_only=True):
        if not os.path.exists(path):
            raise
        self.ckpt = path
        if weights_only:
            self.args.pretrained_model = path
            self.args.ckpt = None
        else:
            self.args.pretrained_model = None
            self.args.ckpt = path

    def model():
        raise NotImplementedError

# def train_worker(gpu, args, cfg, train_loader, train_size, val_loader, val_size, cycle):
#     args.local_rank = gpu

#     if args.launcher == 'none':
#         dist_train = False
#         total_gpus = 1
#     else:
#         total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
#             args.tcp_port, args.local_rank, backend='nccl'
#         )
#         dist_train = True

#     # if args.batch_size is None:
#     #     args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
#     # else:
#     #     assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
#     #     args.batch_size = args.batch_size // total_gpus

#     args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

#     common_utils.set_random_seed(args.seed)

#     # output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
#     output_dir = args.logdir
#     ckpt_dir = output_dir / 'ckpt'
#     output_dir.mkdir(parents=True, exist_ok=True)
#     ckpt_dir.mkdir(parents=True, exist_ok=True)

#     log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
#     logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

#     # log to file
#     logger.info('**********************Start logging**********************')
#     gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
#     logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

#     if dist_train:
#         logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
#     for key, val in vars(args).items():
#         logger.info('{:16} {}'.format(key, val))
#     # log_config_to_file(cfg, logger=logger)
#     if cfg.LOCAL_RANK == 0:
#         os.system('cp %s %s' % (args.cfg_file, output_dir))

#     tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

#     # -----------------------create dataloader & network & optimizer---------------------------
#     train_loader = train_loader()
#     val_loader = val_loader()
#     train_sampler = train_loader.sampler
#     # train_sampler = DistributedSampler(train_loader.dataset)
#     # train_loader = torch.utils.data.DataLoader(
#     #     train_loader.dataset,
#     #     batch_size=train_loader.batch_size,
#     #     sampler=train_sampler,
#     #     num_workers=train_loader.num_workers,
#     #     collate_fn=train_loader.collate_fn,
#     #     pin_memory=train_loader.pin_memory,
#     #     timeout=train_loader.timeout,
#     #     worker_init_fn=train_loader.worker_init_fn,
#     #     multiprocessing_context=train_loader.multiprocessing_context
#     # )
#     train_set = EasyDict({
#         'class_names': cfg.CLASS_NAMES,
#         'grid_size': cfg.GRID_SIZE,
#         'voxel_size': cfg.VOXEL_SIZE,
#         'point_cloud_range': cfg.POINT_CLOUD_RANGE,
#         'point_feature_encoder': {
#             'num_point_features': cfg.NUM_POINT_FEATURES
#         }
#     })

#     model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
#     if args.sync_bn:
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     model.cuda()

#     # for name, module in model.named_modules():
#     #     if 'vfe' in name:
#     #         print(name)
#     #         for name, param in module.named_parameters():
#     #             print(name)
#     #             param.requires_grad = False

#     optimizer = build_optimizer(model, cfg.OPTIMIZATION)

#     # load checkpoint if it is possible
#     start_epoch = it = 0
#     last_epoch = -1
#     if args.pretrained_model is not None:
#         model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)

#     if args.ckpt is not None:
#         it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
#         last_epoch = start_epoch + 1
#     else:
#         ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
#         if len(ckpt_list) > 0:
#             ckpt_list.sort(key=os.path.getmtime)
#             it, start_epoch = model.load_params_with_optimizer(
#                 ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
#             )
#             last_epoch = start_epoch + 1

#     model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
#     if dist_train:
#         model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
#     # logger.info(model)

#     lr_scheduler, lr_warmup_scheduler = build_scheduler(
#         optimizer, total_iters_each_epoch=len(train_loader), total_epochs=(args.epochs+1)*args.num_cycles if args.continuous else args.epochs,
#         last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
#     )

#     # -----------------------start training---------------------------
#     logger.info(f'**********************Start training (Cycle {cycle}, {train_size} samples)**********************')
#     model_func = model_fn_decorator()
#     optim_cfg = cfg.OPTIMIZATION
#     total_epochs = start_epoch+args.epochs if args.continuous else args.epochs
#     start_iter = it
#     rank = cfg.LOCAL_RANK
#     ckpt_save_dir = ckpt_dir
#     ckpt_save_interval = args.ckpt_save_interval
#     max_ckpt_save_num = args.max_ckpt_save_num
#     merge_all_iters_to_one_epoch = args.merge_all_iters_to_one_epoch
#     accumulated_iter = start_iter
#     es_ctrl = EarlyStoppingController(patience=10, min_delta=0.25)

#     with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
#         total_it_each_epoch = len(train_loader)
#         if merge_all_iters_to_one_epoch:
#             assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
#             train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
#             total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

#         dataloader_iter = iter(train_loader)
#         for cur_epoch in tbar:
#             if train_sampler is not None:
#                 train_sampler.set_epoch(cur_epoch)

#             # train one epoch
#             if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
#                 cur_scheduler = lr_warmup_scheduler
#             else:
#                 cur_scheduler = lr_scheduler
#             accumulated_iter = train_one_epoch(
#                 model, optimizer, train_loader, model_func,
#                 lr_scheduler=cur_scheduler,
#                 accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
#                 rank=rank, tbar=tbar, tb_log=tb_log,
#                 leave_pbar=(cur_epoch + 1 == total_epochs),
#                 total_it_each_epoch=total_it_each_epoch,
#                 dataloader_iter=dataloader_iter
#             )

#             # save trained model
#             trained_epoch = cur_epoch + 1
#             if trained_epoch % ckpt_save_interval == 0 and rank == 0:

#                 ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
#                 ckpt_list.sort(key=os.path.getmtime)

#                 if ckpt_list.__len__() >= max_ckpt_save_num:
#                     for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
#                         os.remove(ckpt_list[cur_file_idx])

#                 ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
#                 save_checkpoint(
#                     checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
#                 )

#             if args.early_stopping:
#                 # eval on val set
#                 val_loss = 0
#                 alf.query._disable_dropout_torch(model)
#                 if rank == 0:
#                     pbar = tqdm.tqdm(enumerate(val_loader),
#                         total=len(val_loader),
#                         desc='val',
#                         leave=(cur_epoch + 1 == total_epochs),
#                         dynamic_ncols=True)
#                 else:
#                     pbar = enumerate(val_loader)
#                 with torch.no_grad():
#                     for i, batch_dict in pbar:
#                         load_data_to_gpu(batch_dict)
#                         loss, _, _ = model_func(model, batch_dict)
#                         val_loss += loss.item()

#                 tmpdir = output_dir / 'tmp'
#                 os.makedirs(tmpdir, exist_ok=True)
#                 dist.barrier()
#                 pickle.dump(val_loss, open(os.path.join(tmpdir, f'val_loss_part_{rank}.pkl'), 'wb'))
#                 dist.barrier()
#                 val_loss = 0
#                 for i in range(total_gpus):
#                     val_loss += pickle.load(open(os.path.join(tmpdir, f'val_loss_part_{i}.pkl'), 'rb'))
#                 dist.barrier()

#                 if rank == 0:
#                     shutil.rmtree(tmpdir)
#                     tb_log.add_scalar('val/loss', val_loss, accumulated_iter)

#                 es_ctrl.epoch = trained_epoch
#                 if es_ctrl.should_stop(val_loss):
#                     if rank == 0:
#                         # Restore best weights from best epoch
#                         best_epoch = es_ctrl.best_epoch
#                         logger.info(f'Early stopping at epoch {trained_epoch}')
#                         logger.info(f'Loading weights from best epoch {best_epoch}')
#                         best_ckpt = ckpt_save_dir / ('checkpoint_epoch_%d.pth' % best_epoch)
#                         shutil.copyfile(best_ckpt, ckpt_save_dir / 'checkpoint_best.pth')
#                     break
#                 model.train()

#     logger.info(f'**********************End training (Cycle {cycle}, {train_size} samples)**********************\n\n\n')


def test_worker(gpu, args, cfg, test_loader, eval_output_dir, mcdropout=False, n_times=1, log_level=logging.INFO, leave=True):
    args.local_rank = gpu
    print(args)
    if mcdropout > 1:
        log_level = logging.WARNING

    # np.random.seed(args.seed)

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK, log_level=log_level)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    # for key, val in vars(args).items():
    #     logger.info('{:16} {}'.format(key, val))
    # log_config_to_file(cfg, logger=logger)

    # ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_loader = test_loader()
    # test_loader = torch.utils.data.DataLoader(
    #     test_loader.dataset,
    #     batch_size=test_loader.batch_size,
    #     sampler=DistributedSampler(test_loader.dataset, shuffle=False),
    #     num_workers=test_loader.num_workers,
    #     collate_fn=test_loader.collate_fn,
    #     pin_memory=test_loader.pin_memory,
    #     timeout=test_loader.timeout,
    #     worker_init_fn=test_loader.worker_init_fn,
    #     multiprocessing_context=test_loader.multiprocessing_context
    # )
    test_set = EasyDict({
        'class_names': cfg.CLASS_NAMES,
        'grid_size': cfg.GRID_SIZE,
        'voxel_size': cfg.VOXEL_SIZE,
        'point_cloud_range': cfg.POINT_CLOUD_RANGE,
        'point_feature_encoder': {
            'num_point_features': cfg.NUM_POINT_FEATURES
        }
    })

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    # eval_single_ckpt
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # eval_one_epoch
    epoch_id = args.ckpt
    save_to_file = False
    result_dir = eval_output_dir

    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = test_loader
    dataset = test_loader.dataset
    class_names = cfg.CLASS_NAMES

    det_annos = []
    gt_annos = []

    logger.info('*************** %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    if mcdropout:
        logger.warn('MC dropout enabled during inference')
        def _enable_dropout_torch(model):
            def apply_dropout(m):
                if type(m) == nn.Dropout:
                    m.train()
            model.apply(apply_dropout)
        _enable_dropout_torch(model)

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader)*n_times, leave=leave, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for _ in range(n_times):
        for i, batch_dict in enumerate(dataloader):
            gt_dict = process_gt_batch_dict(batch_dict)
            gt_annos += gt_dict

            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict)

            annos = args.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if save_to_file else None
            )

            det_annos += annos

            if cfg.LOCAL_RANK == 0:
                progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        gt_annos = common_utils.merge_results_dist(gt_annos, len(dataset), tmpdir=result_dir / 'tmpdir')

    # logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK == 0:
        with open(result_dir / 'result.pkl', 'wb') as f:
            pickle.dump(det_annos, f)
        with open(result_dir / 'gt.pkl', 'wb') as f:
            pickle.dump(gt_annos, f)
