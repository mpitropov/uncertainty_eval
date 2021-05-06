import os
import pprint
import logging
import pickle

from callback import Callback
from logger import DataLogger, FileLogger, CustomLogger
from data_persistence import JSONManager

def get_logdir_path(logdir_root, dataset_name,
    model_name, initializer_name,
    qbs_name, seed):
    logdir_path = os.path.join(
        logdir_root,
        dataset_name,
        model_name # ,
        # initializer_name,
        # qbs_name,
        # str(seed)
    )
    if not os.path.isdir(logdir_path):
        os.makedirs(logdir_path)
    return logdir_path

class ActiveLearning:
    def __init__(self, model, data_loader, data_pool, data_initializer,
                 query_method, query_batch_size, num_cycles, metrics=[],
                 query_with_gt=False,
                 logdir_root=None, loggers=[], callbacks=[],
                 weights_only=True, overwrite=False):

        # assert( query_method.name != 'init_model' )
        self.logdir_path = get_logdir_path(
            logdir_root=logdir_root,
            model_name=model.name,
            dataset_name=data_loader.name,
            initializer_name=data_initializer.name,
            qbs_name="None", # query_batch_size.name,
            seed=data_initializer.seed_num
        )

        self.logger = logging.getLogger(__name__)

        self.model = model
        self.data_loader = data_loader
        self.data_pool = data_pool
        self.data_initializer = data_initializer
        self.query_method = query_method
        self.query_batch_size = query_batch_size
        self.num_cycles = num_cycles
        self.metrics = metrics
        self.query_with_gt = query_with_gt
        self.loggers = loggers
        self.callbacks = callbacks
        self.weights_only = weights_only

        self.model_logger = FileLogger(name='saved_models', ext='pth')
        self.loggers.append(self.model_logger)

        self.unlabeled_output_logger = CustomLogger(name='output/unlabeled')
        self.loggers.append(self.unlabeled_output_logger)
        
        self.test_output_logger = CustomLogger(name='output/test')
        self.loggers.append(self.test_output_logger)

        self.pool_logger = DataLogger(name='queried_data')
        self.loggers.append(self.pool_logger)

        self.metric_loggers = {}
        # for metric in metrics:
        #     for k in metric.keys():
        #         self.metric_loggers[k] = DataLogger(name=k, rel_path='eval')
        #         self.loggers.append(self.metric_loggers[k])
            
        self.overwrite = overwrite
        

    def get_base_log_path(self, cycle):
        return os.path.join(
            self.logdir_path, 
            'init_model' if cycle == 0 else self.query_method.name
        )
    
    def update_loggers(self, cycle):
        base_log_path = self.get_base_log_path(cycle)
        for logger in self.loggers:
            logger.set_cycle(cycle)
            logger.set_base_path(base_log_path)
    
    def exec_callback(self, cb_name, *args, **kwargs):
        print('No callbacks')
        # for cb in self.callbacks:
        #     cb_fn = getattr(cb, cb_name)
        #     if cb_fn.__code__ != getattr(Callback, cb_name).__code__:
        #         self.logger.info(f'Callback: {cb.__class__.__name__}.{cb_name}')
        #         cb_fn(*args, **kwargs)

    def run(self):
        self.exec_callback('on_al_begin', loggers=self.loggers)
        for cycle in range(0, self.num_cycles + 1):
            self.update_loggers(cycle)
            self.exec_callback('on_cycle_begin', cycle=cycle)

            # Query and label unlabeled data using query method
            self.exec_callback('on_query_begin', cycle=cycle,
                pool_log_path=self.pool_logger.get_log_path(),
                unlabeled_output_path=self.unlabeled_output_logger.get_log_path())
            queried_data = self.query_and_label(cycle)
            self.exec_callback('on_query_end', cycle=cycle,
                queried_data=queried_data, 
                pool_log_path=self.pool_logger.get_log_path,
                unlabeled_output_path=self.unlabeled_output_logger.get_log_path())

            # Train and save model
            self.exec_callback('on_train_begin', cycle=cycle,
                model_path=self.model_logger.get_log_path())
            self.train_model(cycle, from_cycle=0)
            self.exec_callback('on_train_end', cycle=cycle, 
                model_path=self.model_logger.get_log_path())

            # Evaluate model
            self.exec_callback('on_eval_begin', cycle=cycle,
                test_output_path=self.test_output_logger.get_log_path())
            gts, preds, eval_results = self.evaluate_model(cycle)
            self.exec_callback('on_eval_end', cycle=cycle, eval_results=eval_results,
                test_output_path=self.test_output_logger.get_log_path())

            self.exec_callback('on_cycle_end', cycle=cycle)
        self.exec_callback('on_al_end')


    @staticmethod
    def get_model_predictions(model, data_loader, data_pool, split,
            output_logger, mode='test', overwrite=False, *args, **kwargs):
        log_path = output_logger.get_log_path()
        gts_path = os.path.join(log_path, 'gts.pkl')
        preds_path = os.path.join(log_path, 'preds.pkl')
        if (not overwrite) and os.path.isdir(log_path) and os.path.isfile(gts_path) and os.path.isfile(preds_path):
            # Load existing results if possible
            output_logger.info('Loading existing prediction results')
            with open(gts_path, 'rb') as gts_f:
                gts = pickle.load(gts_f)
            with open(preds_path, 'rb') as preds_f:
                preds = pickle.load(preds_f)
        else:
            # Predict and save prediction results
            if split == 'unlabeled':
                data_split = data_loader.data_loader(
                    train_indices=data_pool.unlabeled_data(), 
                    mode=mode
                )
            elif split == 'labeled':
                data_split = data_loader.data_loader(
                    train_indices=data_pool.labeled_data(), 
                    mode=mode
                )
            elif split == 'test':
                data_split = data_loader.test_data()
            else:
                raise ValueError(f'Unrecognized split {split}')

            gts, preds = model.predict(data_split, *args, **kwargs)

            output_logger.info('Saving prediction results')
            with open(gts_path, 'wb') as gts_f:
                pickle.dump(gts, gts_f)
            with open(preds_path, 'wb') as preds_f:
                pickle.dump(preds, preds_f)
        return gts, preds


    def query_and_label(self, cycle):
        queried_data = None
        if (not self.overwrite) and os.path.isfile(self.pool_logger.get_log_path()):
            queried_data = self.pool_logger.get_logged_data()

        if queried_data is None:
            if cycle == 0:
                queried_data = self.data_initializer.data()
            else:
                gts = preds = None
                # if self.query_method.name != 'random':
                gts, preds = self.get_model_predictions(self.model, self.data_loader, self.data_pool, split='unlabeled', output_logger=self.unlabeled_output_logger)
                if self.query_with_gt:
                    scores, sorted_scores = self.query_method(preds, self.data_pool, self.data_loader, gt=gts)
                else:
                    scores, sorted_scores = self.query_method(preds, self.data_pool, self.data_loader)
                score_idx = sorted_scores[:self.query_batch_size.get()]
                queried_data = self.data_pool.unlabeled_data()[score_idx]
            self.pool_logger.log_data(queried_data)
        else:
            self.logger.info(f'Loading existing queried data for cycle {cycle}')

        self.data_pool.mark_as_labeled(queried_data, cycle)
        return queried_data


    def train_model(self, cycle, from_cycle=0):
        if (not self.overwrite) and os.path.isfile( self.model_logger.get_log_path() ):
            # Load existing model
            self.logger.info(f'Loading existing model for cycle {cycle}')
            self.model.load( self.model_logger.get_log_path(), 
                weights_only=self.weights_only )
        else:
            # Start from previous cycle, if specified
            if cycle != 0 and from_cycle is not None:
                self.model.load(
                    self.model_logger.get_log_path(
                        self.get_base_log_path(from_cycle),
                        cycle=from_cycle
                    ),
                    weights_only=self.weights_only
                )

            # Get training and validation dataloaders
            labeled_data = self.data_loader.train_data(self.data_pool.labeled_data())
            labeled_data_size = self.data_pool.labeled_data_size()
            val_data = self.data_loader.val_data()
            val_data_size = self.data_loader.val_data_size()

            # Train model
            self.model.train(labeled_data, labeled_data_size, val_data, val_data_size, cycle)

            # Save model checkpoint
            self.model.save( self.model_logger.get_log_path(), 
                weights_only=self.weights_only )


    def evaluate_model(self, cycle):
        eval_results = {}
        gts = preds = None

        # just give back stuff
        gts, preds = self.get_model_predictions(self.model, self.data_loader, self.data_pool, split='test', output_logger=self.test_output_logger)
        return gts, preds, {}

        if len(self.metrics) > 0:
            gts, preds = self.get_model_predictions(self.model, self.data_loader, self.data_pool, split='test', output_logger=self.test_output_logger)
            for metric in self.metrics:
                eval_results = { **eval_results, **metric(gts, preds) }

        self.logger.info(f'Cycle {cycle}, Data size: {self.data_pool.labeled_data_size()}, Eval: \n{pprint.pformat(eval_results)}')

        for k, v in eval_results.items():
            self.metric_loggers[k].log_data(v)

        return gts, preds, eval_results
    
