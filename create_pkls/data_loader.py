from initial_sampling import *

import numpy as np
import collections
from functools import partial


def id_fn(x, *args, **kwargs):
    return x

def empty_builder(*args, **kwargs):
    return {}


class Data():
    def __init__(self, data, prepare_data_fn=None):
        """ A map-style dataset

        Args:
            data (list, dict): an iterable with `__getitem__` and `__len__` implemented
                that represents data or metadata.
            load_func (func, optional): if the data is metadata, this function
                will be called to load the actual data. Defaults to None.
            preprocess (func, optional): if specified, this function will be
                called on the loaded data to perform necessary formatting. Defaults to None.
        """
        self.data = data
        self.prepare_data_fn = id_fn if prepare_data_fn is None else prepare_data_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.prepare_data_fn( self.data[idx] )


class DataSource():
    def __init__(self, name='unnamed_dataset'):
        """ Keeps track of (meta-)data for a dataset

        Args:
            name (str): name of the dataset in lower case separated by underscore,
                e.g. nuscenes_mini, cadc
            config (dict): configuration dict including information like
                data path on disk, dataset version, train/val/test split etc.
            preprocess (func): preprocess function that formats the raw data
        """
        self.name = name

        # Load (meta-)data based on config
        self.train_split = []
        self.val_split = []
        self.test_split = []

    def prepare_data(self, data, split):
        """ Load the actual data based on a provided meta-data.
        If the data is not meta-data, then this function is by default an identity function.

        Args:
            data (any): (meta-)data, should be an element of `train/val/test_split`
            split (str): One of ['train', 'val', 'test']

        Returns:
            any: The actual data
        """
        return data
    
    def data(self, train_indices=[], val_indices=[], test_indices=[], mode='test'):
        train_split = self.train_split if train_indices is None else [self.train_split[i] for i in train_indices]
        val_split = self.val_split if val_indices is None else [self.val_split[i] for i in val_indices]
        test_split = self.test_split if test_indices is None else [self.test_split[i] for i in test_indices]
        if len(train_split) == 0 and len(val_split) == 0: # CADC
            splits = test_split
        else:
            splits = train_split + val_split + test_split
        return Data( splits, partial(self.prepare_data, split=mode) )

    def train_data(self, indices=None):
        """ Returns an iterable whose `__getitem__` method returns a processed data point
        e.g. preprocess( prepare_data( meta_info[idx] ) ), and 

        Args:
            indices (list): when provided

        Returns:
            Data:
        """
        return self.data(train_indices=indices, mode='train')

    def val_data(self, indices=None):
        return self.data(val_indices=indices, mode='val')

    def test_data(self, indices=None):
        return self.data(test_indices=indices, mode='test')


class DataLoader(object):
    def __init__(self,
                 platform,
                 data_source,
                 train_cfg={},
                 train_cfg_builder=empty_builder,
                 train_cfg_fn=id_fn,
                 val_cfg={},
                 val_cfg_builder=empty_builder,
                 val_cfg_fn=id_fn,
                 test_cfg={},
                 test_cfg_builder=empty_builder,
                 test_cfg_fn=id_fn,
                 init_loader=True):
        """ A data loader template class contains functions that return appropriate
        platform-specific data loaders for specified data splits.

        This template assumes that the data loader is initalized by providing
        a data object (list, dict, etc.) as the first argument followed by
        other additional arguments. To create a data loader for a new platform,
        simply override `self.dataloader_cls` to be the class of the platform-specific dataloader
        (e.g. `torch.utils.data.DataLoader`).
        
        If the data loader class does not conform with the assumed argument placement,
        the functions in this template class need to be overriden.

        Args:
            platform (str):
            data_source (DataSource):
            train_cfg (dict, optional): keyword arguments for train loader. Defaults to {}.
            train_cfg_builder (func, optional): a function that returns the keyword arguments for train loader based on data. This is useful when specifying samplers for PyTorch DataLoader. Defaults to empty_builder.
            train_cfg_fn (func, optional): a function that takes in a data loader and returns a data loader. This is useful when setting up data loaders that need to be configured by calling certain member functions such as TensorFlow Dataset. Defaults to id_fn.
            val_cfg (dict, optional): same as train_cfg, but for validation set. Defaults to {}.
            val_cfg_builder (func, optional): same as train_cfg_builder, but for validation set . Defaults to empty_builder.
            val_cfg_fn (func, optional): same as train_cfg_fn, but for validation set. Defaults to id_fn.
            test_cfg (dict, optional): same as train_cfg, but for test set. Defaults to {}.
            test_cfg_builder (func, optional): same as train_cfg_builder, but for test set. Defaults to empty_builder.
            test_cfg_fn (func, optional): same as train_cfg_fn, but for test set. Defaults to id_fn.
        """
        super().__init__()
        self.name = data_source.name
    
        if platform in ('tensorflow', 'pytorch'):
            self.platform = platform
        else:
            raise ValueError('Invalid name for argument `platform`. Supported names are `tensorflow` and `pytorch`')

        self.data_source = data_source
        self.train_cfg = train_cfg
        self.train_cfg_builder = empty_builder if train_cfg_builder is None else train_cfg_builder
        self.train_cfg_fn = id_fn if train_cfg_fn is None else train_cfg_fn
        self.val_cfg = val_cfg
        self.val_cfg_builder = empty_builder if val_cfg_builder is None else val_cfg_builder
        self.val_cfg_fn = id_fn if val_cfg_fn is None else val_cfg_fn
        self.test_cfg = test_cfg
        self.test_cfg_builder = empty_builder if test_cfg_builder is None else test_cfg_builder
        self.test_cfg_fn = id_fn if test_cfg_fn is None else test_cfg_fn
        self.init_loader = init_loader

        def dataloader_cls_error(*args, **kwargs):
            raise AttributeError('Platform-specific data loader class is not defined. Use a subclass (e.g. TFDataLoader) or set `dataloader.dataloader_cls`.')
        self.dataloader_cls = dataloader_cls_error
    
    @staticmethod
    def _dataloader_initializer(
        dataloader_cls, 
        cfg, cfg_builder, cfg_fn, 
        data, *args, **kwargs
    ):
        return cfg_fn(
            dataloader_cls(
                data, **cfg, **cfg_builder(data)
            ), data
        )
    
    def data_loader(self, train_indices=[], val_indices=[], test_indices=[], mode='test'):
        data = self.data_source.data(train_indices, val_indices, test_indices, mode)
        initializer = partial(
            self._dataloader_initializer,
            self.dataloader_cls,
            getattr(self, mode + '_cfg'),
            getattr(self, mode + '_cfg_builder'),
            getattr(self, mode + '_cfg_fn'),
            data
        )
        if self.init_loader:
            return initializer()
        else:
            return initializer

    def train_loader(self, indices=None):
        return self.data_loader(train_indices=indices, mode='train')
    
    def val_loader(self, indices=None):
        return self.data_loader(val_indices=indices, mode='val')
    
    def test_loader(self, indices=None):
        return self.data_loader(test_indices=indices, mode='test')
    
    # Alias functions for compatibility
    def train_data(self, indices=None):
        return self.train_loader(indices)
    def val_data(self, indices=None):
        return self.val_loader(indices)
    def test_data(self, indices=None):
        return self.test_loader(indices)
    def train_data_size(self):
        return len(self.data_source.train_data())
    def val_data_size(self):
        return len(self.data_source.val_data())
    def test_data_size(self):
        return len(self.data_source.test_data())

class TFDataLoader(DataLoader):
    def __init__(self, data_source, **kwargs):
        super().__init__(platform='tensorflow', data_source=data_source, **kwargs)
        import tensorflow as tf
        self.dataloader_cls = tf.data.Dataset

class TFTensorSlicesDataLoader(DataLoader):
    def __init__(self, data_source, **kwargs):
        super().__init__(platform='tensorflow', data_source=data_source, **kwargs)
        import tensorflow as tf
        self.dataloader_cls = tf.data.Dataset.from_tensor_slices

class TFTensorsDataLoader(DataLoader):
    def __init__(self, data_source, **kwargs):
        super().__init__(platform='tensorflow', data_source=data_source, **kwargs)
        import tensorflow as tf
        self.dataloader_cls = tf.data.Dataset.from_tensors

class TFGeneratorDataLoader(DataLoader):
    def __init__(self, data_source, **kwargs):
        super().__init__(platform='tensorflow', data_source=data_source, **kwargs)
        import tensorflow as tf
        self.dataloader_cls = tf.data.Dataset.from_generator

class PyTorchDataLoader(DataLoader):
    def __init__(self, data_source, **kwargs):
        super().__init__(platform='pytorch', data_source=data_source, **kwargs)
        import torch
        self.dataloader_cls = torch.utils.data.DataLoader


class DataPool():
    """
    This class allows to control the AL pool just by the index of data point.
    The actual data can be controlled and managed by another class.
    """

    def __init__(self, indices):
        if isinstance(indices, (collections.Sequence, np.ndarray)):
            self.indices = np.asarray(indices, dtype=int)
        elif isinstance(indices, int):
            self.indices = np.arange(indices)
        else:
            raise ValueError('Invalid type for argument `indices`.')
        
        self.labels = np.zeros(len(self.indices))
        # self.labeled_per_cycle = {}

    def mark_as_labeled(self, indices, cycle=None):
        for idx in indices:
            self.labels[np.where(self.indices == idx)] = 1

        # if cycle is not None:
        #     if cycle > 0:
        #         if cycle in self.labeled_per_cycle:
        #             print("Cycle number ", cycle, "has already been used.")
        #         else:
        #             self.labeled_per_cycle[cycle] = indices
        #     else:
        #         print("Cycle number 0 reserved for the initial data")

    def mark_as_unlabeled(self, indices, cycle=None):
        for idx in indices:
            self.labels[np.where(self.indices == idx)] = 0

        # if cycle is not None and cycle > -1:
        #     try:
        #         del self.labeled_per_cycle[cycle]
        #     except KeyError:
        #         print("Cycle ", cycle, " not found.")

    def unlabeled_data(self):
        indices = np.where(self.labels == 0)
        return self.indices[indices]

    def unlabeled_data_size(self):
        indices = np.where(self.labels == 0)
        return len(indices[0])

    def labeled_data(self):
        indices = np.where(self.labels == 1)
        return self.indices[indices]

    def labeled_data_size(self):
        indices = np.where(self.labels == 1)
        return len(indices[0])

    # def save_data(self, file_address):
    #     self.data_persistence.save_data(self.indices, "indices", file_address)
    #     self.data_persistence.save_data(self.labels, "labels", file_address)
    #     num_cycles = np.array([self.labeled_per_cycle.items().__len__()])
    #     self.data_persistence.save_data(num_cycles, "num_cycles", file_address)
    #     for item in self.labeled_per_cycle.items():
    #         self.data_persistence.save_data(np.asarray(item[1]), "cycle_"+str(item[0]), file_address)

IndexBasedPool = DataPool
