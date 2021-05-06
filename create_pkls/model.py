import os
import logging
from zipfile import ZipFile

import numpy as np

class Model():
    def __init__(self, name, platform):
        """
        @param name: specifies the name of model. For example, VGG-16, ResNet, etc
        @param platform: specifies the platform in which the model has been implemented. For example, tensorflow, pytorch, etc.
        """
        super().__init__()
        self.name = name
        if platform in ('tensorflow', 'pytorch'):
            self.platform = platform
        else:
            raise ValueError('Invalid name for argument `platform`. Supported names are `tensorflow` and `pytorch`')

    def random_init(self, seed):
        logging.warn('Trying to initialize model with unimplemented method `Model.random_init(seed)`.')

    def predict(self, data):
        raise NotImplementedError

    def train(self, train_data, train_data_size, val_data, val_data_size, cycle):
        raise NotImplementedError

    def save(self, path, weights_only=True):
        raise NotImplementedError

    def load(self, path, weights_only=True):
        raise NotImplementedError


class EnsembleModel(Model):
    def __init__(self, model, n_models=5, seeds=None, collate_fn=None, tmpdir='/tmp', name=None):
        if name is None:
            name = f'{model.name}.ensemble' 
        super().__init__(name, model.platform)
        self.model = model
        self.n_models = n_models
        self.collate_fn = collate_fn
        self.tmpdir = os.path.join(tmpdir, name)
        os.makedirs(self.tmpdir, exist_ok=True)
        
        # Generate random seeds if not provided
        if seeds is None:
            seeds = np.random.randint(np.iinfo(np.uint32).max, size=n_models)
        if len(seeds) != n_models:
            raise ValueError
        for i, seed in enumerate(seeds):
            ckpt_path = os.path.join(self.tmpdir, f'{i}.ckpt')
            self.model.random_init(seed)
            self.model.save(ckpt_path)

    def predict(self, *args, **kwargs):
        print('starting predict')
        print(args)
        print('after print')
        res = [None] * self.n_models
        for i in range(self.n_models):
            ckpt_path = os.path.join(self.tmpdir, f'{i}.ckpt')
            self.model.load(ckpt_path)
            res[i] = self.model.predict(*args, **kwargs)
        return self.collate_fn(res) if callable(self.collate_fn) else res

    def train(self, *args, **kwargs):
        for i in range(self.n_models):
            ckpt_path = os.path.join(self.tmpdir, f'{i}.ckpt')
            self.model.load(ckpt_path)
            self.model.train(*args, **kwargs)
            self.model.save(ckpt_path)
    
    def save(self, path, *args, **kwargs):
        with ZipFile(path, 'w') as f:
            for i in range(self.n_models):
                ckpt_path = os.path.join(self.tmpdir, f'{i}.ckpt.save')
                self.model.save(ckpt_path, *args, **kwargs)
                f.write(ckpt_path, f'{i}.ckpt')
                os.remove(ckpt_path)
    
    def load(self, path, *args, **kwargs):
        with ZipFile(path, 'r') as f:
            f.extractall(self.tmpdir)

    def __getattr__(self, name):
        if name in ['model', 'n_models', 'collate_fn', 'tmpdir']:
            raise AttributeError
        return getattr(self.model, name)


class MCDropoutModel(Model):
    def __init__(self, model, n_forward_passes=5, collate_fn=None, name=None):
        if name is None:
            name = f'{model.name}.mcdropout'
        super().__init__(name, model.platform)
        self.model = model
        self.n_forward_passes = n_forward_passes
        self.collate_fn = collate_fn

    def predict(self, data, *args, **kwargs):
        res = [None] * self.n_forward_passes
        try:
            for i in range(self.n_forward_passes):
                res[i] = self.model.predict(data, *args, mcdropout=i+1, **kwargs)
        except TypeError as e:
            raise TypeError('To use MCDropoutModel, your model needs to support `Model.predict(..., mcdropout=int)`.')
        return self.collate_fn(res) if callable(self.collate_fn) else res
    
    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        self.model.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        self.model.load(*args, **kwargs)
    
    def __getattr__(self, name):
        if name in ['model', 'n_forward_passes', 'collate_fn']:
            raise AttributeError
        return getattr(self.model, name)