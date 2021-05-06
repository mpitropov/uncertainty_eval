import os
import logging

from data_persistence import (
    HDF5Manager,
    JSONManager,
    PickleManager
)

class PyLoggerWrapper():
    def __init__(self, name, init_logger=False, base_path=None, rel_path=''):
        self.name = name
        self._cycle = 0
        self._pylogger = logging.getLogger(name) if init_logger else None
        self._rel_path = rel_path
        if base_path is not None:
            self.set_base_path(base_path)
    
    def set_logger(self, logger):
        self._pylogger = logger

    def set_cycle(self, cycle):
        self._cycle = cycle
    
    def set_base_path(self, base_path):
        log_path = os.path.join(base_path, self._rel_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path, exist_ok=True)
        self._base_path = base_path

    def get_log_path(self, base_path=None):
        base_path = self._base_path if base_path is None else base_path
        print('base_path', base_path)
        return os.path.join(base_path, self._rel_path)
    
    def __getattr__(self, name):
        if self._pylogger is not None and hasattr(self._pylogger, name):
            return getattr(self._pylogger, name)
        if hasattr(logging, name):
            return getattr(logging, name)


class DataLogger(PyLoggerWrapper):
    def __init__(self, name, data_type='dict', file_format='json',
        base_path=None, rel_path=''):
        self.data_type = data_type
        self.file_format = file_format
        if file_format == 'json':
            self.dp_cls = JSONManager
        elif file_format == 'pkl':
            self.dp_cls = PickleManager
        elif file_format == 'hdf5':
            self.dp_cls = HDF5Manager
        else:
            raise ValueError(f'unrecognized file format: `{file_format}`')
        self.dp = None
        super().__init__(name=name, init_logger=False,
            base_path=base_path, rel_path=rel_path)


    def set_base_path(self, base_path):
        super().set_base_path(base_path)
        self.dp = self.dp_cls(self.get_log_path())

    def get_log_path(self, base_path=None):
        return os.path.join(
            super().get_log_path(base_path),
            f'{self.name}.{self.file_format}'
        )

    def get_logged_data(self, base_path=None, cycle=None):
        if base_path is not None:
            log_path = self.get_log_path(base_path)
            dp = self.dp_cls(log_path)
        else:
            dp = self.dp
        cycle = self._cycle if cycle is None else cycle
        return dp.load_data(name=f'cycle_{cycle}')

    def log_data(self, data, base_path=None, cycle=None):
        if base_path is not None:
            log_path = self.get_log_path(base_path)
            dp = self.dp_cls(log_path)
        else:
            dp = self.dp
        cycle = self._cycle if cycle is None else cycle
        dp.save_data(name=f'cycle_{cycle}', data=data)


class FileLogger(PyLoggerWrapper):
    def __init__(self, name, ext='txt', base_path=None, rel_path=''):
        super().__init__(name=name, init_logger=False,
            base_path=base_path, rel_path=os.path.join(rel_path, name))
        self.ext = ext

    def get_log_path(self, base_path=None, cycle=None):
        log_path = super().get_log_path(base_path)
        cycle = self._cycle if cycle is None else cycle
        return os.path.join(log_path, f'cycle_{cycle}.{self.ext}')


class CustomLogger(PyLoggerWrapper):
    def __init__(self, name, base_path=None, rel_path=''):
        super().__init__(name=name, init_logger=False,
            base_path=base_path, rel_path=rel_path)

    def get_log_path(self, base_path=None, cycle=None):
        log_path = os.path.join(
            super().get_log_path(base_path),
            self.name
        )
        cycle = self._cycle if cycle is None else cycle
        if cycle != 0:
            log_path = os.path.join(log_path, f'cycle_{cycle}')
        if not os.path.isdir(log_path):
            os.makedirs(log_path, exist_ok=True)
        return log_path
