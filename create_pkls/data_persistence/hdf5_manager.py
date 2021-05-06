import collections

import h5py
import numpy as np
import os

from .data_persistence import DataPersistence


class HDF5Manager(DataPersistence):
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name if file_name.endswith('.hdf5') else file_name + '.hdf5'
        dir_name = os.path.dirname(file_name)
        if not os.path.isdir(dir_name): os.makedirs(dir_name)

    def save_data(self, name, data, pass_writing=False):
        """

        @param name:
        @param data:
        @return:
        """
        """
        r: Readonly, file must exist
        r+: Read/write, file must exist
        w:Create file, truncate if exists
        w- or x:Create file, fail if exists
        a:Read/write if exists, create otherwise (default)
        """
        # if os.path.isfile(self.file_name):
        #     raise ValueError(f'the file name {self.file_name} already exists.')

        mode = "a"
        h5_file = h5py.File(self.file_name, mode)
        if pass_writing and name in h5_file: return
        if isinstance(data, (list, np.ndarray)):
            tmp_data = np.asarray(data)
            dtype = tmp_data.dtype
            if dtype.char == 'U':
                dtype = h5py.special_dtype(vlen=str)
            if name in h5_file:
                self.remove_dataset(name)
            d_set = h5_file.create_dataset(name, shape=tmp_data.shape, dtype=dtype)
            d_set[:] = tmp_data
        else:
            if isinstance(data, str):
                if name in h5_file:
                    self.remove_dataset(name)
                dt = h5py.special_dtype(vlen=str)
                h5_file.create_dataset(name, data=data, dtype=dt)
            elif isinstance(data, int):
                if name in h5_file:
                    self.remove_dataset(name)
                h5_file.create_dataset(name, data=data, dtype='int64')
            else:
                if name in h5_file:
                    self.remove_dataset(name)
                h5_file.create_dataset(name, data=data, dtype='float64')
        h5_file.close()
        # del (d_set)

    def load_data(self, name):
        h5_file = h5py.File(self.file_name, "r")
        if name in h5_file:
            data = np.copy(h5_file[name])
        else:
            data=[]
        h5_file.close()
        return data

    def remove_dataset(self, name):
        """
        NOTE: This does not reduce the size of file.
        """
        h5_file = h5py.File(self.file_name, "a")
        if name in h5_file:
            del h5_file[name]
        h5_file.close()

    def _ds_exist(self, name):
        h5_file = h5py.File(self.file_name, "r")
        if name in h5_file:
            return True
        else:
            return False
        h5_file.close()

    def union_data(self, name, data):
        # NOTE: this code is just for 1-d data. at teh current version of the framework, we need just update 1-d data.
        if self._ds_exist(name):
            new_data = np.union1d(self.load_data(name), np.asarray(data))
        else:
            new_data = data
        self.remove_dataset(name)
        self.save_data(name, list(new_data))
