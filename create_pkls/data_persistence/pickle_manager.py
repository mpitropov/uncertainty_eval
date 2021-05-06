import os
import pickle
import numpy as np

from .data_persistence import DataPersistence

class PickleManager(DataPersistence):
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name if file_name.endswith('.pkl') else file_name + '.pkl'
        dir_name = os.path.dirname(file_name)
        if not os.path.isdir(dir_name): os.makedirs(dir_name)
    
    def save_data(self, name, data, pass_writing=False):
        try:
            with open(self.file_name, 'rb') as file:
                pickle_data = pickle.load(file)
        except:
            pickle_data = {}
        if isinstance(data, np.ndarray):
            data = data.tolist()
        pickle_data[name] = data
        with open(self.file_name, 'wb') as file:
            pickle.dump(pickle_data, file)
    
    def load_data(self, name):
        with open(self.file_name, 'rb') as file:
            pickle_data = pickle.load(file)
            if name in pickle_data:
                data = pickle_data[name]
            else:
                data = None
        return data
    
    def union_data(self, name, data):
        if isinstance(data, np.ndarray):
            data = data.tolist()
        existing = self.load_data(name)
        if existing is None:
            new_data = data
        else:
            new_data = existing + data
        self.save_data(name, new_data)
