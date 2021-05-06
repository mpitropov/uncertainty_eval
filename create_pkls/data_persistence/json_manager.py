import os
import json
import numpy as np

from .data_persistence import DataPersistence

class JSONManager(DataPersistence):
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name if file_name.endswith('.json') else file_name + '.json'
        dir_name = os.path.dirname(file_name)
        if not os.path.isdir(dir_name): os.makedirs(dir_name)
    
    def save_data(self, name, data, pass_writing=False):
        try:
            with open(self.file_name, 'r') as file:
                json_data = json.load(file)
        except:
            json_data = {}
        if isinstance(data, np.ndarray):
            data = data.tolist()
        json_data[name] = data
        with open(self.file_name, 'w') as file:
            json.dump(json_data, file)
    
    def load_data(self, name):
        with open(self.file_name, 'r') as file:
            json_data = json.load(file)
            if name in json_data:
                data = json_data[name]
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
