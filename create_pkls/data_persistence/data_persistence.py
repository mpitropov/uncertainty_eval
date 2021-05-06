import h5py
import numpy as np
from abc import ABC, abstractmethod


class DataPersistence(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def save_data(self, data, name, file_address):
        pass

    @abstractmethod
    def load_data(self, name, file_address, extension=True):
        pass

