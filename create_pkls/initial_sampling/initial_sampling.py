from abc import ABC, abstractmethod
import numpy as np


class InitialSampling(ABC):

    def __init__(self, name, seed_num, pool, data_source, num_data):
        self.name = name
        self.seed_num = seed_num
        self.pool = pool
        self.data_source = data_source
        self.num_data = num_data

    @abstractmethod
    def data(self):
        pass
