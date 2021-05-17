import numpy as np

class ScoringRule:
    def __init__(self):
        self.tp_value_list = []
        self.fp_value_list = []
    def add_tp(self, *args):
        print('invalid')
        raise NotImplementedError
    def add_fp(self, *args):
        print('invalid')
        raise NotImplementedError
    def mean_tp(self):
        return np.mean(self.tp_value_list)
    def mean_fp(self):
        return np.mean(self.fp_value_list)
    def mean(self):
        return np.mean(np.concatenate((self.tp_value_list, self.fp_value_list)))