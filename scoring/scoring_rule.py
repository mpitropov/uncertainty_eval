import numpy as np

class ScoringRule:
    def __init__(self):
        self.tp_value_list = []
        self.loc_err_value_list = []
        self.dup_value_list = []
        self.fp_value_list = []
        self.fn_value_list = []
    def add_tp(self, *args):
        print('invalid')
        raise NotImplementedError
    def add_loc_err(self, *args):
        print('invalid')
        raise NotImplementedError
    def add_dup(self, *args):
        print('invalid')
        raise NotImplementedError
    def add_fp(self, *args):
        print('invalid')
        raise NotImplementedError
    def add_fn(self, *args):
        print('invalid')
        raise NotImplementedError
    def mean_tp(self):
        return np.mean(self.tp_value_list)
    def mean_loc_err(self):
        return np.mean(self.loc_err_value_list)
    def mean_dup(self):
        return np.mean(self.dup_value_list)
    def mean_fp(self):
        return np.mean(self.fp_value_list)
    def mean_fn(self):
        return np.mean(self.fn_value_list)
    def mean(self):
        return np.mean(np.concatenate((self.tp_value_list, self.loc_err_value_list, \
                                        self.dup_value_list, self.fp_value_list, self.fn_value_list)))
    def var(self):
        return np.var(np.concatenate((self.tp_value_list, self.loc_err_value_list, \
                                        self.dup_value_list, self.fp_value_list, self.fn_value_list)))
