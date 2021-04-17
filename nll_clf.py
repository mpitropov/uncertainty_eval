import numpy as np

from scoring_rule import ScoringRule

class NLLCLF(ScoringRule):
    def add_tp(self, score):
        self.value_list.append(-np.log(score))
        self.tp_value_list.append(-np.log(score))

    def add_bg_tp(self, score):
        self.value_list.append(-np.log(score))
        self.fp_value_list.append(-np.log(score))

    def add_fp(self, score):
        self.value_list.append(-np.log(1.0 - score))
        self.fp_value_list.append(-np.log(1.0 - score))

    # Ignorance CLF Score Mean
    def mean(self):
        return np.mean(self.value_list)

    def mean_tp(self):
        return np.mean(self.tp_value_list)

    def mean_fp(self):
        return np.mean(self.fp_value_list)