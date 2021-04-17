import numpy as np

from scoring_rule import ScoringRule

class BINARYBRIERSCORE(ScoringRule):
    def add_tp(self, score):
        self.value_list.append( np.power(score - 1.0, 2) )
        self.tp_value_list.append( np.power(score - 1.0, 2) )

    def add_bg_tp(self, score):
        self.value_list.append( np.power(score - 1.0, 2) )
        self.fp_value_list.append( np.power(score - 1.0, 2) )

    def add_fp(self, score):
        self.value_list.append( np.power(score - 0, 2) )
        self.fp_value_list.append( np.power(score - 0, 2) )

    # Binary Brier CLF Score
    def mean(self):
        return np.mean(self.value_list)

    def mean_tp(self):
        return np.mean(self.tp_value_list)

    def mean_fp(self):
        return np.mean(self.fp_value_list)