import numpy as np

from scoring_rule import ScoringRule

class BINARYBRIERSCORE(ScoringRule):
    def add_tp(self, score):
        # print(score)
        self.value_list.append( np.power(score - 1.0, 2) )

    def add_fp(self, score):
        # print(score)
        self.value_list.append( np.power(score - 0, 2) )

    # Binary Brier CLF Score
    def mean(self):
        return np.mean(self.value_list)
