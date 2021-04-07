import numpy as np

from scoring_rule import ScoringRule

class NLLCLF(ScoringRule):
    def add_tp(self, score):
        # print(score)
        self.value_list.append(1.0 * -np.log(score))

    def add_fp(self, score):
        # print(score)
        self.value_list.append((1.0 - 0.0) * -np.log(1.0 - score))

    # Ignorance CLF Score Mean
    def mean(self):
        return np.mean(self.value_list)
