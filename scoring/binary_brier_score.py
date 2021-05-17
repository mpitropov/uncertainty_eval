import numpy as np
from scoring.scoring_rule import ScoringRule

class BINARYBRIERSCORE(ScoringRule):
    def add_tp(self, pred_list):
        self.tp_value_list = [np.power(obj.pred_score - 1.0, 2) for obj in pred_list]

    def add_bg_tp(self, pred_list):
        self.fp_value_list = [np.power(obj.data['score_all'][-1] - 1.0, 2) for obj in pred_list]

    def add_fp(self, pred_list):
        self.fp_value_list = [np.power(obj.pred_score - 0, 2) for obj in pred_list]