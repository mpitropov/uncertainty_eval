import numpy as np
from scoring.scoring_rule import ScoringRule

class NLLCLF(ScoringRule):
    def add_tp(self, pred_list):
        self.tp_value_list = [-np.log(obj.pred_score) for obj in pred_list]

    def add_bg_tp(self, pred_list):
        self.fp_value_list = [-np.log(obj.data['score_all'][-1]) for obj in pred_list]

    def add_fp(self, pred_list):
        self.fp_value_list = [-np.log(1.0 - obj.pred_score) for obj in pred_list]