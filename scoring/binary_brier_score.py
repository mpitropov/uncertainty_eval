import numpy as np
from scoring.scoring_rule import ScoringRule

class BINARYBRIERSCORE(ScoringRule):
    def add_tp(self, pred_list):
        self.tp_value_list = [np.power(obj.pred_score - 1.0, 2) for obj in pred_list]

    def add_dup(self, pred_list):
        self.dup_value_list = [np.power(obj.pred_score - 1.0, 2) for obj in pred_list]

    # We can't assume that it is correctly classified
    def add_loc_err(self, pred_list):
        self.loc_err_value_list = [np.power(obj.data['score_all'][obj.gt_label - 1] - 1.0, 2) for obj in pred_list]

    def add_bg_tp(self, pred_list):
        self.fp_value_list = [np.power(obj.data['score_all'][-1] - 1.0, 2) for obj in pred_list]

    def add_fp(self, pred_list):
        self.fp_value_list = [np.power(obj.pred_score - 0, 2) for obj in pred_list]