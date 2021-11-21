import numpy as np
from scoring.scoring_rule import ScoringRule

class BRIERSCORE(ScoringRule):
    def get_pos_matches(self, pred_list):
        softmax_scores = np.array([obj.data['score_all'] for obj in pred_list])

        gt_index = np.array([obj.gt_label - 1 for obj in pred_list], dtype=int)
        gt_labels = np.zeros(softmax_scores.shape)
        # Add 1 to every row of gt_labels at the gt_index
        np.add.at(gt_labels, (range(len(gt_labels)), gt_index), 1)

        return np.sum(np.power(softmax_scores - gt_labels, 2), axis=1)

    def add_tp(self, pred_list):
        if len(pred_list) == 0:
            return
        self.tp_value_list = self.get_pos_matches(pred_list)

    def add_dup(self, pred_list):
        if len(pred_list) == 0:
            return
        self.dup_value_list = self.get_pos_matches(pred_list)

    def add_loc_err(self, pred_list):
        if len(pred_list) == 0:
            return
        self.loc_err_value_list = self.get_pos_matches(pred_list)

    def add_fp(self, pred_list):
        if len(pred_list) == 0:
            return
        softmax_scores = np.array([obj.data['score_all'] for obj in pred_list])

        bg_index = len(softmax_scores[0]) - 1
        bg_labels = np.zeros(softmax_scores.shape)
        bg_labels[:,bg_index] += 1

        self.fp_value_list = np.sum(np.power(softmax_scores - bg_labels, 2), axis=1)

    def add_fn(self, fn_bg):
        self.fn_value_list = [0.9 ** 2 for obj in fn_bg]
