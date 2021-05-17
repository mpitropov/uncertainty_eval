import numpy as np
from scoring.scoring_rule import ScoringRule

class BRIERSCORE(ScoringRule):
    def add_tp(self, pred_list):
        softmax_scores = np.array([obj.data['score_all'] for obj in pred_list])
        
        gt_index = pred_list[0].pred_label - 1
        gt_labels = np.zeros(softmax_scores.shape)
        gt_labels[:,gt_index] += 1        

        self.tp_value_list = np.sum(np.power(softmax_scores - gt_labels, 2), axis=1)

    def add_fp(self, pred_list):
        softmax_scores = np.array([obj.data['score_all'] for obj in pred_list])

        bg_index = len(softmax_scores[0]) - 1
        bg_labels = np.zeros(softmax_scores.shape)
        bg_labels[:,bg_index] += 1        

        self.fp_value_list = np.sum(np.power(softmax_scores - bg_labels, 2), axis=1)
