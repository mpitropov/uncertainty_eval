import numpy as np

from scoring_rule import ScoringRule

class BRIERSCORE(ScoringRule):
    def add_tp(self, pred_label, softmax_scores):
        # print(score)
        gt_label = pred_label - 1 # Subtract 1 to convert to index in softmax output
        class_sum = 0
        for score_idx in range(len(softmax_scores)):
            if gt_label == score_idx:
                class_sum += np.power(softmax_scores[score_idx] - 1.0, 2)
            else:
                class_sum += np.power(softmax_scores[score_idx] - 0.0, 2)
        self.value_list.append(class_sum)
        self.tp_value_list.append(class_sum)

    def add_fp(self, bg_label, softmax_scores):
        class_sum = 0
        for score_idx in range(len(softmax_scores)):
            if bg_label == score_idx:
                class_sum += np.power(softmax_scores[score_idx] - 1.0, 2)
            else:
                class_sum += np.power(softmax_scores[score_idx] - 0.0, 2)
        self.value_list.append(class_sum)
        self.fp_value_list.append(class_sum)

    # Brier CLF Score
    def mean(self):
        return np.mean(self.value_list)

    def mean_tp(self):
        return np.mean(self.tp_value_list)

    def mean_fp(self):
        return np.mean(self.fp_value_list)
