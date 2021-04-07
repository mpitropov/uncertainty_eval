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

    def add_fp(self, score):
        print('invalid')
        raise NotImplementedError

    # Brier CLF Score
    def mean(self):
        return np.mean(self.value_list)
