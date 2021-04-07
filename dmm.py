import numpy as np

from scoring_rule import ScoringRule

class DMM(ScoringRule):
    def add_tp(self, gt_box_means, pred_box_means, pred_box_vars):
        # print(score)
        target_minus_pred = gt_box_means - pred_box_means
        pred_vars_minus_actual = pred_box_vars - \
            target_minus_pred * np.transpose(target_minus_pred)
        self.value_list.append(
            np.sum(np.abs(target_minus_pred) + \
                    np.abs(pred_vars_minus_actual)
                )
            )

    def add_fp(self, score):
        print('invalid')
        raise NotImplementedError

    # DMM REG Score P=1
    def mean(self):
        return np.mean(self.value_list)
