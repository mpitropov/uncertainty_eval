import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from scoring_rule import ScoringRule

class NLLREG(ScoringRule):
    def add_tp(self, gt_box_means, pred_box_means, pred_box_vars):
        pred_multivariate_normal_dists = MultivariateNormal(
            torch.tensor(pred_box_means),
            torch.diag(torch.tensor(pred_box_means)) + 1e-2 * torch.eye(pred_box_vars.shape[0]))
        negative_log_prob = - \
            pred_multivariate_normal_dists.log_prob(torch.tensor(gt_box_means))
        self.value_list.append(float(negative_log_prob))

    def add_fp(self, score):
        print('invalid')
        raise NotImplementedError

    # Ignorance REG Score Mean
    def mean(self):
        return np.mean(self.value_list)
