import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.von_mises import VonMises

from scoring_rule import ScoringRule

class NLLREG(ScoringRule):
    def add_tp(self, gt_box_means, pred_box_means, pred_box_vars):
        pred_multivariate_normal_dists = MultivariateNormal(
            torch.tensor(pred_box_means[:-1]),
            torch.diag(torch.tensor(pred_box_vars[:-1])) + 1e-2 * torch.eye(pred_box_vars[:-1].shape[0]))

        # concentration (a reciprocal measure of dispersion, so 1/k is analogous to variance). 
        pred_von_mises_dists = VonMises(
            torch.tensor(pred_box_means[-1:]),
            torch.tensor(1/pred_box_vars[-1:]))
        
        negative_log_prob = -pred_multivariate_normal_dists.log_prob(torch.tensor(gt_box_means[:-1]))
        negative_log_prob += -pred_von_mises_dists.log_prob(torch.tensor(gt_box_means[-1:])).squeeze()
        
        self.value_list.append(float(negative_log_prob))

    def add_fp(self, score):
        print('invalid')
        raise NotImplementedError

    # Ignorance REG Score Mean
    def mean(self):
        return np.mean(self.value_list)
