import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.von_mises import VonMises
from scoring.scoring_rule import ScoringRule

class NLLREG(ScoringRule):        
    def add_tp(self, gt_list, pred_list):
        pred_box_means = np.array([obj.data['boxes_lidar'] for obj in pred_list])
        pred_box_vars = np.array([obj.data['pred_vars'] for obj in pred_list])
        gt_box_means = np.array([gt_list[int(obj.matched_idx)].data['gt_boxes'] for obj in pred_list])

        pred_var_mat = [np.diag(i[:-1]) for i in pred_box_vars]
        pred_multivariate_normal_dists = MultivariateNormal(
                torch.tensor(pred_box_means[:,:-1]),
                torch.tensor(pred_var_mat) + 1e-2 * torch.eye(pred_box_vars.shape[1]-1))

        pred_von_mises_dists = VonMises(
                torch.tensor(pred_box_means[:,-1:]),
                torch.tensor(1/pred_box_vars[:,-1:]))

        negative_log_prob = -pred_multivariate_normal_dists.log_prob(torch.tensor(gt_box_means[:,:-1]))
        negative_log_prob += -pred_von_mises_dists.log_prob(torch.tensor(gt_box_means[:,-1:])).squeeze()
        
        self.tp_value_list = negative_log_prob.numpy()