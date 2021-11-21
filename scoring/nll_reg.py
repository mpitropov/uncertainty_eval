from contextlib import suppress
import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.von_mises import VonMises
from scoring.scoring_rule import ScoringRule

class NLLREG(ScoringRule):        
    def get_pos_matches(self, gt_list, pred_list):
        pred_box_means = np.array([obj.data['boxes_lidar'] for obj in pred_list])
        pred_box_vars = np.array([obj.data['pred_vars'] for obj in pred_list])
        gt_box_means = np.array([gt_list[int(obj.matched_idx)].data['gt_boxes'] for obj in pred_list])

        pred_var_mat = [np.diag(i[:-1]) for i in pred_box_vars]
        pred_multivariate_normal_dists = MultivariateNormal(
            torch.tensor(pred_box_means[:,:-1], dtype=torch.double),
            torch.tensor(pred_var_mat, dtype=torch.double) + 1e-2 * torch.eye(pred_box_vars.shape[1]-1, dtype=torch.double))

        pred_von_mises_dists = VonMises(
            torch.tensor(pred_box_means[:,-1:], dtype=torch.double),
            torch.tensor(1/(pred_box_vars[:,-1:] + 1e-2), dtype=torch.double))

        negative_log_prob = -pred_multivariate_normal_dists.log_prob(torch.tensor(gt_box_means[:,:-1], dtype=torch.double))
        negative_log_prob += -pred_von_mises_dists.log_prob(torch.tensor(gt_box_means[:,-1:], dtype=torch.double)).squeeze()

        return negative_log_prob.numpy()

    def add_tp(self, gt_list, pred_list):
        if len(pred_list) == 0:
            return
        self.tp_value_list = self.get_pos_matches(gt_list, pred_list)

    def add_dup(self, gt_list, pred_list):
        if len(pred_list) == 0:
            return
        self.dup_value_list = self.get_pos_matches(gt_list, pred_list)

    def add_loc_err(self, gt_list, pred_list):
        if len(pred_list) == 0:
            return
        self.loc_err_value_list = self.get_pos_matches(gt_list, pred_list)

class NLLREG_Calibration():
    def __init__(self):
        self.init = []

    def calibrate(self, gt_list, pred_list):
        pred_box_means = np.array([obj.data['boxes_lidar'] for obj in pred_list])
        pred_box_vars = np.array([obj.data['pred_vars'] for obj in pred_list])
        gt_box_means = np.array([gt_list[int(obj.matched_idx)].data['gt_boxes'] for obj in pred_list])

        # normalize predicted mean angle to [gt angle - pi, gt angle + pi]
        for i in range(len(pred_box_means)):
            if pred_box_means[i][6] > gt_box_means[i][6] + np.pi:
                pred_box_means[i][6] -= 2*np.pi
            if pred_box_means[i][6] < gt_box_means[i][6] - np.pi:
                pred_box_means[i][6] += 2*np.pi

        reg_var_names = ['x', 'y', 'z', 'l', 'w', 'h', 'rz']
        np_eps = np.finfo(float).eps

        best_T_list = []

        print('Finding best T for regression calibration')
        for i in range(7):
            lowest_nll = np.Inf
            best_T = None
            for curr_T in np.arange(start = 0.05, stop = 2.0, step = 0.05):
                if i != 6:
                    dists = Normal(
                        torch.tensor(pred_box_means[:,i], dtype=torch.double),
                        torch.tensor(np.sqrt(pred_box_vars[:,i] / curr_T), dtype=torch.double)
                    )
                    var = -dists.log_prob(torch.tensor(gt_box_means[:,i], dtype=torch.double))
                else:
                    dists = VonMises(
                            torch.tensor(pred_box_means[:,-1:], dtype=torch.double),
                            torch.tensor(1/(pred_box_vars[:,-1:] / curr_T), dtype=torch.double))
                    var = -dists.log_prob(torch.tensor(gt_box_means[:,-1:], dtype=torch.double)).squeeze()
                nll_mean = var.mean()
                if nll_mean < lowest_nll:
                    lowest_nll = nll_mean
                    best_T = curr_T
                    print('    Found new low NLL: lowest_nll, best_T', lowest_nll, best_T)

            print('NLL REG Calibration', reg_var_names[i], 'best_T:', best_T)
            best_T_list.append(float(best_T.round(2)))

        return best_T_list
