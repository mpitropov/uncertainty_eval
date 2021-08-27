from contextlib import suppress
import numpy as np
import torch
from torch.distributions.normal import Normal
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
                torch.tensor(pred_box_means[:,:-1], dtype=torch.double),
                torch.tensor(pred_var_mat, dtype=torch.double) + 1e-2 * torch.eye(pred_box_vars.shape[1]-1, dtype=torch.double))

        pred_von_mises_dists = VonMises(
                torch.tensor(pred_box_means[:,-1:], dtype=torch.double),
                torch.tensor(1/pred_box_vars[:,-1:], dtype=torch.double))

        negative_log_prob = -pred_multivariate_normal_dists.log_prob(torch.tensor(gt_box_means[:,:-1], dtype=torch.double))
        negative_log_prob += -pred_von_mises_dists.log_prob(torch.tensor(gt_box_means[:,-1:], dtype=torch.double)).squeeze()

        self.tp_value_list = negative_log_prob.numpy()

class NLLREG_Calibration():
    def __init__(self):
        self.init = []

    def calibrate(self, gt_list, pred_list):
        pred_box_means = np.array([obj.data['boxes_lidar'] for obj in pred_list])
        pred_box_vars = np.array([obj.data['pred_vars'] for obj in pred_list])
        gt_box_means = np.array([gt_list[int(obj.matched_idx)].data['gt_boxes'] for obj in pred_list])

        pred_normal_dists = []
        for i in range(6):
                pred_normal_dists.append(Normal(
                        torch.tensor(pred_box_means[:,i], dtype=torch.double),
                        torch.tensor(np.sqrt(pred_box_vars[:,i]), dtype=torch.double) + 1e-2

                ))

        for i in range(7):
                print('NLL index', i)
                T_bot = 0.001
                T_top = 5.0
                nll_bot = None
                nll_top = None
                chosen_T = None
                increment = T_top / 2
                for j in range(30):
                        if nll_bot == None:
                                if i < 6:
                                        dists = Normal(
                                                torch.tensor(pred_box_means[:,i], dtype=torch.double),
                                                torch.tensor(np.sqrt(pred_box_vars[:,i] / T_bot), dtype=torch.double)

                                        )
                                        var = -dists.log_prob(torch.tensor(gt_box_means[:,i], dtype=torch.double))
                                else:
                                        # dists = VonMises(
                                        #         torch.tensor(pred_box_means[:,-1:], dtype=torch.double),
                                        #         torch.tensor(1/(pred_box_vars[:,-1:] / T_bot), dtype=torch.double))
                                        # var = -dists.log_prob(torch.tensor(gt_box_means[:,-1:], dtype=torch.double)).squeeze()
                                        dists = VonMises(
                                                torch.tensor(gt_box_means[:,-1:], dtype=torch.double),
                                                torch.tensor(1/(pred_box_vars[:,-1:] / T_bot), dtype=torch.double))
                                        var = -dists.log_prob(torch.tensor(pred_box_means[:,-1:], dtype=torch.double)).squeeze()
                                        print('nll bot sum', var.numpy().sum())
                                        print('nll bot mean', var.numpy().mean())
                                nll_bot = var.numpy().sum()
                                # print('nll bot', nll_bot, T_bot)
                        if nll_top == None:
                                if i < 6:
                                        dists = Normal(
                                                torch.tensor(pred_box_means[:,i], dtype=torch.double),
                                                torch.tensor(np.sqrt(pred_box_vars[:,i] / T_top), dtype=torch.double)

                                        )
                                        var = -dists.log_prob(torch.tensor(gt_box_means[:,i], dtype=torch.double))
                                else:
                                        # dists = VonMises(
                                        #         torch.tensor(pred_box_means[:,-1:], dtype=torch.double),
                                        #         torch.tensor(1/(pred_box_vars[:,-1:] / T_top), dtype=torch.double))
                                        # var = -dists.log_prob(torch.tensor(gt_box_means[:,-1:], dtype=torch.double)).squeeze()
                                        dists = VonMises(
                                                torch.tensor(gt_box_means[:,-1:], dtype=torch.double),
                                                torch.tensor(1/(pred_box_vars[:,-1:] / T_top), dtype=torch.double))
                                        var = -dists.log_prob(torch.tensor(pred_box_means[:,-1:], dtype=torch.double)).squeeze()
                                        print('nll top sum', var.numpy().sum())
                                        print('nll top mean', var.numpy().mean())
                                nll_top = var.numpy().sum()
                                # print('nll top', nll_top, T_top)
                        
                        chosen_T = T_bot + increment
                        # print('chosen_T', chosen_T)
                        if i < 6:
                                dists = Normal(
                                        torch.tensor(pred_box_means[:,i], dtype=torch.double),
                                        torch.tensor(np.sqrt(pred_box_vars[:,i] / chosen_T), dtype=torch.double)

                                )
                                var = -dists.log_prob(torch.tensor(gt_box_means[:,i], dtype=torch.double))
                        else:
                                # dists = VonMises(
                                #         torch.tensor(pred_box_means[:,-1:], dtype=torch.double),
                                #         torch.tensor(1/(pred_box_vars[:,-1:] / chosen_T), dtype=torch.double))
                                # var = -dists.log_prob(torch.tensor(gt_box_means[:,-1:], dtype=torch.double)).squeeze()
                                dists = VonMises(
                                        torch.tensor(gt_box_means[:,-1:], dtype=torch.double),
                                        torch.tensor(1/(pred_box_vars[:,-1:] / chosen_T), dtype=torch.double))
                                var = -dists.log_prob(torch.tensor(pred_box_means[:,-1:], dtype=torch.double)).squeeze()
                                print('nll mid sum', var.numpy().sum())
                                print('nll mid mean', var.numpy().mean())
                        nll_mid = var.numpy().sum()
                        # print('nll mid', nll_mid, chosen_T)
                        if nll_bot < nll_mid:
                                T_top = chosen_T
                                # print('T_top set to', T_top)
                        elif nll_top < nll_mid:
                                T_bot = chosen_T
                                # print('T_bot set to', T_bot)
                        # Unclear: bring in the bounds
                        else:
                                nll_bot = None # reset these
                                nll_top = None
                                T_diff = increment / 2
                                T_top -= T_diff
                                T_bot += T_diff
                                # print('T_top set to', T_top)
                                # print('T_bot set to', T_bot)
                        # update increment
                        increment /= 2
                print('chosen T', chosen_T)
