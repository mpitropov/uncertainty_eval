import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.von_mises import VonMises

from scoring_rule import ScoringRule

class ENERGYSCORE(ScoringRule):
    def calc_energy_score(self, distr, gt_box_means=None):
        # Energy Score.
        sample_set = distr.sample((1000,)) # github code is 1001, paper mentions 1000
        sample_set_1 = sample_set[:-1]
        sample_set_2 = sample_set[1:]

        # Following is how it's described in paper
        # Github implementation first norm term: (sample_set_1 - gt_box_means) 
        # https://github.com/asharakeh/probdet/blob/master/src/core/evaluation_tools/scoring_rules.py#L156-L161
        energy_score = torch.norm((sample_set_1 - sample_set_2), dim=1).mean(0)
        
        if gt_box_means is not None:
            energy_score = torch.norm((sample_set - gt_box_means), dim=1).mean(0) - \
                        0.5 * \
                        energy_score
        
        return energy_score.item()

    def add_tp(self, gt_box_means, pred_box_means, pred_box_vars):
        pred_multivariate_normal_dists = MultivariateNormal(
            torch.tensor(pred_box_means[:-1]),
            torch.diag(torch.tensor(pred_box_vars[:-1])) + 1e-2 * torch.eye(pred_box_vars[:-1].shape[0]))

        # concentration (a reciprocal measure of dispersion, so 1/k is analogous to variance). 
        pred_von_mises_dists = VonMises(
            torch.tensor(pred_box_means[-1:]),
            torch.tensor(1/pred_box_vars[-1:]))

        energy_score = self.calc_energy_score(pred_multivariate_normal_dists, gt_box_means[:-1])
        energy_score += self.calc_energy_score(pred_von_mises_dists, gt_box_means[-1:])

        self.value_list.append(float(energy_score))

    def add_fp(self, pred_box_means, pred_box_vars):
        pred_multivariate_normal_dists = MultivariateNormal(
            torch.tensor(pred_box_means[:-1]),
            torch.diag(torch.tensor(pred_box_vars[:-1])) + 1e-2 * torch.eye(pred_box_vars[:-1].shape[0]))
        
        pred_von_mises_dists = VonMises(
            torch.tensor(pred_box_means[-1:]),
            torch.tensor(1/pred_box_vars[-1:]))

        fp_energy_score = self.calc_energy_score(pred_multivariate_normal_dists)
        fp_energy_score += self.calc_energy_score(pred_von_mises_dists)

        self.value_list.append(float(fp_energy_score))

    # Energy Score Mean
    def mean(self):
        return np.mean(self.value_list)
