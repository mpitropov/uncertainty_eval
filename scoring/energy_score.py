import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.von_mises import VonMises
from scoring.scoring_rule import ScoringRule

class ENERGYSCORE(ScoringRule):
    def calc_energy_score(self, distr, gt_box_means=None, periodic_distr=False):
        # Energy Score.
        sample_set = distr.sample((1000,)) # github code is 1001, paper mentions 1000
        sample_set_1 = sample_set[:-1]
        sample_set_2 = sample_set[1:]

        # Following is how it's described in paper
        # Github implementation first norm term: (sample_set_1 - gt_box_means) 
        # https://github.com/asharakeh/probdet/blob/master/src/core/evaluation_tools/scoring_rules.py#L156-L161
        sample1_minus_sample2 = sample_set_1 - sample_set_2
        if periodic_distr:
            bool_mask = sample1_minus_sample2 > np.pi
            sample1_minus_sample2[bool_mask] -= np.pi*2
            bool_mask = sample1_minus_sample2 < -np.pi
            sample1_minus_sample2[bool_mask] += np.pi*2

        energy_score = torch.norm(sample1_minus_sample2, dim=2).mean(0)

        if gt_box_means is not None:
            sample_minus_gt = sample_set - gt_box_means
            if periodic_distr:
                bool_mask = sample_minus_gt > np.pi
                sample_minus_gt[bool_mask] -= np.pi*2
                bool_mask = sample_minus_gt < -np.pi
                sample_minus_gt[bool_mask] += np.pi*2
            energy_score = torch.norm(sample_minus_gt, dim=2).mean(0) - \
                        0.5 * \
                        energy_score

        return energy_score.numpy()

    def get_pos_matches(self, gt_list, pred_list):
        pred_box_means = np.array([obj.data['boxes_lidar'] for obj in pred_list])
        pred_box_vars = np.array([obj.data['pred_vars'] for obj in pred_list])
        gt_box_means = np.array([gt_list[int(obj.matched_idx)].data['gt_boxes'] for obj in pred_list])

        pred_var_mat = [np.diag(i[:-1]) for i in pred_box_vars]
        pred_multivariate_normal_dists = MultivariateNormal(
                torch.tensor(pred_box_means[:,:-1]),
                torch.tensor(pred_var_mat) + 1e-2 * torch.eye(pred_box_vars.shape[1]-1))

        pred_von_mises_dists = VonMises(
                torch.tensor(pred_box_means[:,-1:]),
                torch.tensor(1/(pred_box_vars[:,-1:] + 1e-2)))

        energy_score = self.calc_energy_score(pred_multivariate_normal_dists, gt_box_means[:,:-1])
        energy_score += self.calc_energy_score(pred_von_mises_dists, gt_box_means[:,-1:], periodic_distr=True)
        return energy_score

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

    def add_fp(self, pred_list):
        if len(pred_list) == 0:
            return
        print('fp energy score calculation might take some time')
        pred_box_means = np.array([obj.data['boxes_lidar'] for obj in pred_list])
        pred_box_vars = np.array([obj.data['pred_vars'] for obj in pred_list])

        pred_var_mat = [np.diag(i[:-1]) for i in pred_box_vars]

        # Divide list by 2 in order to reduce memory requirements
        split = int(len(pred_box_means)/2)

        # Split is small, don't need to split
        if split < 100:
            pred_multivariate_normal_dists = MultivariateNormal(
                    torch.tensor(pred_box_means[:,:-1]),
                    torch.tensor(pred_var_mat) + 1e-2 * torch.eye(pred_box_vars.shape[1]-1))

            pred_von_mises_dists = VonMises(
                    torch.tensor(pred_box_means[:,-1:]),
                    torch.tensor(1/(pred_box_vars[:,-1:] + 1e-2)))

            fp_energy_score = self.calc_energy_score(pred_multivariate_normal_dists)
            fp_energy_score += self.calc_energy_score(pred_von_mises_dists, periodic_distr=True)

            self.fp_value_list = fp_energy_score
            return

        print('fp energy score calculation: split 1')
        pred_multivariate_normal_dists = MultivariateNormal(
                torch.tensor(pred_box_means[:split,:-1]),
                torch.tensor(pred_var_mat[:split]) + 1e-2 * torch.eye(pred_box_vars.shape[1]-1))

        pred_von_mises_dists = VonMises(
                torch.tensor(pred_box_means[:split,-1:]),
                torch.tensor(1/(pred_box_vars[:split,-1:] + 1e-2)))

        fp_energy_score_1 = self.calc_energy_score(pred_multivariate_normal_dists)
        fp_energy_score_1 += self.calc_energy_score(pred_von_mises_dists, periodic_distr=True)

        print('fp energy score calculation: split 2')
        pred_multivariate_normal_dists = MultivariateNormal(
                torch.tensor(pred_box_means[split:,:-1]),
                torch.tensor(pred_var_mat[split:]) + 1e-2 * torch.eye(pred_box_vars.shape[1]-1))

        pred_von_mises_dists = VonMises(
                torch.tensor(pred_box_means[split:,-1:]),
                torch.tensor(1/(pred_box_vars[split:,-1:] + 1e-2)))

        fp_energy_score_2 = self.calc_energy_score(pred_multivariate_normal_dists)
        fp_energy_score_2 += self.calc_energy_score(pred_von_mises_dists, periodic_distr=True)

        self.fp_value_list = np.concatenate((fp_energy_score_1, fp_energy_score_2))