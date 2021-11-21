import numpy as np
from scoring.scoring_rule import ScoringRule

class DMM(ScoringRule):
    def get_pos_matches(self, gt_list, pred_list):
        pred_box_means = np.array([obj.data['boxes_lidar'] for obj in pred_list]).reshape(-1,7,1)
        pred_box_vars = np.array([obj.data['pred_vars'] for obj in pred_list]).reshape(-1,7,1)
        gt_box_means = np.array([gt_list[int(obj.matched_idx)].data['gt_boxes'] for obj in pred_list]).reshape(-1,7,1)
        
        target_minus_pred = gt_box_means - pred_box_means
        bool_mask = target_minus_pred[:,-1] > np.pi
        target_minus_pred[:,-1][bool_mask] -= np.pi*2
        bool_mask = target_minus_pred[:,-1] < -np.pi
        target_minus_pred[:,-1][bool_mask] += np.pi*2
        
        pred_vars_minus_actual = pred_box_vars - \
            np.matmul(target_minus_pred,target_minus_pred.transpose(0,2,1))

        pred_vars_minus_actual_norm = np.linalg.norm(pred_vars_minus_actual, ord=1, axis=(1,2))
        target_minus_pred_norm = np.linalg.norm(np.abs(target_minus_pred), ord=1, axis=(1,2))
        return target_minus_pred_norm + pred_vars_minus_actual_norm

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
