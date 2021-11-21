import numpy as np
from scoring.scoring_rule import ScoringRule

# class UncertaintyMetrics:
#     POINT_THRESHOLDS = [0, 100, 250, 9999]
#     DIST_THRESHOLDS = [0, 20, 40, 99]
#     data = {}
#     def __init__(self):
#         self.data = {}
#         self.distance = {}

#     def init_class(self, class_name):
#         self.data[class_name] = {
#             'point_filter': {
#                 0: 12
#             },
#             'distance_filter': []
#         }

#     def add_preds(self, class_name, tp_list, fp_ml_list, fp_bg_list):
#         if class_name not in self.data:
#             self.init_class(class_name)


class ShannonEntropy(ScoringRule):
    MIN_POINTS = 250
    def add_tp(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.tp_value_list = np.array([obj.data['shannon_entropy'] for obj in pred_list[pt_filter]])

    def add_dup(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.dup_value_list = np.array([obj.data['shannon_entropy'] for obj in pred_list[pt_filter]])

    def add_loc_err(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.loc_err_value_list = np.array([obj.data['shannon_entropy'] for obj in pred_list[pt_filter]])

    def add_fp(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.fp_value_list = np.array([obj.data['shannon_entropy'] for obj in pred_list[pt_filter]])

class AleatoricEntropy(ScoringRule):
    MIN_POINTS = 250
    def add_tp(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.tp_value_list = np.array([obj.data['aleatoric_entropy'] for obj in pred_list[pt_filter]])

    def add_dup(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.dup_value_list = np.array([obj.data['aleatoric_entropy'] for obj in pred_list[pt_filter]])

    def add_loc_err(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.loc_err_value_list = np.array([obj.data['aleatoric_entropy'] for obj in pred_list[pt_filter]])

    def add_fp(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.fp_value_list = np.array([obj.data['aleatoric_entropy'] for obj in pred_list[pt_filter]])

class MutualInfo(ScoringRule):
    MIN_POINTS = 250
    def add_tp(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.tp_value_list = np.array([obj.data['mutual_info'] for obj in pred_list[pt_filter]])

    def add_dup(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.dup_value_list = np.array([obj.data['mutual_info'] for obj in pred_list[pt_filter]])

    def add_loc_err(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.loc_err_value_list = np.array([obj.data['mutual_info'] for obj in pred_list[pt_filter]])

    def add_fp(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.fp_value_list = np.array([obj.data['mutual_info'] for obj in pred_list[pt_filter]])

class EpistemicTotalVar(ScoringRule):
    MIN_POINTS = 250
    def add_tp(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.tp_value_list = np.array([obj.data['epistemic_total_var'] for obj in pred_list[pt_filter]])

    def add_dup(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.dup_value_list = np.array([obj.data['epistemic_total_var'] for obj in pred_list[pt_filter]])

    def add_loc_err(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.loc_err_value_list = np.array([obj.data['epistemic_total_var'] for obj in pred_list[pt_filter]])

    def add_fp(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.fp_value_list = np.array([obj.data['epistemic_total_var'] for obj in pred_list[pt_filter]])

class AleatoricTotalVar(ScoringRule):
    MIN_POINTS = 250
    def add_tp(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.tp_value_list = np.array([obj.data['aleatoric_total_var'] for obj in pred_list[pt_filter]])

    def add_dup(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.dup_value_list = np.array([obj.data['aleatoric_total_var'] for obj in pred_list[pt_filter]])

    def add_loc_err(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.loc_err_value_list = np.array([obj.data['aleatoric_total_var'] for obj in pred_list[pt_filter]])

    def add_fp(self, pred_list):
        if len(pred_list) == 0:
            return
        pt_filter = np.array([obj.data['num_pts_in_pred'] for obj in pred_list]) > self.MIN_POINTS
        self.fp_value_list = np.array([obj.data['aleatoric_total_var'] for obj in pred_list[pt_filter]])
