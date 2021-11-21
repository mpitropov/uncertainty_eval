import pickle
import numpy as np
from scoring.scoring_rule import ScoringRule

from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

class NLLCLF(ScoringRule):
    def add_tp(self, pred_list):
        if len(pred_list) == 0:
            return
        self.tp_value_list = [-np.log(obj.pred_score) for obj in pred_list]

    def add_dup(self, pred_list):
        if len(pred_list) == 0:
            return
        self.dup_value_list = [-np.log(obj.pred_score) for obj in pred_list]

    def add_loc_err(self, pred_list):
        if len(pred_list) == 0:
            return

        # After calibration can cause some scores to be 0
        eps = np.finfo(float).eps

        # # Assume correctly classified (FP_ML)
        # self.loc_err_value_list = [-np.log(obj.pred_score) for obj in pred_list]

        # We can't assume that it is correctly classified
        self.loc_err_value_list = [-np.log(max(obj.data['score_all'][obj.gt_label - 1], eps)) for obj in pred_list]

    def add_bg_tp(self, pred_list):
        self.fp_value_list = [-np.log(obj.data['score_all'][-1]) for obj in pred_list]

    def add_fp(self, pred_list):
        self.fp_value_list = [-np.log(1.0 - obj.pred_score) for obj in pred_list]

class NLLCLF_Calibration():
    def __init__(self):
        self.init = []

    def calibrate(self, tp_pred_list, fp_pred_list):
        np_eps = np.finfo(float).eps
        lowest_nll = np.Inf
        best_T = None
        print('Finding best T for classification calibration')
        for curr_T in np.arange(start = 0.05, stop = 3.0, step = 0.05):
            tp_nll = [-np.log(max(max(softmax(obj.data['score_all'] / curr_T)[:3]), np_eps)) for obj in tp_pred_list]
            fp_nll = [-np.log(max(softmax(obj.data['score_all'] / curr_T)[3], np_eps)) for obj in fp_pred_list]
            nll_mean = np.concatenate((tp_nll,fp_nll)).mean()
            if nll_mean < lowest_nll:
                lowest_nll = nll_mean
                best_T = curr_T
                print('    Found new low NLL: lowest_nll, best_T', lowest_nll, best_T)

        print('NLL CLF Calibration best_T:', best_T)
        return best_T.round(2)

    def calibrate_isotonic_reg(self, tp_pred_list, fp_pred_list):
        X = []
        y = []
        for obj in tp_pred_list:
            X.append(obj.pred_score)
            y.append(1.0)
        for obj in fp_pred_list:
            X.append(obj.pred_score)
            y.append(0.0)
        iso_reg_model = IsotonicRegression().fit(X, y)

        # save the model to disk
        filename = 'mimo_a_isotonic_reg.sav'
        pickle.dump(iso_reg_model, open(filename, 'wb'))

    def calibrate_multiclass_isotonic_reg(self, tp_pred_list, fp_pred_list):
        # LinearSVM is the default base estimator but not working
        # base_clf = GaussianNB()
        base_clf = MLPClassifier()
        calibrated_clf = CalibratedClassifierCV(base_estimator=base_clf, method='isotonic')
        X = []
        y = []
        for obj in tp_pred_list:
            X.append(obj.data['score_all'])
            true_class = np.argmax(obj.data['score_all'][:3]) # From foreground class
            y.append(true_class)
        for obj in fp_pred_list:
            X.append(obj.data['score_all'])
            y.append(3)
        multiclass_iso_reg_model = calibrated_clf.fit(X, y)

        # save the model to disk
        filename = 'mimo_a_multiclass_isotonic_reg.sav'
        pickle.dump(multiclass_iso_reg_model, open(filename, 'wb'))