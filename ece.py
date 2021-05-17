import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import vonmises

def calculate_ece(conf_mat, num_bins, name):
    # Calculate Precision from TP and FP
    accuracy = []
    bin_total = []
    for bin_mat in conf_mat:
        # If the bin is empty
        if bin_mat['TP'] + bin_mat['FP'] == 0:
            accuracy.append(0)
            bin_total.append(0)
            continue
        # Calculate precision
        precision = bin_mat['TP'] / (bin_mat['TP'] + bin_mat['FP'])
        accuracy.append(precision)
        bin_total.append(bin_mat['TP'] + bin_mat['FP'])

    accuracy = accuracy[1:] # the first element is always going to be zero
    bin_total = bin_total[1:]
    accuracy = np.array(accuracy)
    bin_upper_range = np.arange(0.5*(1/num_bins), 1.0 + 0.5*(1/num_bins), 1/num_bins)
    ece = np.sum(np.array(bin_total) * np.abs(accuracy - bin_upper_range) / np.sum(bin_total))
    # key = ':'.join([car_filter.name(), car_filter.name(i)])
    key = ':'.join([name])
    
    return accuracy, ece, key

def calculate_ece_reg(gt_list, pred_list, histogram_bin_count=15):
    pred_means = [obj.data['boxes_lidar'] for obj in pred_list]
    pred_vars = [obj.data['pred_vars'] for obj in pred_list]
    gt_means = [gt_list[int(obj.matched_idx)].data['gt_boxes'] for obj in pred_list] 

    # Compute regression calibration errors. False negatives cant be evaluated since
    # those do not have ground truth.
    all_predicted_means = torch.tensor(pred_means)

    all_predicted_covariances = torch.tensor(pred_vars)

    all_predicted_gt = torch.tensor(gt_means)

    # The assumption of uncorrelated components is not accurate, especially when estimating full
    # covariance matrices. However, using scipy to compute multivariate cdfs is very very
    # time consuming for such large amounts of data.
    reg_maximum_calibration_error = []
    reg_expected_calibration_error = []

    # Regression calibration is computed for every box dimension
    # separately, and averaged after.
    for box_dim in range(all_predicted_gt.shape[1]):
        all_predicted_means_current_dim = all_predicted_means[:, box_dim]
        all_predicted_gt_current_dim = all_predicted_gt[:, box_dim]
        all_predicted_covariances_current_dim = all_predicted_covariances[:, box_dim]
        if box_dim == all_predicted_gt.shape[1] - 1:
            #cdf(x, kappa, loc=0, scale=1)
            all_predicted_scores = vonmises.cdf(
                all_predicted_gt_current_dim, 
                1/all_predicted_covariances_current_dim, 
                loc=all_predicted_means_current_dim, 
                scale=torch.sqrt(all_predicted_covariances_current_dim))
            all_predicted_scores = torch.tensor(all_predicted_scores)
        else:
            normal_dists = torch.distributions.Normal(
                all_predicted_means_current_dim,
                scale=torch.sqrt(all_predicted_covariances_current_dim))
            all_predicted_scores = normal_dists.cdf(
                all_predicted_gt_current_dim)

        reg_calibration_error = []
        histogram_bin_step_size = 1.0 / histogram_bin_count
        for i in torch.arange(
                0.0,
                1.0 - histogram_bin_step_size,
                histogram_bin_step_size):
            # Get number of elements in bin
            elements_in_bin = (
                all_predicted_scores < (i + histogram_bin_step_size))
            num_elems_in_bin_i = elements_in_bin.type(
                torch.FloatTensor).sum()

            # Compute calibration error from "Accurate uncertainties for deep
            # learning using calibrated regression" paper.
            reg_calibration_error.append(
                (num_elems_in_bin_i / all_predicted_scores.shape[0] - (i + histogram_bin_step_size)) ** 2)

        calibration_error = torch.stack(reg_calibration_error)
        reg_maximum_calibration_error.append(calibration_error.max())
        reg_expected_calibration_error.append(calibration_error.mean())


    return np.array(reg_maximum_calibration_error).mean(), np.array(reg_expected_calibration_error).mean()

# Plot Reliability Diagram
def plot_reliability(acc, ece, save_path):
    interval = 1 / len(acc)
    x = np.arange(interval/2, 1+interval/2, 1/len(acc))

    plt.figure(figsize=(3,3))
    plt.bar(x, acc, width=0.08, edgecolor='k')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.text(0,1.01,'ECE={}'.format(str(ece)[:5]))

    plt.plot([0,1], [0,1], 'k--')
    plt.tight_layout()
    plt.savefig(save_path)
#     plt.show()
