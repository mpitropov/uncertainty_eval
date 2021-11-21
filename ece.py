import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import vonmises

def calculate_ece(conf_mat, num_bins, name):
    # Calculate Precision from TP and FP
    accuracy = []
    confidence = []
    bin_totals = []
    for bin_mat in conf_mat:
        # If the bin is empty
        if bin_mat['TP'] + bin_mat['FP'] == 0:
            accuracy.append(0)
            confidence.append(0)
            bin_totals.append(0)
            continue
        # Calculate precision
        precision = bin_mat['TP'] / (bin_mat['TP'] + bin_mat['FP'])
        accuracy.append(precision)
        confidence.append(np.sum(bin_mat['confidence_list']) / (bin_mat['TP'] + bin_mat['FP']))
        bin_totals.append(bin_mat['TP'] + bin_mat['FP'])

    accuracy = np.array(accuracy[1:]) # the first element is always going to be zero
    confidence = np.array(confidence[1:])
    bin_totals = bin_totals[1:]
    bin_middle_range = np.arange(0.5*(1.0/num_bins), 1.0 + 0.5*(1.0/num_bins), 1.0/num_bins)
    ece = np.sum(np.array(bin_totals) * np.abs(accuracy - confidence) / np.sum(bin_totals))
    max_ce = np.max(np.abs(accuracy - confidence))
    weight_per_bin = np.array(bin_totals) / np.sum(bin_totals) # percentage of how many objects are in each bin
    # print('bin_middle_range', bin_middle_range)
    # print('bin_totals', bin_totals)
    # print('bin_totals sum', np.sum(bin_totals))
    # print('confidence', confidence)
    # print('bin_totals weighted ', np.array(bin_totals) / np.sum(bin_totals))
    # print('acc - conf', np.abs(accuracy - confidence))
    # print('multiplied', np.array(bin_totals) * np.abs(accuracy - confidence))
    # print('divided', np.array(bin_totals) * np.abs(accuracy - confidence) / np.sum(bin_totals))
    # print('ece', ece)
    # print('max_ce', max_ce)
    # exit(0)
    key = ':'.join([name])
    
    return accuracy, confidence, ece, max_ce, weight_per_bin, key

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

    # acc added
    acc_list_list = []

    # Regression calibration is computed for every box dimension
    # separately, and averaged after.
    for box_dim in range(all_predicted_gt.shape[1]):
        all_predicted_means_current_dim = all_predicted_means[:, box_dim]
        all_predicted_gt_current_dim = all_predicted_gt[:, box_dim]
        all_predicted_covariances_current_dim = all_predicted_covariances[:, box_dim]
        if box_dim == all_predicted_gt.shape[1] - 1:
            # cdf(x, kappa, loc=0, scale=1)
            # normalize predicted mean angle to [gt angle - pi, gt angle + pi]
            for i in range(len(all_predicted_means_current_dim)):
                if all_predicted_means_current_dim[i] > all_predicted_gt_current_dim[i] + np.pi:
                    all_predicted_means_current_dim[i] -= 2*np.pi
                if all_predicted_means_current_dim[i] < all_predicted_gt_current_dim[i] - np.pi:
                    all_predicted_means_current_dim[i] += 2*np.pi

            # print('all_predicted_gt_current_dim min', all_predicted_gt_current_dim.min())
            # print('all_predicted_gt_current_dim max', all_predicted_gt_current_dim.max())
            # print('all_predicted_means_current_dim min', all_predicted_means_current_dim.min())
            # print('all_predicted_means_current_dim max', all_predicted_means_current_dim.max())
            # print('here')
            # print('all_predicted_gt_current_dim', all_predicted_gt_current_dim)
            # print('all_predicted_means_current_dim', all_predicted_means_current_dim)
            # print('all_predicted_covariances_current_dim', all_predicted_covariances_current_dim)

            all_predicted_scores = vonmises.cdf(
                all_predicted_means_current_dim,
                1/all_predicted_covariances_current_dim, 
                loc=all_predicted_gt_current_dim)
            # print('all_predicted_means_current_dim', all_predicted_means_current_dim[:5])
            # print('all_predicted_gt_current_dim', all_predicted_gt_current_dim[:5])
            # print('all_predicted_scores', all_predicted_scores[:5])
            # print('all_predicted_scores min', all_predicted_scores.min())
            # print('all_predicted_scores max', all_predicted_scores.max())
            # print('all_predicted_scores mean', all_predicted_scores.mean())

            all_predicted_scores = torch.tensor(all_predicted_scores)
        else:
            normal_dists = torch.distributions.Normal(
                all_predicted_means_current_dim,
                scale=torch.sqrt(all_predicted_covariances_current_dim))
            all_predicted_scores = normal_dists.cdf(
                all_predicted_gt_current_dim)

        reg_calibration_error = []
        acc_list = []
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
            # Following equation 9 in the paper with weights adding to 1.0
            p_j = (i + histogram_bin_step_size) # Confidence level
            p_hat_j = num_elems_in_bin_i / all_predicted_scores.shape[0] # Empirical frequency
            w_j = num_elems_in_bin_i / all_predicted_scores.shape[0] # How many items are in the bin
            reg_calibration_error.append( w_j * (p_j - p_hat_j) ** 2 )

            # Add the bin accuracy (empirical frequency)
            acc_list.append(p_hat_j)

        calibration_error = torch.stack(reg_calibration_error)
        reg_maximum_calibration_error.append(calibration_error.max())
         # Take sum instead because our weights add up to 1.0
        reg_expected_calibration_error.append(calibration_error.sum())
        acc_list_list.append(acc_list)

    return np.array(reg_maximum_calibration_error).mean(), \
            np.array(reg_expected_calibration_error).mean(), \
            np.array(acc_list_list)

def plot_reliability_clf(class_name, acc, conf, ece, max_ce, weight_per_bin, save_path):
    interval = 1 / len(acc)
    x = np.arange(interval/2, 1+interval/2, interval)

    plt.figure(figsize=(4,4))
    plt.grid(linestyle='dotted')
    plt.bar(x, acc, width=interval, edgecolor='k', label='Prediction')
    # Plotting the gap requires you to calculate the gap and use the acc as the bottom position
    gap = conf - acc
    # Create red colour with alpha value based on the weight of each bin
    rgba_colors = np.zeros((len(x),4))
    rgba_colors[:,0] = 1.0
    rgba_colors[:, 3] = weight_per_bin
    plt.bar(x, gap, bottom=acc, width=interval, edgecolor='red', color=rgba_colors, label='Gap')
    # plt.bar(x, gap, bottom=acc, width=interval, edgecolor='red', color='red', alpha=0.3, label='Gap')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.xlim([0,1])
    plt.ylim([0,1])

    plt.plot([],[], ' ', label='ECE={}'.format(ece.round(3)))
    plt.plot([],[], ' ', label='MAXCE={}'.format(max_ce.round(3)))
    plt.plot([0,1], [0,1], 'k--', label='Perfect prediction')
    plt.legend(loc='upper left', framealpha=0.3)
    plt.title(class_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_reliability_reg(class_name, acc, ece, save_path):
    interval = 1 / (len(acc[0]) + 1)
    x = np.arange(0, 1.0 + interval/2, interval)
    plt.figure(figsize=(5,5))

    labels = ['x', 'y', 'z', 'l', 'w', 'h', 'rz']
    for i in range(len(acc)):
        acc_plot = np.concatenate(([0], acc[i], [1.0]))
        plt.plot(x, acc_plot, label=labels[i])
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.text(0,1.01,'ECE={}'.format(str(ece)[:5]))

    plt.plot([0,1], [0,1], 'k--', label='Perfect prediction')
    plt.legend(loc='upper left', framealpha=0.3)
    plt.title(class_name)
    plt.tight_layout()
    plt.grid(linestyle='dotted')
    plt.savefig(save_path, dpi=300)
    plt.close()
