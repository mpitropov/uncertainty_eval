
import numpy as np
import matplotlib.pyplot as plt

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