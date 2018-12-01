import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# static constants
HYPERPARAMS = ['learning_rate', 'total_num_iters', 'n_h', 'n_h_adv', 'dropout_rate', 'alpha']

# k gives the number of classes the protected variable can take on
def get_metrics(y_pred, y, z, hyperparams, k = 2, y_select = 0):
    metrics = dict()

    # add hyperparameters for experiment
    for i in range(len(hyperparams)):
        metrics[HYPERPARAMS[i]] = hyperparams[i]

    # performance metrics
    pred = predict(y_pred)
    metrics['accuracy'] = get_accuracy(pred, y)
    metrics['roc_auc'] = roc_auc_score(y, y_pred) # CAN TAKE IN ARRAYS?
    metrics['roc_curve_fpr'], metrics['roc_curve_tpr'], metrics['roc_curve_thresholds'] = roc_curve(y, y_pred, pos_label=1)

    # fairness metrics
    for i in range(k):
        metrics['conditional_roc_auc_' + str(i)] = roc_auc_score(y[z == i], y_pred[z == i])
        metrics['conditional_roc_curve_fpr'], metrics['conditional_roc_curve_tpr'], metrics['conditional_roc_curve_thresholds'] = \
        roc_curve(y[z == i], y_pred[z == i], pos_label=1)
        metrics['count_' + str(i)] = np.sum(z == i)
        metrics['y_hat_' + str(i)] = np.sum(pred[z == i] == y_select)/np.sum(z == i)
        metrics['accuracy_' + str(i)] = get_accuracy(pred[z == i], y[z == i]) # WATCH OUT - DOESN'T COUNT INSTANCES WHERE PREDICT i BUT ISN'T ACTUALLY i
        metrics['true_neg_' + str(i)], metrics['false_neg_' + str(i)], metrics['false_pos_' + str(i)], metrics['true_pos_' + str(i)]  = \
        confusion_matrix(pred[z == i], y[z == i])
        metrics['fp_' + str(i)] = metrics['false_pos_' + str(i)] / (metrics['false_pos_' + str(i)] + metrics['true_neg_' + str(i)])
        metrics['fn_' + str(i)] = metrics['false_neg_' + str(i)] / (metrics['false_neg_' + str(i)] + metrics['true_pos_' + str(i)])
        metrics['calibration_pos_' + str(i)] = np.sum((pred == 1) & (z == i) & (y == 1))/np.sum((pred == 1) & (z == i)) # p(y | pred = 1, z)
        metrics['calibration_neg_' + str(i)] = np.sum((pred == 0) & (z == i) & (y == 1))/np.sum((pred == 0) & (z == i)) # p(y | pred, z)

    for i in range(k):
        metrics['parity_gap_' + str(i)] = demographic_parity_gap(pred, z, i, y_select) # |prop(i) - prop(not(i))| -- MAY BE INTERESTING TO REMOVE ABS VALUE HERE
        metrics['fp_gap_' + str(i)] = false_positive_gap(metrics, i, k) # |fp_k - fp_not_k| -- MAY BE INTERESTING TO REMOVE ABS VALUE HERE
        metrics['fn_gap_' + str(i)] = false_negative_gap(metrics, i, k) # |fn_k - fn_not_k| -- MAY BE INTERESTING TO REMOVE ABS VALUE HERE
        metrics['calibration_pos_gap_' + str(i)] = calibration_gap(metrics, i, k, 1)
        metrics['calibration_neg_gap_' + str(i)] = calibration_gap(metrics, i, k, 0)
    return metrics


def predict(prob):   
    return prob >= 0.5


def get_accuracy(y_pred, y):
    return np.sum(1 for i in range(len(y)) if (y_pred[i] == y[i])) / len(y)


# 1 - METRICS FROM WADSWORTH PAPER
# protected variable z (array of 1s or 0s); calculate for demographic y_select (1 or 0)
def demographic_parity_gap(y_pred, z, k, y_select):
    prop_k = np.sum(y_pred[z == k] == y_select) / np.sum(z == k)
    prop_not = np.sum(y_pred[z != k] == y_select) / np.sum(z != k)
    return np.abs(prop_k - prop_not)


def false_positive_gap(metrics, k, num_k):
    fp_k = metrics['false_pos_' + str(k)] / (metrics['false_pos_' + str(k)] + metrics['true_neg_' + str(k)])

    fp_not_num = 0
    fp_not_denom = 0
    for i in range(num_k):
        if i != k:
            fp_not_num += metrics['count_' + str(i)]*metrics['false_pos_' + str(i)]
            fp_not_denom += metrics['count_' + str(i)]*(metrics['false_pos_' + str(i)] + metrics['true_neg_' + str(i)])

    fp_not = fp_not_num/fp_not_denom
    return np.abs(fp_k - fp_not)


def false_negative_gap(metrics, k, num_k):
    fn_k = metrics['false_neg_' + str(k)] / (metrics['false_neg_' + str(k)] + metrics['true_pos_' + str(k)])

    fn_not_num = 0
    fn_not_denom = 0
    for i in range(num_k):
        if i != k:
            fn_not_num += metrics['count_' + str(i)]*metrics['false_neg_' + str(i)]
            fn_not_denom += metrics['count_' + str(i)]*(metrics['false_neg_' + str(i)] + metrics['true_pos_' + str(i)])

    fn_not = fn_not_num/fn_not_denom
    return np.abs(fn_k - fn_not)

def calibration_gap(metrics, k, num_k, q):
    typ = 'pos'
    if (q == 0):
        typ = 'neg'
    cal_k = metrics['calibration_' + typ + '_' + str(k)]
    cal_not = np.mean([metrics['calibration_' + typ + '_' + str(i)] for i in range(num_k) if (i != k)])
    return np.abs(cal_k - cal_not)


# 2 - METRICS FROM ZHANG PAPER (somewhat redundant, but coded again for simplicity)
def confusion_matrix(y_pred, y):
    true_pos = np.sum((y_pred == 1) & (y == 1))
    false_pos = np.sum((y_pred == 1) & (y == 0))
    true_neg = np.sum((y_pred == 0) & (y == 0))
    false_neg = np.sum((y_pred == 0) & (y == 1))
    return true_neg, false_neg, false_pos, true_pos

