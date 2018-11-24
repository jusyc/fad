import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# static constants
HYPERPARAMS = ['learning_rate', 'total_num_iters', 'n_h', 'n_h_adv', 'dropout_rate', 'alpha']

def get_metrics(y_pred, y, z, hyperparams, y_select = 0):
    metrics = dict()

    # add hyperparameters for experiment
    for i in range(len(hyperparams)):
        metrics[HYPERPARAMS[i]] = hyperparams[i]

    metrics['accuracy'] = get_accuracy(y_pred, y)
    metrics['roc_auc'] = roc_auc_score(y, y_pred)
    metrics['count_0'] = np.sum(z == 0)
    metrics['count_1'] = np.sum(z == 1)
    metrics['accuracy_0'] = get_accuracy(y_pred[z == 0], y[z == 0])
    metrics['accuracy_1'] = get_accuracy(y_pred[z == 1], y[z == 1])
    metrics['true_neg_0'], metrics['false_neg_0'], metrics['false_pos_0'], metrics['true_pos_0']  = \
        confusion_matrix(y_pred[z == 0], y[z == 0])
    metrics['true_neg_1'], metrics['false_neg_1'], metrics['false_pos_1'], metrics['true_pos_1']  = \
        confusion_matrix(y_pred[z == 1], y[z == 1])
    metrics['parity_gap'] = demographic_parity_gap(y_pred, z, y_select)
    metrics['fp_gap'] = false_positive_gap(metrics)
    metrics['fn_gap'] = false_negative_gap(metrics)

    return metrics


def predict(prob):
    return prob >= 0.5


def get_accuracy(y_pred, y):
    return np.sum(1 for i in range(len(y)) if (predict(y_pred[i]) == y[i]))*100.0/len(y)


# 1 - METRICS FROM WADSWORTH PAPER
# protected variable z (array of 1s or 0s); calculate for demographic y_select (1 or 0)
def demographic_parity_gap(y_pred, z, y_select):
    prop_0 = np.sum(predict(y_pred[z == 0]) == y_select) / np.sum(z == 0)
    prop_1 = np.sum(predict(y_pred[z == 1]) == y_select) / np.sum(z == 1)
    return np.abs(prop_0 - prop_1)


def false_positive_gap(metrics):
    fp_0 = metrics['false_pos_0'] / (metrics['false_pos_0'] + metrics['true_neg_0'])
    fp_1 = metrics['false_pos_1'] / (metrics['false_pos_1'] + metrics['true_neg_1'])
    return np.abs(fp_0 - fp_1)


def false_negative_gap(metrics):
    fp_0 = metrics['false_neg_0'] / (metrics['false_neg_0'] + metrics['true_pos_0'])
    fp_1 = metrics['false_neg_1'] / (metrics['false_neg_1'] + metrics['true_pos_1'])
    return np.abs(fp_0 - fp_1)


# 2 - METRICS FROM ZHANG PAPER (somewhat redundant, but coded again for simplicity)
def confusion_matrix(y_pred, y):
    true_pos = np.sum((predict(y_pred) == 1) & (y == 1))
    false_pos = np.sum((predict(y_pred) == 1) & (y == 0))
    true_neg = np.sum((predict(y_pred) == 0) & (y == 0))
    false_neg = np.sum((predict(y_pred) == 0) & (y == 1))
    return true_neg, false_neg, false_pos, true_pos

