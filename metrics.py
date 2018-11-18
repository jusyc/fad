import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def predict(prob):
    return (prob >= 0.5)

def get_accuracy(y_pred, y):
    return sum(1 for i in range(len(y)) if (predict(y_pred[i]) == y[i]))*100.0/len(y)

# 1 - METRICS FROM WADSWORTH PAPER
# protected variable z (array of 1s or 0s); calculate for demographic y_select (1 or 0)
def demographic_parity_gap(y_pred, z, y_select):
    m = len(y_pred)
    prop_0 = sum(1 for i in range(m) if (z[i] == 0 and predict(y_pred[i]) == y_select))/(m-sum(z[i] for i in range(m)))
    prop_1 = sum(1 for i in range(m) if (z[i] == 1 and predict(y_pred[i]) == y_select))/sum(z[i] for i in range(m))
    return np.abs(prop_0[0] - prop_1[0])

def false_positive_gap(y_pred, y, z):
    m = len(y_pred)
    fp_0 = sum(1 for i in range(m) if (z[i] == 0 and predict(y_pred[i] > y[i])))/(m-sum(z[i] for i in range(m)))
    fp_1 = sum(1 for i in range(m) if (z[i] == 1 and predict(y_pred[i] > y[i])))/sum(z[i] for i in range(m))
    return np.abs(fp_0[0] - fp_1[0])

def false_negative_gap(y_pred, y, z):
    m = len(y_pred)
    fn_0 = sum(1 for i in range(m) if (z[i] == 0 and predict(y_pred[i] < y[i])))/(m-sum(z[i] for i in range(m)))
    fn_1 = sum(1 for i in range(m) if (z[i] == 1 and predict(y_pred[i] < y[i])))/sum(z[i] for i in range(m))
    return np.abs(fn_0[0] - fn_1[0])
    
# 2 - METRICS FROM ZHANG PAPER (somewhat redundant, but coded again for simplicity)
def confusion_matrix(y_pred, y, z, z_select):
    m = len(z)
    m_select = sum(1 for i in range(m) if (z[i] == z_select))
    true_pos = sum(1 for i in range(m) if (z[i] == z_select and y[i] == 1 and predict(y_pred[i]) == 1))
    true_neg = sum(1 for i in range(m) if (z[i] == z_select and y[i] == 0 and predict(y_pred[i]) == 0))
    false_pos = sum(1 for i in range(m) if (z[i] == z_select and predict(y_pred[i]) > y[i]))
    false_neg = sum(1 for i in range(m) if (z[i] == z_select and y[i] > predict(y_pred[i])))
    return [true_pos, true_neg, false_pos, false_neg]

def get_performance_metrics(y_pred, y):
    accuracy = get_accuracy(y_pred, y)
    roc_auc = roc_auc_score(y, y_pred)
    print('Accuracy: {:.4f}; roc_auc score: {:.4f}'.format(accuracy, roc_auc))
    return np.array([accuracy, roc_auc])

def get_fairness_metrics(y_pred, y, z, y_select = 0):
    cm_0 = confusion_matrix(y_pred, y, z, z_select=0)
    cm_1 = confusion_matrix(y_pred, y, z, z_select=0)
    dpg = demographic_parity_gap(y_pred, z, y_select)
    fpg = false_positive_gap(y_pred, y, z)
    fng = false_negative_gap(y_pred, y, z)

    print('Demographic parity gap: {:.4f}; False positive gap: {:.4f}; False negative gap: {:.4f}'.format(dpg, fpg, fng))

    return np.array([dpg, fpg, fng, cm_0[0], cm_0[1], cm_0[2], cm_0[3], cm_1[0], cm_1[1], cm_1[2], cm_1[3]])
