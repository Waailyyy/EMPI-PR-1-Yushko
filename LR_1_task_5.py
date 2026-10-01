# ПР-1 Юшко ТЦР-43 — Завдання 2.5. Метрики якості класифікації

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

df = pd.read_csv('data_metrics.csv')
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')
print(df.head())

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

def yushko_find_TP(y_true, y_pred):
    # true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))

def yushko_find_FN(y_true, y_pred):
    # false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))

def yushko_find_FP(y_true, y_pred):
    # false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))

def yushko_find_TN(y_true, y_pred):
    # true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))

def yushko_find_conf_matrix_values(y_true, y_pred):
    TP = yushko_find_TP(y_true, y_pred)
    FN = yushko_find_FN(y_true, y_pred)
    FP = yushko_find_FP(y_true, y_pred)
    TN = yushko_find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def yushko_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = yushko_find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

y_true = df.actual_label.values
print('TP:', yushko_find_TP(y_true, df.predicted_RF.values))
print('FN:', yushko_find_FN(y_true, df.predicted_RF.values))
print('FP:', yushko_find_FP(y_true, df.predicted_RF.values))
print('TN:', yushko_find_TN(y_true, df.predicted_RF.values))
print("\nsklearn confusion_matrix RF:\n", confusion_matrix(y_true, df.predicted_RF.values))
print("yushko_confusion_matrix RF:\n", yushko_confusion_matrix(y_true, df.predicted_RF.values))
print("yushko_confusion_matrix LR:\n", yushko_confusion_matrix(y_true, df.predicted_LR.values))

assert np.array_equal(yushko_confusion_matrix(y_true, df.predicted_RF.values),
                      confusion_matrix(y_true, df.predicted_RF.values)), 'yushko_confusion_matrix() is not correct for RF'
assert np.array_equal(yushko_confusion_matrix(y_true, df.predicted_LR.values),
                      confusion_matrix(y_true, df.predicted_LR.values)), 'yushko_confusion_matrix() is not correct for LR'
print("\nConfusion matrix check: OK")

def yushko_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = yushko_find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + FN + FP + TN)

def yushko_recall_score(y_true, y_pred):
    TP, FN, FP, TN = yushko_find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

def yushko_precision_score(y_true, y_pred):
    TP, FN, FP, TN = yushko_find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

def yushko_f1_score(y_true, y_pred):
    recall = yushko_recall_score(y_true, y_pred)
    precision = yushko_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

checks = [(yushko_accuracy_score, accuracy_score, 'accuracy'),
          (yushko_recall_score, recall_score, 'recall'),
          (yushko_precision_score, precision_score, 'precision'),
          (yushko_f1_score, f1_score, 'f1')]
for mine, ref, name in checks:
    for col in ['predicted_RF', 'predicted_LR']:
        assert np.isclose(mine(y_true, df[col].values), ref(y_true, df[col].values)), f'yushko_{name}_score failed on {col}'
print("All metric checks: OK\n")

for name, f in [('Accuracy', yushko_accuracy_score), ('Recall', yushko_recall_score),
                ('Precision', yushko_precision_score), ('F1', yushko_f1_score)]:
    print('%s RF: %.3f' % (name, f(y_true, df.predicted_RF.values)))
    print('%s LR: %.3f' % (name, f(y_true, df.predicted_LR.values)))

for t in [0.5, 0.25]:
    pred = (df.model_RF >= t).astype('int').values
    print('scores with threshold = %s' % t)
    print('Accuracy RF: %.3f' % yushko_accuracy_score(y_true, pred))
    print('Recall RF: %.3f' % yushko_recall_score(y_true, pred))
    print('Precision RF: %.3f' % yushko_precision_score(y_true, pred))
    print('F1 RF: %.3f' % yushko_f1_score(y_true, pred))
    print('')

from sklearn.metrics import roc_curve, roc_auc_score
fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_true, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_true, df.model_LR.values)

plt.figure()
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF')
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

auc_RF = roc_auc_score(y_true, df.model_RF.values)
auc_LR = roc_auc_score(y_true, df.model_LR.values)
print('AUC RF:%.3f' % auc_RF)
print('AUC LR:%.3f' % auc_LR)

plt.figure()
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
