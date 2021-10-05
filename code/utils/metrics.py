import os
import sys

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import jaccard_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

def scores(y_test, y_pred, verbose=1):
    kappa = cohen_kappa_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    mcc = jaccard_score(y_test, y_pred)
    jaccard = matthews_corrcoef(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if verbose:
        print(f'|----------------------------------------------|')
        print(f'Mathew\'s correlation: {mcc}')
        print(f'Balanced Accuracy: {balanced_acc}')
        print(f'Jaccard Score: {jaccard}')
        print(f'Kappa: {kappa}')
        print(f'R2: {r2}')
        print(f'ROC AUC: {roc_auc}')
        print(report)
        print(matrix)
        print(f'|----------------------------------------------|')
    return kappa, matrix, report, mcc, jaccard, balanced_acc, r2, roc_auc