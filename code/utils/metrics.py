import os
import sys

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score



def scores(y_test, y_pred, verbose=1):
    kappa = cohen_kappa_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    if verbose:
        print(f'Kappa: {kappa}')
        print(report)
    return kappa, matrix, report