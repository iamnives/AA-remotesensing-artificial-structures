import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scipy.stats import uniform
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
from utils import visualization as viz
from utils import data
from utils import metrics
from datetime import timedelta
import time
import numpy as np


def model(dfs):
    
    train_size = int(19386625*0.05)
    X_train, y_train, X_test, y_test = data.load(
        train_size, datafiles=dfs, normalize=True, balance=False, osm_roads=True)

    start = time.time()
    print(f'Tuning on {X_train.shape}')
    tuning_params = {
        'loss': ['hinge'],
        'penalty': ['elasticnet', 'l2', 'l1', 'none'],
        'alpha': 10.0**-np.arange(1,7),
        'l1_ratio': uniform(0, 1),
        'early_stopping': [True],
        'class_weight': ['balanced'],
        'tol': [1e-3],
        'max_iter': [1000, 500, 1500]
    }

    print(
        f'# Tuning hyper-parameters for Stochastic gradient descent (SVM) on { X_train.shape[0] } samples')
    print()
    kappa_scorer = make_scorer(cohen_kappa_score)
    gs = RandomizedSearchCV(SGDClassifier(), tuning_params, cv=3, scoring={
                            'kappa': kappa_scorer}, refit='kappa', return_train_score=False, n_iter=200, verbose=2, n_jobs=-1)
    gs.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(gs.best_params_)
    print()
    print(gs.best_score_)
    print()
    
    clf = gs.best_estimator_
    y_pred = clf.predict(X_test)

    kappa = cohen_kappa_score(y_test, y_pred)

    matrix = confusion_matrix(y_test, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))
    print(matrix)

    end = time.time()
    elapsed = end-start
    print("Run time: " + str(timedelta(seconds=elapsed)))

def main(argv):
    model(None)


if __name__ == "__main__":
    main(sys.argv)
