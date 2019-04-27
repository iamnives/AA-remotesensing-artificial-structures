import os
import sys

import gdal

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

from utils import visualization as viz
from utils import data


def main(argv):
    X, y = data.load()

    N_COMPUTING = 100000
    n_samples = X.shape[0]

    n_samples_per = N_COMPUTING/n_samples

    # Split the dataset in two equal parts
    _, X_train, _, y_train = train_test_split(
        X, y, test_size=n_samples_per)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=n_samples_per/2)

    del X
    del y

    # Shuffle the data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    X_train = X_train[indices]
    y_train = y_train[indices]

    indices = np.arange(X_test.shape[0])
    np.random.shuffle(indices)

    X_test = X_test[indices]
    y_test = y_test[indices]


    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    X_test = preprocessing.normalize(X_test)
    X_train = preprocessing.normalize(X_train)
    # Set the parameters by cross-validation
    tuning_params = [ {'n_estimators': [1,5,10,20,100, 150, 200, 500], "n_jobs": [-1]} ]

    scores = ['f1_weighted', 'accuracy', 'precision_weighted', 'recall_weighted']

    print("# Tuning hyper-parameters for %s" % scores)
    print()

    gs = GridSearchCV(RandomForestClassifier(), tuning_params, cv=5,
                        scoring=scores, refit='precision_weighted', return_train_score=True)
    gs.fit(X_train, y_train)

    print("Best parameters set found on development set: precision_weighted")
    print()
    print(gs.best_params_)
    print()

    clf = gs.best_estimator_
    y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 =  f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Accuracy: {accuracy}')
    print(f'F1: {f1}')
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))

    viz.plot_confusionmx(matrix)
    viz.plot_gridcv(gs.cv_results_, scores, "n_estimators", 500)

if __name__== "__main__":
  main(sys.argv)