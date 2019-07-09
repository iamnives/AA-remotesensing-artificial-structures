
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from feature_selection import fselector
import matplotlib.pyplot as plt
import xgboost as xgb
from joblib import dump, load
from xgboost import plot_importance
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import gdal
from utils import data
from utils import visualization as viz
import time
from datetime import timedelta
import numpy as np
from tqdm import tqdm
import argparse
import csv   
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing

import time

def get_Data():
    train_size = int(19386625*0.2)
    X, y, X_test, y_test = data.load(
        train_size, normalize=False, balance=False, osm_roads=False)

    return X, y, X_test, y_test


def models():
    boosted = xgb.XGBClassifier(colsample_bytree=0.7553707061597048,
                                gamma=5,
                                gpu_id=0,
                                learning_rate=0.2049732654267658,
                                max_depth=8,
                                min_child_weight=1,
                                max_delta_step=9.075685204314162,
                                n_estimators=1500,
                                n_jobs=4,
                                objective='multi:softmax',
                                predictor='cpu_predictor',
                                tree_method='gpu_hist')

    forest = RandomForestClassifier(n_estimators=500,
                                    min_samples_leaf=4,
                                    min_samples_split=2,
                                    max_depth=130,
                                    class_weight='balanced',
                                    n_jobs=-1, verbose=1)

    sv = svm.SVC(C=6.685338321430641, gamma=6.507029881541734)

    sgc = SGDClassifier(alpha=0.2828985957487874, class_weight='balanced', early_stopping=True,
                        l1_ratio=0.12293886358853467, loss='perceptron', max_iter=1000, penalty='l2', tol=0.001)


    return boosted, sv, sgc, forest
def get_metrics(y_pred, y_true):
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return kappa, report


def write_to_file(line):
    with open('./finalrun.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(line)

def main(argv):
    # DATASET codes: static 1, timeseries(s1s2) 2, timeseries dem 3
    # LABELS codes: estruturas 1, estradas 2, estrutura separada 3
    write_to_file(['MODEL', 'DATASET', 'SAMPLE', 'LABELS', 'ISROAD', 'CLASS', 'PRECISION', 'RECALL', 'F1SCORE', 'KAPPA', 'TRAINTIME', 'PREDICTTIME'])

    boosted, sv, sgc, forest = models()
    dataset = 3
    n_classes = 3
    X, y, X_test, y_test = get_Data()

    print("Fitting XGB...")
    start = time.time()
    boosted.fit(X,y)
    end = time.time()
    traintime = end-start

    start = time.time()
    pred = boosted.predict(X_test)
    end = time.time()
    predtime = end-start

    kappa, report = get_metrics(pred, y_test)
    for i in list(range(0, n_classes)):
        line = ['XGB', dataset, X.shape[0], 1, False, i, report[str(i)]['precision'], report[str(i)]['recall'], report[str(i)]['f1-score'], kappa, traintime, predtime]
        write_to_file(line)

    print("Fitting RF...")
    start = time.time()
    forest.fit(X,y)
    end = time.time()
    traintime = end-start

    start = time.time()
    pred = pred.predict(X_test)
    end = time.time()
    predtime = end-start

    kappa, report = get_metrics(pred, y_test)
    for i in list(range(0, n_classes)):
        line = ['RF', dataset, X.shape[0], 1, False, i, report[str(i)]['precision'], report[str(i)]['recall'], report[str(i)]['f1-score'], kappa, traintime, predtime]
        write_to_file(line)

    print("Normalization for SVMs: Loading...")
    normalizer = preprocessing.Normalizer().fit(X)
    X = normalizer.transform(X)
    X_test = normalizer.transform(X_test)
    print("Done!")

    print("Fitting SGD...")
    start = time.time()
    sgc.fit(X,y)
    end = time.time()
    traintime = end-start

    start = time.time()
    pred = sgc.predict(X_test)
    end = time.time()
    predtime = end-start

    kappa, report = get_metrics(pred, y_test)
    for i in list(range(0, n_classes)):
        line = ['SGD', dataset, X.shape[0], 1, False, i, report[str(i)]['precision'], report[str(i)]['recall'], report[str(i)]['f1-score'], kappa, traintime, predtime]
        write_to_file(line)

    print("Fitting svm...")
    # svm cant handle full training data
    X_train, X_test, y_train, y_test = train_test_split(
        X_test, y_test, test_size=20000, train_size=100_000)

    start = time.time()
    sv.fit(X_train,y_train)
    end = time.time()
    traintime = end-start

    start = time.time()
    pred = sv.predict(X_test)
    end = time.time()
    predtime = end-start

    kappa, report = get_metrics(pred, y_test)
    for i in list(range(0, n_classes)):
        line = ['SVM', dataset, X_train.shape[0], 1, False, i, report[str(i)]['precision'], report[str(i)]['recall'], report[str(i)]['f1-score'], kappa, traintime, predtime]
        write_to_file(line)

if __name__ == "__main__":
    main(sys.argv)
