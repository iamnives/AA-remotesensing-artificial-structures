import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import csv
import xgboost as xgb
from utils import visualization as viz
from utils import data
from boruta import BorutaPy
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, ElasticNet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# l1 = lasso, l2 = ridge


class Fselector:
    def __init__(self, clf, mode="elastic", thold=0.25):
        self.thold = thold
        self.mode = "importances"
        self.clf = clf

        if mode == "lasso":
            self.selector = ElasticNet(alpha=6.7783432762922965e-06, l1_ratio=1, copy_X=True)
        elif mode == "boruta":
            self.selector = BorutaPy(clf, n_estimators='auto', verbose=1, perc=95)
        elif mode == "elastic":
            self.selector = ElasticNetCV(l1_ratio=np.geomspace(
                0.1, 1), normalize=True, cv=3, copy_X=True, n_jobs=-1)
        elif mode == "importances":
            if thold < 0.80:
                print("Warning: thold for feature importances is inverted from the other models, are you sure you want a thold lower than 0.8?")
            self.selector = None

    def select(self, X, y):
        if self.mode == "boruta":
            sfm = self.selector.fit(X, y)
            return sfm

        if self.mode == "importances":
            self.clf.fit(X, y)
            importances = self.clf.feature_importances_
            indices = np.argsort(importances)[::-1]

            total_importance = 0

            imp = self.thold
            for imp in importances[indices]:
                total_importance += imp
                if total_importance >= self.thold:
                    break

            sfm = SelectFromModel(self.clf, threshold=imp)
            sfm.fit(X, y)
            return sfm

        if self.mode != "importances":
            self.selector.fit(X, y)
            sfm = SelectFromModel(self.selector, threshold=0.25)
            sfm.fit(X, y)

            return sfm


def importances_test(argv):
    imps = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    n_classes = 3

    train_size = int(19386625*0.2)
    X, y, X_test, y_test = data.load(
        train_size, normalize=False, balance=False, osm_roads=False, split_struct=False)

    f_names = data.get_features()

    forest_transformed = xgb.XGBClassifier(colsample_bytree=0.7553707061597048,
                                           gamma=5,
                                           gpu_id=0,
                                           learning_rate=0.2049732654267658,
                                           max_depth=8,
                                           min_child_weight=1,
                                           max_delta_step=9.075685204314162,
                                           n_estimators=1500,
                                           n_jobs=4,
                                           objective='multi:softmax',  # binary:hinge if binary classification
                                           predictor='gpu_predictor',
                                           tree_method='gpu_hist')

    importances = np.array([0.01089848, 0.0023375, 0.00423722, 0.01034001, 0.00214008, 0.01552957,
                   0.01527289, 0.00191506, 0.00163826, 0.0015426, 0.00158133, 0.00134435,
                   0.0014973, 0.00147868, 0.00145128, 0.00136524, 0.00138426, 0.00526585,
                   0.00177788, 0.00559436, 0.00318752, 0.00206053, 0.00171499, 0.00161261,
                   0.00714606, 0.10342276, 0.05041501, 0.01287329, 0.00535889, 0.00369522,
                   0.0040654, 0.00264086, 0.00253616, 0.00196853, 0.00178495, 0.00256285,
                   0.00229026, 0.00456279, 0.00224456, 0.00206856, 0.00181751, 0.00210288,
                   0.00395745, 0.00181428, 0.0025117, 0.00174094, 0.00207488, 0.00208558,
                   0.00160785, 0.00184317, 0.00182251, 0.00255113, 0.00224134, 0.00282709,
                   0.00265786, 0.00289279, 0.00301179, 0.0023039, 0.0103916, 0.00195739,
                   0.00239043, 0.00205036, 0.00208437, 0.00235835, 0.00346185, 0.00229981,
                   0.00220893, 0.00224484, 0.00195264, 0.00204797, 0.0022098, 0.00312089,
                   0.00360755, 0.00209681, 0.00285087, 0.12099829, 0.00234435, 0.00183119,
                   0.00220757, 0.00600978, 0.00722492, 0.00335292, 0.00388622, 0.00773073,
                   0.00830841, 0.00403531, 0.00658356, 0.00516254, 0.00223089, 0.00336122,
                   0.00420028, 0.00409018, 0.00400201, 0.00464223, 0.00638007, 0.00420069,
                   0.00615229, 0.00357109, 0.00384229, 0.00584501, 0.00533407, 0.00403268,
                   0.01020813, 0.00859572, 0.00696929, 0.00293149, 0.00215568, 0.01377079,
                   0.0021877, 0.00223974, 0.00282237, 0.00232623, 0.00223629, 0.00216405,
                   0.00336877, 0.00224604, 0.0023854, 0.00237986, 0.00246003, 0.00225075,
                   0.00248474, 0.00518136, 0.00220346, 0.0019646, 0.0030761, 0.00242861,
                   0.00224683, 0.00309225, 0.00198086, 0.00254123, 0.00229161, 0.0024428,
                   0.00244258, 0.00196201, 0.07294901, 0.00255527, 0.12117738, 0.00211922,
                   0.00603639, 0.00931404, 0.00601439, 0.02677349, 0.0034595])

    indices = np.argsort(importances)[::-1]
    f_names_sorted = np.array(f_names)[indices]

    print(importances[indices])
    print(f_names_sorted)

    for thold in imps:
        print(f'Testing thold {thold}')
        total_importance = 0
        for idx, imp in enumerate(importances[indices]):
            total_importance += imp
            if total_importance >= thold:
                print("Transforming data...")
                transf_x = np.array([ xv[importances>=imp] for xv in tqdm(X) ])
                trans_x_test = np.array([ xv[importances>=imp] for xv in tqdm(X_test) ])

                print("Fitting transformed...")
                start = time.time()
                forest_transformed.fit(transf_x, y)
                traintime = time.time()-start

                print("Predicting transformed...")
                start = time.time()
                y_pred = forest_transformed.predict(trans_x_test)
                predtime = time.time()-start

                kappa = cohen_kappa_score(y_test, y_pred)
                report = classification_report(
                    y_test, y_pred, output_dict=True)

                for i in list(range(1, n_classes+1)):
                    with open('./importances_score.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['IMPORTANCES', thold, X.shape[1], transf_x.shape[1], kappa, i, report[str(
                            i)]['precision'], report[str(i)]['recall'], report[str(i)]['f1-score'], traintime, predtime])
                break

def main(argv):

    enet = ElasticNetCV(l1_ratio=np.geomspace(0.1,1), normalize=False, cv=5, copy_X=True, n_jobs=-1)
    n_classes = 3

    train_size = int(19386625*0.05)
    X, y, X_test, y_test = data.load(
        train_size, normalize=True, balance=False, osm_roads=False, split_struct=False)

    enet.fit(X, y)
    print(f"Alpha: {enet.alpha_}")
    print(f"Intercept: {enet.intercept_}")
    print(f"L1 ratio: {enet.l1_ratio_}")
    print(f"Iterations: {enet.n_iter_}")


def elastic_net_test(argv):
    train_size = int(19386625*0.05)
    X_sfm, y_sfm, _, _ = data.load(
        train_size, normalize=True, balance=False, osm_roads=False, split_struct=False)

    X, y, X_test, y_test = data.load(
        train_size, normalize=False, balance=False, osm_roads=False, split_struct=False)
    
    f_names = data.get_features()

    forest_transformed = xgb.XGBClassifier(colsample_bytree=0.7553707061597048,
                                            gamma=5,
                                            gpu_id=0,
                                            learning_rate=0.2049732654267658,
                                            max_depth=8,
                                            min_child_weight=1,
                                            max_delta_step=9.075685204314162,
                                            n_estimators=1500,
                                            n_jobs=-1,
                                            objective='multi:softmax',
                                            predictor='gpu_predictor',
                                            tree_method='gpu_hist')

    regr = ElasticNet(alpha=6.7783432762922965e-06, l1_ratio=1.0, max_iter=2000)
    regr.fit(X_sfm, y_sfm)
    n_classes = 3
    
    thold = 0.00001

    sfm = SelectFromModel(regr, threshold=thold)
    print(f"Features selected uesing SelectFromModel with threshold {sfm.threshold}." )
    sfm.fit(X_sfm, y_sfm)

    print(regr.coef_)

    print(X.shape)
    print("Transforming data...")
    X_transf = sfm.transform(X)
    X_test_transf = sfm.transform(X_test)

    print("Fitting transformed data...")
    print(X_transf.shape)

    start = time.time()
    forest_transformed.fit(X_transf, y)
    traintime = time.time()-start

    print("Predicting transformed...")
    start = time.time()
    y_pred = forest_transformed.predict(X_test_transf)
    predtime = time.time()-start

    kappa = cohen_kappa_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    for i in list(range(1, n_classes+1)):
        with open('./importances_score.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ELASTIC', thold, X.shape[1], X_transf.shape[1], kappa, i, report[str(
                i)]['precision'], report[str(i)]['recall'], report[str(i)]['f1-score'], traintime, predtime])

    plt.plot(regr.coef_, linestyle='none', marker='h', markersize=5, color='blue', label=r'Ridge; $\alpha = 6.78e-6$')
    plt.xlabel('Índice do coeficiente',fontsize=16)
    plt.ylabel('Magnitude dos coeficientes', fontsize=16)
    plt.legend(fontsize=13, loc=4)
    plt.show()
                    
def boruta_test(argv):
    train_size = int(19386625*0.05)

    X, y, X_test, y_test = data.load(
        train_size, normalize=False, balance=False, osm_roads=False, split_struct=False)
    
    f_names = data.get_features()
    print(X.shape)

    forest_transformed = xgb.XGBClassifier(colsample_bytree=0.7553707061597048,
                                            gamma=5,
                                            gpu_id=0,
                                            learning_rate=0.2049732654267658,
                                            max_depth=8,
                                            min_child_weight=1,
                                            max_delta_step=9.075685204314162,
                                            n_estimators=1500,
                                            n_jobs=-1,
                                            objective='multi:softmax',
                                            predictor='gpu_predictor',
                                            tree_method='gpu_hist',
                                            seed=47)

    n_classes = 3
    
    imps = [100, 95, 90, 80, 70, 50]

    for thold in imps:
        print(f"Features selected using Boruta with percentile {thold}." )

        sfm = BorutaPy(forest_transformed, n_estimators='auto', verbose=1, perc=thold, random_state=47)

        sfm.fit(X, y)

        print("Transforming data...")
        X_transf = sfm.transform(X)
        X_test_transf = sfm.transform(X_test)
        print(X_transf.shape)

        print("Fitting transformed data...")
    
        start = time.time()
        forest_transformed.fit(X_transf, y)
        traintime = time.time()-start

        print("Predicting transformed...")
        start = time.time()
        y_pred = forest_transformed.predict(X_test_transf)
        predtime = time.time()-start

        kappa = cohen_kappa_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        for i in list(range(1, n_classes+1)):
            with open('./importances_score.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['BORUTA', thold, X.shape[1], X_transf.shape[1], kappa, i, report[str(
                    i)]['precision'], report[str(i)]['recall'], report[str(i)]['f1-score'], traintime, predtime])

if __name__ == "__main__":
    boruta_test(sys.argv)
