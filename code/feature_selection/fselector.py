import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, ElasticNet
from sklearn.feature_selection import SelectFromModel
from boruta import BorutaPy
from utils import data
from utils import visualization as viz
import xgboost as xgb
import csv
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import time

# l1 = lasso, l2 = ridge
class Fselector:
    def __init__(self, clf, mode="elastic", thold=0.25):
        self.thold = thold
        self.mode = "importances"
        self.clf = clf

        if mode == "lasso":
            self.selector = ElasticNet(alpha=1e-05, l1_ratio=1, copy_X=True)
        elif mode == "boruta":
            self.selector = BorutaPy(clf, n_estimators='auto', verbose=1)
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


def main(argv):
    imps = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    n_classes = 3

    train_size = int(19386625*0.2)
    X, y, X_test, y_test = data.load(
        train_size, normalize=False, balance=False, osm_roads=False, split_struct=False)

    f_names = data.get_features()

    forest = xgb.XGBClassifier(colsample_bytree=0.7553707061597048,
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

    forest.fit(X, y)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    total_importance = 0

    for thold in imps:
        for imp in importances[indices]:
            total_importance += imp
            if total_importance >= thold:

                sfm = SelectFromModel(forest, threshold=imp)
                sfm.fit(X, y)
                transf_x = sfm.transform(X)
                transf_y = sfm.transform(y)
                trans_x_test = sfm.transform(X_test)

                start = time.time()
                forest_transformed.fit(transf_x, transf_y)
                traintime = time.time()-start

                start = time.time()
                y_pred = forest_transformed.predict(trans_x_test)
                predtime = time.time()-start

                kappa = cohen_kappa_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                for i in list(range(1, n_classes+1)):
                    with open('./importances_score.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['IMPORTANCES', X.shape[0], transf_x.shape[0], kappa, traintime, predtime])
                
  
    
    

if __name__ == "__main__":
    main(sys.argv)
