import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, ElasticNet
from sklearn.feature_selection import SelectFromModel
from boruta import BorutaPy

# l1 = lasso, l2 = ridge
class Fselector:
    def __init__(self, clf, mode="elastic", thold=0.25):
        self.thold = thold
        self.mode = "importances"
        self.clf = clf

        if mode == "lasso":
            self.selector = ElasticNet(alpha=0.5, l1_ratio=1, copy_X=True)
        elif mode == "boruta":
            self.selector = BorutaPy(clf, n_estimators=1500, verbose=1)
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
