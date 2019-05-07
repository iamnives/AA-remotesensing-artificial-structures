import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
 
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from utils import data
import matplotlib.pyplot as plt  
import numpy as np
from datetime import timedelta
import time
from utils import visualization as viz
from utils import data

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load

def train():
    start = time.time()

    train_size = 200_000
    X, y, X_test , y_test  = data.load(train_size, normalize=True, balance=False) 

    # Build a forest and compute the feature importances
    forest = xgb.XGBClassifier(colsample_bytree=0.5483193137202504, 
                            gamma=0.1, 
                            gpu_id=0, 
                            learning_rate=0.6783980222181293,
                            max_depth=6,
                            min_child_weight=1,
                            n_estimators=1500,
                            nthread=4,
                            objective='multi:softmax', 
                            predictor='gpu_predictor', 
                            tree_method='gpu_hist', 
                            verbose=2)

    forest.fit(X, y)
    y_pred = forest.predict(X_test)

    kappa = cohen_kappa_score(y_test, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))

    end=time.time()
    elapsed=end-start
    print("Run time: " + str(timedelta(seconds=elapsed)))

    dump(forest, 'boosted.model')

train()