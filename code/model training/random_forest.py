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
from sklearn.ensemble import RandomForestClassifier

from joblib import dump, load

def train():
    start = time.time()

    train_size = 200_000
    X, y, X_test , y_test  = data.load(train_size, normalize=True, balance=False) 

    # Build a forest and compute the feature importances
    forest = RandomForestClassifier(n_estimators=500,
                                min_samples_leaf=4, 
                                min_samples_split=2, 
                                max_depth=130,
                                class_weight='balanced',
                                n_jobs=-1, verbose=1)

    forest.fit(X, y)
    y_pred = forest.predict(X_test)

    kappa = cohen_kappa_score(y_test, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))

    end=time.time()
    elapsed=end-start
    print("Run time: " + str(timedelta(seconds=elapsed)))

    dump(forest, 'forest.model')

train()