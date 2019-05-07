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

import gdal
#inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "lisboa-setubal/"
DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/classification.tiff"

def predict():
    start = time.time()
    X, y = data.load_prediction(DS_FOLDER)
    print(X.shape)
    #load saved model
    forest = load('boosted.model')

    y_pred = forest.predict(X)

    kappa = cohen_kappa_score(y, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y, y_pred))

    end=time.time()
    elapsed=end-start
    print("Run time: " + str(timedelta(seconds=elapsed)))
predict()
