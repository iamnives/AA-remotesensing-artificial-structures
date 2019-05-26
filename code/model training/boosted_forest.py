import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import xgboost as xgb 
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from utils import data
import matplotlib.pyplot as plt  
import numpy as np
from datetime import timedelta
import time
from utils import visualization as viz

import gdal
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

from joblib import dump, load

#inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/" + ROI + "/timeseries/boosted_20px_ts_s1_s2_idxfixed_roadstyped_align_classification.tiff"
REF_FILE = DATA_FOLDER + "clipped/" + ROI  + "/ignored/static/clipped_sentinel2_B03.vrt"
start = time.time() 

train_size = int(19386625*0.2)
X, y, X_test , y_test  = data.load(train_size, normalize=False, balance=False) 

# Build a forest and compute the feature importances
forest = xgb.XGBClassifier(colsample_bytree=0.5483193137202504, 
                        gamma=0.1, 
                        gpu_id=0, 
                        learning_rate=0.6783980222181293,
                        max_depth=6,
                        min_child_weight=1,
                        n_estimators=1500,
                        n_jobs=4,
                        objective='multi:softmax', # binary:hinge if binary classification
                        predictor='gpu_predictor', 
                        tree_method='gpu_hist')

forest.fit(X, y)
y_pred = forest.predict(X_test)

kappa = cohen_kappa_score(y_test, y_pred)
print(f'Kappa: {kappa}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

dump(forest, '../sensing_data/models/boosted.joblib')
print("Saved model to disk")
# Testing trash
X, y, shape = data.load_prediction(ratio=0.5, normalize=False)
 
# reduce to half size maybe just for the lol
print(X.shape, y.shape)

forest.get_booster().set_param('predictor', 'cpu_predictor')
y_pred = forest.predict(X)

kappa = cohen_kappa_score(y, y_pred)
print(f'Kappa: {kappa}')
print(classification_report(y, y_pred))

yr = y_pred.reshape(shape)

viz.createGeotiff(OUT_RASTER, yr, REF_FILE , gdal.GDT_Byte)

end=time.time()
elapsed=end-start
print("Run time: " + str(timedelta(seconds=elapsed)))