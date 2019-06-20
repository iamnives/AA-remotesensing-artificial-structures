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

import gdal
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier

from joblib import dump, load

#inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/" + ROI + "/timeseries/test.tiff"
REF_FILE = DATA_FOLDER + "clipped/" + ROI  + "/ignored/static/clipped_sentinel2_B03.vrt"


train_size = int(19386625*0.2)
X, y, X_test , y_test  = data.load(train_size, normalize=False, balance=False) 

start = time.time()
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
print(confusion_matrix(y_test, y_pred))

end=time.time()
elapsed=end-start
print("Training time: " + str(timedelta(seconds=elapsed)))

dump(forest, '../sensing_data/models/forest.joblib')
print("Saved model to disk")

# Testing trash
X, y, shape = data.load_prediction(ratio=0.5, normalize=False)
print(X.shape, y.shape)

y_pred = forest.predict(X)

print(X.shape, y.shape)

kappa = cohen_kappa_score(y, y_pred)
print(f'Kappa: {kappa}')
print(classification_report(y, y_pred))

yr = y_pred.reshape(shape)

viz.createGeotiff(OUT_RASTER, yr, REF_FILE + "clipped_sentinel2_B03.vrt", gdal.GDT_Byte)

end=time.time()
elapsed=end-start
print("Total run time: " + str(timedelta(seconds=elapsed)))
