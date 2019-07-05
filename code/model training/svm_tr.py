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
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn import svm

from joblib import dump, load

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/" + ROI + \
    "/timeseries/svm_100k_ts_s2_idx_roadstrack_align_classification.tiff"

OUT_PROBA_RASTER = DATA_FOLDER + "results/" + ROI + \
    "/timeseries/svm_100k_ts_s2_idx_roadstrack_align_classification_proba_"

REF_FILE = DATA_FOLDER + "clipped/" + ROI + \
    "/ignored/static/clipped_sentinel2_B08.vrt"


def main(argv):
    real_start = time.time()

    train_size = 100_000
    X, y, X_test , y_test  = data.load(train_size, normalize=True, balance=False) 

    start = time.time()
    # Build a sv and compute the feature importances
    sv = svm.SVC(C=6.685338321430641, gamma=6.507029881541734)

    print("Fitting data...")
    sv.fit(X, y)

    end = time.time()
    elapsed = end-start
    print("Training time: " + str(timedelta(seconds=elapsed)))
    
    y_pred = sv.predict(X_test)

    kappa = cohen_kappa_score(y_test, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))

    dump(sv, '../sensing_data/models/svm.joblib')
    print("Saved model to disk")
    # Testing trash
    X, y, shape = data.load_prediction(DS_FOLDER, normalize=True)
    
    start_pred = time.time()
    y_pred = sv.predict(X)
    print("Predict time: " + str(timedelta(seconds=time.time()-start_pred)))

    kappa = cohen_kappa_score(y, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y, y_pred))

    yr = y_pred.reshape(shape)

    viz.createGeotiff(OUT_RASTER, yr, DS_FOLDER + "clipped_sentinel2_B03.vrt", gdal.GDT_Byte)

    end=time.time()
    elapsed=end-real_start
    print("Total run time: " + str(timedelta(seconds=elapsed)))

if __name__ == "__main__":
    main(sys.argv)
