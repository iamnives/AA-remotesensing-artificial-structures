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
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
TS_FOLDER = DS_FOLDER + "tstats/"
TS1_FOLDER = DS_FOLDER + "t1stats/"

OUT_RASTER = DS_FOLDER + "results/roads_test.tif"
def test_pred():
    start = time.time()
    X, y, pic_shape = data.load_prediction(ratio=1, normalize=False)

    viz.createGeotiff(OUT_RASTER, y, DS_FOLDER + "ref.tif", gdal.GDT_UInt16)

    end=time.time()
    elapsed=end-start
    print("Run time: " + str(timedelta(seconds=elapsed)))

if __name__ == '__main__':   
    test_pred()
