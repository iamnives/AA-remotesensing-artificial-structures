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
OUT_RASTER = DATA_FOLDER + "results/test_roads.tiff"

def testi():
    start = time.time()
    X, y, pic_shape = data.load_prediction(DS_FOLDER, normalize=False)

   
    yr = y.reshape(pic_shape)
    viz.createGeotiff(OUT_RASTER, yr, DS_FOLDER + "clipped_sentinel2_B03.vrt")

    end=time.time()
    elapsed=end-start
    print("Run time: " + str(timedelta(seconds=elapsed)))
testi()
