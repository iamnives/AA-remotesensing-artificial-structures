"""
Created on Sun Mar  3 21:42:16 2019

@author: André Neves
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import numpy as np
from datetime import timedelta
import time
from utils import visualization as viz
from utils import data
import gdal
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.linear_model import SGDClassifier
from joblib import dump, load
import matplotlib.pyplot as plt


# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/" + ROI + \
    "/timeseries/SGD_20px_ts_fs1_s2_idx_roadstrack_align_classification.tiff"

REF_FILE = DATA_FOLDER + "clipped/" + ROI + \
    "/ignored/static/clipped_sentinel2_B08.vrt"


def main(argv):
    train_size = int(19386625*0.2)
    X, y, X_test, y_test = data.load(
        train_size, normalize=True, balance=False, osm_roads=True)

    start = time.time()

    sgc = SGDClassifier(alpha=0.2828985957487874, class_weight='balanced', early_stopping=True,
                        l1_ratio=0.12293886358853467, loss='perceptron', max_iter=1000, penalty='l2', tol=0.001)

    print("Fitting data...")
    sgc.fit(X, y)

    end = time.time()
    elapsed = end-start
    print("Training time: " + str(timedelta(seconds=elapsed)))

    y_pred = sgc.predict(X_test)

    kappa = cohen_kappa_score(y_test, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


    dump(sgc, '../sensing_data/models/sgc.joblib')
    print("Saved model to disk")
    # Testing trash
    X, y, shape = data.load_prediction(
        ratio=1, normalize=True, osm_roads=True)

    start_pred = time.time()

    y_pred = sgc.predict(X)

    kappa = cohen_kappa_score(y, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y, y_pred))
    print("Predict time: " + str(timedelta(seconds=time.time()-start_pred)))

    y_pred = y_pred.reshape(shape)

    viz.createGeotiff(OUT_RASTER, y_pred,
                      REF_FILE, gdal.GDT_Byte)

    end = time.time()
    elapsed = end-start
    print("Total run time: " + str(timedelta(seconds=elapsed)))


if __name__ == "__main__":
    main(sys.argv)
