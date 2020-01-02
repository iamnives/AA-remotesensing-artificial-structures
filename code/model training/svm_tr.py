import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import gdal
from utils import visualization as viz
import time
from datetime import timedelta
import numpy as np
from utils import data
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/" + ROI + \
    "/static/svm/svm_100k_static_group3_classification.tiff"

OUT_PROBA_RASTER = DATA_FOLDER + "results/" + ROI + \
    "/static/svm/svm_100k_static_group3_classification_proba_"

REF_FILE = DATA_FOLDER + "clipped/" + ROI + \
    "/ignored/static/clipped_sentinel2_B08.vrt"


def main(argv):
    real_start = time.time()

    split_struct=False
    osm_roads=False

    # train_size = int(100_000)
    train_size = int(19_386_625*0.2)
    X_train, y_train, X_test, y_test,_,_,_ = data.load(train_size, normalize=True, osm_roads=osm_roads, split_struct=split_struct)

    start = time.time()
    # Build a sv and compute the feature importances
    sv = svm.SVC(C=6.685338321430641, gamma=6.507029881541734)

    print("Fitting data...")
    sv.fit(X_train, y_train)

    end = time.time()
    elapsed = end-start
    print("Training time: " + str(timedelta(seconds=elapsed)))

    yt_pred = sv.predict(X_train)
    kappa = cohen_kappa_score(y_train, yt_pred)
    print(f'Train Kappa: {kappa}')
    print(classification_report(y_train, yt_pred))

    y_pred = sv.predict(X_test)
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))
    return 0

    dump(sv, '../sensing_data/models/svm_static_group3.joblib')
    print("Saved model to disk")
    # Testing trash
    X, y, shape = data.load_prediction(
        ratio=1, normalize=True, osm_roads=osm_roads, split_struct=split_struct)

    start_pred = time.time()
    y_pred = sv.predict(X)
    print("Predict time: " + str(timedelta(seconds=time.time()-start_pred)))

    kappa = cohen_kappa_score(y, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y, y_pred))

    yr = y_pred.reshape(shape)

    viz.createGeotiff(OUT_RASTER, yr, DS_FOLDER +
                      "clipped_sentinel2_B08.vrt", gdal.GDT_Byte)

    end = time.time()
    elapsed = end-real_start
    print("Total run time: " + str(timedelta(seconds=elapsed)))


if __name__ == "__main__":
    main(sys.argv)
