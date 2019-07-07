"""
Created on Sun Mar  3 21:42:16 2019

@author: André Neves
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
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
from xgboost import plot_importance
from joblib import dump, load
import xgboost as xgb
import matplotlib.pyplot as plt


# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/" + ROI + \
    "/timeseries/boosted_20px_ts_s1_s2_dem_idx_classification.tiff"
OUT_PROBA_RASTER = DATA_FOLDER + "results/" + ROI + \
    "/timeseries/boosted_20px_ts_s1_s2_dem_idx_classification"

REF_FILE = DATA_FOLDER + "clipped/" + ROI + \
    "/ignored/static/clipped_sentinel2_B08.vrt"

OUT_FEATURES = DATA_FOLDER + "results/" + ROI + \
    "/timeseries/boosted_20px_ts_s1_s2_idx_features.pdf"

def str_2_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(argv):
    parser = argparse.ArgumentParser(description='Trains a xgboost model.')
    parser.add_argument("--roads", type=str_2_bool, nargs='?',
                    const=True, default=False,
                    help="Activate OSM roads")

    args = parser.parse_args()

    road_flag = args.roads

    obj = 'binary:hinge' if args.roads else 'multi:softmax'

    real_start = time.time()
    train_size = int(19386625*0.2)
    X, y, X_test, y_test = data.load(
        train_size, normalize=False, balance=False, osm_roads=road_flag)

    start = time.time()
    # Build a forest and compute the feature importances
    forest = xgb.XGBClassifier(colsample_bytree=0.5483193137202504,
                               gamma=0.1,
                               gpu_id=0,
                               learning_rate=0.6783980222181293,
                               max_depth=6,
                               min_child_weight=1,
                               n_estimators=1500,
                               n_jobs=4,
                               objective=obj,  # binary:hinge if binary classification
                               predictor='gpu_predictor',
                               tree_method='gpu_hist')

    print("Fitting data...")
    forest.fit(X, y)

    end = time.time()
    elapsed = end-start
    print("Training time: " + str(timedelta(seconds=elapsed)))

    y_pred = forest.predict(X_test)

    kappa = cohen_kappa_score(y_test, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="c", align="center")
    plt.xticks(range(X.shape[1]), data.feature_map(
        indices), rotation='90', horizontalalignment="right")
    plt.xlim([-1, X.shape[1]])

    plt.savefig(OUT_FEATURES, bbox_inches='tight')

    dump(forest, '../sensing_data/models/boosted.joblib')
    print("Saved model to disk")

    # Testing trash
    X, y, shape = data.load_prediction(
        ratio=1, normalize=False, osm_roads=road_flag)

    start_pred = time.time()
    # batch test
    X_h = X[:len(X)//2]
    X_h1 = X[len(X)//2:]

    forest.get_booster().set_param('predictor', 'cpu_predictor')

    print("Predict 0%...")
    y_pred = forest.predict_proba(X_h)
    print("Predict 50%...")
    y_pred2 = forest.predict_proba(X_h1)
    print("Predict 100%...")

    y_pred_proba = np.concatenate((y_pred, y_pred2))
    y_pred_classes = np.array(
        [np.argmax(yi, axis=-1) + 1 for yi in tqdm(y_pred_proba)])
    print("Predict time: " + str(timedelta(seconds=time.time()-start_pred)))

    kappa = cohen_kappa_score(y, y_pred_classes)
    print(f'Kappa: {kappa}')
    print(classification_report(y, y_pred_classes))

    y_pred_classes_reshaped = y_pred_classes.reshape(shape)

    viz.createGeotiff(OUT_RASTER, y_pred_classes_reshaped,
                      REF_FILE, gdal.GDT_Byte)

    print("Creating uncertainty matrix...")
    start_matrix = time.time()

    y_pred_proba_reshaped = y_pred_proba.reshape((shape[0], shape[1], 3))

    viz.createGeotiff(OUT_PROBA_RASTER + "estrutura.tiff",
                      y_pred_proba_reshaped[:, :, 0], REF_FILE, gdal.GDT_Float32)
    # viz.createGeotiff(OUT_PROBA_RASTER + "estrada.tiff",
    #                   y_pred_proba_reshaped[:, :, 1], REF_FILE, gdal.GDT_Float32)
    viz.createGeotiff(OUT_PROBA_RASTER + "restante.tiff",
                      y_pred_proba_reshaped[:, :, 1], REF_FILE, gdal.GDT_Float32)
    viz.createGeotiff(OUT_PROBA_RASTER + "agua.tiff",
                      y_pred_proba_reshaped[:, :, 2], REF_FILE, gdal.GDT_Float32)

    end = time.time()
    elapsed = end-start_matrix
    print("Matrix creation time: " + str(timedelta(seconds=elapsed)))

    end = time.time()
    elapsed = end-real_start
    print("Total run time: " + str(timedelta(seconds=elapsed)))


if __name__ == "__main__":
    main(sys.argv)
