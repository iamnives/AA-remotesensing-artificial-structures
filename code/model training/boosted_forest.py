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
from feature_selection import fselector
from sklearn.model_selection import train_test_split

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/" + ROI + \
    "timeseries/xgb/boosted_20px_tstest_group1_classification.tiff"
OUT_PROBA_RASTER = DATA_FOLDER + "results/" + ROI + \
    "timeseries/xgb/boosted_20px_tstest_group1_classification"

REF_FILE = DATA_FOLDER + "clipped/" + ROI + \
    "ignored/static/clipped_sentinel2_B08.vrt"

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
    parser.add_argument("--fselect", type=str_2_bool, nargs='?',
                    const=True, default=False,
                    help="Activate feature selection")

    args = parser.parse_args()

    road_flag = args.roads
    selector_flag = args.fselect

    if road_flag:
        print("Using roads...")

    if selector_flag:
        print("Using feature selection...")

    obj = 'binary:hinge'

    real_start = time.time()
    
    split_struct=False
    osm_roads=False

    train_size = int(19386625*0.2)
    # train_size = int(1607*1015*0.2)
    
    X_train, y_train, X_test, y_test, _, _, _ = data.load(train_size, map_classes=False, normalize=False, osm_roads=osm_roads, split_struct=split_struct, gt_raster='cos_new_gt_2015t.tif')

    start = time.time()

    forest = xgb.XGBClassifier(colsample_bytree=0.7553707061597048,
                            gamma=5,
                            gpu_id=0,
                            learning_rate=0.2049732654267658,
                            max_depth=8,
                            min_child_weight=1,
                            max_delta_step=9.075685204314162,
                            n_estimators=1500,
                            n_jobs=-1,
                            #objective=obj,  # binary:hinge if binary classification
                            predictor='cpu_predictor',
                            tree_method='gpu_hist')


    if selector_flag:
        print("Feature importances running...")
        # svm cant handle full training data
        x_train_feature, _, y_train_feature, _ = train_test_split(
            X_test, y_test, test_size=0, train_size=100_000)

        selector = fselector.Fselector(forest, mode="importances", thold=0.80)
        transformer = selector.select(x_train_feature, y_train_feature)

        features = transformer.get_support()
        # feature_names = data.get_features()
        # feature_names = feature_names[features]
        print(features)
        print("Transforming data...")
        print("Before: ", X_train.shape)
        X = transformer.transform(X)
        X_test = transformer.transform(X_test)
        print("After: ", X_train.shape)

    print("Fitting data...")
    forest.fit(X_train, y_train)

    end = time.time()
    elapsed = end-start
    print("Training time: " + str(timedelta(seconds=elapsed)))


    yt_pred = forest.predict(X_train)

    yt_pred[yt_pred > 0.5] = 1
    yt_pred[yt_pred <= 0.5] = 0  

    kappa = cohen_kappa_score(y_train, yt_pred)
    print(f'Train Kappa: {kappa}')
    print(classification_report(y_train, yt_pred))

    y_pred = forest.predict(X_test)

    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0 

    kappa = cohen_kappa_score(y_test, y_pred)
    print(f'Validation Kappa: {kappa}')
    print(classification_report(y_test, y_pred))

    dump(forest, '../sensing_data/models/boosted_test_group1.joblib')
    print("Saved model to disk")

    # Testing trash
    X, y, shape = data.load_prediction(
        ratio=1, map_classes=False, normalize=False, osm_roads=osm_roads, split_struct=split_struct, gt_raster='cos_new_gt_2015t.tif')

    start_pred = time.time()
    # batch test
    X_h = X[:len(X)//2]
    X_h1 = X[len(X)//2:]

    forest.get_booster().set_param('predictor', 'cpu_predictor')

    print("Predict 0%...")
    y_pred = forest.predict(X_h)
    print("Predict 50%...")
    y_pred2 = forest.predict(X_h1)
    print("Predict 100%...")

    y_pred_classes = np.concatenate((y_pred, y_pred2))
    y_pred_classes[y_pred_classes > 0.5] = 1
    y_pred_classes[y_pred_classes <= 0.5] = 0 

    print("Predict time: " + str(timedelta(seconds=time.time()-start_pred)))

    kappa = cohen_kappa_score(y, y_pred_classes)
    print(f'Kappa: {kappa}')
    print(classification_report(y, y_pred_classes))

    y_pred_classes_reshaped = y_pred_classes.reshape(shape)

    viz.createGeotiff(OUT_RASTER, y_pred_classes_reshaped,
                      REF_FILE, gdal.GDT_Byte)

    print("Creating uncertainty matrix...")
    start_matrix = time.time()

    return 

    y_pred_proba_reshaped = y_pred_proba.reshape((shape[0], shape[1], 4))

    viz.createGeotiff(OUT_PROBA_RASTER + "estrutura_urbana.tiff",
                      y_pred_proba_reshaped[:, :, 0], REF_FILE, gdal.GDT_Float32)
    viz.createGeotiff(OUT_PROBA_RASTER + "estrada.tiff",
                    y_pred_proba_reshaped[:, :, 1], REF_FILE, gdal.GDT_Float32)
    # viz.createGeotiff(OUT_PROBA_RASTER + "outras.tiff",
    #                     y_pred_proba_reshaped[:, :, 2], REF_FILE, gdal.GDT_Float32)
    viz.createGeotiff(OUT_PROBA_RASTER + "natural.tiff",
                      y_pred_proba_reshaped[:, :, 3], REF_FILE, gdal.GDT_Float32)
    viz.createGeotiff(OUT_PROBA_RASTER + "agua.tiff",
                      y_pred_proba_reshaped[:, :, 4], REF_FILE, gdal.GDT_Float32)

    end = time.time()
    elapsed = end-start_matrix
    print("Matrix creation time: " + str(timedelta(seconds=elapsed)))

    end = time.time()
    elapsed = end-real_start
    print("Total run time: " + str(timedelta(seconds=elapsed)))


if __name__ == "__main__":
    main(sys.argv)

