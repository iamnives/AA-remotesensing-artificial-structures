"""
Created on Sun Mar  3 21:42:16 2019

@author: André Neves
"""

import os
import sys

from scipy.stats.stats import obrientransform
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
from utils import metrics
import gdal
from joblib import dump, load
import xgboost as xgb
import matplotlib.pyplot as plt
from feature_selection import fselector
from sklearn.model_selection import train_test_split
from utils.classaccuracymetrics import cls_quantity_accuracy, calc_class_accuracy_metrics
import pprint as pp

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "bigsquare/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/" + ROI + \
    "new_test.tiff"
OUT_PROBA_RASTER = DATA_FOLDER + "results/" + ROI + \
    "boosted_newfmz_test_classification"

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

    gt_raster = "cos_indi_ground_binary_gt.tiff"
    out_ref_raster = DS_FOLDER + "cos_indi_ground_binary_big.tif"
    X_train, y_train, X_test, y_test, _, _, norm = data.load(normalize=False, map_classes=False, binary=False, test_size=0.2, osm_roads=False, army_gt=False, urban_atlas=False, split_struct=False, indexes=False, gt_raster=gt_raster) 

    start = time.time()

    #XGB_binary_building = {'colsample_bytree': 0.7343021353976351, 'gamma': 0, 'learning_rate': 0.16313076998849083, 'max_delta_step': 8.62355770678575, 'max_depth': 8, 'min_child_weight': 3, 'n_estimators': 1500, 'predictor': 'cpu_predictor', 'tree_method': 'hist'}


    forest = xgb.XGBClassifier(colsample_bytree=0.7806582714598144,
                            gamma=0,
                            gpu_id=0,
                            learning_rate=0.1365824474680162,
                            max_depth=6,
                            min_child_weight=5,
                            max_delta_step=4.613093902915149,
                            n_estimators=1500,
                            n_jobs=-1,
                            objective=obj,  # binary:hinge if binary classification
                            predictor='cpu_predictor',
                            tree_method='gpu_hist'
                            )


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
    
    metrics.scores(y_train, yt_pred)

    y_pred = forest.predict(X_test)

    y_pred = y_pred - 1
    y_test = y_test - 1

    non_zero = np.count_nonzero(y_pred)

    #binary only
    acc_hist = [len(y_pred) - non_zero, non_zero]
    cls_area = np.array(acc_hist)
    print(cls_area)

    acc_metrics = calc_class_accuracy_metrics(y_test, y_pred, cls_area=cls_area, cls_names=['(0) non built-up', '(1) built-up'])

    metrics.scores(y_test, y_pred)
    pp.pprint(acc_metrics)

    if False:
        print("Class Mapping: Loading...")
        y_pred_binary = np.array([data._class_split_map_ua_binary(yi) for yi in tqdm(y_pred)])
        y_true_binary = np.array([data._class_split_map_ua_binary(yi) for yi in tqdm(y_test)])
        print("Class Mapping: Done!      ")
        print("Binary metrics.......................")
        metrics.scores(y_true_binary, y_pred_binary)

    dump(forest, '../sensing_data/models/boosted_test_ua_cluster.joblib')
    print("Saved model to disk")
    return 
    # Testing trash, useless gt:raster for no gt
    predict_folder = "D:/AA-remotesensing-artificial-structures/sensing_data/clipped/fullsqure/clipped/"
    predict_dfs = [predict_folder+'clipped_sentinel1_asc.tif',predict_folder+'clipped_sentinel1_desc.tif',predict_folder+'clipped_sentinel2_complete.tif']
    #out_ref_raster = predict_folder+'clipped_sentinel1_asc.tif'
    X, y, shape = data.load_prediction(datafiles=None, ratio=1, map_classes=False, binary=False, urban_atlas=False, znorm=False, normalize=False, osm_roads=False, split_struct=False, gt_raster=predict_folder+'cos_indi_ground_binary_big.tif')

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

    print("Predict time: " + str(timedelta(seconds=time.time()-start_pred)))

    y_pred_classes_reshaped = y_pred_classes.reshape(shape)

    viz.createGeotiff(OUT_RASTER, y_pred_classes_reshaped,out_ref_raster, gdal.GDT_Byte)

    non_zero_idx = np.nonzero(y)
    y_pred_classes_clean = y_pred_classes[non_zero_idx]
    y_clean = y[non_zero_idx]

    metrics.scores(y_clean, y_pred_classes_clean)

    print("Creating uncertainty matrix...")
    start_matrix = time.time()
    create_proba = False
    if create_proba:
        print("Predict 0%...")
        y_pred_proba = forest.predict_proba(X_h)
        print("Predict 50%...")
        y_pred2_proba = forest.predict_proba(X_h1)
        print("Predict 100%...")
        y_pred_proba = np.concatenate((y_pred_proba, y_pred2_proba))

        y_pred_proba_reshaped = y_pred_proba.reshape((shape[0], shape[1], 5))
        
        viz.createGeotiff(OUT_PROBA_RASTER + "estrutura.tiff",
                        y_pred_proba_reshaped[:, :, 0], out_ref_raster, gdal.GDT_Float32)
        viz.createGeotiff(OUT_PROBA_RASTER + "rural.tiff",
                        y_pred_proba_reshaped[:, :, 1], out_ref_raster, gdal.GDT_Float32)
        viz.createGeotiff(OUT_PROBA_RASTER + "outras.tiff",
                            y_pred_proba_reshaped[:, :, 2], out_ref_raster, gdal.GDT_Float32)
        viz.createGeotiff(OUT_PROBA_RASTER + "natural.tiff",
                        y_pred_proba_reshaped[:, :, 3], out_ref_raster, gdal.GDT_Float32)
        viz.createGeotiff(OUT_PROBA_RASTER + "indi.tiff",
                        y_pred_proba_reshaped[:, :, 4], out_ref_raster, gdal.GDT_Float32)


    end = time.time()
    elapsed = end-start_matrix
    print("Matrix creation time: " + str(timedelta(seconds=elapsed)))

    end = time.time()
    elapsed = end-real_start
    print("Total run time: " + str(timedelta(seconds=elapsed)))


if __name__ == "__main__":
    main(sys.argv)

