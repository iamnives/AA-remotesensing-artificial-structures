import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import gdal
from utils import visualization as viz
import time
from datetime import timedelta
import numpy as np
from utils import data
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/" + ROI + \
    "timeseries/rf/forest_20px_tsfull_group2_classification.tiff"

OUT_PROBA_RASTER = DATA_FOLDER + "results/" + ROI + \
    "timeseries/rf/forest_20px_tsfull_group2_classification_proba_"

REF_FILE = DATA_FOLDER + "clipped/" + ROI + \
    "ignored/static/clipped_sentinel2_B08.vrt"

OUT_FEATURES = DATA_FOLDER + "results/" + ROI + \
    "timeseries/rf/forest_20px_tsfull_group2_features.pdf"


def main(argv):
    real_start = time.time()
    train_size = int(19386625*0.2)

    split_struct=True
    osm_roads=False

    X_train, y_train, X_test, y_test,_,_,_ = data.load(
        train_size, normalize=False, osm_roads=osm_roads, split_struct=split_struct)

    start = time.time()
    # Build a forest and compute the feature importances
    forest = RandomForestClassifier(n_estimators=500,
                                    min_samples_leaf=4,
                                    min_samples_split=2,
                                    max_depth=130,
                                    class_weight='balanced',
                                    n_jobs=-1, verbose=1)
    print("Fitting data...")
    forest.fit(X_train, y_train)

    end = time.time()
    elapsed = end-start
    print("Training time: " + str(timedelta(seconds=elapsed)))

    yt_pred = forest.predict(X_train)
    kappa = cohen_kappa_score(y_train, yt_pred)
    print(f'Train Kappa: {kappa}')
    print(classification_report(y_train, yt_pred))

    y_pred = forest.predict(X_test)
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f'Validation Kappa: {kappa}')
    print(classification_report(y_test, y_pred))
    return 0

    dump(forest, '../sensing_data/models/forest_tsfull_group2.joblib')
    print("Saved model to disk")

    X, y, shape = data.load_prediction(ratio=1, normalize=None, osm_roads=osm_roads, split_struct=split_struct, army_gt=False)
    
    start_pred = time.time()
    y_pred_classes = forest.predict(X)

    # y_pred_proba = forest.predict_proba(X)
    # y_pred_classes = np.array(
    #     [np.argmax(yi, axis=-1) + 1 for yi in tqdm(y_pred_proba)])
    print("Predict time: " + str(timedelta(seconds=time.time()-start_pred)))

    kappa = cohen_kappa_score(y, y_pred_classes)
    print(f'Kappa: {kappa}')
    print(classification_report(y, y_pred_classes))

    yr = y_pred_classes.reshape(shape)

    viz.createGeotiff(OUT_RASTER, yr, REF_FILE, gdal.GDT_Byte)

    print("Creating uncertainty matrix...")
    start_matrix = time.time()

    # y_pred_proba_reshaped = y_pred_proba.reshape((shape[0], shape[1], 3))

    # viz.createGeotiff(OUT_PROBA_RASTER + "estrutura.tiff",
    #                   y_pred_proba_reshaped[:, :, 0], REF_FILE, gdal.GDT_Float32)
    # # viz.createGeotiff(OUT_PROBA_RASTER + "estrada.tiff",
    # #                   y_pred_proba_reshaped[:, :, 1], REF_FILE, gdal.GDT_Float32)
    # viz.createGeotiff(OUT_PROBA_RASTER + "restante.tiff",
    #                   y_pred_proba_reshaped[:, :, 1], REF_FILE, gdal.GDT_Float32)
    # viz.createGeotiff(OUT_PROBA_RASTER + "agua.tiff",
    #                   y_pred_proba_reshaped[:, :, 2], REF_FILE, gdal.GDT_Float32)

    end = time.time()
    elapsed = end-start_matrix
    print("Matrix creation time: " + str(timedelta(seconds=elapsed)))

    end = time.time()
    elapsed = end-real_start
    print("Total run time: " + str(timedelta(seconds=elapsed)))


if __name__ == "__main__":
    main(sys.argv)
