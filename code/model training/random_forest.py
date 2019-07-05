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


# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/" + ROI + \
    "/timeseries/forest_20px_ts_s2_idx_roadstrack_align_classification.tiff"

OUT_PROBA_RASTER = DATA_FOLDER + "results/" + ROI + \
    "/timeseries/forest_20px_ts_s2_idx_roadstrack_align_classification_proba_"

REF_FILE = DATA_FOLDER + "clipped/" + ROI + \
    "/ignored/static/clipped_sentinel2_B08.vrt"


def main(argv):
    real_start = time.time()
    train_size = int(19386625*0.2)
    X, y, X_test, y_test = data.load(
        train_size, normalize=False, balance=False)

    start = time.time()
    # Build a forest and compute the feature importances
    forest = RandomForestClassifier(n_estimators=500,
                                    min_samples_leaf=4,
                                    min_samples_split=2,
                                    max_depth=130,
                                    class_weight='balanced',
                                    n_jobs=-1, verbose=1)
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
    plt.show()

    dump(forest, '../sensing_data/models/forest.joblib')
    print("Saved model to disk")

    # Testing trash
    X, y, shape = data.load_prediction(ratio=1, normalize=False)
    
    start_pred = time.time()
    y_pred = forest.predict(X)
    print("Predict time: " + str(timedelta(seconds=time.time()-start_pred)))

    kappa = cohen_kappa_score(y, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y, y_pred))

    yr = y_pred.reshape(shape)

    viz.createGeotiff(OUT_RASTER, yr, REF_FILE +
                      "clipped_sentinel2_B03.vrt", gdal.GDT_Byte)

    end = time.time()
    elapsed = end-real_start
    print("Total run time: " + str(timedelta(seconds=elapsed)))


if __name__ == "__main__":
    main(sys.argv)
