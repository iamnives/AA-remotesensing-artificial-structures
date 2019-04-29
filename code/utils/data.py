import os
import sys

import gdal
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#inicialize data location
DATA_FOLDER = "../sensing_data/"
DS_FOLDER = DATA_FOLDER + "clipped/"
LB_FOLDER = DATA_FOLDER + "labels/"
OUT_RASTER = DATA_FOLDER + "results/classification.tiff"

def _class_map(x):
    if x == 4: return 2
    if x >= 1 and x <= 13:
        return 1
    elif x > 13 and x <= 42:
        return 3
    elif x > 42 and x <= 48:
            return 4
    return -1

def load(train_size):
    X = []

    src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER)]

    labelDS = gdal.Open(DS_FOLDER + "clipped_cos_50982.tif", gdal.GA_ReadOnly)

    # Extract band's data and transform into a numpy array
    labelBands = labelDS.GetRasterBand(1).ReadAsArray()
    # Prepare training data (set of pixels used for training) and labels
    isTrain = np.nonzero(labelBands)
    y = labelBands[isTrain]

    print("Labels array shape, should be (n,): " + str(y.shape))

    # Get list of raster bands info as array, already indexed by labels non zero
    test_ds = None
    for _, raster in enumerate(src_dss):
        if("cos_50982.tif" not in raster):
            # Open raster dataset
            print("Opening raster: " + raster)
            rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)
            # Extract band's data and transform into a numpy array
            test_ds = rasterDS.GetRasterBand(1).ReadAsArray()
            X.append(test_ds[isTrain])
        
    print("Done!") 

    X = np.dstack(tuple(X))[0]

    n_samples = X.shape[0]

    n_samples_per = train_size/n_samples

    # Split the dataset in two equal parts
    _, X_train, _, y_train = train_test_split(
        X, y, test_size=n_samples_per)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=n_samples_per/2)

    del X
    del y

    # Shuffle the data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    X_train = X_train[indices]
    y_train = y_train[indices]

    indices = np.arange(X_test.shape[0])
    np.random.shuffle(indices)

    X_test = X_test[indices]
    y_test = y_test[indices]


    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    X_test = preprocessing.normalize(X_test)
    X_train = preprocessing.normalize(X_train)

    return X_train, np.array([_class_map(y) for y in y_train]) , X_test ,  np.array([_class_map(y) for y in y_test])