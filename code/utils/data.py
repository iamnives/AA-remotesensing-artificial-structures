"""
Created on Sun Mar  3 21:42:16 2019

@author: André Neves
"""

import os
import sys

import gdal
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import RUSBoostClassifier

import scipy.signal

from tqdm import tqdm

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
TS_FOLDER = DS_FOLDER + "tstats/"
TS1_FOLDER = DS_FOLDER + "t1stats/"
STATIC_FOLDER = DS_FOLDER + "static/"

CACHE_FOLDER = DS_FOLDER + "cache/"

# Class to text for plotting features


def feature_map(u):
    src_dss = [f for f in os.listdir(DS_FOLDER) if (
        "cos" not in f) and ("xml" not in f) and ("_" in f)]
    ts_dss = [f for f in os.listdir(TS_FOLDER) if (
        "cos" not in f) and ("xml" not in f) and ("_" in f)]
    ts1_dss = [f for f in os.listdir(TS1_FOLDER) if (
        "cos" not in f) and ("xml" not in f) and ("_" in f)]

    src_dss = src_dss + ts_dss + ts1_dss
    src_dss.sort()
    text_classes = dict(zip(range(len(src_dss)), src_dss))
    return np.array([text_classes[x] for x in u])

# Class to text for plotting and analysis, only works if map_classes = True

def reverse_road_class_map(u):
    text_classes = {
        1: "Edificação artificial permanente",
        2: "Estradas",
        3: "Natural",
        4: "Água",
    }
    return np.array([text_classes[x] for x in u])

# Class to text for plotting and analysis, only works if map_classes = True

def reverse_class_map(u):
    text_classes = {
        1: "Edificação artificial permanente",
        2: "Natural",
        3: "Água",
    }
    return np.array([text_classes[x] for x in u])

def _army_map(X):
    return X

def _class_map(x):  
    if x >= 1 and x <= 13:
        return 1
    elif x > 13 and x <= 42:
        return 2
    elif x > 42 and x <= 48:
        return 3
    return 2

def _class_split_map(x):
    if x == 1: 
        return 1
    if x == 2: 
        return 2
    if x >= 3 and x <= 13:
        return 3
    elif x > 13 and x <= 42:
        return 4
    elif x > 42 and x <= 48:
        return 5
    return 4

def _road_and_map(x): 
    if x == 4:
        return 2
    if x >= 1 and x <= 13:
        return 1
    elif x > 13 and x <= 42:
        return 3
    elif x > 42 and x <= 48:
        return 4
    return 3

def _road_map(x):  # roads vs all
    if x == 4:
        return 1
    elif x > 42 and x <= 48:
        return 3
    return 2

def _class_map_binary(x):
    if x >= 1 and x <= 13:
        return 1
    else:
        return 2
    return 2

def get_features():
    src_dss = [f for f in os.listdir(DS_FOLDER) if (
        "cos" not in f) and ("xml" not in f) and ("_" in f)]
    ts_dss = [f for f in os.listdir(TS_FOLDER) if (
        "cos" not in f) and ("xml" not in f) and ("_" in f)]
    ts1_dss = [f for f in os.listdir(TS1_FOLDER) if (
        "cos" not in f) and ("xml" not in f) and ("_" in f)]

    src_dss = src_dss + ts_dss + ts1_dss
    src_dss.sort()
    return np.array(src_dss)

def load_prediction(ratio=1, normalize=None, map_classes=True, binary=False, osm_roads=False, convolve=False, army_gt=False, split_struct=False):
    print("Prediction data: Loading...")
    src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER) if (
        "cos" not in f) and ("xml" not in f) and ("_" in f)]
    ts_dss = [TS_FOLDER + f for f in os.listdir(TS_FOLDER) if (
        "cos" not in f) and ("xml" not in f) and ("_" in f)]
    ts1_dss = [TS1_FOLDER + f for f in os.listdir(TS1_FOLDER) if (
        "cos" not in f) and ("xml" not in f) and ("_" in f)]

    src_dss = src_dss + ts_dss + ts1_dss
    src_dss.sort()

    refDs = gdal.Open(
        DS_FOLDER + "/ignored/static/clipped_sentinel2_B08.vrt", gdal.GA_ReadOnly)
    band = refDs.GetRasterBand(1).ReadAsArray()
    shape = tuple([int(ratio*i) for i in band.shape])

    try:
        print("Trying to load cached data...")
        X = np.load(CACHE_FOLDER + "pred_data.npy")
        print("Using cached data...")
    except Exception:
        print("Failed to load cached data...")
        print("Reconstructing data...")
        X = []

        for raster in tqdm(src_dss):
            # Open raster dataset
            rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)
            # Extract band's data and transform into a numpy array
            test_ds = rasterDS.GetRasterBand(1).ReadAsArray()
            test_ds = test_ds[:shape[0], :shape[1]]

            if convolve:
                # Blur kernel
                filter_kernel = [[1, 1, 1],
                                 [1, -8, 1],
                                 [1, 1, 1]]
                test_ds = scipy.signal.convolve2d(
                    test_ds, filter_kernel, mode='same', boundary='fill', fillvalue=0)

            X.append(test_ds.flatten())

        print("Transposing data...")
        # Transpose attributes matrix
        X = np.dstack(tuple(X))[0]
        X = X.astype(np.float32)

        X[~np.isfinite(X)] = -1

        print("Saving data to file cache...")
        np.save(CACHE_FOLDER + "pred_data.npy", X)

    if normalize is not None:
        X = normalize.transform(X)

    labelDS = gdal.Open(
        DS_FOLDER + "clipped_cos_50982.tif", gdal.GA_ReadOnly)
    y = labelDS.GetRasterBand(1).ReadAsArray()[
        :shape[0], :shape[1]].flatten()

    maping_f = _class_map
    if binary:
        maping_f = _class_map_binary

    if osm_roads:
        labelDS = gdal.Open(
            DS_FOLDER + "roads_cos_50982.tif", gdal.GA_ReadOnly)
        roads = labelDS.GetRasterBand(1).ReadAsArray()[
            :shape[0], :shape[1]].flatten()
        y[roads == 4] = roads[roads == 4]
        maping_f = _road_and_map

    if split_struct:
            maping_f = _class_split_map

    if army_gt:
        maping_f = _army_map
        
    if map_classes:
        y = np.array([maping_f(yi) for yi in tqdm(y)])

    print("Prediction data: Done!")
    return X, y, shape

def load_timeseries(img_size):
    ts_dss = [TS_FOLDER + f for f in os.listdir(TS_FOLDER) if (
        "cos" not in f) and ("xml" not in f) and ("_" in f)]
    ts1_dss = [TS1_FOLDER + f for f in os.listdir(TS1_FOLDER) if (
        "cos" not in f) and ("xml" not in f) and ("_" in f)]
    image_files = ts_dss + ts1_dss  # Your list of files

    image_height = img_size[0]
    image_width = img_size[1]

    # Create empty HxWxN array/matrix
    image_stack = np.empty((image_height, image_width, len(image_files)))

    for i, fname in enumerate(image_files):
            # Extract band's data and transform into a numpy array
        label_ds = gdal.Open(fname, gdal.GA_ReadOnly)
        label_bands = label_ds.GetRasterBand(1).ReadAsArray()
        image_stack[:, :, i] = label_bands  # Set the i:th slice to this image
    return image_files

def load(train_size, datafiles=None, normalize=True, map_classes=True, binary=False, balance=False, test_size=0.2, osm_roads=False, convolve=False, army_gt=False, split_struct=False):

    try:
        print("Trying to load cached data...")
        X = np.load(CACHE_FOLDER + "train_data.npy")
        print("Using cached data...")
    except Exception:
        print("Failed to load cached data...")
        print("Reconstructing data...")
        X = []

        if(datafiles is None):
            src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER) if (
                "cos" not in f) and ("xml" not in f) and ("_" in f)]
            ts_dss = [TS_FOLDER + f for f in os.listdir(TS_FOLDER) if (
                "cos" not in f) and ("xml" not in f) and ("_" in f)]
            ts1_dss = [TS1_FOLDER + f for f in os.listdir(TS1_FOLDER) if (
                "cos" not in f) and ("xml" not in f) and ("_" in f)]

            src_dss = src_dss + ts_dss + ts1_dss
        else:
            src_dss = datafiles
        src_dss.sort()

        # Extract band's data and transform into a numpy array
        cos_ds = gdal.Open(
            DS_FOLDER + "clipped_cos_50982.tif", gdal.GA_ReadOnly)
        cos_bands = cos_ds.GetRasterBand(1).ReadAsArray()[:, :]

        if osm_roads:
            roads_ds = gdal.Open(
                DS_FOLDER + "roads_cos_50982.tif", gdal.GA_ReadOnly)
            roads = roads_ds.GetRasterBand(1).ReadAsArray()

        # Prepare training data (set of pixels used for training) and labels
        isTrain = np.nonzero(cos_bands)

        # Get list of raster bands info as array, already indexed by labels non zero
        print("Datasets: Loading...")
        for raster in tqdm(src_dss):
            if(("cos_50982.tif" not in raster) and ("xml" not in raster)):
                # Open raster dataset
                rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)
                # Extract band's data and transform into a numpy array
                test_ds = rasterDS.GetRasterBand(1).ReadAsArray()

                if convolve:
                    # This should do moving average try 5x5
                    filter_kernel = [[1, 1, 1],
                                     [1, 0, 1],
                                     [1, 1, 1]]

                    test_ds = scipy.signal.convolve2d(
                        test_ds, filter_kernel, mode='same', boundary='fill', fillvalue=0)

                X.append(test_ds[isTrain])
        print("Transposing data...")
        # Transpose attributes matrix
        X = np.dstack(tuple(X))[0]

        print("Datasets: Done!           ")
        print("Datasets: Features array shape, should be (n,k): " + str(X.shape))

        print("Saving data to file cache...")
        np.save(CACHE_FOLDER + "train_data.npy", X)

    cos_ds = gdal.Open(
        DS_FOLDER + "clipped_cos_50982.tif", gdal.GA_ReadOnly)
    cos_bands = cos_ds.GetRasterBand(1).ReadAsArray()[:, :]
    
    isTrain = np.nonzero(cos_bands)
    y = cos_bands[isTrain]

    maping_f = _class_map
    if binary:
        maping_f = _class_map_binary

    if osm_roads:
        roads_ds = gdal.Open(
            DS_FOLDER + "roads_cos_50982.tif", gdal.GA_ReadOnly)
        roads = roads_ds.GetRasterBand(1).ReadAsArray()    
        roads = roads[isTrain]
        y[roads == 4] = roads[roads == 4]
        maping_f = _road_and_map

    if split_struct:
        maping_f = _class_split_map

    if army_gt:
        maping_f = _army_map

    if map_classes:
        print("Class Mapping: Loading...")
        y = np.array([maping_f(yi) for yi in tqdm(y)])
        print("Class Mapping: Done!      ")

    print("Train test split...")
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(int(train_size*test_size)), train_size=min(X.shape[0], train_size), stratify=y, random_state=42)

    print("Cleaning and typing data...")
    # Prevents overflow on algoritms computations
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    X_train[~np.isfinite(X_train)] = -1
    X_test[~np.isfinite(X_test)] = -1

    normalizer = None
    if normalize:
        print("Normalization: Loading...")
        normalizer = preprocessing.Normalizer().fit(X_train)
        X_train = normalizer.transform(X_train)
        X_test = normalizer.transform(X_test)
        print("Done!")

    if balance:
        print("Data balance: Loading...")
        smt = TomekLinks(sampling_strategy='not minority',
                         n_jobs=4, random_state=42)
        X_train, y_train = smt.fit_sample(X_train, y_train)
        print("Features array shape after balance: " + str(X_train.shape))
    print("Dataset loaded!")

    return X_train, y_train, X_test, y_test, normalizer
