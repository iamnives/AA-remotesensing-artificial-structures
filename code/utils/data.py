"""
Created on Sun Mar  3 21:42:16 2019

@author: André Neves
"""

import os
import sys

import gdal
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold, NearMiss
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import RUSBoostClassifier
from scipy import stats
import scipy.signal
from collections import Counter

from tqdm import tqdm

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "bigsquare/"

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

def _class_map_ua(x):
    if x >= 1 and x <= 7: 
        #built up
        return 0
    return 1

def _class_split_map_ua(x):
    if x >= 1 and x <= 3: 
        #dense fabric
        return 1
    if x > 3 and x <= 6: 
        #less dense fabric
        return 2
    elif (x > 6 and x <= 14) or (x == 17):
        # roads and other structures 17 sport
        return 3
    elif x > 25 and x <= 27:
        # water bodies
        return 5
    return 4 # non built up

def _class_split_map_ua_binary(x):
    if x >= 1 and x <= 3: 
        #urban fabric
        return 1
    return 0 # non built up

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
    if (x >= 1 and x <= 3) or x == 5:
        return 1
    return 0

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

def load_prediction(datafiles=None, ratio=1, znorm=False, normalize=False, map_classes=False, urban_atlas=False, binary=False, osm_roads=False, army_gt=False, split_struct=False, gt_raster="cos_ground.tiff"):
    print("Prediction data: Loading...")

    if(datafiles is None):
        src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER) if (
            'ground' not in f) and ("xml" not in f) and ("_" in f) and ("decis" not in f)]
        ts_dss = [TS_FOLDER + f for f in os.listdir(TS_FOLDER) if (
            'ground' not in f) and ("xml" not in f) and ("_" in f)]
        ts1_dss = [TS1_FOLDER + f for f in os.listdir(TS1_FOLDER) if (
            'ground' not in f) and ("xml" not in f) and ("_" in f)]

        src_dss = src_dss + ts_dss + ts1_dss
    else:
        src_dss = datafiles
    src_dss.sort()
    print("SRC Images: ", src_dss)

    refDs = gdal.Open(gt_raster, gdal.GA_ReadOnly)
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
        
        print("Datasets: Loading...")
        for raster in tqdm(src_dss):
            # Open raster dataset
            raster_ds = gdal.Open(raster, gdal.GA_ReadOnly)
            n_bands = raster_ds.RasterCount
            # Extract band's data and transform into a numpy array
            for band in range(1, n_bands+1):
                test_ds = raster_ds.GetRasterBand(band).ReadAsArray()
                test_ds = test_ds[:shape[0], :shape[1]]
                X.append(test_ds.flatten())

        print("Transposing data...")
        # Transpose attributes matrix
        X = np.dstack(tuple(X))[0]
        X = X.astype(np.float32)

        X[~np.isfinite(X)] = -1

        print("Saving data to file cache...")
        np.save(CACHE_FOLDER + "pred_data.npy", X)

    if normalize:
        X = normalize.transform(X)
    elif znorm:
        print("Z-Normalization: Loading...")
        X = stats.zscore(X, axis=1)
        print("Done!")

    labelDS = gdal.Open(gt_raster, gdal.GA_ReadOnly)

    y = labelDS.GetRasterBand(1)
    y = y.ReadAsArray()[:shape[0], :shape[1]].flatten()

    maping_f = _class_map
    if binary:
        maping_f = _class_map_binary

    if osm_roads:
        labelDS = gdal.Open(
            DS_FOLDER + "roads_cos_50982.tiff", gdal.GA_ReadOnly)
        roads = labelDS.GetRasterBand(1).ReadAsArray()[
            :shape[0], :shape[1]].flatten()
        y[roads == 4] = roads[roads == 4]
        maping_f = _road_and_map

    if split_struct:
            maping_f = _class_split_map

    if army_gt:
        maping_f = _army_map

    if urban_atlas:
        maping_f = _class_split_map_ua
        print("Class Mapping: UA2018...")

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

def load(datafiles=None, normalize=False, znorm=False, map_classes=False, binary=False, test_size=0.2, osm_roads=False, army_gt=False, urban_atlas=False, split_struct=False, indexes=False, gt_raster="cos_ground.tif"):

    try:
        print("Trying to load cached data...")
        X_train = np.load(CACHE_FOLDER + "train_data.npy")
        y_train = np.load(CACHE_FOLDER + "train_labels.npy")

        X_test = np.load(CACHE_FOLDER + "test_data.npy")
        y_test = np.load(CACHE_FOLDER + "test_labels.npy")

        X_val = np.load(CACHE_FOLDER + "val_data.npy")
        y_val = np.load(CACHE_FOLDER + "val_labels.npy")
        print("Using cached data...", X_train.shape)
        normalizer = None
    except Exception:
        print("Failed to load cached data...")
        print("Reconstructing data...")
       
        if(datafiles is None):
            src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER) if (
                'ground' not in f) and ("xml" not in f) and ("_" in f) and ("decis" not in f)]
            ts_dss = [TS_FOLDER + f for f in os.listdir(TS_FOLDER) if (
                'ground' not in f) and ("xml" not in f) and ("_" in f)]
            ts1_dss = [TS1_FOLDER + f for f in os.listdir(TS1_FOLDER) if (
                'ground' not in f) and ("xml" not in f) and ("_" in f)]

            src_dss = src_dss + ts_dss + ts1_dss
        else:
            src_dss = datafiles
        src_dss.sort()
        print("SRC Images: ", src_dss)
        
        gt_ds = gdal.Open(
            DS_FOLDER + gt_raster, gdal.GA_ReadOnly)
        gt_bands = gt_ds.GetRasterBand(1)
        gt_bands = gt_bands.ReadAsArray()[:, :]

        ref_ds = gdal.Open(
            DS_FOLDER + gt_raster, gdal.GA_ReadOnly)
        
        ref_bands = ref_ds.GetRasterBand(1)
        ref_bands = ref_bands.ReadAsArray()[:, :]

        (unique, counts) = np.unique(gt_bands, return_counts=True)
        frequencies = np.asarray((unique, counts)).T

        print("Pre zero frequencies")
        print(frequencies)

        is_train = np.nonzero(ref_bands)
        print("Pixels: ", gt_bands.size)

        y = gt_bands[is_train].flatten()
        
        (unique, counts) = np.unique( y, return_counts=True)
        frequencies = np.asarray((unique, counts)).T

        print("Real frequencies")
        print(frequencies) 

        # Prepare training data (set of pixels used for training) and labels
        # Create empty HxW array/matrix
        # X = np.empty((len(src_dss), len(cos_bands[is_train])))
        X = []
        # Get list of raster bands info as array, already indexed by labels non zero
        print("Datasets: Loading...")
        for i, raster in enumerate(tqdm(src_dss)):
            # Open raster dataset
            raster_ds = gdal.Open(raster, gdal.GA_ReadOnly)
            n_bands = raster_ds.RasterCount
            # Extract band's data and transform into a numpy array
            for band in range(1, n_bands+1):
                test_ds = raster_ds.GetRasterBand(band).ReadAsArray()
                X.append(test_ds[is_train].flatten())

        # dont remove transpose after loading, time sucks if you do it at load 
        # more resource heavy but it takes way less time
        print("Transposing data...")
        # Transpose attributes matrix 
        # X = X.T
        X = np.dstack(tuple(X))[0]

        normalizer = None
        if normalize:
            print("Normalization: Loading...")
            normalizer = preprocessing.Normalizer().fit(X)
            X = normalizer.transform(X)
            print("Done!")
        elif znorm:
            print("Z-Normalization: Loading...")
            X = stats.zscore(X, axis=1)
            print("Done!")


        print("Datasets: Done!           ")
        print("Datasets: Features array shape, should be (n,k): " + str(X.shape))
        
        # resample dataset to wanted distributioms
        #us = EditedNearestNeighbours()
        #X, y = us.fit_resample(X, y)
        print("after processing: ", sorted(Counter(y).items()))

        maping_f = _class_map

        if urban_atlas:
            maping_f = _class_split_map_ua
            print("Class Mapping: UA2018...")

        if binary:
            maping_f = _class_map_binary
            print("Class Mapping: Binary...")

        if osm_roads:
            roads_ds = gdal.Open(
                DS_FOLDER + "roads_cos_50982.tiff", gdal.GA_ReadOnly)
            roads = roads_ds.GetRasterBand(1).ReadAsArray()    
            #roads = roads[is_train]
            y[roads == 4] = roads[roads == 4]
            maping_f = _road_and_map
            print("Class Mapping: Roads...")

        if split_struct:
            maping_f = _class_split_map
            print("Class Mapping: Split...")

        if army_gt:
            maping_f = _army_map
            print("Class Mapping: Army...")

        if map_classes:
            print("Class Mapping: Loading...")
            y = np.array([maping_f(yi) for yi in tqdm(y)])
            print("Class Mapping: Done!      ")

            (unique, counts) = np.unique(y, return_counts=True)
            frequencies = np.asarray((unique, counts)).T

            print(frequencies)

        if not indexes: # switch to if indexes exist use indexes.
            indices = np.arange(X.shape[0])
            print("Train validation split...")
            X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(X, y, indices, test_size=test_size, stratify=y, random_state=42)

            np.save(CACHE_FOLDER + "main_indexes_1.npy", idx1)
            np.save(CACHE_FOLDER + "main_indexes_2.npy", idx2)
            indices = np.arange(X_train.shape[0])
            print("Train test split...")
            X_train, X_val, y_train, y_val, idx1, idx2 = train_test_split(X_train, y_train, indices, stratify=y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2
            np.save(CACHE_FOLDER + "train_indexes_1.npy", idx1)
            np.save(CACHE_FOLDER + "train_indexes_2.npy", idx2)
        else:
            print("Trying to load cached indexes data...")
            idx1 = np.load(CACHE_FOLDER + "main_indexes_1.npy")
            idx2 = np.load(CACHE_FOLDER + "main_indexes_2.npy")
            X_train = X[idx1]
            y_train = y[idx1]
            X_test = X[idx2]
            y_test = y[idx2]

            idx1 = np.load(CACHE_FOLDER + "train_indexes_1.npy")
            idx2 = np.load(CACHE_FOLDER + "train_indexes_2.npy")
            X_val = X_train[idx2]
            y_val = y_train[idx2]
            X_train = X_train[idx1]
            y_train = y_train[idx1]
            

        print("Cleaning and typing data...")
        # Prevents overflow on algoritms computations
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        X_val = X_val.astype(np.float32)

        X_train[~np.isfinite(X_train)] = -1
        X_test[~np.isfinite(X_test)] = -1
        X_val[~np.isfinite(X_val)] = -1

        (unique, counts) = np.unique(y_train, return_counts=True)
        frequencies = np.asarray((unique, counts)).T

        print("Train frequencies")
        print(frequencies) 

        (unique, counts) = np.unique(y_test, return_counts=True)
        frequencies = np.asarray((unique, counts)).T

        print("Test frequencies")
        print(frequencies) 


        print("Saving data to file cache...")

        np.save(CACHE_FOLDER + "train_data.npy", X_train)
        np.save(CACHE_FOLDER + "train_labels.npy", y_train)

        np.save(CACHE_FOLDER + "test_data.npy", X_test)
        np.save(CACHE_FOLDER + "test_labels.npy", y_test)

        np.save(CACHE_FOLDER + "val_data.npy", X_val)
        np.save(CACHE_FOLDER + "val_labels.npy", y_val)

    return X_train, y_train, X_test, y_test, X_val, y_val, normalizer
