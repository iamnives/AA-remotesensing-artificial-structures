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

from tqdm import tqdm

#inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
TS_FOLDER = DS_FOLDER + "tstats/"
TS1_FOLDER = DS_FOLDER + "t1stats/"

# Class to text for plotting features
def feature_map(u):
    src_dss = [f.split("clipped")[1].split('.')[0][1:] for f in os.listdir(DS_FOLDER) if ("cos" not in f) and ("xml" not in f) ]
    src_dss.sort()
    text_classes = dict(zip(range(len(src_dss)), src_dss))
    return np.array([text_classes[x] for x in u])

# Class to text for plotting and analysis, only works if map_classes = True
def reverse_class_map(u):
    text_classes = {
        1: "Edificação artificial permanente",
        2: "Vias ferreas e estradas",
        3: "Vegetação",
        4: "Águas",
    }
    return np.array([text_classes[x] for x in u])

def _class_map(x):
    if x == 4: return 2
    if x >= 1 and x <= 13:
        return 1
    elif x > 13 and x <= 42: 
        return 3
    elif x > 42 and x <= 48:
            return 4
    return 0

def _class_map_binary(x):
    if x >= 1 and x <= 13:
        return 1
    else:
        return 2
    return 0

def get_features():
    src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER) if ("cos_50982.tif"  not in f) and ("xml" not in f) and ("_" in f) ]
    src_dss.sort()
    return np.array(src_dss)

def load_prediction(ratio=1, normalize=True, map_classes=True, binary=False, osm_roads=True):
    print("Prediction data: Loading...")
    src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER) if ("cos_50982.tif" not in f) and ("xml" not in f) and ("_" in f)]
    ts_dss = [TS_FOLDER + f for f in os.listdir(TS_FOLDER) if ("cos" not in f) and ("xml" not in f) and ("_" in f)]
    ts1_dss = [TS1_FOLDER + f for f in os.listdir(TS1_FOLDER) if ("cos" not in f) and ("xml" not in f) and ("_" in f)]

    src_dss = src_dss + ts_dss + ts1_dss
    src_dss.sort()
    X = []
    
    refDs = gdal.Open(DS_FOLDER + "/ignored/static/clipped_sentinel2_B03.vrt", gdal.GA_ReadOnly)
    band = refDs.GetRasterBand(1).ReadAsArray()
    shape = tuple([int(ratio*i) for i in band.shape])
    
    for raster in tqdm(src_dss):
        # Open raster dataset
        rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)
        # Extract band's data and transform into a numpy array
        test_ds = rasterDS.GetRasterBand(1).ReadAsArray()
        X.append(test_ds[:shape[0],:shape[1]].flatten())

    # Transpose attributes matrix
    X = np.dstack(tuple(X))[0]
    X = X.astype(np.float32)

    X[~np.isfinite(X)] = -1

    if normalize:
        normalizer = preprocessing.Normalizer().fit(X)
        X = normalizer.transform(X)  

    labelDS = gdal.Open(DS_FOLDER + "clipped_cos_50982.tif", gdal.GA_ReadOnly)
    y = labelDS.GetRasterBand(1).ReadAsArray()[:shape[0],:shape[1]].flatten()

    if osm_roads:
        labelDS = gdal.Open(DS_FOLDER + "roads_cos_50982.tif", gdal.GA_ReadOnly)
        roads = labelDS.GetRasterBand(1).ReadAsArray()[:shape[0],:shape[1]].flatten()
        y[roads == 4] = roads[roads == 4]

    maping_f = _class_map
    if binary:
        maping_f = _class_map_binary

    if map_classes:
        y = np.array([maping_f(yi) for yi in tqdm(y)])


    print("Prediction data: Done!")
    return X, y, shape

def load_timeseries(img_size):
    ts_dss = [TS_FOLDER + f for f in os.listdir(TS_FOLDER) if ("cos" not in f) and ("xml" not in f) and ("_" in f)]
    ts1_dss = [TS1_FOLDER + f for f in os.listdir(TS1_FOLDER) if ("cos" not in f) and ("xml" not in f) and ("_" in f)]
    image_files = ts_dss + ts1_dss # Your list of files
    
    image_height = img_size[0]
    image_width = img_size[1]

    image_stack = np.empty((image_height, image_width, len(image_files))) # Create empty HxWxN array/matrix

    for i, fname in enumerate(image_files):
            # Extract band's data and transform into a numpy array
            label_ds = gdal.Open(fname, gdal.GA_ReadOnly)
            label_bands = label_ds.GetRasterBand(1).ReadAsArray()
            image_stack[:, :, i] = label_bands # Set the i:th slice to this image
    return image_files


def load(train_size, datafiles=None, normalize=True, map_classes=True, binary=False, balance=False, test_size=0.2, osm_roads=True):
    X = []

    if(datafiles is None):
        src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER) if ("cos" not in f) and ("xml" not in f) and ("_" in f)]
        ts_dss = [TS_FOLDER + f for f in os.listdir(TS_FOLDER) if ("cos" not in f) and ("xml" not in f) and ("_" in f)]
        ts1_dss = [TS1_FOLDER + f for f in os.listdir(TS1_FOLDER) if ("cos" not in f) and ("xml" not in f) and ("_" in f)]

        src_dss = src_dss + ts_dss + ts1_dss
    else: src_dss = datafiles
    src_dss.sort()

    # Extract band's data and transform into a numpy array
    labelDS = gdal.Open(DS_FOLDER + "clipped_cos_50982.tif", gdal.GA_ReadOnly)
    labelBands = labelDS.GetRasterBand(1).ReadAsArray()

    labelDS = gdal.Open(DS_FOLDER + "roads_cos_50982.tif", gdal.GA_ReadOnly)
    roads = labelDS.GetRasterBand(1).ReadAsArray()

    # Prepare training data (set of pixels used for training) and labels
    isTrain = np.nonzero(labelBands)

    y = labelBands[isTrain]
    roads = roads[isTrain]

    if osm_roads:
        y[roads == 4] = roads[roads == 4]

    # Get list of raster bands info as array, already indexed by labels non zero
    print("Datasets: Loading...")
    for raster in tqdm(src_dss):
        if(("cos_50982.tif"  not in raster) and ("xml" not in raster)):
            # Open raster dataset
            rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)
            # Extract band's data and transform into a numpy array
            test_ds = rasterDS.GetRasterBand(1).ReadAsArray()
            X.append(test_ds[isTrain])
    
    # Transpose attributes matrix
    X = np.dstack(tuple(X))[0]

    print("Datasets: Done!           ") 
    print("Datasets: Features array shape, should be (n,k): " + str(X.shape))

    maping_f = _class_map
    if binary:
        maping_f = _class_map_binary

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(int(train_size*test_size)), train_size=min(X.shape[0], train_size), stratify=y, random_state=42)

    # Prevents overflow on algoritms computations
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    X_train[~np.isfinite(X_train)] = -1
    X_test[~np.isfinite(X_test)] = -1
    
    if normalize:
        print("Normalization: Loading...")
        normalizer = preprocessing.Normalizer().fit(X_train)
        X_train = normalizer.transform(X_train) 
        X_test = normalizer.transform(X_test)
        print("Done!")

    if balance:
        print("Data balance: Loading...")
        smt = TomekLinks(sampling_strategy='not minority', n_jobs=4, random_state=42)
        X_train, y_train = smt.fit_sample(X_train, y_train)
        print("Features array shape after balance: " + str(X_train.shape)) 

    if map_classes:
        print("Class Mapping: Loading...")
        y_train = np.array([maping_f(yi) for yi in tqdm(y_train)])
        y_test = np.array([maping_f(yi) for yi in tqdm(y_test)])
        print("Class Mapping: Done!      ")


    return X_train, y_train , X_test , y_test