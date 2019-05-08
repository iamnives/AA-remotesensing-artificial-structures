import os
import sys

import gdal
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from tqdm import tqdm

#inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/" + ROI + "classification.tiff"

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
    return -1

def _class_map_binary(x):
    if x >= 1 and x <= 13:
        return 1
    else:
        return 2
    return -1

def get_features():
    src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER) if ("cos_50982.tif"  not in f) and ("xml" not in f) ]
    src_dss.sort()
    return np.array(src_dss)

def load_prediction(src_folder, normalize=True):
    src_dss = [src_folder + f for f in os.listdir(src_folder) if ("cos_50982.tif" not in f) and ("xml" not in f) ]
    src_dss.sort()
    X = []

    refDs = gdal.Open(src_folder + "clipped_sentinel2_B03.vrt", gdal.GA_ReadOnly)
    band = refDs.GetRasterBand(1).ReadAsArray()

    nonZero = np.nonzero(band)
    for raster in tqdm(src_dss):
        # Open raster dataset
        rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)
        # Extract band's data and transform into a numpy array
        test_ds = rasterDS.GetRasterBand(1).ReadAsArray()
        X.append(test_ds[nonZero])

    # Transpose attributes matrix
    X = np.dstack(tuple(X))[0]
    X = X.astype(np.float64)
    X[np.isnan(X)]=-1

    if normalize:
        normalizer = preprocessing.Normalizer().fit(X)
        X = normalizer.transform(X) 

    labelDS = gdal.Open(src_folder + "clipped_cos_50982.tif", gdal.GA_ReadOnly)
    return X, labelDS.GetRasterBand(1).ReadAsArray()

def load(train_size, datafiles=None, normalize=True, map_classes=True, binary=False, balance=False, test_size=0.2):
    X = []

    if(datafiles is None):
        src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER)]
    else: src_dss = datafiles
    src_dss.sort()
    labelDS = gdal.Open(DS_FOLDER + "clipped_cos_50982.tif", gdal.GA_ReadOnly)

    # Extract band's data and transform into a numpy array
    labelBands = labelDS.GetRasterBand(1).ReadAsArray()
    # Prepare training data (set of pixels used for training) and labels
    isTrain = np.nonzero(labelBands)
    y = labelBands[isTrain]

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
    X = X.astype(np.float64)

    print("Datasets: Done!           ") 
    print("Datasets: Features array shape, should be (n,k): " + str(X.shape))

    maping_f = _class_map
    if binary:
        maping_f = _class_map_binary

    if map_classes:
        print("Class Mapping: Loading...")
        y = np.array([maping_f(yi) for yi in tqdm(y)])
        print("Class Mapping: Done!      ")



    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(int(train_size*test_size)), stratify=y, random_state=42)

    # Prevents overflow on algoritms computations
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    X_train[np.isnan(X_train)]=-1
    X_test[np.isnan(X_test)]=-1
    
    if normalize:
        normalizer = preprocessing.Normalizer().fit(X_train)
        X_train = normalizer.transform(X_train) 
        X_test = normalizer.transform(X_test)

   

     # Split the dataset in two equal parts
    X_train, _, y_train , _ = train_test_split(
        X_train, y_train, train_size=min(X_train.shape[0], train_size), stratify=y_train ,random_state=42)

    if balance:
        smt = SMOTE(sampling_strategy='auto', random_state=42)
        X_train, y_train = smt.fit_sample(X_train, y_train)
        print("Features array shape after balance: " + str(X_train.shape)) 

    return X_train, y_train , X_test , y_test