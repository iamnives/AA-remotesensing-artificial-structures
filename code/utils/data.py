import os
import sys

import gdal
import numpy as np
#inicialize data location
DATA_FOLDER = "../sensing_data/"
DS_FOLDER = DATA_FOLDER + "clipped/"
LB_FOLDER = DATA_FOLDER + "labels/"
OUT_RASTER = DATA_FOLDER + "results/classification.tiff"

def load():
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
        if("cos.tif" not in raster):
            # Open raster dataset
            print("Opening raster: " + raster)
            rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)
            # Extract band's data and transform into a numpy array
            test_ds = rasterDS.GetRasterBand(1).ReadAsArray()
            X.append(test_ds[isTrain])
        
    print("Done!") 

    X = np.dstack(tuple(X))[0]
    return X, y