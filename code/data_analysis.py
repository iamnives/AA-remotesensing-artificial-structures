import os
import sys

import gdal

import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
#inicialize data location
DATA_FOLDER = "../sensing_data/"
DS_FOLDER = DATA_FOLDER + "clipped/"
LB_FOLDER = DATA_FOLDER + "labels/"

outRaster = DATA_FOLDER + "results/classification.tiff"

X = []

src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER)]

labelDS = gdal.Open(DS_FOLDER + "clipped_cos.tif", gdal.GA_ReadOnly)

# Extract band's data and transform into a numpy array
labelBands = labelDS.GetRasterBand(1).ReadAsArray()
# Prepare training data (set of pixels used for training) and labels
isTrain = np.nonzero(labelBands)
y = labelBands[isTrain]


# Get list of raster bands info as array, already indexed by labels non zero
test_ds = None
for idx, raster in enumerate(src_dss):
    if("cos.tif" not in raster):
        # Open raster dataset
        print("Opening raster: " + raster)
        rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)
        # Extract band's data and transform into a numpy array
        test_ds = rasterDS.GetRasterBand(1).ReadAsArray()
        X.append(test_ds[isTrain])
    
print("Done!")

X = np.dstack(tuple(X))[0]

plt.hist(y, bins=np.arange(y.min(), y.max()+4), align='left')
plt.xticks(np.arange(y.min(), y.max()+4))
plt.show()