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

from utils import data

train_size = 5_000_000
X, y, _ , _ = data.load(train_size) 

# # Get list of raster bands info as array, already indexed by labels non zero
# test_ds = None
# for idx, raster in enumerate(src_dss):
#     if("cos_50982.tif" not in raster):
#         # Open raster dataset
#         print("Opening raster: " + raster)
#         rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)
#         # Extract band's data and transform into a numpy array
#         test_ds = rasterDS.GetRasterBand(1).ReadAsArray()
#         X.append(test_ds[isTrain])
    
# print("Done!")

# X = np.dstack(tuple(X))[0]

plt.hist(y, bins=np.arange(y.min(), y.max()+1), align='left', color='c')
plt.xticks(np.arange(y.min(), y.max()+1))

plt.show()