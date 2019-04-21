import os
import sys

import gdal

import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

print("Normalizing data N(0,1)!")
# subtract means form the input data
X -= np.mean(X, axis=1)[:,None]
# normalize the data
X /= np.sqrt(np.sum(X*X, axis=1))[:,None]
print("Done!")

X = np.dstack(tuple(X))[0]

N_COMPUTING = 100000
n_samples = X.shape[0]

n_samples_per = N_COMPUTING/n_samples

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

print("Labels array shape, should be (n,): " + str(y_train.shape))
print("Training array shape, should be (n,k): " + str(X_train.shape))

fig7, ax7 = plt.subplots()
ax7.set_title('Multiple Samples with Different sizes')
ax7.hist(y_test)

plt.show()
