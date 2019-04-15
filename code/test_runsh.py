import numpy as np
import os
import gdal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import sys

#inicialize data location
DATA_FOLDER = "../sensing_data/"
DS_FOLDER = DATA_FOLDER + "clipped/"
LB_FOLDER = DATA_FOLDER + "labels/"

outRaster = DATA_FOLDER + "results/classification.tiff"

bandsData = []

src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER) if ".vrt" in f]

labelDS = gdal.Open(DS_FOLDER + "clipped_cos.tif", gdal.GA_ReadOnly)

# Extract band's data and transform into a numpy array
labelBands = labelDS.GetRasterBand(1).ReadAsArray()
# Prepare training data (set of pixels used for training) and labels
isTrain = np.nonzero(labelBands)
trainingLabels = labelBands[isTrain]

print("Labels: " + str(trainingLabels.shape))
# Get list of raster bands info as array, already indexed by labels non zero

for idx, raster in enumerate(src_dss):
    # Open raster dataset
    print("Opening raster: " + raster)
    rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)

    # Extract band's data and transform into a numpy array
    band = rasterDS.GetRasterBand(1)
    bandsData.append(band.ReadAsArray()[isTrain])
    
print("Done!")
measures= np.vstack(tuple(bandsData))
bandsData=None
trainingData = measures.T
print(trainingData.shape)

imgplot = plt.imshow(labelBands)
plt.show()