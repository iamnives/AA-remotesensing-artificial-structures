import numpy as np
import os
import gdal
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
plt.switch_backend('Qt4Agg') 

#inicialize data location
DATA_FOLDER = "../sensing_data/"
DS_FOLDER = DATA_FOLDER + "clipped/"
LB_FOLDER = DATA_FOLDER + "labels/"

outRaster = DATA_FOLDER + "results/classification.tiff"

bandsData = []
geo_transform = None
projection = None

src_dss = [DS_FOLDER + f for f in os.listdir(DS_FOLDER) if ".vrt" in f]

labelDS = gdal.Open(DS_FOLDER + "clipped_cos.tif", gdal.GA_ReadOnly)

# Extract band's data and transform into a numpy array
labelBands = labelDS.GetRasterBand(1).ReadAsArray()
# Prepare training data (set of pixels used for training) and labels
isTrain = np.nonzero(labelBands)
trainingLabels = labelBands[isTrain]

print("Labels: " + str(trainingLabels.shape))

# Get list of raster bands info as array, already indexed by labels non zero

for raster in src_dss:
    # Open raster dataset
    rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)
    # Get spatial reference
    geo_transform = rasterDS.GetGeoTransform()
    projection = rasterDS.GetProjectionRef()

    # Extract band's data and transform into a numpy array
    band = rasterDS.GetRasterBand(1)
    bandsData.append(band.ReadAsArray()[isTrain])



print(np.count_nonzero(bandsData[0]==-32767))


#probs wrong
measures= np.vstack(tuple(bandsData))

print(measures)
print(measures.shape)
imgplot = plt.imshow(labelBands)
plt.show()