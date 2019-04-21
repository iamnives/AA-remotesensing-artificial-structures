import os
import sys

import gdal

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

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

print("Labels array shape, should be (n,): " + str(y.shape))

# Get list of raster bands info as array, already indexed by labels non zero
test_ds = None
for idx, raster in enumerate(src_dss):
    # Open raster dataset
    print("Opening raster: " + raster)
    rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)
    # Extract band's data and transform into a numpy array
    test_ds = rasterDS.GetRasterBand(1).ReadAsArray()
    X.append(test_ds[isTrain])
    
print("Done!") 

X = np.dstack(tuple(X))[0]
print(X.shape)

N_COMPUTING = 100000
n_samples = X.shape[0]

n_samples_per = N_COMPUTING/n_samples

# Split the dataset in two equal parts
_, X_train, _, y_train = train_test_split(
    X, y, test_size=n_samples_per)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=n_samples_per/2)

# Shuffle the data
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)

X_train = X_train[indices]
y_train = y_train[indices]

indices = np.arange(X_test.shape[0])
np.random.shuffle(indices)

X_test = X_test[indices]
y_test = y_test[indices]

print("Train: " + str(X_train.shape), "Test: " + str(X_test.shape))

# Set the parameters by cross-validation
tuning_params = [{'n_estimators': [1, 2, 4]}]

scores = ['f1_micro', 'accuracy', 'precision', 'roc_auc', 'recall']

print("# Tuning hyper-parameters for %s" % scores)
print()

clf = GridSearchCV(RandomForestClassifier(), tuning_params, cv=5,
                    scoring=scores, refit='f1_micro' ,verbose=1)
clf.fit(X_train, y_train)

print("Best parameters set found on development set: f1_micro")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
full_scores = pd.DataFrame(clf.cv_results_)
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

