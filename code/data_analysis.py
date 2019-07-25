import os
import sys

import gdal

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit

from utils.data import _class_map, _class_split_map
import scipy.stats as stats
from tqdm import tqdm

DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
TS_FOLDER = DS_FOLDER + "tstats/"
TS1_FOLDER = DS_FOLDER + "t1stats/"

train_size = int(19386625*0.20)

labelDS = gdal.Open(DS_FOLDER + "clipped_cos_50982.tif", gdal.GA_ReadOnly)
labelBands = labelDS.GetRasterBand(1).ReadAsArray()[:, :]

labelDS = gdal.Open(DS_FOLDER + "roads_cos_50982.tif", gdal.GA_ReadOnly)
roads = labelDS.GetRasterBand(1).ReadAsArray()
isTrain = np.nonzero(labelBands)

y = labelBands[isTrain]
roads = roads[isTrain]
y_roads = labelBands[isTrain]

y = np.array([_class_split_map(yi) for yi in tqdm(y)])
# y_roads[roads == 4] = roads[roads == 4]
# y_roads = np.array([_class_map(yi) for yi in tqdm(y_roads)])

# Split the dataset in two equal parts
_, _, y_train, _ = train_test_split(
    y, y, train_size=0.2, stratify=y, random_state=42)

# # Split the dataset in two equal parts
# _, _, y_train_roads, _ = train_test_split(
#     y_roads, y_roads, train_size=0.2, stratify=y_roads, random_state=42)

unique, counts = np.unique(y_train, return_counts=True)
barlist=plt.bar(unique, counts)
barlist[0].set_color('tab:orange')
barlist[1].set_color('tab:red')
barlist[2].set_color('tab:purple')
barlist[3].set_color('tab:green')
barlist[4].set_color('tab:cyan')

print("NOOSM", unique, counts)
plt.xticks((1, 2, 3, 4, 5), ('Estrutura - alta densidade', 'Estrutura - baixa densidade', 'Restante', 'Natural', 'Água'))
plt.title('')
plt.xlabel('Classe')
plt.ylabel('Amostras')

plt.show()
