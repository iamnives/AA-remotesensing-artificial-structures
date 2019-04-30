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

train_size = 1_000_000
X, y, _ , _ = data.load(train_size, balance=False) 

plt.hist(y, bins=np.arange(y.min(), y.max()+2), align='left', color='c')
plt.xticks(np.arange(y.min(), y.max()+2))

plt.show()