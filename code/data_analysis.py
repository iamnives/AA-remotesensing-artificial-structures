import os
import sys

import gdal
 
import numpy as np
import matplotlib
import scipy.stats as stats
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit

from utils import data
import scipy.stats as stats

train_size = 100_000
X, y, Xt , yt = data.load(train_size, normalize=False, balance=True) 

print(X.shape)
f = plt.figure(1)
plt.title("Training set class distribution")
plt.hist(y, bins=np.arange(y.min(), y.max()+2), align='left', color='c',  histtype='bar', ec='black')
plt.xticks(np.arange(y.min(), y.max()+2))
plt.xlabel('Sample number')
plt.ylabel('Class')

print(Xt.shape)
fg = plt.figure(2)
plt.title("Testing set class distribution")
plt.hist(yt, bins=np.arange(yt.min(), yt.max()+2), align='left', color='c',  histtype='bar', ec='black')
plt.xticks(np.arange(yt.min(), yt.max()+2))
plt.xlabel('Sample number')
plt.ylabel('Class')

plt.show()
