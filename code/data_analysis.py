import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gdal

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit

from utils import data
import scipy.stats as stats
from tqdm import tqdm

DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
TS_FOLDER = DS_FOLDER + "tstats/"
TS1_FOLDER = DS_FOLDER + "t1stats/"

DS_PLOT_FOLDER = DATA_FOLDER + "results/" + ROI + "static/data_plots/group2/"

train_size = int(19386625*0.2)
X, y, _, _ = data.load(
        train_size, normalize=False, balance=False, osm_roads=False, split_struct=True, army_gt=False)

def data_dist():
    unique, counts = np.unique(y, return_counts=True)
    barlist=plt.bar(unique, counts)
    barlist[0].set_color('tab:orange')
    barlist[1].set_color('tab:purple')
    barlist[2].set_color('tab:green')
    barlist[3].set_color('tab:cyan')

    print("NOOSM", unique, counts)
    plt.xticks((1, 2, 3, 4), ('Estrutura', 'Estrada', 'Natural', 'Água'))
    plt.title('')
    plt.xlabel('Classe')
    plt.ylabel('Amostras')

    plt.show()

X_dense = X[y == 1]
X_rural = X[y == 1] 
X_estrutura = X[y == 1] 

X_natural = X[y == 2]
X_agua = X[y == 3]

feature_names = data.get_features()

# extract wat we want to evaluate
for idx, feature in enumerate(feature_names):
    feature_est_dense = X_dense[:,idx]
    feature_est_rural = X_rural[:,idx]
    feature_est = X_estrutura[:,idx]

    feature_nat = X_natural[:,idx]
    feature_wat = X_agua[:,idx]

    data = [feature_est_dense, feature_est_rural, feature_est, feature_nat, feature_wat]
    fig1, ax1 = plt.subplots()
    ax1.set_title(f'Box Plot of attribute: {feature}')
    ax1.boxplot(data, showfliers=False)
    plt.xticks([1, 2, 3, 4, 5], ['Estrutura urbana', 'Estrutura rural' , 'Outras estruturas' ,'Natural', 'Água'])
    plt.xticks(rotation=45, ha='right')
    plt.savefig(DS_PLOT_FOLDER + f'{feature}.pdf', bbox_inches='tight')