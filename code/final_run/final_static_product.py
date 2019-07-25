"""
Created on Sun Mar  3 21:42:16 2019

@author: André Neves
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
from tqdm import tqdm
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import numpy as np
from datetime import timedelta
import time
from utils import visualization as viz
from utils import data
import gdal
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from xgboost import plot_importance
from joblib import dump, load
import xgboost as xgb
import matplotlib.pyplot as plt
from feature_selection import fselector
from sklearn.model_selection import train_test_split

# inicialize data location
DATA_FOLDER = "../sensing_data/portugal/"

DS_FOLDER_S1 = DATA_FOLDER + "s1/"
DS_FOLDER_S2 = DATA_FOLDER + "s2/"

def reproject():
    src_folders = [DS_FOLDER_S2 + f for f in os.listdir(DS_FOLDER_S2) if "10x10_" not in f]
    src_folders.sort()
    
    for idx, f in enumerate(tqdm(src_folders)):
        for f1 in os.listdir(f):
            if ".jp2" in f1:
                gdal.Warp("10x10_" + f + "/" + f1, f + "/" + f1, dstSRS="EPSG:32629",
                          resampleAlg="near", format="GTiff", xRes=10, yRes=10)

def model():
    forest = xgb.XGBClassifier(colsample_bytree=0.7553707061597048,
                            gamma=5,
                            gpu_id=0,
                            learning_rate=0.2049732654267658,
                            max_depth=8,
                            min_child_weight=1,
                            max_delta_step=9.075685204314162,
                            n_estimators=1500,
                            n_jobs=4,
                            objective='multi:softmax',  # binary:hinge if binary classification
                            predictor='cpu_predictor',
                            tree_method='gpu_hist')
    return forest

def main(argv):
    reproject()


if __name__ == "__main__":
    main(sys.argv)