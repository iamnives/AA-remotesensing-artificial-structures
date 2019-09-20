
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
import csv

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"


def write_to_file(line):
    with open('./features_static.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(line)


def run():
    f_names = data.get_features()
    for idx, feature in enumerate(f_names):
        line = [idx, feature]
        write_to_file(line)

if __name__ == "__main__":
    run()
