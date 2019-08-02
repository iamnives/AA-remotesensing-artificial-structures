"""
Created on Sun Mar  3 21:42:16 2019

@author: André Neves
"""
import os
import sys
import gdal
from utils import visualization as viz
from utils import data
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "arbitrary/"

S1_FOLDER = DATA_FOLDER + "clipped/" + ROI + "ts1/"
S2_FOLDER = DATA_FOLDER + "clipped/" + ROI + "ts/"


def main(argv):

    s1_names = [f for f in os.listdir(S1_FOLDER)]
    s2_names = [f for f in os.listdir(S2_FOLDER)]
    s1_names.sort()
    s2_names.sort()

    for idx, f in tqdm(enumerate(s1_names)):
        os.rename(S1_FOLDER + f, S1_FOLDER + str(idx//2) + "clipped_" + f[7:])

    # for idx, f in tqdm(enumerate(s2_names)):
    #     os.rename(S2_FOLDER + f, S2_FOLDER + str(idx//13) + "clipped_" + f[6:])


if __name__ == "__main__":
    main(sys.argv)