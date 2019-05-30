"""
Created on Thu May  30

@author: André Neves
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import scipy.signal
import gdal
from utils import visualization as viz
import time
from datetime import timedelta
import numpy as np

import scipy.signal

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
TS_FOLDER = DS_FOLDER + "tstats/"
TS1_FOLDER = DS_FOLDER + "t1stats/"

OUT_RASTER = DS_FOLDER + "results/roads_test.tif"

def test_pred():
    start = time.time()

    raster_ds = gdal.Open(DS_FOLDER + "/ignored/static/clipped_sentinel2_B03.vrt", gdal.GA_ReadOnly)
    band = raster_ds.GetRasterBand(1).ReadAsArray()

    # Square average kernel gives box blur.
    filter_kernel = [[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]]
    band_convolved = scipy.signal.convolve2d(band, filter_kernel, mode='same', boundary='fill', fillvalue=0)

    viz.createGeotiff(DS_FOLDER + "/ignored/static/convolved.tif", band_convolved, DS_FOLDER + "/ignored/static/clipped_sentinel2_B03.vrt", gdal.GDT_Float32)

    end = time.time()
    elapsed = end-start
    print("Run time: " + str(timedelta(seconds=elapsed)))
    

if __name__ == '__main__':
    test_pred()
