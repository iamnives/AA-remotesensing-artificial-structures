from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import gdal
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import visualization as viz


# inicialize data location
DATA_FOLDER = "D:/sat_data_big/gee/"
SRC_S2 = DATA_FOLDER + "\clipped_l2_rgb/"
SRC_FOLDER = SRC_S2 + "tiles/64/"

SRC_GT = SRC_FOLDER + "clipped_cos_clipped_fixed/"
SRC_S2 = SRC_FOLDER + "clipped_sentinel2_complete_fixed_fixed/"

def main(argv):

    src_dss_s2 = [f for f in os.listdir(SRC_S2) if (
        ".jp2" in f) or (".tif" in f) or (".img" in f)]

    src_dss = src_dss_s2
    print("Validating dataset...")
    no_data_file_count = 0
    for f in src_dss:

        tile_gt = gdal.Open(SRC_GT + f, gdal.GA_ReadOnly)
        tile_s2 = gdal.Open(SRC_S2 + f, gdal.GA_ReadOnly)

        band_s2 = tile_s2.GetRasterBand(1).ReadAsArray()
        band_gt = tile_gt.GetRasterBand(1).ReadAsArray()

        zeros_s2 = np.count_nonzero(band_s2 == 0)
        zeros_gt = np.count_nonzero(band_gt == 0)
        tile_gt = None
        tile_s2 = None

        if zeros_s2 > 0 or zeros_gt > 0:
            os.remove(SRC_GT + f)
            os.remove(SRC_S2 + f)
            no_data_file_count += 1
        
    print(no_data_file_count)

if __name__ == "__main__":
    main(sys.argv)
