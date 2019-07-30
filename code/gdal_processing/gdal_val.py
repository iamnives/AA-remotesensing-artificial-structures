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
DATA_FOLDER = "../sensing_data/"
ROI = "arbitrary/"
SRC = DATA_FOLDER + "clipped/" + ROI
SRC_FOLDER = SRC + "ts1/"


def main(argv):
    print("Validating dataset...")
    src_dss = [f for f in os.listdir(SRC_FOLDER) if (
        ".jp2" in f) or (".tif" in f) or (".img" in f)]
    src_dss.sort()

    refDs = gdal.Open("../sensing_data/clipped/" + ROI +
                      "ignored/static/clipped_sentinel2_B08.vrt", gdal.GA_ReadOnly)
    band = refDs.GetRasterBand(1).ReadAsArray()
    ref_shape = band.shape

    for f in tqdm(src_dss):
        refDs = gdal.Open(SRC_FOLDER + f, gdal.GA_ReadOnly)
        band = refDs.GetRasterBand(1).ReadAsArray()
        bshape = band.shape
        if ref_shape != bshape:
            print(f"Failed validation, raster with different shape: {f}, {bshape}")
            print(f"It may still work, but needs checking to validate results.")
            return 1
    print("Passed validation")

if __name__ == "__main__":
    main(sys.argv)
