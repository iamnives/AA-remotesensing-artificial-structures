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
ROI = "vila-de-rei/"
SRC = DATA_FOLDER + "clipped/" + ROI
SRC_FOLDER = SRC + "ts1/"


def main(argv):
    print("Validating dataset...")
    src_dss = [f for f in os.listdir(SRC_FOLDER) if (
        ".jp2" in f) or (".tif" in f) or (".img" in f)]
    src_dss.sort()
    ref_shape = (1939, 2501)
    # Reference files
    for f in tqdm(src_dss):
        refDs = gdal.Open(SRC_FOLDER + f, gdal.GA_ReadOnly)
        band = refDs.GetRasterBand(1).ReadAsArray()
        bshape = band.shape
        if ref_shape != bshape:
            print("Failed")
            return 1
    print("Passed")


# np.mean(a, axis=0), np.quantile(a, 0.25, axis=0),
if __name__ == "__main__":
    main(sys.argv)
