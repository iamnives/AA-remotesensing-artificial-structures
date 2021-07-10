import numpy as np
import gdal
import os
import sys
import scipy
import cv2
import math
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# inicialize data location
DATA_FOLDER = "D:/sat_data_big/gee/"
SRC_S2 = DATA_FOLDER + "\clipped_l2_rgb/"
DST_FOLDER = SRC_S2 + "tiles/"

# [32, 64, 128, 256, 512]
TILE_SIZES = [64]

def main(argv):
    src2_dss = [SRC_S2 + f for f in os.listdir(SRC_S2) if ".tif" in f and ".xml" not in f]

    src_dss = src2_dss
    print("Processing the following files:\n")
    src_dss.reverse()
    print(src_dss)

    for size in TILE_SIZES:
        print(f"Creating tiles of size: {size}")
        # load image into memory
        img_name_1 = src_dss[0].split('/')[-1].split('.')[0]
        img_name_cos = src_dss[1].split('/')[-1].split('.')[0]

        ref_ds = gdal.Open(src_dss[0], gdal.GA_ReadOnly)
        ref_ds_cos = gdal.Open(src_dss[1], gdal.GA_ReadOnly)

        rows, cols = size, size
        band = ref_ds.GetRasterBand(1)
        xsize, ysize = band.XSize, band.YSize
        x_edge, y_edge = int(xsize - cols + 1), int(ysize - rows + 1)
        # x_extra, y_extra = int(x_edge%cols), int(y_edge%rows)

        out_out_dir = DST_FOLDER + f"{size}/{img_name_1}/"
        out_out_dir_cos = DST_FOLDER + f"{size}/{img_name_cos}/"

        if not os.path.exists(out_out_dir):
            os.makedirs(out_out_dir)
        
        if not os.path.exists(out_out_dir_cos):
            os.makedirs(out_out_dir_cos)

        current_tile = 0
        for j in tqdm(range(0, y_edge, rows)):
            for i in range(0, x_edge, cols):
                gdal.Translate(out_out_dir + "test_" + str(i) + "_" + str(j) + "_" + str(current_tile) + ".tif", ref_ds, srcWin=[i, j, size, size])
                gdal.Translate(out_out_dir_cos + "test_" + str(i) + "_" + str(j) + "_" + str(current_tile) + ".tif", ref_ds_cos, srcWin=[i, j, size, size])
                current_tile += 1
    



if __name__ == "__main__":
    main(sys.argv)
