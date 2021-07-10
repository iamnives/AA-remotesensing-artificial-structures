import numpy
import gdal
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# inicialize data location
DATA_FOLDER = "D:/sat_data_big/gee/ua2018-data/clipped"

LABELS = DATA_FOLDER + "cos_clipped.tif"
DST_FOLDER = GEE_DATA_FOLDER + "clipped/"




def main(argv):
    src2_dss = [SRC_S2 + f for f in os.listdir(SRC_S2)]

    src_dss = src2_dss
    #src_dss.append(LABELS)
    
    for f in tqdm(src_dss):
        outFile = f.split("/")[-1].split('.')[0]
        gdal.Warp(DST_FOLDER + 'clipped_' + outFile + '.tif', f, dstSRS="EPSG:32629", resampleAlg="near", format="GTiff", xRes=10, yRes=10, cutlineDSName=MASK, cropToCutline=True)


if __name__ == "__main__":
    main(sys.argv)
