import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gdal
import numpy
from tqdm import tqdm

#inicialize data location
DATA_FOLDER = "../sensing_data/raw/timeseries/"
ROI = "vila-de-rei/"
MASK = "../vector_data/" + ROI + "ROI.shp" 


SRC = DATA_FOLDER + ROI + "/s1_corrected/"
DST_FOLDER = "../sensing_data/" + "clipped/" + ROI + "ts1/"

 
def main(argv):
	src_dss = [SRC + f for f in os.listdir(SRC) if ".dim" not in f]
	src_dss.sort()

	for idx, f in enumerate(tqdm(src_dss)):
		for f1 in os.listdir(f):
			if ".img" in f1:
				gdal.Warp(DST_FOLDER + str(idx) +'clipped_' + f1, f + "/" + f1 , dstSRS="EPSG:32629", resampleAlg="near", format="GTiff", xRes=10, yRes=10, cutlineDSName=MASK, cropToCutline=1)


if __name__== "__main__":
  main(sys.argv)