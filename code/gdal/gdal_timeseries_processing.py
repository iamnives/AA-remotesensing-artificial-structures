import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gdal
import numpy
from tqdm import tqdm

#inicialize data location
DATA_FOLDER = "../sensing_data/raw/"
ROI = "vila-de-rei/"
SRC_S2 = DATA_FOLDER + "timeseries/" + ROI

DST_FOLDER = "../sensing_data/" + "clipped/" + ROI + "ts/"
MASK = "../vector_data/" + ROI + "ROI.shp" 
 
def main(argv):
	src_dss = [SRC_S2 + f for f in os.listdir(SRC_S2) if ".zip" not in f and "S2A" in f]
	src_dss.sort()

	for idx, f in enumerate(tqdm(src_dss)):
		for f1 in os.listdir(f):
			#cut_command = f"gdalwarp -t_srs EPSG:32629 -r near -of GTiff -tr 10 10 -tap -cutline {MASK} -crop_to_cutline {f} {DST_FOLDER + 'clipped_' + outFile}"
			gdal.Warp(DST_FOLDER + str(idx) + 'clipped_' + f1, f + "/" + f1 , dstSRS="EPSG:32629", resampleAlg="near", format="GTiff", xRes=10, yRes=10, cutlineDSName=MASK, cropToCutline=1)
			#os.system(cut_command)

if __name__== "__main__":
  main(sys.argv)