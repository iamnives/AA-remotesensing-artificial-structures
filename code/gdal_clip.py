import gdal
import numpy
import os
import sys
import osr

#inicialize data location
DATA_FOLDER = "../sensing_data/"
SRC_S1 = DATA_FOLDER + "datasets/s1/"
SRC_S2 = DATA_FOLDER + "datasets/s2/"
SRC_DEM = DATA_FOLDER + "dem/"
SRC_INDEXES = DATA_FOLDER + "indexes/"

LABELS = DATA_FOLDER + "labels/cos.tif"
DST_FOLDER = DATA_FOLDER + "clipped/"
MASK = "../vector_data/ROI_test.shp"

def main(argv):
	src_dss = [SRC_S1 + f for f in os.listdir(SRC_S1) if ".vrt" in f]
	src2_dss = [SRC_S2 + f for f in os.listdir(SRC_S2) if ".vrt" in f]
	src3_dss = [SRC_DEM + f for f in os.listdir(SRC_DEM) if ".vrt" in f]
	#src_dss = [SRC_INDEXES + f for f in os.listdir(SRC_INDEXES) if ".vrt" in f]

	src_dss = src_dss + src3_dss + src2_dss + [LABELS]

	for f in src_dss:
		outFile = f.split("/")[-1]
		cut_command = f"gdalwarp -t_srs EPSG:32629 -r near -of GTiff -tr 10 10 -tap -cutline {MASK} -crop_to_cutline {f} {DST_FOLDER + 'clipped_' + outFile}"
		os.system(cut_command)
	

if __name__== "__main__":
  main(sys.argv)