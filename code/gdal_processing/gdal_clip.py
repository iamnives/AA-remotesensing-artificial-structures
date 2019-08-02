import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gdal
import numpy

#inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "arbitrary/"
SRC_S1 = DATA_FOLDER + "datasets/" + ROI + "s1/"
SRC_S2 = DATA_FOLDER + "datasets/s2/"
SRC_DEM = DATA_FOLDER + "dem/"

LABELS = DATA_FOLDER + "labels/"+ ROI +"cos_50982.tif"
DST_FOLDER = DATA_FOLDER + "clipped/" + ROI
MASK = "../vector_data/" + ROI + "ROI.shp" 

def main(argv):
	# src_dss = [SRC_S1 + f for f in os.listdir(SRC_S1) if ".vrt" in f]
	# src2_dss = [SRC_S2 + f for f in os.listdir(SRC_S2) if ".vrt" in f]
	src3_dss = [SRC_DEM + f for f in os.listdir(SRC_DEM) if ".vrt" in f]

	# src_dss = src_dss + src3_dss + src2_dss + [LABELS]

	for f in src3_dss:
		outFile = f.split("/")[-1]
		#cut_command = f"gdalwarp -t_srs EPSG:32629 -r near -of GTiff -tr 10 10 -tap -cutline {MASK} -crop_to_cutline {f} {DST_FOLDER + 'clipped_' + outFile}"
		gdal.Warp(DST_FOLDER + 'clipped_' + outFile, f , dstSRS="EPSG:32629", resampleAlg="near", format="GTiff", xRes=10, yRes=10, cutlineDSName=MASK, cropToCutline=True)
		#os.system(cut_command)
	

if __name__== "__main__":
  main(sys.argv)