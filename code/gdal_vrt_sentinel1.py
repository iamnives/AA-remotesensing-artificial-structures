
import gdal
import numpy
import os
import sys
import osr

#inicialize data location
DATA_FOLDER = "../sensing_data/"
SRC_FOLDER = DATA_FOLDER + "datasets/s1/"

def main(argv):
    # Reference files
    src_dss = [f for f in os.listdir(SRC_FOLDER) if ".img" in f]
    vrt_options = gdal.BuildVRTOptions(resolution='lowest')
    for f in src_dss:
        gdal.BuildVRT(SRC_FOLDER + 'sentinel1_' + f.split(".")[0] + '.vrt', SRC_FOLDER + f, options=vrt_options)
    
if __name__== "__main__":
  main(sys.argv)