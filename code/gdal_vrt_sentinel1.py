
import gdal
import numpy
import os
import sys
import osr

#inicialize data location
DATA_FOLDER = "../sensing_data/"
SRC_FOLDER = DATA_FOLDER + "datasets/s1"
dS1_FNAME = "s1b-iw-grd-"

def main(argv):
    # Bands to process and merge (add intensity, gamma, beta, etc.. on main PC)
    bands =	{
        "vv":[],
        "vh": [],
    }
    # Reference files
    src_dss = [f for f in os.listdir(SRC_FOLDER) if S1_FNAME in f]
    for f in src_dss:
       bands[f.split("-")[3]].append(SRC_FOLDER + f)

    vrt_options = gdal.BuildVRTOptions(resolution='lowest')
    for b in bands:
        gdal.BuildVRT(SRC_FOLDER + 'sentinel1_' + b + '.vrt', bands[b], options=vrt_options)
    
if __name__== "__main__":
  main(sys.argv)