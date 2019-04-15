import gdal
import sys
import os 

#inicialize data location
DATA_FOLDER = "../sensing_data/"
SRC_FOLDER = DATA_FOLDER + "dem/"

def calculate_slope(elevation, dst):
    gdal.DEMProcessing(dst + '_slope.tif', elevation, 'slope')
 

def calculate_aspect(elevation, dst):
    gdal.DEMProcessing(dst + '_aspect.tif', elevation, 'aspect')

def main(argv):
    src_dss = [SRC_FOLDER + f for f in os.listdir(SRC_FOLDER)]
    for fname in src_dss:
        calculate_slope(fname, ".." +  fname.split(".")[2])
        calculate_aspect(fname, ".." + fname.split(".")[2])

    # Build VRT
    vrt_options = gdal.BuildVRTOptions(resolution='lowest')

    for data in ["v3.tif","_slope.tif", "_aspect.tif"]:
        src_dss = [SRC_FOLDER + f for f in os.listdir(SRC_FOLDER) if data in f ]
        gdal.BuildVRT(SRC_FOLDER + data.split(".")[0] + '.vrt', src_dss, options=vrt_options)

    
 
if __name__== "__main__":
  main(sys.argv)