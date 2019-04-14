import gdal
import sys
import os 

#inicialize data location
DATA_FOLDER = "../sensing_data/"
SRC_FOLDER = DATA_FOLDER + "dem/"

DEM = SRC_FOLDER + "DEM.tiff"

def calculate_slope(elevation):
    gdal.DEMProcessing(elevation + '_slope.tif', elevation, 'slope')
 

def calculate_aspect(elevation):
    gdal.DEMProcessing(elevation + '_aspect.tif', elevation, 'aspect')

def main(argv):
    src_dss = [SRC_FOLDER + f for f in os.listdir(SRC_FOLDER)]
    for fname in src_dss:
        calculate_slope(fname)
        calculate_aspect(fname)

 
if __name__== "__main__":
  main(sys.argv)