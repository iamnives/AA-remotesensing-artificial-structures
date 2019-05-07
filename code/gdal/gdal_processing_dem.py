import gdal
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "lisboa-setubal/"

SRC_FOLDER = DATA_FOLDER + "clipped/" + ROI

def calculate_slope(elevation, dst):
    gdal.DEMProcessing(dst + '_slope.tif', elevation, 'slope')
 

def calculate_aspect(elevation, dst):
    gdal.DEMProcessing(dst + '_aspect.tif', elevation, 'aspect')

def main(argv):
    src_dss = [SRC_FOLDER + f for f in os.listdir(SRC_FOLDER) if "dem.vrt" in f]
    for fname in src_dss:
        calculate_slope(fname, ".." +  fname.split(".")[2])
        calculate_aspect(fname, ".." + fname.split(".")[2])


if __name__== "__main__":
  main(sys.argv)