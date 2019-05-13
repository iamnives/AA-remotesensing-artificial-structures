import gdal
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#inicialize data location
DATA_FOLDER = "../sensing_data/"
SRC_FOLDER = DATA_FOLDER + "dem/"
def main(argv):
    # Build VRT
    vrt_options = gdal.BuildVRTOptions(resolution='lowest')
    src_dss = [SRC_FOLDER + f for f in os.listdir(SRC_FOLDER) if ".hgt" in f ]
    gdal.BuildVRT(SRC_FOLDER + 'dem.vrt', src_dss, options=vrt_options)

if __name__== "__main__":
  main(sys.argv)