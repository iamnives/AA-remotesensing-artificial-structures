import gdal
import numpy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import osr
from pathlib import Path

#inicialize data location
DATA_FOLDER = "G:\My Drive\gee Sentinel2-fixed-2yr"
SRC_FOLDER = DATA_FOLDER + "\data"
OUT_FOLDER = DATA_FOLDER + "\\vrt"

images = []

def main(argv):

    for path in Path(SRC_FOLDER).rglob('*.tif'):
      str_path = str(path)
      images.append(str_path)

    vrt_options = gdal.BuildVRTOptions(resolution='highest')
    gdal.BuildVRT(OUT_FOLDER + '\sentinel2_complete.vrt', images, options=vrt_options)
    
if __name__== "__main__":
  main(sys.argv)