import gdal
import numpy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import osr
from pathlib import Path

#inicialize data location
DATA_FOLDER = "E:\sat_data\s2-2018-nocloud"
SRC_FOLDER = DATA_FOLDER + "\data"
OUT_FOLDER = DATA_FOLDER + "\\vrt"

bands =	{
  "TCI":[],
  "B01": [],
  "B02": [],
  "B03": [],
  "B04": [],
  "B05": [],
  "B06": [],
  "B07": [],
  "B08": [],
  "B8A": [],
  "B09": [],
  "B10": [],
  "B11": [],
  "B12": [],
  "PVI": [],
}

def main(argv):

    for path in Path(SRC_FOLDER).rglob('*.jp2'):
      str_path = str(path)
      bands[str_path.split('_')[-1].split('.')[0]].append(str_path)
    vrt_options = gdal.BuildVRTOptions(resolution='highest')
    for b in bands:
        if(len(bands[b])> 0):
          gdal.BuildVRT(OUT_FOLDER + '\sentinel2_' + b + '.vrt', bands[b], options=vrt_options)
    
if __name__== "__main__":
  main(sys.argv)