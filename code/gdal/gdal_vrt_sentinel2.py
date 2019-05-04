import gdal
import numpy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import osr

#inicialize data location
DATA_FOLDER = "../sensing_data/"
SRC_FOLDER = DATA_FOLDER + "datasets/s2/"

bands =	{
  "AOT":[],
  "SCL": [],
  "WVP": [],
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
  "B11": [],
  "B12": [],
  "B01": [],
}

def main(argv):
    # Reference files
    for f in os.listdir(SRC_FOLDER):
       bands[f.split("_")[3]].append(SRC_FOLDER + f)

    vrt_options = gdal.BuildVRTOptions(resolution='lowest')
    for b in bands:
        if(len(bands[b])> 0):
          gdal.BuildVRT(SRC_FOLDER + 'sentinel2_' + b + '.vrt', bands[b], options=vrt_options)
    
if __name__== "__main__":
  main(sys.argv)