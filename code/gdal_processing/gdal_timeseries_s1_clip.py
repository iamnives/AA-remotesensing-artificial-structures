import gdal
import numpy
from tqdm import tqdm
import os
import sys, getopt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# inicialize data location
DATA_FOLDER = "../sensing_data/raw/timeseries/"


def main(argv):
    roi = None

    try:
      opts, args = getopt.getopt(argv,"r:", ["roi="])
    except getopt.GetoptError:
      print('usage: gdal_timeseries_s1_clip.py -r <roifolder>')
      sys.exit(2)

    for opt, arg in opts:
      if opt == '-r':
          roi = arg + "/"

    if roi is None:
        print('usage: gdal_timeseries_s1_clip.py -r <roifolder>')
        sys.exit(2)

    mask = "../vector_data/" + roi + "ROI.shp"
    src = DATA_FOLDER + roi + "s1-corrected/"
    dst_folder = "../sensing_data/" + "clipped/" + roi + "ts1/"

    src_dss = [src + f for f in os.listdir(src) if ".dim" not in f]
    src_dss.sort()
    
    for idx, f in enumerate(tqdm(src_dss)):
        for f1 in os.listdir(f):
            if ".img" in f1:
                gdal.Warp(dst_folder + str(idx) + 'clipped_' + f1, f + "/" + f1, dstSRS="EPSG:32629",
                          resampleAlg="near", format="GTiff", xRes=10, yRes=10, cutlineDSName=mask, cropToCutline=1)


if __name__ == "__main__":
    main(sys.argv[1:])
