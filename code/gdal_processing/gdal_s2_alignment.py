import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gdal
from utils.visualization import createGeotiff

DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"
SRC = DATA_FOLDER + "clipped/" + ROI
SRC_FOLDER = SRC + "ts/"

DST_FOLDER = DATA_FOLDER + "clipped/" + ROI + "/tsalg/"
imFilename = DST_FOLDER + "18clipped_pad_pad_ndvi.tif"

rast_src = gdal.Open(imFilename, gdal.GA_Update)
gt = rast_src.GetGeoTransform()
gtl = list(gt)
gtl[0] -= 0 # moves X axis - left + right
gtl[3] += 5  # moves Y axis - down + up
rast_src.SetGeoTransform(tuple(gtl))
rast_src = None # -0 +5