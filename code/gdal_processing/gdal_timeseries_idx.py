import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gdal
import numpy as np
from utils.visualization import createGeotiff

from tqdm import tqdm

#inicialize data location
ROI = "vila-de-rei/"
SRC_FOLDER = "../sensing_data/" + "clipped/" + ROI + "ts/"
DST_FOLDER =  "../sensing_data/" + "clipped/" + ROI + "ts/"


def ndvi(nir, red, ref):
	return (nir - red) / (nir + red)

def ndbi(swir, nir, ref):
	return (swir - nir) / (swir + nir)

def ndwi(green, nir, ref):
	return (green - nir) / (green + nir)

def evi(nir, red, ref, blue=None):
	return 2.4*(nir - red) / (nir + 2.4*red + 10_000)

# createGeotiff(DST_FOLDER + name, data, ref, gdal.GDT_Float32)

def getBand(f):
	refDs = gdal.Open(SRC_FOLDER + f, gdal.GA_ReadOnly)
	return refDs.GetRasterBand(1).ReadAsArray().astype(np.float)

def main(argv):
	src_dss = [f for f in os.listdir(SRC_FOLDER) if ".tif" in f]
	src_dss.sort()
	sets = [f.split("clipped")[0] for f in src_dss]
	sets = np.unique(sets)

	for f_id in tqdm(sets):
		data = [f for f in os.listdir(SRC_FOLDER) if f.split("clipped")[0] == f_id]
		
		ref = SRC_FOLDER + data[7]

		nir = getBand(data[7])
		green = getBand(data[2])
		red = getBand(data[3])
		swir = getBand(data[10])

		id1 = ndvi(nir, red, ref)
		id2 = ndwi(green, nir, ref)
		id3 = ndbi(swir, nir, ref)
		id4 = evi(nir, red, ref)
		
		id1[~np.isfinite(id1)] = 0
		id2[~np.isfinite(id2)] = 0
		id3[~np.isfinite(id3)] = 0
		id4[~np.isfinite(id4)] = 0

		createGeotiff(SRC_FOLDER + f_id + "clipped_pad_pad_ndvi.tif" , id1 ,ref ,gdal.GDT_Float32)
		createGeotiff(SRC_FOLDER + f_id + "clipped_pad_pad_ndwi.tif" , id2 ,ref ,gdal.GDT_Float32)
		createGeotiff(SRC_FOLDER + f_id + "clipped_pad_pad_ndbi.tif" , id3 ,ref ,gdal.GDT_Float32)
		createGeotiff(SRC_FOLDER + f_id + "clipped_pad_pad_evi.tif" , id4 ,ref ,gdal.GDT_Float32)


		
if __name__== "__main__":
  main(sys.argv)