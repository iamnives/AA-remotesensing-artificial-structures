import numpy as np
import gdal
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import visualization as viz

SRC = 'D:/AA-remotesensing-artificial-structures/sensing_data/clipped/bigsquare/'
IMAGE_2 = 'individual.tif'
IMAGE_1 = 'cos_binary.tif'

def main(argv):
    #cos
    raster_ds_1 = gdal.Open(SRC + IMAGE_1, gdal.GA_ReadOnly)
    raster_band_1 = raster_ds_1.GetRasterBand(1).ReadAsArray()
    #indi
    raster_ds_2 = gdal.Open(SRC + IMAGE_2, gdal.GA_ReadOnly)
    raster_band_2 = raster_ds_2.GetRasterBand(1).ReadAsArray()

    raster_band_1[np.where((raster_band_1 == 1) & (raster_band_2 == 5))] = 2

    viz.createGeotiff(SRC + "cos_indi_binary.tiff",
                      raster_band_1, SRC + IMAGE_1, gdal.GDT_Byte)


if __name__ == "__main__":
    main(sys.argv)
