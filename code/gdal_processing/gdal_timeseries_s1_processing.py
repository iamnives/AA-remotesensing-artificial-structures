import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import visualization as viz
import matplotlib
import numpy as np
import gdal


# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "arbitrary/"
SRC = DATA_FOLDER + "clipped/" + ROI
SRC_FOLDER = SRC + "ts1/"

DST_FOLDER = DATA_FOLDER + "clipped/" + ROI + "/t1stats/"

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def main(argv):
    bands = {
        "VV": [],
        "VH": [],
        "VVVH": []
    }

    src_dss = [f for f in os.listdir(SRC_FOLDER) if (
        ".jp2" in f) or (".tif" in f) or (".img" in f)]
    src_dss.sort()

    for vh, vv in pairwise(src_dss):
        raster_ds = gdal.Open(SRC_FOLDER + vv, gdal.GA_ReadOnly)
        vv_data = raster_ds.GetRasterBand(1).ReadAsArray()
        raster_ds = gdal.Open(SRC_FOLDER + vh, gdal.GA_ReadOnly)
        vh_data = raster_ds.GetRasterBand(1).ReadAsArray()
        vv_vh_data = np.divide(vv_data, vh_data)
        viz.createGeotiff(SRC_FOLDER + vv.split(".")[0] +"VH.img", vv_vh_data,
                          "../sensing_data/clipped/arbitrary/ignored/static/clipped_sentinel2_B08.vrt", gdal.GDT_Float32)

    src_dss = [f for f in os.listdir(SRC_FOLDER) if (
    ".jp2" in f) or (".tif" in f) or (".img" in f)]
    src_dss.sort()

    for f in src_dss:
        try:
            bands[f.split("_")[2].split(".")[0]].append(SRC_FOLDER + f)
        except KeyError:
            print("ignoring")

    ref_ds = gdal.Open(
        "../sensing_data/clipped/arbitrary/ignored/static/clipped_sentinel2_B08.vrt", gdal.GA_ReadOnly)
    band = ref_ds.GetRasterBand(1).ReadAsArray()
    ref_shape = band.shape

    for b in tqdm(bands):
        # change to np array (0,m) when possible timeseries.append([bandsData], axis=0), or faster (n,m) -> a[0..n] = [1,2,...]
        timeseries = []
        if(len(bands[b]) > 0):
            # Open raster dataset
            for raster in bands[b]:
                raster_ds = gdal.Open(raster, gdal.GA_ReadOnly)
                # Extract band's data and transform into a numpy array
                bands_data = raster_ds.GetRasterBand(1).ReadAsArray()
                # static fix for clip mismatch problem
                timeseries.append(bands_data[:ref_shape[0], :ref_shape[1]])

        timeseries = np.array(timeseries)
        timeseries[~np.isfinite(timeseries)] = 0

        # Using quartiles, change to 0.05 quantiles later if load isn't too much...
        mean_ts = np.mean(timeseries, axis=0)  # mean
        q0 = np.quantile(timeseries, 0.00, axis=0)  # minimum
        q1 = np.quantile(timeseries, 0.25, axis=0)  # first quantile
        q2 = np.quantile(timeseries, 0.50, axis=0)  # median
        q3 = np.quantile(timeseries, 0.75, axis=0)  # third quantile
        q4 = np.quantile(timeseries, 1.0, axis=0)  # maximum
        std = np.std(timeseries, axis=0)  # standard dev
        variance = np.sqrt(std)  # variance

        viz.createGeotiff(DST_FOLDER + b + "_mean.tiff", mean_ts,
                          "../sensing_data/clipped/arbitrary/ignored/static/clipped_sentinel2_B08.vrt", gdal.GDT_Float32)
        viz.createGeotiff(DST_FOLDER + b + "_q0.tiff", q0,
                          "../sensing_data/clipped/arbitrary/ignored/static/clipped_sentinel2_B08.vrt", gdal.GDT_Float32)
        viz.createGeotiff(DST_FOLDER + b + "_q1.tiff", q1,
                          "../sensing_data/clipped/arbitrary/ignored/static/clipped_sentinel2_B08.vrt", gdal.GDT_Float32)
        viz.createGeotiff(DST_FOLDER + b + "_q2.tiff", q2,
                          "../sensing_data/clipped/arbitrary/ignored/static/clipped_sentinel2_B08.vrt", gdal.GDT_Float32)
        viz.createGeotiff(DST_FOLDER + b + "_q3.tiff", q3,
                          "../sensing_data/clipped/arbitrary/ignored/static/clipped_sentinel2_B08.vrt", gdal.GDT_Float32)
        viz.createGeotiff(DST_FOLDER + b + "_q4.tiff", q4,
                          "../sensing_data/clipped/arbitrary/ignored/static/clipped_sentinel2_B08.vrt", gdal.GDT_Float32)
        viz.createGeotiff(DST_FOLDER + b + "_var.tiff", variance,
                          "../sensing_data/clipped/arbitrary/ignored/static/clipped_sentinel2_B08.vrt", gdal.GDT_Float32)

if __name__ == "__main__":
    main(sys.argv)
