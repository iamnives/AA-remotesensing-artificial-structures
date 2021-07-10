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
ROI = "vila-de-rei/"
SRC = DATA_FOLDER + "clipped/" + ROI
SRC_FOLDER = SRC + "ts1/"

DST_FOLDER = DATA_FOLDER + "clipped/" + ROI + "t1stats_decis/"

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
    ref_dss = SRC_FOLDER + src_dss[0]  # Get the first one

    for vh, vv in pairwise(src_dss):
        raster_ds = gdal.Open(SRC_FOLDER + vv, gdal.GA_ReadOnly)
        vv_data = raster_ds.GetRasterBand(1).ReadAsArray()
        raster_ds = gdal.Open(SRC_FOLDER + vh, gdal.GA_ReadOnly)
        vh_data = raster_ds.GetRasterBand(1).ReadAsArray()
        vv_vh_data = np.divide(vv_data, vh_data)
        viz.createGeotiff(SRC_FOLDER + vv.split(".")[0] +"VH.tif", vv_vh_data,
                          ref_dss, gdal.GDT_Float32)

    src_dss = [f for f in os.listdir(SRC_FOLDER) if (
    ".jp2" in f) or (".tif" in f) or (".img" in f)]
    src_dss.sort()

    # 3clipped_Gamma0_VH.img
    for f in src_dss:
        try:
            key = f.split("_")[2].split(".")[0]
            bands[key].append(SRC_FOLDER + f)
        except KeyError:
            print("ignoring", f)
    
    refDs = gdal.Open("../sensing_data/clipped/" + ROI +
                      "ignored/static/clipped_sentinel2_B08.vrt", gdal.GA_ReadOnly)
    band = refDs.GetRasterBand(1).ReadAsArray()
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
        nq_tiles = 10

        q0 = np.quantile(timeseries, 0.00, axis=0)  # minimum

        n_tiles = [q0]

        increment = 1/nq_tiles
        q_tile = 0.00
        for n in range(nq_tiles-1):
            q_tile += increment
            q = np.quantile(timeseries, q_tile, axis=0)
            n_tiles.append(q)

        std = np.std(timeseries, axis=0)  # standard dev
        variance = np.sqrt(std)  # variance

        d_type = gdal.GDT_Float32

        viz.createGeotiff(DST_FOLDER + b + "_mean.tiff", mean_ts,
                          ref_dss, d_type)
        viz.createGeotiff(DST_FOLDER + b + "_var.tiff", variance,
                          ref_dss, d_type)
        viz.createGeotiff(DST_FOLDER + b + "_q0.tiff", q0,
                          ref_dss, d_type)

        for idx, q in enumerate(n_tiles):
            viz.createGeotiff(DST_FOLDER + b + f"_q{idx}.tiff", q,
                          ref_dss, d_type)
if __name__ == "__main__":
    main(sys.argv)
