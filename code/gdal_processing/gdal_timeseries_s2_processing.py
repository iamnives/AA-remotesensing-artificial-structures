import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import visualization as viz
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import gdal

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"
SRC = DATA_FOLDER + "clipped/" + ROI
SRC_FOLDER = SRC + "ts/"

DST_FOLDER = DATA_FOLDER + "clipped/" + ROI + "tstats/"


def main(argv):
    bands = {
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
        "ndvi": [],
        "evi": [],
        "ndbi": [],
        "savi": [],
        "mndwi": [],
        "ndwi": [],
        "bsi": [],
        "ui": [],
        "idi": [],
        "vibi": [],
        "nbi": [],
        "baei": [],
        "bui": [],
        "ndvire": [],
        "ndti" : [],
    }



    src_dss = [f for f in os.listdir(SRC_FOLDER) if (
        ".jp2" in f) or (".tif" in f) or (".img" in f)]
    src_dss.sort()

    # Reference files
    for f in src_dss:
        try:
            bands[f.split("_")[3].split(".")[0]].append(SRC_FOLDER + f)
        except KeyError:
            print("ignoring: ", f)

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
                rasterDS = gdal.Open(raster, gdal.GA_ReadOnly)
                # Extract band's data and transform into a numpy array
                bandsData = rasterDS.GetRasterBand(1).ReadAsArray()
                # static fix for clip mismatch problem
                timeseries.append(bandsData[:ref_shape[0], :ref_shape[1]])

        timeseries = np.array(timeseries)
        timeseries[~np.isfinite(timeseries)] = 0

        # Using quartiles, change to 0.05 quantiles later if load isn't too much...
        mean_ts = np.mean(timeseries, axis=0)  # mean
        nq_tiles = 5

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
                          "../sensing_data/clipped/vila-de-rei/ignored/static/clipped_sentinel2_B08.vrt", d_type)
        viz.createGeotiff(DST_FOLDER + b + "_var.tiff", variance,
                          "../sensing_data/clipped/vila-de-rei/ignored/static/clipped_sentinel2_B08.vrt", d_type)
        viz.createGeotiff(DST_FOLDER + b + "_q0.tiff", q0,
                          "../sensing_data/clipped/vila-de-rei/ignored/static/clipped_sentinel2_B08.vrt", d_type)

        for idx, q in enumerate(n_tiles):
            viz.createGeotiff(DST_FOLDER + b + f"_q{idx}.tiff", q,
                          "../sensing_data/clipped/vila-de-rei/ignored/static/clipped_sentinel2_B08.vrt", d_type)
        


# np.mean(a, axis=0), np.quantile(a, 0.25, axis=0),
if __name__ == "__main__":
    main(sys.argv)
