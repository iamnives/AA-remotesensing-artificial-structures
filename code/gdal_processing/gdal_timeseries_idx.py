import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
from utils.visualization import createGeotiff
import numpy as np
import gdal

# inicialize data location
ROI = "vila-de-rei/"
SRC_FOLDER = "../sensing_data/" + "clipped/" + ROI + "ts/"
DST_FOLDER = "../sensing_data/" + "clipped/" + ROI + "ts/"


def ndti_calc(swir1, swir2):
    return (swir1 - swir2)/(swir1 + swir2)

def ndvire_calc(rededge1, red):
    return (rededge1 - red)/(rededge1 + red)

def savi_calc(nir, red):
    return (nir - red)/((nir + red + 0.5) * 0.5)

def mndwi_calc(green, swir1):
    return (green - swir1)/(green + swir1)

def ndvi_calc(nir, red):
    """
    Normalized
    difference
    vegetation index
    """
    return (nir - red) / (nir + red + 1)

def ndbi_calc(swir, nir):
    """
    Normalized
    difference
    built-up index
    """
    return (swir - nir) / (swir + nir + 1)

def ndwi_calc(green, nir):
    """
    Normalized
    difference
    water index
    """
    return (green - nir) / (green + nir + 1)

def evi_calc(nir, red, blue=None):
    """
    """
    return 2.4*(nir - red) / (nir + 2.4*red + 10_000)

def bui_calc(ndbi, ndvi):
    """
    Buil-up index
    """
    return ndbi-ndvi

def baei_calc(red, green, swir, L=0.3):
    """
    Built-up area
    extraction
    index
    """
    return (red + L)/(green + swir)

def nbi_calci(red, nir, swir):
    """
    new build-up index
    """
    return (swir * red)/nir

def vibi_calc(ndvi, ndbi):
    """
    Vegetation
    index built-up
    index
    """
    return ndvi/(ndvi + ndbi)

def ibi_calc(ndbi, savi, mndwi):
    """
    Index-based
    built-up index 
    """
    return (ndbi - ((savi+mndwi)/2)) / (ndbi + ((savi+mndwi)/2))

def ui_calc(swir, nir):
    """
    Urban index
    wrong formula in paper
    """
    return ( ((swir - nir)/(swir - nir)) + 1.0 )*100

def bsi_calc(swir, red, nir, blue):
    """
    Bare soil index
    """
    return ((swir + red)-(nir + blue))/((swir + red)+(nir + blue))

def getBand(f):
    refDs = gdal.Open(SRC_FOLDER + f, gdal.GA_ReadOnly)
    return refDs.GetRasterBand(1).ReadAsArray().astype(np.float)


def main(argv):
    src_dss = [f for f in os.listdir(SRC_FOLDER) if ".tif" in f]
    src_dss.sort()
    sets = [f.split("clipped")[0] for f in src_dss]
    sets = np.unique(sets)

    for f_id in tqdm(sets):
        data = [f for f in os.listdir(
            SRC_FOLDER) if f.split("clipped")[0] == f_id]

        ref = SRC_FOLDER + data[7]

        nir = getBand(data[7])
        green = getBand(data[2])
        red = getBand(data[3])
        blue = getBand(data[1])
        swir1 = getBand(data[10])
        swir2 = getBand(data[11])
        rededge1 = getBand(data[4])

        ndvi = ndvi_calc(nir, red)
        ndwi = ndwi_calc(green, nir)
        ndbi = ndbi_calc(swir1, nir)
        evi = evi_calc(nir, red)
        savi = savi_calc(nir, red)
        mndwi = mndwi_calc(green, swir1) 
        bsi = bsi_calc(swir1, red, nir, blue)
        ui = ui_calc(swir1, nir)
        idi = ibi_calc(ndbi, savi, mndwi)
        vibi = vibi_calc(ndvi, ndbi)
        nbi = nbi_calci(red, nir, swir1)
        baei = baei_calc(red, green, swir1, L=0.3)
        bui = bui_calc(ndbi, ndvi)
        ndti = ndti_calc(swir1, swir2)
        ndvire = ndvire_calc(rededge1, red)
        
        idxs = [(ndvi, 'ndvi'),(ndwi,'ndwi'), (ndbi,'ndbi'),(evi,'evi'),(savi,'savi'),(mndwi,'mndwi'),(bsi,'bsi'),(ui,'ui'),(idi,'idi'),(vibi,'vibi'),(nbi,'nbi'),(baei,'baei'),(bui,'bui'),(ndti,'ndti'),(ndvire,'ndvire')]

        for idx, name in idxs:
            idx[~np.isfinite(idx)] = 0

            createGeotiff(SRC_FOLDER + f_id +
                        f"clipped_pad_pad_{name}.tif", idx, ref, gdal.GDT_Float32)


if __name__ == "__main__":
    main(sys.argv)
