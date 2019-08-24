from tqdm import tqdm
import subprocess
import osr
import gdal
import numpy
import os
import sys


# inicialize data location
DATA_FOLDER = "../vector_data/"
ROI = "fundao/"

DST_FILE = DATA_FOLDER + ROI + "consolidated.shp"


def main(argv):

    estructures = [f for f in os.listdir(
        DATA_FOLDER + ROI + "edificios") if "shp" in f]
    hidro = [f for f in os.listdir(
        DATA_FOLDER + ROI + "hidrografia") if "shp" in f]

    to_merge = estructures + hidro

    subprocess.call(["ogr2ogr", "-f", "ESRI Shapefile", DST_FILE, to_merge[0]])

    # for shape in tqdm(to_merge[1:]):
    #     subprocess.call(["ogr2ogr", "-f", "ESRI Shapefile",
    #                     "-update",  "-append", DST_FILE, shape])


if __name__ == "__main__":
    main(sys.argv)
