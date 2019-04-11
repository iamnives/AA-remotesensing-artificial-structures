import gdal
import numpy
import os
import sys
import osr

#inicialize data location
DATA_FOLDER = "./sensing_data/"
REF_FOLDER = DATA_FOLDER + "reference/"
DST_FOLDER = DATA_FOLDER + "indexes/"
SRC_FOLDER = DATA_FOLDER + "warped/"

INDEXES =	{
  "NDVI": ('"A-B/A+B"', "B08", "B04"),
  "EVI": ('"(A-B)/(A+6*B7.5*C+1)"', "B08", "B04", "B02"),
  "NDWI": ('"A-B/A+B"', "B03", "B08"),
  "NDBI": ('"A-B/A+B"', "B11", "B08"),
  "NDBIv2": ('"A-B/A+B"', "B12", "B08"),
}
GDAL_PROCESS = "gdal_calc.py"

def main(argv):
    ref_dss = [REF_FOLDER + f for f in os.listdir(REF_FOLDER)]
    src_dss = [SRC_FOLDER + f for f in os.listdir(SRC_FOLDER)]
    files = ref_dss + src_dss

    for index, value in INDEXES.items():
        output = index + ".tiff"
        bands=[]

        #find band files
        for fname in files:
            # Apply file type filter   
            if fname.endswith(".tiff"):
                for band in value[1:]:
                    if "_"+band+"_" in fname:
                        bands.insert(fname)
                        break

        # Call process, place non normalized difference indexes in special cases.
        if index == "EVI":
            gdal_calc_str = f'{GDAL_PROCESS} -A {bands[0]} -B {bands[1]} -C {bands[2]} --outfile={output} --calc={value[0]}'
        else:
            gdal_calc_str = f'{GDAL_PROCESS} -A {bands[0]} -B {bands[1]} --outfile={output} --calc={value[0]}'
            os.system(gdal_calc_str)
  
if __name__== "__main__":
  main(sys.argv)