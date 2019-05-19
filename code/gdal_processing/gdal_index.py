import gdal
import numpy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import osr

#inicialize data location
DATA_FOLDER = "../sensing_data/"
REF_FOLDER = DATA_FOLDER + "warped/"

INDEXES =	{
  "NDVI": ('"(A-B)/(A+B)"', "B08", "B04"),
  "EVI": ('"2,5 * (A-B)/((A+6*B-7,5*C) + 1)"', "B08", "B04", "B02"),
  "NDWI": ('"(A-B)/(A+B)"', "B03", "B08"),
  "NDBI": ('"(A-B)/(A+B)"', "B11", "B08"),
}
GDAL_PROCESS = "C:/OSGeo4W64/bin/gdal_calc.py"

def main(argv):
    ref_dss = [REF_FOLDER + f for f in os.listdir(REF_FOLDER)]
    files = ref_dss
    files.sort()

    for index, value in INDEXES.items():
        output = REF_FOLDER + "clipped_" + index + ".tif"
        bands=[]

        #find band files
        for fname in files:
            # Apply file type filter   
            if fname.endswith(".tif"):
                for band in value[1:]:
                    if "_"+band in fname:
                        bands.append(fname)
                        break
        
        # Call process, place non normalized difference indexes in special cases.
        if index != "NDWI":
            bands.reverse()
        if index == "EVI":
            gdal_calc_str = f'python {GDAL_PROCESS} -A {bands[0]} -B {bands[1]} -C {bands[2]} --outfile={output} --calc={value[0]}'
        else:
            gdal_calc_str = f'python {GDAL_PROCESS} -A {bands[0]} -B {bands[1]} --outfile={output} --calc={value[0]}'
        os.system(gdal_calc_str)
  
if __name__== "__main__":
  main(sys.argv)