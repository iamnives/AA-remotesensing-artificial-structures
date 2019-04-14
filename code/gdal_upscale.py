import gdal
import numpy
import os
import sys
import osr

#inicialize data location
DATA_FOLDER = "../sensing_data/"
REF_FOLDER = DATA_FOLDER + "reference/"
DST_FOLDER = DATA_FOLDER + "warped/"
SRC_FOLDER = DATA_FOLDER + "clipped/"

# Function to read the original file's projection:
def GetGeoInfo(FileName):
    src_ds = gdal.Open(FileName, gdal.GA_ReadOnly)
    nDV = src_ds.GetRasterBand(1).GetNoDataValue()

    # Get reference pixel size
    gt = src_ds.GetGeoTransform()
    xsize = gt[1]
    ysize =-gt[5]

    geoT = src_ds.GetGeoTransform()
    # Get reference dataset EPSG
    src_proj = osr.SpatialReference(wkt=src_ds.GetProjection())
    projection = src_proj.GetAttrValue('AUTHORITY',1)

    dataType = src_ds.GetRasterBand(1).DataType
    dataType = gdal.GetDataTypeName(dataType)
    return nDV, xsize, ysize, geoT, projection, dataType 

def warpRasters(src, nDV, xRes, yRes, projection, dataType, totalFiles):
    for idx, fName in enumerate(src, start=1):

        name = fName.split("/")
        name = name[-1]

        print(str(idx) + "/" + str(totalFiles) + " warping " + fName + "...")
        src_ds = gdal.Open(fName, gdal.GA_ReadOnly)

        gdal.Warp(
            DST_FOLDER + "warped_" + projection + "_" + str(xRes) + "x" + str(yRes) + "_" +  name, 
            src_ds, 
            dstSRS='EPSG:' + projection,
            xRes=xRes, yRes=yRes)


def main(argv):
    # Reference files
    ref_dss = [REF_FOLDER + f for f in os.listdir(REF_FOLDER)]
    # Warping files
    src_dss = [SRC_FOLDER + f for f in os.listdir(SRC_FOLDER)]

    # Get reference info
    nDV, xsize, ysize, geoT, projection, _ = GetGeoInfo(ref_dss[0])
    toWrapN = len(src_dss)
    print("Warping " + str(toWrapN) + " files...")

    #Warp everything!
    warpRasters(
        src=src_dss, 
        nDV = nDV, 
        xRes=xsize, 
        yRes=ysize, 
        projection=projection, 
        dataType= gdal.GDT_UInt16,
        totalFiles=toWrapN)
    print("Wraping finished, dst files can be found at: " + DST_FOLDER)
  
if __name__== "__main__":
  main(sys.argv)