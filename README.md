<div align="center">
  <img src="https://raw.githubusercontent.com/AMNeves/AA-remotesensing-artificial-structures/master/images/logo.png">
</div>

# Geospatial machine learning framework (Work in progress)
Master's thesis development code

## Introduction


## Requirements

gdal fixed by: conda create -n mynewenv -c conda-forge gdal rasterio


## Data


aria2c --http-user=amneves --http-passwd=amnandre12 --check-certificate=false --max-concurrent-downloads=2 -M products.meta4

Labels extracted with: gdal_rasterize -l COS2015_ModComb -a ID -ts 10980.0 10980.0 -a_nodata 0.0 -te -119191.40909999982 -300404.8039999973 162129.08110000088 276083.76740000054 -ot Float32 -of GTiff "E:/Master Folder/COSmodcomb/COS2015_ModComb.shp" C:/Users/amnev/Desktop/AA-remotesensing-artificial-structures/sensing_data/labels/cos.tif