
import gdal
import numpy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import osr

#inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "lisboa-setubal/"
SRC_FOLDER = DATA_FOLDER + "datasets/" + ROI + "s1/"

bands = {
        'Gamma0_VH': [], 
        'Gamma0_VH_ASM': [], 
        'Gamma0_VH_Contrast': [], 
        'Gamma0_VH_Dissimilarity': [], 
        'Gamma0_VH_Energy': [], 
        'Gamma0_VH_Entropy': [], 
        'Gamma0_VH_GLCMCorrelation': [], 
        'Gamma0_VH_GLCMMean': [], 
        'Gamma0_VH_GLCMVariance': [], 
        'Gamma0_VH_Homogeneity': [], 
        'Gamma0_VH_MAX': [], 
        'Gamma0_VV': [], 
        'Gamma0_VV_ASM': [], 
        'Gamma0_VV_Contrast': [], 
        'Gamma0_VV_Dissimilarity': [], 
        'Gamma0_VV_Energy': [], 
        'Gamma0_VV_Entropy': [], 
        'Gamma0_VV_GLCMCorrelation': [], 
        'Gamma0_VV_GLCMMean': [], 
        'Gamma0_VV_GLCMVariance': [], 
        'Gamma0_VV_Homogeneity': [], 
        'Gamma0_VV_MAX': []
         }

def main(argv):
    # Reference files
    src_dss = [f for f in os.listdir(SRC_FOLDER) if ".img" in f]

    for f in src_dss:
        bands[f.split(".")[0]].append(SRC_FOLDER + f)

    vrt_options = gdal.BuildVRTOptions(resolution='lowest')
    for b in bands:
        if(len(bands[b])> 0):
          gdal.BuildVRT(SRC_FOLDER + 'sentinel1_' + b + '.vrt', bands[b], options=vrt_options)
    
if __name__== "__main__":
  main(sys.argv)

