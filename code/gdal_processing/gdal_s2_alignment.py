# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:42:16 2019

@author: Ricardo
"""

from __future__ import print_function
import cv2
from osgeo import gdal, osr
import numpy as np
from timeit import default_timer as timer
import os

#inicialize data location
DATA_FOLDER = "../sensing_data/raw/timeseries/"
ROI = "vila-de-rei/"
MASK = "../vector_data/" + ROI + "ROI.shp" 


SRC = DATA_FOLDER + ROI + "/s2/"
DST_FOLDER = "../sensing_data/" + "clipped/" + ROI + "ts/"

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.5
gdal.UseExceptions()


def getWarpMatrix(newFilename,refFilename):
  im1 = cv2.imread(newFilename, cv2.IMREAD_COLOR)  
  im2= cv2.imread(refFilename, cv2.IMREAD_COLOR)
  
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
  
  for i, match in enumerate(matches):
   points1[i, :] = keypoints1[match.queryIdx].pt
   points2[i, :] = keypoints2[match.trainIdx].pt  
  
  warp_matrix = np.eye(2, 3, dtype=np.float32)
  (warp_matrix, mask) = cv2.estimateAffine2D(points2, points1, warp_matrix, maxIters=10000, confidence=0.99)

  return warp_matrix



def align_image(refFile, imFile, warp_matrix , out):

#  height, width, channels = im2.shape
#  im1Reg = cv2.warpPerspective(im1, h, (width, height))
  ref = gdal.Open(refFile)
  img = gdal.Open(imFile)  
  
  #warp_matrix = np.eye(2, 3, dtype=np.float32)
  print(warp_matrix)
  
  #fica a zero pois não é necessário rotações na imagem
  warp_matrix[0,1] = 0.00
  warp_matrix[1,0] = 0.00
  warp_matrix[0,0] = 1.00
  warp_matrix[1,1] = 1.00
  
  #deslocamento a ser feito nas coordenadas da imagem
  dx = -warp_matrix[0,2]*10.0
  dy = warp_matrix[1,2]*10.0

  driver = gdal.GetDriverByName("GTiff")
  dst_ds = driver.CreateCopy(out, img, strict=0)
  
 
  geo = ref.GetGeoTransform()
  new_geo = ([geo[0]+dx, geo[1], geo[2],geo[3]+dy,geo[4],geo[5]])
  dst_ds.SetGeoTransform(new_geo)
  dst_ds.SetProjection(ref.GetProjection())

  dst_ds = None
     

def align_all_images(folder):
    
    refImage_10m_path = DATA_FOLDER + ""
    
    #Obtem a matriz de transformação baseada na imagem tci
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            
            file_path = subdir+'/'+file
            
            if "TCI_10m" in file and (file.endswith(".tif") or file.endswith(".jp2")):
                print("A calcular a matriz: ", file )
                warp_matrix = getWarpMatrix(file_path, refImage_10m_path)  

    #aplica a matriz de transformação a todas as outras imagens 
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            
            file_path = subdir+'/'+file
            
            if file.endswith(".tif") or file.endswith(".jp2"):
                align_image(refImage_10m_path, file_path, warp_matrix, subdir+"/aligned_"+file)
              


def align_all_products(folder): 
    for subdir, dirs, files in os.walk(folder):
        for d in dirs:
            if ".SAFE" in d:
                curr_dir = subdir+"/"+d
                print("Alignig product: "+curr_dir)
                align_all_images(curr_dir)
        
        
    
if __name__ == '__main__':
   
  # Read reference image
  start = timer()
  align_all_products("../../temp")
  duration = timer()-start
  print("Duration: ",duration)

  
  
  
  
  
