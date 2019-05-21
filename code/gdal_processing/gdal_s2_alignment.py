import cv2
import gdal
import numpy as np

#inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"
SRC = DATA_FOLDER + "clipped/" + ROI
SRC_FOLDER = SRC +  "tstats/"
 
DST_FOLDER = DATA_FOLDER + "clipped/" + ROI + "/tsalg/"

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
 
 
def alignImages(im1, im2):
 
  # Convert images to grayscale
  im1Gray =im1
  im2Gray = im2
   
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
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
   
  return im1Reg, h
 
 
if __name__ == '__main__':
   
  # Read reference image
  refFilename = SRC_FOLDER + "B01_mean.tiff"
  print("Reading reference image : ", refFilename)
  refDs = gdal.Open(refFilename, gdal.GA_ReadOnly)
  imReference = refDs.GetRasterBand(1).ReadAsArray().astype(np.float)

  for f in files:
    # add for each image in dir
    # Read image to be aligned
    imFilename = f
    print("Reading image to align : ", imFilename);  
    refDs = gdal.Open(imFilename, gdal.GA_ReadOnly)
    im = refDs.GetRasterBand(1).ReadAsArray().astype(np.float)
    
    print("Aligning images ...")
    # Registered image will be resotred in imReg. 
    # The estimated homography will be stored in h. 
    imReg, h = alignImages(im, imReference)
    
    # Write aligned image to disk. 
    outFilename = DST_FOLDER + f
    print("Saving aligned image : ", outFilename); 
    cv2.imwrite(outFilename, imReg)
    
    # Print estimated homography
    print("Estimated homography : \n",  h)