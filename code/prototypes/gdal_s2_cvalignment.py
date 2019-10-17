import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
import numpy as np
import gdal
import cv2
from utils.visualization import createGeotiff


# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"
SRC = DATA_FOLDER + "clipped/" + ROI
SRC_FOLDER = SRC + "ts/"

DST_FOLDER = DATA_FOLDER + "clipped/" + ROI + "/tsalg/"

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.20


def align_images(im1, im2):

    # Convert images to grayscale
    im1_gray = im1
    im2_grey = im2

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_grey, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    # Draw top matches
    imMatches = cv2.drawMatches(
        im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2)
    return h


if __name__ == '__main__':

    # Read reference image
    refFilename = SRC_FOLDER + "0clipped_T29SND_20160430T112122_B08.tif"
    labelDS = gdal.Open(refFilename, gdal.GA_ReadOnly)
    imReference = labelDS.GetRasterBand(1).ReadAsArray()
    imReference = cv2.imread(refFilename, 0)

    # Read to align image
    imFilename = SRC_FOLDER + "6clipped_T29SND_20160629T112112_B08.tif"
    labelDS = gdal.Open(imFilename, gdal.GA_ReadOnly)
    im = labelDS.GetRasterBand(1).ReadAsArray()
    im = cv2.imread(imFilename, 0)

    # The estimated homography will be stored in h.
    h = align_images(im, imReference)

    # Write aligned image to disk.
    outFilename = DST_FOLDER + "test.tif"
    width = im.shape[0]
    height = im.shape[1]

    imReg = cv2.warpPerspective(im, h, (width, height))
    print(imReg.shape)

    createGeotiff(outFilename, im, refFilename, gdal.GDT_UInt16)
 