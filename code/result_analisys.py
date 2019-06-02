"""
Created on Sun Mar  3 21:42:16 2019

@author: André Neves
"""
import sys
import gdal
from utils import visualization as viz
from utils import data
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from pandas_ml import ConfusionMatrix

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

SRC_FOLDER = DATA_FOLDER + "results/" + ROI
SRC = SRC_FOLDER + "timeseries/"

GT_SRC = SRC_FOLDER + "GT/"
COS_SRC = DATA_FOLDER + "clipped/" + ROI


def scl_map(x_elem):
    """Maps one element from x to the given class on our standard

    Parameters:
        x_elem (int): class value from the scl
    Returns:
        int: class number in the classification format
   """
    if x_elem == 4:
        return 3
    if x_elem == 5:
        return 1
    if x_elem == 6:
        return 4
    return 5  # anomalies or noclass


def main(argv):
    """Runs main code for result analysis

    Parameters:
        argv (array): console arguments if given
    Returns:
        None
   """
    result_10m = gdal.Open(
        SRC + "boosted_20px_ts_s1_s2_idxfixed_roads_truealign_classification.tiff", gdal.GA_ReadOnly)
    result_10m = result_10m.GetRasterBand(1).ReadAsArray()

    result_20m = gdal.Open(
        SRC + "boosted_20px_ts_s1_s2_idxfixed_roads_truealign_20m_classification.tiff", gdal.GA_ReadOnly)
    result_20m = result_20m.GetRasterBand(1).ReadAsArray()

    gt = gdal.Open(GT_SRC + "T29SND_20190525T112121_SCL.tif", gdal.GA_ReadOnly)
    gt = gt.GetRasterBand(1).ReadAsArray()
    gt = gt[:result_10m.shape[0], :result_10m.shape[1]]

    gt_20m = gdal.Open(
        GT_SRC + "T29SND_20190525T112121_SCL_20m.tif", gdal.GA_ReadOnly)
    gt_20m = gt_20m.GetRasterBand(1).ReadAsArray()

    cos = gdal.Open(COS_SRC + "clipped_cos_50982.tif", gdal.GA_ReadOnly)
    cos = cos.GetRasterBand(1).ReadAsArray()
    cos = cos[:result_10m.shape[0], :result_10m.shape[1]]

    roads = gdal.Open(COS_SRC + "roads_cos_50982.tif", gdal.GA_ReadOnly)
    roads = roads.GetRasterBand(1).ReadAsArray()
    roads = roads[:result_10m.shape[0], :result_10m.shape[1]]
    cos[roads == 4] = roads[roads == 4]
 
    cos_20 = gdal.Open(COS_SRC + "clipped_20_cos_50982.tif", gdal.GA_ReadOnly)
    cos_20 = cos_20.GetRasterBand(1).ReadAsArray()
    cos_20 = cos_20[:result_10m.shape[0], :result_10m.shape[1]]

    roads = gdal.Open(COS_SRC + "roads_20_cos_50982.tif", gdal.GA_ReadOnly)
    roads = roads.GetRasterBand(1).ReadAsArray()
    roads = roads[:cos_20.shape[0], :cos_20.shape[1]]
    cos_20[roads == 4] = roads[roads == 4]

    cos = np.array([data._class_map(yi)
                    for yi in tqdm(cos.flatten())]).reshape((1937, 2501))
    cos_20 = np.array([data._class_map(yi)
                       for yi in tqdm(cos_20.flatten())]).reshape((1937, 2501))

    print("Shapes: ")
    print(result_10m.shape, result_20m.shape, gt.shape,
          gt_20m.shape, cos.shape, cos_20.shape)

    print("10m result matrix")
    print(confusion_matrix(result_10m.flatten(), cos.flatten()))
    print("20m result matrix:")
    print(confusion_matrix(result_20m.flatten(), cos_20.flatten()))

    gt = np.array([scl_map(yi)
                    for yi in tqdm(gt.flatten())]).reshape((1937, 2501))
    gt_20m = np.array([scl_map(yi)
                        for yi in tqdm(gt_20m.flatten())]).reshape((1937, 2501))

    print("10m scl matrix")
    print(confusion_matrix(gt.flatten(), cos.flatten()))
    print("20m scl matrix:")
    print(confusion_matrix(gt_20m.flatten(), cos_20.flatten()))

    cm = ConfusionMatrix(result_10m.flatten(), cos.flatten())
    print(cm)

if __name__ == "__main__":
    main(sys.argv)
