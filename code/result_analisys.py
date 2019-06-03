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
import matplotlib
import matplotlib.pyplot as plt

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

SRC_FOLDER = DATA_FOLDER + "results/" + ROI
SRC = SRC_FOLDER + "timeseries/"

GT_SRC = SRC_FOLDER + "GT/"
COS_SRC = DATA_FOLDER + "clipped/" + ROI


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(title+'.pdf')
    return ax


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


def reverse_scl_map(x_elem):
    """Maps one element from x to the given class on SCL standart

    Parameters:
        x_elem (int): class value from the classification
    Returns:
        int: class number in the SCL format
   """
    if x_elem == 2:
        return 1
    if x_elem == 3:
        return 2
    if x_elem == 4:
        return 3
    return x_elem  # no mapping needed


def scl_gt_map(x_elem):
    """Maps one element from SCL to the given class on SCL ranged from 1 to 4

    Parameters:
        x_elem (int): class value from the classification
    Returns:
        int: class number in the SCL format
   """
    if x_elem == 3:
        return 2
    if x_elem == 4:
        return 3
    if x_elem == 5:
        return 4
    return x_elem  # else no mapping needed


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

    print("Mapping cos...")
    cos = np.array([data._class_map(yi)
                    for yi in tqdm(cos.flatten())]).reshape((1937, 2501))
    cos_20 = np.array([data._class_map(yi)
                       for yi in tqdm(cos_20.flatten())]).reshape((1937, 2501))

    print("Shapes: ")
    print(result_10m.shape, result_20m.shape, gt.shape,
          gt_20m.shape, cos.shape, cos_20.shape)

    print("Mapping scl...")
    gt = np.array([scl_map(yi)
                   for yi in tqdm(gt.flatten())]).reshape((1937, 2501))
    gt_20m = np.array([scl_map(yi)
                       for yi in tqdm(gt_20m.flatten())]).reshape((1937, 2501))

    classes = ["NON_VEGETATED", "VEGETATION", "WATER", "SCL Anomaly"]
    classes_cos = ["NON_VEGETATED", "ROAD", "VEGETATION", "WATER"]
    classes_scl = ["NON_VEGETATED", "ROAD", "VEGETATION", "WATER", "SCL Anomaly"]

    plot_confusion_matrix(cos.flatten(), gt.flatten(), classes=classes_scl,
                          normalize=True, title="SentinelSCL-COS 10m Normalized confusion matrix")
    plot_confusion_matrix(cos_20.flatten(), gt_20m.flatten(), classes=classes_scl,
                          normalize=True, title="SentinelSCL-COS 20m Normalized confusion matrix")

    plot_confusion_matrix(cos.flatten(), result_10m.flatten(
    ), classes=classes_cos, normalize=True, title="XGBoost-COS 10m Normalized confusion matrix")
    plot_confusion_matrix(cos.flatten(), result_20m.flatten(
    ), classes=classes_cos, normalize=True, title="XGBoost-COS 20m Normalized confusion matrix")

    print("Mapping results...")
    result_10_mapped = np.array(
        [reverse_scl_map(yi) for yi in tqdm(result_10m.flatten())]).reshape((1937, 2501))

    result_20_mapped = np.array(
        [reverse_scl_map(yi) for yi in tqdm(result_20m.flatten())]).reshape((1937, 2501))

    print("Re-mapping scl...")
    gt_mapped = np.array(
        [scl_gt_map(yi) for yi in tqdm(gt.flatten())]).reshape((1937, 2501))

    gt_20_mapped = np.array(
        [scl_gt_map(yi) for yi in tqdm(gt_20m.flatten())]).reshape((1937, 2501))

    plot_confusion_matrix(gt_mapped.flatten(), result_10_mapped.flatten(
    ), classes=classes, normalize=True, title="XGBoost-SentinelSCL 10m Normalized confusion matrix")
    plot_confusion_matrix(gt_20_mapped.flatten(), result_20_mapped.flatten(
    ), classes=classes, normalize=True, title="XGBoost-SentinelSCL 20m Normalized confusion matrix")


if __name__ == "__main__":
    main(sys.argv)
