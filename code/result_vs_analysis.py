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

FCG_SRC= SRC_FOLDER + "fgc/"

GT_SRC = SRC_FOLDER + "GT/"
COS_SRC = DATA_FOLDER + "clipped/" + ROI

OUT_RASTER = DATA_FOLDER + "results/" + ROI + \
    "timeseries/xgb/GT_group1_classification.tiff"
OUT_RASTER_2 = DATA_FOLDER + "results/" + ROI + \
    "timeseries/xgb/GT_group2_classification.tiff"

REF_FILE = DATA_FOLDER + "clipped/" + ROI + \
    "ignored/static/clipped_sentinel2_B08.vrt"

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
    # classes = classes[unique_labels(y_true, y_pred)]
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

def split_map(x):
    if x >= 1 and x <= 3:
        return 1
    return x-2

def main(argv):
    """Runs main code for result analysis

    Parameters:
        argv (array): console arguments if given
    Returns:
        None
   """

#    # open rasters
#     fgc_pred = gdal.Open(
#         FCG_SRC + "rasterized_generated.tif", gdal.GA_ReadOnly)

#     fgc_true = gdal.Open(
#         FCG_SRC + "rasterized_gt.tif", gdal.GA_ReadOnly)
#     # get result data
#     fgc_pred = fgc_pred.GetRasterBand(1).ReadAsArray()
#     fgc_true = fgc_true.GetRasterBand(1).ReadAsArray()

#     fgc_true = fgc_true[:fgc_pred.shape[0], :fgc_pred.shape[1]]

#     print("generated vs true")
#     kappa = cohen_kappa_score(fgc_true.flatten(), fgc_pred.flatten())
#     print(f'Kappa: {kappa}')
#     print(classification_report(fgc_true.flatten(), fgc_pred.flatten()))
#     print(confusion_matrix(fgc_true.flatten(), fgc_pred.flatten()))

    # open rasters
    result_10m_boosted_split = gdal.Open(
        SRC + "boosted_20px_ts_s1_s2_dem_idx_split_classification.tiff", gdal.GA_ReadOnly)

    result_10m_boosted = gdal.Open(
        SRC + "boosted_20px_ts_s1_s2_dem_idx_classification.tiff", gdal.GA_ReadOnly)

    # get result data
    result_10m_boosted = result_10m_boosted.GetRasterBand(1).ReadAsArray()
    result_10m_boosted_split = result_10m_boosted_split.GetRasterBand(1).ReadAsArray()

    # open and map groung truth
    cos = gdal.Open(COS_SRC + "clipped_cos_50982.tif", gdal.GA_ReadOnly)
    cos = cos.GetRasterBand(1).ReadAsArray()
    cos = cos[:result_10m_boosted_split.shape[0], :result_10m_boosted_split.shape[1]]

    print("Mapping cos...")
    cos_g1 = np.array([data._class_map(yi)
                    for yi in tqdm(cos.flatten())])
    cos_g2 = np.array([data._class_split_map(yi)
                        for yi in tqdm(cos.flatten())])


    viz.createGeotiff(OUT_RASTER, cos_g1.reshape(cos.shape),
    REF_FILE, gdal.GDT_Byte)

    viz.createGeotiff(OUT_RASTER_2, cos_g2.reshape(cos.shape),
    REF_FILE, gdal.GDT_Byte)

    return 1

    print("Mapping split...") 
    result_10m_boosted_split_mapped = np.array([split_map(yi)
                    for yi in tqdm(result_10m_boosted_split.flatten())])

 
    classes_cos = ["ESTRUTURAS", "NATURAL", "ÁGUA"]
 
    print("cos vs split")
    kappa = cohen_kappa_score(cos, result_10m_boosted_split_mapped)
    print(f'Kappa: {kappa}')
    print(classification_report(cos, result_10m_boosted_split_mapped))
    print(confusion_matrix(cos, result_10m_boosted_split_mapped))

    print("cos vs boosted")
    kappa = cohen_kappa_score(cos, result_10m_boosted.flatten())
    print(f'Kappa: {kappa}')
    print(classification_report(cos, result_10m_boosted.flatten()))
    print(confusion_matrix(cos, result_10m_boosted.flatten()))

    print("boosted vs split")
    kappa = cohen_kappa_score(result_10m_boosted.flatten(), result_10m_boosted_split_mapped)
    print(f'Kappa: {kappa}')
    print(classification_report(result_10m_boosted.flatten(), result_10m_boosted_split_mapped))
    print(confusion_matrix(result_10m_boosted.flatten(), result_10m_boosted_split_mapped))


if __name__ == "__main__":
    main(sys.argv)
