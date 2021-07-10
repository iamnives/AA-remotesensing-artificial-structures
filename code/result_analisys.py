"""
Created on Sun Mar  3 21:42:16 2019

@author: André Neves
"""
import sys
import gdal
from utils import visualization as viz
from utils import data
import numpy as np
from numpy import interp
from itertools import cycle
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_auc_score,  roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "bigsquare/"

SRC_FOLDER = DATA_FOLDER + "results/" + ROI
SRC = SRC_FOLDER

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
    return ax

def scl_map(x_elem):
    """Maps one element from x to the given class on our standard

    Parameters:
        x_elem (int): class value from the scl
    Returns:
        int: class number in the classification format
   """
    if x_elem == 4:
        return 2
    if x_elem == 5:
        return 1
    if x_elem == 6:
        return 3
    return 4  # anomalies or noclass

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

def ghsl_map(x_elem):
    """Maps one element from GHSL to the given class on our standart ranged from 1 to 4

    Parameters:
        x_elem (int): class value from the classification
    Returns:
        int: class number in the SCL format
   """
    if x_elem == 1:
        return 3
    if x_elem == 2:
        return 2
    return 1  # else its structure

def calc_roc(y_test, y_score, n_classes=5):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    tholds = dict() 
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], tholds[i] = roc_curve((y_test==i), y_score[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area

    #fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])    
    return fpr, tpr, roc_auc, tholds

def plot_roc(fpr, tpr, roc_auc, n_classes=5, optimal_points=None, tholds=None):
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    #plt.plot(fpr["micro"], tpr["micro"],
    #        label='micro-average ROC curve (area = {0:0.2f})'
    #            ''.format(roc_auc["micro"]),
    #        color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

        if optimal_points is not None:
            plt.plot(fpr[i][optimal_points[i]], tpr[i][optimal_points[i]], color='red', marker="*")
            print('class: ', i ,tholds[i][optimal_points[i]])
            plt.text(fpr[i][optimal_points[i]], tpr[i][optimal_points[i]], tholds[i][optimal_points[i]], fontsize=9)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def main(argv):
    """Runs main code for result analysis

    Parameters:
        argv (array): console arguments if given
    Returns:
        None
   """
    result_10m = gdal.Open(SRC + "recall almost 1, precision 0.2.tif", gdal.GA_ReadOnly)
    result_10m = result_10m.GetRasterBand(1).ReadAsArray()

    train_zone = gdal.Open(SRC + "small_square.tif", gdal.GA_ReadOnly)
    train_zone = train_zone.GetRasterBand(1).ReadAsArray()

    cos = gdal.Open(COS_SRC + "cos_indi_ground_binary_big.tif", gdal.GA_ReadOnly)
    cos = cos.GetRasterBand(1).ReadAsArray()

    train_zone = train_zone.ravel()
    result_10m = result_10m.ravel()
    cos = cos.ravel()

    #valid_idx = np.where((train_zone==0) & (result_10m !=0))
    valid_idx = np.where((result_10m !=0))

    cos = cos-1
    cos = cos[valid_idx]
    cos_og = np.copy(cos)

    result_10m = result_10m-1
    result_10m = result_10m[valid_idx]

    #cos[np.where((cos==0) | (cos==1) | (cos==2) | (cos==4))] = 1
    #cos[cos==3] = 0
    #result_10m[np.where((result_10m==0) | (result_10m==1) | (result_10m==2) | (result_10m==4))] = 1
    #result_10m[result_10m==3] = 0

    kappa = cohen_kappa_score(cos, result_10m)
    print(f'Kappa: {kappa}')
    print(classification_report(cos, result_10m))
    
    #classes_cos = ["HIGH DENSITY",  "RURAL", "NON RESIDENTIAL", "NON BUILT UP", "INDIVIDUAL"]
    classes_cos = [0, 1, 2, 3, 4]
    plot_confusion_matrix(result_10m, cos, classes=classes_cos,
                          normalize=True, title="SentinelSCL-COS Individual 10m Normalized confusion matrix")

    # Caluate rocs and probas
    class_1 = gdal.Open(SRC + "boosted_newfmz_test_classificationestrutura.tiff", gdal.GA_ReadOnly)
    class_1 = class_1.GetRasterBand(1).ReadAsArray().ravel()[valid_idx]

    #class_2 = gdal.Open(SRC + "boosted_newfmz_test_classificationrural.tiff", gdal.GA_ReadOnly)
    #class_2 = class_2.GetRasterBand(1).ReadAsArray().ravel()[valid_idx]

    #class_3 = gdal.Open(SRC + "boosted_newfmz_test_classificationoutras.tiff", gdal.GA_ReadOnly)
    #class_3 = class_3.GetRasterBand(1).ReadAsArray().ravel()[valid_idx]

    #class_4 = gdal.Open(SRC + "boosted_newfmz_test_classificationnatural.tiff", gdal.GA_ReadOnly)
    #class_4 = class_4.GetRasterBand(1).ReadAsArray().ravel()[valid_idx]

    #class_5 = gdal.Open(SRC + "boosted_newfmz_test_classificationindi.tiff", gdal.GA_ReadOnly)
    #class_5 = class_5.GetRasterBand(1).ReadAsArray().ravel()[valid_idx]

    #y_score = np.array([class_1, class_2, class_3, class_4, class_5])
    fpr, tpr, roc_auc, tholds = calc_roc(cos_og, [class_1], n_classes=1)

    roc_optimal = np.array([0,1])
    optimal_points = np.empty(5, dtype=int)

    for idx in range(0,1):
        points = np.stack((fpr[idx], tpr[idx]), axis=-1)
        
        distances = np.array([np.linalg.norm(a-roc_optimal) for a in points])

        optimal_points[idx] = np.argmin(distances)

    plot_roc(fpr, tpr, roc_auc, n_classes=1, optimal_points=optimal_points, tholds=tholds)

if __name__ == "__main__":
    main(sys.argv)
