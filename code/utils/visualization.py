import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sn
import gdal

#inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"

DS_FOLDER = DATA_FOLDER + "clipped/" + ROI
OUT_RASTER = DATA_FOLDER + "results/" + ROI + "classification.tiff"

def createGeotiff(outRaster, data, ref):
        
    labelDS = gdal.Open(ref, gdal.GA_ReadOnly)
    geo_transform = labelDS.GetGeoTransform()
    projection = labelDS.GetProjection()
    # Create a GeoTIFF file with the given data
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    rasterDS = driver.Create(outRaster, cols, rows, 1, gdal.GDT_Byte)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    band = rasterDS.GetRasterBand(1)
    band.WriteArray(data)
    dataset = None

#PARAM: results = clf.cv_results_ , scorng = metric objects
def plot_gridcv(results, scoring, param, limL, limH):
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluation " + param,
            fontsize=16)

    plt.xlabel(param)
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(limL, limH)
    ax.set_ylim(0.3, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_' + param].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k', 'b', 'c', 'y', 'r', 'm']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()

def plot_confusionmx(matrix):
    # plot confusion matrix
    df_cm = pd.DataFrame(matrix, index = [i for i in range(0,matrix.shape[0])],
                    columns = [i for i in  range(0,matrix.shape[0])])
    plt.figure(figsize = (13,13))
    sn.heatmap(df_cm, annot=True)
    plt.show()