#! /usr/bin/env python
############################################################################
#  classaccuracymetrics.py
#
#  Copyright 2020 RSGISLib.
#
#  RSGISLib: 'The remote sensing and GIS Software Library'
#
#  RSGISLib is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  RSGISLib is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with RSGISLib.  If not, see <http://www.gnu.org/licenses/>.
#
#
# Purpose:  Provide a set of functions to calculate the accuracy of a
#           classification.
#
# Author: Pete Bunting
# Email: petebunting@mac.com
# Date: 03/02/2020
# Version: 1.0
#
# History:
# Version 1.0 - Created.
#
###########################################################################

import numpy


def cls_quantity_accuracy(y_true, y_pred, cls_area):
    """
    A function to calculate quantity allocation & disagreement for a
    land cover classification. The labels must be integers from 1 - N, 
    where N is the number of classes.

    :param y_true: A list or 1D numpy array of true labels.
    :param y_pred: A list or 1D numpy array of predicted labels.
    :param cls_area: A dict or 1D numpy array of area/n_pixels identified by the
                     classifier. len(cls_area) == numpy.unique(y_true).
    
    :return: dict with 'Quantity Disagreement (Q)', 
                       'Allocation Disagreement (A)', 
                       'Proportion Correct (C)', 
                       'Total Disagreement (D)'.

    Reference: Pontius, R. G., Jr, & Millones, M. (2011). Death to Kappa: birth
    of quantity disagreement and allocation disagreement for accuracy assessment. 
    International Journal of Remote Sensing, 32(15), 4407–4429.
    
    """
    from sklearn.metrics import confusion_matrix
    
    # check inputs:
    if not isinstance(y_true, numpy.ndarray):
        y_true = numpy.array(y_true)
    if not isinstance(y_pred, numpy.ndarray):
        y_pred = numpy.array(y_pred)
    if not isinstance(cls_area, numpy.ndarray):
        cls_area = numpy.array(cls_area)

    for arr in [y_true, y_pred, cls_area]:
        if arr.ndim != 1:
            raise SystemExit('Error: All input arrays must be one dimensional.')

    if numpy.unique(y_true).size != cls_area.size:
        raise SystemExit('Error: Number of classes != Number of classes in area.')

    # create confusion matrix:
    cm = confusion_matrix(y_true, y_pred)

    # convert absolute areas into proportional areas:
    prop_area = (cls_area / cls_area.sum()).reshape(-1, 1)  # same as Comparison Total (see Ref.)

    # normalise the confusion matrix by proportional area:
    norm_cm = cm.astype(float) / cm.sum(axis=1)[:,].reshape(-1, 1)
    norm_cm = norm_cm * prop_area
    comp_total = norm_cm.sum(axis=1)  # same as proportional area
    ref_total = norm_cm.sum(axis=0)

    quantity_disagreement = sum(numpy.abs(ref_total - comp_total)) / 2
    commission = [(row.sum() - row[idx]) for idx, row in enumerate(norm_cm)]
    ommission = ref_total - numpy.diag(norm_cm)
    allocation_disagreement = sum(2 * numpy.min(numpy.array([commission, ommission]), axis=0)) / 2
    prop_correct = sum(numpy.diag(norm_cm)) / numpy.sum(norm_cm)
    disagreement = quantity_disagreement + allocation_disagreement

    Ex = sum(ref_total * comp_total)
    Rx = 1 - Ex
    jey = 2

    pij = (norm_cm/sum(norm_cm) * (comp_total/sum(comp_total)))

    out_dict = dict()
    out_dict['Quantity Disagreement (Q)'] = quantity_disagreement
    out_dict['Allocation Disagreement (A)'] = allocation_disagreement
    out_dict['Proportion Correct (C)'] = prop_correct
    out_dict['Total Disagreement (D)'] = disagreement
    
    out_dict['Kappa Standard'] = (Rx - disagreement)/Rx
    out_dict['Kappa Allocation'] = (Rx - disagreement)/(Rx - quantity_disagreement)
    out_dict['Kappa Histo'] = (Rx - quantity_disagreement)/(Rx)
    out_dict['Kappa No'] = ((1 - (1/jey))-disagreement)/(1 - (1/jey))

    Y = sum(sum(pij**2)) + out_dict['Kappa Allocation'] * (1 - sum(sum(pij**2)))

    Z = (1/jey) + out_dict['Kappa Allocation'] * (sum(numpy.minimum(numpy.full(sum(pij).shape, 1/jey), sum(pij))) - (1/jey))

    out_dict['Kappa Quantity'] = (prop_correct - Z)/(Y - Z)

    return out_dict


def calc_class_accuracy_metrics(ref_samples, pred_samples, cls_area=None, cls_names=None):
    """
    A function which calculates a set of classification accuracy metrics for a set
    of reference and predicted samples. Optionally, the area classified for each
    class can be provided allowing further metrics to be calculated.
    
    :param ref_samples: a 1d array of reference samples represented by a numeric class id
    :param pred_samples: a 1d array of predicted samples represented by a numeric class id
    :param cls_area: a 1d array with the area of each class classified (i.e., pixel count)
    :param cls_names: a 1d list of the class names (labels) in the order of the class ids.
    
    """
    import sklearn.metrics
    
    acc_metrics = sklearn.metrics.classification_report(ref_samples, pred_samples, target_names=cls_names, output_dict=True)
    cohen_kappa = sklearn.metrics.cohen_kappa_score(ref_samples, pred_samples)
    acc_metrics['cohen_kappa'] = cohen_kappa
    
    cm = sklearn.metrics.confusion_matrix(ref_samples, pred_samples)
    user_accuracy = [(row[idx] / row.sum()) * 100 for idx, row in enumerate(cm)]
    producer_accuracy = [(col[idx] / col.sum()) * 100 for idx, col in enumerate(cm.T)]
    
    # convert absolute areas into proportional areas:
    prop_area = (cls_area / cls_area.sum()).reshape(-1, 1)  # same as Comparison Total (see Ref.)
    # normalise the confusion matrix by proportional area:
    norm_cm = cm.astype(float) / cm.sum(axis=1)[:,].reshape(-1, 1)
    norm_cm = norm_cm * prop_area
    comp_total = norm_cm.sum(axis=1)  # same as proportional area
    ref_total = norm_cm.sum(axis=0)
    commission = [(row.sum() - row[idx]) for idx, row in enumerate(norm_cm)]
    ommission = ref_total - numpy.diag(norm_cm)
    # Sum the normalised cm columns to estimate the proportion of scene for each class.
    cls_area_prop = numpy.sum(norm_cm, axis=0)
        
    acc_metrics['confusion_matrix'] = cm.tolist()
    acc_metrics['user_accuracy'] = user_accuracy
    acc_metrics['producer_accuracy'] = producer_accuracy
    
    acc_metrics['norm_confusion_matrix'] = norm_cm.tolist()
    acc_metrics['commission'] = commission
    acc_metrics['ommission'] = ommission.tolist()
    acc_metrics['est_prop_cls_area'] = cls_area_prop.tolist()
    
    if cls_area is not None:
        quantity_metrics = cls_quantity_accuracy(ref_samples, pred_samples, cls_area)
        acc_metrics['quantity_metrics'] = quantity_metrics
    
    return acc_metrics        
    

def calc_acc_metrics_vecsamples(in_vec_file, in_vec_lyr, ref_col, cls_col, cls_img, img_cls_name_col='ClassName', img_hist_col='Histogram', out_json_file=None, out_csv_file=None):
    """
    A function which calculates classification accuracy metrics using a set of 
    reference samples in a vector file. 
    This would be often be used alongside the ClassAccuracy QGIS plugin.
    
    :param in_vec_file: the input vector file with the reference points
    :param in_vec_lyr: the input vector layer name with the reference points.
    :param ref_col: the name of the reference classification column in the input vector file.
    :param cls_col: the name of the classification column in the input vector file.
    :param cls_img: an image of the classification from which the area 
                    (pixel counts) of each class are extracted to normalise the
                    confusion matrix. Should have a RAT with class names and histogram.
    :param img_cls_name_col: The name of the column in the image attribute table which specifies the 
                             class name.
    :param img_hist_col: The name of the column in the image attribute table which contains the 
                         histogram (i.e., number of pixels within the class).
    :param out_json_file: if specified the generated metrics and confusion matrix are written to 
                          a JSON file (Default=None).
    :param out_csv_file: if specified the generated metrics and confusion matrix are written to 
                         a CSV file (Default=None).
    
    """
    import rsgislib
    import rsgislib.vectorutils
    import rsgislib.rastergis
    import rsgislib.rastergis.ratutils
    
    # Read columns from vector file.
    ref_vals = numpy.array(rsgislib.vectorutils.readVecColumn(in_vec_file, in_vec_lyr, ref_col))
    cls_vals = numpy.array(rsgislib.vectorutils.readVecColumn(in_vec_file, in_vec_lyr, cls_col))
    
    # Find unique class values
    unq_cls_names = numpy.unique(numpy.concatenate((numpy.unique(ref_vals), numpy.unique(cls_vals))))    
    
    # Create LUTs assigning each class a unique int ID. 
    cls_name_lut = dict()
    cls_id_lut = dict()
    for cls_id, cls_name in enumerate(unq_cls_names):
        cls_name_lut[cls_name] = cls_id
        cls_id_lut[cls_id] = cls_name
    
    # Create cls_id arrays
    ref_int_vals = numpy.zeros_like(ref_vals, dtype=int)
    cls_int_vals = numpy.zeros_like(cls_vals, dtype=int)
    for cls_name in unq_cls_names:
        ref_int_vals[ref_vals == cls_name] = cls_name_lut[cls_name]
        cls_int_vals[cls_vals == cls_name] = cls_name_lut[cls_name]
    
    try:
        rat_cols = rsgislib.rastergis.getRATColumns(cls_img)
    except:
        raise Exception("The input image does not have a RAT...")
        
    if img_cls_name_col not in rat_cols:
        raise Exception("The RAT does not contain the class name column specified ('{}')".format(img_cls_name_col))
    if img_hist_col not in rat_cols:
        raise Exception("The RAT does not contain the histogram column specified ('{}')".format(img_hist_col))
    
    img_hist_data = rsgislib.rastergis.ratutils.getColumnData(cls_img, img_hist_col)
    img_clsname_data = rsgislib.rastergis.ratutils.getColumnData(cls_img, img_cls_name_col)
    img_clsname_data[0] = ''
    
    rsgis_utils = rsgislib.RSGISPyUtils()
    pxl_size_x, pxl_size_y = rsgis_utils.getImageRes(cls_img)
    pxl_area = pxl_size_x * pxl_size_y
    
    # Find the class areas (pixel counts)
    cls_pxl_count_dict = dict()
    cls_area_dict = dict()
    tot_area = 0.0
    cls_pxl_counts = numpy.zeros_like(unq_cls_names, dtype=int)
    for i, cls_name in enumerate(img_clsname_data):
        cls_name = str(cls_name.decode())
        if (i > 0) and (cls_name !=''):
            if cls_name not in unq_cls_names:
                raise Exception("Class ('{}') found in image which was not in point samples...".format(cls_name))
            cls_pxl_counts[cls_name_lut[cls_name]] = img_hist_data[i]
            cls_pxl_count_dict[cls_name] = img_hist_data[i]
            cls_area_dict[cls_name] = img_hist_data[i] * pxl_area
            tot_area = tot_area + (img_hist_data[i] * pxl_area)
    
    acc_metrics = calc_class_accuracy_metrics(ref_int_vals, cls_int_vals, cls_pxl_counts, unq_cls_names)
    
    acc_metrics['pixel_count'] = cls_pxl_count_dict
    acc_metrics['pixel_area'] = cls_area_dict
    
    if out_json_file is not None:
        import json
        with open(out_json_file, 'w') as out_json_file_obj:
            json.dump(acc_metrics, out_json_file_obj, sort_keys=True,indent=4, separators=(',', ': '), ensure_ascii=False)
    
    if out_csv_file is not None:
        import csv
        with open(out_csv_file, mode='w') as out_csv_file_obj:
            acc_metrics_writer = csv.writer(out_csv_file_obj, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            # Overall Accuracy
            acc_metrics_writer.writerow(['overall accuracy', acc_metrics['accuracy']])
            acc_metrics_writer.writerow(['cohen kappa', acc_metrics['cohen_kappa']])
            acc_metrics_writer.writerow([''])
            
            # Quantity Metrics
            acc_metrics_writer.writerow(['Allocation Disagreement (A)', acc_metrics['quantity_metrics']['Allocation Disagreement (A)']])
            acc_metrics_writer.writerow(['Quantity Disagreement (Q)', acc_metrics['quantity_metrics']['Quantity Disagreement (Q)']])
            acc_metrics_writer.writerow(['Proportion Correct (C)', acc_metrics['quantity_metrics']['Proportion Correct (C)']])
            acc_metrics_writer.writerow(['Total Disagreement (D)', acc_metrics['quantity_metrics']['Total Disagreement (D)']])
            acc_metrics_writer.writerow([''])
            
            # Individual Class Scores
            acc_metrics_writer.writerow(['class', 'f1-score', 'precision', 'recall', 'support'])
            for cls_name in unq_cls_names:
                acc_metrics_writer.writerow([cls_name, acc_metrics[cls_name]['f1-score'], acc_metrics[cls_name]['precision'], acc_metrics[cls_name]['recall'], acc_metrics[cls_name]['support']])
            # Overall macro and weighted
            acc_metrics_writer.writerow([''])
            acc_metrics_writer.writerow(['macro avg', acc_metrics['macro avg']['f1-score'], acc_metrics['macro avg']['precision'], acc_metrics['macro avg']['recall'], acc_metrics['macro avg']['support']])
            acc_metrics_writer.writerow(['weighted avg', acc_metrics['weighted avg']['f1-score'], acc_metrics['weighted avg']['precision'], acc_metrics['weighted avg']['recall'], acc_metrics['weighted avg']['support']])
            acc_metrics_writer.writerow([''])
            
            # Output the confusion matrix
            acc_metrics_writer.writerow(['Point Count Confusion Matrix'])
            cm_top_row = ['']
            for cls_name in unq_cls_names:
                cm_top_row.append(cls_name)
            cm_top_row.append('User Acc')
            acc_metrics_writer.writerow(cm_top_row)
            for cls_name, cm_row, user_acc in zip(unq_cls_names, acc_metrics['confusion_matrix'], acc_metrics['user_accuracy']):
                row = [cls_name]
                for val in cm_row:
                    row.append(val)
                row.append(user_acc)
                acc_metrics_writer.writerow(row)
            cm_bot_row = ['Producer']
            for prod_val in acc_metrics['producer_accuracy']:
                cm_bot_row.append(prod_val)
            acc_metrics_writer.writerow(cm_bot_row)
            acc_metrics_writer.writerow([''])
            
            acc_metrics_writer.writerow(['Normalised Confusion Matrix'])
            # Output the normalised confusion matrix
            cm_top_row = ['']
            for cls_name in unq_cls_names:
                cm_top_row.append(cls_name)
            acc_metrics_writer.writerow(cm_top_row)
            for cls_name, cm_row in zip(unq_cls_names, acc_metrics['norm_confusion_matrix']):
                row = [cls_name]
                for val in cm_row:
                    row.append(val)
                acc_metrics_writer.writerow(row)
            
            acc_metrics_writer.writerow([''])
            acc_metrics_writer.writerow(['class', 'commission', 'ommision'])
            for i, cls_name in enumerate(unq_cls_names):
                acc_metrics_writer.writerow([cls_name, acc_metrics['commission'][i], acc_metrics['ommission'][i]])
                       
            acc_metrics_writer.writerow([''])
            acc_metrics_writer.writerow(['class', 'pixel count', 'pixel area', 'Est. Prop. Area', 'Est. Area'])
            for i, cls_name in enumerate(unq_cls_names):
                acc_metrics_writer.writerow([cls_name, cls_pxl_count_dict[cls_name], cls_area_dict[cls_name], acc_metrics['est_prop_cls_area'][i], (tot_area * acc_metrics['est_prop_cls_area'][i])])
    
    if (out_json_file is None) and (out_csv_file is None):
        import pprint
        pprint.pprint(acc_metrics)


