import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from sklearn.metrics import classification_report

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from scipy.stats import uniform
from sklearn.utils import class_weight

from utils import visualization as viz
from utils import data
from utils import metrics

import gdal

from datetime import timedelta
import time

import matplotlib
import matplotlib.pyplot as plt 

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
import numpy as np

#inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"
DS_FOLDER = DATA_FOLDER + "clipped/" + ROI

OUT_RASTER = DATA_FOLDER + "results/" + ROI + "/timeseries/neural_20px_ts_s1_s2_idx_roads_clean_classification.tiff"
REF_FILE = DATA_FOLDER + "clipped/" + ROI  + "/ignored/static/clipped_sentinel2_B03.vrt"

# Tensorflow trash
def model(dfs):
  start = time.time()
  train_size = int(19386625*0.2)
  X_train, y_train, X_test , y_test = data.load(train_size, normalize=True, balance=False)


  input_shape = X_train.shape[1]
  logits = 4

  y_train = y_train - 1
  y_test = y_test - 1

  class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

  y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=logits)

  dnn = Sequential()
  # Define DNN structure
  dnn.add(Dense(32, input_dim=input_shape, activation='relu'))
  dnn.add(Dense(64, input_dim=input_shape, activation='relu'))
  dnn.add(Dropout(0.4))
  dnn.add(Dense(units=logits, activation='softmax'))

  dnn.compile(
      loss='categorical_crossentropy',
      optimizer='Nadam',
      metrics=['accuracy']
      )
  dnn.summary()

  dnn.fit(X_train, y_train_onehot,
            epochs=10, validation_split = 0.2, class_weight=class_weights)

  y_pred_onehot = dnn.predict(X_test)
  y_pred = [np.argmax(pred) for pred in y_pred_onehot]

  kappa = cohen_kappa_score(y_test, y_pred)
  print(f'Kappa: {kappa}')
  print(classification_report(y_test, y_pred))

  # Testing trash
  X, y, shape = data.load_prediction(ratio=0.5 , normalize=True)
  print(X.shape, y.shape)

  y_pred = dnn.predict(X)
  y_pred = [np.argmax(pred) for pred in y_pred]

  kappa = cohen_kappa_score(y-1, y_pred)
  print(f'Kappa: {kappa}')
  print(classification_report(y-1, y_pred))

  y_pred = np.array(y_pred)
  yr = y_pred.reshape(shape)

  viz.createGeotiff(OUT_RASTER, yr, REF_FILE, gdal.GDT_Byte)

   # serialize model to YAML
  model_yaml = dnn.to_yaml()
  with open("../sensing_data/models/dnn_tf.yaml", "w") as yaml_file:
      yaml_file.write(model_yaml)
  # serialize weights to HDF5
  dnn.save_weights("../sensing_data/models/dnn_tf.h5")
  print("Saved model to disk")

  end=time.time()
  elapsed=end-start
  print("Run time: " + str(timedelta(seconds=elapsed)))

def main(argv):
  model(None)

if __name__== "__main__":
  main(sys.argv)