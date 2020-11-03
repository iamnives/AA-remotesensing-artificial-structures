import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sklearn import preprocessing
from tqdm import tqdm
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.callbacks import EarlyStopping
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
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras import Sequential


# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"
DS_FOLDER = DATA_FOLDER + "clipped/" + ROI

OUT_RASTER = DATA_FOLDER + "results/" + ROI + \
    "/timeseries/1dcnn/neural_20px_tsfull_group2_classification.tiff"
REF_FILE = DATA_FOLDER + "clipped/" + ROI + \
    "/ignored/static/clipped_sentinel2_B08.vrt"


def model(dfs):
    start = time.time()
    train_size = int(19386625*0.2)

    split_struct = True
    osm_roads = False

    X_train, y_train, X_test, y_test, _, _, normalizer = data.load(
        train_size, normalize=True, osm_roads=osm_roads, split_struct=split_struct)

   
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    input_shape = X_train.shape[1]
    logits = 5

    y_train = y_train - 1
    y_test = y_test - 1

    #class_weights = class_weight.compute_class_weight('balanced',
     #                                                 np.unique(y_train),
        #                                              y_train)

    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=logits)

    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train_onehot.shape[1]

    model_cnn = Sequential()
    model_cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model_cnn.add(MaxPooling1D(pool_size=2))

    model_cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model_cnn.add(MaxPooling1D(pool_size=2))
    
    model_cnn.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(128, activation='relu'))
    model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(64, activation='relu'))
    model_cnn.add(Dense(n_outputs, activation='softmax'))

    model_cnn.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['mae', 'acc']
    )
    model_cnn.summary()

    es = EarlyStopping(monitor='val_loss', min_delta=0.0001,
                       patience=5, verbose=0, mode='auto')

    model_cnn.fit(X_train, y_train_onehot,
            epochs=100, validation_split=0.2, callbacks=[es])

    yt_pred_onehot = model_cnn.predict(X_train)
    yt_pred = [np.argmax(pred) for pred in yt_pred_onehot]

    kappa = cohen_kappa_score(y_train, yt_pred)
    print(f'Train Kappa: {kappa}')
    print(classification_report(y_train, yt_pred))

    y_pred_onehot = model_cnn.predict(X_test)
    y_pred = [np.argmax(pred) for pred in y_pred_onehot]

    kappa = cohen_kappa_score(y_test, y_pred)
    print(f'Validation Kappa: {kappa}')
    print(classification_report(y_test, y_pred))
    
    # Testing trash
    X, y, shape = data.load_prediction(
        ratio=1, normalize=normalizer, osm_roads=osm_roads, split_struct=split_struct, army_gt=False)
    print(X.shape, y.shape)

    y_pred = model_cnn.predict(X)
    y_pred = [np.argmax(pred) for pred in y_pred]

    kappa = cohen_kappa_score(y-1, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y-1, y_pred))

    y_pred = np.array(y_pred)
    yr = y_pred.reshape(shape)

    viz.createGeotiff(OUT_RASTER, yr, REF_FILE, gdal.GDT_Byte)

    end = time.time()
    elapsed = end-start
    print("Run time: " + str(timedelta(seconds=elapsed)))


def main(argv):
    model(None)


def predict():
    yaml_file = open("../sensing_data/models/dnn_tf_1_1.yaml", 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    dnn_pred = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    dnn_pred.load_weights("../sensing_data/models/dnn_tf_1_1.h5")
    print("Loaded model from disk")

    dnn_pred.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )

    dnn_pred.summary()

    X, y, shape = data.load_prediction(
        ratio=1, normalize=False, osm_roads=False, split_struct=False, army_gt=False)

    normalizer = preprocessing.Normalizer().fit(X)
    X = normalizer.transform(X)

    y_pred = dnn_pred.predict(X)
    y_pred = [np.argmax(pred) for pred in tqdm(y_pred)]

    kappa = cohen_kappa_score(y-1, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y-1, y_pred))

    y_pred = np.array(y_pred)
    yr = y_pred.reshape(shape)

    viz.createGeotiff(OUT_RASTER, yr, REF_FILE, gdal.GDT_Byte)


if __name__ == "__main__":
    # predict()
    main(sys.argv)
