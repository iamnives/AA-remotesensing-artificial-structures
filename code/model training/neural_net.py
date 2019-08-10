import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib
import time
from datetime import timedelta
import gdal
from utils import metrics
from utils import data
from utils import visualization as viz
from sklearn.utils import class_weight
from scipy.stats import uniform
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_yaml
from tqdm import tqdm
from sklearn import preprocessing

# inicialize data location
DATA_FOLDER = "../sensing_data/"
ROI = "vila-de-rei/"
DS_FOLDER = DATA_FOLDER + "clipped/" + ROI

OUT_RASTER = DATA_FOLDER + "results/" + ROI + \
    "/static/ann/neural_20px_static_group3_classification.tiff"
REF_FILE = DATA_FOLDER + "clipped/" + ROI + \
    "/ignored/static/clipped_sentinel2_B08.vrt"

def model(dfs):
    start = time.time()
    train_size = int(19386625*0.2)

    split_struct=False
    osm_roads=True

    X_train, y_train, X_test, y_test = data.load(
        train_size, normalize=True, osm_roads=osm_roads, split_struct=split_struct, army_gt=False)

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
    dnn.add(Dense(256, input_dim=input_shape, activation='relu'))
    dnn.add(Dense(512, input_dim=input_shape, activation='relu'))
    dnn.add(Dense(512, input_dim=input_shape, activation='relu'))
    dnn.add(Dropout(0.2))
    dnn.add(Dense(units=logits, activation='softmax'))

    dnn.compile(
        loss='categorical_crossentropy',
        optimizer='Nadam',
        metrics=['accuracy']
    )
    dnn.summary()

    es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')

    dnn.fit(X_train, y_train_onehot,
            epochs=100, validation_split=0.2, class_weight=class_weights, callbacks=[es])

    y_pred_onehot = dnn.predict(X_test)
    y_pred = [np.argmax(pred) for pred in y_pred_onehot]

    kappa = cohen_kappa_score(y_test, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))


    # serialize model to YAML
    model_yaml = dnn.to_yaml()
    with open("../sensing_data/models/dnn_tf_static_group3.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    dnn.save_weights("../sensing_data/models/dnn_tf_static_group3.h5")
    print("Saved model to disk")

    # Testing trash
    X, y, shape = data.load_prediction(ratio=1, normalize=True, osm_roads=osm_roads, split_struct=split_struct, army_gt=False)
    print(X.shape, y.shape)

    y_pred = dnn.predict(X)
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

    X, y, shape = data.load_prediction(ratio=1, normalize=False, osm_roads=False, split_struct=False, army_gt=False)

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
    #predict()
    main(sys.argv)
