import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
import tensorflow as tf
import matplotlib
import time
from datetime import timedelta
from utils import metrics
from utils import data
from utils import visualization as viz
from scipy.stats import uniform
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch



# TensorFlow and tf.keras


def gen_graph(history, title):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy ' + title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss ' + title)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()



class MyHyperModel(HyperModel):

    def __init__(self, classes):
        self.classes = classes

    def build(self, hp):
        model = tf.keras.Sequential()
        for i in range(hp.Int('num_layers', 2, 20)):
            model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=32,
                                                max_value=512,
                                                step=32),
                                activation='relu'))
        model.add( tf.keras.layers.Dense(self.classes, activation='softmax'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model

def predict_metrics(model, X_test, y_test):
    y_pred_onehot = model.predict(X_test)

    y_pred = [np.argmax(pred) for pred in y_pred_onehot]

    kappa = cohen_kappa_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))
    viz.plot_confusionmx(matrix)

# Tensorflow trash
def model(dfs):
    X_train, y_train, X_test, y_test, val_x, val_y, norm = data.load(normalize=True, znorm=False, datafiles=dfs, map_classes=True, binary=False, test_size=0.2, osm_roads=False, army_gt=False, urban_atlas=True, split_struct=False, gt_raster="ua_ground.tif") 

    logits = 2

    hypermodel = MyHyperModel(classes=logits)

    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=10,
        directory='./dnn_tunes',
        project_name='au2018_tune')

    tuner.search(X_train, y_train,
                epochs=5,
                validation_data=(val_x, val_y))

    tuner.search_space_summary()

    tuner.search(X_train, y_train,
             epochs=5,
             validation_data=(val_x, val_y))

    models = tuner.get_best_models(num_models=2)

    tuner.results_summary()


def main(argv):
    model(None)


if __name__ == "__main__":
    main(sys.argv)
