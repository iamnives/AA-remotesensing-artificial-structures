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


# Tensorflow trash
def model(dfs):
    start = time.time()
    train_size = 100_000
    X_train, y_train, X_test, y_test = data.load(
        train_size, normalize=True, balance=False)

    input_shape = X_train.shape[1]
    logits = 4

    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=logits)

    dnn = Sequential()
    # Define DNN structure
    dnn.add(Dense(32, input_dim=input_shape, activation='relu'))
    dnn.add(Dense(units=logits, activation='softmax'))

    dnn.compile(
        loss='categorical_crossentropy',
        optimizer='Adadelta',
        metrics=['accuracy']
    )
    dnn.summary()

    history_rmsprop = dnn.fit(X_train, y_train_onehot,
                              epochs=10, validation_split=0.1)

    # plot the accuracy
    gen_graph(history_rmsprop,
              "ResNet50 RMSprop")

    y_pred_onehot = dnn.predict(X_test)

    y_pred = [np.argmax(pred) for pred in y_pred_onehot]

    kappa = cohen_kappa_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))

    end = time.time()
    elapsed = end-start
    print("Run time: " + str(timedelta(seconds=elapsed)))

    viz.plot_confusionmx(matrix)


def main(argv):
    model(None)


if __name__ == "__main__":
    main(sys.argv)
