import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
from sklearn import svm
import numpy as np



def model(dfs):
    start = time.time()
    train_size = 100_000

    X_train, y_train, X_test, y_test = data.load(
        train_size, normalize=True, balance=False, osm_roads=False)
    # Set the parameters by cross-validation
    C_s = uniform(loc=0, scale=8)
    gamma = uniform(loc=0, scale=8)

    tuning_params = {'C': C_s, 'gamma': gamma, 'class_weight': ['balanced']}

    kappa_scorer = make_scorer(cohen_kappa_score)
    gs = RandomizedSearchCV(svm.SVC(), tuning_params, cv=3, scoring={
                            'kappa': kappa_scorer}, refit='kappa', return_train_score=False,  n_iter=50, verbose=1, n_jobs=-1)
    gs.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(gs.best_params_)
    print()
    print(gs.best_score_)
    print()
    
    clf = gs.best_estimator_
    y_pred = clf.predict(X_test)

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
