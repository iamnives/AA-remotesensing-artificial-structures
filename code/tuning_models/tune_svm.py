import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import  make_scorer

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from scipy.stats import uniform

from utils import visualization as viz
from utils import data
from utils import metrics

from datetime import timedelta
import time

def model(dfs):
  start = time.time()
  train_size = 100_000 

  X_train, y_train, X_test , y_test = data.load(train_size, normalize=True, balance=False)
  # Set the parameters by cross-validation
  C_s = uniform(loc=0,scale=8)
  gamma = uniform(loc=0,scale=8)
  
  tuning_params = {'C': C_s, 'gamma':gamma, 'class_weight': ['balanced', None]} 

  kappa_scorer = make_scorer(cohen_kappa_score)
  gs = RandomizedSearchCV(svm.SVC(), tuning_params, cv=5, scoring={'kappa': kappa_scorer}, refit='kappa', return_train_score=True,  n_iter=10, verbose=2, n_jobs=-1)
  gs.fit(X_train, y_train)

  print("Best parameters set found on development set:")
  print()
  print(gs.best_params_)
  print()

  clf = gs.best_estimator_
  y_pred = clf.predict(X_test)

  kappa = cohen_kappa_score(y_test, y_pred)
  matrix = confusion_matrix(y_test, y_pred)

  print(f'Kappa: {kappa}')
  print(classification_report(y_test, y_pred))

  end=time.time()
  elapsed=end-start
  print("Run time: " + str(timedelta(seconds=elapsed)))

  viz.plot_confusionmx(matrix)
  viz.plot_gridcv(gs.cv_results_, ["kappa"], "C", 0, 10)
  viz.plot_gridcv(gs.cv_results_, ["kappa"], "gamma", 0, 10)

def main(argv):
  model(None)

if __name__== "__main__":
  main(sys.argv)