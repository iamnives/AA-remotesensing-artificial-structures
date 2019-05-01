import os
import sys

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

from utils import visualization as viz
from utils import data

def main(argv):

    train_size = 100_000
    X_train, y_train, X_test , y_test = data.load(train_size, normalize=True, balance=True)

    scores = ['f1_weighted', 'accuracy', 'precision_weighted', 'recall_weighted']

    print("# Tuning hyper-parameters for %s" % scores)
    print()

    xgb_model = xgb.XGBClassifier()
    #brute force scan for all parameters, here are the tricks
    #usually max_depth is 6,7,8
    #learning rate is around 0.05, but small changes may make big diff
    #tuning min_child_weight subsample colsample_bytree can have 
    #much fun of fighting against overfit 
    #n_estimators is how many round of boosting
    #finally, ensemble xgboost with multiple seeds may reduce variance
    n_trees = [1,10,20,150, 300, 500]
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'tree_method': ['gpu_hist'],
              'gpu_id': [0],
              'max_bin': [16],
              'objective':['multi:softmax'],
              #params tuning
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'verbosity': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': n_trees , #number of trees, change it to 1000 for better results
              'missing':[-999],}

    gs = GridSearchCV(xgb_model, parameters, cv=5, n_jobs=5, 
                        scoring=scores, refit='precision_weighted', return_train_score=True)
    gs.fit(X_train, y_train)

    print("Best parameters set found on development set: precision_weighted")
    print()
    print(gs.best_params_)
    print()

    clf = gs.best_estimator_
    y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 =  f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Accuracy: {accuracy}')
    print(f'F1: {f1}')
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))

    viz.plot_confusionmx(matrix)
    viz.plot_gridcv(gs.cv_results_, scores, "n_estimators",  n_trees[0], n_trees[-1])

if __name__== "__main__":
  main(sys.argv)