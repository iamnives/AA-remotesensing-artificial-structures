import os
import sys

from sklearn.ensemble import RandomForestClassifier
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

    train_size = 7_000_000
    X_train, y_train, X_test , y_test = data.load(train_size, balance=True) 

    # N_s = [1, 5, 10,20,100, 150, 200, 300, 500]
    # min_samples_leaf = [1, 3, 4, 5]
    # min_samples_split = [2, 8, 10, 12]

    # tuning_params = [ {
    #                   'bootstrap': [True],
    #                   'max_depth': [80, 90, 100, 110, None],
    #                   'max_features': [2, 3, 'auto'],
    #                   'min_samples_leaf': min_samples_leaf,
    #                   'min_samples_split': min_samples_split,
    #                   'n_estimators': N_s,
    #                   'n_jobs': [4]
    #               } ]

    # Set the parameters by cross-validation, best params before feature selection TODO dont mess till FS ends
    N_s = [100]
    min_samples_leaf = [1]
    min_samples_split = [2]

    tuning_params = [ {
                      'n_estimators': N_s,
                      'n_jobs': [-1]
                  } ]

    scores = ['f1_weighted', 'precision_weighted', 'accuracy', 'recall_weighted']

    print(f'# Tuning hyper-parameters for {scores} on { X_train.shape[0] } samples')
    print()

    gs = GridSearchCV(RandomForestClassifier(), tuning_params, cv=5,
                        scoring=scores, refit='precision_weighted', return_train_score=True)
    gs.fit(X_train, y_train)

    print("Best parameters set found on development set: precision_weighted")
    print()
    print(gs.best_params_)
    print()

    clf = gs.best_estimator_
    y_pred = clf.predict(X_test)

    kappa = cohen_kappa_score(y_test, y_pred)
    
    matrix = confusion_matrix(y_test, y_pred)
    print(f'Kappa: {kappa}')
    print(classification_report(y_test, y_pred))

    viz.plot_confusionmx(matrix)
    viz.plot_gridcv(gs.cv_results_, scores, "n_estimators",  N_s[0], N_s[-1])

if __name__== "__main__":
  main(sys.argv)