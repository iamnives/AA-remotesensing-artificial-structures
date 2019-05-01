import os
import sys

from sklearn import svm
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

    train_size = 10_000
    X_train, y_train, X_test , y_test = data.load(train_size, datafiles=argv[1], normalize=True, balance=True)
    # Set the parameters by cross-validation

    C_s = [0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 300]
    gamma = [0.1, 0.5, 1, 2, 5, 10, 25, 50]
    
    tuning_params = [ {'C': C_s, 'gamma':['auto']} ]

    scores = ['f1_weighted', 'accuracy', 'precision_weighted', 'recall_weighted']

    print("# Tuning hyper-parameters for %s" % scores)
    print()

    gs = GridSearchCV(svm.SVC(), tuning_params, cv=5,
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
    viz.plot_gridcv(gs.cv_results_, scores, "C", C_s[0], C_s[-1])

if __name__== "__main__":
  main(sys.argv)