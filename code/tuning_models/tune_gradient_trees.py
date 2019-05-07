import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import xgboost as xgb 
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

def main(argv):

    start = time.time()
    train_size = 100_000
    X_train, y_train, X_test , y_test = data.load(train_size, normalize=False, balance=False)


    xgb_model = xgb.XGBClassifier()
    #brute force scan for all parameters, here are the tricks
    #usually max_depth is 6,7,8
    #learning rate is around 0.05, but small changes may make big diff
    #tuning min_child_weight subsample colsample_bytree can have 
    #much fun of fighting against overfit 
    #n_estimators is how many round of boosting
    #finally, ensemble xgboost with multiple seeds may reduce variance
    n_trees =  [500, 1000, 1500, 2000]
    parameters = {'nthread':[4],
              'tree_method': ['gpu_hist'],
              'predictor':['gpu_predictor'],
              'gpu_id': [0],
              'objective':['multi:softmax'],
              #params tuning
              'learning_rate': uniform(), #`eta` value
              'max_depth':[ 3, 4, 5, 6, 8, 10, 12, 15],
              'min_child_weight': [ 1, 3, 5, 7 ],
              "gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
              'colsample_bytree': uniform(),
              'n_estimators': n_trees,
              'verbose': [1] }

    kappa_scorer = make_scorer(cohen_kappa_score)
    gs = RandomizedSearchCV(xgb_model, parameters, cv=5, scoring={'kappa': kappa_scorer}, refit='kappa', return_train_score=True,  n_iter=10, verbose=2)
    gs.fit(X_train, y_train)

    print("Best parameters set found on development set: ")
    print()
    print(gs.best_params_)
    print()

    clf = gs.best_estimator_
    y_pred = clf.predict(X_test)
    
    kappa, matrix, report = metrics.scores(y_test, y_pred)

    end=time.time()
    elapsed=end-start
    print("Run time: " + str(timedelta(seconds=elapsed)))

    viz.plot_confusionmx(matrix)
    viz.plot_gridcv(gs.cv_results_, ["kappa"], "n_estimators", 500, 2000)

if __name__== "__main__":
  main(sys.argv)