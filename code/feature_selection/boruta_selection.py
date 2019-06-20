import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
 
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from utils import data
import matplotlib.pyplot as plt 
import numpy as np
from datetime import timedelta
import time
from utils import visualization as viz
from utils import data

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from boruta import BorutaPy

# Initialize time variable for calculus of runtime
start = time.time()

# Load the dataset with optimal attrubutes after cross validation
train_size = int(19386625*0.2)
X, y, X_test , y_test  = data.load(train_size, normalize=False, balance=False, osm_roads=True) 

# Build a forest and compute the feature importances
forest = RandomForestClassifier(n_estimators=500,
                            min_samples_leaf=4, 
                            min_samples_split=2, 
                            max_depth=130,
                            class_weight='balanced',
                            n_jobs=-1, verbose=1)

# Get scores before selection
forest.fit(X, y)
y_pred = forest.predict(X_test)
kappa = cohen_kappa_score(y_test, y_pred)
print(f'Kappa: {kappa}')
print(classification_report(y_test, y_pred))

# Define Boruta feature selection method
boruta_selector = BorutaPy(forest, n_estimators=500, verbose=1)

# find all relevant features
boruta_selector.fit(X, y)

# check selected features
print("Support: " + str(boruta_selector.support_))

# check ranking of features
print("Ranking: " + str(boruta_selector.ranking_))

# call transform() on X to filter it down to selected features
X = boruta_selector.transform(X)
X_test = boruta_selector.transform(X_test)
print("Transformed data: " + str(X.shape))

# Get scores after selection
forest.fit(X, y)
y_pred = forest.predict(X_test)
kappa = cohen_kappa_score(y_test, y_pred)
print(f'Kappa: {kappa}')
print(classification_report(y_test, y_pred))


end=time.time()
elapsed=end-start
print("Run time: " + str(timedelta(seconds=elapsed)))