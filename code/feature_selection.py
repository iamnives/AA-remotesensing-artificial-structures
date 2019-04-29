from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from utils import data
import matplotlib.pyplot as plt 
import numpy as np

from utils import visualization as viz
from utils import data

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

sample_size = 100_000
X, y, X_test , y_test = data.load(sample_size) 

# Build a forest and compute the feature importances
forest = RandomForestClassifier(n_estimators=500, n_jobs=4)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

y_pred = forest.predict(X_test)
print(classification_report(y_test, y_pred))

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="c", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()