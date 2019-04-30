import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from utils import data
from sklearn.model_selection import GridSearchCV

sample_size = 500_000
X, y, _ , _s = data.load(sample_size, balance=True) 

clf = LassoCV(cv=10)
clf.fit(X, y)

print(f'CV alpha: { clf.alpha_ }')
print(clf.coef_)
# Do a SelectFromModel after this

coefs = clf.coef_
indices = np.argsort(coefs)[::-1]

# Print the feature ranking
print("Coefitients per feature:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], coefs[indices[f]]))

# Plot the feature importances of the forest
fig = plt.figure()

plt.title("Feature coeficients")
plt.bar(range(X.shape[1]), coefs[indices],
       color="c", align="center")
plt.xticks(range(X.shape[1]), data.feature_map(indices), rotation='45', horizontalalignment="right")
plt.xlim([-1, X.shape[1]])
fig.tight_layout()
plt.show()