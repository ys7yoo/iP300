#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# cross validation using grid search
from sklearn.model_selection import GridSearchCV
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    
from sklearn.neighbors import KNeighborsClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# confusion matrix
from sklearn import metrics
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html




#from tqdm import tqdm # for progressive bar

from sklearn.externals import joblib # to save model


from P300.data import load_data


# ## k-nearest neighbor classifier

# In[2]:


subject_train = 'A'
subject_test = 'A'
X_train, Y_train, Code_train = load_data(subject_train, 'train', 85)
X_test, Y_test, Code_test = load_data(subject_test, 'test', 100)


# In[ ]:


n_neighbors = np.arange(3, 10, 1, dtype=int)
#n_neighbors = np.int(np.linspace(3, 10, 8))
print(n_neighbors)
parameters = { 'n_neighbors':n_neighbors}

clf = KNeighborsClassifier(n_jobs=-1)

clf_cv = GridSearchCV(clf, parameters, cv=5)
clf_cv.fit(X_train, Y_train)

