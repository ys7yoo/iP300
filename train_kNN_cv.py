#!/usr/bin/env python

import numpy as np
from scipy.io import loadmat


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


import sys

if len(sys.argv) < 2:
    print ('choose training dataset (A, B, AandB)')
    exit()

subject_train = str(sys.argv[1]).strip()
print(subject_train)


## load training data

#subject_train = 'A'
#subject_train = 'B'
#subject_test = 'A'
#subject_test = 'B'

if subject_train == 'AandB':
    subject = 'A'
    X_train_A, Y_train_A, Code_train_A = load_data(subject, 'train', 85)
    #X_test_A, Y_test_A, Code_test_A = load_data(subject, 'test', 100)

    subject = 'B'
    X_train_B, Y_train_B, Code_train_B = load_data(subject, 'train', 85)
    #X_test_B, Y_test_B, Code_test_B = load_data(subject, 'test', 100)

    # combine
    X_train = np.vstack((X_train_A,X_train_B))
    #print(X_train.shape)
    Y_train = np.concatenate((Y_train_A, Y_train_B))
    #print(Y_train.shape)

    #X_test = np.vstack((X_test_A, X_test_B))
    #print(X_test.shape)
    #Y_test = np.concatenate((Y_test_A, Y_test_B))
    #print(Y_test.shape)

else:
    X_train, Y_train, Code_train = load_data(subject_train, 'train', 85)
    #X_test, Y_test, Code_test = load_data(subject_test, 'test', 100)



## cross validation for kNN classifier


k = np.arange(3, 16, 1, dtype=int)
print(k)
parameters = {'n_neighbors':k}

clf = KNeighborsClassifier(n_jobs=-1)

clf_cv = GridSearchCV(clf, parameters, cv=5)
clf_cv.fit(X_train, Y_train)


# save model
num_ch = 64
joblib.dump(clf_cv, "{}_kNN_cv_ch{}.model".format(subject_train,num_ch))

