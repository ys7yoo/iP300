import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt
#%matplotlib inline 

## general SVM 
from sklearn.svm import SVC

## Let's start from linear SVM 
from sklearn.svm import LinearSVC
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
from sklearn import svm

from tqdm import tqdm # for progressive bar

from sklearn.externals import joblib # to save model

from P300.data import load_data




# load training data
num_epoches = 85

X_train_A, Y_train_A, C_train_A = load_data('A', 'train', num_epoches)
X_train_B, Y_train_B, C_train_B = load_data('B', 'train', num_epoches)


# combine
X_train = np.vstack((X_train_A,X_train_B))
print(X_train.shape)
Y_train = np.concatenate((Y_train_A, Y_train_B))
print(Y_train.shape)


# train SVM
clf = svm.SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, max_iter=-1, random_state=42)
#clf = svm.LinearSVC(C=1.0, loss="hinge", class_weight='balanced', probability=True, max_iter=5000, random_state=42)
clf.fit(X_train, Y_train)


# training score
clf.score(X_train, Y_train)


# save model
model_subject='AandB'
num_ch = 64
joblib.dump(clf, "{}_SVC_linear_ch{}.model".format(model_subject,num_ch))



# test scores
X_test_A, Y_test_A, C_test_A = load_data('A', 'test', num_epoches)
X_test_B, Y_test_B, C_test_B = load_data('B', 'test', num_epoches)


clf.score(X_test_A, Y_test_A)


clf.score(X_test_B, Y_test_B)


