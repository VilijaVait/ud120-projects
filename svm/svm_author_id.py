#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#use in case need to reduce training set
features_train = features_train[:len(features_train)/1]
labels_train = labels_train[:len(labels_train)/1]

t0 = time()
clf = SVC(C=10000,kernel = 'rbf')
clf.fit(features_train, labels_train)

accuracy = clf.score(features_test, labels_test)
pred = clf.predict(features_test)
pred_chris = sum(pred)

print 'Accuracy: {}'.format(round(accuracy,4))
print 'Emails in Chris class in test set: {}'.format(pred_chris)
print 'training time: {} secs'.format(round(time()-t0),3)
