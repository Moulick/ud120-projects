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
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC

clf = SVC(kernel='rbf', C=10000)

# Cut down training dataset t 1%
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
print "Training Time:", round(t1-t0, 3), "s"

pred = clf.predict(features_test)
print 'pred :', pred
t2 = time()
print "Prediction Time:", round(t2-t1, 3), "s"


unique, count = np.unique(pred, return_counts=True)
counts = dict(zip(unique, count))
print counts

print 'Chris occured', counts[1], 'times!'

# print '10 :', pred[10]
# print '26 :', pred[26]
# print '50 :', pred[50]




score = clf.score(features_test, labels_test)
print score
t3 = time()
print "Scoring Time:", round(t3-t2, 3), "s"

#########################################################

# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Training Time: 22.783 s
# pred : [0 0 1 ..., 1 0 0]
# Prediction Time: 1.88 s
# {0: 875, 1: 883}
# Chris occured 883 times!
# 0.976109215017
# Scoring Time: 1.876 s
#
# Process finished with exit code 0