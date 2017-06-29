#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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


print 'len feature_train', len(features_train[0])

#########################################################
### your code goes here ###

from sklearn import tree

clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)

t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
print "Training Time:", round(t1-t0, 3), "s"

pred = clf.predict(features_test)
print 'pred :', pred
t2 = time()
print "Training Time:", round(t2-t1, 3), "s"


unique, count = np.unique(pred, return_counts=True)
counts = dict(zip(unique,count))
print counts


print 'Chris occured', counts[1], 'times!'

# print '10 :', pred[10]
# print '26 :', pred[26]
# print '50 :', pred[50]


score = clf.score(features_test, labels_test)
print score
t3 = time()
print "Training Time:", round(t3-t2, 3), "s"

print(score)

#########################################################


