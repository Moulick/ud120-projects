#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.ensemble import  AdaBoostClassifier
import numpy as np
from time import time


features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary


# Cut down training dataset to 1%
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


clf = AdaBoostClassifier(n_estimators=100)


# Timming of training
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
pred = clf.predict(features_test)
# print 'pred :', pred
t2 = time()
print "Prediction Time:", round(t2 - t1, 3), "s"

score = clf.score(features_test, labels_test)
print 'score :',score*100
t3 = time()
print "Scoring Time:", round(t3 - t2, 3), "s"


class MyClass:
    # simple example class
    i = 12345

    def f(self):
        return 'hello world'

    def g(self):
        return self.f()




try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
