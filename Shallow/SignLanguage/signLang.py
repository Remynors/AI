# -*- coding: utf-8 -*-
"""
American Sign language Project
28/01/2018
Remis Norvilis
"""
import os
SIGNLANG_PATH = "~/ml/datasets/signLang"

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

# load data function
def load_signLand_data(signLang_path=SIGNLANG_PATH, file="sign_mnist_train.csv"):
    csv_path = os.path.join(signLang_path, file)
    return pd.read_csv(csv_path)

## Load train data
signs_train = load_signLand_data()
#signs.head()
signs_train.info()
print("Training data array shape:signs_t ",signs_train.shape)
print("Training data length: ",len(signs_train.index))

file_test = "sign_mnist_test.csv"
signs_test = load_signLand_data(SIGNLANG_PATH,file_test)
signs_test.info()
print("Testing data array shape:signs_t ",signs_test.shape)
print("Testing data length: ",len(signs_test.index))

## Process data
#extract labels column into y
y_train = signs_train.loc[:,'label']
y_test = signs_test.loc[:,'label']

#extract pixel array into X
X_train= signs_train
X_train.drop('label',axis=1,inplace=True)
X_test= signs_test
X_test.drop('label',axis=1,inplace=True)

## Display images
#format into 28x28 pixel image
index = 5
someSign = X_train.iloc[index].reshape(28,28)

#show original image
#plt.imshow(someSign)

#show binary image
print("printing ", y_train[index])
plt.imshow(someSign, cmap = matplotlib.cm.gray, interpolation="nearest")
plt.show()

##Convert Pandas dataframe to np array
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

#display random item
index = 300
someSign = X_train[index].reshape(28,28)
plt.imshow(someSign, cmap = matplotlib.cm.gray, interpolation="nearest")
plt.show()

A_sign = X_train[5].reshape(28,28)
plt.imshow(A_sign, cmap = matplotlib.cm.gray, interpolation="nearest")
plt.show()

#shuffle training set
shuffle_index = np.random.permutation(len(signs_train.index))
X_train,y_train = X_train[shuffle_index],y_train[shuffle_index]


## TRAINING ###

## Training binary classifier
y_train_5 = (y_train == 5) 
y_test_5 = (y_test == 5) 

print("Training...")
sgd_clf = SGDClassifier(random_state = 42, max_iter=3000)
sgd_clf.fit(X_train, y_train_5)

### VALIDATION ###
# manual testing for few samples
print("Testing...")
print(y_train_5[19],sgd_clf.predict([X_train[19]]))
print(y_train_5[20],sgd_clf.predict([X_train[20]]))
print(y_test_5[95],sgd_clf.predict([X_test[95]]))
print(y_test_5[94],sgd_clf.predict([X_test[94]]))
print(y_test_5[111],sgd_clf.predict([X_test[111]]))

#cross-validation using K-fold with 3 folds
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3,scoring="accuracy")
#getting 99.9% accuracy, array([ 0.99901672,  0.99967217,  0.99934433])

# confusion matric computation returning predictions made on each test fold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
# array([[26241,    10],
#       [    8,  1196]])
# non-5 images, 26241 corectly classified as not-5 (true negatives TN)
#               10 wrongly classified as 5 (false positives FP)
# 5 images (positive class), 8 wrongly classified (flase negatives FN)
#                            1196 correctly classified (true positives TP)


# Precision of the classifier =  TP/(TP + FP)
# Recall (sensitivity) = TP/(TP+FN)
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)  
# 0.99170812603648428
recall_score(y_train_5, y_train_pred)
# 0.99335548172757471
# !!! Sign 5 is correct 99.2% of the time and it detects 99.3% of all 5s

#Combining precision and recall into once metric - F1 score (harmonic mean)
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)
# 0.99253112033195035, 99.2%





 