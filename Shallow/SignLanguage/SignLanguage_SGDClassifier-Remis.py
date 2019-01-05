
# coding: utf-8

# SGDClassifier with max_iter=200

# In[1]:


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



# In[2]:


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
#index = 200
#someSign = X_train.iloc[index].reshape(28,28)

#show original image
#plt.imshow(someSign)

#show binary image
#print("printing ", y_train[index])
#plt.imshow(someSign, cmap = matplotlib.cm.gray, interpolation="nearest")
#plt.show()





# In[3]:


##Convert Pandas dataframe to np array
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

#shuffle training set
shuffle_index = np.random.permutation(len(signs_train.index))
X_train,y_train = X_train[shuffle_index],y_train[shuffle_index]
someSign = X_train[0]



# In[4]:


# Display all sign language alphabet
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 10))
for index in range(0, 26):
    # find first element of letter and get index inside y_train[]
    signIndex = next((i for i in range(1000) if y_train[i] == index), None)
    #print(index, signIndex)
    if index == 0:
        SIGN_A = signIndex
    if signIndex == None:
        continue
    nextSign = X_train[signIndex]
    nextSignImage = nextSign.reshape(28,28)
    plt.subplot(5, 6, index+1)
    plt.axis('off')
    plt.imshow(nextSignImage, cmap = matplotlib.cm.gray, interpolation="nearest")
    plt.title('%c' % (65+index))
plt.show()


# ## Multiclass classification¶
# 
#  The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion). 
# 
# Using binary classifiers for multiclass classificatino
# OvA (One-vesus-All) strategy we need to train 24 binary classifiers
# OvO (One-verus-One) strategy, this means we would need to train 276 binary classifiers (N*(N-1)/2), but training sets are smaller because we only use two classes out of 24.
# 
# Scikit-Learn automatically detects when you try to use a binary classifier for multi-class classification task, and it automatically runs OvA (except for SVM where it uses OvO, since SVM scale poorly with the size).
# 

# In[5]:


## SGDClassifier
sgd_clf = SGDClassifier(random_state = 42, max_iter=200,n_jobs=-1) #use all cpu cores


# In[6]:


# train for multi-class classification
sgd_clf.fit(X_train, y_train)


# In[8]:


#cross-validation using K-fold with 3 folds
from sklearn.model_selection import cross_val_score
# Cross-validate SGDClassifier accuracy using cross_val_score()
print("SGD classifier Train scross validation, ",
      cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
      )


# In[10]:


print("testing SGD classifier accuracy...", 
      sgd_clf.score(X_test, y_test,sample_weight=None)
)


# In[11]:


#after classifier gets trained, classes_ array holds the list of target classes
sgd_clf.classes_


# In[12]:


print("Number of binary classifiers trained: ",len(sgd_clf.classes_))


# In[13]:


## Forcing to use OvO classifier strategy with SGDCClassifier 
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42,max_iter=200,tol=None), n_jobs=-1)
ovo_clf.fit(X_train, y_train)


# In[14]:


print("Number of binary classifiers trained: ",len(ovo_clf.estimators_))


# In[16]:


# Cross-validate OvO SGDClassifier accuracy using cross_val_score()
print("testing SGD OvO classifier accuracy...", 
      cross_val_score(ovo_clf, X_train, y_train, cv=3, scoring="accuracy")
      )


# In[17]:


print("testing OvO SGD classifier accuracy...",
      ovo_clf.score(X_test, y_test,sample_weight=None)
      )


# # ERROR ANALYSIS

# In[19]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
#calculaing confusion matrix for SGDClassifier
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx


# In[20]:


# plotting confusion matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# In[21]:


#comparing error rates rather than absolute number of erors
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums


# In[22]:


np.fill_diagonal(norm_conf_mx, 0)
#plt.figure(figsize=(10, 10))
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
#save_fig("confusion_matrix_errors_plot", tight_layout=False)
plt.show()


# In[23]:


## calculaing confusion matrix for TEST SGDClassifier
y_test_pred = cross_val_predict(sgd_clf, X_test, y_test, cv=3)
conf_mx = confusion_matrix(y_test, y_test_pred)
conf_mx


# In[24]:


# plotting confusion matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# In[25]:


#comparing error rates rather than absolute number of erors
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums


# In[26]:


np.fill_diagonal(norm_conf_mx, 0)
#plt.figure(figsize=(10, 10))
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
#save_fig("confusion_matrix_errors_plot", tight_layout=False)
plt.show()

