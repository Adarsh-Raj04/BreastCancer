# -*- coding: utf-8 -*-


import numpy as np
import sklearn.datasets

breast_cancer = sklearn.datasets.load_breast_cancer()

print(breast_cancer)

X = breast_cancer.data
Y= breast_cancer.target
print(X)
print(Y)

print(X.shape,Y.shape)

import pandas as pd
data= pd.DataFrame(breast_cancer.data,columns = breast_cancer.feature_names)

data['class']=breast_cancer.target

data.head()

data.describe()

print(data['class'].value_counts())

print(breast_cancer.target_names)

data.groupby('class').mean()

"""0-malignant
1-benign

Train and test split
"""

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test =  train_test_split(X,Y)

print(Y.shape,Y_train.shape,Y_test.shape)

X_train,X_test,Y_train,Y_test =  train_test_split(X,Y,test_size=0.1)

print(Y.shape,Y_train.shape,Y_test.shape)

print(Y.mean(),Y_train.mean(),Y_test.mean())

#stratify to correct distribution of data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify =Y)

print(Y.mean(),Y_train.mean(),Y_test.mean())

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify =Y,random_state =1)

print(X_train.mean(),X_test.mean(),X.mean())

print(X_train)

#Logistic Regression model
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

#training the model
classifier.fit(X_train,Y_train)

#Evaluation of model 
from sklearn.metrics import accuracy_score

prediction_on_training_data = classifier.predict(X_train)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)

print("Accuracy on training data",accuracy_on_training_data)

#prediction on test data
prediction_on_test_data = classifier.predict(X_test)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)
print("Accuracy on training data",accuracy_on_test_data)

#Detecting whether patient has benign or malignant
input_data = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
#change input to numpy array
input_data_as_numpy_array = np.asarray(input_data)
print(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print("the breast cancer is Malignant")

else:
  print("the breast cancer is Benign")

input_data=(12.05,14.63,78.04,449.3,0.1031,0.09092,0.06592,0.02749,0.1675,0.06043,0.2636,0.7294,1.848,19.87,0.005488,0.01427,0.02322,0.00566,0.01428,0.002422,13.76,20.7,89.88,582.6,0.1494,0.2156,0.305,0.06548,0.2747,0.08301)
input_data_as_numpy_array = np.asarray(input_data)
print(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print("the breast cancer is Malignant")

else:
  print("the breast cancer is Benign")

