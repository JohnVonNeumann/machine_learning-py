# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:18:11 2017

@author: lw

Data Preprocessing

Importing the libraries
"""

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
dataset = pd.read_csv('Data.csv') #use pandas to read csv data from a file
X = dataset.iloc[:, :-1].values #independent variable vector, so, country, age, salary, the number is a stop point, -1 means the first 
                                #from the right will not be included, as we calc UPTO the stop point, not including
Y = dataset.iloc[:, 3].values #dependent variable vector, purchased, in this instance

#Sorting out missing data values
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN', strategy="mean", axis=0) # for all NaN cells, replace with columnar mean
imputer = imputer.fit(X[:, 1:3]) #fits the imputer method onto the vector/array we need to be fixed, in this case, 
X[:, 1:3] = imputer.transform(X[:, 1:3]) #transforms any missing data points, based on our choices on line 25

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # label encoder is used to translate cat data into num data
                                                              # OneHotEncoder enables encoding of cat-to-num data in order to be read flatly, ie not with rising numbers, tilting results
labelencoder_X = LabelEncoder() # assign the method to a var we can use
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # for the first column (0) of the x vector, transform cat data into num data, then record it into the x/indep array
onehotencoder = OneHotEncoder(categorical_features = [0]) # this is just one of man optional params, at first i thought this should be set to all, but because we translated to num earlier, it's 0
X = onehotencoder.fit_transform(X).toarray() # reassign to X as it is ultimately what we wish to change, run fot_transform to shorten code base.

#Encoding of indep (purchased) vector
labelencoder_Y = LabelEncoder() # much of this is the same as above
Y = labelencoder_Y.fit_transform(Y) # a much simpler version of waht i wrote, makes sense that seeing as it's just 1s and 0s there is no need at all to use onehotencoder


#Splitting the dataset for training and test use
from sklearn.cross_validation import train_test_split #train_test_split is used to segregate test and train data in order to ensure we have datasets that minimise the risk of us overfitting our models
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # here we crtreate test and train sets, we outline set size, and becuase it always equals 1, we dont need to use the other number.

# Feature Scaling
from sklearn.preprocessing import StandardScaler # SS used to scale euclidean distances between values, in order to produce accurate results
standard_scaler = StandardScaler() # assign the method to a var
standard_scaler = standard_scaler.fit_transform(X, Y = None) # here is where I fuck up really, so i try and push both of them into the same method call, which really makes no sense
                                                             # now that i think about it