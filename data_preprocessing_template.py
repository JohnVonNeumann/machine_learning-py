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
y = dataset.iloc[:, 3].values #dependent variable vector, purchased, in this instance

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
onehotencoder = OneHotEncoder() # assign the method to a var we can use
onehotencoder.fit_transform(X[:, 0]) # shitty hack attempting to copy what we did last time, i didnt think it would work haha