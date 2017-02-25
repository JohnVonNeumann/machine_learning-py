# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:18:11 2017

@author: lw

"""

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
dataset = pd.read_csv('Salary_Data.csv') #use pandas to read csv data from a file
X = dataset.iloc[:, :-1].values #independent variable vector
Y = dataset.iloc[:, 1].values #dependent variable vector

#Splitting the dataset for training and test use
from sklearn.cross_validation import train_test_split #train_test_split is used to segregate test and train data in order to ensure we have datasets that minimise the risk of us overfitting our models
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # here we crtreate test and train sets, we outline set size, and becuase it always equals 1, we dont need to use the other number.

