# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:18:11 2017

@author: lw

THIS TEMPLATE WILL BE USED TO START MOST ML PROJECTS AND FILES, IT CONTAINS 
BASIC STEPS THAT WILL BE REQUIRED IN JUST ABOUT EVERY PROJECT, SHOULD MORE 
STEPS BE REQUIRED (IE: YOU ARE FACED WITH DIRTY AND CATEGORICAL DATA, OR 
REAL DATA, USE THE DATA_PREPROCESSING_TEMPLATE INSTEAD) AS THAT TEMPLATE WILL
PROVIDE A BETTER GUIDE. THIS ONE IS ONLY BEING USED FOR THE REMAINDER OF MY 
COURSE AS MOST OF THE DATASETS I WILL BE WORKING WITH, WILL BE MAINTAINED
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

# Feature Scaling
#OPTIONAL, MAY BE REQUIRED FROM TIME TO TIME.
"""
from sklearn.preprocessing import StandardScaler # SS used to scale euclidean distances between values, in order to produce accurate results
sc_X = StandardScaler() #cleaned up syntax, much nicer
X_train = sc_X.fit_transform(X_train) #we recall X_train made on ln44 and scale it using the method
X_test = sc_X.transform(X_test) # here we have no need to fit as well as tform, as it was done on the prev line
#no need to scale y as we already have them in accetpable ranges due to a simple 2feature encode