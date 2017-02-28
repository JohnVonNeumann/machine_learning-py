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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0) # here we crtreate test and train sets, we outline set size, and becuase it always equals 1, we dont need to use the other number.

# Fitting the simple linear regression to training set
from sklearn.linear_model import LinearRegression # import the LR class
regressor = LinearRegression() # create a var for the methods to reside in
regressor.fit(X_train, Y_train) # using the LR class, we call the fit method on our x_train and y_train datasets, or the indep and dep sets.

# using our machine to predict test set results
y_pred = regressor.predict(X_test) # y_pred because we are predicting the dependent var, that is we are predicting what the salary of someone should be given a career of n years

# visualise the training set results 
plt.scatter(X_train, Y_train, color = 'red') #scatter red points with the training dataset, this is real data
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #create a blue regression line for real salary increases vs career length
plt.title('Salary vs Experience (Training Set)') #create a title for the graph
plt.xlabel('Years of Experience') #label the x axis
plt.ylabel('Salary') # label the y axis
plt.show() # output the graph
