# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:35:09 2019

@author: danny
"""

# Simple Linear Regression


#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Datasets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitting dataset into Training set nd Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

#Fitting SLR to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(x_train, y_train)

#Predicting the test set result 
y_pred = regressor.predict(x_test)

#Visualizing the Training set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experinece (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show() 

#Visualizing the Test set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experinece (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show() 