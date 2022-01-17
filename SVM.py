#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:18:37 2021

@author: Apple
"""

#importing Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# importing dataset from computer
dataset = pd.read_csv('Polynomial_dataset.csv')
X=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1].values

# using StandardScaler
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)
sc_y=StandardScaler()
y=sc_y.fit_transform(np.reshape(y,(10,1)))


# Set SVR for dataset
from sklearn.svm import SVR
svm_reg= SVR(kernel='rbf')
svm_reg.fit(X,y)


# Visualization
plt.scatter(X, y, color='red')
plt.plot(X, svm_reg.predict(X), color = 'black')
plt.title('SVR Regression')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()

#for predicting 
predict=sc_y.inverse_transform(svm_reg.predict(sc_X.transform(np.array([[6.5]]))))