#importing Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# importing dataset from computer
dataset = pd.read_csv('Exam_Dataset.csv')
X=dataset.iloc[:, : -1].values
y=dataset.iloc[:, 3].values

#training and testing data (Divide data into two parts)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Linear_Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_prdict = reg.predict(X_test)

