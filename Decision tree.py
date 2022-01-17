#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 19:21:24 2021

@author: Apple
"""

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



# Set Decisiontree regressor
from sklearn.tree import DecisionTreeRegressor
Dtree_reg= DecisionTreeRegressor(random_state=0)
Dtree_reg.fit(X,y)


# Visualization
x_grid=np.arange(min(X),max(X),0.1)
x_grid=np.reshape(len(x_grid),1)
plt.scatter(X, y, color='red')
plt.plot(x_grid, Dtree_reg.predict(x_grid), color = 'blue')
plt.title('Decision tree')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()


#Visualization on old
plt.scatter(X, y, color='red')
plt.plot(X, Dtree_reg.predict(X), color = 'black')
plt.title('Decision Regression')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()
#for predicting 

Dtree_reg.predict(np.reshape(6.5(1,1)))