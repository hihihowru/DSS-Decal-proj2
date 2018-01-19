import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd

import csv
import math as m
import random

filename = 'imports-85.data.txt'

with open(filename) as f:
    reader = csv.reader(f)
    data = list(reader)

curbweight = [point[13] for point in data]
enginesize = [point[16] for point in data]
points = [(float(p[13]), float(p[16])) for p in data]

Y = df['curbweight']
X = df['enginesize']
 
X=X.reshape(len(X),1)
Y=Y.reshape(len(Y),1)
 
# Split the data into training/testing sets
X_train = X[:-205]
X_test = X[-205:]
 
# Split the targets into training/testing sets
Y_train = Y[:205]
Y_test = Y[205:]
 
# Plot outputs
plt.scatter(X_test, Y_test,  color='black')
plt.title('Test Data')
plt.xlabel('enginesize')
plt.ylabel('curbweight')
plt.xticks(())
plt.yticks(())
 
plt.show()

# Create linear regression object
regr = linear_model.LinearRegression()
 
# Train the model using the training sets
regr.fit(X_train, Y_train)
 
# Plot outputs
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3)

# Individual prediction using linear regression model
print( str(round(regr.predict(1000))) )
