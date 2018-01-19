import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd

import csv
import math as m
import random

filename = 'zoo.data.txt'

with open(filename) as f:
    reader = csv.reader(f)
    data = list(reader)

legs = [p[13] for p in points]
animaltype = [p[17] for p in points]
points = [(float(p[13]), float(p[17])) for p in data]

Y = df['legs']
X = df['animalype']
 
X=X.reshape(len(X),1)
Y=Y.reshape(len(Y),1)
 
# Split the data into training/testing sets
X_train = X[:-101]
X_test = X[-101:]
 
# Split the targets into training/testing sets
Y_train = Y[:101]
Y_test = Y[101:]
 
# Plot outputs
plt.scatter(X_test, Y_test,  color='black')
plt.title('Test Data')
plt.xlabel('animaltype')
plt.ylabel('legs')
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
