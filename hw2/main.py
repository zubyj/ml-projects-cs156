# Predicting house prices using the 4 following predictive models:
#   1. Simple Linear Regression
#   2. Multiple Linear Regression
#   3. Decision Tree Regression
#   4. Random Forest Regression
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics._regression import mean_squared_error
from numpy.lib.scimath import sqrt
from test_set import test_set

# Read dataset of houses & use house price as predictive output (dependent variable)
# Every other housing quality is an indepedent variable. 
dataset = pd.read_csv('Housing-Data-one-zip-4.csv')
x = dataset.iloc[: , dataset.columns != 'price'].values
y = dataset.iloc[: , -2].values

# Convert categorical (non-numeric) data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2,3,4])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Transform year built to age of building 
x[: , -1] = 2021 - x[: , -1]

# Simple Linear Regression Model
#
# Train using sqft_living as independent variable
# Divide into training & test set
sqft_living = x[: , 17:18]
x_train, x_test, y_train, y_test = train_test_split(sqft_living, y, test_size=.2, random_state=1)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
# Show root mean squared value (R squared)
rms = sqrt(mean_squared_error(y_test, y_pred))
print("RMS " + str(rms))
# Plot linear regression for training data
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.show()
# Plot linear regression for test data
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, y_pred, color='blue')
plt.show()
# Predict house prices for some test cases
sqft_living_test = test_set[: , 2:3]
for sqft_living in sqft_living_test:
    print("SQFT LIVING : " + str(sqft_living))
    print("PRED PRICE " + str(regressor.predict([sqft_living])))
