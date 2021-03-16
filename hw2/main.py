# Predicting house prices using the 4 following predictive models:
#   1. Simple Linear Regression
#   2. Multiple Linear Regression
#   3. Decision Tree Regression
#   4. Random Forest Regression
#

from test_set import houses_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics._regression import mean_squared_error
from numpy.lib.scimath import sqrt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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

# Divide into training & test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)

# Simple Linear Regression Model
#
regressor = LinearRegression()
# Train using sqft_living as independent variable
sqft_living_train = x_train[: , 17:18]
regressor.fit(sqft_living_train, y_train)
r_square = regressor.score(sqft_living_train, y_train)
print('Simple Linear Regression (R square) : ' + str(r_square))
# Plot linear regression line for training set
plt.scatter(sqft_living_train, y_train, color='red')
plt.plot(sqft_living_train, regressor.predict(sqft_living_train), color='blue')
plt.title('House Price vs Sqft_living (Training data)')
plt.xlabel('Square ft living')
plt.ylabel('House Price')
plt.show()
# Plot linear regression line for test set
sqft_living_test = x_test[: , 17:18]
plt.scatter(sqft_living_test, y_test, color='red')
plt.plot(sqft_living_train, regressor.predict(sqft_living_train), color='blue')
plt.title('House Price vs Square ft living (Test data)')
plt.xlabel('Square ft living')
plt.ylabel('House Price')
plt.show()
# Predict house prices for some test cases
sqft_living_test = np.array(houses_data)[: , 23:24]
for sqft_living in sqft_living_test:
    print('Sq feet living ' + str(sqft_living[0]))
    pred_price = str(regressor.predict([sqft_living])[0])
    print('Predicted cost : $' + pred_price) 
    print()

def print_house_price_preds():
    print('R Square : ' + str(regressor.score(x_train, y_train)))
    for house in houses_data:
        print('Given features : ' + str(house))
        print('Predicted House Price ' + str(regressor.predict([house])))
    print()

# Multiple Linear Regression Model
print("Multiple Linear Regression")
regressor.fit(x_train, y_train)
print_house_price_preds()
print("Coefficients " + regressor.coef_)
print("Intercepts : " + regressor.intercept_)

# Decision Tree Regression Model
print("Decision Tree Regression")
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train, y_train)
print_house_price_preds()

# Random Forest Regression
print("Random Forest Regression")
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x, y)
print_house_price_preds()