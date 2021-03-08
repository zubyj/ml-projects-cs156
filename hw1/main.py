import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Importing the dataset
dataset = pd.read_csv('Data-hw1.csv')
print(dataset)
x = dataset.iloc[: , [0, 1, 3]].values
y = dataset.iloc[: , 3:4].values

# Replace missing data with mean value. 
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer.fit(x[: , 1:])
x[: , 1:] = imputer.transform(x[: , 1:])
y = y.reshape(-1, 1)
imputer.fit(y)
y = imputer.transform(y)

# Convert from non-numeric/categorical to numeric data
ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder() , [0])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

# Split into test & training set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state = 1)

# Train Simple Linear Regression model on training set. 
# Experience vs Salary
yrs_exp_train = x_train[: , 2:3]
print(yrs_exp_train)
regressor = LinearRegression()
regressor.fit(yrs_exp_train, y_train)

# Visualizing training set results.
# plt.scatter(yrs_exp_train, y_train, color='red') 
# plt.plot(yrs_exp_train, regressor.predict(yrs_exp_train), color='blue')
# plt.show()

# Comparing test set results to actual results. 
y_pred = regressor.predict(x_test[: , 2:3])
# print('Expected')
# print(y_test)
# print('Actual')
# print(y_pred)

# Given years of experience, predict salary.
print(regressor.predict([[3.1]]))
print(regressor.predict([[7]]))
