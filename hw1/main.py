import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Data-hw1.csv')
print(dataset)
x = dataset.iloc[: , [0, 1, 3]].values
y = dataset.iloc[: , 2].values

# Eliminating missing data
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer.fit(x[: , 1:])
x[: , 1:] = imputer.transform(x[: , 1:])
y.reshape(-1, 1)
imputer.fit(y)
y = imputer.transform(y)

# Convert from categorical/non-numeric to numeric data
ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder() , [0])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

# Split into test & training set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state = 1)
print('xtrain')
print(x_train)
print('ytrain')
print(y_train)    
print('xtest')
print(x_test)
print('ytest')
print(y_test)