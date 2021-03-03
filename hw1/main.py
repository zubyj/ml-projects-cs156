import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Importing the dataset
dataset = pd.read_csv('Data-hw1.csv')
print(dataset)
x = dataset.iloc[: , :-1].values
# print("X")
# print(x)
y = dataset.iloc[: , -1].values
# print("Y")
# print(y)

# Data preprocessing
# Eliminating missing data
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer.fit(x[: , 1:])
x[: , 1:] = imputer.transform(x[: , 1:])
# print(x)

y = y.reshape(1, -1)
imputer.fit(y)
y = imputer.transform(y)
# print(y)

# Convert from categorical/non-numeric to numeric data
ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder() , [0])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
print(x)

#   

