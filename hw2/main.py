# Predicting house prices using the 4 following predictive models:
#   1. Simple Linear Regression
#   2. Multiple Linear Regression
#   3. Decision Tree Regression
#   4. Random Forest Regression
#

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

# Read dataset of houses & use house price as predictive output (dependent variable)
# Every other housing quality is an indepedent variable. 
dataset = pd.read_csv('Housing-Data-one-zip-4.csv')
print(dataset)
x = dataset.iloc[: , dataset.columns != 'price'].values
y = dataset.iloc[: , -2].values

# Convert categorical (non-numeric) data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2,3,4])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Transform year built to age of building 
x[: , -1] = 2021 - x[: , -1]
print(x[: , -1])

# Divide into training & test set
sqft_living = x[: , -9]
x_train, x_test, y_train, y_test = train_test_split(sqft_living, y, test_size=.2, random_state=1)
print(x_train)