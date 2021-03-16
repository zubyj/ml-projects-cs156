# Predict employee's salary given certain features. 
# Using Reflex based Model  (Simple Linear Regression) 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Importing the dataset & splicing. 
dataset = pd.read_csv('Data-hw1.csv')
print(dataset)
x = dataset.iloc[: , :3].values
y = dataset.iloc[: , 3:].values

# Replace missing data with mean value.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[: , 1:])
x[: , 1:] = imputer.transform(x[: , 1:])
imputer.fit(y)
y = imputer.transform(y)

# Convert categorical (non-numeric) data
ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder(), [0])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

# Split into training & test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)

# Train using Simple Linear Regression Model
# Salary vs Experience
exp_train = x_train[: , 4:]
regressor = LinearRegression()
regressor.fit(exp_train, y_train)
salary_train_pred = regressor.predict(exp_train)

# Visualize training set results
plt.scatter(exp_train, y_train, color='red')
plt.plot(exp_train, salary_train_pred, color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predicting test set results
exp_test = x_test[: , 4:]
salary_test_pred = regressor.predict(exp_test)
print('Salary (actual)')
print(y_test)
print('Salary (predicted)')
print(salary_test_pred)

# Visualize test set results
plt.scatter(exp_test, y_test, color='red')
plt.plot(exp_test, salary_test_pred, color='blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predicting salary given years of experience. 
exp1 = 3.1
exp2 = 7.1
pred_1 = regressor.predict([[exp1]])
pred_2 = regressor.predict([[exp2]])
print(str(exp1) + ' Years Experience pays ' + str(pred_1))
print(str(exp2) + ' Years Experience pays' + str(pred_2))

# Visualize results for predicted salaries. 
plt.plot(exp_train, salary_train_pred, color='blue')
plt.scatter(3.1, pred_1, color='black')
plt.scatter(7, pred_2, color='black')
plt.title('Predicting Salary from 3.1 and 7 Yrs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()