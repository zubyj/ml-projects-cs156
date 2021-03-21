# Using Machine Learning Classification Algorithms to 
# build a predictive model.
# Given a persons features, will they survive the Titanic sinking?

'''
Backstory 
The sinking of the Titanic is one of the most infamous shipwrecks in 
history. On April 15, 1912, during her maiden voyage, the widely 
considered “unsinkable” RMS Titanic sank after colliding with an iceberg.
Unfortunately, there weren’t enough lifeboats for everyone onboard,
resulting in the death of 1502 out of 2224 passengers and crew.
While there was some element of luck involved in surviving, it seems
some groups of people were more likely to survive than others.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_set import passengers

# Split data into each person's features (x) and if person survived (y)
dataset = pd.read_csv('Data-Hw3.csv')
x = dataset.iloc[: , 2:-1].values
y = dataset.iloc[: , -1].values

# Need to convert categorical data (only needed for logistic regression)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Replace missing data with mean value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[: , 2:3])
x[: , 2:3] = imputer.transform(x[: , 2:3])  

# Split into training set (75%) and test set (25%)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1)

def print_results(y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    print("Confusion Matrix " +  str(cm))
    print("Accuracy Score " + str(acc_score))

    print("y actual : " + str(y_pred))
    print("y expected " + str(y_test))

    # Print results for data set
    for passenger in passengers[0]:
        print()
        print('Given features '  + str(passenger))
        if regressor.predict([passenger]) == 0:
            print("The passenger will die")
        else:
            print("The passenger will survive")

# Logistic Regression
print()
print("LOGISTIC REGRESSION")
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train, y_train)
# Predicing test set results
y_pred = regressor.predict(x_test)
# Creating the confusion matrix for results.
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)
print_results(y_pred, y_test)

# K Nearest Neighbor Classification 
print()
print("K NEAREST NEIGHBOR CLASSIFICATION")
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print_results(y_pred, y_test)