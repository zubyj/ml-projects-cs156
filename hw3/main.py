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
print(x)

# Split into training set (75%) and test set (25%)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1)

# # Feature Scaling (not needed since values all below 100)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)
# print(x_train)
# print(x_test)

# For each classification algorithm below, prints
# actual vs expected results, confusion matrix, & accuracy score.
from sklearn.metrics import confusion_matrix, accuracy_score
def print_results(y_test, y_pred):
    print('y actual : ' + str(y_pred))
    print('y expected ' + str(y_test))
    # Given test set of passengers, predict if they live or die.
    for passenger in passengers[0]:
        print()
        print('Given features '  + str(passenger))
        if regressor.predict([passenger]) == 0:
            print('The passenger will die')
        else:
            print('The passenger will survive')
    cm = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    print('\nConfusion Matrix ' +  str(cm))
    print('\nAccuracy Score ' + str(acc_score))

# Logistic Regression
print('\nLOGISTIC REGRESSION')
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print_results(y_test, y_pred)

# K Nearest Neighbor Classification 
print()
print('\nK NEAREST NEIGHBOR CLASSIFICATION')
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print_results(y_test, y_pred)

# Support Vector Machine
print('\nLINEAR SUPPORT VECTOR MACHINES')
from sklearn.svm import SVC
classifier = SVC(kernel =  'linear', random_state = 1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print_results(y_test, y_pred)

# Kernel SVM
print('\nKERNEL SVM')
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print_results(y_test, y_pred)

# Naive Bayes 
print('\nNAIVE BAYES')
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print_results(y_test, y_pred)

# Decision Tree Classification
print('\nDECISION TREE CLASSIFICATION')
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print_results(y_test, y_pred)

# Random Forest Classification
print('\n RANDOM FOREST CLASSIFICATION')
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print_results(y_test, y_pred)

'''
Top 3 Models from the accuracy
1. Logistical Regression  - 0.7982062780269058
2. Linear Support Vector Machine - 0.7847533632286996
3. Naive Bayes - 0.7847533632286996
'''

'''
Additional Info

model & scores with & without feature scaling

no feature scaling
log reg - 0.7982062780269058
k nearest 0.7443946188340808
svm -  0.7847533632286996
kernel svm -  0.5695067264573991
naive bayes - 0.7847533632286996
decision tree - 0.7802690582959642
random forest - 0.7443946188340808

w/feature scaling
log reg - 0.8026905829596412
k nearest 0.7802690582959642
svm -  0.7847533632286996
kernel svm -  0.7847533632286996
naive bayes - 0.7847533632286996
decision tree - 0.7802690582959642
random forest - 0.7399103139013453


'''