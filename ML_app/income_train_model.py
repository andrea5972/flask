"""
Last Updated: Dec 12, 2018
Relative File Path: app/income_train_model.py
Description: Classification of the census adult income dataset
Dataset:number of attributes (Columns): 15
        number of instances (Rows): 32560
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib


# Importing the data itself
data_set  = pd.read_csv('adult.csv')

data_set.columns = ["age", "workclass", "fnlwgt", "education", "education-num",
                "marital-status", "occupation", "relationship", "race",
                "sex", "capital-gain", "capital-loss", "hours-per-week",
                "native-country", "salary"]

data_set = data_set.drop(['fnlwgt', 'education', 'race','native-country', 'workclass','capital-gain', 'occupation',
                    'capital-loss','hours-per-week', 'relationship'], axis=1)
# data_set contains:
        # age               32561 non-null int64
        # education-num     32561 non-null int64
        # marital.status    32561 non-null object
        # sex               32561 non-null object

# Replace categorical data
data_set['salary'] = data_set['salary'].map({'<=50K': 0, '>50K': 1}).astype(int)

data_set['sex'] = data_set['sex'].map({'Male': 0, 'Female': 1}).astype(int)

data_set['marital-status'] = data_set['marital-status'].map(
                            {'Married-spouse-absent': 0,
                            'Widowed': 1, 'Married-civ-spouse': 2,
                            'Separated': 3, 'Divorced': 4,
                            'Never-married': 5,
                            'Married-AF-spouse': 6}).astype(int)



# Only the features that are important
X = data_set[['sex', 'marital-status', 'education-num', 'age']]
# Taking the labels (Income)
y = data_set['salary']


# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber',
    random_state=0
)
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it in other programs
joblib.dump(model, 'classifier_model.pkl')

# Find the error rate on the training set
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)
