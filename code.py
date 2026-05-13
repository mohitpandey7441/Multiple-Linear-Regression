import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Read the dataset
dataset = pd.read_csv(r"C:\AVSCODE\ML_Folder\Multiple Linear Regression\Investment.csv")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]


x = pd.get_dummies(x, dtype = int)

from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


#== we build ml model 

# y = mx + c 

m = regressor.coef_
print(m)

# if constant is missing how to get constant
c = regressor.intercept_
print(c)

# How to add constant in dataFrame

x = np.append(arr=np.full((50,1), 42467).astype(int), values=x, axis=1)

import statsmodels.api as sm
x_opt = x[:,[0, 1, 2, 3, 4, 5]]


# OrdinaryLeast Squares

regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()















