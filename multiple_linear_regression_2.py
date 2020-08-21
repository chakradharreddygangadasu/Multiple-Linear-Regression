#Multiple linear regression 

""" In this case, the dataset consists of 4 independent variables that includes R&D Spend, Administration
Marketing Spend, state and 1 dependent variable, profit. The goal is to create multiple linear regression model 
to predict the profits with optimum independent variable selection through backward elimination."""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

#dealing with categorical variables by creating Dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x.iloc[:, 3] = labelencoder_x.fit_transform(x.iloc[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()
x = pd.DataFrame(x)

#avoiding the dummy variable trap
x = x.iloc[:,1:]

#splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#creating the multiple linear regression model with all the independent variables
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

#predictions using the test data
y_pred = regression.predict(x_test)

#Creating the optimum model with backward elemination method

import statsmodels.api as sm 
x = np.append(arr = np.ones((50,1)).astype(int),values = x,axis = 1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

# select the variable with hignest p-value(the p-value is greater than the desired significant level)

x_opt = x[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,3,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:,[0,3]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

# Here p-values of al the variables is less than the desired significant level(sl =0.05)
# Therefor we pick these independent variables as the optimum variables

x_train_opt, x_test_opt, y_train_opt, y_test_opt = train_test_split(x_opt, y, test_size = 0.2, random_state =0)
regressor_opt = LinearRegression()
regressor_opt.fit(x_train_opt, y_train_opt)

y_pred_opt = regressor_opt.predict(x_test_opt)



