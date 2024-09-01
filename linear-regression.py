import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression101:
    def __init__(self):
        self.coefficient_ = 0
        self.intercept_ = 0
    def fit(X, y):
        # work in progress
        pass
    def predict(X):
        # work in progress
        pass

# Importing the dataset 
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

regressor = LinearRegression101()

''' We are not concerned about the training and test set.
    just building a LinearRegression() class based on the math cocept linear curve fitting. '''

regressor.fit(X,y)                 #let's train

y_pred = regressor.predict(X)
print(y_pred)

# let's print the coefficient_ and intercept_

print(regressor.coefficient_)
print(regressor.intercept_)

# let's Plot

plt.scatter(X, y, color = "red")
plt.plot(X, regressor.predict(X), color = "blue")
plt.title("Linear Regression YoE vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

