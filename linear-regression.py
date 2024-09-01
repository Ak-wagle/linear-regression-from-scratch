import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression101:

    def __init__(self):
        self.coefficient_ = 0
        self.intercept_ = 0

    def fit(self, X, y):
        # converting y to a column vector as it's a 1D array here
        y = y.reshape(-1,1)

        # these are nothing but the summations that we use in the normal equation in linear curve fitting
        sum_X = np.sum(X)
        sum_y = np.sum(y)
        sum_X_square = np.sum(X**2)   
        sum_X_mul_y = np.sum(X*y)
        n = X.shape[0]

        # calculating the coefficient (or slope) and intercept
        self.coefficient_ = ((n * sum_X_mul_y) - (sum_X * sum_y)) / ((n * sum_X_square) - (sum_X)**2)
        self.intercept_ = (sum_y-(self.coefficient_*sum_X))/n

    def predict(self,X_input):
        # converting X_input to 2D array if it's not
        X_input = np.array(X_input)
        if X_input.ndim == 1:
            X_input = X_input.reshape(-1,1)

        # this our dear y = mx + c
        return self.coefficient_ * X_input + self.intercept_

dataset = pd.read_csv("Salary_Data.csv")      # Dataset contains years of experience and salary
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# let's create and train the model
regressor = LinearRegression101()

''' We are not concerned about the training and test set here.
    just building a LinearRegression() class based on the math cocept linear curve fitting. '''

regressor.fit(X,y)                         

# predicting a salary for a given years of experience i.e, X = 4.2
y_pred = regressor.predict([[4.2]])         
print(f"For 4.2 years of experience you may expect the salary Rs. {y_pred}")

# let's print the coefficient_ and intercept_
print(f"The coefficient value is: {regressor.coefficient_}")
print(f"The intercept value is: {regressor.intercept_}")

# let's Plot the results

plt.scatter(X, y, color = "red")
plt.plot(X, regressor.predict(X), color = "blue")
plt.title("Linear Regression YoE vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()