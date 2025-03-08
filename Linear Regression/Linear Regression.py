#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 4, 5, 4, 6])

model = LinearRegression()
model.fit(X, Y)


beta_0 = model.intercept_ 
beta_1 = model.coef_[0] 

print(f"Intercept (beta_0): {beta_0:.2f}")
print(f"Slope (beta_1): {beta_1:.2f}")

Y_pred = model.predict(X)

residuals = Y - Y_pred
RSS = np.sum(residuals**2)
print(f"\nResidual Sum of Squares (RSS): {RSS:.2f}")


n = len(Y)
RSE = np.sqrt(RSS / (n - 2))
print(f"Residual Standard Error (RSE): {RSE:.2f}")

MSE = mean_squared_error(Y, Y_pred)
RMSE = np.sqrt(MSE)
print(f"Mean Squared Error (MSE): {MSE:.2f}")
print(f"Root Mean Squared Error (RMSE): {RMSE:.2f}")

print("\nRegression Equation:")
print(f"Y = {beta_0:.2f} + {beta_1:.2f} * X")

