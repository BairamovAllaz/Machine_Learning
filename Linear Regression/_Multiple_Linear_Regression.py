#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

X = np.array([
    [1, 230, 38, 69],  
    [1, 44, 39, 45],  
    [1, 17, 46, 69],  
    [1, 151, 41, 58],
    [1, 180, 11, 58]  
])

y = np.array([22, 10, 9, 18, 13]);

X_Transpose = X.T;  # Transpose of X
X_Transpose_X = np.dot(X_Transpose,X); #X^T X
X_Transpose_X_inv = np.linalg.inv(X_Transpose_X);# (X^T X)^-1
X_transpose_y = np.dot(X_Transpose, y)  # X^T y
beta = np.dot(X_Transpose_X_inv, X_transpose_y)  # β = (X^T X)^-1 X^T y
print(beta);

print("Regression Coefficients (β):")
print(f"Intercept (β0): {beta[0]:.4f}")
print(f"TV (β1): {beta[1]:.4f}")
print(f"Radio (β2): {beta[2]:.4f}")
print(f"Newspaper (β3): {beta[3]:.4f}")


y_pred = np.dot(X, beta)  # ŷ = X β
print(y_pred);

residuals = y - y_pred;
print(f"Residuals {residuals}");

# Residual Sum of Squares (RSS)
RSS = np.sum(residuals**2);

print(f"\nResidual Sum of Squares (RSS): {RSS:.4f}");


data_to_predict = np.array([
    [1, 200, 40, 70], 
]);

predicted = np.dot(data_to_predict,beta);
print("\nPredicted Revenue for New Data:")
print(predicted)


print(" ======  ");

#Deciding on Important Variables using forward_selection


def forward_selection(X, y):
    predictors = []
    remaining_predictors = list(range(1, X.shape[1]))  # Exclude intercept (column 0)
    best_rss = np.inf

    while remaining_predictors:
        best_predictor = None
        for predictor in remaining_predictors:
            model_predictors = [0] + predictors + [predictor]  # Include intercept
            X_subset = X[:, model_predictors]
            beta_subset = np.dot(np.linalg.inv(np.dot(X_subset.T, X_subset)), np.dot(X_subset.T, y))  # β = (X^T X)^-1 X^T y
            y_pred_subset = np.dot(X_subset, beta_subset)
            rss = np.sum((y - y_pred_subset)**2)  # Residual Sum of Squares (RSS)
            if rss < best_rss:
                best_rss = rss
                best_predictor = predictor

        # Add the best predictor to the model
        if best_predictor is not None:
            predictors.append(best_predictor)
            remaining_predictors.remove(best_predictor)
        else:
            break  # Stop if no predictor improves the model

    return predictors
    
print(forward_selection(X,y));


print("-----------");


#Or another way scikit-learn  libary provides built-in functionality for forward selection and other feature selection methods.

model = LinearRegression()
sfs = SequentialFeatureSelector(
        estimator=model,
        n_features_to_select=2,
        direction='forward' #forward selection
    )

sfs.fit(X, y);
selected_features = sfs.get_support(indices=True)
print("Selected Features (Column Indices):", selected_features)


# In[ ]:




