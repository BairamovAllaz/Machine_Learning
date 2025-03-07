#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pip', 'install --upgrade scikit-learn')


# In[31]:


from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pylab as plt


# In[36]:


X, y = fetch_california_housing(return_X_y=True)

df = pd.DataFrame(X, columns=[
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
    "Population", "AveOccup", "Latitude", "Longitude"
])
df["Target"] = y  # Adding the target column
print(df.head())
print(fetch_california_housing()['DESCR'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("scale",StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=1))
])

pipe.get_params(); 

mod = GridSearchCV(estimator=pipe,
            param_grid={'model__n_neighbors' : [1,2,3,4,5,6,7,8,9,10]},
             cv=3)
mod.fit(X_train,y_train);

pd.DataFrame(mod.cv_results_);


print("/b")

#An simple example of this code

import numpy as np

# Example data: Features and target variable
X = np.array([
    [1500, 3, 10, 1],
    [1800, 4, 5, 0.5],
    [1200, 2, 20, 3],
    [2000, 4, 15, 1.5],
    [1600, 3, 8, 2]
])

y = np.array([400, 500, 350, 550, 420])  # Price in $1000

# Step 1: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Create a pipeline with scaling and KNN regression
pipe = Pipeline([
    ("scale", StandardScaler()),  # Standardize the features
    ("model", KNeighborsRegressor(n_neighbors=2))  # Apply KNN regression with 2 neighbors
])

# Step 3: Fit the model to the training data
pipe.fit(X_train, y_train)

# Step 4: Make predictions on the test set
predictions = pipe.predict(X_test)

# Step 5: Display the predictions
print("Predicted Prices:", predictions)


print("Model Accuracy on Test Data:", pipe.score(X_test, y_test))

# Plotting: Predicted vs Actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('KNN Regression: Actual vs Predicted Prices')
plt.show()

