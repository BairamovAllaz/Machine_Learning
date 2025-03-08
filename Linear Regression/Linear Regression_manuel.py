#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


# In[ ]:





# In[2]:


X = np.array([1,2,3,4,5]);
Y = np.array([2,4,5,4,6]);


# In[3]:


mean_X = np.mean(X);
mean_Y = np.mean(Y);


# In[4]:


numerator = np.sum((X - mean_X) * (Y - mean_Y));
denominator = np.sum((X - mean_X) ** 2);
beta_1 = numerator / denominator #slope
beta_0 = mean_Y - (beta_1 * mean_X); #Intercept

print(f"Intercept (beta 0): {beta_0:.2f}");
print(f"Slope (beta 1): {beta_1:.2f}");


# In[5]:


#make predictions

Y_prediction = beta_0 + (beta_1 * X);


# In[7]:


residuals = Y - Y_prediction;
RSS = np.sum(residuals**2);
print(f"\nResidual Sum of Squares (RSS): {RSS:.2f}");


# In[8]:


n = len(Y); #number of observation
RSE = np.sqrt(RSS/(n-2)); ## Residual Standard Error
print(f"Residual Standard Error (RSE): {RSE:.2f}")


# In[15]:


sigma_squared = RSS / (n-2);
se_beta_1 = np.sqrt(sigma_squared / denominator)
se_beta_0 = np.sqrt(sigma_squared * (1/n + mean_X**2 / denominator))

print(f"\nStandard Error of beta_0: {se_beta_0:.2f}")
print(f"Standard Error of beta_1: {se_beta_1:.2f}")


print("\nRegression Equation:")
print(f"Y = {beta_0:.2f} + {beta_1:.2f} * X")


# In[19]:


alpha = 0.05  # Significance level
df = n - 2  # Degrees of freedom
t_critical = t.ppf(1 - alpha/2, df)  # Critical t-value

ci_beta_0 = (beta_0 - t_critical * se_beta_0, beta_0 + t_critical * se_beta_0)
ci_beta_1 = (beta_1 - t_critical * se_beta_1, beta_1 + t_critical * se_beta_1)

print(f"\n95% Confidence Interval for beta_0: ({ci_beta_0[0]:.2f}, {ci_beta_0[1]:.2f})")
print(f"95% Confidence Interval for beta_1: ({ci_beta_1[0]:.2f}, {ci_beta_1[1]:.2f})")


# In[16]:


t_beta_0 = beta_0 / se_beta_0  # t-statistic for beta_0
t_beta_1 = beta_1 / se_beta_1  # t-statistic for beta_1


print(f"\nHypothesis Testing:");
print(f"t-statistic for beta_0: {t_beta_0:.2f}");
print(f"t-statistic for beta_1: {t_beta_1:.2f}");


# In[20]:


if t_beta_1 > t_critical:
    print("95% meaningful realtion between X and Y");
else:
    print("Not enough realtion between X and Y");


# In[13]:


#Ploting data points

plt.scatter(X,Y,color = "blue", label="Data points");
plt.plot(X, Y_prediction, color="red", label="Regression line")

plt.xlabel("X")
plt.ylabel("Y")

plt.title("Linear Regression from Scratch")
plt.legend()

