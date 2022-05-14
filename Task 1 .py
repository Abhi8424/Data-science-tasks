#!/usr/bin/env python
# coding: utf-8

# # Name - Abhishek Poojari 
# # Task 1 - Prediction Using Supervised ML
# # GripMay22 - Data science and Business Analytics

# In[7]:


# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# # Step 1 - Reading the data

# In[2]:


df=pd.read_csv("http://bit.ly/w-data")
print("Data imported successfully")
df.head(10)


# In[3]:


df.info()


# In[4]:


# Statistical information about data

df.describe()


# # Step 2 - Input Data visualization

# In[5]:


# Plotting the distribution of scores

plt.scatter(df.Hours,df.Scores)
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Score")
plt.title("Hours Vs Percenatge")
plt.show()


# positive relationship between no. of hours studied and Percentage Score.

# In[6]:


df.corr()


# # Step 3 - Data preprocessing

# In[8]:


x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


# # Step 4 - Model Training

# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
regressor = LinearRegression()
regressor.fit(x_train, y_train)

print("Training Complete")


# # Step 5 - Plotting The Line Of Regression

# In[12]:


# Plotting the regression line

line = regressor.intercept_ + regressor.coef_*x
plt.plot(x, line, color = 'pink')

# Plotting for the test data

plt.scatter(x,y)
plt.show()


# # Step 6 - Making Predictions

# In[13]:


# Testing the data
print(x_test)

# Model Prediction
y_pred = regressor.predict(x_test)


# # Step 7 - Comparing Actual Result to the Predicted Model Result

# In[14]:


# Comparing the Actual Vs Predicted

actual_predicted = pd.DataFrame({'Actual':y_test , 'Predicted' :y_pred})
actual_predicted


# In[15]:


# Estimating training and test score

print("Training Score:" , regressor.score(x_train,y_train))
print("Testing Score:", regressor.score(x_test,y_test))


# In[16]:


# Accuracy Of The Model

regressor.score(x_test, y_test)


# # Step 8 - Making Prediction about Scores for studying different no. of hours

# In[17]:


hours= 9.25
own_pred = regressor.predict([[hours]])
print("No. Of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred))


# # Step 9 - Evaluate The Model

# In[18]:


from sklearn import metrics
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))


#  CONCLUSION : If a student studies 9.25 hours , he will score 90.58 marks.
