import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"D:\FSDS\Multiple Linear Regression Model\Salary_Data.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #regressor - model, linerarregression- Algorithm
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

comparision = pd.DataFrame({'Actual' : y_test, 'Prediction': y_pred})
print(comparision)

plt.scatter(x_test, y_test, color = 'Blue')
plt.plot(x_train, regressor.predict(x_train), color = 'Red')
plt.title('Salary of Employee Based On Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Predict The Future
m_coef = regressor.coef_
print(m_coef)


c_intercept = regressor.intercept_
print(c_intercept)


y_12 = m_coef * 12 + c_intercept
print(y_12)

y_20 = m_coef * 20 + c_intercept
print(y_20)

bias = regressor.score(x_train,y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

#Stats Implementation to the code


#mean, Median and mode
dataset.mean()

dataset['Salary'].mean()

dataset.median()

dataset['Salary'].median()

dataset['YearsExperience'].mean()

dataset['YearsExperience'].median()

#Variance
dataset.var()

dataset['Salary'].var()

dataset['YearsExperience'].var()

#Standard Deviation
dataset.std()

dataset['Salary'].std()

dataset['YearsExperience'].std()

# COFFIENT OF VARIATION

from scipy.stats import variation

variation(dataset.values) 

variation(dataset['Salary'])

#CORRELATION

dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])


#Skewness

dataset.skew()

dataset['Salary'].skew()

#Standard Error

#dataset.se

#z-Scroe

import scipy.stats as stats

dataset.apply(stats.zscore) #this will give z-score of entrie dataframe

#stats.dataset['Salary']

#ssr
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#sse
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

#sst
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#r2
r_square = 1 - SSR/SST
print(r_square)


import pickle
#Save the trained model ti disk
filename = 'Regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been picked and saved as Regression_model.pkl")





