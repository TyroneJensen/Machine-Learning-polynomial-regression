#This program uses polynomial regression to train, predict and plot the relationship between temperature and humidty
#
#Author: T.Jensen
#
#Background:
#Monitoring of temperature and humidty (amount of water vapour in the air)is essential for growers espeacilly in indoor environments.
#Maintaining favourable environments significantly enhances plant growth, health and resistance to disease.
#The general relationship(in a vacuum) is an increase in temperature will decrease humidity.
# 
#Method of data acquisition:
#The data (which is my own) is the log records of a wireless temperature and humidity sensor in an outdoor greenhouse.
#Temperature and humidity readings were taken every 5 minutes for a 24 hour period.
#The logged records were downloaded in .csv file format
#
#Outcomes:
#-to train data to fit a linear and polynomial regression model
#-to visually plot the relationship between two environmental variable
#-to predict humidity as a function of temperature
#
#Conclusion:
#Given a controlled environment and enough data, one could accurately predict (with Polynomial regression) the humidity using temperature  

#import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#import dataset
dataset = pd.read_csv('sensorReading1.csv')

#clean-up data, remove all NaN values
dataset.fillna(method ='ffill', inplace = True)

#slice data to select columns of interest
X1= dataset.iloc[:,2] #temp
y1= dataset.iloc[:,3] #humidity

#reshape data by adding another dimension (1D to 2D) 
X = X1 [:,np. newaxis]
y = y1[:,np. newaxis]

# Fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X, y, color='b')    #plot data
    plt.plot(X, lin_reg.predict(X), color='r')  #plot regression line
    plt.title('Linear Regression results')  
    plt.xlabel('Temperature (C)')
    plt.ylabel('Humidity (%)')
    plt.show()
    return

viz_linear()

# Set the degree of the Polynomial Regression model
poly_reg = PolynomialFeatures(degree=2)
# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_poly = poly_reg.fit_transform(X)

# Fitting Polynomial Regression to the dataset
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='b')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='r')
    plt.title('Polynomial Regression results')
    plt.xlabel('Temperature (C)')
    plt.ylabel('Humidity (%)')
    plt.show()
    return

viz_polymonial()

# Predicting a new result with Linear Regression
print("Using Linear regression the humidity (%) predicted at 25C,28C, 30C respectively is: ")
print(lin_reg.predict([[25]]))
print(lin_reg.predict([[28]]))
print(lin_reg.predict([[30]]))

print()

# Predicting a new result with Polymonial Regression
print("Using Polynomial regression the humidity (%) predicted at 25C,28C, 30C respectively is: ")
print(pol_reg.predict(poly_reg.fit_transform([[25]])))
print(pol_reg.predict(poly_reg.fit_transform([[28]])))
print(pol_reg.predict(poly_reg.fit_transform([[30]])))
