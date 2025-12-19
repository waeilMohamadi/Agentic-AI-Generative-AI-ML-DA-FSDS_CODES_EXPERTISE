# Import core numerical and data handling libraries
import numpy as np                  # For numerical operations
import pandas as pd                 # For data manipulation and analysis
import matplotlib.pyplot as plt     # For data visualization
import pickle                       # For saving (serializing) the trained model

# Load the dataset from CSV file
dataset = pd.read_csv(r'C:\Users\waeil\OneDrive\Desktop\ML_AI\Salary_Data.csv')

# Separate independent variable (X) and dependent variable (y)
# X = YearsExperience
# y = Salary
x = dataset.iloc[:, :-1]            # All rows, all columns except last
y = dataset.iloc[:, -1]             # All rows, only last column

# Split the dataset into training and testing sets
# 80% training data, 20% testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# Import Linear Regression model
from sklearn.linear_model import LinearRegression

# Create the Linear Regression object
regressor = LinearRegression()

# Train (fit) the model using training data
regressor.fit(x_train, y_train)

# Predict salaries using the test data
y_pred = regressor.predict(x_test)

# Compare actual vs predicted salary values
comparision = pd.DataFrame({
    'Actual': y_test,
    'Prediction': y_pred
})
print(comparision)

# Visualize the test data points
plt.scatter(x_test, y_test, color='red')

# Plot the regression line (trained on training data)
plt.plot(x_train, regressor.predict(x_train), color='blue')

# Add chart title and labels
plt.title('Salary of employee based on experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# -----------------------------------
# Model validation / future prediction
# -----------------------------------

# Intercept (c) of the regression line (salary when experience = 0)
c_inter = regressor.intercept_
print(f'Intercept: {c_inter}')

# Coefficient (m) of the regression line (salary increase per year)
m_coef = regressor.coef_
print(f'Coefficient: {m_coef}')

# Predict salary for 12 years of experience using formula: y = mx + c
y_12 = m_coef * 12 + c_inter
print("Predicted salary for 12 years:", y_12)

# Predict salary for 20 years of experience
y_20 = m_coef * 20 + c_inter
print("Predicted salary for 20 years:", y_20)

# -----------------------------------
# Model performance (Bias & Variance)
# -----------------------------------

# R² score on training data (bias)
bias_training = regressor.score(x_train, y_train)
print("Training R² (Bias):", bias_training)

# R² score on testing data (variance)
variance_testing = regressor.score(x_test, y_test)
print("Testing R² (Variance):", variance_testing)

# -----------------------------------
# Descriptive statistics
# -----------------------------------

# Mean
dataset.mean()
dataset['Salary'].mean()
dataset['YearsExperience'].mean()

# Median
dataset.median()
dataset['Salary'].median()
dataset['YearsExperience'].median()

# Variance
dataset.var()
dataset['Salary'].var()
dataset['YearsExperience'].var()

# Standard deviation
dataset.std()
dataset['Salary'].std()
dataset['YearsExperience'].std()

# Coefficient of variation
from scipy.stats import variation
variation(dataset.values)
variation(dataset['Salary'])
variation(dataset['YearsExperience'])

# Correlation
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])
dataset['Salary'].corr(dataset['Salary'])  # Always 1

# Skewness
dataset.skew()

# Standard error of mean
dataset.sem()

# -----------------------------------
# Z-score normalization
# -----------------------------------

import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])
stats.zscore(dataset['YearsExperience'])

# -----------------------------------
# ANOVA (Analysis of Variance)
# -----------------------------------

# Mean of dependent variable
y_mean = np.mean(y)

# Regression Sum of Squares (SSR)
SSR = np.sum((y_pred - y_mean) ** 2)
print("SSR:", SSR)

# Limit y values to match y_pred size
y = y[0:6]

# Sum of Squared Errors (SSE)
SSE = np.sum((y - y_pred) ** 2)
print("SSE:", SSE)

# Total Sum of Squares (SST)
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values - mean_total) ** 2)
print("SST:", SST)

# Calculate R² manually
r_square = 1 - (SSR / SST)
print("R² (manual):", r_square)

# Compare R² values
print("Training R²:", bias_training)
print("Testing R²:", variance_testing)

# -----------------------------------
# Save the trained model (Pickling)
# -----------------------------------

filename = 'linear_regression_model.pkl'

# Save model to disk
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)

print("Model has been pickled and saved as linear_regression_model.pkl")

# Print current working directory
import os
print(os.getcwd())

# ML Developer 🚀
