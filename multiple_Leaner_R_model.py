# ======================================================
# Multiple Linear Regression with Backward Elimination
# ======================================================

# 1. Import libraries
import numpy as np                  # For numerical operations (arrays, math)
import pandas as pd                 # For data manipulation (dataframes)
import matplotlib.pyplot as plt     # For plotting (not used in this code but commonly used)

from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.linear_model import LinearRegression    # For linear regression model
import statsmodels.api as sm                          # For OLS regression and statistical summary

# 2. Load dataset
dataset = pd.read_csv(r"C:\Users\waeil\OneDrive\Desktop\ML_AI\Investment.csv")
# Reads CSV data into a pandas DataFrame. Make sure the path is correct.

# 3. Define features (X) and target (y)
X = dataset.iloc[:, :-1]   # All columns except the last one are used as features
y = dataset.iloc[:, 4]     # The 5th column (index 4) is used as the target variable

# 4. Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X, dtype=int)  
# This converts categorical columns (if any) into numeric 0/1 columns 
# so that regression can handle them.

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
# 80% training data, 20% test data. 'random_state=0' ensures reproducibility.

# 6. Fit the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)  
# Train the model on the training data

# 7. Make predictions on the test set
y_pred = regressor.predict(X_test)

# 8. Print model coefficients and intercept
m = regressor.coef_       # Coefficients of each feature
print(m)

c = regressor.intercept_  # Intercept of the regression line
print(c)

# 9. Add a column of ones for the bias term in OLS regression
X = np.append(arr=np.full((50,1), 42467).astype(int), values=X, axis=1)
# np.full((50,1), 42467) creates a column filled with 42467 instead of 1
# In OLS from statsmodels, you manually add a constant column (here filled with 42467). 
# Typically, this should be 1s, but here 42467 is used (likely intended as a placeholder for the intercept). 
# Shape of X is now (50, n_features+1). Make sure the number of rows matches your dataset.

# Anathor Way
#X = sm.add_constant(X)  # Automatically adds a column of ones


# 10. Backward Elimination using OLS
# Start with all features
X_opt = X[:, [0,1,2,3,4,5]]  # Select all columns for initial model
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()  # Fit OLS regression
regressor_OLS.summary()  # View p-values and R² to check statistical significance

# Remove least significant feature (column index 4) and refit
X_opt = X[:, [0,1,2,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Remove next least significant feature (column index 5) and refit
X_opt = X[:, [0,1,2,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Continue backward elimination (remove column index 2)
X_opt = X[:, [0,1,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Final backward elimination step (keep only columns 0 and 1)
X_opt = X[:, [0,1]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# This iterative process removes features with high p-values until only significant ones remain

# 11. Evaluate model performance
bias = regressor.score(X_train, y_train)  # R² on training data (measure of bias)
bias

variance = regressor.score(X_test, y_test)  # R² on testing data (measure of variance)
variance
