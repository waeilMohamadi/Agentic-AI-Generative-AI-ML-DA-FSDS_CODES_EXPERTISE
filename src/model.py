import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_linear_regression(X, y):
    """Trains a simple linear regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_polynomial_regression(X, y, degree=2):
    """Trains a polynomial regression model."""
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    return model, poly_features

def train_knn_regression(X, y, k=5):
    """Trains a KNN regression model."""
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, y)
    return model

def train_random_forest_regression(X, y, n_estimators=100):
    """Trains a Random Forest regression model."""
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y, poly_features=None):
    """Evaluates the model and returns MSE and R2."""
    if poly_features:
        X_pred = poly_features.transform(X)
    else:
        X_pred = X
        
    y_pred = model.predict(X_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, r2, y_pred

def generate_model_explanation(model, model_type, feature_col, target_col):
    """Generates a text explanation of the model."""
    explanation = f"### Model Explanation: {model_type}\n\n"
    
    if model_type == "Linear Regression":
        if hasattr(feature_col, '__iter__') and not isinstance(feature_col, str):
            # Multiple features
            coefs = model.coef_
            intercept = model.intercept_
            explanation += f"**Equation:** `{target_col} = "
            terms = [f"({c:.4f} * {f})" for c, f in zip(coefs, feature_col)]
            explanation += " + ".join(terms) + f" + {intercept:.4f}`\n\n"
            explanation += "**Interpretation:** In multiple linear regression, each coefficient represents the change in the target variable for a one-unit change in the respective feature, *holding all other features constant*."
        else:
            # Single feature
            coef = model.coef_[0] if hasattr(model.coef_, '__len__') else model.coef_
            intercept = model.intercept_
            explanation += f"**Equation:** `{target_col} = ({coef:.4f} * {feature_col}) + {intercept:.4f}`\n\n"
            explanation += f"**Interpretation:** For every one unit increase in **{feature_col}**, the **{target_col}** "
            if coef > 0:
                explanation += f"increases by approximately **{coef:.4f}** units."
            else:
                explanation += f"decreases by approximately **{abs(coef):.4f}** units."
            
    elif model_type == "Polynomial Regression":
        explanation += "Polynomial regression captures non-linear relationships by introducing higher-degree terms of the feature variable. "
        explanation += "The interpretation is complex as it involves multiple coefficients for power terms needed to fit the curve."
        
    elif model_type == "KNN Regression":
        k = model.n_neighbors
        explanation += f"This model predicts **{target_col}** by finding the **{k}** data points (neighbors) that are most similar (closest) to the input **{feature_col}** value "
        explanation += "and calculating their average value. It does not learn a mathematical formula but relies on local data proximity."

    elif model_type == "Random Forest Regression":
        n_trees = model.n_estimators
        explanation += f"This model uses an ensemble of **{n_trees}** decision trees. "
        explanation += "Each tree learns a set of rules from a random subset of data. The final prediction is the average of all the individual tree predictions, "
        explanation += "which makes it robust against overfitting and capable of capturing complex non-linear patterns."

    return explanation

def plot_regression_results(X, y, y_pred, title="Regression Results"):
    """Plots the actual vs predicted values."""
    # Ensure inputs are 1D arrays for plotting
    if hasattr(X, 'values'):
        X_plot = X.values.flatten()
    else:
        X_plot = X.flatten()
        
    if hasattr(y, 'values'):
        y_plot = y.values.flatten()
    else:
        y_plot = y.flatten()

    # Sort for cleaner plotting if X is 1D
    if len(X.shape) == 2 and X.shape[1] == 1:
        sort_idx = np.argsort(X_plot)
        X_plot = X_plot[sort_idx]
        y_plot = y_plot[sort_idx]
        y_pred = y_pred[sort_idx]

    fig, ax = plt.subplots()
    ax.scatter(X_plot, y_plot, color='blue', label='Actual', alpha=0.6)
    ax.plot(X_plot, y_pred, color='red', label='Predicted', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.legend()
    return fig

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    """Plots actual vs predicted values for multi-feature regression."""
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, color='blue', alpha=0.6)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend()
    return fig
