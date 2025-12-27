import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file):
    """Loads CSV data into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def show_stats(df):
    """Returns descriptive statistics of the DataFrame."""
    return df.describe()

def show_missing_values(df):
    """Returns numeric count of missing values."""
    return df.isnull().sum()

def plot_correlation(df):
    """Plots a correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if numeric_df.empty:
        st.warning("No numeric columns found for correlation plot.")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    return fig

def plot_distribution(df, column):
    """Plots distribution (histogram + KDE) for a specific column."""
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    return fig

def plot_pairplot(df):
    """Plots pairplot for numeric columns."""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if numeric_df.empty:
        return None
    fig = sns.pairplot(numeric_df)
    return fig

 # Handling Missing Values
def impute_missing_values(df, column, strategy):
    """Imputes missing values in a column using the specified strategy."""
    if strategy == "Mean":
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())
    elif strategy == "Median":
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
    elif strategy == "Mode":
        if not df[column].mode().empty:
            df[column] = df[column].fillna(df[column].mode()[0])
    return df

def impute_all_missing_values(df, strategy):
    """Imputes missing values in all applicable columns using the specified strategy."""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if strategy in ["Mean", "Median"] and col not in numeric_cols:
                continue # Skip non-numeric for mean/median
            
            impute_missing_values(df, col, strategy)
    return df

def convert_to_numeric(df):
    """Attempts to convert object columns to numeric by removing non-numeric characters."""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # 1. Remove commas (common in numbers like 1,000)
                cleaned = df[col].astype(str).str.replace(',', '', regex=False)
                
                # 2. Extract the first valid number pattern (integer or float, positive or negative)
                # Pattern explains: Optional -/+, digits, optional part (dot + digits)
                extracted = cleaned.str.extract(r'([-+]?\d*\.?\d+)', expand=False)
                
                # Coerce to numeric
                converted = pd.to_numeric(extracted, errors='coerce')
                
                if converted.notna().sum() > 0:
                     df[col] = converted
            except Exception:
                pass
    return df
