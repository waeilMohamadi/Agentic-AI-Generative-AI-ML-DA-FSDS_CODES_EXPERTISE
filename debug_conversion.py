
import pandas as pd
import numpy as np
import re

def convert_to_numeric(df):
    """Attempts to convert object columns to numeric by removing non-numeric characters."""
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"Converting column: {col}")
            try:
                # 1. Remove commas
                cleaned = df[col].astype(str).str.replace(',', '', regex=False)
                
                # 2. Extract first valid number
                extracted = cleaned.str.extract(r'([-+]?\d*\.?\d+)', expand=False)
                
                print(f"Sample extracted values for {col}: {extracted.dropna().head().tolist()}")

                # Coerce to numeric
                converted = pd.to_numeric(extracted, errors='coerce')
                
                valid_count = converted.notna().sum()
                print(f"Valid numeric values in {col}: {valid_count} / {len(df)}")

                if valid_count > 0:
                     df[col] = converted
                     print(f"-> Successfully converted {col} to numeric.")
                else:
                     print(f"-> Failed to convert {col} (all NaNs).")
            except Exception as e:
                print(f"Error converting {col}: {e}")
    return df

try:
    file_path = "c:/Users/waeil/OneDrive/Desktop/ANTIGRAVITY/MyResume_Project/archive/uranium_production_summary_us.csv"
    df = pd.read_csv(file_path)
    
    print("Initial dtypes:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
    
    # Simulate App Logic
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"\nInitial Numeric Cols: {numeric_cols}")
    
    if len(numeric_cols) < 2:
        print("\nTriggering conversion...")
        df = convert_to_numeric(df)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        print(f"Post-Conversion Numeric Cols: {numeric_cols}")
        for col in df.columns:
            print(f"Final Type {col}: {df[col].dtype}")

except Exception as e:
    print(f"Script Error: {e}")
