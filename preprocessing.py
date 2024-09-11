import pandas as pd
import numpy as np

def preprocess_data(df, x_columns):
    # Log initial state
    print("Initial DataFrame shape:", df.shape)
    
    # Handle missing values
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    
    if not missing_summary.empty:
        print("Missing values before treatment:")
        print(missing_summary)
    
    # Drop columns with more than 70% missing values
    threshold = 0.7
    cols_to_drop = missing_summary[missing_summary / len(df) > threshold].index
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Log columns dropped
    if not cols_to_drop.empty:
        print("Columns dropped due to more than 70% missing values:")
        print(cols_to_drop)
    
    # Impute missing values for the remaining columns
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                # Impute categorical columns with the mode
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
            else:
                # Impute numerical columns with the mean
                mean_value = df[col].mean()
                df[col].fillna(mean_value, inplace=True)
            
            # Log imputation
            print(f"Missing values in column '{col}' treated with: {df[col].mode()[0] if df[col].dtype == 'object' else mean_value}")
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Get updated list of feature columns
    updated_x_columns = [col for col in x_columns if col in df.columns]
    
    # Log final state
    print("DataFrame shape after preprocessing:", df.shape)
    
    return df, updated_x_columns
