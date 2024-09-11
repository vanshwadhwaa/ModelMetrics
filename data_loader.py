import pandas as pd
import logging

# Set up logging
logging.basicConfig(filename='data_loader.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    try:
        # Check the file extension and load the data accordingly
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        elif file_path.endswith('.xls'):
            df = pd.read_excel(file_path, engine='xlrd')
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file extension. Please use .xls, .xlsx, or .csv files.")
        logging.info("Data loaded successfully from '%s'. Shape: %s", file_path, df.shape)
        return df
    except Exception as e:
        logging.error("Error loading file '%s': %s", file_path, str(e))
        raise

def validate_columns(df, x_columns, y_column):
    try:
        missing_columns = [col for col in x_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in the data.")
        if y_column not in df.columns:
            raise ValueError(f"Column '{y_column}' not found in the data.")
        logging.info("Columns validated successfully.")
        return df[x_columns + [y_column]]
    except Exception as e:
        logging.error("Error validating columns: %s", str(e))
        raise
