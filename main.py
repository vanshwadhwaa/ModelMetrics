import streamlit as st
import pandas as pd
from data_loader import validate_columns
from preprocessing import preprocess_data
from model_training import train_models
from utils import plot_results
import logging

# Set up logging
logging.basicConfig(filename='main.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    st.title("Machine Learning Model Comparison")

    # File upload
    file = st.file_uploader("Upload your Excel file", type=["xlsx", "csv"])
    if file is not None:
        # Load the file into a DataFrame
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            st.write("Preview of the uploaded file:")
            st.write(df.head())
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return

        # Display available columns
        all_columns = df.columns.tolist()

        # Checkboxes for feature columns
        st.write("Select feature columns (X):")
        x_columns = [col for col in all_columns if st.checkbox(col, value=True, key=f"x_{col}")]

        # Checkbox for target column
        st.write("Select target column (Y):")
        y_column = st.selectbox("Select target column", options=[col for col in all_columns if col not in x_columns])

        if not x_columns or not y_column:
            st.warning("Please select feature columns and a target column.")
            return

        # Select task type
        task = st.selectbox("Select task type", options=['regression', 'classification'])

        # Button to start processing
        if st.button("Run Model"):
            try:
                # Validate and select the columns
                df = validate_columns(df, x_columns, y_column)

                # Preprocess the data (handle null values and categorical variables)
                df, updated_x_columns = preprocess_data(df, x_columns)

                # Ensure y_column is present in df
                if y_column not in df.columns:
                    st.error(f"Target column '{y_column}' is not in the DataFrame after preprocessing.")
                    return

                # Separate features and target
                X = df[updated_x_columns]
                y = df[y_column]

                # Train models and get accuracy
                results = train_models(X, y, task=task)

                # Plot the results
                plot_results(results, task=task)

            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                logging.error("Error in main execution: %s", str(e))

if __name__ == "__main__":
    main()
