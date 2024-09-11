# Machine Learning Model Comparison with Streamlit

## Overview

This project is an interactive Streamlit application that allows users to compare machine learning models for both regression and classification tasks. It provides a user-friendly interface to upload datasets, select features, preprocess data, and visualize model performance metrics.

## Features

- **Interactive User Interface**: Upload Excel or CSV files, select features and target columns, and choose task types (regression or classification).
- **Data Preprocessing**: Handles missing values by imputing numerical columns with the mean and categorical columns with the mode. One-hot encoding is applied to categorical features.
- **Model Training**: Compares different machine learning models based on the selected task type:
  - **Regression**: Linear Regression, Decision Tree Regressor, Random Forest Regressor.
  - **Classification**: Logistic Regression, Decision Tree Classifier, Random Forest Classifier.
- **Results Visualization**: Displays model performance using bar charts to compare metrics like RMSE for regression or accuracy for classification.
- **Error Handling**: Robust error handling with user feedback and logging for any issues encountered during execution.

## Getting Started

### Prerequisites

Ensure you have Python installed (version 3.7 or higher). You will also need the following Python libraries:

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

Install the required libraries using:


pip install streamlit pandas numpy scikit-learn matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/ml-model-comparison.git
    cd ml-model-comparison
    ```

2. Run the application:
    ```bash
    streamlit run main.py
    ```

## File Structure

- **main.py**: Main file to run the Streamlit application.
- **data_loader.py**: Contains functions to validate and load data.
- **preprocessing.py**: Handles preprocessing steps like missing value imputation and one-hot encoding.
- **model_training.py**: Trains and evaluates models for regression and classification tasks.
- **utils.py**: Utility functions for plotting model performance results.

## How to Use

1. **Upload a File**: Click on the "Upload your Excel file" button to upload an Excel or CSV file.
2. **Select Features and Target**: Choose the features (X) and the target (Y) columns from the dataset.
3. **Select Task Type**: Choose between regression or classification.
4. **Run Model**: Click on "Run Model" to start the processing. The application will validate columns, preprocess the data, train models, and visualize the results.

## Example Workflow

1. Upload a dataset with features and a target column.
2. Select which columns to use as features and which one as the target.
3. Choose the type of task you want to perform (regression or classification).
4. Run the models to see a comparison of their performance.

## Logging

Logs are stored in the `main.log` file for debugging and tracking purposes. If any errors occur during processing, detailed error messages are logged.

## Contributing

If you would like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue to discuss what you would like to change.



## Contact

For any questions or issues, please contact
Vansh Wadhwa 
vanshwadhwa4802@gmail.com

```bash
