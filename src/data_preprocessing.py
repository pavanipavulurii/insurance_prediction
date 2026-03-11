# 1. Load raw data
# 2. Identify X (inputs) and y (output)
# 3. Split data into train and test

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data():

    # Load dataset
    data = pd.read_csv("../data/raw/insurance_data.csv")

    # Input features
    X = data[['Age', 'Annual_Income_LPA', 'Policy_Term_Years', 'Sum_Assured_Lakhs']]

    # Target variable
    y = data['Annual_Premium_Thousands']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test