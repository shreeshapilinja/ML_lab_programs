import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load housing data
def load_housing_data():
    csv_path = "housing.csv"  # Specify the path to the housing.csv file
    return pd.read_csv(csv_path)

housing = load_housing_data()

# Data Cleaning: Identify and delete duplicate rows
housing.drop_duplicates(inplace=True)

# Data Integration: Identify and delete columns with a single value
columns_to_drop = []
for column in housing.columns:
    if housing[column].nunique() == 1:
        columns_to_drop.append(column)

housing.drop(columns_to_drop, axis=1, inplace=True)

# Data Transformation: Preprocessing pipeline
def preprocess_data(data):
    # Separate numerical and categorical features
    num_attribs = list(data.select_dtypes(include=['float64', 'int64']))
    cat_attribs = list(data.select_dtypes(include=['object']))

    # Numerical pipeline: impute missing values and scale features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    # Categorical pipeline: encode categorical features
    cat_pipeline = Pipeline([
        ('ordinal_encoder', OrdinalEncoder()),
        ('one_hot_encoder', OneHotEncoder()),
    ])

    # Full pipeline: apply transformations to all features
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    # Apply preprocessing pipeline to the data
    processed_data = full_pipeline.fit_transform(data)

    return processed_data

# Preprocess housing data
housing_preprocessed = preprocess_data(housing)