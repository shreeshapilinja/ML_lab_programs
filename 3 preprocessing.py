import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
housing = pd.read_csv('housing.csv')

# Data Cleaning - Identifying and Deleting Duplicate Rows
housing_cleaned = housing.drop_duplicates()
print("Number of rows after removing duplicates:", len(housing_cleaned))

# Data Integration - Identifying and Deleting Columns with a Single Value
columns_to_drop = []
for column in housing_cleaned.columns:
    if housing_cleaned[column].nunique() == 1:
        columns_to_drop.append(column)
housing_integrated = housing_cleaned.drop(columns=columns_to_drop)
print("Columns after removing single-valued columns:")
print(housing_integrated.columns)

# Data Transformation - Preprocessing Pipeline
num_attribs = list(housing_integrated.select_dtypes(include=['float64', 'int64']))
cat_attribs = list(housing_integrated.select_dtypes(include=['object']))
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
    ('ordinal_encoder', OrdinalEncoder()),
    ('one_hot_encoder', OneHotEncoder()),
])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
housing_preprocessed = full_pipeline.fit_transform(housing_integrated)
print("Preprocessed data shape:", housing_preprocessed.shape)