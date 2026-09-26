
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np # Added numpy import

DATA_PATH = "tourism_project/data/tourism.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)
print(f"Original dataset shape: {df.shape}")

# Remove unnecessary columns (e.g., CustomerID is an identifier)
columns_to_drop = ['CustomerID']
df = df.drop(columns=columns_to_drop, errors='ignore') # Operate directly on df
print(f"Shape after dropping CustomerID: {df.shape}")

# Handle specific data quality issues (e.g., "Fe Male" should be "Female")
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].str.strip().replace({'Fe Male': 'Female', 'Fe male': 'Female'})

# Handle missing values
print("\nHandling missing values...")

# Identify numerical and categorical columns for imputation
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

# Impute missing values for numerical columns with the median
for col in numerical_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Imputed numerical column '{col}' with median: {median_val}")

# Impute missing values for categorical columns with the mode
for col in categorical_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"Imputed categorical column '{col}' with mode: {mode_val}")

print(f"Shape after imputation: {df.shape}")

# Separate features (X) and target (y)
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

# Split the data into training and testing sets
# Using stratify=y for classification tasks to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the split datasets locally as CSV files
X_train.to_csv('Xtrain.csv', index=False)
X_test.to_csv('Xtest.csv', index=False)
y_train.to_csv('ytrain.csv', index=False)
y_test.to_csv('ytest.csv', index=False)

print("Data loaded, processed, split, and saved successfully:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
