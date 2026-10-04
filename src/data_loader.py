import pandas as pd
import os


# Location of our dataset
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "dataset.csv"
)


# Load dataset
df = pd.read_csv(DATA_PATH)


# Display basic information
print("====================================")
print("   STUDENT DROPOUT RISK PROJECT")
print("====================================")

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nTarget Distribution:")
print(df["Target"].value_counts())

print("\nTarget Percentage:")
print(df["Target"].value_counts(normalize=True) * 100)

print("\n========== FINAL DATA CHECK ==========")

print("\n1. Dataset Shape:")
print(df.shape)

print("\n2. Missing Values:")
print(df.isnull().sum().sum())

print("\n3. Duplicate Records:")
print(df.duplicated().sum())