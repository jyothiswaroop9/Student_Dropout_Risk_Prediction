
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# DATASET PATH
# --------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "dataset.csv"
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("Target", axis=1)

y = df["Target"]


# --------------------------------------------------
# DISPLAY INFORMATION
# --------------------------------------------------

print("\n==========================================")
print("       DATA PREPROCESSING")
print("==========================================")

print("\nFeature Data (X):")
print("Rows:", X.shape[0])
print("Columns:", X.shape[1])

print("\nTarget Data (y):")
print("Rows:", y.shape[0])

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Classes:")
print(y.unique())

# --------------------------------------------------
# CATEGORICAL AND NUMERICAL FEATURES
# --------------------------------------------------

categorical_features = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]


numerical_features = [
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]


print("\nCategorical Features:")
print(len(categorical_features))
print(categorical_features)

print("\nNumerical Features:")
print(len(numerical_features))
print(numerical_features)

# --------------------------------------------------
# ENCODE TARGET
# --------------------------------------------------

target_mapping = {
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2
}

y_encoded = y.map(target_mapping)


print("\nTarget Encoding:")
print(target_mapping)

print("\nEncoded Target Values:")
print(y_encoded.value_counts().sort_index())


# --------------------------------------------------
# ONE-HOT ENCODING
# --------------------------------------------------

X_encoded = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=False
)


print("\nOriginal Feature Count:")
print(X.shape[1])

print("\nEncoded Feature Count:")
print(X_encoded.shape[1])

print("\nEncoded Dataset Shape:")
print(X_encoded.shape)

# --------------------------------------------------
# SCALE NUMERICAL FEATURES
# --------------------------------------------------
# --------------------------------------------------
# SCALE NUMERICAL FEATURES
# --------------------------------------------------
# --------------------------------------------------
# SCALE NUMERICAL FEATURES
# --------------------------------------------------

# --------------------------------------------------
# SCALE NUMERICAL FEATURES
# --------------------------------------------------

scaler = StandardScaler()

X_encoded[numerical_features] = scaler.fit_transform(
    X_encoded[numerical_features]
)

print("\nNumerical Features Scaled Successfully.")

print("\nFinal Encoded Dataset Shape:")
print(X_encoded.shape)

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\n==========================================")
print("          TRAIN / TEST SPLIT")
print("==========================================")

print("\nTraining Features:")
print(X_train.shape)

print("\nTesting Features:")
print(X_test.shape)

print("\nTraining Target:")
print(y_train.shape)

print("\nTesting Target:")
print(y_test.shape)