import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# DATASET PATH
# --------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "dataset.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("Target", axis=1)

y = df["Target"]


# --------------------------------------------------
# FEATURE GROUPS
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


# --------------------------------------------------
# ENCODE TARGET
# --------------------------------------------------

target_mapping = {
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2
}

y_encoded = y.map(target_mapping)


# --------------------------------------------------
# ONE-HOT ENCODE CATEGORICAL FEATURES
# --------------------------------------------------

X_encoded = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=False
)


# --------------------------------------------------
# SCALE NUMERICAL FEATURES
# --------------------------------------------------

scaler = StandardScaler()

X_encoded[numerical_features] = scaler.fit_transform(
    X_encoded[numerical_features]
)


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


# --------------------------------------------------
# CREATE MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

print("\n==========================================")
print("       MACHINE LEARNING MODEL TRAINING")
print("==========================================")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    print(f"{name} training completed.")

print("\n==========================================")
print("       ALL MODELS TRAINED SUCCESSFULLY")
print("==========================================")

# --------------------------------------------------
# SAVE TRAINED MODEL AND PREPROCESSING OBJECTS
# --------------------------------------------------

import joblib
import os

os.makedirs(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models"
    ),
    exist_ok=True
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "logistic_regression.pkl"
)

SCALER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "scaler.pkl"
)

COLUMNS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "model_columns.pkl"
)


# Save Logistic Regression
joblib.dump(
    models["Logistic Regression"],
    MODEL_PATH
)

# Save scaler
joblib.dump(
    scaler,
    SCALER_PATH
)

# Save encoded column names
joblib.dump(
    X_encoded.columns.tolist(),
    COLUMNS_PATH
)

print("\nModel and preprocessing objects saved successfully.")