import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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
# TARGET ENCODING
# --------------------------------------------------

target_mapping = {
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2
}

reverse_mapping = {
    0: "Dropout",
    1: "Enrolled",
    2: "Graduate"
}

y_encoded = y.map(target_mapping)


# --------------------------------------------------
# ONE-HOT ENCODING
# --------------------------------------------------

X_encoded = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=False
)


# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_encoded[numerical_features] = scaler.fit_transform(
    X_encoded[numerical_features]
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


model = LogisticRegression(
    max_iter=2000
)

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_student(student_data):

    student_df = pd.DataFrame([student_data])

    # One-hot encode
    student_encoded = pd.get_dummies(
        student_df,
        columns=categorical_features,
        drop_first=False
    )

    # Match training columns
    student_encoded = student_encoded.reindex(
        columns=X_encoded.columns,
        fill_value=0
    )

    # Scale numerical features
    student_encoded[numerical_features] = scaler.transform(
        student_encoded[numerical_features]
    )

    # Prediction
    prediction = model.predict(
        student_encoded
    )[0]

    # Probability
    probabilities = model.predict_proba(
        student_encoded
    )[0]

    confidence = probabilities[prediction] * 100

    predicted_class = reverse_mapping[prediction]

    return predicted_class, confidence


# --------------------------------------------------
# TEST PREDICTION
# --------------------------------------------------

if __name__ == "__main__":

    print("\n==========================================")
    print("       STUDENT PREDICTION SYSTEM")
    print("==========================================")

    # Example student
    sample_student = X.iloc[0].to_dict()

    prediction, confidence = predict_student(
    sample_student
)


# --------------------------------------------------
# RISK LEVEL
# --------------------------------------------------

if prediction == "Dropout":
    risk_level = "High Risk"

elif prediction == "Enrolled":
    risk_level = "Medium Risk"

else:
    risk_level = "Low Risk"


print("\nPrediction:")
print(prediction)

print(f"\nConfidence: {confidence:.2f}%")

print(f"\nRisk Level:")
print(risk_level)