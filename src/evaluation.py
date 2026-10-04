import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


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
# MODEL EVALUATION
# --------------------------------------------------

print("\n==========================================")
print("             MODEL EVALUATION")
print("==========================================")


results = []


for name, model in models.items():

    print(f"\nEvaluating {name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"\n{name}")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Dropout",
                "Enrolled",
                "Graduate"
            ],
            zero_division=0
        )
    )


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n==========================================")
print("          MODEL COMPARISON")
print("==========================================")

print(
    results_df.to_string(
        index=False
    )
)

# --------------------------------------------------
# CONFUSION MATRIX VISUALIZATION
# --------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns


for name, model in models.items():

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Dropout",
            "Enrolled",
            "Graduate"
        ],
        yticklabels=[
            "Dropout",
            "Enrolled",
            "Graduate"
        ]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
# MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

results_plot = results_df.set_index("Model")[metrics]

results_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Machine Learning Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)

plt.xticks(rotation=0)

plt.legend(
    title="Metrics",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.show()