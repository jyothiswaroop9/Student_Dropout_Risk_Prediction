import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


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
# TARGET DISTRIBUTION
# --------------------------------------------------

target_counts = df["Target"].value_counts()


print("\n==========================================")
print("          EXPLORATORY DATA ANALYSIS")
print("==========================================")

print("\nTarget Distribution:")
print(target_counts)


# --------------------------------------------------
# 1ST SEMESTER GRADE BY STUDENT OUTCOME
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Target",
    y="Curricular units 1st sem (grade)"
)

plt.title("1st Semester Grade by Student Outcome")
plt.xlabel("Student Outcome")
plt.ylabel("1st Semester Grade")

plt.tight_layout()

plt.show()

# --------------------------------------------------
# 2ND SEMESTER GRADE BY STUDENT OUTCOME
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Target",
    y="Curricular units 2nd sem (grade)"
)

plt.title("2nd Semester Grade by Student Outcome")
plt.xlabel("Student Outcome")
plt.ylabel("2nd Semester Grade")

plt.tight_layout()

plt.show()

# --------------------------------------------------
# 2ND SEMESTER APPROVED UNITS BY OUTCOME
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Target",
    y="Curricular units 2nd sem (approved)"
)

plt.title("2nd Semester Approved Units by Student Outcome")
plt.xlabel("Student Outcome")
plt.ylabel("Approved Units")

plt.tight_layout()

plt.show()

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

numeric_df = df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(16, 12))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.show()

# --------------------------------------------------
# OUTLIER ANALYSIS
# --------------------------------------------------

outlier_features = [
    "Age at enrollment",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)"
]

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df[outlier_features]
)

plt.title("Outlier Analysis of Important Features")
plt.xticks(rotation=25)
plt.ylabel("Values")

plt.tight_layout()

plt.show()