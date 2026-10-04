import pandas as pd
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
# SELECT NUMERICAL COLUMNS
# --------------------------------------------------

numeric_df = df.select_dtypes(include="number")


# --------------------------------------------------
# STATISTICAL ANALYSIS
# --------------------------------------------------

statistics = pd.DataFrame({
    "Mean": numeric_df.mean(),
    "Median": numeric_df.median(),
    "Standard Deviation": numeric_df.std(),
    "Variance": numeric_df.var(),
    "Minimum": numeric_df.min(),
    "Q1": numeric_df.quantile(0.25),
    "Q3": numeric_df.quantile(0.75),
    "Maximum": numeric_df.max()
})


# --------------------------------------------------
# CALCULATE IQR
# --------------------------------------------------

statistics["IQR"] = (
    statistics["Q3"] - statistics["Q1"]
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n==========================================")
print("       STATISTICAL ANALYSIS")
print("==========================================")

print("\nNumerical Features:")
print(numeric_df.columns.tolist())

print("\nStatistical Summary:")
print(statistics.round(2))