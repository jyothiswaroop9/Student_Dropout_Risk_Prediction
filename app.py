import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "dataset.csv"
)

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_regression.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_columns.pkl"
)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
model_columns = joblib.load(COLUMNS_PATH)
# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    font-size: 17px;
    color: #666666;
    margin-bottom: 30px;
}

.metric-card {
    background-color: white;
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

.metric-title {
    font-size: 15px;
    color: #666666;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: black;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="dashboard-title">🎓 Student Dropout Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'AI-powered student outcome analysis and prediction system'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🎓 Student Risk System")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Student Prediction",
        "Dataset"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This system uses machine learning to analyze "
    "student information and predict academic outcomes."
)

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.header("📊 Project Dashboard")

    # Dataset statistics
    total_students = len(df)

    dropout_count = (
        df["Target"] == "Dropout"
    ).sum()

    enrolled_count = (
        df["Target"] == "Enrolled"
    ).sum()

    graduate_count = (
        df["Target"] == "Graduate"
    ).sum()

    # --------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Students</div>
                <div class="metric-value">{total_students:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Dropout</div>
                <div class="metric-value">{dropout_count:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Enrolled</div>
                <div class="metric-value">{enrolled_count:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Graduate</div>
                <div class="metric-value">{graduate_count:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------
    # CHARTS
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📈 Student Outcome Distribution")

        target_counts = df["Target"].value_counts()

        st.bar_chart(target_counts)

    with col2:

        st.subheader("📋 Outcome Percentage")

        target_percentage = (
            df["Target"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        percentage_df = target_percentage.rename(
            "Percentage (%)"
        ).to_frame()

        st.dataframe(
            percentage_df,
            use_container_width=True
        )

    # --------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------

    st.subheader("ℹ️ About the Project")

    st.write(
        """
        The Student Dropout Risk Prediction system uses machine
        learning techniques to analyze student academic and
        demographic information.

        The system predicts one of three outcomes:

        • Dropout  
        • Enrolled  
        • Graduate
        """
    )

# --------------------------------------------------
# STUDENT PREDICTION PAGE
# --------------------------------------------------
elif page == "Student Prediction":

    st.header("🔍 Student Prediction")

    st.write(
        "Enter the student's information to predict their "
        "academic outcome."
    )

    # --------------------------------------------------
    # STUDENT INPUTS
    # --------------------------------------------------

    st.subheader("👤 Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        marital_status = st.number_input(
            "Marital status",
            min_value=1,
            max_value=6,
            value=1
        )

        application_mode = st.number_input(
            "Application mode",
            min_value=1,
            max_value=18,
            value=8
        )

        application_order = st.number_input(
            "Application order",
            min_value=0,
            max_value=9,
            value=1
        )

    with col2:

        course = st.number_input(
            "Course",
            min_value=1,
            max_value=17,
            value=2
        )

        daytime = st.selectbox(
            "Daytime / Evening Attendance",
            [0, 1],
            index=1
        )

        previous_qualification = st.number_input(
            "Previous qualification",
            min_value=1,
            max_value=17,
            value=1
        )

    with col3:

        nationality = st.number_input(
            "Nationality",
            min_value=1,
            max_value=21,
            value=1
        )

        age = st.number_input(
            "Age at enrollment",
            min_value=17,
            max_value=70,
            value=20
        )

        gender = st.selectbox(
            "Gender",
            [0, 1]
        )

    st.subheader("🏠 Family & Financial Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        mother_qualification = st.number_input(
            "Mother's qualification",
            min_value=1,
            max_value=29,
            value=13
        )

        father_qualification = st.number_input(
            "Father's qualification",
            min_value=1,
            max_value=34,
            value=14
        )

    with col2:

        mother_occupation = st.number_input(
            "Mother's occupation",
            min_value=1,
            max_value=32,
            value=6
        )

        father_occupation = st.number_input(
            "Father's occupation",
            min_value=1,
            max_value=46,
            value=8
        )

    with col3:

        displaced = st.selectbox(
            "Displaced",
            [0, 1]
        )

        debtor = st.selectbox(
            "Debtor",
            [0, 1]
        )

        tuition = st.selectbox(
            "Tuition fees up to date",
            [0, 1],
            index=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        special_needs = st.selectbox(
            "Educational special needs",
            [0, 1]
        )

    with col2:

        scholarship = st.selectbox(
            "Scholarship holder",
            [0, 1]
        )

    with col3:

        international = st.selectbox(
            "International",
            [0, 1]
        )

    st.subheader("📚 Academic Performance")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 1st Semester")

        sem1_credited = st.number_input(
            "Credited units",
            min_value=0,
            max_value=20,
            value=0
        )

        sem1_enrolled = st.number_input(
            "Enrolled units",
            min_value=0,
            max_value=26,
            value=6
        )

        sem1_evaluations = st.number_input(
            "Evaluations",
            min_value=0,
            max_value=45,
            value=8
        )

        sem1_approved = st.number_input(
            "Approved units",
            min_value=0,
            max_value=26,
            value=5
        )

        sem1_grade = st.number_input(
            "Grade",
            min_value=0.0,
            max_value=18.88,
            value=12.0
        )

        sem1_without_eval = st.number_input(
            "Without evaluations",
            min_value=0,
            max_value=12,
            value=0
        )

    with col2:

        st.markdown("### 2nd Semester")

        sem2_credited = st.number_input(
            "Credited units ",
            min_value=0,
            max_value=19,
            value=0
        )

        sem2_enrolled = st.number_input(
            "Enrolled units ",
            min_value=0,
            max_value=23,
            value=6
        )

        sem2_evaluations = st.number_input(
            "Evaluations ",
            min_value=0,
            max_value=33,
            value=8
        )

        sem2_approved = st.number_input(
            "Approved units ",
            min_value=0,
            max_value=20,
            value=5
        )

        sem2_grade = st.number_input(
            "Grade ",
            min_value=0.0,
            max_value=18.57,
            value=12.0
        )

        sem2_without_eval = st.number_input(
            "Without evaluations ",
            min_value=0,
            max_value=12,
            value=0
        )

    st.subheader("🌍 Economic Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        unemployment = st.number_input(
            "Unemployment rate",
            min_value=7.6,
            max_value=16.2,
            value=11.57
        )

    with col2:

        inflation = st.number_input(
            "Inflation rate",
            min_value=-0.8,
            max_value=3.7,
            value=1.23
        )

    with col3:

        gdp = st.number_input(
            "GDP",
            min_value=-4.06,
            max_value=3.51,
            value=0.0
        )

    # --------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------

    st.write("")

    analyze = st.button(
        "🔍 ANALYZE STUDENT",
        use_container_width=True
    )

    if analyze:

        # Create original feature dictionary
        student_data = {

            "Marital status": marital_status,
            "Application mode": application_mode,
            "Application order": application_order,
            "Course": course,
            "Daytime/evening attendance": daytime,
            "Previous qualification": previous_qualification,
            "Nacionality": nationality,
            "Mother's qualification": mother_qualification,
            "Father's qualification": father_qualification,
            "Mother's occupation": mother_occupation,
            "Father's occupation": father_occupation,
            "Displaced": displaced,
            "Educational special needs": special_needs,
            "Debtor": debtor,
            "Tuition fees up to date": tuition,
            "Gender": gender,
            "Scholarship holder": scholarship,
            "Age at enrollment": age,
            "International": international,

            "Curricular units 1st sem (credited)": sem1_credited,
            "Curricular units 1st sem (enrolled)": sem1_enrolled,
            "Curricular units 1st sem (evaluations)": sem1_evaluations,
            "Curricular units 1st sem (approved)": sem1_approved,
            "Curricular units 1st sem (grade)": sem1_grade,
            "Curricular units 1st sem (without evaluations)": sem1_without_eval,

            "Curricular units 2nd sem (credited)": sem2_credited,
            "Curricular units 2nd sem (enrolled)": sem2_enrolled,
            "Curricular units 2nd sem (evaluations)": sem2_evaluations,
            "Curricular units 2nd sem (approved)": sem2_approved,
            "Curricular units 2nd sem (grade)": sem2_grade,
            "Curricular units 2nd sem (without evaluations)": sem2_without_eval,

            "Unemployment rate": unemployment,
            "Inflation rate": inflation,
            "GDP": gdp
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([student_data])

        # Categorical features
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

        # One-hot encode
        input_encoded = pd.get_dummies(
            input_df,
            columns=categorical_features,
            drop_first=False
        )

        # Make sure all 260 columns exist
        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        # Scale numerical columns
        input_encoded[numerical_features] = scaler.transform(
            input_encoded[numerical_features]
        )

        # Prediction
        prediction = model.predict(input_encoded)[0]

        probabilities = model.predict_proba(
            input_encoded
        )[0]

        class_names = {
            0: "Dropout",
            1: "Enrolled",
            2: "Graduate"
        }

        prediction_name = class_names[prediction]

        confidence = probabilities[prediction] * 100

        # Risk classification
        if prediction_name == "Dropout":
            risk_level = "High Risk"
        elif prediction_name == "Enrolled":
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        # --------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------

        st.markdown("---")

        st.subheader("📊 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Outcome",
                prediction_name
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        with col3:

            st.metric(
                "Risk Level",
                risk_level
            )

        if prediction_name == "Dropout":

            st.error(
                "⚠️ High dropout risk detected. "
                "The student may require additional academic support."
            )

        elif prediction_name == "Enrolled":

            st.warning(
                "⚠️ The student is currently classified as enrolled."
            )

        else:

            st.success(
                "✅ The student is predicted to graduate."
            )

        # Probability chart

        st.subheader("📈 Prediction Probabilities")

        probability_df = pd.DataFrame(
            {
                "Outcome": [
                    "Dropout",
                    "Enrolled",
                    "Graduate"
                ],
                "Probability": probabilities * 100
            }
        )

        probability_df = probability_df.set_index(
            "Outcome"
        )

        st.bar_chart(probability_df)
# --------------------------------------------------
# DATASET PAGE
# --------------------------------------------------

elif page == "Dataset":

    st.header("📁 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Features",
            df.shape[1] - 1
        )

    with col3:
        st.metric(
            "Target Classes",
            df["Target"].nunique()
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Target Distribution")

    st.dataframe(
        df["Target"].value_counts().rename("Students"),
        use_container_width=True
    )