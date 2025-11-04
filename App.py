# medical_expense_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------
st.set_page_config(page_title="Medical Expense Predictor", layout="wide")
st.title("💉 Medical Expense Prediction App")
st.markdown("### Predicting patient medical expenses using multiple regression models.")

# ---------------------------------------------------------------
# STEP 1: BUSINESS PROBLEM
# ---------------------------------------------------------------
st.subheader("📘 STEP 1: Business Problem Understanding")
st.write("""
Predict medical expenses based on patient's details such as age, sex, BMI, smoker status, and number of children.
""")

# ---------------------------------------------------------------
# STEP 2: DATA UPLOAD
# ---------------------------------------------------------------
st.subheader("📂 STEP 2: Load and Collect Data")

uploaded_file = st.file_uploader("Upload your insurance dataset (Excel format)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ Data loaded successfully!")
    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    # ---------------------------------------------------------------
    # DATA UNDERSTANDING
    # ---------------------------------------------------------------
    st.subheader("🔍 Data Understanding")
    with st.expander("View Dataset Info"):
        buffer = []
        df.info(buf=buffer)
        info_str = "\n".join(buffer)
        st.text(info_str)

    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", list(df.columns))
    st.write("**Size:**", df.size)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Sex counts:")
        st.write(df["sex"].value_counts())
    with col2:
        st.write("Smoker counts:")
        st.write(df["smoker"].value_counts())
    with col3:
        st.write("Region counts:")
        st.write(df["region"].value_counts())

    # ---------------------------------------------------------------
    # EXPLORATORY DATA ANALYSIS
    # ---------------------------------------------------------------
    st.subheader("📊 Exploratory Data Analysis")
    continous = ['age', 'bmi', 'expenses']
    discrete_categorical = ['sex', 'smoker', 'region']

    st.write("#### Continuous Feature Description")
    st.dataframe(df[continous].describe())

    st.write("#### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[continous].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("#### Pair Plot of Continuous Variables")
    st.pyplot(sns.pairplot(df[continous]))

    st.write("#### Scatter Plot: BMI vs Expenses")
    fig, ax = plt.subplots()
    sns.scatterplot(x='bmi', y='expenses', hue='sex', data=df, ax=ax)
    st.pyplot(fig)

    # ---------------------------------------------------------------
    # DATA PREPROCESSING
    # ---------------------------------------------------------------
    st.subheader("⚙️ STEP 3: Data Preprocessing")

    st.write("Missing values:", df.isnull().sum().sum())
    st.write("Duplicate rows:", df.duplicated().sum())

    df.drop_duplicates(inplace=True)

    # Drop region as it’s not significant
    df.drop('region', axis=1, inplace=True)

    # Encoding
    df['sex'].replace({'male': 1, 'female': 0}, inplace=True)
    df['smoker'].replace({'yes': 1, 'no': 0}, inplace=True)

    st.write("✅ Data after preprocessing:")
    st.dataframe(df.head())

    # ---------------------------------------------------------------
    # MODELING SECTION
    # ---------------------------------------------------------------
    st.subheader("🤖 STEP 4: Model Building and Evaluation")

    x = df.drop('expenses', axis=1)
    y = df['expenses']

    test_size = st.slider("Select Test Size (for Train-Test Split)", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random State", value=9, step=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    model_option = st.selectbox(
        "Choose Regression Model",
        ["Linear Regression", "Lasso Regression", "Ridge Regression", "ElasticNet Regression"]
    )

    if st.button("🚀 Train Model"):
        if model_option == "Linear Regression":
            model = LinearRegression()
            model.fit(x_train, y_train)

        elif model_option == "Lasso Regression":
            alpha = st.slider("Select Alpha (Regularization Strength)", 1, 100, 60)
            model = Lasso(alpha=alpha)
            model.fit(x_train, y_train)

        elif model_option == "Ridge Regression":
            alpha = st.slider("Select Alpha (Regularization Strength)", 1, 100, 10)
            model = Ridge(alpha=alpha)
            model.fit(x_train, y_train)

        elif model_option == "ElasticNet Regression":
            alpha = st.select_slider("Alpha", options=[0.1, 0.2, 1, 2, 5, 10], value=10.0)
            l1_ratio = st.select_slider("L1 Ratio", options=[0.2, 0.4, 0.6, 0.8, 1.0], value=1.0)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            model.fit(x_train, y_train)

        # Prediction and Metrics
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='r2').mean()

        st.write("### 📈 Model Results")
        st.write(f"**Train RMSE:** {train_rmse:.2f}")
        st.write(f"**Test RMSE:** {test_rmse:.2f}")
        st.write(f"**Train R²:** {train_r2:.3f}")
        st.write(f"**Test R²:** {test_r2:.3f}")
        st.write(f"**Cross Validation Score:** {cv_score:.3f}")

        st.write("**Model Coefficients:**")
        coef_df = pd.DataFrame({"Feature": x.columns, "Coefficient": model.coef_})
        st.dataframe(coef_df)

        # Visualization of predictions
        st.write("#### Predicted vs Actual (Test Data)")
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred_test, ax=ax)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{model_option} - Actual vs Predicted")
        st.pyplot(fig)

    # ---------------------------------------------------------------
    # FINAL PREDICTION SECTION
    # ---------------------------------------------------------------
    st.subheader("💡 STEP 5: Predict on New Data")

    st.write("Enter patient details to predict medical expenses:")
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.5)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])

    if 'model' in locals():
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [1 if sex == 'male' else 0],
            'bmi': [bmi],
            'children': [children],
            'smoker': [1 if smoker == 'yes' else 0]
        })

        if st.button("🔮 Predict Expense"):
            pred_expense = model.predict(input_data)[0]
            st.success(f"💰 Predicted Medical Expense: ${pred_expense:,.2f}")
    else:
        st.info("Train a model first to enable predictions.")
else:
    st.warning("👆 Please upload your Excel dataset to start analysis.")
