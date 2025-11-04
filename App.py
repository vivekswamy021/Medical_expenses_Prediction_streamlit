import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import io
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------
st.set_page_config(page_title="Medical Expense Predictor", layout="wide")
st.title("💉 Medical Expense Prediction App")
st.markdown("### Predict medical expenses based on patient's profile using Regression Models")

# ---------------------------------------------------------------
# STEP 1: BUSINESS PROBLEM
# ---------------------------------------------------------------
st.subheader("📘 STEP 1: Business Problem Understanding")
st.write("""
Predict patient medical expenses based on demographic and lifestyle factors
""")

# ---------------------------------------------------------------
# STEP 2: DATA UPLOAD
# ---------------------------------------------------------------
st.subheader("📂 STEP 2: Load and Collect Data")

uploaded_file = st.file_uploader("Upload your insurance dataset (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded and data loaded successfully!")

    st.write("### 📋 Dataset Preview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Sex Distribution:")
        st.bar_chart(df["sex"].value_counts())
    with col2:
        st.write("Smoker Distribution:")
        st.bar_chart(df["smoker"].value_counts())
    with col3:
        st.write("Region Distribution:")
        st.bar_chart(df["region"].value_counts())

    # ---------------------------------------------------------------
    # STEP 3: EXPLORATORY DATA ANALYSIS
    # ---------------------------------------------------------------
    st.subheader("📊 STEP 3: Exploratory Data Analysis")

    continous = ['age', 'bmi', 'expenses']
    discrete_categorical = ['sex', 'smoker', 'region']

    st.write("### Continuous Variables Summary")
    st.dataframe(df[continous].describe())

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[continous].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("### Pairplot of Continuous Features")
    pair_fig = sns.pairplot(df[continous])
    st.pyplot(pair_fig.fig)

    st.write("### Scatterplot: BMI vs Expenses")
    fig, ax = plt.subplots()
    sns.scatterplot(x="bmi", y="expenses", hue="sex", data=df, ax=ax)
    st.pyplot(fig)

    # ---------------------------------------------------------------
    # STEP 4: DATA PREPROCESSING
    # ---------------------------------------------------------------
    st.subheader("⚙️ STEP 4: Data Preprocessing")

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop region (low impact)
    if "region" in df.columns:
        df.drop("region", axis=1, inplace=True)

    # Encode categorical columns
    df["sex"].replace({"male": 1, "female": 0}, inplace=True)
    df["smoker"].replace({"yes": 1, "no": 0}, inplace=True)

    st.write("✅ Cleaned Data Preview")
    st.dataframe(df.head())

  
    model_option = st.selectbox(
        "Select Regression Model",
        ["Linear Regression", "Lasso Regression", "Ridge Regression", "ElasticNet Regression"]
    )

        # Model selection
        if model_option == "Linear Regression":
            model = LinearRegression()

        elif model_option == "Lasso Regression":
            alpha = st.slider("Lasso Alpha", 1, 100, 60)
            model = Lasso(alpha=alpha)

        elif model_option == "Ridge Regression":
            alpha = st.slider("Ridge Alpha", 1, 100, 10)
            model = Ridge(alpha=alpha)

        elif model_option == "ElasticNet Regression":
            alpha = st.select_slider("Alpha", options=[0.1, 0.2, 1, 2, 5, 10], value=10.0)
            l1_ratio = st.select_slider("L1 Ratio", options=[0.2, 0.4, 0.6, 0.8, 1.0], value=1.0)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)


        # Plot Actual vs Predicted
        st.write("### Predicted vs Actual (Test Set)")
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_test_pred, ax=ax)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{model_option} - Actual vs Predicted")
        st.pyplot(fig)

        # Save model to session state
        st.session_state["trained_model"] = model
        st.session_state["X_columns"] = X.columns.tolist()

    # ---------------------------------------------------------------
    # STEP 6: PREDICT NEW DATA
    # ---------------------------------------------------------------
    st.subheader("💡 STEP 6: Predict on New Data")

    if "trained_model" in st.session_state:
        model = st.session_state["trained_model"]
        st.write("Enter new patient details:")

        age = st.number_input("Age", min_value=1, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.5)
        children = st.number_input("Children", min_value=0, max_value=10, value=1)
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["yes", "no"])

        if st.button("🔮 Predict Expense"):
            new_df = pd.DataFrame({
                "age": [age],
                "sex": [1 if sex == "male" else 0],
                "bmi": [bmi],
                "children": [children],
                "smoker": [1 if smoker == "yes" else 0]
            })
            prediction = model.predict(new_df)[0]
            st.success(f"💰 Predicted Medical Expense: ${prediction:,.2f}")
    else:
        st.info("Train a model first to make predictions.")
else:
    st.warning("👆 Please upload your Excel dataset to start analysis.")

