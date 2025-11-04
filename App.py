import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------
# Streamlit Page Config
# ---------------------------------------
st.set_page_config(page_title="Medical Expenses Predictor", layout="wide")
st.title("💊 Medical Expenses Prediction App")
st.markdown("Predict insurance costs using multiple regression techniques.")

# ---------------------------------------
# File Upload Section
# ---------------------------------------
uploaded_file = st.file_uploader("📂 Upload the insurance.xlsx file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # ---------------------------------------
    # Data Overview
    # ---------------------------------------
    with st.expander("🔍 Dataset Info"):
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())
        st.write("**Data Types:**")
        st.write(df.dtypes)

    # ---------------------------------------
    # EDA Section
    # ---------------------------------------
    st.subheader("📈 Exploratory Data Analysis")
    continous = ['age', 'bmi', 'expenses']
    discrete_categorical = ['sex', 'smoker', 'region']

    st.write("**Continuous Features Summary:**")
    st.dataframe(df[continous].describe())

    st.write("**Categorical Features Summary:**")
    st.dataframe(df[discrete_categorical].describe())

    # Correlation heatmap
    fig, ax = plt.subplots()
    sns.heatmap(df[continous].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Scatterplot
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='bmi', y='expenses', hue='sex', data=df, ax=ax2)
    st.pyplot(fig2)

    # ---------------------------------------
    # Data Cleaning and Encoding
    # ---------------------------------------
    st.subheader("🧹 Data Preprocessing")
    df.drop_duplicates(inplace=True)

    df['sex'] = df['sex'].replace({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].replace({'yes': 1, 'no': 0})

    if 'region' in df.columns:
        df.drop('region', axis=1, inplace=True)

    st.success("✅ Data cleaned and encoded successfully!")

    # ---------------------------------------
    # Train Test Split
    # ---------------------------------------
    X = df.drop('expenses', axis=1)
    y = df['expenses']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    # ---------------------------------------
    # Model Selection
    # ---------------------------------------
    st.subheader("🤖 Train a Model")

    model_choice = st.selectbox(
        "Choose Regression Model:",
        ["Linear Regression", "Lasso Regression", "Ridge Regression", "ElasticNet Regression"]
    )

    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Lasso Regression":
        alpha = st.slider("Select Alpha (λ)", 1, 100, 60)
        model = Lasso(alpha=alpha)
    elif model_choice == "Ridge Regression":
        alpha = st.slider("Select Alpha (λ)", 1, 100, 1)
        model = Ridge(alpha=alpha)
    else:
        alpha = st.slider("Select Alpha (λ)", 1, 100, 10)
        l1_ratio = st.slider("Select L1 Ratio", 0.0, 1.0, 1.0)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    if st.button("🚀 Train Model"):
        model.fit(x_train, y_train)

        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='r2').mean()

        st.subheader("📊 Model Performance")
        st.write(f"**Train RMSE:** {train_rmse:.2f}")
        st.write(f"**Test RMSE:** {test_rmse:.2f}")
        st.write(f"**Train R2:** {train_r2:.3f}")
        st.write(f"**Test R2:** {test_r2:.3f}")
        st.write(f"**Cross Validation R2:** {cv_score:.3f}")

        # Save model
        filename = model_choice.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, filename)

        with open(filename, "rb") as f:
            st.download_button(
                label=f"💾 Download {model_choice} Model",
                data=f,
                file_name=filename,
                mime="application/octet-stream"
            )

    # ---------------------------------------
    # Prediction Section
    # ---------------------------------------
    st.subheader("🧮 Predict New Medical Expense")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 0, 100, 30)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    with col2:
        children = st.number_input("Children", 0, 10, 0)
        sex = st.selectbox("Sex", ["male", "female"])
    with col3:
        smoker = st.selectbox("Smoker", ["yes", "no"])

    sex_val = 1 if sex == "male" else 0
    smoker_val = 1 if smoker == "yes" else 0

    input_data = np.array([[age, sex_val, bmi, children, smoker_val]])

    if st.button("🔮 Predict Expense"):
        if 'model' not in locals():
            st.error("Please train a model first.")
        else:
            prediction = model.predict(input_data)[0]
            st.success(f"💰 Predicted Medical Expense: **${prediction:.2f}**")

else:
    st.warning("👆 Please upload the `insurance.xlsx` file to continue.")
