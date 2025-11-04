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

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Medical Expenses Prediction", layout="wide")
st.title("💊 Medical Expenses Prediction App")
st.markdown("### Predict insurance costs using regression models")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload your insurance.xlsx file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    st.subheader("📊 Data Overview")
    st.dataframe(df.head())

    # -----------------------------
    # DATA UNDERSTANDING
    # -----------------------------
    with st.expander("🔍 Dataset Info"):
        buf = []
        df.info(buf=buf)
        s = "\n".join(buf)
        st.text(s)
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())

    # -----------------------------
    # EDA SECTION
    # -----------------------------
    st.subheader("📈 Exploratory Data Analysis")
    continous = ['age','bmi','expenses']
    discrete_categorical = ['sex','smoker','region']
    discrete_count = ['children']

    st.write("**Continuous Features Summary:**")
    st.dataframe(df[continous].describe())

    st.write("**Categorical Features Summary:**")
    st.dataframe(df[discrete_categorical].describe())

    # Correlation heatmap
    fig, ax = plt.subplots()
    sns.heatmap(df[continous].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Pairplot
    st.write("**Pairplot:**")
    sns.pairplot(df[continous])
    st.pyplot(plt)

    # Scatterplot BMI vs Expenses
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['bmi'], y=df['expenses'], hue=df['sex'], ax=ax)
    st.pyplot(fig)

    # -----------------------------
    # DATA CLEANING & ENCODING
    # -----------------------------
    st.subheader("🧹 Data Preprocessing")

    st.write("Dropping duplicates...")
    df.drop_duplicates(inplace=True)

    st.write("Encoding categorical columns...")
    df['sex'].replace({'male':1,'female':0},inplace=True)
    df['smoker'].replace({'yes':1,'no':0},inplace=True)

    st.write("Dropping 'region' column...")
    df.drop('region', axis=1, inplace=True)

    st.success("✅ Data cleaned and encoded successfully!")

    # -----------------------------
    # TRAIN TEST SPLIT
    # -----------------------------
    x = df.drop('expenses', axis=1)
    y = df['expenses']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    st.subheader("🏋️ Model Training and Evaluation")

    model_choice = st.selectbox("Choose a model to train", ["Linear Regression", "Lasso Regression", "Ridge Regression", "ElasticNet Regression"])

    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

    elif model_choice == "Lasso Regression":
        model = Lasso(alpha=60)
        model.fit(x_train, y_train)
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

    elif model_choice == "Ridge Regression":
        model = Ridge(alpha=1)
        model.fit(x_train, y_train)
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

    elif model_choice == "ElasticNet Regression":
        model = ElasticNet(alpha=10, l1_ratio=1)
        model.fit(x_train, y_train)
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

    # -----------------------------
    # MODEL PERFORMANCE
    # -----------------------------
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='r2').mean()

    st.write("### 📊 Model Performance")
    st.write(f"**Train RMSE:** {train_rmse:.2f}")
    st.write(f"**Test RMSE:** {test_rmse:.2f}")
    st.write(f"**Train R2:** {train_r2:.3f}")
    st.write(f"**Test R2:** {test_r2:.3f}")
    st.write(f"**Cross Validation (CV) R2:** {cv_score:.3f}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    filename = model_choice.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, filename)
    with open(filename, "rb") as f:
        st.download_button(label=f"💾 Download {model_choice} Model", data=f, file_name=filename)

    # -----------------------------
    # PREDICTION SECTION
    # -----------------------------
    st.subheader("🧮 Predict New Data")
    st.markdown("Enter patient details to predict medical expenses")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    with col2:
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
        sex = st.selectbox("Sex", ["male", "female"])
    with col3:
        smoker = st.selectbox("Smoker", ["yes", "no"])

    sex_val = 1 if sex == "male" else 0
    smoker_val = 1 if smoker == "yes" else 0

    input_data = np.array([[age, sex_val, bmi, children, smoker_val]])
    if st.button("🔮 Predict Expenses"):
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Medical Expense: **${prediction:.2f}**")

else:
    st.warning("👆 Please upload the `insurance.xlsx` file to start.")
