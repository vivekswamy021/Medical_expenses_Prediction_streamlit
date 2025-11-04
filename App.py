import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="💰 Medical Expense Prediction", layout="wide")

# -----------------------------
# APP TITLE
# -----------------------------
st.title("💰 Medical Expense Prediction & EDA")
st.markdown("""
This app allows you to explore the insurance dataset and predicts **medical expenses** based on:  
**Age, Sex, BMI, Children, and Smoking Status** using pre-trained models.
""")

# -----------------------------
# STEP 1: UPLOAD DATA
# -----------------------------
st.sidebar.header("📁 Upload Dataset (Excel or CSV)")
uploaded_file = st.sidebar.file_uploader("Upload insurance dataset", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    # Preprocessing: drop region, encode categorical columns
    if 'region' in df.columns:
        df.drop('region', axis=1, inplace=True)
    if 'sex' in df.columns:
        df['sex'] = df['sex'].replace({'male':1, 'female':0})
    if 'smoker' in df.columns:
        df['smoker'] = df['smoker'].replace({'yes':1, 'no':0})

    st.header("🔍 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # STEP 2: EXPLORATORY DATA ANALYSIS
    # -----------------------------
    st.sidebar.header("📊 EDA Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
    )

    numeric_cols = ['age', 'bmi', 'children', 'expenses']
    st.header(f"📈 {analysis_type}")

    if analysis_type == "Univariate Analysis":
        feature = st.selectbox("Select Feature for Histogram", numeric_cols)
        bins = st.slider("Number of bins", 5, 50, 10)
        fig, ax = plt.subplots()
        ax.hist(df[feature], bins=bins, rwidth=0.8, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif analysis_type == "Bivariate Analysis":
        x_feature = st.selectbox("Select X-axis Feature", numeric_cols, index=0)
        y_feature = st.selectbox("Select Y-axis Feature", numeric_cols, index=1)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_feature], y=df[y_feature], ax=ax)
        ax.set_title(f"{x_feature} vs {y_feature}")
        st.pyplot(fig)

    elif analysis_type == "Multivariate Analysis":
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Pairplot (Scatter Matrix)")
        pairplot_fig = sns.pairplot(df[numeric_cols])
        st.pyplot(pairplot_fig.fig)

# -----------------------------
# STEP 3: USER INPUT FORM
# -----------------------------
st.sidebar.header("🧍 Enter Patient Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=1)
smoker = st.sidebar.selectbox("Smoker", ["Yes", "No"])

sex_val = 1 if sex=="Male" else 0
smoker_val = 1 if smoker=="Yes" else 0
input_features = np.array([[age, sex_val, bmi, children, smoker_val]])

st.subheader("📝 Input Summary")
input_dict = {
    "Age": age,
    "Sex": sex,
    "BMI": bmi,
    "Children": children,
    "Smoker": smoker
}
st.table(input_dict)

# -----------------------------
# STEP 4: MODEL SELECTION
# -----------------------------
st.sidebar.header("⚙️ Choose Model")

available_models = {
    "Linear Regression": "linear_regression_model.pkl",
    "Lasso Regression": "lasso_model.pkl",
    "Ridge Regression": "ridge_model.pkl",
    "ElasticNet Regression": "elasticnet_model.pkl"
}

model_choice = st.sidebar.selectbox("Select Model", list(available_models.keys()))
selected_model_file = available_models[model_choice]

# -----------------------------
# STEP 5: LOAD MODEL & PREDICT
# -----------------------------
st.header("📂 Prediction")

if os.path.exists(selected_model_file):
    model = joblib.load(selected_model_file)
    if st.button("🔮 Predict Medical Expense"):
        prediction = model.predict(input_features)[0]
        st.success(f"💰 Predicted Medical Expense using {model_choice}: **${prediction:,.2f}**")
else:
    st.warning(f"⚠️ {selected_model_file} not found. Please place the .pkl model file in the same directory.")
