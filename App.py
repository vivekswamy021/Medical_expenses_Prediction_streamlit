import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="💰 Medical Expense Prediction", layout="wide")

st.title("💰 Medical Expense Prediction & EDA Dashboard")
st.markdown("Predict insurance expenses using pre-trained `.pkl` models and explore the dataset.")

# -----------------------------
# UPLOAD DATA
# -----------------------------
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload insurance dataset (Excel or CSV)", type=["xlsx","csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Preprocessing
        if 'region' in df.columns:
            df.drop('region', axis=1, inplace=True)
        if 'sex' in df.columns:
            df['sex'] = df['sex'].replace({'male':1, 'female':0})
        if 'smoker' in df.columns:
            df['smoker'] = df['smoker'].replace({'yes':1, 'no':0})

        st.header("🔍 Dataset Preview")
        st.dataframe(df.head())

        # EDA
        st.sidebar.header("📊 EDA")
        analysis_type = st.sidebar.selectbox("Select Analysis", ["Univariate","Bivariate","Multivariate"])
        numeric_cols = ['age','bmi','children','expenses']

        if analysis_type=="Univariate":
            feature = st.selectbox("Select Feature for Histogram", numeric_cols)
            bins = st.slider("Bins", 5, 50, 10)
            fig, ax = plt.subplots()
            ax.hist(df[feature], bins=bins, color='skyblue', edgecolor='black')
            st.pyplot(fig)

        elif analysis_type=="Bivariate":
            x_feature = st.selectbox("X-axis", numeric_cols, index=0)
            y_feature = st.selectbox("Y-axis", numeric_cols, index=1)
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x_feature], y=df[y_feature], ax=ax)
            st.pyplot(fig)

        elif analysis_type=="Multivariate":
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8,5))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error reading dataset: {e}")

# -----------------------------
# USER INPUT
# -----------------------------
st.sidebar.header("🧍 Enter Patient Details")

age = st.sidebar.number_input("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["Male","Female"])
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
children = st.sidebar.number_input("Children", 0, 10, 1)
smoker = st.sidebar.selectbox("Smoker", ["Yes","No"])

sex_val = 1 if sex=="Male" else 0
smoker_val = 1 if smoker=="Yes" else 0
input_features = np.array([[age, sex_val, bmi, children, smoker_val]], dtype=float)

st.subheader("📝 Input Summary")
st.table({
    "Age": age,
    "Sex": sex,
    "BMI": bmi,
    "Children": children,
    "Smoker": smoker
}, width=200)

# -----------------------------
# MODEL SELECTION
# -----------------------------
st.sidebar.header("⚙️ Choose Model")
available_models = {
    "Linear Regression": "linear_model.pkl",
    "Lasso Regression": "lasso_model.pkl",
    "Ridge Regression": "ridge_model.pkl",
    "ElasticNet Regression": "elastic_model.pkl"
}

model_choice = st.sidebar.selectbox("Select Model", list(available_models.keys()))
model_file = available_models[model_choice]

# -----------------------------
# LOAD MODEL & PREDICT
# -----------------------------
st.header("📂 Prediction")

try:
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        if st.button("🔮 Predict Medical Expense"):
            prediction = model.predict(input_features)[0]
            st.success(f"💰 Predicted Medical Expense using {model_choice}: **${prediction:,.2f}**")
    else:
        st.warning(f"⚠️ Model file not found: {model_file}. Upload it in the app directory.")
except ValueError as ve:
    st.error(f"ValueError: Check the input features. {ve}")
except Exception as e:
    st.error(f"Unexpected error: {e}")
