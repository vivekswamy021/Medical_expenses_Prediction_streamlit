import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Medical Expense Prediction Dashboard", page_icon="💰", layout="wide")

# ------------------------------------------------
# Title
# ------------------------------------------------
st.title("💰 Medical Expense Prediction Dashboard")
st.markdown("""
This Streamlit app predicts **medical expenses** based on patient details  
and visualizes **EDA insights** using charts from your ML notebook.

---

### Models Used
- **Linear Regression**
- **Lasso Regression**
- **Ridge Regression**
- **Elastic Net**
""")

# ------------------------------------------------
# Sidebar Upload Section
# ------------------------------------------------
st.sidebar.header("⚙️ Upload Data and Models")

data_file = st.sidebar.file_uploader("Upload `insurance.xlsx`", type=["xlsx"])
if data_file is not None:
    df = pd.read_excel(data_file)
else:
    st.sidebar.warning("Upload insurance.xlsx to visualize EDA.")
    df = None

# Model upload
st.sidebar.subheader("Upload Model Files (.pkl)")
linear_model_file = st.sidebar.file_uploader("Linear Model", type=["pkl"])
lasso_model_file = st.sidebar.file_uploader("Lasso Model", type=["pkl"])
ridge_model_file = st.sidebar.file_uploader("Ridge Model", type=["pkl"])
elastic_model_file = st.sidebar.file_uploader("ElasticNet Model", type=["pkl"])

# Load default or uploaded models
def load_model(file, default_name):
    try:
        if file is not None:
            return joblib.load(file)
        return joblib.load(default_name)
    except:
        return None

linear_model = load_model(linear_model_file, "linear_model.pkl")
lasso_model = load_model(lasso_model_file, "lasso_model.pkl")
ridge_model = load_model(ridge_model_file, "ridge_model.pkl")
elastic_model = load_model(elastic_model_file, "elastic_model.pkl")

# ------------------------------------------------
# Tabs for EDA and Prediction
# ------------------------------------------------
tab1, tab2 = st.tabs(["📊 Data Visualization (EDA)", "🔮 Predict Medical Expenses"])

# =========================================================
# 📊 TAB 1 — EDA VISUALIZATION
# =========================================================
with tab1:
    st.header("📊 Exploratory Data Analysis")

    if df is not None:
        st.subheader("Basic Data Overview")
        st.dataframe(df.head())

        continous = ['age', 'bmi', 'expenses']
        discrete_categorical = ['sex', 'smoker', 'region']

        st.markdown("### 🧾 Summary Statistics")
        st.write(df[continous].describe())

        st.markdown("### 🧍‍♂️ Categorical Summary")
        st.write(df[discrete_categorical].describe())

        # Pairplot
        st.subheader("Pairplot for Continuous Features")
        sns.pairplot(df[continous])
        st.pyplot(plt.gcf())
        plt.clf()

        # Correlation heatmap
        st.subheader("Heatmap of Continuous Features")
        corr = df[continous].corr()
        plt.figure(figsize=(5, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())
        plt.clf()

        # Scatter sex vs expenses
        st.subheader("Expenses by Gender")
        plt.figure(figsize=(5, 4))
        sns.scatterplot(x='sex', y='expenses', data=df)
        st.pyplot(plt.gcf())
        plt.clf()

        # Histogram for sex
        st.subheader("Distribution of Gender")
        plt.figure(figsize=(5, 4))
        sns.histplot(df['sex'])
        st.pyplot(plt.gcf())
        plt.clf()

        # BMI vs expenses with hue=sex
        st.subheader("BMI vs Expenses by Gender")
        plt.figure(figsize=(5, 4))
        sns.scatterplot(x='bmi', y='expenses', hue='sex', data=df)
        st.pyplot(plt.gcf())
        plt.clf()

    else:
        st.warning("Please upload your `insurance.xlsx` file to view EDA visualizations.")

# =========================================================
# 🔮 TAB 2 — MODEL PREDICTIONS
# =========================================================
with tab2:
    st.header("🔮 Medical Expense Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=35)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=28.5)

    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["yes", "no"])

    with col3:
        children = st.number_input("Children", min_value=0, max_value=10, value=2)
        region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

    def preprocess(age, sex, bmi, children, smoker):
        df = pd.DataFrame({
            'age': [age],
            'sex': [1 if sex == 'male' else 0],
            'bmi': [bmi],
            'children': [children],
            'smoker': [1 if smoker == 'yes' else 0]
        })
        return df

    if st.button("💡 Predict Expenses"):
        df_input = preprocess(age, sex, bmi, children, smoker)

        st.subheader("🧾 Input Data")
        st.dataframe(df_input)

        # Predictions
        preds = {}
        if linear_model:
            preds["Linear Regression"] = round(linear_model.predict(df_input)[0], 2)
        if lasso_model:
            preds["Lasso Regression"] = round(lasso_model.predict(df_input.drop(columns=['sex'], errors='ignore'))[0], 2)
        if ridge_model:
            preds["Ridge Regression"] = round(ridge_model.predict(df_input)[0], 2)
        if elastic_model:
            preds["Elastic Net"] = round(elastic_model.predict(df_input)[0], 2)

        results_df = pd.DataFrame(list(preds.items()), columns=["Model", "Predicted Expense ($)"])
        st.subheader("💰 Model Predictions")
        st.table(results_df)

        # Visualization: Comparison Chart
        st.subheader("📈 Model Comparison Chart")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Model", y="Predicted Expense ($)", data=results_df, ax=ax, palette="viridis")
        plt.xticks(rotation=15)
        st.pyplot(fig)

        best_model = max(preds, key=preds.get)
        st.success(f"✅ Best Model: **{best_model}** (${preds[best_model]})")

st.markdown("---")
