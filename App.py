import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# CONFIGURATION
# -------------------------------
st.set_page_config(page_title="ML Model Dashboard", layout="wide")

# Replace this with your actual GitHub raw base URL
# Example: https://raw.githubusercontent.com/<username>/<repo>/main/
GITHUB_BASE_URL = "https://raw.githubusercontent.com/<your-username>/<your-repo-name>/main/"

MODEL_FILES = {
    "Linear Regression": "linear_model.pkl",
    "Lasso Regression": "lasso_model.pkl",
    "Ridge Regression": "ridge_model.pkl",
    "ElasticNet Regression": "elastic_model.pkl"
}

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

@st.cache_resource
def load_model_from_github(file_name):
    """Load model pickle file from GitHub"""
    url = GITHUB_BASE_URL + file_name
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to load {file_name} from GitHub.")
        return None
    return pickle.load(io.BytesIO(response.content))

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, color="skyblue", ax=ax)
    ax.set_title("Residuals Distribution")
    st.pyplot(fig)

def plot_actual_vs_pred(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, color='green')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

def plot_coefficients(model, feature_names):
    fig, ax = plt.subplots(figsize=(8, 4))
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_})
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, ax=ax, palette="viridis")
    ax.set_title("Model Feature Importance (Coefficients)")
    st.pyplot(fig)

# -------------------------------
# APP UI
# -------------------------------

st.title("📊 ML Model Prediction Dashboard")

with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model_name = st.selectbox("Select Model", list(MODEL_FILES.keys()))
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        st.info("Upload a CSV file to start predictions.")
        df = None

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model_from_github(MODEL_FILES[selected_model_name])

if model is not None and df is not None:
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Assuming last column is the target variable
    target_col = st.selectbox("Select Target Column (if available)", options=df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Predictions
    y_pred = model.predict(X)

    st.subheader("📈 Prediction Results")
    result_df = df.copy()
    result_df["Predicted"] = y_pred
    st.dataframe(result_df.head())

    # -------------------------------
    # METRICS
    # -------------------------------
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    st.metric("R² Score", f"{r2:.4f}")
    st.metric("MSE", f"{mse:.4f}")

    # -------------------------------
    # VISUALIZATIONS
    # -------------------------------
    st.subheader("📊 Charts and Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        plot_actual_vs_pred(y, y_pred)
    with col2:
        plot_residuals(y, y_pred)

    st.subheader("📉 Model Coefficients")
    plot_coefficients(model, X.columns)

else:
    st.warning("👈 Please upload a dataset and select a model to continue.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Matplotlib, and scikit-learn.")
