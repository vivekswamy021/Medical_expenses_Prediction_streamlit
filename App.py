import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------
# Load all saved models
# -----------------------------------------------
linear_model = joblib.load("linear_model.pkl")
lasso_model = joblib.load("lasso_model.pkl")
ridge_model = joblib.load("ridge_model.pkl")
elastic_model = joblib.load("elastic_model.pkl")

st.set_page_config(page_title="Medical Expenses Predictor", page_icon="💰", layout="centered")

# -----------------------------------------------
# Title and Description
# -----------------------------------------------
st.title("💰 Medical Expenses Prediction App")
st.markdown("""
This app predicts **medical expenses** based on patient information using:
- Linear Regression  
- Lasso Regression  
- Ridge Regression  
- Elastic Net  

Upload your trained models or use the default ones stored in this app.
""")

# -----------------------------------------------
# Sidebar Inputs
# -----------------------------------------------
st.sidebar.header("🩺 Enter Patient Details")

age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=35)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=28.5)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=2)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# -----------------------------------------------
# Preprocessing Input Data
# -----------------------------------------------
def preprocess_input(age, sex, bmi, children, smoker, region):
    df = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'male' else 0],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == 'yes' else 0],
        # region dropped in model
    })
    return df

# -----------------------------------------------
# Prediction
# -----------------------------------------------
if st.sidebar.button("🔍 Predict Expenses"):
    df_input = preprocess_input(age, sex, bmi, children, smoker, region)
    
    # Model predictions
    linear_pred = linear_model.predict(df_input)[0]
    lasso_pred = lasso_model.predict(df_input.drop(columns=['sex'], errors='ignore'))[0]
    ridge_pred = ridge_model.predict(df_input)[0]
    elastic_pred = elastic_model.predict(df_input)[0]
    
    st.subheader("📊 Prediction Results")
    st.write("### Input Data")
    st.dataframe(df_input)

    results = {
        "Linear Regression": round(linear_pred, 2),
        "Lasso Regression": round(lasso_pred, 2),
        "Ridge Regression": round(ridge_pred, 2),
        "Elastic Net": round(elastic_pred, 2)
    }

    st.write("### 💵 Predicted Medical Expenses")
    st.table(pd.DataFrame(results.items(), columns=["Model", "Predicted Expense ($)"]))

    # Best model suggestion
    best_model_name = max(results, key=results.get)
    st.success(f"✅ Best predicted model: **{best_model_name}** (${results[best_model_name]})")

# -----------------------------------------------
# Footer
# -----------------------------------------------
st.markdown("---")
st.caption("Developed by Vivek Swamy | Streamlit + Scikit-learn + Colab")

