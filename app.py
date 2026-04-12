# 1. IMPORTS

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

# 2. LOAD SAVED MODELS + SCALER

@st.cache_resource
def load_models():
    cnn_model = torch.load("cnn_model.pth", map_location=torch.device('cpu'))
    resnet_model = torch.load("resnet_model.pth", map_location=torch.device('cpu'))

    cnn_model.eval()
    resnet_model.eval()

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return cnn_model, resnet_model, scaler

cnn_model, resnet_model, scaler = load_models()

# 3. PAGE CONFIG

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("🏦 Customer Churn Prediction System")
st.markdown("### Powered by Deep Learning (CNN vs ResNet)")

# 4. SIDEBAR INPUTS

st.sidebar.header("Customer Information")

def user_input():
    data = {}

    # Example inputs (adjust based on dataset columns)
    data['Customer_Age'] = st.sidebar.slider("Age", 18, 80, 35)
    data['Dependent_count'] = st.sidebar.slider("Dependents", 0, 5, 2)
    data['Months_on_book'] = st.sidebar.slider("Months with Bank", 6, 60, 24)
    data['Total_Relationship_Count'] = st.sidebar.slider("Products Held", 1, 6, 3)
    data['Months_Inactive_12_mon'] = st.sidebar.slider("Inactive Months", 0, 12, 2)
    data['Contacts_Count_12_mon'] = st.sidebar.slider("Contacts Last Year", 0, 6, 2)
    data['Credit_Limit'] = st.sidebar.number_input("Credit Limit", 1000, 50000, 10000)
    data['Total_Revolving_Bal'] = st.sidebar.number_input("Revolving Balance", 0, 5000, 1000)

    return pd.DataFrame([data])

input_df = user_input()

# 5. PREPROCESS INPUT (SAME PIPELINE)

def preprocess_input(df):
    df_scaled = scaler.transform(df)
    tensor = torch.tensor(df_scaled, dtype=torch.float32).unsqueeze(1)
    return tensor

input_tensor = preprocess_input(input_df)

# 6. PREDICTION FUNCTION

def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1).numpy()[0]
        pred = np.argmax(prob)
    return pred, prob[1]  # probability of churn

cnn_pred, cnn_prob = predict(cnn_model, input_tensor)
resnet_pred, resnet_prob = predict(resnet_model, input_tensor)

# 7. DISPLAY RESULTS

st.subheader("📊 Prediction Results")

col1, col2 = st.columns(2)

def format_result(pred, prob):
    label = "Attrited Customer" if pred == 1 else "Existing Customer"
    return label, prob

cnn_label, cnn_conf = format_result(cnn_pred, cnn_prob)
resnet_label, resnet_conf = format_result(resnet_pred, resnet_prob)

with col1:
    st.markdown("### 🧠 Deep CNN")
    st.metric("Prediction", cnn_label)
    st.progress(float(cnn_conf))
    st.write(f"Confidence: **{cnn_conf:.2%}**")

with col2:
    st.markdown("### 🚀 ResNet (Improved Model)")
    st.metric("Prediction", resnet_label)
    st.progress(float(resnet_conf))
    st.write(f"Confidence: **{resnet_conf:.2%}**")

# 8. MODEL COMPARISON (PRODUCT THINKING)

st.subheader("📈 Model Comparison")

# Example improvement (replace with your real results)
cnn_acc = 0.86
resnet_acc = 0.91

improvement = (resnet_acc - cnn_acc) * 100

st.success(f"✅ ResNet model is **{improvement:.1f}% more accurate** than the CNN model")

# 9. INTERPRETATION

st.subheader("💡 What This Means")

if resnet_pred == 1:
    st.error("⚠️ This customer is likely to churn. Consider retention strategies.")
else:
    st.success("✅ This customer is likely to stay.")

# 10. USER EXPERIENCE ENHANCEMENTS

st.markdown("---")
st.markdown("### 🛠 System Improvements")

st.markdown("""
- 🔍 **Better Accuracy**: ResNet improves prediction reliability  
- 📊 **Confidence Scores**: Shows certainty of predictions  
- ⚖️ **Model Comparison**: Transparency between models  
- 🎯 **Actionable Insight**: Clear churn vs retain signal  
""")
