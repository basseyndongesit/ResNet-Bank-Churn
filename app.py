# 1. IMPORTS
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle

# 2. MODEL DEFINITIONS (REQUIRED)
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self._to_linear = None
        self._get_output()

        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def _get_output(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 30)  # number of features
            x = self.conv(x)
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return torch.relu(out)


class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = ResidualBlock(1, 32)
        self.pool1 = nn.MaxPool1d(2)

        self.layer2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool1d(2)

        self.layer3 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool1d(2)

        self.layer4 = ResidualBlock(128, 256)
        self.pool4 = nn.MaxPool1d(2)

        self._to_linear = None
        self._get_output()

        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def _get_output(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 30)
            x = self.pool1(self.layer1(x))
            x = self.pool2(self.layer2(x))
            x = self.pool3(self.layer3(x))
            x = self.pool4(self.layer4(x))
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.pool4(self.layer4(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 3. LOAD MODELS (FIXED)
@st.cache_resource
def load_models():
    cnn_model = DeepCNN()
    resnet_model = ResNet1D()

    cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
    resnet_model.load_state_dict(torch.load("resnet_model.pth", map_location="cpu"))

    cnn_model.eval()
    resnet_model.eval()

    import pickle

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("features.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    return cnn_model, resnet_model, scaler, feature_columns

cnn_model, resnet_model, scaler, feature_columns = load_models()

# 4. STREAMLIT UI
st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("🏦 Customer Churn Prediction System")
st.markdown("### Deep Learning Powered (CNN vs ResNet)")

# =========================================
# 5. USER INPUT
# =========================================
st.sidebar.header("Enter Customer Details")

def user_input():
    data = {
        "Customer_Age": st.sidebar.slider("Age", 18, 80, 35, key="input_age"),
        "Dependent_count": st.sidebar.slider("Dependents", 0, 5, 2, key="input_dependents"),
        "Months_on_book": st.sidebar.slider("Months with Bank", 6, 60, 24, key="input_months"),
        "Total_Relationship_Count": st.sidebar.slider("Products Held", 1, 6, 3, key="input_products"),
        "Months_Inactive_12_mon": st.sidebar.slider("Inactive Months", 0, 12, 2, key="input_inactive"),
        "Contacts_Count_12_mon": st.sidebar.slider("Contacts", 0, 6, 2, key="input_contacts"),
        "Credit_Limit": st.sidebar.number_input("Credit Limit", 1000, 50000, 10000, key="input_credit"),
        "Total_Revolving_Bal": st.sidebar.number_input("Balance", 0, 5000, 1000, key="input_balance")
    }
    return pd.DataFrame([data])

# 6. PREPROCESS
def preprocess(df):
    # Create full feature dataframe
    full_df = pd.DataFrame(columns=feature_columns)

    # Fill user inputs
    for col in df.columns:
        full_df[col] = df[col]

    # Fill missing columns with 0
    full_df = full_df.fillna(0)

    df_scaled = scaler.transform(full_df)
    return torch.tensor(df_scaled, dtype=torch.float32).unsqueeze(1)

# 7. PREDICTION
def predict(model, x):
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1).numpy()[0]
        pred = np.argmax(prob)
    return pred, prob[1]

input_df = user_input()

input_tensor = preprocess(input_df)

cnn_pred, cnn_prob = predict(cnn_model, input_tensor)
resnet_pred, resnet_prob = predict(resnet_model, input_tensor)

# 8. DISPLAY RESULTS
st.subheader("📊 Predictions")

col1, col2 = st.columns(2)

def label(pred):
    return "Attrited Customer" if pred == 1 else "Existing Customer"

with col1:
    st.markdown("### 🧠 CNN")
    st.metric("Prediction", label(cnn_pred))
    st.progress(float(cnn_prob))
    st.write(f"Confidence: {cnn_prob:.2%}")

with col2:
    st.markdown("### 🚀 ResNet")
    st.metric("Prediction", label(resnet_pred))
    st.progress(float(resnet_prob))
    st.write(f"Confidence: {resnet_prob:.2%}")

# 9. BUSINESS IMPACT
# st.subheader("💰 Business Impact")

# st.success("Estimated Annual ROI: **$360,000**")

# st.markdown("""
# - Retained Customers: **432**
# - Revenue Saved: **$432,000**
# - Cost: **$72,000**
# """)
