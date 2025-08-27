import streamlit as st
import pandas as pd
import torch
from nam.model import NAM
import numpy as np

st.title("Neural Additive Model (NAM) Predictor & Feature Effect Visualizer")

# Load model
HIDDEN_DIM = 32
MODEL_PATH = "models/trained_nam.pth"
FEATURES_PATH = "data/X_train.csv"  # To get feature names
input_dim = pd.read_csv(FEATURES_PATH).shape[1]

@st.cache_resource
def load_model():
    model = NAM(input_dim, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()
feature_names = pd.read_csv(FEATURES_PATH).columns

uploaded_file = st.file_uploader("Upload your CSV (same features as training)", type='csv')
if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("Preview of your uploaded data:", user_df.head())
    # Predict
    with torch.no_grad():
        X = torch.tensor(user_df.values, dtype=torch.float32)
        logits = model(X).squeeze(1)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)
        user_df["Prob_Diabetes"] = probs
        user_df["Prediction"] = preds
        st.write("Results:", user_df)

st.header("Feature Effect Curves (Interpretability)")
import matplotlib.pyplot as plt
for i, name in enumerate(feature_names):
    x_vals = np.linspace(-2, 2, 100)
    X_dummy = np.zeros((100, len(feature_names)))
    X_dummy[:, i] = x_vals
    x_tensor = torch.tensor(X_dummy, dtype=torch.float32)
    with torch.no_grad():
        effect = model.feature_nets[i](x_tensor[:, i:i+1]).numpy().squeeze()
    fig, ax = plt.subplots()
    ax.plot(x_vals, effect)
    ax.set_title(f"Effect Curve: {name}")
    ax.set_xlabel(name)
    ax.set_ylabel("f(x)")
    st.pyplot(fig)
