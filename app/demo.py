import streamlit as st
import pandas as pd
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nam.model import NAM
import numpy as np

st.title("Neural Additive Model (NAM) Predictor & Feature Effect Visualizer")

# Add explanation section
st.markdown("""
## About Neural Additive Models (NAMs)

Neural Additive Models combine the predictive power of neural networks with the interpretability of traditional models. 
Unlike black-box models, NAMs allow you to see exactly how each feature contributes to predictions through **effect curves**.

### What are Effect Curves?

Effect curves are visual representations showing how each individual feature contributes to the model's predictions:
- **X-axis**: The range of values for that specific feature
- **Y-axis**: The contribution (effect) of that feature to the model's prediction

### Why are Effect Curves Important?

1. **Interpretability**: See exactly how each feature influences predictions
2. **Trust**: Medical professionals can understand and trust the model's decisions
3. **Debugging**: Identify if a feature is behaving unexpectedly
4. **Insights**: Domain experts can validate if the relationships make sense

This is the key innovation of NAMs - they combine the predictive power of neural networks with the interpretability of simpler models.
""")

# Load model
HIDDEN_DIM = 32
MODEL_PATH = "models/trained_nam.pth"
FEATURES_PATH = "data/X_train.csv"  # To get feature names

try:
    input_dim = pd.read_csv(FEATURES_PATH).shape[1]
    
    @st.cache_resource
    def load_model():
        model = NAM(input_dim, hidden_dim=HIDDEN_DIM)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        return model

    model = load_model()
    feature_names = pd.read_csv(FEATURES_PATH).columns

    st.header("Upload Data for Prediction")
    uploaded_file = st.file_uploader("Upload your CSV (same features as training)", type='csv')
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("Preview of your uploaded data:", user_df.head())
        # Predict
        with torch.no_grad():
            X = torch.tensor(user_df.values, dtype=torch.float32)
            logits = model(X)  # The model already returns the correct shape
            probs = torch.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(int)
            user_df["Prob_Diabetes"] = probs
            user_df["Prediction"] = preds
            st.write("Results:", user_df)

    st.header("Feature Effect Curves (Interpretability)")
    st.markdown("""
    These curves show how each feature contributes to the model's predictions. 
    The y-axis shows the effect on the prediction, and the x-axis shows the feature values.
    """)
    
    # Only show a few feature effects to avoid overwhelming the UI
    num_features_to_show = min(5, len(feature_names))
    for i, name in enumerate(feature_names[:num_features_to_show]):
        x_vals = np.linspace(-2, 2, 100)
        X_dummy = np.zeros((100, len(feature_names)))
        X_dummy[:, i] = x_vals
        x_tensor = torch.tensor(X_dummy, dtype=torch.float32)
        with torch.no_grad():
            effect = model.feature_nets[i](x_tensor[:, i:i+1]).numpy().squeeze()
        st.subheader(f"Effect Curve: {name}")
        chart_data = pd.DataFrame({
            'x': x_vals,
            'effect': effect
        })
        st.line_chart(chart_data.set_index('x'))
        
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.info("Please make sure you have run the data preparation and model training scripts first.")
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please check the console for more details.")
