"""
Setup script for Streamlit Cloud deployment.
This script prepares the required data and trains the model when deployed on Streamlit Cloud.
"""
import os
import subprocess
import sys

def run_setup():
    print("Setting up Neural Additive Model for Streamlit Cloud...")
    
    # Check if required files exist
    if os.path.exists("data/X_train.csv") and os.path.exists("models/trained_nam.pth"):
        print("Required files already exist. Skipping setup.")
        return
    
    print("Required files not found. Running data preparation and model training...")
    
    # Run data preparation
    print("Running data preparation...")
    result = subprocess.run([sys.executable, "scripts/data_prep.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error in data preparation:")
        print(result.stderr)
        return
    print("Data preparation completed.")
    
    # Run model training
    print("Training model...")
    result = subprocess.run([sys.executable, "scripts/train_nam.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error in model training:")
        print(result.stderr)
        return
    print("Model training completed.")
    
    print("Setup completed successfully!")

if __name__ == "__main__":
    run_setup()