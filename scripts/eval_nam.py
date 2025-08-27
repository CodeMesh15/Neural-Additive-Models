import torch
import pandas as pd
from nam.model import NAM
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Parameters
MODEL_PATH = 'models/trained_nam.pth'
X_TEST_PATH = 'data/X_test.csv'
Y_TEST_PATH = 'data/y_test.csv'
HIDDEN_DIM = 32  # Should match the value used in training

# Load test data
X_test = pd.read_csv(X_TEST_PATH).values
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

# Load trained NAM model
input_dim = X_test.shape[1]
model = NAM(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Generate predictions
X_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    logits = model(X_tensor).squeeze(1)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)

# Evaluate
accuracy = accuracy_score(y_test, preds)
roc_auc = roc_auc_score(y_test, probs)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")
print("\nDetailed classification report:")
print(classification_report(y_test, preds, digits=4))
