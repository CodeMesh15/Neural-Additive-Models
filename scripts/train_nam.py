import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nam.model import NAM
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001

# Load data
X_train = pd.read_csv('data/X_train.csv').values
y_train = pd.read_csv('data/y_train.csv').values.ravel()

# Prepare Torch tensors/dataset
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate model (input_dim = #features)
model = NAM(input_dim=X_train.shape[1], hidden_dim=32)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(Xb)  # The model already returns the correct shape
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), 'models/trained_nam.pth')
print("NAM model trained and saved!")
