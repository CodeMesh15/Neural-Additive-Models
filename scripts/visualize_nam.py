import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nam.model import NAM

X_train = pd.read_csv('data/X_train.csv')
model = NAM(input_dim=X_train.shape[1], hidden_dim=32)
model.load_state_dict(torch.load('models/trained_nam.pth'))
model.eval()

feature_names = X_train.columns
for i, name in enumerate(feature_names):
    x_min, x_max = X_train[name].min(), X_train[name].max()
    x_vals = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    X_dummy = np.zeros((100, X_train.shape[1]))
    X_dummy[:, i] = x_vals.squeeze()
    x_tensor = torch.tensor(X_dummy, dtype=torch.float32)
    with torch.no_grad():
        effects = model.feature_nets[i](x_tensor[:, i:i+1]).numpy().squeeze()
    plt.plot(x_vals, effects)
    plt.title(f"Feature Effect: {name}")
    plt.xlabel(name)
    plt.ylabel("f(x)")
    plt.show()
