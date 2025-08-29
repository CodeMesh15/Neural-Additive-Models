import torch
import torch.nn as nn

class NAM(nn.Module):
    """
    Neural Additive Model: Each feature has a small sub-network. 
    Feature-wise sub-networks' outputs are summed for the final prediction.
    Suitable for binary classification (can be easily adapted for regression).
    """
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super().__init__()
        self.feature_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(input_dim)
        ])
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        # x: [batch_size, input_dim]
        # Compute each feature’s effect independently, then sum them
        outs = [fn(x[:, i:i+1]) for i, fn in enumerate(self.feature_nets)]
        out_sum = torch.stack(outs, dim=2).sum(dim=2).squeeze()  # shape: [batch_size] or [batch_size, output_dim]
        return out_sum + self.bias  # Output is logits for binary (use sigmoid in loss/eval)

