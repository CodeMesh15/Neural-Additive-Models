import torch.nn as nn

def nam_feature_layer(hidden_dim=32):
    """
    Factory for a standard feature network block.
    Can be enhanced with dropout, batchnorm etc.
    """
    return nn.Sequential(
        nn.Linear(1, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )

# Example usage in NAM model: 
# Replace in NAM.__init__ if modularizing further.
