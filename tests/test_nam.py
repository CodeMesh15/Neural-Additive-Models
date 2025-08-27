import torch
from nam.model import NAM

def test_forward_shape():
    batch_size = 5
    input_dim = 8
    x = torch.randn(batch_size, input_dim)
    model = NAM(input_dim=input_dim)
    out = model(x)
    assert out.shape[0] == batch_size, "Output should match batch size"
    print("Forward shape test passed.")

if __name__ == "__main__":
    test_forward_shape()

# You can add further tests for data integrity, feature nets, etc.
