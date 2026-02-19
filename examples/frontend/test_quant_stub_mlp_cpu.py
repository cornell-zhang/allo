import numpy as np
import torch
import torch.nn as nn
import allo

torch.set_grad_enabled(False)
torch.manual_seed(0)

# This test is designed for "quantize/dequantize module support" in the frontend:
# - Uses QuantStub / DeQuantStub modules (common in quantization flows)
# - Keeps the core compute as standard FP32 ops (Linear/ReLU/Linear)
# Expected behavior (initial support): treat quant/dequant as identity and match PyTorch.


class QuantStubMLP(nn.Module):
    def __init__(self, in_dim=16, hid=32, out_dim=8):
        super().__init__()
        from torch.ao.quantization import QuantStub, DeQuantStub

        self.quant = QuantStub()
        self.fc1 = nn.Linear(in_dim, hid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid, out_dim)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)     # quant boundary (module)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dequant(x)   # dequant boundary (module)
        return x


def main():
    m = QuantStubMLP().eval()
    x = torch.randn(4, 16, dtype=torch.float32)

    # PyTorch golden
    golden = m(x).detach().numpy()

    # Allo compile + run on CPU (LLVM)
    llvm_mod = allo.frontend.from_pytorch(m, example_inputs=[x], target="llvm")
    out = llvm_mod(x.detach().numpy())

    np.testing.assert_allclose(out, golden, rtol=1e-5, atol=1e-6)
    print("PASS: QuantStub/DeQuantStub MLP compiled by Allo and matches PyTorch.")


if __name__ == "__main__":
    main()
