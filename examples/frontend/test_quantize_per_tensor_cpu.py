import numpy as np
import torch
import torch.nn as nn
import allo

torch.set_grad_enabled(False)
torch.manual_seed(0)

# This test introduces real quantize/dequantize ops:
#   torch.quantize_per_tensor(x, ...).dequantize()
# The intended "initial support" in the frontend is typically to treat
# quantize+dequantize as a no-op boundary (float in/out) so translation doesn't crash.

class QDQMLP(nn.Module):
    def __init__(self, in_dim=16, hid=32, out_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid, out_dim)

        # fixed quant params for determinism
        self.scale = 0.05
        self.zero_point = 0
        self.dtype = torch.qint8

    def forward(self, x):
        # quantize -> dequantize boundary (functions/methods, not stubs)
        xq = torch.quantize_per_tensor(x, self.scale, self.zero_point, self.dtype)
        x = xq.dequantize()

        x = self.fc1(x)
        x = self.relu(x)

        yq = torch.quantize_per_tensor(x, self.scale, self.zero_point, self.dtype)
        x = yq.dequantize()

        x = self.fc2(x)
        return x

def main():
    m = QDQMLP().eval()
    x = torch.randn(4, 16, dtype=torch.float32)

    golden = m(x).detach().numpy()

    llvm_mod = allo.frontend.from_pytorch(m, example_inputs=[x], target="llvm")
    out = llvm_mod(x.detach().numpy())

    np.testing.assert_allclose(out, golden, rtol=1e-5, atol=1e-6)
    print("PASS: quantize_per_tensor/dequantize model compiled and matches PyTorch.")

if __name__ == "__main__":
    main()
