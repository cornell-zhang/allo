import numpy as np
import torch
import torch.nn as nn
import allo

torch.set_grad_enabled(False)
torch.manual_seed(0)

class MLP(nn.Module):
    def __init__(self, in_dim=16, hid=32, out_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.relu = nn.ReLU()          # <-- module, not torch.relu
        self.fc2 = nn.Linear(hid, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)              # <-- call_module
        x = self.fc2(x)
        return x

m = MLP().eval()
x = torch.randn(4, 16, dtype=torch.float32)

golden = m(x).detach().numpy()

llvm_mod = allo.frontend.from_pytorch(m, example_inputs=[x], target="llvm")
res = llvm_mod(x.detach().numpy())

np.testing.assert_allclose(res, golden, rtol=1e-5, atol=1e-6)
print("Passed: Allo(LLVM CPU) matches PyTorch FP32 MLP.")
