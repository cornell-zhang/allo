import torch
import numpy as np
from allo.frontend.pytorch import from_pytorch  # adjust import if different

class QDQ(torch.nn.Module):
    def forward(self, x):
        # qint8 fake-quant via QDQ
        scale = torch.tensor(0.1, dtype=torch.float32)
        zp = 3
        xq = torch.quantize_per_tensor(x, scale=scale.item(), zero_point=zp, dtype=torch.qint8)
        return xq.dequantize()

def ref(x_np):
    x = x_np.astype(np.float32)
    s = 0.1
    z = 3.0
    qmin, qmax = -128.0, 127.0
    y = (np.clip(np.round(x / s) + z, qmin, qmax) - z) * s  # numpy round is bankers
    return y.astype(np.float32)

def main():
    m = QDQ().eval()

    x_t = torch.tensor([[-1.55, -1.5, -0.05, 0.05, 1.5, 1.55]], dtype=torch.float32)
    x_np = x_t.detach().cpu().numpy()

    y_pt = m(x_t).detach().cpu().numpy()

    allo_mod = from_pytorch(m, (x_t,), target="llvm", mode="csim", verbose=True)

    y_allo = np.array(allo_mod(x_np))

    print("pt :", y_pt)
    print("allo:", y_allo)


if __name__ == "__main__":
    main()
