import numpy as np
import torch
import torch.nn as nn
import allo

torch.set_grad_enabled(False)
torch.manual_seed(0)


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
        # explicit quantize -> dequantize boundaries
        xq = torch.quantize_per_tensor(x, self.scale, self.zero_point, self.dtype)
        x = xq.dequantize()

        x = self.fc1(x)
        x = self.relu(x)

        yq = torch.quantize_per_tensor(x, self.scale, self.zero_point, self.dtype)
        x = yq.dequantize()

        x = self.fc2(x)
        return x


def print_fx_graph(model):
    print("=" * 80)
    print("FX GRAPH")
    print("=" * 80)
    traced = torch.fx.symbolic_trace(model)
    print(traced.graph)
    print()


def test_mlir_only(model, x):
    print("=" * 80)
    print("TEST 1: FRONTEND -> MLIR")
    print("=" * 80)
    mlir_mod = allo.frontend.from_pytorch(
        model,
        example_inputs=[x],
        target="mlir",
        verbose=False,
    )
    print("PASS: MLIR generation succeeded.")
    print()
    print("Generated MLIR module:")
    print(mlir_mod.module)
    print()
    return mlir_mod


def test_llvm_execution(model, x):
    print("=" * 80)
    print("TEST 2: FRONTEND -> LLVM -> CPU EXECUTION")
    print("=" * 80)

    golden = model(x).detach().numpy()

    llvm_mod = allo.frontend.from_pytorch(
    model,
    example_inputs=[x],
    target="llvm",
    verbose=False,
    weights_as_args=True,
    )
    print("PASS: LLVM build succeeded.")
    print()

    out = llvm_mod(x.detach().numpy())
    out = np.asarray(out)

    max_abs_err = np.max(np.abs(out - golden))
    mean_abs_err = np.mean(np.abs(out - golden))

    print("Output shape:", out.shape)
    print("Golden shape:", golden.shape)
    print("max abs err:", max_abs_err)
    print("mean abs err:", mean_abs_err)
    print()

    np.testing.assert_allclose(out, golden, rtol=1e-4, atol=1e-5)
    print("PASS: LLVM output matches PyTorch.")
    print()
    return out, golden


def main():
    m = QDQMLP().eval()
    x = torch.randn(4, 16, dtype=torch.float32)

    print_fx_graph(m)
    test_mlir_only(m, x)
    test_llvm_execution(m, x)

    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()