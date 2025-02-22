# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch.nn as nn
import allo


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 32)  # 8*16 * 32*16
        self.linear2 = torch.nn.Linear(32, 10)

    def forward(self, data):
        out = self.linear1(data)
        out = self.linear2(out)
        out = F.relu(out)
        return out


model = MLP()
model.eval()
example_inputs = [torch.rand(8, 16)]
llvm_mod = allo.frontend.from_pytorch(
    model, example_inputs=example_inputs, verbose=True
)
golden = model(*example_inputs)
np_inputs = [x.detach().numpy() for x in example_inputs]
res = llvm_mod(*np_inputs)
torch.testing.assert_close(res, golden.detach().numpy())
print("Passed!")
