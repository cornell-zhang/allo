# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch.nn as nn
import allo


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = x + y
        x = F.relu(x)
        return x


model = Model()
model.eval()
example_inputs = [torch.rand(1, 3, 10, 10), torch.rand(1, 3, 10, 10)]
llvm_mod = allo.frontend.from_pytorch(
    model, example_inputs=example_inputs, verbose=True
)

golden = model(*example_inputs)
np_inputs = [x.detach().numpy() for x in example_inputs]
res = llvm_mod(*np_inputs)
torch.testing.assert_allclose(res, golden.detach().numpy())
