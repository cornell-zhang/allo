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
example_inputs = [torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)]
mod = allo.frontend.from_pytorch(model, example_inputs=example_inputs)
