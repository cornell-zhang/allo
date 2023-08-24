# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import allo

IC = 3
OC = 16
IH = 32
IW = 32

# 1. Build your own model
class Model(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Model, self).__init__()

        self.conv = nn.Conv2d(
            kernel_size=3, in_channels=in_ch, out_channels=out_ch, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.weight = nn.Parameter(
            torch.randint(0, 2, (OC, IC, 3, 3), dtype=torch.float32) * 2 - 1
        )
        self.bn.weight = nn.Parameter(torch.rand(OC, dtype=torch.float32))
        self.bn.bias = nn.Parameter(torch.rand(OC, dtype=torch.float32))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


model = Model(IC, OC)
model.eval()
mod = allo.frontend.from_pytorch(model)
