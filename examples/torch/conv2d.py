# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

import allo
from allo import maxpool

torch.set_printoptions(precision=4)

in_channels = 3
out_channels = 5
kernel_size = 3

pool_size = (2, 3)
stride_size = (1, 1)

out_features = 2

N = 2  # batch size
C = 3  # num of channels
H = 4  # image height
W = 5  # image width


class ImageClassify(nn.Module):
    def __init__(self):
        super(ImageClassify, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(pool_size, stride=stride_size)
        self.linear = nn.Linear(
            out_channels
            * ((H - kernel_size + 1 - pool_size[0]) // stride_size[0] + 1)
            * ((W - kernel_size + 1 - pool_size[1]) // stride_size[1] + 1),
            out_features,
        )
        self.relu = F.relu

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


model = ImageClassify()
model.eval()
example_inputs = [torch.empty(N, C, H, W).uniform_(-10, 10)]
llvm_mod = allo.frontend.from_pytorch(
    model,
    example_inputs=example_inputs,
    verbose=True,
)
print(llvm_mod)
golden = model(*example_inputs)
np_inputs = [x.detach().numpy() for x in example_inputs]
res = llvm_mod(*np_inputs)
torch.testing.assert_close(res, golden.detach().numpy(), atol=1e-1, rtol=1e-2)
print("SUCCESS!")
