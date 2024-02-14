import torch
import torch.nn as nn
import torch.nn.functional as F
import allo

in_channels = 3
out_channels = 10
kernel_size = 3

N = 2 # batch size
C = 3 # num of channels
H = 8 # image height
W = 5 # image width

out_features = 2

class ImageClassify(nn.Module):
    def __init__(self):
        super(ImageClassify, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.linear = nn.Linear(C * H * W, out_features)
        self.relu = F.relu
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(-1, C * H * W)
        x = self.linear(x)
        return x

model = ImageClassify()
model.eval()
example_inputs = [torch.rand(N, C, H, W)]
llvm_mod = allo.frontend.from_pytorch(
    model, example_inputs=example_inputs, verbose=False
)
golden = model(*example_inputs)
np_inputs = [x.detach().numpy() for x in example_inputs]
res = llvm_mod(*np_inputs)
torch.testing.assert_close(res, golden.detach().numpy())
