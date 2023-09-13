# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import allo
import numpy as np
from transformers import AutoConfig, BertLayer


batch_size = 2
seq_len = 512
hidden_size = 768

config = AutoConfig.from_pretrained("bert-base-uncased")
model = BertLayer(config).eval()
print(model)

# trace module
example_inputs = [torch.rand(batch_size, seq_len, hidden_size)]
concrete_args = {
    "past_key_value": None,
    "attention_mask": None,
    "head_mask": None,
    "output_attentions": False,
}
llvm_mod = allo.frontend.from_pytorch(
    model, example_inputs=example_inputs, concrete_args=concrete_args, verbose=True
)

golden = model(*example_inputs)
np_inputs = [x.detach().numpy() for x in example_inputs]
res = llvm_mod(*np_inputs)
np.testing.assert_allclose(res, golden[0].detach().numpy(), atol=1e-3)
