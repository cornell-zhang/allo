# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import allo
import numpy as np
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertLayer, BertEncoder


def test_bert_module(module):
    batch_size = 2
    seq_len = 512
    hidden_size = 768

    example_inputs = [torch.rand(batch_size, seq_len, hidden_size)]
    llvm_mod = allo.frontend.from_pytorch(
        module,
        example_inputs=example_inputs,
        verbose=False,
    )

    golden = module(*example_inputs)
    np_inputs = [x.detach().numpy() for x in example_inputs]
    res = llvm_mod(*np_inputs)
    np.testing.assert_allclose(res, golden[0].detach().numpy(), atol=1e-3)


config = AutoConfig.from_pretrained("bert-base-uncased")
bert_layer_module = BertLayer(config).eval()
test_bert_module(bert_layer_module)
del bert_layer_module
bert_encoder_module = BertEncoder(config).eval()
test_bert_module(bert_encoder_module)
del bert_encoder_module
