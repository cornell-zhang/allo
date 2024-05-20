<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Examples

This folder contains examples of using Allo to design hardware accelerators for various applications.

## PolyBench
[PolyBench](https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1) is a C-based benchmark suite that contains a set of kernels commonly used in scientific computing. We provide Allo implementations under the [`polybench`](polybench/) folder. For example, you can directly run the GEMM kernel using the CPU backend by typing the following command:
```bash
python3 polybench/gemm.py
```

For comparison between Allo and other baseline systems, please refer to our [PLDI'24 artifact repository](https://github.com/cornell-zhang/allo-pldi24-artifact) for more details.


## Transformer Models
We propose an analytical framework to predict the performance of FPGA-based spatial accelerators for large language model (LLM) inference. Please refer to our [FCCM'24 paper](https://arxiv.org/abs/2312.15159) for more details.

We provide the Allo implementation of the Transformer kernel library under [nn.py](https://github.com/cornell-zhang/allo/tree/main/allo/library). Example usage can be found in [test_nn.py](https://github.com/cornell-zhang/allo/tree/main/tests/test_nn.py).

To facilitate the usage of the HLS library, we also provide a script to quickly generate HLS code for each kernel. Currently, we support the following arguments:

| Argument | Description |
| --- | --- |
| `--func` | The kernel name. Supported kernels: `gemm`, `sdp`, `softmax`, `layernorm`, `gelu`. |
| `--dtype` | The data type of the kernel. Supported data types: `float`, `int8`, etc. |
| `--H` | The number of attention heads. |
| `--L` | The input sequence length. |
| `--D` | The hidden dimension. |
| `--Dffn` | The hidden dimension of the FFN layer. |
| `--M0` | The size of systolic array (first dimension). |
| `--M1` | The size of systolic array (second dimension). |

For example, you can generate the HLS code for the int8 GEMM kernel with 16x16 systolic array using the following command:
```bash
python3 transformer_hls.py --func gemm --dtype float --M0 16 --M1 16
```

If you use our framework in your research, please use the following bibtex entry to cite us:
```bibtex
@article{chen2024llmfpga,
    author = {Chen, Hongzheng and Zhang, Jiahao and Du, Yixiao and Xiang, Shaojie and Yue, Zichao and Zhang, Niansong and Cai, Yaohui and Zhang, Zhiru},
    title = {Understanding the Potential of FPGA-Based Spatial Acceleration for Large Language Model Inference},
    journal = {ACM Trans. Reconfigurable Technol. Syst.},
    year = {2024},
    month = {apr},
    url = {https://doi.org/10.1145/3656177},
    doi = {10.1145/3656177},
    note = {FCCM'24 Journal Track},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    issn = {1936-7406},
}
```
