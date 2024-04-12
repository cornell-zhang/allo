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

In this folder, we provide the Allo implementations of Transformer-based models under the [`transformer`](transformer/) folder. We will release the end-to-end code and the corresponding kernel library for large-scale LLMs soon.

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
