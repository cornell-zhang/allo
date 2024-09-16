<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

<img src="tutorials/allo-icon.png" width=128/> Accelerator Design Language
==============================================================================

[**Documentation**](https://cornell-zhang.github.io/allo) 

![GitHub](https://img.shields.io/github/license/cornell-zhang/allo)
[![CircleCI](https://circleci.com/gh/cornell-zhang/allo.svg?style=shield)](https://circleci.com/gh/cornell-zhang/allo.svg?style=shield)

Allo is a Python-embedded Accelerator Design Language (ADL) and compiler that facilitates the construction of large-scale, high-performance hardware accelerators in a modular and composable manner. Allo has several key features:
* **Progressive hardware customizations**: Allo decouples hardware customizations from algorithm specifications and treats each hardware customization as a primitive that performs a rewrite on the program. Allo not only decouples the loop-based transformations, but also extends the decoupling to memory, communication, and data types. All the transformations are built on top of [MLIR](https://mlir.llvm.org/) that is easier to target different backends.
* **Reusable parameterized kernel templates**: Allo supports declaring type variables during kernel creation and instantiating the kernel when building the hardware executable, which is an important feature for building reusable hardware kernel libraries. Allo introduces a concise grammar for creating kernel templates, eliminating the need for users to possess complicated metaprogramming expertise.
* **Composable schedules**: Allo empowers users to construct kernels incrementally from the bottom up, adding customizations one at a time while validating the correctness of each submodule. Ultimately, multiple schedules are progressively integrated into a complete design using the `.compose()` primitive. This approach, unachievable by prior top-down methods, significantly enhances productivity and debuggability.


## Getting Started

Please check out the [Allo documentation](https://cornell-zhang.github.io/allo) for installation instructions and tutorials.
If you encounter any problems, please feel free to open an [issue](https://github.com/cornell-zhang/allo/issues).


## Publications
Please refer to our [PLDI'24 paper](https://dl.acm.org/doi/10.1145/3656401) for more details. If you use Allo in your research, please use the following bibtex entry to cite us:
```bibtex
@article{chen2024allo,
    author = {Hongzheng Chen and Niansong Zhang and Shaojie Xiang and Zhichen Zeng and Mengjia Dai and Zhiru Zhang},
    title = {Allo: A Programming Model for Composable Accelerator Design},
    journal = {Proc. ACM Program. Lang.},
    year = {2024},
    month = {jun},
    url = {https://doi.org/10.1145/3656401},
    doi = {10.1145/3656401},
    articleno = {171},
    volume = {8},
    number = {PLDI},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    issue_date = {June 2024},
}
```

## Related Projects
* Accelerator Programming Languages: [Exo](https://github.com/exo-lang/exo), [Halide](https://github.com/halide/Halide), [TVM](https://github.com/apache/tvm)
* Accelerator Design Languages: [Dahlia](https://github.com/cucapra/dahlia), [HeteroCL](https://github.com/cornell-zhang/heterocl), [PyLog](https://github.com/hst10/pylog), [ScaleHLS](https://github.com/hanchenye/scalehls), [Spatial](https://github.com/stanford-ppl/spatial)
* Compiler Frameworks: [MLIR](https://mlir.llvm.org/)
