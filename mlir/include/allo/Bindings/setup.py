# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import setuptools

def setup():
    setuptools.setup(
        name="allo_mlir",
        description="Allo-MLIR: A Allo-MLIR Dialect for Heterogeneous Computing",
        version="0.1",
        author="Allo",
        setup_requires=[],
        install_requires=[],
        packages=setuptools.find_packages(),
        url="https://github.com/cornell-zhang/allo",
        python_requires=">=3.12",
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Accelerator Design",
            "Operating System :: OS Independent",
        ],
        zip_safe=True,
    )


if __name__ == "__main__":
    setup()
