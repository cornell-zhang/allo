# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import setuptools


def setup():
    with open("README.md", encoding="utf-8") as fp:
        long_description = fp.read()

    setuptools.setup(
        name="allo",
        description="Allo",
        version="0.1",
        author="Allo Community",
        long_description=long_description,
        long_description_content_type="text/markdown",
        setup_requires=[],
        install_requires=[
            "packaging",
            "psutil",
        ],
        packages=setuptools.find_packages(),
        url="https://github.com/cornell-zhang/allo",
        python_requires=">=3.8",
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering",
            "Topic :: System :: Hardware",
            "Operating System :: OS Independent",
        ],
        zip_safe=True,
    )


if __name__ == "__main__":
    setup()
