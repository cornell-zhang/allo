# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import setuptools


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


def setup():
    with open("README.md", encoding="utf-8") as fp:
        long_description = fp.read()

    setuptools.setup(
        name="allo",
        description="Allo",
        version="0.2",
        author="Allo Community",
        long_description=long_description,
        long_description_content_type="text/markdown",
        setup_requires=[],
        install_requires=parse_requirements("requirements.txt"),
        packages=setuptools.find_packages(),
        url="https://github.com/cornell-zhang/allo",
        python_requires=">=3.12",
        classifiers=[
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering",
            "Topic :: System :: Hardware",
            "Operating System :: OS Independent",
        ],
        zip_safe=True,
    )


if __name__ == "__main__":
    setup()
