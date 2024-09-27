# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools import find_packages


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        # Ensure CMake is installed
        try:
            subprocess.check_call(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        # Call the build process for each extension
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        cmake_args = [
            "-DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir",
            "-DPython3_EXECUTABLE=`which python3`",
        ]

        build_temp = os.path.join(ext.sourcedir, "build")
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(
            ["cmake", "-G Ninja", ext.sourcedir] + cmake_args,
            cwd=build_temp,
            check=True,
        )
        subprocess.check_call(["ninja"], cwd=build_temp, check=True)


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


if __name__ == "__main__":
    with open("README.md", encoding="utf-8") as fp:
        long_description = fp.read()

    setup(
        name="allo",
        description="Allo",
        version="0.3",
        author="Allo Community",
        long_description=long_description,
        long_description_content_type="text/markdown",
        setup_requires=[],
        install_requires=parse_requirements("requirements.txt"),
        packages=find_packages(),
        ext_modules=[CMakeExtension("mlir", sourcedir="mlir")],
        cmdclass={"build_ext": CMakeBuild},
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
