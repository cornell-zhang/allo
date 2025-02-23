# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
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
        # Retrieve LLVM_BUILD_DIR from environment variable
        llvm_build_dir = os.environ.get("LLVM_BUILD_DIR")
        if not llvm_build_dir:
            raise RuntimeError("LLVM_BUILD_DIR environment variable is not set")

        cmake_args = [
            f"-DMLIR_DIR={llvm_build_dir}/lib/cmake/mlir",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]

        build_temp = os.path.join(ext.sourcedir, "build")
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        BUILD_WITH = os.environ.get("BUILD_WITH")
        if not BUILD_WITH or BUILD_WITH == "ninja":
            subprocess.run(
                ["cmake", "-G Ninja", ext.sourcedir] + cmake_args,
                cwd=build_temp,
                check=True,
            )
            if NUM_THREADS := os.environ.get("NUM_THREADS"):
                subprocess.run(
                    ["ninja", f"-j{NUM_THREADS}"], cwd=build_temp, check=True
                )
            else:
                subprocess.run(["ninja"], cwd=build_temp, check=True)
        elif BUILD_WITH == "make":
            subprocess.run(
                ["cmake", "-G Unix Makefiles", ext.sourcedir] + cmake_args,
                cwd=build_temp,
                check=True,
            )
            if NUM_THREADS := os.environ.get("NUM_THREADS"):
                subprocess.run(["make", f"-j{NUM_THREADS}"], cwd=build_temp, check=True)
            else:
                subprocess.run(["make", "-j"], cwd=build_temp, check=True)
        else:
            raise RuntimeError(f"Unsupported BUILD_WITH={BUILD_WITH}")


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
