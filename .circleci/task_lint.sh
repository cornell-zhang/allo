#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

echo "Check license header..."
python3 scripts/lint/check_license_header.py HEAD~1
python3 scripts/lint/check_license_header.py origin/main

echo "Check Python formats using black..."
bash ./scripts/lint/git-black.sh HEAD~1
bash ./scripts/lint/git-black.sh origin/main

echo "Check C/C++ formats using clang-format..."
bash ./scripts/lint/git-clang-format.sh HEAD~1
bash ./scripts/lint/git-clang-format.sh origin/main

echo "Running pylint on allo"
python3 -m pylint allo --rcfile=./scripts/lint/pylintrc
