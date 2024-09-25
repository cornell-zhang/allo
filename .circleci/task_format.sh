#!/bin/bash
# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

echo "Check license header..."
python3 scripts/check_license_header.py HEAD~1
python3 scripts/check_license_header.py origin/main

echo "Check C/C++ formats using clang-format..."
bash ./scripts/git-clang-format.sh HEAD~1
bash ./scripts/git-clang-format.sh origin/main
