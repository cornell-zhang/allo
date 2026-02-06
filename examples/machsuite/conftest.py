# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# Only collect tests from test_machsuite.py to avoid module name collisions
# (e.g., md/grid/md.py vs md/knn/md.py) when pytest discovers subdirectories.
_this_dir = os.path.dirname(__file__)
collect_ignore = [
    os.path.join(_this_dir, d)
    for d in os.listdir(_this_dir)
    if os.path.isdir(os.path.join(_this_dir, d)) and d != "__pycache__"
]
