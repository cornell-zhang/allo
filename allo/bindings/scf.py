# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=c-extension-no-member

import sys
from . import _liballo

sys.modules[__name__] = _liballo._load_submodule("scf")
