# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.ir import Context, Location
from hcl_mlir.exceptions import APIError


class UniqueName:
    # each dict is symbol name -> set of unique names
    # e.g. x -> {x, x_0, x_1, x_2}
    dicts = {
        "scalar": {},
        "loop": {},
        "tensor": {},
        "stage": {},
        "schedule": {},
        "axis": {},
        "r": {},  # reduction axis
        "instance": {},
        "op": {},
        "project": {},
    }

    def __init__(self):
        pass

    @classmethod
    def reset(cls):
        for _, v in cls.dicts.items():
            v.clear()

    @classmethod
    def get(cls, name, case):
        if case not in cls.dicts:
            raise APIError(f"Unrecognized case in UniqueName.get(): {case}")

        if name is None or name == "":
            # name is not given
            # generate a name if name is not given
            case_set = cls.dicts[case]
            set_size = len(case_set)
            name = case + "_" + str(set_size)
            cls.dicts[case][name] = set()  # add a new set
            return name

        # pylint: disable=no-else-return
        if name in cls.dicts[case]:
            # name is not unique
            # generate a unique name
            case_dict = cls.dicts[case]
            set_size = len(case_dict[name])
            uname = name + "_" + str(set_size)
            cls.dicts[case][name].add(uname)
            return uname
        else:
            # name is unique
            # add the dictionary name -> {} to dicts
            cls.dicts[case][name] = set()
            return name


class GlobalContext:
    in_context = False

    def __init__(self):
        self.ctx = None
        self.loc = None
        GlobalContext.in_context = True

    def get_context(self):
        return self.ctx

    def set_context(self):
        self.ctx = Context()
        hcl_d.register_dialect(self.ctx)
        self.loc = Location.unknown(self.ctx)

    def get_location(self):
        return self.loc

    def exit_context(self):
        GlobalContext.in_context = False


global_ctx = GlobalContext()
get_context = global_ctx.get_context
set_context = global_ctx.set_context
get_location = global_ctx.get_location
exit_context = global_ctx.exit_context
