# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Ported from heterocl/ast/type_rules.py

import ast
import types as python_types
from .types import AlloType, Index, Float, Int, UInt, Fixed, UFixed


def get_typing_rule(node):
    return registry[node]


def sort_type_classes(types):
    """Sort the types in the order of Int, UInt, Fixed, UFixed, Float, Struct.

    Parameters
    ----------
    types : list of Type
        The list of types to be sorted.

    Returns
    -------
    list of Type
        The sorted list of types.
    """
    if isinstance(types, tuple):
        types = list(types)
    elif not isinstance(types, list):
        raise RuntimeError(
            f"sort_type_classes input should be a list or tuple, got {type(types)}"
        )
    for t in types:
        if not isinstance(t, type):
            raise RuntimeError(
                f"sort_type_classes input should be a list of types, got a list of {t} : {type(t)}"
            )
        if not issubclass(t, AlloType):
            raise RuntimeError(
                f"sort_type_classes input should be a list of Type subclass, got {t}"
            )
    type_classes = [Int, UInt, Index, Fixed, UFixed, Float]
    type_classes = [t.__name__ for t in type_classes]
    return sorted(types, key=lambda t: type_classes.index(t.__name__))


class TypingRule:
    """Type inference rule for a set of operations."""

    def __init__(self, inf_rules, commutative=False):
        """
        Parameters
        ----------
        inf_rules : a dictionary or a collection of dictionaries
            The inference rules for the operation class
            Each item should be (input types, lambda function)
        """
        # Check argument types
        if isinstance(inf_rules, dict):
            inf_rules = [inf_rules]
        elif not isinstance(inf_rules, tuple):
            inf_rules = list(inf_rules)
        elif not isinstance(inf_rules, list):
            raise TypeError(
                f"inf_rules must be a dict or a collection of dict, not {type(inf_rules)}"
            )

        self.commutative = commutative
        # Inference rules
        self.inf_rules = {}
        # a dictionary of the form:
        # { input types (tuple) : inference function (lambda func) }
        # merge the collection of inference rules into a single dictionary
        for rule_set in inf_rules:
            for itype, inf_rule in rule_set.items():
                # check itype type
                if not isinstance(itype, tuple):
                    raise TypeError(f"itype must be a tuple, not {type(itype)}")
                for t in itype:
                    if not isinstance(t, type):
                        raise TypeError(
                            f"itype must be a tuple of Class, not {type(t)}"
                        )
                # check inf_rule type
                if not isinstance(inf_rule, python_types.LambdaType):
                    raise TypeError(
                        f"inf_rule must be a lambda function, not {type(inf_rule)}"
                    )
                # sort the input types
                if commutative:
                    itype = tuple(sort_type_classes(itype))
                else:
                    itype = tuple(itype)
                # check if the input types are already in the dictionary
                if itype in self.inf_rules:
                    raise RuntimeError(
                        f"Duplicate inference rule for input types {itype}"
                    )
                # add the rule to the dictionary
                self.inf_rules[itype] = inf_rule

    def __call__(self, *args):
        """Call the inference rule with the given input types.

        It automatically finds the typing rule based on the input types.
        If no rule is found, it will raise an error.

        Parameters
        ----------
        args : list of input types

        Returns
        -------
        Type
            The inferred output type
        """
        if self.commutative:
            itype_classes = sort_type_classes([type(t) for t in args])
        else:
            itype_classes = [type(t) for t in args]
        itype_classes = tuple(itype_classes)
        if itype_classes not in self.inf_rules:
            raise RuntimeError(
                f"Typing rule is not defined with input types {itype_classes}"
            )
        rule = self.inf_rules[itype_classes]
        res_type = rule(*args)
        return res_type


def add_sub_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: Int(max(t1.bits, t2.bits) + 1),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits + 1) + 1),
        (Int, Index): lambda t1, t2: Int(max(t1.bits, t2.bits + 1) + 1),
        (Int, Fixed): lambda t1, t2: Fixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs
        ),
        (Int, UFixed): lambda t1, t2: Fixed(
            max(t1.bits, t2.bits - t2.fracs + 1) + t2.fracs + 1, t2.fracs
        ),
        (Int, Float): lambda t1, t2: t2,
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: Int(max(t1.bits + 1, t2.bits) + 1),
        (UInt, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits) + 1),
        (UInt, Index): lambda t1, t2: UInt(max(t1.bits, t2.bits) + 1),
        (UInt, Fixed): lambda t1, t2: Fixed(
            max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs
        ),
        (UInt, UFixed): lambda t1, t2: UFixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs
        ),
        (UInt, Float): lambda t1, t2: t2,
    }
    index_rules = {
        (Index, Int): lambda t1, t2: Int(max(t1.bits + 1, t2.bits) + 1),
        (Index, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits) + 1),
        (Index, Index): lambda t1, t2: Index(),
        (Index, Fixed): lambda t1, t2: Fixed(
            max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs
        ),
        (Index, UFixed): lambda t1, t2: UFixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs
        ),
        (Index, Float): lambda t1, t2: t2,
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits) + t1.fracs + 1, t1.fracs
        ),
        (Fixed, UInt): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs + 1, t1.fracs
        ),
        (Fixed, Index): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs + 1, t1.fracs
        ),
        (Fixed, Fixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs) + 1,
            max(t1.fracs, t2.fracs),
        ),
        (Fixed, UFixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs + 1)
            + max(t1.fracs, t2.fracs)
            + 1,
            max(t1.fracs, t2.fracs),
        ),
        (Fixed, Float): lambda t1, t2: t2,
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs + 1, t2.bits) + t1.fracs + 1, t1.fracs
        ),
        (UFixed, UInt): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits) + t1.fracs + 1, t1.fracs
        ),
        (UFixed, Index): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits) + t1.fracs + 1, t1.fracs
        ),
        (UFixed, Fixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs + 1, t2.bits - t2.fracs)
            + max(t1.fracs, t2.fracs)
            + 1,
            max(t1.fracs, t2.fracs),
        ),
        (UFixed, UFixed): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs) + 1,
            max(t1.fracs, t2.fracs),
        ),
        (UFixed, Float): lambda t1, t2: t2,
    }
    float_rules = {
        (Float, Int): lambda t1, t2: t1,
        (Float, UInt): lambda t1, t2: t1,
        (Float, Index): lambda t1, t2: t1,
        (Float, Fixed): lambda t1, t2: t1,
        (Float, UFixed): lambda t1, t2: t1,
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def mul_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: Int(t1.bits + t2.bits),
        (Int, UInt): lambda t1, t2: Int(t1.bits + t2.bits),
        (Int, Index): lambda t1, t2: Int(t1.bits + t2.bits),
        (Int, Fixed): lambda t1, t2: Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        (Int, UFixed): lambda t1, t2: Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        (Int, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    uint_rules = {
        # (Uint, Int) covered by (Int, Uint)
        (UInt, UInt): lambda t1, t2: UInt(t1.bits + t2.bits),
        (UInt, Index): lambda t1, t2: UInt(t1.bits + t2.bits),
        (UInt, Fixed): lambda t1, t2: Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        (UInt, UFixed): lambda t1, t2: UFixed(
            t1.bits + t2.bits, max(t1.fracs, t2.fracs)
        ),
        (UInt, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    index_rules = {
        # (Index, Int) covered by (Int, Index)
        # (Index, UInt) covered by (UInt, Index)
        (Index, Index): lambda t1, t2: Index(),
        (Index, Fixed): lambda t1, t2: Fixed(
            t1.bits + t2.bits, max(t1.fracs, t2.fracs)
        ),
        (Index, UFixed): lambda t1, t2: UFixed(
            t1.bits + t2.bits, max(t1.fracs, t2.fracs)
        ),
        (Index, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    fixed_rules = {
        # (Fixed, Int) covered by (Int, Fixed)
        # (Fixed, UInt) covered by (UInt, Fixed)
        # (Fixed, Index) covered by (Index, Fixed)
        (Fixed, Fixed): lambda t1, t2: Fixed(t1.bits + t2.bits, t1.fracs + t2.fracs),
        (Fixed, UFixed): lambda t1, t2: Fixed(t1.bits + t2.bits, t1.fracs + t2.fracs),
        (Fixed, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    ufixed_rules = {
        # (UFixed, Int) covered by (Int, UFixed)
        # (UFixed, UInt) covered by (UInt, UFixed)
        # (UFixed, Index) covered by (Index, UFixed)
        # (UFixed, Fixed) covered by (Fixed, UFixed)
        (UFixed, UFixed): lambda t1, t2: UFixed(t1.bits + t2.bits, t1.fracs + t2.fracs),
        (UFixed, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    float_rules = {
        # (Float, (Int, UInt, Index, Fixed, UFixed)) covered
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits))
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
        commutative=True,
    )


def div_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: t1,
        (Int, UInt): lambda t1, t2: t1,
        (Int, Index): lambda t1, t2: t1,
        (Int, Fixed): lambda t1, t2: Fixed(t1.bits + t2.bits, t1.bits - t2.fracs),
        (Int, UFixed): lambda t1, t2: Fixed(t1.bits + t2.bits + 1, t1.bits - t2.fracs),
        (Int, Float): lambda t1, t2: t2,
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: Int(t1.bits),
        (UInt, UInt): lambda t1, t2: t1,
        (UInt, Index): lambda t1, t2: t1,
        (UInt, Fixed): lambda t1, t2: Fixed(t1.bits + t2.bits, t1.bits - t2.fracs),
        (UInt, UFixed): lambda t1, t2: UFixed(t1.bits + t2.bits, t1.bits - t2.fracs),
        (UInt, Float): lambda t1, t2: t2,
    }
    index_rules = {
        (Index, Int): lambda t1, t2: Int(t1.bits),
        (Index, UInt): lambda t1, t2: t1,
        (Index, Index): lambda t1, t2: Index(),
        (Index, Fixed): lambda t1, t2: Fixed(t1.bits + t2.bits, t1.bits - t2.fracs),
        (Index, UFixed): lambda t1, t2: UFixed(t1.bits + t2.bits, t1.bits - t2.fracs),
        (Index, Float): lambda t1, t2: t2,
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: Fixed(t1.bits + t2.bits, t2.bits + t1.fracs),
        (Fixed, UInt): lambda t1, t2: Fixed(t1.bits + t2.bits + 1, t2.bits + t1.fracs),
        (Fixed, Index): lambda t1, t2: Fixed(t1.bits + t2.bits + 1, t2.bits + t1.fracs),
        (Fixed, Fixed): lambda t1, t2: Fixed(
            t1.bits + t2.bits, t2.bits - t2.fracs + t1.fracs
        ),
        (Fixed, UFixed): lambda t1, t2: Fixed(
            t1.bits + t2.bits + 1, t2.bits - t2.fracs + t1.fracs
        ),
        (Fixed, Float): lambda t1, t2: t2,
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: Fixed(t1.bits + t2.bits + 1, t2.bits + t1.fracs),
        (UFixed, UInt): lambda t1, t2: UFixed(t1.bits + t2.bits, t2.bits + t1.fracs),
        (UFixed, Index): lambda t1, t2: UFixed(t1.bits + t2.bits, t2.bits + t1.fracs),
        (UFixed, Fixed): lambda t1, t2: Fixed(
            t1.bits + t2.bits, t2.bits - t2.fracs + t1.fracs
        ),
        (UFixed, UFixed): lambda t1, t2: UFixed(
            t1.bits + t2.bits, t2.bits - t2.fracs + t1.fracs
        ),
        (UFixed, Float): lambda t1, t2: t2,
    }
    float_rules = {
        (Float, Int): lambda t1, t2: t1,
        (Float, UInt): lambda t1, t2: t1,
        (Float, Index): lambda t1, t2: t1,
        (Float, Fixed): lambda t1, t2: t1,
        (Float, UFixed): lambda t1, t2: t1,
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def mod_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits + 1)),
        (Int, Index): lambda t1, t2: Int(max(t1.bits, t2.bits + 1)),
        (Int, Fixed): lambda t1, t2: Fixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (Int, UFixed): lambda t1, t2: Fixed(
            max(t1.bits, t2.bits - t2.fracs + 1) + t2.fracs, t2.fracs
        ),
        (Int, Float): lambda t1, t2: t2,
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: Int(max(t1.bits + 1, t2.bits)),
        (UInt, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (UInt, Index): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (UInt, Fixed): lambda t1, t2: Fixed(
            max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (UInt, UFixed): lambda t1, t2: UFixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (UInt, Float): lambda t1, t2: t2,
    }
    index_rules = {
        (Index, Int): lambda t1, t2: Int(max(t1.bits + 1, t2.bits)),
        (Index, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (Index, Index): lambda t1, t2: Index(),
        (Index, Fixed): lambda t1, t2: Fixed(
            max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (Index, UFixed): lambda t1, t2: UFixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (Index, Float): lambda t1, t2: t2,
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs
        ),
        (Fixed, UInt): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs
        ),
        (Fixed, Index): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs
        ),
        (Fixed, Fixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (Fixed, UFixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs + 1) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (Fixed, Float): lambda t1, t2: t2,
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs + 1, t2.bits) + t1.fracs, t1.fracs
        ),
        (UFixed, UInt): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs
        ),
        (UFixed, Index): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs
        ),
        (UFixed, Fixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs + 1, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (UFixed, UFixed): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (UFixed, Float): lambda t1, t2: t2,
    }
    float_rules = {
        (Float, Int): lambda t1, t2: t1,
        (Float, UInt): lambda t1, t2: t1,
        (Float, Index): lambda t1, t2: t1,
        (Float, Fixed): lambda t1, t2: t1,
        (Float, UFixed): lambda t1, t2: t1,
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def cmp_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: (Int(max(t1.bits, t2.bits)), UInt(1)),
        (Int, UInt): lambda t1, t2: (Int(max(t1.bits, t2.bits + 1)), UInt(1)),
        (Int, Index): lambda t1, t2: (Int(max(t1.bits, t2.bits + 1)), UInt(1)),
        (Int, Fixed): lambda t1, t2: (
            Fixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            UInt(1),
        ),
        (Int, UFixed): lambda t1, t2: (
            Fixed(max(t1.bits, t2.bits - t2.fracs + 1) + t2.fracs, t2.fracs),
            UInt(1),
        ),
        (Int, Float): lambda t1, t2: (t2, UInt(1)),
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: (Int(max(t1.bits + 1, t2.bits)), UInt(1)),
        (UInt, UInt): lambda t1, t2: (UInt(max(t1.bits, t2.bits)), UInt(1)),
        (UInt, Index): lambda t1, t2: (UInt(max(t1.bits, t2.bits)), UInt(1)),
        (UInt, Fixed): lambda t1, t2: (
            Fixed(max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            UInt(1),
        ),
        (UInt, UFixed): lambda t1, t2: (
            UFixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            UInt(1),
        ),
        (UInt, Float): lambda t1, t2: (t2, UInt(1)),
    }
    index_rules = {
        (Index, Int): lambda t1, t2: (Int(max(t1.bits + 1, t2.bits)), UInt(1)),
        (Index, UInt): lambda t1, t2: (UInt(max(t1.bits, t2.bits)), UInt(1)),
        (Index, Index): lambda t1, t2: (Index(), UInt(1)),
        (Index, Fixed): lambda t1, t2: (
            Fixed(max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            UInt(1),
        ),
        (Index, UFixed): lambda t1, t2: (
            UFixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            UInt(1),
        ),
        (Index, Float): lambda t1, t2: (t2, UInt(1)),
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
            UInt(1),
        ),
        (Fixed, UInt): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs),
            UInt(1),
        ),
        (Fixed, Index): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs),
            UInt(1),
        ),
        (Fixed, Fixed): lambda t1, t2: (
            Fixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            UInt(1),
        ),
        (Fixed, UFixed): lambda t1, t2: (
            Fixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs + 1)
                + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            UInt(1),
        ),
        (Fixed, Float): lambda t1, t2: (t2, UInt(1)),
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs + 1, t2.bits) + t1.fracs, t1.fracs),
            UInt(1),
        ),
        (UFixed, UInt): lambda t1, t2: (
            UFixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
            UInt(1),
        ),
        (UFixed, Index): lambda t1, t2: (
            UFixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
            UInt(1),
        ),
        (UFixed, Fixed): lambda t1, t2: (
            Fixed(
                max(t1.bits - t1.fracs + 1, t2.bits - t2.fracs)
                + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            UInt(1),
        ),
        (UFixed, UFixed): lambda t1, t2: (
            UFixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            UInt(1),
        ),
        (UFixed, Float): lambda t1, t2: (t2, UInt(1)),
    }
    float_rules = {
        (Float, Int): lambda t1, t2: (t1, UInt(1)),
        (Float, UInt): lambda t1, t2: (t1, UInt(1)),
        (Float, Index): lambda t1, t2: (t1, UInt(1)),
        (Float, Fixed): lambda t1, t2: (t1, UInt(1)),
        (Float, UFixed): lambda t1, t2: (t1, UInt(1)),
        (Float, Float): lambda t1, t2: (Float(max(t1.bits, t2.bits)), UInt(1)),
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def select_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits + 1)),
        (Int, Index): lambda t1, t2: Int(max(t1.bits, t2.bits + 1)),
        (Int, Fixed): lambda t1, t2: Fixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (Int, UFixed): lambda t1, t2: Fixed(
            max(t1.bits, t2.bits - t2.fracs + 1) + t2.fracs, t2.fracs
        ),
        (Int, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    uint_rules = {
        (UInt, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (UInt, Index): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (UInt, Fixed): lambda t1, t2: Fixed(
            max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (UInt, UFixed): lambda t1, t2: UFixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (UInt, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    index_rules = {
        (Index, Index): lambda t1, t2: Index(),
        (Index, Fixed): lambda t1, t2: Fixed(
            max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (Index, UFixed): lambda t1, t2: UFixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (Index, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    fixed_rules = {
        (Fixed, Fixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (Fixed, UFixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs + 1) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (Fixed, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    ufixed_rules = {
        (UFixed, UFixed): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (UFixed, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    float_rules = {
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
        commutative=True,
    )


def shift_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: t1,
        (Int, UInt): lambda t1, t2: t1,
        (Int, Index): lambda t1, t2: t1,
    }
    uint_rules = {
        (UInt, UInt): lambda t1, t2: t1,
        (UInt, Index): lambda t1, t2: t1,
    }
    index_rules = {
        (Index, Index): lambda t1, t2: Index(),
    }
    return TypingRule([int_rules, uint_rules, index_rules], commutative=True)


def and_or_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, Index): lambda t1, t2: Int(max(t1.bits, t2.bits)),
    }
    uint_rules = {
        (UInt, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (UInt, Index): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
    }
    index_rules = {
        (Index, Index): lambda t1, t2: Index(),
    }
    return TypingRule([int_rules, uint_rules, index_rules], commutative=True)


def logic_op_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: UInt(1),
        (Int, UInt): lambda t1, t2: UInt(1),
        (UInt, UInt): lambda t1, t2: UInt(1),
    }
    return TypingRule([int_rules])


def pow_rule():
    def select_float(t1, _):
        if t1.bits <= 32:
            return Float(32)
        return Float(64)

    int_rule = {
        (Int, Int): select_float,
        (Int, UInt): select_float,
        (Int, Index): select_float,
        (Int, Fixed): select_float,
        (Int, UFixed): select_float,
        (Int, Float): select_float,
    }
    uint_rule = {
        (UInt, UInt): select_float,
        (UInt, Index): select_float,
        (UInt, Fixed): select_float,
        (UInt, UFixed): select_float,
        (UInt, Float): select_float,
    }
    index_rule = {
        (Index, Index): select_float,
        (Index, Fixed): select_float,
        (Index, UFixed): select_float,
        (Index, Float): select_float,
    }
    fixed_rule = {
        (Fixed, Fixed): select_float,
        (Fixed, UFixed): select_float,
        (Fixed, Float): select_float,
    }
    ufixed_rule = {(UFixed, UFixed): select_float, (UFixed, Float): select_float}
    float_rule = {
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    # Commutative=True here doesn't mean that power operation is commutative.
    # It means that the type rule is commutative, to reduce the number of rules.
    # e.g. hcl.power(a, b) and hcl.power(b, a) will have the same type rule.
    # because MLIR math op in LLVM 15 only has float pow op.
    return TypingRule(
        [int_rule, uint_rule, index_rule, fixed_rule, ufixed_rule, float_rule],
        commutative=True,
    )


def intrin_rule():
    # covers:
    # expr, log, log2, log10, sqrt,
    # sin, cos, tanh
    unaryrules = {
        (Float,): lambda t: t,
        (Int,): lambda t: Float(32) if t.bits <= 32 else Float(64),
        (UInt,): lambda t: Float(32) if t.bits <= 32 else Float(64),
        (Index,): lambda t: Float(32) if t.bits <= 32 else Float(64),
        (Fixed,): lambda t: Float(32) if t.bits <= 32 else Float(64),
        (UFixed,): lambda t: Float(32) if t.bits <= 32 else Float(64),
    }
    return TypingRule([unaryrules])


registry = {
    ast.Add: add_sub_rule(),
    ast.Sub: add_sub_rule(),
    ast.Mult: mul_rule(),
    ast.Div: div_rule(),
    ast.FloorDiv: div_rule(),
    ast.Mod: mod_rule(),
    ast.Pow: pow_rule(),
    ast.LShift: shift_rule(),
    ast.RShift: shift_rule(),
    ast.BitOr: and_or_rule(),
    ast.BitXor: and_or_rule(),
    ast.BitAnd: and_or_rule(),
    ast.And: logic_op_rule(),
    ast.Or: logic_op_rule(),
    ast.Eq: cmp_rule(),
    ast.NotEq: cmp_rule(),
    ast.Lt: cmp_rule(),
    ast.LtE: cmp_rule(),
    ast.Gt: cmp_rule(),
    ast.GtE: cmp_rule(),
    ast.USub: intrin_rule(),
    ast.UAdd: intrin_rule(),
    ast.Invert: intrin_rule(),
    ast.IfExp: select_rule(),
}
