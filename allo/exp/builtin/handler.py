# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import ast
import types as python_types

# Registry for builtin handlers
#   function name (str) -> Handler class
BUILTIN_HANDLERS = {}


def register_builtin_handler(name):
    def decorator(cls):
        BUILTIN_HANDLERS[name] = cls
        return cls

    return decorator


def register_custom_handler(name=None):
    def decorator(cls):
        handler_name = name if name else cls.__name__
        BUILTIN_HANDLERS[handler_name] = cls

        def custom_func(*args, **kwargs):
            pass

        custom_func.__dict__["__allo_handler__"] = handler_name
        return custom_func

    return decorator


class BuiltinHandler(abc.ABC):
    def __init__(self, builder):
        """
        Args:
            builder: The IRBuilder instance invoking this handler.
        """
        self.builder = builder

    @abc.abstractmethod
    def build(self, node: ast.Call, *args):
        """
        Build the IR for the builtin function call.

        Args:
            node: The ast.Call node.
            args: The arguments passed to the function call
        """
        pass

    @staticmethod
    def infer(*args):
        """
        Infer the result type and operand types for the builtin function.

        Args:
            *args: The arguments passed to the function call.

        Returns:
            A tuple of (result_types..., operand_types..., annotations...).
            - result_types: The inferred result types.
            - operand_types: The expected operand types.
            - annotations: Optional annotations for the operation (e.g., "unsigned").
        """
        raise NotImplementedError("infer method is not implemented for this handler.")

    def get_affine_expr(self, node: ast.Call, ivs: list, symbols: list):
        """
        Build the affine expression for the builtin function call.

        Args:
            node: The ast.Call node.
            ivs: The list of induction variables.
        """
        return None


class TypingRule:
    """Type inference rule for a set of operations."""

    def __init__(self, inf_rules):
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
        itype_classes = [type(t) for t in args]
        itype_classes = tuple(itype_classes)
        if itype_classes not in self.inf_rules:
            raise RuntimeError(
                f"Typing rule is not defined with input types {itype_classes}"
            )
        rule = self.inf_rules[itype_classes]
        res_type = rule(*args)
        return res_type
