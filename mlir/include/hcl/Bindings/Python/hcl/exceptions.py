# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from contextvars import ContextVar

# By default, Python ignores deprecation warnings.
# we have to enable it to see the warning.
warnings.simplefilter("always", DeprecationWarning)

PrintLog = ContextVar("PrintLog", default=False)

class bcolors:
    """ANSI color escape codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class HCLException(Exception):
    """Base class for all HeteroCL exceptions.

    Exception is the base class for warnings and errors.
    Developers can subclass this class to provide additional information

    Parameters
    ----------
    message : str
        The error message.
    """

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message


class HCLWarning(HCLException):
    """Base class for all HeteroCL warnings.

    Warning is the base class for all warnings.
    Developers can subclass this class to provide additional information

    Parameters
    ----------
    message : str
        The warning message.

    line : int, optional
        The line number of the warning.

    category_str : str, optional
        The warning category string.

    category : Warning, optional
        The warning category.
    """

    def __init__(self, message, line=None, category_str=None, category=None):
        message = bcolors.OKBLUE + message + bcolors.ENDC
        if category_str is not None:
            message = "\n{} {}".format(category_str, message)
        if line is not None:
            message += bcolors.BOLD + " (line {})".format(line) + bcolors.ENDC
        HCLException.__init__(self, message)
        self.category = category

    def warn(self):
        warnings.warn(self.message, category=self.category)

    def log(self):
        if PrintLog.get():
            print(self.message)


class HCLError(HCLException):
    """Base class for all HeteroCL errors.

    Error is the base class for all errors.
    Developers can subclass this class to provide additional information

    Parameters
    ----------
    message : str
        The error message.

    line: int, optional
        The line number of the error.

    category_str : str, optional
        The error category string.
    """

    def __init__(self, message, line=None, category_str=None):
        message = bcolors.OKBLUE + message + bcolors.ENDC
        if category_str is not None:
            message = "{} {}".format(category_str, message)
        if line is not None:
            message += bcolors.BOLD + " (line {})".format(line) + bcolors.ENDC
        HCLException.__init__(self, message)

    def error(self):
        raise self.message

""" Inherited Error subclasses """

class DTypeError(HCLError):
    """A subclass for specifying data type related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Data Type]" + bcolors.ENDC
        HCLError.__init__(self, msg, line, category_str)


class APIError(HCLError):
    """A subclass for specifying API related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[API]" + bcolors.ENDC
        HCLError.__init__(self, msg, line, category_str)


class DSLError(HCLError):
    """A subclass for specifying imperative DSL related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Imperative]" + bcolors.ENDC
        HCLError.__init__(self, msg, line, category_str)


class TensorError(HCLError):
    """A subclass for specifying tensor related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Tensor]" + bcolors.ENDC
        HCLError.__init__(self, msg, line, category_str)


class DeviceError(HCLError):
    """A subclass for specifying device related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Device]" + bcolors.ENDC
        HCLError.__init__(self, msg, line, category_str)


class AssertError(HCLError):
    """A subclass for specifying assert related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Assert]" + bcolors.ENDC
        HCLError.__init__(self, msg, line, category_str)

""" New Error subclasses """

class HCLNotImplementedError(HCLError):
    """A subclass for specifying not implemented exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Not Implemented]" + bcolors.ENDC
        HCLError.__init__(self, msg, line, category_str)

class MLIRLimitationError(HCLError):
    """A subclass for specifying MLIR limitation exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[MLIR Limitation]" + bcolors.ENDC
        HCLError.__init__(self, msg, line, category_str)

class HCLValueError(HCLError):
    """A subclass for specifying HCL value exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Value Error]" + bcolors.ENDC
        HCLError.__init__(self, msg, line, category_str)

""" New Warning subclasses """
class DTypeWarning(HCLWarning):
    """A subclass for specifying data type related warning"""

    def __init__(self, msg, line=None):
        category_str = bcolors.WARNING + "[Data Type]" + bcolors.ENDC
        HCLWarning.__init__(self, msg, line, category_str, RuntimeWarning)


class HCLDeprecationWarning(HCLWarning):
    """A subclass for specifying deprecation warning"""

    def __init__(self, msg, line=None):
        category_str = bcolors.WARNING + "[Deprecation]" + bcolors.ENDC
        HCLWarning.__init__(self, msg, line, category_str, DeprecationWarning)

class APIWarning(HCLWarning):
    """A subclass for specifying API related warning"""

    def __init__(self, msg, line=None):
        category_str = bcolors.WARNING + "[API]" + bcolors.ENDC
        HCLWarning.__init__(self, msg, line, category_str, RuntimeWarning)

class PassWarning(HCLWarning):
    """A subclass for specifying pass related warning"""

    def __init__(self, msg, line=None):
        category_str = bcolors.WARNING + "[Pass]" + bcolors.ENDC
        HCLWarning.__init__(self, msg, line, category_str, RuntimeWarning)