# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar


# XlsInt: Arbitrary Precision Integer
class XlsIntMeta(type):
    _cache: dict[tuple[int, bool], type] = {}

    def __getitem__(cls, params):
        if isinstance(params, tuple):
            width, signed = params
        else:
            width, signed = params, True

        key = (width, signed)
        if key not in cls._cache:
            new_cls = type(
                f"XlsInt_{width}_{'s' if signed else 'u'}",
                (XlsInt,),
                {"WIDTH": width, "SIGNED": signed},
            )
            cls._cache[key] = new_cls
        return cls._cache[key]


class XlsInt(metaclass=XlsIntMeta):
    WIDTH: int = 32
    SIGNED: bool = True

    def __init__(self, value: int = 0):
        self._value = self._wrap(int(value))

    def _wrap(self, val: int) -> int:
        mask = (1 << self.WIDTH) - 1
        val = val & mask
        if self.SIGNED and (val & (1 << (self.WIDTH - 1))):
            val = val - (1 << self.WIDTH)
        return val

    @property
    def value(self) -> int:
        return self._value

    def __int__(self) -> int:
        return self._value

    def __repr__(self) -> str:
        sign = "s" if self.SIGNED else "u"
        return f"XlsInt<{self.WIDTH},{sign}>({self._value})"

    def __eq__(self, other) -> bool:
        if isinstance(other, XlsInt):
            return self._value == other._value
        return self._value == other

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other) -> bool:
        other_val = other._value if isinstance(other, XlsInt) else other
        return self._value < other_val

    def __le__(self, other) -> bool:
        other_val = other._value if isinstance(other, XlsInt) else other
        return self._value <= other_val

    def __gt__(self, other) -> bool:
        other_val = other._value if isinstance(other, XlsInt) else other
        return self._value > other_val

    def __ge__(self, other) -> bool:
        other_val = other._value if isinstance(other, XlsInt) else other
        return self._value >= other_val

    def __add__(self, other) -> XlsInt:
        other_val = other._value if isinstance(other, XlsInt) else other
        return type(self)(self._value + other_val)

    def __radd__(self, other) -> XlsInt:
        return self.__add__(other)

    def __sub__(self, other) -> XlsInt:
        other_val = other._value if isinstance(other, XlsInt) else other
        return type(self)(self._value - other_val)

    def __rsub__(self, other) -> XlsInt:
        other_val = other._value if isinstance(other, XlsInt) else other
        return type(self)(other_val - self._value)

    def __mul__(self, other) -> XlsInt:
        other_val = other._value if isinstance(other, XlsInt) else other
        return type(self)(self._value * other_val)

    def __rmul__(self, other) -> XlsInt:
        return self.__mul__(other)

    def __floordiv__(self, other) -> XlsInt:
        other_val = other._value if isinstance(other, XlsInt) else other
        if other_val == 0:
            raise ZeroDivisionError("XlsInt division by zero")
        return type(self)(self._value // other_val)

    def __mod__(self, other) -> XlsInt:
        other_val = other._value if isinstance(other, XlsInt) else other
        return type(self)(self._value % other_val)

    def __and__(self, other) -> XlsInt:
        other_val = other._value if isinstance(other, XlsInt) else other
        return type(self)(self._value & other_val)

    def __or__(self, other) -> XlsInt:
        other_val = other._value if isinstance(other, XlsInt) else other
        return type(self)(self._value | other_val)

    def __xor__(self, other) -> XlsInt:
        other_val = other._value if isinstance(other, XlsInt) else other
        return type(self)(self._value ^ other_val)

    def __lshift__(self, other) -> XlsInt:
        shift = other._value if isinstance(other, XlsInt) else other
        return type(self)(self._value << shift)

    def __rshift__(self, other) -> XlsInt:
        shift = other._value if isinstance(other, XlsInt) else other
        return type(self)(self._value >> shift)

    def __invert__(self) -> XlsInt:
        return type(self)(~self._value)

    def __neg__(self) -> XlsInt:
        return type(self)(-self._value)

    def __getitem__(self, key) -> XlsInt:
        if isinstance(key, slice):
            hi = key.start if key.start is not None else self.WIDTH
            lo = key.stop if key.stop is not None else 0
            width = hi - lo
            mask = (1 << width) - 1
            val = (self._value >> lo) & mask
            return XlsInt[width, False](val)
        return (self._value >> key) & 1

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            hi = key.start if key.start is not None else self.WIDTH
            lo = key.stop if key.stop is not None else 0
            width = hi - lo
            mask = (1 << width) - 1
            val = value._value if isinstance(value, XlsInt) else value
            val = val & mask
            clear_mask = ~(mask << lo)
            self._value = self._wrap((self._value & clear_mask) | (val << lo))


def ac_int(width: int, signed: bool = True):
    return XlsInt[width, signed]


# XlsChannel: Streaming I/O Channel
T = TypeVar("T")


class XlsChannelMeta(type):
    _cache: dict = {}

    def __getitem__(cls, elem_type):
        if elem_type not in cls._cache:
            type_name = (
                elem_type.__name__ if hasattr(elem_type, "__name__") else str(elem_type)
            )
            new_cls = type(
                f"XlsChannel_{type_name}",
                (XlsChannel,),
                {"ELEM_TYPE": elem_type},
            )
            cls._cache[elem_type] = new_cls
        return cls._cache[elem_type]


class XlsChannel(Generic[T], metaclass=XlsChannelMeta):
    ELEM_TYPE = int

    def __init__(self, name: str = "channel"):
        self.name = name
        self._queue: deque = deque()

    def write(self, value: T) -> None:
        self._queue.append(value)

    def read(self) -> T:
        if not self._queue:
            raise RuntimeError(f"Channel '{self.name}' is empty - read blocked")
        return self._queue.popleft()

    def empty(self) -> bool:
        return len(self._queue) == 0

    def size(self) -> int:
        return len(self._queue)

    def load(self, data: list) -> None:
        for item in data:
            self.write(item)

    def drain(self) -> list:
        result = []
        while not self.empty():
            result.append(self.read())
        return result

    def __repr__(self) -> str:
        return f"XlsChannel[{self.ELEM_TYPE}]('{self.name}', size={self.size()})"


# XlsMemory: On-chip Memory (SRAM/BRAM)
class XlsMemoryMeta(type):
    _cache: dict = {}

    def __getitem__(cls, params):
        if isinstance(params, tuple):
            elem_type, size = params
        else:
            elem_type, size = params, 256

        key = (elem_type, size)
        if key not in cls._cache:
            new_cls = type(
                f"XlsMemory_{size}",
                (XlsMemory,),
                {"ELEM_TYPE": elem_type, "SIZE": size},
            )
            cls._cache[key] = new_cls
        return cls._cache[key]


class XlsMemory(metaclass=XlsMemoryMeta):
    ELEM_TYPE = int
    SIZE: int = 256

    def __init__(self):
        self._data = [0] * self.SIZE

    def __getitem__(self, idx: int):
        if not 0 <= idx < self.SIZE:
            raise IndexError(f"XlsMemory index {idx} out of range [0, {self.SIZE})")
        return self._data[idx]

    def __setitem__(self, idx: int, value):
        if not 0 <= idx < self.SIZE:
            raise IndexError(f"XlsMemory index {idx} out of range [0, {self.SIZE})")
        self._data[idx] = value._value if isinstance(value, XlsInt) else value

    def load(self, data: list) -> None:
        for i, val in enumerate(data[: self.SIZE]):
            self._data[i] = val

    def dump(self) -> list:
        return list(self._data)

    def __repr__(self) -> str:
        return f"XlsMemory[{self.ELEM_TYPE}, {self.SIZE}]"


# Test Framework
class XlsTestRunner:
    def __init__(self, name: str = "XLS Functional Tests"):
        self.name = name
        self._tests: list[tuple[str, callable]] = []
        self._results: list[tuple[str, bool, str]] = []

    def test(self, name: str):
        def decorator(func):
            self._tests.append((name, func))
            return func

        return decorator

    def add_test(self, name: str, func: callable):
        self._tests.append((name, func))

    def run_all(self) -> bool:
        print(f"\n{'=' * 60}")
        print(f" {self.name}")
        print(f"{'=' * 60}")
        print("  ", end="")

        all_passed = True
        failures = []
        for name, func in self._tests:
            try:
                result = func()
                passed = result if isinstance(result, bool) else True
                msg = ""
            except AssertionError as e:
                passed = False
                msg = str(e)
            except Exception as e:  # pylint: disable=broad-exception-caught
                passed = False
                msg = f"Exception: {e}"

            self._results.append((name, passed, msg))
            if passed:
                print(".", end="", flush=True)
            else:
                print("F", end="", flush=True)
                all_passed = False
                failures.append((name, msg))

        print()
        for name, msg in failures:
            print(f"  FAIL: {name}")
            if msg:
                print(f"        {msg}")

        self._print_summary()
        return all_passed

    def _print_summary(self):
        passed = sum(1 for _, p, _ in self._results if p)
        failed = len(self._results) - passed

        print(f"\n{'─' * 60}")
        if failed == 0:
            print(f" ✓ All {passed} tests passed!")
        else:
            print(f" ✗ {failed}/{len(self._results)} tests failed")
        print(f"{'─' * 60}\n")


def run_directed_test(
    _name: str,
    func: callable,
    inputs: list,
    expected: list,
) -> tuple[bool, str]:
    try:
        actual = func(*inputs)
        if not isinstance(actual, (list, tuple)):
            actual = [actual]
        if not isinstance(expected, (list, tuple)):
            expected = [expected]

        actual_vals = [int(a) if isinstance(a, XlsInt) else a for a in actual]
        expected_vals = [int(e) if isinstance(e, XlsInt) else e for e in expected]

        passed = actual_vals == expected_vals
        msg = "" if passed else f"{inputs} -> {actual_vals} (expected {expected_vals})"
        return passed, msg
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, f"Exception - {e}"


def run_test_vectors(
    name: str,
    func: callable,
    test_vectors: list[tuple],
) -> bool:
    print(f"\n{name}: ", end="")

    failures = []
    for i, (inputs, expected) in enumerate(test_vectors):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        passed, msg = run_directed_test(f"vector[{i}]", func, inputs, expected)
        if passed:
            print(".", end="", flush=True)
        else:
            print("F", end="", flush=True)
            failures.append((i, msg))

    print()
    for idx, msg in failures:
        print(f"  FAIL vector[{idx}]: {msg}")

    return len(failures) == 0
