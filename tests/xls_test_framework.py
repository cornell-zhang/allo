"""
XLS[cc] Test Framework

This module provides a test framework for verifying the correctness of 
generated XLS[cc] C++ code from Allo.

Usage:
    from xls_test_framework import XLSTestRunner, XLSTest
    
    # Create test runner
    runner = XLSTestRunner()
    
    # Test combinational logic
    runner.test_combinational(
        allo_func=add,
        schedule=s,
        test_cases=[
            ((2, 3), 5),
            ((10, -5), 5),
        ]
    )
    
    # Test sequential logic (arrays)
    runner.test_sequential(
        allo_func=vvadd,
        schedule=s,
        test_cases=[
            (([1, 2, 3], [4, 5, 6]), [5, 7, 9]),
        ]
    )
"""

import os
import subprocess
import tempfile
import shutil
import numpy as np
from typing import Callable, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import allo

# Path to the mock XLS headers
XLSCC_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_HEADERS_DIR = os.path.join(XLSCC_TESTS_DIR, "xlscc_tests")


@dataclass
class TestResult:
    """Result of a single test case."""
    passed: bool
    test_name: str
    inputs: Any
    expected: Any
    actual: Any
    error_message: Optional[str] = None


class XLSTestRunner:
    """Test runner for XLS[cc] generated code."""
    
    def __init__(self, output_dir: Optional[str] = None, verbose: bool = False):
        """
        Initialize the test runner.
        
        Args:
            output_dir: Directory for test outputs. If None, uses xlscc_tests/
            verbose: If True, print detailed output for each test
        """
        self.output_dir = output_dir or MOCK_HEADERS_DIR
        self.verbose = verbose
        self.results: List[TestResult] = []
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _compile_cpp(self, cpp_code: str, test_name: str) -> Tuple[bool, str, str]:
        """
        Compile C++ code with g++ and return (success, executable_path, error_msg).
        """
        # Create temp directory for this test
        test_dir = os.path.join(self.output_dir, test_name)
        os.makedirs(test_dir, exist_ok=True)
        
        # Write the C++ code
        cpp_path = os.path.join(test_dir, "test.cpp")
        exe_path = os.path.join(test_dir, "test")
        
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
        
        # Compile with g++
        compile_cmd = [
            "g++", "-std=c++17", "-O2",
            f"-I{MOCK_HEADERS_DIR}",
            cpp_path, "-o", exe_path
        ]
        
        try:
            result = subprocess.run(
                compile_cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            if result.returncode != 0:
                return False, "", f"Compilation error:\n{result.stderr}"
            return True, exe_path, ""
        except subprocess.TimeoutExpired:
            return False, "", "Compilation timeout"
        except Exception as e:
            return False, "", f"Compilation exception: {e}"
    
    def _run_executable(self, exe_path: str, input_data: str = "") -> Tuple[bool, str, str]:
        """
        Run an executable and return (success, stdout, stderr).
        """
        try:
            result = subprocess.run(
                [exe_path],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Execution timeout"
        except Exception as e:
            return False, "", f"Execution exception: {e}"
    
    def _generate_combinational_harness(self, 
                                         cpp_code: str, 
                                         func_name: str,
                                         test_cases: List[Tuple[Tuple, Any]]) -> str:
        """
        Generate a test harness for combinational logic.
        
        The harness calls the function with each input and prints the result.
        """
        # Extract return type and parameter types from function signature
        # For now, assume int types
        
        harness = f'''
// Test harness for combinational logic
#include <iostream>
#include <cstdint>

// Include the mock headers
#include "xls_builtin.h"
#include "xls_int.h"

// Wrap the ac_int alias
template <int Width, bool Signed = true>
using ac_int = XlsInt<Width, Signed>;

// Generated code (remove the #include directives that reference XLS paths)
'''
        
        # Remove XLS-specific includes from the generated code
        lines = cpp_code.split('\n')
        filtered_lines = []
        for line in lines:
            if '#include' in line and ('xls' in line.lower() or '/xls' in line):
                continue  # Skip XLS includes
            filtered_lines.append(line)
        harness += '\n'.join(filtered_lines)
        
        # Generate main() with test cases
        harness += '''

int main() {
'''
        for i, (inputs, expected) in enumerate(test_cases):
            args_str = ", ".join(str(x) for x in inputs)
            harness += f'''
    // Test case {i + 1}
    {{
        int result = {func_name}({args_str});
        std::cout << result << std::endl;
    }}
'''
        harness += '''
    return 0;
}
'''
        return harness
    
    def _generate_sequential_harness(self,
                                       cpp_code: str,
                                       func_name: str,
                                       input_arrays: List[np.ndarray],
                                       output_size: int) -> str:
        """
        Generate a test harness for sequential logic with channels.
        
        The harness:
        1. Creates a TestBlock instance
        2. Pushes input data to input channels
        3. Runs the top function
        4. Reads output data from output channel
        """
        
        harness = f'''
// Test harness for sequential logic with channels
#include <iostream>
#include <cstdint>

// Include the mock headers
#include "xls_builtin.h"
#include "xls_int.h"

// Wrap the ac_int alias
template <int Width, bool Signed = true>
using ac_int = XlsInt<Width, Signed>;

// Generated code (remove the #include directives that reference XLS paths)
'''
        
        # Remove XLS-specific includes from the generated code
        lines = cpp_code.split('\n')
        filtered_lines = []
        for line in lines:
            if '#include' in line and ('xls' in line.lower() or '/xls' in line):
                continue  # Skip XLS includes
            filtered_lines.append(line)
        harness += '\n'.join(filtered_lines)
        
        # Generate main()
        harness += '''

int main() {
    TestBlock block;
    
'''
        # Push input data to channels
        for i, arr in enumerate(input_arrays):
            flat = arr.flatten()
            for val in flat:
                harness += f'    block.v{i}_in.push_input({int(val)});\n'
        
        # Run the function enough times to process all data
        # For sequential designs, we need to run multiple times
        total_inputs = sum(arr.size for arr in input_arrays)
        total_runs = total_inputs + output_size + 100  # Add margin for state machine
        
        harness += f'''
    // Run the function multiple times to process all data
    for (int i = 0; i < {total_runs}; i++) {{
        try {{
            block.{func_name}();
        }} catch (...) {{
            // Input channels may be empty
        }}
    }}
    
    // Read output data
    for (int i = 0; i < {output_size}; i++) {{
        try {{
            int val = block.out.pop_output();
            std::cout << val << " ";
        }} catch (...) {{
            std::cout << "ERR ";
        }}
    }}
    std::cout << std::endl;
    
    return 0;
}}
'''
        return harness
    
    def test_combinational(self,
                           allo_func: Callable,
                           schedule,
                           test_cases: List[Tuple[Tuple, Any]],
                           test_name: Optional[str] = None,
                           use_memory: bool = False) -> List[TestResult]:
        """
        Test a combinational Allo function.
        
        Args:
            allo_func: The Allo function to test
            schedule: The allo.customize() schedule object
            test_cases: List of ((input1, input2, ...), expected_output) tuples
            test_name: Name for the test (default: function name)
            use_memory: Whether to use memory mode
        
        Returns:
            List of TestResult objects
        """
        test_name = test_name or allo_func.__name__
        
        if self.verbose:
            print(f"\n--- Testing {test_name} (combinational) ---")
        
        # Build the XLS module
        try:
            mod = schedule.build(
                target="xlscc",
                project=os.path.join(self.output_dir, f"{test_name}_prj"),
                use_memory=use_memory
            )
            cpp_code = str(mod)
        except Exception as e:
            result = TestResult(
                passed=False,
                test_name=test_name,
                inputs=None,
                expected=None,
                actual=None,
                error_message=f"Build failed: {e}"
            )
            self.results.append(result)
            print("F", end="", flush=True)
            return [result]
        
        # Generate test harness
        harness = self._generate_combinational_harness(
            cpp_code, allo_func.__name__, test_cases
        )
        
        # Compile
        success, exe_path, err = self._compile_cpp(harness, test_name)
        if not success:
            result = TestResult(
                passed=False,
                test_name=test_name,
                inputs=None,
                expected=None,
                actual=None,
                error_message=err
            )
            self.results.append(result)
            print("F", end="", flush=True)
            if self.verbose:
                print(f"\n{err}")
            return [result]
        
        # Run and check results
        success, stdout, stderr = self._run_executable(exe_path)
        if not success:
            result = TestResult(
                passed=False,
                test_name=test_name,
                inputs=None,
                expected=None,
                actual=None,
                error_message=f"Execution failed: {stderr}"
            )
            self.results.append(result)
            print("F", end="", flush=True)
            return [result]
        
        # Parse output and compare with expected
        outputs = stdout.strip().split('\n')
        results = []
        
        for i, ((inputs, expected), output_str) in enumerate(zip(test_cases, outputs)):
            try:
                actual = int(output_str.strip())
                passed = actual == expected
            except ValueError:
                actual = output_str
                passed = False
            
            result = TestResult(
                passed=passed,
                test_name=f"{test_name}[{i}]",
                inputs=inputs,
                expected=expected,
                actual=actual
            )
            results.append(result)
            self.results.append(result)
            
            if passed:
                print(".", end="", flush=True)
            else:
                print("F", end="", flush=True)
                if self.verbose:
                    print(f"\n  FAIL: {inputs} -> expected {expected}, got {actual}")
        
        return results
    
    def test_sequential(self,
                        allo_func: Callable,
                        schedule,
                        test_cases: List[Tuple[Tuple, Any]],
                        test_name: Optional[str] = None,
                        use_memory: bool = False) -> List[TestResult]:
        """
        Test a sequential Allo function with array inputs.
        
        Args:
            allo_func: The Allo function to test
            schedule: The allo.customize() schedule object
            test_cases: List of ((input_array1, input_array2, ...), expected_output_array) tuples
            test_name: Name for the test (default: function name)
            use_memory: Whether to use memory mode
        
        Returns:
            List of TestResult objects
        """
        test_name = test_name or allo_func.__name__
        
        if self.verbose:
            print(f"\n--- Testing {test_name} (sequential) ---")
        
        # Build the XLS module
        try:
            mod = schedule.build(
                target="xlscc",
                project=os.path.join(self.output_dir, f"{test_name}_prj"),
                use_memory=use_memory
            )
            cpp_code = str(mod)
        except Exception as e:
            result = TestResult(
                passed=False,
                test_name=test_name,
                inputs=None,
                expected=None,
                actual=None,
                error_message=f"Build failed: {e}"
            )
            self.results.append(result)
            print("F", end="", flush=True)
            return [result]
        
        results = []
        
        for case_idx, (inputs, expected) in enumerate(test_cases):
            # Convert inputs to numpy arrays
            input_arrays = [np.array(arr, dtype=np.int32) for arr in inputs]
            expected_arr = np.array(expected, dtype=np.int32).flatten()
            
            # Generate test harness for this specific test case
            harness = self._generate_sequential_harness(
                cpp_code, 
                allo_func.__name__,
                input_arrays,
                expected_arr.size
            )
            
            case_name = f"{test_name}_case{case_idx}"
            
            # Compile
            success, exe_path, err = self._compile_cpp(harness, case_name)
            if not success:
                result = TestResult(
                    passed=False,
                    test_name=f"{test_name}[{case_idx}]",
                    inputs=inputs,
                    expected=expected,
                    actual=None,
                    error_message=err
                )
                results.append(result)
                self.results.append(result)
                print("F", end="", flush=True)
                if self.verbose:
                    print(f"\n{err}")
                continue
            
            # Run
            success, stdout, stderr = self._run_executable(exe_path)
            if not success:
                result = TestResult(
                    passed=False,
                    test_name=f"{test_name}[{case_idx}]",
                    inputs=inputs,
                    expected=expected,
                    actual=None,
                    error_message=f"Execution failed: {stderr}"
                )
                results.append(result)
                self.results.append(result)
                print("F", end="", flush=True)
                continue
            
            # Parse output
            try:
                output_vals = [int(x) for x in stdout.strip().split() if x != "ERR"]
                actual_arr = np.array(output_vals, dtype=np.int32)
                
                # Compare
                if actual_arr.shape == expected_arr.shape and np.array_equal(actual_arr, expected_arr):
                    passed = True
                else:
                    passed = False
            except Exception as e:
                actual_arr = stdout
                passed = False
            
            result = TestResult(
                passed=passed,
                test_name=f"{test_name}[{case_idx}]",
                inputs=inputs,
                expected=expected_arr.tolist() if isinstance(expected_arr, np.ndarray) else expected,
                actual=actual_arr.tolist() if isinstance(actual_arr, np.ndarray) else actual_arr
            )
            results.append(result)
            self.results.append(result)
            
            if passed:
                print(".", end="", flush=True)
            else:
                print("F", end="", flush=True)
                if self.verbose:
                    print(f"\n  FAIL: expected {expected_arr}, got {actual_arr}")
        
        return results
    
    def summary(self) -> str:
        """Return a summary of all test results."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        summary = f"\n\n{'=' * 60}\n"
        summary += f"Test Summary: {passed} passed, {failed} failed\n"
        summary += f"{'=' * 60}\n"
        
        if failed > 0:
            summary += "\nFailed tests:\n"
            for r in self.results:
                if not r.passed:
                    summary += f"  - {r.test_name}\n"
                    if r.error_message:
                        summary += f"    Error: {r.error_message[:100]}...\n"
                    else:
                        summary += f"    Expected: {r.expected}\n"
                        summary += f"    Actual: {r.actual}\n"
        
        return summary
    
    def print_summary(self):
        """Print test summary."""
        print(self.summary())
    
    def all_passed(self) -> bool:
        """Return True if all tests passed."""
        return all(r.passed for r in self.results)


def run_tests():
    """Run example tests to verify the framework works."""
    print("XLS[cc] Test Framework - Self Test")
    print("=" * 60)
    
    # Simple combinational test
    from allo.ir.types import int32
    
    def add(a: int32, b: int32) -> int32:
        return a + b
    
    s = allo.customize(add)
    
    runner = XLSTestRunner(verbose=True)
    
    runner.test_combinational(
        allo_func=add,
        schedule=s,
        test_cases=[
            ((2, 3), 5),
            ((10, -5), 5),
            ((0, 0), 0),
            ((-1, 1), 0),
        ]
    )
    
    runner.print_summary()
    return runner.all_passed()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)

