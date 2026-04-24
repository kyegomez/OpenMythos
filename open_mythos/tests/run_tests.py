"""
Test Runner - Run all tests

Usage:
    python tests/run_tests.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def main():
    """Run all tests."""
    print("Running OpenMythos Tests...")
    print("=" * 50)
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__.replace("run_tests.py", ""),  # tests directory
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ])
    
    print("=" * 50)
    
    if exit_code == 0:
        print("All tests passed!")
    else:
        print(f"Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
