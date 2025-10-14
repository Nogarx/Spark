
import sys
import os
import pytest

if __name__ == '__main__':
    # Add the parent directory (where your package resides) to sys.path
    #repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    #sys.path.insert(0, repo_root)

    # Run pytest on the package or tests directory
    # Adjust 'tests' if your test folder has a different name
    sys.exit(
        pytest.main([
            'spark/tests/utils_shape.py', 
            'spark/tests/initializers.py',
            'spark/tests/module_execution.py',
            'spark/tests/brain_execution.py'
        ])
    )