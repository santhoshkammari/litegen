#!/bin/bash

set -e

# Check if dist directory exists before deleting
if [ -d "dist" ]; then
    rm -r dist
fi
pip uninstall -y ailite

python -m build

# Install the wheel file
pip install dist/*.whl

# Upload to PyPI
twine upload dist/*