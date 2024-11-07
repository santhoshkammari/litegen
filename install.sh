#!/bin/bash

set -e

# Check if dist directory exists before deleting
if [ -d "dist" ]; then
    rm -r dist
fi
pip uninstall -y ailite

python -m build

twine upload dist/*