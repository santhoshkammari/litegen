#!/bin/bash

set -e


echo "Uninstalling existing package..."
pip uninstall -y ailite

echo "Building package..."
python -m build

echo "Installing wheel file..."
pip install dist/*.whl

echo "Uploading to PyPI..."
twine upload dist/*

echo "Deployment completed successfully!"