#!/bin/bash

set -e

# Git operations
echo "Performing Git operations..."
git add .
if git diff --staged --quiet; then
    echo "No changes to commit"
else
    git commit -m "Updated files before deployment"
    git push
fi

# Check if dist directory exists before deleting
if [ -d "dist" ]; then
    echo "Removing existing dist directory..."
    rm -r dist
fi

echo "Uninstalling existing package..."
pip uninstall -y ailite

echo "Building package..."
python -m build

echo "Installing wheel file..."
pip install dist/*.whl

echo "Uploading to PyPI..."
twine upload dist/*

echo "Deployment completed successfully!"