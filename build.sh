#!/bin/bash

set -e


echo "Uninstalling existing package..."
pip uninstall -y ailite

if [ -d "ailite/dist" ]; then
    rm -r ailite/dist
fi

echo "Building package..."
poetry build

echo "Installing wheel file..."
pip install dist/*.whl

echo "Uploading to PyPI..."
poetry publish

echo "Deployment completed successfully!"