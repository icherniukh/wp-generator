#!/bin/bash

# Nothing-Style Wallpaper Generator Installer
# Packages the tool into a standalone binary and installs it globally

set -e

INSTALL_DIR="/usr/local/bin"
BINARY_NAME="nothing-wp"

echo "📦 Packaging Nothing Wallpaper Generator into a standalone binary..."

# Ensure dependencies are installed for the build
echo "Installing Python dependencies..."
pip install -r requirements.txt --quiet

# Use PyInstaller to create a single-file executable
# --collect-all ensures data files for heavy libraries are included
echo "Building binary (this may take a minute)..."
pyinstaller --onefile \
            --name "$BINARY_NAME" \
            --collect-all colour \
            --collect-all scipy \
            --collect-all numpy \
            --hidden-import engine.gradient \
            --hidden-import engine.filters \
            --hidden-import engine.colors \
            --hidden-import tools.sampler \
            main.py --noconfirm

# Move the resulting binary to /usr/local/bin
echo "Installing binary to $INSTALL_DIR/$BINARY_NAME..."
sudo mv "dist/$BINARY_NAME" "$INSTALL_DIR/$BINARY_NAME"

# Clean up build artifacts
echo "Cleaning up build artifacts..."
rm -rf build dist "$BINARY_NAME.spec"

echo "Installation complete! 🎉"
echo "You can now run '$BINARY_NAME' from any directory."
