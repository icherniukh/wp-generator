#!/bin/bash

# Wallpaper Generator Installer
# Installs Python dependencies and creates a global shell wrapper

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="/usr/local/bin"
BINARY_NAME="wp-generator"

echo "Installing Python dependencies..."
pip install -r "$REPO_DIR/requirements.txt" --quiet

echo "Installing $BINARY_NAME to $INSTALL_DIR..."
sudo tee "$INSTALL_DIR/$BINARY_NAME" > /dev/null <<EOF
#!/bin/bash
exec python3 "$REPO_DIR/main.py" "\$@"
EOF
sudo chmod +x "$INSTALL_DIR/$BINARY_NAME"

# Remove old binary name if present
if [ -f "$INSTALL_DIR/nothing-wp" ]; then
    echo "Removing old 'nothing-wp' binary..."
    sudo rm "$INSTALL_DIR/nothing-wp"
fi

echo "Done. Run '$BINARY_NAME --help' from any directory."
