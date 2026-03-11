#!/bin/bash

# Nothing-Style Wallpaper Generator Installer
# Installs the CLI tool globally as "nothing-wp"

set -e

INSTALL_DIR="/usr/local/bin"
REPO_DIR="$(pwd)"

echo "Installing Nothing Wallpaper Generator..."

# Ensure dependencies are installed
echo "Installing Python dependencies..."
pip install -r requirements.txt --quiet

# Create the wrapper script
WRAPPER_SCRIPT="$INSTALL_DIR/nothing-wp"

echo "Creating executable at $WRAPPER_SCRIPT..."

# Use sudo if necessary to write to /usr/local/bin
sudo bash -c "cat > \"$WRAPPER_SCRIPT\" << EOF
#!/bin/bash
# Wrapper script to execute the generator from its install directory
cd \"$REPO_DIR\"
python main.py \"\\\$@\"
EOF"

sudo chmod +x "$WRAPPER_SCRIPT"

echo "Installation complete! 🎉"
echo "You can now generate wallpapers from anywhere by typing: nothing-wp"
