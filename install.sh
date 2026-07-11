#!/bin/bash

# Wallpaper Generator Installer
# Installs Python dependencies and creates a shell wrapper in ~/.local/bin

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.local/bin"
BINARY_NAME="wp-generator"

echo "Installing locked Python dependencies..."
uv sync --project "$REPO_DIR" --quiet

mkdir -p "$INSTALL_DIR"

echo "Installing $BINARY_NAME to $INSTALL_DIR..."
cat > "$INSTALL_DIR/$BINARY_NAME" <<EOF
#!/bin/bash
exec uv run --project "$REPO_DIR" python "$REPO_DIR/main.py" "\$@"
EOF
chmod +x "$INSTALL_DIR/$BINARY_NAME"

# Remove old system-wide binary if present (may need sudo once to clean up)
if [ -f "/usr/local/bin/nothing-wp" ] || [ -f "/usr/local/bin/$BINARY_NAME" ]; then
    echo "Old system binary found in /usr/local/bin — remove it with:"
    echo "  sudo rm -f /usr/local/bin/nothing-wp /usr/local/bin/$BINARY_NAME"
fi

# Warn if ~/.local/bin is not on PATH
if ! echo "$PATH" | tr ':' '\n' | grep -qx "$INSTALL_DIR"; then
    echo ""
    echo "Add this to your ~/.zshrc (or ~/.bashrc) to put it on PATH:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
echo "Done. Run '$BINARY_NAME --help' from any directory."
