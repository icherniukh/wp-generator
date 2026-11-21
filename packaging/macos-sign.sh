#!/usr/bin/env bash
# Signs every Mach-O file in a PyInstaller folder build with the hardened runtime,
# main executable last, then verifies the signature.
# Usage: packaging/macos-sign.sh <dist folder> <signing identity>
set -euo pipefail
dir=$1
identity=$2
main="$dir/wp-generator"

while IFS= read -r -d '' file; do
    [ "$file" = "$main" ] && continue
    if file -b "$file" | grep -q "Mach-O"; then
        codesign --force --options runtime --timestamp --sign "$identity" "$file"
    fi
done < <(find "$dir" -type f -print0)

codesign --force --options runtime --timestamp --sign "$identity" "$main"
codesign --verify --strict --verbose=2 "$main"
