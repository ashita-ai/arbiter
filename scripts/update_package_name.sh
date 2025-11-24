#!/bin/bash
# Update package name from "arbiter" to "arbiter-ai" for PyPI publication
# Import name stays "arbiter" - only PyPI package name changes
# Usage: ./scripts/update_package_name.sh

set -e

OLD_NAME="arbiter"
NEW_NAME="arbiter-ai"

echo "📦 Updating PyPI package name: $OLD_NAME → $NEW_NAME"
echo "   (Import name stays: from arbiter import evaluate)"
echo ""

# Update pyproject.toml
echo "📝 Updating pyproject.toml"
sed -i.bak 's/^name = "arbiter"$/name = "arbiter-ai"/' pyproject.toml
rm pyproject.toml.bak

# Update README.md - pip install commands
echo "📝 Updating README.md"
sed -i.bak 's/pip install arbiter/pip install arbiter-ai/g' README.md
sed -i.bak 's|pypi.org/project/arbiter|pypi.org/project/arbiter-ai|g' README.md
rm README.md.bak

# Update CONTRIBUTING.md
echo "📝 Updating CONTRIBUTING.md"
sed -i.bak 's/pip install arbiter/pip install arbiter-ai/g' CONTRIBUTING.md
sed -i.bak 's|pypi.org/project/arbiter|pypi.org/project/arbiter-ai|g' CONTRIBUTING.md
rm CONTRIBUTING.md.bak

# Update TRANSFER_CHECKLIST.md if it exists
if [ -f "TRANSFER_CHECKLIST.md" ]; then
    echo "📝 Updating TRANSFER_CHECKLIST.md"
    sed -i.bak 's/for `arbiter`/for `arbiter-ai`/g' TRANSFER_CHECKLIST.md
    sed -i.bak 's/Find existing publisher for `arbiter`/Find existing publisher for `arbiter-ai`/g' TRANSFER_CHECKLIST.md
    rm TRANSFER_CHECKLIST.md.bak
fi

echo ""
echo "✅ Package name updates complete!"
echo ""
echo "📋 Changed:"
echo "   - PyPI package name: arbiter → arbiter-ai"
echo "   - pip install commands updated"
echo "   - PyPI URLs updated"
echo ""
echo "📋 Unchanged (intentionally):"
echo "   - Python import: 'from arbiter import evaluate' (stays the same)"
echo "   - Repository name: arbiter"
echo "   - Directory structure: arbiter/"
echo ""
echo "🔍 Review changes:"
echo "   git diff pyproject.toml README.md CONTRIBUTING.md"
echo ""
echo "✅ Next steps:"
echo "1. Review changes carefully"
echo "2. Rebuild package: rm -rf dist/ && python -m build"
echo "3. Verify: twine check dist/*"
echo "4. Commit: git commit -am 'Change PyPI package name to arbiter-ai'"
