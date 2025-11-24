#!/bin/bash
# Update GitHub URLs after repository transfer to ashita-ai organization
# Usage: ./scripts/update_github_urls.sh

set -e

OLD_OWNER="evanvolgas"
NEW_OWNER="ashita-ai"
REPO_NAME="arbiter"

echo "🔄 Updating GitHub URLs from $OLD_OWNER/$REPO_NAME to $NEW_OWNER/$REPO_NAME"
echo ""

# Function to update URLs in a file
update_file() {
    local file=$1
    if [ -f "$file" ]; then
        echo "📝 Updating $file"
        sed -i.bak "s|github.com/$OLD_OWNER/$REPO_NAME|github.com/$NEW_OWNER/$REPO_NAME|g" "$file"
        rm "$file.bak"
    else
        echo "⚠️  File not found: $file"
    fi
}

# Update pyproject.toml
update_file "pyproject.toml"

# Update CONTRIBUTING.md
update_file "CONTRIBUTING.md"

# Update README.md
update_file "README.md"

# Update AGENTS.md (if it contains GitHub URLs)
if [ -f "AGENTS.md" ] && grep -q "github.com/$OLD_OWNER/$REPO_NAME" AGENTS.md; then
    update_file "AGENTS.md"
fi

# Update GitHub Actions workflow
update_file ".github/workflows/publish-to-pypi.yml"

echo ""
echo "✅ URL updates complete!"
echo ""
echo "📋 Next steps:"
echo "1. Review changes: git diff"
echo "2. Update git remote: git remote set-url origin git@github.com:$NEW_OWNER/$REPO_NAME.git"
echo "3. Commit changes: git commit -am 'Update GitHub URLs after org transfer'"
echo "4. Push to new location: git push origin feature/pypi-publishing"
echo "5. Recreate PR at: https://github.com/$NEW_OWNER/$REPO_NAME/compare"
echo ""
echo "🔐 Don't forget to update PyPI trusted publishing:"
echo "   https://pypi.org/manage/account/publishing/"
echo "   Owner: $OLD_OWNER → $NEW_OWNER"
