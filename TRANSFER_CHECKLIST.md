# Repository Transfer Checklist

Guide for transferring arbiter from `evanvolgas` to `ashita-ai` organization.

## Pre-Transfer

- [ ] Verify you have admin access to `ashita-ai` organization
- [ ] Confirm no existing `arbiter` repository in `ashita-ai`
- [ ] Note current PR #22 details (will need to recreate)
- [ ] Save local work: `git push origin feature/pypi-publishing`

## Transfer Repository

### Via GitHub Web UI

1. Go to: https://github.com/evanvolgas/arbiter/settings
2. Scroll to **"Danger Zone"** section
3. Click **"Transfer ownership"**
4. Enter new owner: `ashita-ai`
5. Type `arbiter` to confirm
6. Click **"I understand, transfer this repository"**

### What Transfers Automatically

✅ All code and git history
✅ All issues (#15, #16, #17)
✅ All pull requests (as references, not active PRs)
✅ All releases and tags
✅ GitHub Actions workflows
✅ Repository settings
✅ Branch protection rules

### What Requires Manual Updates

❌ GitHub Actions secrets (PYPI_API_TOKEN if configured)
❌ PyPI trusted publishing configuration
❌ Local git remotes
❌ Active pull requests (need recreation)

## Post-Transfer Updates

### 1. Update URLs in Codebase

```bash
# Run the automated update script
./scripts/update_github_urls.sh

# Review changes
git diff

# Commit changes
git commit -am "Update GitHub URLs after org transfer"
```

**Files Updated by Script:**
- `pyproject.toml` - project.urls section
- `CONTRIBUTING.md` - all documentation links
- `README.md` - badges, links, examples
- `.github/workflows/publish-to-pypi.yml` - workflow comments
- `AGENTS.md` - if contains GitHub URLs

### 2. Update Local Git Configuration

```bash
# Update remote URL
git remote set-url origin git@github.com:ashita-ai/arbiter.git

# Verify
git remote -v

# Push updated branch
git push origin feature/pypi-publishing
```

### 3. Recreate Pull Request

Original PR #22 won't automatically transfer as an active PR.

```bash
# Go to new repository
# https://github.com/ashita-ai/arbiter

# Create new PR from feature/pypi-publishing branch
# Use same title and description from original PR #22
```

**Original PR Details:**
- **Title**: "Prepare package for PyPI publication"
- **Branch**: `feature/pypi-publishing`
- **Closes**: #17 (will auto-transfer to new repo)
- **Description**: See https://github.com/evanvolgas/arbiter/pull/22

### 4. Update PyPI Configuration

If you've already configured PyPI trusted publishing:

1. Go to: https://pypi.org/manage/account/publishing/
2. Find existing publisher for `arbiter-ai`
3. Update configuration:
   - **Owner**: `evanvolgas` → `ashita-ai`
   - **Repository**: `arbiter`
   - **Workflow**: `publish-to-pypi.yml`
   - **Environment**: (leave blank)
4. Save changes

**Note**: If not yet configured, wait until after first release to set up trusted publishing.

### 5. Update GitHub Actions Secrets (if applicable)

If using API token authentication instead of trusted publishing:

1. Go to: https://github.com/ashita-ai/arbiter/settings/secrets/actions
2. Add secret: `PYPI_API_TOKEN`
3. Value: Your PyPI API token

### 6. Verify Everything Works

```bash
# Clone from new location (fresh test)
git clone git@github.com:ashita-ai/arbiter.git
cd arbiter

# Check URLs in files
grep -r "evanvolgas" . --exclude-dir=.git

# Verify builds
uv sync --all-extras
make test
python -m build

# Verify all URLs resolve
curl -I https://github.com/ashita-ai/arbiter
curl -I https://github.com/ashita-ai/arbiter/issues
```

## Post-Transfer Verification

- [ ] Repository accessible at `https://github.com/ashita-ai/arbiter`
- [ ] All issues transferred and accessible
- [ ] All URLs in codebase updated
- [ ] Local git remote updated
- [ ] New PR created from feature/pypi-publishing branch
- [ ] GitHub Actions workflows work (check Actions tab)
- [ ] PyPI trusted publishing configured (or API token set)
- [ ] README badges point to correct repository
- [ ] No references to `evanvolgas/arbiter` remain (except git history)

## Rollback (if needed)

If something goes wrong, you can transfer back:

1. Go to: https://github.com/ashita-ai/arbiter/settings
2. Transfer ownership back to: `evanvolgas`
3. Revert URL changes: `git checkout HEAD -- pyproject.toml CONTRIBUTING.md README.md`

## Timeline

Estimated time: **15-20 minutes**

- Transfer: ~2 minutes
- URL updates: ~5 minutes
- Testing: ~10 minutes

## Support

If issues occur:
- GitHub transfer documentation: https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository
- PyPI trusted publishing: https://docs.pypi.org/trusted-publishers/
