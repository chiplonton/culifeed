# CuliFeed Release Process

This document outlines the step-by-step process for preparing and creating releases of CuliFeed.

## 📋 Prerequisites

- [ ] All changes are merged into `main` branch
- [ ] All tests are passing
- [ ] Code quality checks are satisfied
- [ ] Main branch is protected (requires PR for changes)

## 🏷️ Versioning Strategy

CuliFeed follows [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., `1.2.3`)
- **MAJOR**: Breaking changes or major feature additions
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Version Decision Guide

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Bug fix | PATCH | `1.1.0` → `1.1.1` |
| New feature | MINOR | `1.1.0` → `1.2.0` |
| Breaking change | MAJOR | `1.1.0` → `2.0.0` |

## 🔄 Release Process

### Step 1: Preparation

1. **Check current version and recent changes**
   ```bash
   git log --oneline -10
   git status
   ```

2. **Ensure you're on main branch with latest changes**
   ```bash
   git checkout main
   git pull origin main
   ```

### Step 2: Update Version and Documentation

1. **Update Version in Code**

   Update the version number in `culifeed/__init__.py`:
   ```python
   __version__ = "X.Y.Z"
   ```

2. **Update CHANGELOG.md**

   Add new version section at the top:
   ```markdown
   ## [X.Y.Z] - YYYY-MM-DD

   ### Added / Changed / Fixed / Removed
   - Description of changes
   ```

   Follow the [Keep a Changelog](https://keepachangelog.com/) format:
   - **Added**: New features
   - **Changed**: Changes in existing functionality
   - **Deprecated**: Soon-to-be removed features
   - **Removed**: Removed features
   - **Fixed**: Bug fixes
   - **Security**: Security vulnerability fixes

### Step 3: Create Release Branch

Since `main` is protected, create a release preparation branch:

```bash
# Create release branch
git checkout -b release/vX.Y.Z

# Stage and commit changelog
git add CHANGELOG.md
git commit -m "chore: Prepare release vX.Y.Z

- Update CHANGELOG.md with version X.Y.Z details
- Document new features/fixes/improvements
- [PATCH|MINOR|MAJOR] version bump for [bug fix|new feature|breaking change]

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Step 4: Create Git Tag

```bash
# Create annotated tag with detailed message
git tag -a vX.Y.Z -m "Release vX.Y.Z: [Brief Description]

- [Key improvement 1]
- [Key improvement 2]
- [Key fix 1]

[Detailed description of what this release addresses]"
```

### Step 5: Push Changes

```bash
# Push release branch
git push -u origin release/vX.Y.Z

# Push tag
git push origin vX.Y.Z
```

### Step 6: Create Pull Request

Create PR from `release/vX.Y.Z` to `main`:
- **Title**: `Release vX.Y.Z: [Brief Description]`
- **Description**: Copy the changelog entries for this version
- **Link**: Use the GitHub URL provided in push output

### Step 7: Create GitHub Release

After PR is merged, create GitHub release:

1. **Go to**: https://github.com/chiplonton/culifeed/releases/new
2. **Tag**: `vX.Y.Z` (should already exist)
3. **Title**: `CuliFeed vX.Y.Z: [Brief Description]`
4. **Description**: Use the template below

## 📝 GitHub Release Template

```markdown
## 🔧 [Bug Fixes / New Features / Breaking Changes]

### [Main Feature/Fix Name]
- **[Primary improvement]**: Description of main change
- **[Secondary improvement]**: Description of additional change
- **[Enhancement]**: Description of user experience improvement

## 📝 Technical Improvements

- **[Technical change 1]**: Description
- **[Technical change 2]**: Description
- **[Performance/Security]**: Description

## 🐛 Issues Resolved / ✨ Features Added

Brief explanation of what problems this release solves or what new capabilities it adds.

**Before**: [Description of old behavior]
**After**: [Description of new behavior]

## 📋 What's Changed

- [List of main PRs and commits]
- [Reference to significant changes]

**Full Changelog**: https://github.com/chiplonton/culifeed/compare/vPREV...vX.Y.Z

---

🤖 *This release was prepared with [Claude Code](https://claude.ai/code)*
```

## 🎯 Release Examples

### Patch Release (Bug Fix)
```bash
# Version: 1.1.0 → 1.1.1
git checkout -b release/v1.1.1
# Update CHANGELOG.md with bug fixes
git commit -m "chore: Prepare release v1.1.1"
git tag -a v1.1.1 -m "Release v1.1.1: Fix edittopic parsing"
```

### Minor Release (New Feature)
```bash
# Version: 1.1.1 → 1.2.0
git checkout -b release/v1.2.0
# Update CHANGELOG.md with new features
git commit -m "chore: Prepare release v1.2.0"
git tag -a v1.2.0 -m "Release v1.2.0: Add webhook integration"
```

### Major Release (Breaking Changes)
```bash
# Version: 1.2.0 → 2.0.0
git checkout -b release/v2.0.0
# Update CHANGELOG.md with breaking changes
git commit -m "chore: Prepare release v2.0.0"
git tag -a v2.0.0 -m "Release v2.0.0: New API architecture"
```

## 🚨 Protected Branch Workflow

Since `main` is protected:

1. **Always use release branches** (`release/vX.Y.Z`)
2. **Create PR for all changes** to main
3. **Tag after PR is merged** to main
4. **Create GitHub release** using the merged tag

## ✅ Release Checklist

**Preparation:**
- [ ] All intended changes are merged to main
- [ ] Tests are passing
- [ ] Version number determined (MAJOR.MINOR.PATCH)

**Documentation:**
- [ ] CHANGELOG.md updated with new version
- [ ] Release notes prepared
- [ ] Breaking changes documented (if any)

**Git Operations:**
- [ ] Release branch created (`release/vX.Y.Z`)
- [ ] Changes committed to release branch
- [ ] Git tag created (`vX.Y.Z`)
- [ ] Release branch pushed to origin
- [ ] Git tag pushed to origin

**GitHub:**
- [ ] Pull request created (release branch → main)
- [ ] Pull request reviewed and merged
- [ ] GitHub release created with detailed notes
- [ ] Release published

**Post-Release:**
- [ ] Deployment successful (if applicable)
- [ ] Release announcement made (if applicable)
- [ ] Release branch cleaned up

## 🔧 Troubleshooting

### Protected Branch Rejection
```
remote: error: GH006: Protected branch update failed for refs/heads/main.
```
**Solution**: Use release branch workflow instead of direct push to main.

### Tag Already Exists
```
fatal: tag 'vX.Y.Z' already exists
```
**Solution**: Delete local tag and create new one:
```bash
git tag -d vX.Y.Z
git tag -a vX.Y.Z -m "New message"
```

### Release Branch Conflicts
**Solution**: Rebase against latest main:
```bash
git checkout release/vX.Y.Z
git rebase main
git push --force-with-lease origin release/vX.Y.Z
```

## 📚 References

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [GitHub Releases Documentation](https://docs.github.com/en/repositories/releasing-projects-on-github)
- [CuliFeed Contributing Guidelines](./CONTRIBUTING.md)