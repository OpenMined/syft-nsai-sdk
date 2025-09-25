# 🔄 GitHub Actions CI/CD Setup Guide

Complete guide for setting up automated CI/CD pipelines for the SyftBox Hub package.

## 📁 Files Created

| File | Purpose |
|------|---------|
| `.github/workflows/build-and-test.yml` | Multi-platform testing and building |
| `.github/workflows/publish.yml` | Publishing to PyPI/TestPyPI |
| `.github/workflows/release.yml` | Automated release creation |
| `.github/workflows/codeql.yml` | Security analysis |
| `.github/dependabot.yml` | Automated dependency updates |

## 🚀 Quick Setup Checklist

### 1. Repository Secrets

Go to **Settings** → **Secrets and variables** → **Actions**:

- [ ] Add `PYPI_API_TOKEN` (production PyPI)
- [ ] Add `TESTPYPI_API_TOKEN` (testing PyPI)

### 2. Environment Setup

Go to **Settings** → **Environments**:

- [ ] Create `pypi` environment
  - [ ] Add `PYPI_API_TOKEN` as environment secret
  - [ ] Optional: Enable protection rules
- [ ] Create `testpypi` environment  
  - [ ] Add `TESTPYPI_API_TOKEN` as environment secret

### 3. Branch Protection (Optional)

Go to **Settings** → **Branches**:

- [ ] Protect `main` branch
- [ ] Require status checks: `test`, `build`
- [ ] Require up-to-date branches

## 🔧 How to Get API Tokens

### PyPI Production Token
1. Visit [PyPI Account Settings](https://pypi.org/manage/account/)
2. **API tokens** → **Add API token**
3. Name: `syft-hub-github-actions`
4. Scope: **Entire account** or specific to `syft-hub`
5. Copy token (starts with `pypi-`)

### TestPyPI Token
1. Visit [TestPyPI Account Settings](https://test.pypi.org/manage/account/)
2. Follow same steps as above
3. Copy token

## 🎯 Usage

### Creating a Release

**Full Release (Recommended):**
1. **Actions** → **Create Release** → **Run workflow**
2. Select version bump type:
   - `patch` - Bug fixes (1.0.0 → 1.0.1)
   - `minor` - New features (1.0.0 → 1.1.0)
   - `major` - Breaking changes (1.0.0 → 2.0.0)
3. Options:
   - ✅ **Publish to PyPI** - Automatic publishing
   - ❌ **Pre-release** - Mark as stable
   - ❌ **Draft** - Public release

**Result:**
- Version updated in `pyproject.toml`
- Git tag created (`v1.2.3`)
- GitHub release with changelog
- PyPI package published
- Assets attached to release

### Manual Publishing

**Test First:**
1. **Actions** → **Publish Package** → **Run workflow**
2. Target: `testpypi`
3. Test: `pip install -i https://test.pypi.org/simple/ syft-hub`

**Then Production:**
1. **Actions** → **Publish Package** → **Run workflow**
2. Target: `pypi`

### Automatic Testing

Tests run automatically on:
- ✅ Push to `main`, `develop`, `fixes_launch`
- ✅ Pull requests to `main`, `develop`
- ✅ Manual trigger

## 🛡️ Security Features

### CodeQL Analysis
- **Frequency:** Weekly + on push/PR
- **Scope:** Security vulnerabilities, code quality
- **Results:** Security tab in GitHub

### Dependency Updates
- **Dependabot:** Weekly updates
- **Grouping:** Related packages grouped together
- **Auto-merge:** Configure in repository settings

### Build Security
- **Safety:** Python package vulnerability scanning
- **Bandit:** Static security analysis
- **Non-blocking:** Warnings don't fail builds

## 🔄 Workflow Triggers

| Workflow | Triggers |
|----------|----------|
| **Build and Test** | Push, PR, manual |
| **Publish Package** | Manual, GitHub release |
| **Create Release** | Manual only |
| **CodeQL** | Push, PR, weekly |

## 📊 Monitoring

### Build Status
- Check **Actions** tab for workflow status
- Green ✅ = Success
- Red ❌ = Failed (check logs)
- Yellow 🟡 = In progress

### Notifications
- GitHub notifications for failed workflows
- Email notifications (configure in settings)
- Slack integration (optional)

## 🚨 Troubleshooting

### Common Issues

**Authentication Error:**
```
HTTP Error 403: Invalid or non-existent authentication information
```
- Check API tokens are correct
- Verify token permissions
- Ensure environment secrets are set

**Version Already Exists:**
```
HTTP Error 400: File already exists
```
- Version already published to PyPI
- Use "Create Release" to bump version
- Or manually update version in `pyproject.toml`

**Build Failures:**
- Check Python syntax errors
- Verify dependencies in `pyproject.toml`
- Review test failures in logs

### Manual Verification

**Test package installation:**
```bash
# From TestPyPI
pip install -i https://test.pypi.org/simple/ syft-hub==NEW_VERSION

# From PyPI
pip install syft-hub==NEW_VERSION

# Test import
python -c "import syft_hub; print('✅ Success')"
```

## 🎉 Benefits

### Automated Benefits
- ✅ **Zero-downtime releases** - Automated process
- ✅ **Multi-platform testing** - Linux, macOS, Windows
- ✅ **Version management** - Automatic semver bumping
- ✅ **Security scanning** - Built-in vulnerability detection
- ✅ **Dependency updates** - Automated via Dependabot
- ✅ **Release notes** - Auto-generated from commits

### Quality Assurance
- ✅ **Pre-publish validation** - Package checks before release
- ✅ **Import testing** - Verify package can be imported
- ✅ **Cross-version testing** - Python 3.9-3.12 compatibility
- ✅ **Artifact preservation** - Build files saved for download

## 🔄 Migration from Local Scripts

**Old way (local scripts):**
```bash
./quick-launch.sh patch prod
```

**New way (GitHub Actions):**
1. **Actions** → **Create Release** → **Run workflow**
2. Version: `patch`, Target: Auto-publish to PyPI

**Benefits of migration:**
- 🔒 Secure token storage in GitHub
- 🌐 No local environment dependencies  
- 📋 Automatic testing and validation
- 📊 Audit trail and history
- 👥 Team collaboration ready

---

**Next Steps:**
1. Set up secrets and environments
2. Test with a patch release to TestPyPI
3. Create your first production release
4. Monitor workflows and notifications

For detailed usage examples, see [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md).
