# SyftBox Hub CI/CD with GitHub Actions

This repository uses GitHub Actions workflows for automated building, testing, and publishing of the `syft-hub` package using `uv` and `twine`.

## 🚀 Quick Start

### Automated Workflows (Recommended)

**Create a Release:**
1. Go to **Actions** → **Create Release** → **Run workflow**
2. Choose version bump type (patch/minor/major)
3. Optionally publish to PyPI automatically

**Manual Publish:**
1. Go to **Actions** → **Publish Package** → **Run workflow**
2. Choose target (TestPyPI or PyPI)

### Local Development

For local development and testing, you can still build manually:

```bash
# Setup environment
uv venv .venv --python 3.12
uv sync

# Build package locally
uv pip install build
uv run python -m build

# Check package
uv pip install twine
uv run twine check dist/*
```

## 🔄 GitHub Actions Workflows

### 1. **Build and Test** (`build-and-test.yml`)
**Triggers:** Push to main/develop/fixes_launch, PRs, manual trigger

- ✅ Multi-platform testing (Ubuntu, macOS, Windows)
- ✅ Python versions 3.9-3.12 compatibility
- ✅ Package import and functionality tests
- ✅ Security scanning with Safety and Bandit
- ✅ Build artifacts for distribution

### 2. **Publish Package** (`publish.yml`)
**Triggers:** GitHub releases, manual trigger

- 🚀 Automated publishing to PyPI/TestPyPI
- 🔍 Version existence checking
- 📦 Package validation with twine
- 📊 Deployment summaries and links

### 3. **Create Release** (`release.yml`)
**Triggers:** Manual workflow dispatch

- 🏷️ Automatic version bumping (patch/minor/major)
- 📝 Changelog generation from commits
- 🎯 GitHub release creation with assets
- 🚀 Optional automatic PyPI publishing

### 4. **CodeQL Security** (`codeql.yml`)
**Triggers:** Push, PR, weekly schedule

- 🔐 Advanced security analysis
- 🔍 Code quality scanning
- 📋 Security vulnerability detection

## 🛠️ Local Development Commands

For local testing and development, you can use these manual commands:

```bash
# Setup development environment
uv venv .venv --python 3.12
uv sync
uv pip install -e .

# Build package
uv pip install build twine
rm -rf dist/ build/ *.egg-info  # Clean
uv run python -m build          # Build

# Validate package
uv run twine check dist/*

# Test locally
uv run python -c "import syft_hub; print('✅ Import successful')"

# Manual publish (if needed)
uv run twine upload --repository testpypi dist/*  # TestPyPI
uv run twine upload dist/*                        # PyPI
```

**💡 Recommendation:** Use GitHub Actions workflows instead of manual commands for consistent, secure, and automated releases.

## 🔧 Setup and Prerequisites

### GitHub Repository Setup

#### Required Secrets

Go to **Settings** → **Secrets and variables** → **Actions** and add:

| Secret Name | Description | Value |
|-------------|-------------|-------|
| `PYPI_API_TOKEN` | PyPI API token for production publishing | `pypi-...` |
| `TESTPYPI_API_TOKEN` | TestPyPI API token for test publishing | `pypi-...` |

#### Required Environments

Create these environments in **Settings** → **Environments**:

1. **`pypi`** - For production PyPI publishing
   - Add `PYPI_API_TOKEN` as environment secret
   - Optional: Add protection rules (require reviews)

2. **`testpypi`** - For TestPyPI publishing  
   - Add `TESTPYPI_API_TOKEN` as environment secret

### Getting API Tokens

#### PyPI Production Token:
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Scroll to **API tokens** → **Add API token**
3. Scope: **Entire account** or specific to `syft-hub`
4. Copy the token (starts with `pypi-`)

#### TestPyPI Token:
1. Go to [TestPyPI Account Settings](https://test.pypi.org/manage/account/)
2. Create token same way as above
3. Copy the token

### Local Development Setup

#### Required Tools (for manual development)

For local development, you'll need:
- **uv** - Fast Python package manager
- **build** - Python package building (`uv pip install build`)
- **twine** - Package publishing (`uv pip install twine`)

#### Local Authentication Setup (if publishing manually)

#### For TestPyPI:
```bash
export TWINE_USERNAME_TESTPYPI="__token__"
export TWINE_PASSWORD_TESTPYPI="pypi-your-test-token"
```

#### For PyPI:
```bash
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-your-production-token"
```

Or create a `~/.pypirc` file:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-production-token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-token
```

## 🚀 GitHub Actions Usage

### Production Release Workflow

1. **Create a new release:**
   ```
   GitHub Actions → Create Release → Run workflow
   ```
   - Choose version bump: `patch` (bug fixes), `minor` (features), `major` (breaking)
   - Enable "Automatically publish to PyPI" for immediate release
   - Disable "Create as draft" for public release

2. **Result:**
   - ✅ Version bumped in `pyproject.toml`
   - ✅ Git tag created (`v1.2.3`)
   - ✅ GitHub release with changelog
   - ✅ Package published to PyPI
   - ✅ Release assets attached

### Testing Workflow

1. **Test on TestPyPI first:**
   ```
   GitHub Actions → Publish Package → Run workflow
   ```
   - Target: `testpypi`
   - Test installation: `pip install -i https://test.pypi.org/simple/ syft-hub`

2. **Then publish to production:**
   ```
   GitHub Actions → Publish Package → Run workflow
   ```
   - Target: `pypi`

### Development and PR Workflow

1. **Automatic on every push/PR:**
   - Build and test runs automatically
   - Multi-platform compatibility verified
   - Security scans performed
   - No manual intervention needed

2. **Manual testing:**
   ```
   GitHub Actions → Build and Test → Run workflow
   ```

## 📋 Workflow Examples

### Modern Development Workflow (Recommended)

1. **Make code changes**

2. **Push to branch** - Automatic testing runs

3. **Create PR** - Automatic validation 

4. **Merge to main** - Automatic testing

5. **Create release**:
   ```
   GitHub Actions → Create Release → Run workflow
   ```

### Local Development Workflow (If Needed)

1. **Setup environment**:
   ```bash
   uv venv .venv --python 3.12
   uv sync
   uv pip install -e .
   ```

2. **Make changes to code**

3. **Test locally**:
   ```bash
   uv run python -c "import syft_hub; print('✅ Success')"
   ```

4. **Build and check**:
   ```bash
   uv run python -m build
   uv run twine check dist/*
   ```

## What the Scripts Do

### Automated Steps

1. **Environment Setup**:
   - Installs `uv` if not present
   - Creates virtual environment with Python 3.12
   - Syncs dependencies from `pyproject.toml`

2. **Build Process**:
   - Cleans previous build artifacts
   - Runs pre-publish checks
   - Builds wheel and source distributions
   - Validates package with `twine check`

3. **Version Management**:
   - Reads current version from `pyproject.toml`
   - Bumps version according to semantic versioning
   - Updates `pyproject.toml` with new version

4. **Publishing**:
   - Uploads to specified repository (TestPyPI or PyPI)
   - Provides detailed feedback on success/failure

### Pre-publish Checks

- Package import test
- Syntax validation
- Build artifact verification
- Twine package validation

## Troubleshooting

### Common Issues

1. **Authentication Error**:
   - Verify API tokens are correct
   - Check token permissions
   - Ensure tokens are set in environment or `.pypirc`

2. **Build Failures**:
   - Run `./launch.sh clean` to clear artifacts
   - Check `pyproject.toml` syntax
   - Verify dependencies are available

3. **Version Conflicts**:
   - Check if version already exists on PyPI
   - Use `./launch.sh bump` to increment version

4. **uv Installation Issues**:
   - Restart terminal after installation
   - Check PATH includes `~/.local/bin`
   - Manually install: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Manual Override

If you need more control, you can run individual steps:

```bash
# Manual build process
uv venv -p 3.12 .venv
uv sync
uv pip install build twine
uv run python -m build
twine check dist/*
twine upload --repository testpypi dist/*
```

## Features

### Safety Features

- ✅ Automatic cleanup of old builds
- ✅ Pre-publish validation checks
- ✅ Version conflict detection
- ✅ Colored output for easy reading
- ✅ Error handling with clear messages

### Convenience Features

- ✅ Auto-installation of required tools
- ✅ Cross-platform compatibility (Linux, macOS, Windows)
- ✅ Multiple repository support
- ✅ Semantic version bumping
- ✅ Development environment setup

## Package Information

- **Package Name**: `syft-hub`
- **Current Version**: `0.1.7`
- **Python Version**: `>=3.9` (builds with 3.12)
- **Repository**: SyftBox Hub NSAI SDK

For more information about the package itself, see the main [README.md](README.md).
