#!/bin/bash
set -e
set -u

# Function to detect if running on Windows
is_windows() {
   [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ -n "${WINDIR:-}" ]]
}

# Function to install uv if not present
install_uv() {
   if ! command -v uv &> /dev/null; then
       echo "uv is not installed. Installing..."
       if is_windows; then
           echo "Windows detected. Installing uv..."
           powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
       else
           curl -fsSL https://astral.sh/uv/install.sh | bash
       fi
   fi
}

# Function to setup uv PATH
setup_uv_path() {
   if is_windows; then
       export PATH="${USERPROFILE:-$HOME}/.local/bin:$PATH"
   else
       export PATH="$HOME/.local/bin:$PATH"
   fi
}

# Main script starts here
echo "Setting up FastAPI project with uv..."

# Clean and setup server
rm -rf .venv

# Install and setup uv
install_uv
setup_uv_path

# Verify uv is accessible
if ! command -v uv &> /dev/null; then
   echo "uv installation failed or not in PATH. Please restart your terminal."
   exit 1
fi

# Create virtual environment and install dependencies
uv venv -p 3.12 .venv
uv sync

# Install the package in development mode
echo "Installing syft-hub in development mode..."
uv pip install -e .

echo "Starting FastAPI server..."
echo "🌐 Starting development server on http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Use app.py instead of main.py to avoid CLI conflicts
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8001