#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================================"
echo " Setting up : Open-Domain Question Answering"
echo "============================================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "[-] 'uv' is not installed. Installing astral-sh/uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Ensure uv is available in the current session path
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
else
    echo "[+] 'uv' is already installed."
fi

# Sync project dependencies
echo "[-] Syncing project dependencies from pyproject.toml..."
uv sync

# Download the classical extraction NLP model
echo "[-] Downloading spaCy English model (en_core_web_sm)..."
uv run python -m spacy download en_core_web_sm

echo "============================================================"
echo " Setup Complete!"
echo ""
echo " Next Steps:"
echo " 1. Authenticate with Hugging Face to avoid download limits:"
echo "    uv run hf auth login"
echo " 2. Run the pipeline on the default dataset:"
echo "    uv run task"
echo " or Run the other two datasets as: "
echo "    uv run task --dataset nq_open --sample_size 300"
echo "    uv run task --dataset trivia_qa --sample_size 300"
echo "============================================================"
