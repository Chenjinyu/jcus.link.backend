#!/bin/bash

# Setup script for Document Analysis MCP Server
# This script installs all dependencies and sets up the environment

set -e  # Exit on error

echo "=========================================="
echo "Document Analysis MCP Server Setup"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed"
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed successfully"
    echo "⚠️  Please restart your terminal and run this script again"
    exit 0
fi

echo "✅ uv is already installed"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

required_version="3.10"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.10+ is required"
    echo "Current version: $python_version"
    exit 1
fi

echo "✅ Python version is compatible"
echo ""

# Install dependencies
echo "Installing project dependencies with uv..."
uv sync
echo "✅ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "⚠️  Please edit .env and configure your settings"
else
    echo "⚠️  .env file already exists, skipping..."
fi
echo ""

# Check if Ollama is installed (optional)
echo "Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
    
    # Check if nomic-embed-text model is available
    if ollama list | grep -q "nomic-embed-text"; then
        echo "✅ nomic-embed-text model is available"
    else
        echo "⚠️  nomic-embed-text model not found"
        echo "Installing nomic-embed-text model..."
        ollama pull nomic-embed-text
        echo "✅ Model installed"
    fi
else
    echo "⚠️  Ollama is not installed"
    echo ""
    echo "Ollama is recommended for local embeddings."
    echo "To install Ollama:"
    echo "  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Then run: ollama pull nomic-embed-text"
    echo ""
    echo "Alternatively, you can use OpenAI embeddings by:"
    echo "1. Setting OPENAI_API_KEY in .env"
    echo "2. Using embedding_provider='openai' in tool calls"
fi
echo ""

# Create chroma_db directory
echo "Creating vector database directory..."
mkdir -p chroma_db
echo "✅ Database directory created"
echo ""

# Test the server
echo "Testing MCP server..."
if uv run python document_analysis_mcp.py --help &> /dev/null; then
    echo "✅ Server test passed"
else
    echo "❌ Server test failed"
    echo "Please check for errors above"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and configure your settings"
echo "2. Run the server: uv run python document_analysis_mcp.py"
echo "3. Or run examples: uv run python examples.py"
echo ""
echo "For more information, see README.md"