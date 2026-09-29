#!/bin/bash
# ALTRIX Startup Script
# "Understand the Impact. Not the Identity."

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Check if venv exists
if [ -d ".venv" ]; then
    PYTHON_CMD="./.venv/bin/python3"
    STREAMLIT_CMD="./.venv/bin/streamlit"
else
    PYTHON_CMD="python3"
    STREAMLIT_CMD="streamlit"
fi

echo "=========================================================="
echo "🛡️  ALTRIX — Privacy-Preserving Incremental Ad Impact System"
echo "    “Understand the Impact. Not the Identity.”"
echo "=========================================================="
echo "Starting Streamlit application..."
"$STREAMLIT_CMD" run app.py --server.port 8501 --server.headless true
