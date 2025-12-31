#!/bin/bash

echo "=========================================="
echo "Paper Digest - Debug Mode"
echo "=========================================="
echo ""
echo "This will fetch 2 papers and generate GPT summaries"
echo "WITHOUT sending an email."
echo ""

# Check if OpenAI key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set. Set it with:"
    echo "   export OPENAI_API_KEY='your-key-here'"
    echo ""
    read -p "Continue without GPT summaries? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run in debug mode
export DEBUG_MODE=1
export NUM_PAPERS=2

python digest_email.py
