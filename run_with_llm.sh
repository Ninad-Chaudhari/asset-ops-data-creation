#!/bin/bash
# Wrapper script to run ComplexBench generation with OpenAI API

# Set the OpenAI API key (replace with your key or set as environment variable)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

echo "🚀 Starting ComplexBench dataset generation with LLM calls..."
echo "📊 Processing 101 IoT tasks..."
echo "⏱️  This will take approximately 30-60 minutes..."
echo ""

# Run the main script
python3 extract_complex_bench.py

echo ""
echo "✅ Generation complete!"
echo "📄 Output: complexbench_iot_records.json"

