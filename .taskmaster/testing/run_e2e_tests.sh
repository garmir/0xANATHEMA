#!/bin/bash

# End-to-End Testing Framework Runner
# Runs comprehensive autonomous execution tests for Task Master

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_WORKSPACE="${TEST_WORKSPACE:-/tmp/taskmaster_e2e_tests}"

echo "🧪 Task Master End-to-End Testing Framework"
echo "=========================================="
echo "Test workspace: $TEST_WORKSPACE"
echo "Timestamp: $(date)"
echo ""

# Ensure Python dependencies
echo "📦 Checking Python environment..."
python3 -c "import json, subprocess, logging, uuid, tempfile, shutil" || {
    echo "❌ Missing Python dependencies. Please install required packages."
    exit 1
}

# Ensure task-master is available
echo "🔧 Checking task-master CLI..."
which task-master >/dev/null || {
    echo "❌ task-master CLI not found. Please install task-master-ai."
    exit 1
}

# Run end-to-end tests
echo "🚀 Starting end-to-end test suite..."
echo ""

cd "$SCRIPT_DIR"

if [ "$1" = "--scenario" ]; then
    echo "Running specific scenario: $2"
    python3 end_to_end_framework.py --scenario "$2" --workspace "$TEST_WORKSPACE"
elif [ "$1" = "--report-only" ]; then
    echo "Generating report from existing results..."
    python3 end_to_end_framework.py --report-only --workspace "$TEST_WORKSPACE"
else
    echo "Running full test suite..."
    python3 end_to_end_framework.py --workspace "$TEST_WORKSPACE"
fi

echo ""
echo "✅ End-to-end testing completed!"
echo "📊 Check reports directory for detailed results: $SCRIPT_DIR/reports/"