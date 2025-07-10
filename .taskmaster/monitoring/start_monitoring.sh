#!/bin/bash

# Advanced System Optimization and Monitoring Suite Startup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/suite_config.json"
PYTHON_SCRIPT="$SCRIPT_DIR/ai_optimization_suite.py"

echo "🔧 Advanced System Optimization and Monitoring Suite"
echo "=================================================="
echo "Starting AI-powered performance monitoring..."
echo "Timestamp: $(date)"
echo ""

# Check dependencies
echo "📦 Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

# Check required Python packages
python3 -c "
import sys
required_packages = [
    'psutil', 'numpy', 'pandas', 'sklearn', 
    'flask', 'plotly', 'sqlite3'
]
missing = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing.append(package)

if missing:
    print(f'❌ Missing Python packages: {missing}')
    print('Install with: pip install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('✅ All Python dependencies found')
" || exit 1

# Check configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "✅ Configuration found: $CONFIG_FILE"

# Create necessary directories
mkdir -p "$(dirname "$CONFIG_FILE")/../logs"
mkdir -p "$(dirname "$CONFIG_FILE")/../data"

# Set permissions
chmod +x "$PYTHON_SCRIPT"

echo ""
echo "🚀 Starting monitoring suite..."

# Parse command line arguments
MODE="full"
if [ "$1" = "--api-only" ]; then
    MODE="api"
    echo "   Mode: API Server Only"
elif [ "$1" = "--monitor-only" ]; then
    MODE="monitor"
    echo "   Mode: Monitoring Only"
else
    echo "   Mode: Full Suite (Monitoring + API)"
fi

echo "   Config: $CONFIG_FILE"
echo "   Log Level: $(jq -r '.log_level // "INFO"' "$CONFIG_FILE")"
echo "   API Port: $(jq -r '.api_port // 8080' "$CONFIG_FILE")"
echo "   Self-Healing: $(jq -r '.self_healing_enabled // true' "$CONFIG_FILE")"
echo ""

# Start the suite
cd "$SCRIPT_DIR"

if [ "$MODE" = "api" ]; then
    echo "🌐 Starting API server..."
    python3 "$PYTHON_SCRIPT" --config "$CONFIG_FILE" --api-only
elif [ "$MODE" = "monitor" ]; then
    echo "📊 Starting monitoring only..."
    python3 "$PYTHON_SCRIPT" --config "$CONFIG_FILE" --monitor-only
else
    echo "🔄 Starting full monitoring suite..."
    python3 "$PYTHON_SCRIPT" --config "$CONFIG_FILE"
fi