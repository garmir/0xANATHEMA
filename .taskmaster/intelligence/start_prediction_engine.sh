#!/bin/bash

# Intelligent Task Prediction Engine Startup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/prediction_config.json"
PYTHON_SCRIPT="$SCRIPT_DIR/task_prediction_engine.py"

echo "🧠 Intelligent Task Prediction and Auto-Generation System"
echo "======================================================="
echo "Starting AI-powered task prediction engine..."
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
    'numpy', 'pandas', 'sklearn', 'networkx', 
    'flask', 'sqlite3'
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

# Check task-master data
TASKS_FILE="$(dirname "$SCRIPT_DIR")/tasks/tasks.json"
if [ ! -f "$TASKS_FILE" ]; then
    echo "⚠️  Task-master data not found: $TASKS_FILE"
    echo "   The system will still work but with limited historical data"
else
    echo "✅ Task-master historical data found"
fi

# Create necessary directories
mkdir -p "$(dirname "$CONFIG_FILE")/logs"
mkdir -p "$(dirname "$CONFIG_FILE")/models"
mkdir -p "$(dirname "$CONFIG_FILE")/data"

# Set permissions
chmod +x "$PYTHON_SCRIPT"

echo ""
echo "🚀 Starting prediction engine..."

# Parse command line arguments
MODE="server"
USER_ID="default"

while [[ $# -gt 0 ]]; do
    case $1 in
        --generate)
            MODE="generate"
            shift
            ;;
        --user-id)
            USER_ID="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--generate] [--user-id USER_ID] [--config CONFIG_FILE]"
            exit 1
            ;;
    esac
done

echo "   Mode: $MODE"
echo "   User ID: $USER_ID"
echo "   Config: $CONFIG_FILE"
echo "   API Port: $(jq -r '.api_port // 8081' "$CONFIG_FILE")"
echo "   Learning: $(jq -r '.learning_enabled // true' "$CONFIG_FILE")"
echo "   Auto-Generation: $(jq -r '.auto_generate_enabled // true' "$CONFIG_FILE")"
echo ""

# Start the prediction engine
cd "$SCRIPT_DIR"

if [ "$MODE" = "generate" ]; then
    echo "🔮 Generating task predictions..."
    python3 "$PYTHON_SCRIPT" --config "$CONFIG_FILE" --generate --user-id "$USER_ID"
else
    echo "🌐 Starting prediction API server..."
    echo "   Access at: http://localhost:$(jq -r '.api_port // 8081' "$CONFIG_FILE")/api/predictions"
    echo ""
    python3 "$PYTHON_SCRIPT" --config "$CONFIG_FILE"
fi