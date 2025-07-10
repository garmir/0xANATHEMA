#!/bin/bash

# Task Master Environment Setup Script
# Sets up required environment variables for autonomous execution

export TASKMASTER_HOME="/Users/anam/archive/.taskmaster"
export TASKMASTER_DOCS="/Users/anam/archive/.taskmaster/docs"
export TASKMASTER_LOGS="/Users/anam/archive/.taskmaster/logs"

# Create directories if they don't exist
mkdir -p "$TASKMASTER_HOME"
mkdir -p "$TASKMASTER_DOCS"
mkdir -p "$TASKMASTER_LOGS"

echo "Environment variables set:"
echo "TASKMASTER_HOME=$TASKMASTER_HOME"
echo "TASKMASTER_DOCS=$TASKMASTER_DOCS"
echo "TASKMASTER_LOGS=$TASKMASTER_LOGS"

# Verify directories are accessible
if [[ -w "$TASKMASTER_HOME" && -w "$TASKMASTER_DOCS" && -w "$TASKMASTER_LOGS" ]]; then
    echo "✅ All directories are writable"
else
    echo "❌ Some directories are not writable"
    exit 1
fi

# Add to shell profile for persistence
SHELL_PROFILE=""
if [[ -f "$HOME/.zshrc" ]]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [[ -f "$HOME/.bash_profile" ]]; then
    SHELL_PROFILE="$HOME/.bash_profile"
elif [[ -f "$HOME/.bashrc" ]]; then
    SHELL_PROFILE="$HOME/.bashrc"
fi

if [[ -n "$SHELL_PROFILE" ]]; then
    echo "Adding environment variables to $SHELL_PROFILE"
    cat >> "$SHELL_PROFILE" << EOF

# Task Master Environment Variables
export TASKMASTER_HOME="/Users/anam/archive/.taskmaster"
export TASKMASTER_DOCS="/Users/anam/archive/.taskmaster/docs"
export TASKMASTER_LOGS="/Users/anam/archive/.taskmaster/logs"
EOF
    echo "✅ Environment variables added to shell profile"
else
    echo "⚠️  Could not find shell profile to persist environment variables"
fi

echo "Environment setup complete!"