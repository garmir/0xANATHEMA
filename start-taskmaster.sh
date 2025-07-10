#!/bin/bash
# Task Master AI - Startup Script with API Key Configuration
# Automatically loads environment variables and starts Task Master

set -e

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "🔑 Loading API keys from .env file..."
    source .env
    export OPENAI_API_KEY PERPLEXITY_API_KEY
    echo "✅ Environment variables loaded"
else
    echo "❌ .env file not found. Please create one with your API keys."
    exit 1
fi

# Verify API keys are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not found in .env file"
    exit 1
fi

if [ -z "$PERPLEXITY_API_KEY" ]; then
    echo "⚠️  PERPLEXITY_API_KEY not found - research features will be limited"
fi

echo ""
echo "🚀 Task Master AI - Autonomous Development System"
echo "=================================================="
echo ""
echo "🔧 Configuration:"
echo "  • Primary AI: OpenAI GPT-4o"
echo "  • Research: Perplexity sonar-pro"
echo "  • Status: Ready for autonomous execution"
echo ""

# Show current project status
echo "📊 Current Project Status:"
task-master list

echo ""
echo "🎯 Ready for autonomous execution!"
echo ""
echo "Available commands:"
echo "  task-master next          # Get next available task"
echo "  task-master show <id>     # View task details"
echo "  task-master expand <id>   # Break task into subtasks"
echo "  task-master set-status    # Update task status"
echo ""

# If argument provided, execute it
if [ $# -gt 0 ]; then
    echo "🚀 Executing: task-master $@"
    echo ""
    task-master "$@"
else
    echo "💡 Try: ./start-taskmaster.sh next"
fi