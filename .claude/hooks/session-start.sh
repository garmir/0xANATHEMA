#!/bin/bash
# Claude Session Start Hook
# Automatically runs task-master research and executes steps as prompts

echo "🚀 Initiating automated research-driven development session..."

# Run automated research prompt generator
python3 .taskmaster/scripts/auto-research-prompt-generator.py

# Also try task-master research if available
echo ""
echo "📊 Additional research analysis:"
RESEARCH_OUTPUT=$(task-master research 2>/dev/null || echo "Task-master research command not available")
echo "$RESEARCH_OUTPUT"

echo ""
echo "✅ Automated research-driven session ready"
echo "🎯 Execute research findings as prompts for autonomous development"