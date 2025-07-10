#!/bin/bash
# Claude Code Auto-Recovery Integration
# Wraps any command with autonomous research-driven error recovery

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTONOMOUS_LOOP="$SCRIPT_DIR/autonomous-research-loop.sh"

# Source the autonomous loop functions
source "$AUTONOMOUS_LOOP"

# Enhanced Claude Code integration
claude_with_auto_recovery() {
    local claude_prompt="$*"
    
    log "🤖 Executing Claude Code with auto-recovery: $claude_prompt"
    
    # Create a temporary script to execute the Claude prompt
    local temp_script=$(mktemp)
    cat > "$temp_script" << EOF
#!/bin/bash
# Auto-generated Claude execution script
echo "Executing: $claude_prompt"
# This would integrate with Claude Code API or CLI
# For now, simulate execution
EOF
    
    chmod +x "$temp_script"
    
    # Monitor execution with auto-recovery
    monitor_and_auto_recover "$temp_script"
    
    rm -f "$temp_script"
}

# Task-master integration with auto-recovery
taskmaster_with_auto_recovery() {
    local taskmaster_cmd="$*"
    
    log "🎯 Executing task-master with auto-recovery: $taskmaster_cmd"
    
    # Execute task-master command with monitoring
    monitor_and_auto_recovery "task-master $taskmaster_cmd"
}

# Automatic workflow integration
auto_workflow() {
    log "🔄 Starting automatic workflow with full error recovery"
    
    # Step 1: Get next task
    log "📋 Getting next available task..."
    if ! taskmaster_with_auto_recovery "next"; then
        log "❌ Failed to get next task after auto-recovery"
        return 1
    fi
    
    # Step 2: Execute task with monitoring
    local task_id=$(task-master next --show-id 2>/dev/null || echo "")
    
    if [[ -n "$task_id" ]]; then
        log "🎯 Auto-executing task: $task_id"
        
        # Get task details
        local task_details=$(task-master show "$task_id" 2>/dev/null || echo "")
        
        # Execute task with Claude Code integration
        claude_with_auto_recovery "Execute task $task_id: $task_details"
        
        # Mark task complete if successful
        if [[ $? -eq 0 ]]; then
            taskmaster_with_auto_recovery "set-status --id=$task_id --status=done"
        fi
    fi
    
    # Step 3: Continue loop
    log "🔄 Checking for more tasks..."
    if task-master next 2>&1 | grep -qi "no eligible tasks"; then
        log "✅ All tasks completed successfully"
        return 0
    else
        log "🔄 More tasks available, continuing workflow..."
        auto_workflow  # Recursive call
    fi
}

# Error-prone command wrapper
safe_execute() {
    local cmd="$*"
    
    log "🛡️ Safe execution with auto-recovery: $cmd"
    
    # Try command directly first
    if eval "$cmd" 2>/dev/null; then
        log "✅ Command succeeded on first try"
        return 0
    fi
    
    # If failed, trigger auto-recovery
    log "🚨 Command failed, triggering auto-recovery..."
    monitor_and_auto_recovery "$cmd"
}

# Development workflow with built-in recovery
dev_workflow() {
    local project_path="${1:-.}"
    
    log "🚀 Starting development workflow with auto-recovery"
    log "📁 Project path: $project_path"
    
    cd "$project_path"
    
    # Step 1: Initialize or verify task-master setup
    safe_execute "task-master list"
    
    # Step 2: Start automatic task execution
    auto_workflow
    
    # Step 3: Run any final validation
    log "🔍 Running final validation..."
    safe_execute "task-master validate-dependencies"
    
    log "🎉 Development workflow completed successfully"
}

# CLI Interface
case "${1:-help}" in
    "claude")
        shift
        claude_with_auto_recovery "$@"
        ;;
    "taskmaster"|"tm")
        shift  
        taskmaster_with_auto_recovery "$@"
        ;;
    "auto"|"workflow")
        auto_workflow
        ;;
    "safe")
        shift
        safe_execute "$@"
        ;;
    "dev")
        shift
        dev_workflow "$@"
        ;;
    "help"|*)
        echo "Claude Code Auto-Recovery Integration"
        echo ""
        echo "Usage:"
        echo "  $0 claude <prompt>           # Execute Claude prompt with auto-recovery"
        echo "  $0 taskmaster <cmd>          # Execute task-master command with auto-recovery"  
        echo "  $0 auto                      # Run automatic workflow until completion"
        echo "  $0 safe <command>            # Execute any command with auto-recovery"
        echo "  $0 dev [path]                # Full development workflow with auto-recovery"
        echo ""
        echo "Examples:"
        echo "  $0 claude 'implement user auth'       # Auto-recover from implementation issues"
        echo "  $0 taskmaster next                     # Auto-recover from task-master errors"
        echo "  $0 auto                                # Auto-execute all tasks with recovery"
        echo "  $0 safe 'npm test'                     # Auto-fix test failures"
        echo "  $0 dev .                               # Full project workflow with recovery"
        echo ""
        echo "Features:"
        echo "  ✅ Automatic error detection"
        echo "  🔍 Perplexity research integration"  
        echo "  📋 Solution todo generation"
        echo "  🔄 Recursive execution until success"
        echo "  📊 Full logging and monitoring"
        ;;
esac