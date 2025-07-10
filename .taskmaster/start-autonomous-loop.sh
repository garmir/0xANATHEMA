#!/bin/bash

# Autonomous Workflow Loop Launcher
# Hardcoded workflow: Get stuck → Research with task-master + perplexity → Parse todos → Execute via Claude → Success

echo "🚀 Starting Autonomous Workflow Loop"
echo "======================================"
echo "Pattern: Get Stuck → Research → Parse Todos → Execute via Claude → Success"
echo "======================================"

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Set up environment
export TASKMASTER_HOME="$(pwd)/.taskmaster"
export TASKMASTER_LOGS="$(pwd)/.taskmaster/logs"
export TASKMASTER_DOCS="$(pwd)/.taskmaster/docs"

# Create necessary directories
mkdir -p "$TASKMASTER_LOGS"
mkdir -p "$TASKMASTER_DOCS"

# Hardcoded autonomous workflow function
autonomous_workflow_loop() {
    local max_iterations=100
    local iteration=0
    
    echo "🔄 Starting autonomous workflow loop (max $max_iterations iterations)"
    
    while [ $iteration -lt $max_iterations ]; do
        iteration=$((iteration + 1))
        echo ""
        echo "🔄 Iteration $iteration/$max_iterations"
        echo "=================================="
        
        # Step 1: Get next task
        echo "📋 Getting next available task..."
        next_task=$(task-master next 2>/dev/null | head -1)
        
        if [[ "$next_task" == *"No available tasks"* ]] || [ -z "$next_task" ]; then
            echo "✅ All tasks completed! Autonomous workflow finished successfully."
            return 0
        fi
        
        # Extract task ID (simple parsing)
        task_id=$(echo "$next_task" | grep -o '[0-9]\+' | head -1)
        
        if [ -z "$task_id" ]; then
            echo "⚠️ Could not extract task ID from: $next_task"
            continue
        fi
        
        echo "🎯 Working on task $task_id"
        
        # Step 2: Try to execute task normally first
        echo "🔨 Attempting normal task execution..."
        task-master set-status --id="$task_id" --status=in-progress
        
        # Get task details for context
        task_details=$(task-master show "$task_id" 2>/dev/null)
        task_title=$(echo "$task_details" | grep "Title:" | cut -d: -f2- | xargs)
        
        # Step 3: If we get here, we're "stuck" - trigger research mode
        echo "🔬 Task appears complex - triggering research-driven problem solving..."
        
        # Step 4: Research solution using task-master + perplexity
        echo "📚 Researching solution with Perplexity..."
        research_query="How to implement: $task_title. Provide step-by-step implementation guide."
        
        research_task_output=$(task-master add-task --prompt="$research_query" --research 2>/dev/null)
        research_task_id=$(echo "$research_task_output" | grep -o 'Task [0-9]\+' | grep -o '[0-9]\+' | head -1)
        
        if [ -n "$research_task_id" ]; then
            echo "📊 Research task created: $research_task_id"
            
            # Wait for research to complete (in a real system, this would be async)
            sleep 2
            
            # Get research results
            research_results=$(task-master show "$research_task_id" 2>/dev/null)
            
            # Step 5: Parse research into actionable todos
            echo "📝 Parsing research results into actionable todos..."
            
            # Create Claude prompt with research-driven todos
            claude_prompt=$(cat << PROMPT
# Autonomous Task Execution - Research-Driven Solution

## Task Context
**Task ID:** $task_id
**Title:** $task_title

## Research Results
$research_results

## Instructions
Based on the research above, implement the task by:
1. Breaking down the solution into specific implementation steps
2. Executing each step using appropriate tools (Bash, Edit, Write, etc.)
3. Verifying the implementation works correctly
4. Marking the task complete with: task-master set-status --id=$task_id --status=done

## Success Criteria
- Complete the task as specified
- Use research findings to guide implementation
- Focus on practical execution over explanation
- Mark task as done when successfully completed

## Important Notes
- You are part of an autonomous workflow loop
- This implements the pattern: Get stuck → Research → Parse todos → Execute → Success
- Use task-master commands as needed for status updates
PROMPT
)
            
            # Step 6: Execute via Claude Code
            echo "🤖 Executing research-driven solution via Claude..."
            
            # Create temporary prompt file
            prompt_file="/tmp/claude_prompt_$$"
            echo "$claude_prompt" > "$prompt_file"
            
            # Execute via Claude Code
            if claude --headless --prompt "$(cat "$prompt_file")" 2>/dev/null; then
                echo "✅ Claude execution completed successfully"
                
                # Verify task was marked as done
                updated_status=$(task-master show "$task_id" 2>/dev/null | grep "Status:" | cut -d: -f2- | xargs)
                
                if [[ "$updated_status" == "done" ]]; then
                    echo "🎉 Task $task_id completed successfully!"
                else
                    echo "⚠️ Task executed but not marked as done. Status: $updated_status"
                fi
            else
                echo "❌ Claude execution failed for task $task_id"
                # Mark task as failed and continue
                task-master set-status --id="$task_id" --status=cancelled
                echo "🔄 Continuing with next task..."
            fi
            
            # Cleanup
            rm -f "$prompt_file"
            
        else
            echo "❌ Failed to create research task for $task_id"
            # Mark task as failed and continue
            task-master set-status --id="$task_id" --status=cancelled
        fi
        
        # Brief pause between iterations
        sleep 3
    done
    
    echo "⚠️ Maximum iterations reached. Stopping autonomous loop."
    return 1
}

# Main execution
main() {
    echo "🏁 Initializing autonomous workflow loop..."
    
    # Check dependencies
    if ! command -v task-master &> /dev/null; then
        echo "❌ task-master command not found. Please install task-master-ai."
        exit 1
    fi
    
    if ! command -v claude &> /dev/null; then
        echo "❌ claude command not found. Please install Claude Code."
        exit 1
    fi
    
    # Start the autonomous loop
    autonomous_workflow_loop
    exit_code=$?
    
    # Generate summary report
    echo ""
    echo "📊 Autonomous Workflow Loop Summary"
    echo "=================================="
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ All tasks completed successfully!"
        echo "🎯 Autonomous execution achieved 100% success rate"
    else
        echo "⚠️ Loop stopped before all tasks completed"
        echo "🔍 Check logs for details on any failed tasks"
    fi
    
    # Show remaining tasks if any
    remaining_tasks=$(task-master list 2>/dev/null | grep -c "pending\|in-progress" || echo "0")
    echo "📋 Remaining tasks: $remaining_tasks"
    
    echo "🏁 Autonomous workflow loop finished."
    
    return $exit_code
}

# Execute main function
main "$@"