#!/bin/bash

# LABRYS Unified Execution Script
# Replaces multiple similar scripts with configurable execution modes

REPO_DIR="/Users/anam/temp/0xANATHEMA"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
SESSION_ID="LABRYS-Unified-$(date +%H%M%S)"

# Configuration
CONFIG_FILE="${SCRIPT_DIR}/labrys_config.json"
LOG_DIR=".labrys/logs"
TEMP_DIR=".labrys/temp"

# Default configuration
DEFAULT_CONFIG='{
  "execution_mode": "sequential",
  "terminal_cleanup": true,
  "timeout": 60,
  "max_retries": 3,
  "terminal_app": "Ghostty",
  "terminals": [
    {
      "name": "analytical",
      "description": "ANALYTICAL BLADE OPERATIONS",
      "command": "mkdir -p .labrys/analytical && echo ANALYTICAL_BLADE_INITIALIZED > .labrys/analytical/status.txt && echo OUTPUT_CAPTURED > .labrys/temp/analytical_output.txt"
    },
    {
      "name": "synthesis", 
      "description": "SYNTHESIS BLADE OPERATIONS",
      "command": "mkdir -p .labrys/synthesis && echo SYNTHESIS_BLADE_INITIALIZED > .labrys/synthesis/status.txt && echo OUTPUT_CAPTURED > .labrys/temp/synthesis_output.txt"
    },
    {
      "name": "coordination",
      "description": "COORDINATION SYSTEM",
      "command": "mkdir -p .labrys/coordination && echo COORDINATION_SYSTEM_INITIALIZED > .labrys/coordination/status.txt && echo OUTPUT_CAPTURED > .labrys/temp/coordination_output.txt"
    },
    {
      "name": "validation",
      "description": "VALIDATION FRAMEWORK",
      "command": "mkdir -p .labrys/validation && echo VALIDATION_SYSTEM_INITIALIZED > .labrys/validation/status.txt && echo OUTPUT_CAPTURED > .labrys/temp/validation_output.txt"
    }
  ]
}'

# Load configuration
load_config() {
    if [ -f "$CONFIG_FILE" ]; then
        CONFIG=$(cat "$CONFIG_FILE")
    else
        CONFIG="$DEFAULT_CONFIG"
        echo "$DEFAULT_CONFIG" > "$CONFIG_FILE"
    fi
}

# Parse configuration
parse_config() {
    EXECUTION_MODE=$(echo "$CONFIG" | jq -r '.execution_mode')
    TERMINAL_CLEANUP=$(echo "$CONFIG" | jq -r '.terminal_cleanup')
    TIMEOUT=$(echo "$CONFIG" | jq -r '.timeout')
    MAX_RETRIES=$(echo "$CONFIG" | jq -r '.max_retries')
    TERMINAL_APP=$(echo "$CONFIG" | jq -r '.terminal_app')
}

# Setup directories
setup_directories() {
    mkdir -p "$LOG_DIR"
    mkdir -p "$TEMP_DIR"
    echo "[$TIMESTAMP] LABRYS Unified Executor started: $SESSION_ID" > "$LOG_DIR/session.log"
}

# Execute terminal with configuration
execute_terminal() {
    local terminal_name="$1"
    local terminal_desc="$2"
    local terminal_cmd="$3"
    local output_file="$TEMP_DIR/${terminal_name}_output.txt"
    
    echo "=== EXECUTING TERMINAL: $terminal_name ($terminal_desc) ==="
    echo "[$TIMESTAMP] Starting $terminal_name" >> "$LOG_DIR/session.log"
    
    # Execute based on mode
    if [ "$EXECUTION_MODE" = "parallel" ]; then
        execute_terminal_parallel "$terminal_name" "$terminal_desc" "$terminal_cmd" &
    else
        execute_terminal_sequential "$terminal_name" "$terminal_desc" "$terminal_cmd"
    fi
}

# Sequential execution
execute_terminal_sequential() {
    local terminal_name="$1"
    local terminal_desc="$2"
    local terminal_cmd="$3"
    local output_file="$TEMP_DIR/${terminal_name}_output.txt"
    
    # Try to detect available terminal applications
    if command -v "$TERMINAL_APP" >/dev/null 2>&1; then
        AVAILABLE_TERMINAL="$TERMINAL_APP"
    elif command -v Terminal >/dev/null 2>&1; then
        AVAILABLE_TERMINAL="Terminal"
    elif command -v iTerm >/dev/null 2>&1; then
        AVAILABLE_TERMINAL="iTerm"
    else
        echo "Warning: No supported terminal application found. Using direct execution."
        # Execute directly instead of through AppleScript
        cd "$REPO_DIR"
        eval "$terminal_cmd"
        return $?
    fi
    
    osascript -e "
        tell application \"$AVAILABLE_TERMINAL\"
            activate
            tell application \"System Events\"
                keystroke \"n\" using command down
                delay 2
                keystroke \"cd '$REPO_DIR' && claude --dangerously-skip-permissions\"
                keystroke return
                delay 6
                keystroke \"$terminal_cmd\"
                keystroke return
                delay 3
                keystroke return
                delay 3
                keystroke return
                $([ "$TERMINAL_CLEANUP" = "true" ] && echo "keystroke \"exit\"; keystroke return")
            end tell
        end tell
    " &
    
    # Wait for completion
    wait_for_completion "$terminal_name" "$output_file"
}

# Parallel execution
execute_terminal_parallel() {
    local terminal_name="$1"
    local terminal_desc="$2"
    local terminal_cmd="$3"
    
    execute_terminal_sequential "$terminal_name" "$terminal_desc" "$terminal_cmd"
}

# Wait for completion with timeout
wait_for_completion() {
    local terminal_name="$1"
    local output_file="$2"
    local elapsed=0
    
    echo "Waiting for $terminal_name completion..."
    
    while [ $elapsed -lt $TIMEOUT ] && [ ! -f "$output_file" ]; do
        sleep 3
        elapsed=$((elapsed + 3))
        echo "Elapsed: ${elapsed}s"
    done
    
    if [ -f "$output_file" ]; then
        echo "✓ $terminal_name completed successfully"
        echo "[$TIMESTAMP] $terminal_name completed" >> "$LOG_DIR/session.log"
        return 0
    else
        echo "✗ $terminal_name failed or timed out"
        echo "[$TIMESTAMP] $terminal_name failed" >> "$LOG_DIR/session.log"
        return 1
    fi
}

# Execute all terminals
execute_all_terminals() {
    local terminal_count=$(echo "$CONFIG" | jq '.terminals | length')
    local success_count=0
    local pids=()
    
    for i in $(seq 0 $((terminal_count - 1))); do
        local terminal_name=$(echo "$CONFIG" | jq -r ".terminals[$i].name")
        local terminal_desc=$(echo "$CONFIG" | jq -r ".terminals[$i].description")
        local terminal_cmd=$(echo "$CONFIG" | jq -r ".terminals[$i].command")
        
        if [ "$EXECUTION_MODE" = "parallel" ]; then
            execute_terminal "$terminal_name" "$terminal_desc" "$terminal_cmd" &
            pids+=($!)
        else
            if execute_terminal "$terminal_name" "$terminal_desc" "$terminal_cmd"; then
                success_count=$((success_count + 1))
            else
                echo "Stopping execution due to failure in $terminal_name"
                break
            fi
        fi
    done
    
    # Wait for parallel execution
    if [ "$EXECUTION_MODE" = "parallel" ]; then
        for pid in "${pids[@]}"; do
            wait "$pid" && success_count=$((success_count + 1))
        done
    fi
    
    echo "=== EXECUTION SUMMARY ==="
    echo "Total terminals: $terminal_count"
    echo "Successful: $success_count"
    echo "Failed: $((terminal_count - success_count))"
    
    # Generate final report
    generate_final_report "$terminal_count" "$success_count"
}

# Generate final report
generate_final_report() {
    local total="$1"
    local success="$2"
    local report_file="$TEMP_DIR/final_status.txt"
    
    echo "=== LABRYS UNIFIED EXECUTION REPORT ===" > "$report_file"
    echo "Session ID: $SESSION_ID" >> "$report_file"
    echo "Execution Mode: $EXECUTION_MODE" >> "$report_file"
    echo "Timestamp: $(date)" >> "$report_file"
    echo "Total Terminals: $total" >> "$report_file"
    echo "Successful: $success" >> "$report_file"
    echo "Failed: $((total - success))" >> "$report_file"
    echo "" >> "$report_file"
    
    # Individual terminal status
    for terminal in analytical synthesis coordination validation; do
        if [ -f "$TEMP_DIR/${terminal}_output.txt" ]; then
            echo "$terminal: COMPLETED" >> "$report_file"
        else
            echo "$terminal: FAILED" >> "$report_file"
        fi
    done
    
    echo "Final report generated: $report_file"
}

# Cleanup function
cleanup() {
    if [ "$TERMINAL_CLEANUP" = "true" ]; then
        echo "Performing terminal cleanup..."
        osascript -e "
            tell application \"$TERMINAL_APP\"
                tell application \"System Events\"
                    tell process \"$TERMINAL_APP\"
                        repeat with w in windows
                            try
                                tell w
                                    keystroke \"w\" using command down
                                end tell
                            end try
                        end repeat
                    end tell
                end tell
            end tell
        " 2>/dev/null
    fi
}

# Main execution
main() {
    echo "=== LABRYS UNIFIED EXECUTOR ==="
    echo "Session: $SESSION_ID"
    echo "Mode: ${1:-sequential}"
    
    # Set execution mode from argument
    if [ "$1" = "parallel" ]; then
        EXECUTION_MODE="parallel"
    elif [ "$1" = "sequential" ]; then
        EXECUTION_MODE="sequential"
    fi
    
    load_config
    parse_config
    setup_directories
    
    # Override execution mode if provided
    if [ -n "$1" ]; then
        EXECUTION_MODE="$1"
    fi
    
    echo "Execution mode: $EXECUTION_MODE"
    
    # Execute terminals
    execute_all_terminals
    
    # Cleanup
    cleanup
    
    echo "LABRYS unified execution complete"
    echo "Session logs: $LOG_DIR/session.log"
    echo "Final report: $TEMP_DIR/final_status.txt"
}

# Handle script arguments
case "$1" in
    "sequential"|"parallel")
        main "$1"
        ;;
    "config")
        echo "Configuration file: $CONFIG_FILE"
        if [ -f "$CONFIG_FILE" ]; then
            cat "$CONFIG_FILE" | jq .
        else
            echo "No configuration file found. Default will be created on first run."
        fi
        ;;
    "status")
        if [ -f "$TEMP_DIR/final_status.txt" ]; then
            cat "$TEMP_DIR/final_status.txt"
        else
            echo "No status report found. Run the executor first."
        fi
        ;;
    "help"|"-h"|"--help")
        echo "LABRYS Unified Executor"
        echo "Usage: $0 [mode|command]"
        echo ""
        echo "Modes:"
        echo "  sequential  - Execute terminals one by one (default)"
        echo "  parallel    - Execute terminals in parallel"
        echo ""
        echo "Commands:"
        echo "  config      - Show current configuration"
        echo "  status      - Show last execution status"
        echo "  help        - Show this help message"
        ;;
    *)
        main "sequential"
        ;;
esac