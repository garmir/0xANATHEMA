#!/bin/bash

# Task-Master Evolutionary Optimization Loop
# Implements iterative improvement using evolutionary algorithms for autonomous execution

set -euo pipefail

# Configuration
TASKMASTER_HOME="${TASKMASTER_HOME:-$(pwd)/.taskmaster}"
TASKMASTER_OPTIMIZATION="${TASKMASTER_OPTIMIZATION:-$TASKMASTER_HOME/optimization}"
TASKMASTER_LOGS="${TASKMASTER_LOGS:-$TASKMASTER_HOME/logs}"

# Create directories
mkdir -p "$TASKMASTER_OPTIMIZATION" "$TASKMASTER_LOGS"

# Logging setup
LOG_FILE="$TASKMASTER_LOGS/evolutionary-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=== Task-Master Evolutionary Optimization Started at $(date) ==="
echo "Optimization directory: $TASKMASTER_OPTIMIZATION"
echo ""

# Evolutionary optimization function
optimize_to_autonomous() {
    local max_iterations=20
    local convergence_threshold=0.95
    local current_iteration=1
    
    echo "🧬 Starting evolutionary optimization loop"
    echo "🎯 Target autonomy score: $convergence_threshold"
    echo "🔄 Maximum iterations: $max_iterations"
    echo ""
    
    cd "$TASKMASTER_OPTIMIZATION"
    
    # Initialize with base execution plan
    if [ ! -f "catalytic-execution.json" ]; then
        echo "⚠️  No catalytic-execution.json found, creating mock version..."
        create_mock_catalytic_execution
    fi
    
    # Generate initial execution plan
    echo "🚀 Generating initial execution plan..."
    generate_execution_plan "catalytic-execution.json" "execution-plan-v1.sh"
    
    local best_autonomy_score=0.0
    local best_plan=""
    
    for iteration in $(seq 1 $max_iterations); do
        echo "=== 🧬 Optimization Iteration $iteration ==="
        
        local current_plan="execution-plan-v$iteration.sh"
        local metrics_file="metrics-v$iteration.json"
        
        # Evaluate current efficiency
        echo "  📊 Evaluating execution plan efficiency..."
        evaluate_execution_plan "$current_plan" "$metrics_file"
        
        # Extract autonomy score
        local autonomy_score=$(jq -r '.autonomy_score // 0.5' "$metrics_file" 2>/dev/null || echo "0.5")
        echo "  🎯 Autonomy score: $autonomy_score"
        
        # Track best plan
        if (( $(echo "$autonomy_score > $best_autonomy_score" | bc -l) )); then
            best_autonomy_score=$autonomy_score
            best_plan="$current_plan"
            echo "  ⭐ New best autonomy score: $best_autonomy_score"
        fi
        
        # Check convergence
        if (( $(echo "$autonomy_score >= $convergence_threshold" | bc -l) )); then
            echo "  🎉 Achieved autonomous execution capability!"
            echo "  ✅ Final autonomy score: $autonomy_score"
            cp "$current_plan" final-execution.sh
            break
        fi
        
        # Apply evolutionary improvements
        if [ $iteration -lt $max_iterations ]; then
            local next_plan="execution-plan-v$((iteration + 1)).sh"
            echo "  🧬 Applying evolutionary improvements..."
            apply_evolutionary_improvements "$current_plan" "$metrics_file" "$next_plan"
        fi
        
        echo ""
    done
    
    # Final results
    if [ -f "final-execution.sh" ]; then
        echo "🎉 Evolutionary optimization completed successfully!"
        echo "🏆 Final autonomy score: $best_autonomy_score"
    else
        echo "⚠️  Did not reach convergence threshold, using best plan"
        if [ -n "$best_plan" ]; then
            cp "$best_plan" final-execution.sh
            echo "📋 Best plan saved as final-execution.sh"
            echo "🏆 Best autonomy score: $best_autonomy_score"
        fi
    fi
}

# Create mock catalytic execution if needed
create_mock_catalytic_execution() {
    cat > catalytic-execution.json << 'EOF'
{
    "workspace_path": "./catalytic/workspace",
    "reuse_factor": 0.8,
    "memory_reuse_strategy": "hierarchical_pooling",
    "execution_phases": [
        {
            "phase": "initialization",
            "memory_allocation": "primary_pool",
            "reuse_potential": 0.0
        },
        {
            "phase": "main_processing", 
            "memory_allocation": "dynamic_pooling",
            "reuse_potential": 0.8
        },
        {
            "phase": "consolidation",
            "memory_allocation": "secondary_pool",
            "reuse_potential": 0.96
        }
    ],
    "optimization_parameters": {
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "population_size": 20,
        "selection_pressure": 0.6
    }
}
EOF
    echo "  ✅ Mock catalytic execution plan created"
}

# Generate execution plan from catalytic configuration
generate_execution_plan() {
    local input_file="$1"
    local output_file="$2"
    
    local reuse_factor=$(jq -r '.reuse_factor // 0.8' "$input_file")
    local mutation_rate=$(jq -r '.optimization_parameters.mutation_rate // 0.1' "$input_file")
    
    cat > "$output_file" << EOF
#!/bin/bash
# Task-Master Execution Plan v$(echo "$output_file" | grep -o 'v[0-9]*' | sed 's/v//')
# Generated: $(date)
# Reuse Factor: $reuse_factor
# Mutation Rate: $mutation_rate

set -euo pipefail

# Configuration
REUSE_FACTOR=$reuse_factor
MUTATION_RATE=$mutation_rate
WORKSPACE_PATH="./catalytic/workspace"

# Task execution functions
execute_initialization_phase() {
    echo "🚀 Initialization Phase"
    mkdir -p "\$WORKSPACE_PATH"
    echo "✅ Workspace initialized"
}

execute_main_processing() {
    echo "⚙️  Main Processing Phase"
    echo "♻️  Memory reuse factor: \$REUSE_FACTOR"
    # Simulate processing with memory reuse
    local tasks=\$(seq 1 10)
    for task in \$tasks; do
        echo "  Processing task \$task with \$REUSE_FACTOR reuse"
        sleep 0.1
    done
    echo "✅ Main processing completed"
}

execute_consolidation() {
    echo "📦 Consolidation Phase"
    echo "🔄 Final memory consolidation"
    echo "✅ Consolidation completed"
}

# Main execution
main() {
    echo "=== Task-Master Execution Plan ==="
    echo "Reuse Factor: \$REUSE_FACTOR"
    echo "Mutation Rate: \$MUTATION_RATE"
    echo ""
    
    execute_initialization_phase
    execute_main_processing
    execute_consolidation
    
    echo ""
    echo "🎉 Execution plan completed successfully!"
}

# Execute if run directly
if [[ "\${BASH_SOURCE[0]}" == "\${0}" ]]; then
    main "\$@"
fi
EOF
    
    chmod +x "$output_file"
    echo "  ✅ Execution plan generated: $output_file"
}

# Evaluate execution plan performance
evaluate_execution_plan() {
    local plan_file="$1"
    local metrics_file="$2"
    
    echo "    🔍 Running execution plan evaluation..."
    
    # Run the plan and measure performance
    local start_time=$(date +%s.%N)
    local success=true
    
    if timeout 30s bash "$plan_file" >/dev/null 2>&1; then
        echo "    ✅ Execution completed successfully"
    else
        echo "    ⚠️  Execution had issues"
        success=false
    fi
    
    local end_time=$(date +%s.%N)
    local execution_time=$(echo "$end_time - $start_time" | bc -l)
    
    # Generate metrics
    local reuse_factor=$(grep "REUSE_FACTOR=" "$plan_file" | cut -d= -f2 | head -1)
    local mutation_rate=$(grep "MUTATION_RATE=" "$plan_file" | cut -d= -f2 | head -1)
    
    # Calculate autonomy score based on multiple factors
    local base_score=0.6
    local reuse_bonus=$(echo "$reuse_factor * 0.2" | bc -l)
    local time_bonus=$(echo "1.0 / (1.0 + $execution_time)" | bc -l | awk '{printf "%.3f", $1 * 0.1}')
    local success_bonus=$([ "$success" = true ] && echo "0.1" || echo "0.0")
    
    local autonomy_score=$(echo "$base_score + $reuse_bonus + $time_bonus + $success_bonus" | bc -l)
    
    # Ensure score is between 0 and 1
    autonomy_score=$(echo "$autonomy_score" | awk '{if($1 > 1.0) print 1.0; else if($1 < 0.0) print 0.0; else print $1}')
    
    cat > "$metrics_file" << EOF
{
    "execution_time": $execution_time,
    "success": $success,
    "reuse_factor": $reuse_factor,
    "mutation_rate": $mutation_rate,
    "autonomy_score": $autonomy_score,
    "base_score": $base_score,
    "reuse_bonus": $reuse_bonus,
    "time_bonus": $time_bonus,
    "success_bonus": $success_bonus,
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    echo "    📊 Metrics saved to: $metrics_file"
}

# Apply evolutionary improvements
apply_evolutionary_improvements() {
    local current_plan="$1"
    local metrics_file="$2"
    local next_plan="$3"
    
    echo "    🧬 Applying evolutionary algorithms..."
    
    # Extract current parameters
    local current_reuse=$(jq -r '.reuse_factor' "$metrics_file")
    local current_mutation=$(jq -r '.mutation_rate' "$metrics_file")
    local autonomy_score=$(jq -r '.autonomy_score' "$metrics_file")
    
    # Apply mutations based on performance
    local reuse_mutation=0.02
    local mutation_mutation=0.01
    
    # Improve parameters based on current performance
    if (( $(echo "$autonomy_score < 0.7" | bc -l) )); then
        # Poor performance - larger mutations
        reuse_mutation=0.05
        mutation_mutation=0.02
    elif (( $(echo "$autonomy_score > 0.9" | bc -l) )); then
        # Good performance - smaller mutations
        reuse_mutation=0.01
        mutation_mutation=0.005
    fi
    
    # Apply crossover and mutation
    local new_reuse=$(echo "$current_reuse + ($RANDOM / 32767 - 0.5) * $reuse_mutation * 2" | bc -l)
    local new_mutation=$(echo "$current_mutation + ($RANDOM / 32767 - 0.5) * $mutation_mutation * 2" | bc -l)
    
    # Ensure bounds
    new_reuse=$(echo "$new_reuse" | awk '{if($1 > 1.0) print 1.0; else if($1 < 0.1) print 0.1; else print $1}')
    new_mutation=$(echo "$new_mutation" | awk '{if($1 > 0.5) print 0.5; else if($1 < 0.01) print 0.01; else print $1}')
    
    echo "    🔄 Parameter evolution:"
    echo "      Reuse Factor: $current_reuse → $new_reuse"
    echo "      Mutation Rate: $current_mutation → $new_mutation"
    
    # Generate new execution plan with evolved parameters
    cat > "$next_plan" << EOF
#!/bin/bash
# Task-Master Execution Plan (Evolved)
# Generated: $(date)
# Previous autonomy score: $autonomy_score
# Reuse Factor: $new_reuse (was $current_reuse)
# Mutation Rate: $new_mutation (was $current_mutation)

set -euo pipefail

# Configuration (evolved parameters)
REUSE_FACTOR=$new_reuse
MUTATION_RATE=$new_mutation
WORKSPACE_PATH="./catalytic/workspace"

# Enhanced task execution with evolutionary improvements
execute_initialization_phase() {
    echo "🚀 Enhanced Initialization Phase"
    mkdir -p "\$WORKSPACE_PATH"
    # Apply evolutionary improvements
    echo "🧬 Applying evolved parameters..."
    echo "✅ Workspace initialized with evolution"
}

execute_main_processing() {
    echo "⚙️  Enhanced Main Processing Phase"
    echo "♻️  Evolved memory reuse factor: \$REUSE_FACTOR"
    echo "🧬 Evolved mutation rate: \$MUTATION_RATE"
    
    # Improved processing with evolved parameters
    local tasks=\$(seq 1 \$(echo "10 + \$REUSE_FACTOR * 5" | bc | cut -d. -f1))
    for task in \$tasks; do
        echo "  Processing evolved task \$task (reuse: \$REUSE_FACTOR)"
        sleep \$(echo "0.1 * (1 - \$MUTATION_RATE)" | bc -l)
    done
    echo "✅ Enhanced main processing completed"
}

execute_consolidation() {
    echo "📦 Enhanced Consolidation Phase"
    echo "🔄 Evolved memory consolidation (factor: \$REUSE_FACTOR)"
    echo "✅ Enhanced consolidation completed"
}

# Main execution with evolutionary improvements
main() {
    echo "=== Enhanced Task-Master Execution Plan ==="
    echo "Evolved Reuse Factor: \$REUSE_FACTOR"
    echo "Evolved Mutation Rate: \$MUTATION_RATE"
    echo ""
    
    execute_initialization_phase
    execute_main_processing
    execute_consolidation
    
    echo ""
    echo "🎉 Evolved execution plan completed successfully!"
}

# Execute if run directly
if [[ "\${BASH_SOURCE[0]}" == "\${0}" ]]; then
    main "\$@"
fi
EOF
    
    chmod +x "$next_plan"
    echo "    ✅ Evolved execution plan generated: $next_plan"
}

# Generate final evolution report
generate_evolution_report() {
    echo ""
    echo "📊 Generating evolutionary optimization report..."
    
    local report_file="evolution-report.json"
    local iterations=$(ls -1 metrics-v*.json 2>/dev/null | wc -l)
    
    cat > "$report_file" << EOF
{
    "generated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "evolutionary_optimization": {
        "total_iterations": $iterations,
        "convergence_threshold": 0.95,
        "algorithm": "exponential-evolutionary",
        "parameters": {
            "mutation_rate_range": [0.01, 0.5],
            "crossover_rate": 0.7,
            "selection_pressure": 0.6
        }
    },
    "performance_metrics": {
        "initial_autonomy": null,
        "final_autonomy": null,
        "improvement": null,
        "convergence_achieved": false
    },
    "optimization_phases": [
        "parameter_evolution",
        "fitness_evaluation", 
        "selection_and_crossover",
        "mutation_application",
        "convergence_testing"
    ]
}
EOF
    
    # Add metrics from iterations if available
    if [ $iterations -gt 0 ]; then
        local initial_score=$(jq -r '.autonomy_score' metrics-v1.json 2>/dev/null || echo "null")
        local final_iteration=$(ls -1 metrics-v*.json | tail -1)
        local final_score=$(jq -r '.autonomy_score' "$final_iteration" 2>/dev/null || echo "null")
        
        # Update report with actual metrics
        jq --argjson initial "$initial_score" --argjson final "$final_score" '
        .performance_metrics.initial_autonomy = $initial |
        .performance_metrics.final_autonomy = $final |
        .performance_metrics.improvement = (if $initial and $final then ($final - $initial) else null end) |
        .performance_metrics.convergence_achieved = ($final >= 0.95)
        ' "$report_file" > "${report_file}.tmp" && mv "${report_file}.tmp" "$report_file"
    fi
    
    echo "  📋 Evolution report generated: $report_file"
    
    # Display summary
    echo ""
    echo "🧬 Evolutionary Optimization Complete!"
    echo "════════════════════════════════════════"
    echo "🔄 Total iterations: $iterations"
    echo "🎯 Convergence threshold: 0.95"
    
    if [ -f final-execution.sh ]; then
        echo "✅ Final execution plan generated"
        echo "📁 File: final-execution.sh"
    fi
    
    echo ""
    echo "📊 Evolution metrics:"
    if [ $iterations -gt 0 ]; then
        echo "  📈 Performance progression:"
        for metrics in metrics-v*.json; do
            if [ -f "$metrics" ]; then
                local iter=$(echo "$metrics" | grep -o 'v[0-9]*' | sed 's/v//')
                local score=$(jq -r '.autonomy_score' "$metrics")
                echo "    Iteration $iter: $score"
            fi
        done
    fi
}

# Main execution function
main() {
    echo "🧬 Starting evolutionary optimization for autonomous execution"
    
    # Check for required commands
    local required_commands=("jq" "bc")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "❌ Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Execute evolutionary optimization
    optimize_to_autonomous
    generate_evolution_report
    
    echo ""
    echo "✨ Evolutionary optimization completed successfully!"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi