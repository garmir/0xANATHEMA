# Task-Master PRD Recursive Generation and Optimization System

## Overview
A self-executing system that recursively decomposes project plans into PRDs, then optimizes task execution using computational complexity theory for autonomous execution.

## Prerequisites
- macOS with TouchID configured
- task-master CLI installed
- Working directory: current directory
- Initial project plan at: `task-master-instructions.md`

## Phase 1: Environment Setup

```bash
#!/bin/bash
# Initialize working environment
mkdir -p .taskmaster/{docs,optimization,catalytic,logs}

# Set environment variables using current directory
export TASKMASTER_HOME="$(pwd)/.taskmaster"
export TASKMASTER_DOCS="$TASKMASTER_HOME/docs"
export TASKMASTER_LOGS="$TASKMASTER_HOME/logs"

# Enable logging
exec > >(tee -a "$TASKMASTER_LOGS/execution-$(date +%Y%m%d-%H%M%S).log")
exec 2>&1
```

## Phase 2: Recursive PRD Generation

### 2.1 First-Level PRD Generation
```bash
# Generate initial PRDs from project plan
task-master research \
    --input project-plan.md \
    --output-pattern "$TASKMASTER_DOCS/prd-{n}.md" \
    --log-level info
```

### 2.2 Recursive Decomposition Function
```bash
#!/bin/bash
# Recursive PRD processor with depth tracking
process_prd_recursive() {
    local input_prd="$1"
    local output_dir="$2"
    local depth="${3:-0}"
    local max_depth=5
    
    # Check depth limit
    if [ "$depth" -ge "$max_depth" ]; then
        echo "Max depth reached for $input_prd"
        return
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Generate sub-PRDs
    echo "Processing: $input_prd (depth: $depth)"
    task-master research \
        --input "$input_prd" \
        --output "$output_dir" \
        --depth "$depth"
    
    # Process each generated sub-PRD
    for sub_prd in "$output_dir"/*.md; do
        if [ -f "$sub_prd" ]; then
            # Check if further decomposition needed
            if task-master next --check-atomic "$sub_prd"; then
                echo "Atomic task reached: $sub_prd"
            else
                # Create subdirectory and recurse
                sub_dir="${sub_prd%.md}"
                process_prd_recursive "$sub_prd" "$sub_dir" $((depth + 1))
            fi
        fi
    done
}

# Start recursive processing
for prd in "$TASKMASTER_DOCS"/prd-*.md; do
    if [ -f "$prd" ]; then
        prd_dir="${prd%.md}"
        process_prd_recursive "$prd" "$prd_dir" 1
    fi
done
```

### 2.3 Expected Directory Structure
```
.taskmaster/docs/
├── prd-1.md
├── prd-1/
│   ├── prd-1.1.md
│   ├── prd-1.2.md
│   ├── prd-1.1/
│   │   ├── prd-1.1.1.md
│   │   └── prd-1.1.2.md
│   └── prd-1.2/
│       └── prd-1.2.1.md
├── prd-2.md
└── prd-2/
    └── ...
```

## Phase 3: Computational Optimization

### 3.1 Dependency Analysis
```bash
cd "$TASKMASTER_HOME/optimization"

# Build complete task dependency graph
task-master analyze-dependencies \
    --input "$TASKMASTER_DOCS" \
    --output task-tree.json \
    --include-resources \
    --detect-cycles
```

### 3.2 Space-Efficient Optimization
```bash
# Apply square-root space simulation (Williams, 2025)
task-master optimize \
    --algorithm sqrt-space \
    --input task-tree.json \
    --output sqrt-optimized.json \
    --memory-bound "sqrt(n)"

# Apply tree evaluation optimization (Cook & Mertz)
task-master optimize \
    --algorithm tree-eval \
    --input sqrt-optimized.json \
    --output tree-optimized.json \
    --space-complexity "O(log n * log log n)"
```

### 3.3 Pebbling Strategy Generation
```bash
# Generate pebbling strategy for resource allocation
task-master pebble \
    --strategy branching-program \
    --input tree-optimized.json \
    --output pebbling-strategy.json \
    --minimize memory
```

### 3.4 Catalytic Execution Planning
```bash
# Initialize catalytic workspace
task-master catalytic-init \
    --workspace "$TASKMASTER_HOME/catalytic" \
    --size "10GB"

# Generate catalytic execution plan
task-master catalytic-plan \
    --input pebbling-strategy.json \
    --workspace "$TASKMASTER_HOME/catalytic" \
    --output catalytic-execution.json \
    --reuse-factor 0.8
```

## Phase 4: Evolutionary Optimization Loop

```bash
#!/bin/bash
# Iterative improvement using evolutionary algorithms
optimize_to_autonomous() {
    local max_iterations=20
    local convergence_threshold=0.95
    
    # Initial execution plan
    task-master generate-execution \
        --input catalytic-execution.json \
        --output execution-plan-v1.sh
    
    for iteration in $(seq 1 $max_iterations); do
        echo "=== Optimization Iteration $iteration ==="
        
        # Evaluate current efficiency
        task-master evaluate \
            --input "execution-plan-v$iteration.sh" \
            --metrics time,space,autonomy \
            --output "metrics-v$iteration.json"
        
        # Check autonomy score
        autonomy_score=$(jq -r '.autonomy_score' "metrics-v$iteration.json")
        echo "Autonomy score: $autonomy_score"
        
        if (( $(echo "$autonomy_score >= $convergence_threshold" | bc -l) )); then
            echo "Achieved autonomous execution capability!"
            cp "execution-plan-v$iteration.sh" final-execution.sh
            break
        fi
        
        # Apply evolutionary improvements
        task-master evolve \
            --input "execution-plan-v$iteration.sh" \
            --metrics "metrics-v$iteration.json" \
            --theory exponential-evolutionary \
            --mutation-rate 0.1 \
            --crossover-rate 0.7 \
            --output "execution-plan-v$((iteration + 1)).sh"
    done
}

# Run optimization
optimize_to_autonomous
```

## Phase 5: Final Validation and Queue Generation

```bash
# Comprehensive validation
task-master validate-autonomous \
    --input final-execution.sh \
    --checks "atomicity,dependencies,resources,timing" \
    --output validation-report.json \
    --verbose

# Generate optimized task queue
task-master finalize \
    --input final-execution.sh \
    --validation validation-report.json \
    --output "$TASKMASTER_HOME/task-queue.md" \
    --format markdown \
    --include-metadata
```

## Phase 6: Execution Monitoring

```bash
# Create monitoring dashboard
task-master monitor-init \
    --queue "$TASKMASTER_HOME/task-queue.md" \
    --dashboard "$TASKMASTER_HOME/dashboard.html"

# Execute with real-time monitoring
task-master execute \
    --queue "$TASKMASTER_HOME/task-queue.md" \
    --monitor \
    --checkpoint-interval 5m \
    --resume-on-failure
```

## Helper Functions

### Handle Sudo Operations
```bash
# Configure TouchID for sudo
task-master configure-sudo --method touchid

# Wrapper for sudo operations
sudo_with_touchid() {
    task-master sudo-exec --command "$@"
}
```

### Error Recovery
```bash
# Checkpoint and resume functionality
task-master checkpoint --save
task-master resume --from-last-checkpoint
```

## Theory References Applied

1. **Square-Root Space Simulation** - Reduces memory from O(n) to O(√n)
2. **Tree Evaluation in O(log n · log log n)** - Minimizes evaluation space
3. **Pebbling Strategies** - Optimizes resource allocation timing
4. **Catalytic Computing** - Enables memory reuse without data loss
5. **Evolutionary Algorithms** - Iteratively improves execution efficiency

## Self-Execution Command

```bash
# Save as: task-master-instructions.md
# Execute with:
claude-code --execute task-master-instructions.md \
           --working-dir "$(pwd)" \
           --log-level info \
           --checkpoint \
           --autonomous
```

## Success Criteria

- ✓ All PRDs decomposed to atomic tasks
- ✓ Task dependencies fully mapped
- ✓ Memory usage optimized to O(√n) or better
- ✓ Each task executable without human intervention
- ✓ Checkpoint/resume capability enabled
- ✓ Resource allocation optimized via pebbling
- ✓ Catalytic memory reuse implemented
- ✓ Autonomy score ≥ 0.95

## Troubleshooting

```bash
# Debug mode
export TASKMASTER_DEBUG=1

# Check system state
task-master status --detailed

# View execution logs
tail -f "$TASKMASTER_LOGS/execution-*.log"

# Reset if needed
task-master reset --preserve-prds
```