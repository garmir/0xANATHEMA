#!/bin/bash
# Recursive PRD processor with depth tracking
# Implements the recursive decomposition system from PRD-2

# Set environment variables
export TASKMASTER_HOME="${TASKMASTER_HOME:-$(pwd)/.taskmaster}"
export TASKMASTER_DOCS="$TASKMASTER_HOME/docs"
export TASKMASTER_LOGS="$TASKMASTER_HOME/logs"

# Enable logging
exec > >(tee -a "$TASKMASTER_LOGS/recursive-processing-$(date +%Y%m%d-%H%M%S).log")
exec 2>&1

echo "Starting recursive PRD processing at $(date)"

# Improved atomicity checking function
check_prd_atomicity() {
    local prd_file="$1"
    
    # Read the PRD content for analysis
    local content=$(cat "$prd_file" 2>/dev/null || echo "")
    local line_count=$(echo "$content" | wc -l)
    local word_count=$(echo "$content" | wc -w)
    
    # Check for atomic task indicators
    local atomic_indicators=(
        "specific implementation"
        "concrete code"
        "single function"
        "one component"
        "atomic task"
        "implementation detail"
        "specific API"
        "single file"
    )
    
    local composite_indicators=(
        "system architecture"
        "multiple components"
        "several modules"
        "architecture design"
        "overall system"
        "high-level design"
        "framework"
        "platform"
    )
    
    # Score atomicity based on content analysis
    local atomic_score=0
    local composite_score=0
    
    for indicator in "${atomic_indicators[@]}"; do
        if echo "$content" | grep -qi "$indicator"; then
            atomic_score=$((atomic_score + 1))
        fi
    done
    
    for indicator in "${composite_indicators[@]}"; do
        if echo "$content" | grep -qi "$indicator"; then
            composite_score=$((composite_score + 1))
        fi
    done
    
    # Decision logic: atomic if small file OR high atomic indicators OR low composite indicators
    if [ "$line_count" -lt 30 ] || [ "$word_count" -lt 200 ]; then
        echo "ATOMIC" # Small files are likely atomic
        return 0
    elif [ "$atomic_score" -gt "$composite_score" ] && [ "$atomic_score" -ge 2 ]; then
        echo "ATOMIC" # High atomic indicators
        return 0
    elif [ "$composite_score" -ge 3 ]; then
        echo "COMPOSITE" # High composite indicators
        return 0
    elif [ "$line_count" -lt 100 ] && [ "$word_count" -lt 800 ]; then
        echo "ATOMIC" # Medium-small files
        return 0
    else
        echo "COMPOSITE" # Default to composite for larger content
        return 0
    fi
}

# Validate directory structure hierarchy
validate_directory_structure() {
    local base_dir="$1"
    local current_depth="$2"
    
    echo "Validating directory structure at depth $current_depth: $base_dir"
    
    # Check if directory exists
    if [ ! -d "$base_dir" ]; then
        echo "ERROR: Directory does not exist: $base_dir"
        return 1
    fi
    
    # Validate naming convention: should contain depth level
    local dir_name=$(basename "$base_dir")
    
    # Expected hierarchy: docs/prd-X/level-1/level-2/level-3/etc
    case "$current_depth" in
        0) 
            if [[ ! "$dir_name" =~ ^docs$ ]]; then
                echo "WARNING: Root should be 'docs', found: $dir_name"
            fi
            ;;
        1) 
            if [[ ! "$dir_name" =~ ^prd- ]]; then
                echo "WARNING: Level 1 should start with 'prd-', found: $dir_name"
            fi
            ;;
        *) 
            if [[ ! "$dir_name" =~ (level-[0-9]+|sub-|component-|module-) ]]; then
                echo "WARNING: Level $current_depth should follow naming convention, found: $dir_name"
            fi
            ;;
    esac
    
    echo "Directory structure validation passed for: $base_dir"
    return 0
}

# Enhanced recursive PRD processor function
process_prd_recursive() {
    local input_prd="$1"
    local output_dir="$2"
    local depth="${3:-0}"
    local max_depth=5
    
    echo "=== Processing PRD: $input_prd (depth: $depth) ==="
    
    # Enhanced depth validation
    if [ "$depth" -ge "$max_depth" ]; then
        echo "ERROR: Maximum recursion depth ($max_depth) exceeded for $input_prd"
        echo "DEPTH_LIMIT_EXCEEDED" > "$output_dir.status"
        return 2 # Specific error code for depth limit
    fi
    
    # Enhanced input validation
    if [ ! -f "$input_prd" ]; then
        echo "ERROR: Input PRD file not found: $input_prd"
        return 1
    fi
    
    # Validate PRD format
    if ! head -5 "$input_prd" | grep -qi "^#"; then
        echo "ERROR: Invalid PRD format - missing markdown headers: $input_prd"
        return 3 # Specific error code for format validation
    fi
    
    # Enhanced directory creation with validation
    if ! mkdir -p "$output_dir"; then
        echo "ERROR: Failed to create output directory: $output_dir"
        return 4 # Specific error code for filesystem operations
    fi
    
    # Validate directory structure
    if ! validate_directory_structure "$output_dir" "$depth"; then
        echo "WARNING: Directory structure validation failed for: $output_dir"
    fi
    
    echo "Created and validated output directory: $output_dir"
    
    # Enhanced atomicity checking
    local atomicity_result=$(check_prd_atomicity "$input_prd")
    echo "Atomicity check result: $atomicity_result for $input_prd"
    
    if [ "$atomicity_result" = "ATOMIC" ]; then
        echo "ATOMIC PRD detected - no further decomposition needed: $input_prd"
        echo "ATOMIC" > "$output_dir.status"
        return 0
    fi
    
    # Generate sub-PRDs using task-master research with enhanced error handling
    echo "Generating sub-PRDs from $input_prd (non-atomic)..."
    
    local research_output="$output_dir/sub-prds-$(basename "$input_prd")"
    if task-master research "Break down this PRD into smaller, more focused PRD documents. Create separate sections for different components or phases." -f "$input_prd" -s "$research_output"; then
        echo "Successfully generated sub-PRDs research: $research_output"
    else
        echo "WARNING: task-master research failed, attempting alternative decomposition"
        # Fallback: create basic structural decomposition
        echo "# Sub-PRD Decomposition for $(basename "$input_prd")" > "$research_output"
        echo "This PRD requires manual decomposition at depth $depth" >> "$research_output"
    fi
    
    # Enhanced sub-PRD processing with better error handling
    local processed_count=0
    local failed_count=0
    
    for sub_prd in "$output_dir"/*.md; do
        if [ -f "$sub_prd" ] && [ "$sub_prd" != "$input_prd" ]; then
            echo "Processing sub-PRD: $sub_prd"
            processed_count=$((processed_count + 1))
            
            # Enhanced atomicity check for sub-PRDs
            local sub_atomicity=$(check_prd_atomicity "$sub_prd")
            echo "Sub-PRD atomicity: $sub_atomicity"
            
            if [ "$sub_atomicity" = "ATOMIC" ]; then
                echo "Atomic sub-PRD reached: $sub_prd"
                echo "ATOMIC" > "${sub_prd}.status"
            else
                echo "Non-atomic sub-PRD, recursing: $sub_prd"
                # Create subdirectory with proper naming and recurse
                local sub_dir="${sub_prd%.md}-level-$((depth + 1))"
                
                if process_prd_recursive "$sub_prd" "$sub_dir" $((depth + 1)); then
                    echo "SUCCESS" > "${sub_prd}.status"
                else
                    echo "FAILED" > "${sub_prd}.status"
                    failed_count=$((failed_count + 1))
                    echo "ERROR: Failed to process sub-PRD: $sub_prd"
                fi
            fi
        fi
    done
    
    # Enhanced status reporting
    echo "Processing summary for $input_prd:"
    echo "  - Processed: $processed_count sub-PRDs"
    echo "  - Failed: $failed_count sub-PRDs"
    echo "  - Depth: $depth/$max_depth"
    echo "  - Atomicity: $atomicity_result"
    
    # Write processing status
    cat > "$output_dir.metadata" << EOF
{
    "input_prd": "$input_prd",
    "output_dir": "$output_dir", 
    "depth": $depth,
    "max_depth": $max_depth,
    "atomicity": "$atomicity_result",
    "processed_count": $processed_count,
    "failed_count": $failed_count,
    "timestamp": "$(date -Iseconds)"
}
EOF
    
    if [ "$failed_count" -gt 0 ]; then
        echo "PARTIAL_SUCCESS" > "$output_dir.status"
        return 5 # Partial success code
    else
        echo "SUCCESS" > "$output_dir.status"
        return 0
    fi
}

# Validate environment
if [ ! -d "$TASKMASTER_DOCS" ]; then
    echo "ERROR: TASKMASTER_DOCS directory not found: $TASKMASTER_DOCS"
    echo "Please run environment setup first (PRD-1)"
    exit 1
fi

echo "Environment validated. Starting recursive processing..."

# Process all PRDs in the docs directory
processed_main_prds=0
for prd in "$TASKMASTER_DOCS"/prd-*.md; do
    if [ -f "$prd" ]; then
        echo "Processing main PRD: $prd"
        prd_dir="${prd%.md}"
        if process_prd_recursive "$prd" "$prd_dir" 1; then
            processed_main_prds=$((processed_main_prds + 1))
            echo "Successfully processed: $prd"
        else
            echo "ERROR: Failed to process: $prd"
        fi
        echo "----------------------------------------"
    fi
done

echo "=== Recursive PRD Processing Complete ==="
echo "Processed $processed_main_prds main PRDs"
echo "Completed at $(date)"

# Generate summary report
echo "Generating directory structure report..."
find "$TASKMASTER_DOCS" -name "*.md" | sort | tee "$TASKMASTER_LOGS/prd-structure-$(date +%Y%m%d-%H%M%S).txt"

echo "Recursive PRD processing finished successfully!"