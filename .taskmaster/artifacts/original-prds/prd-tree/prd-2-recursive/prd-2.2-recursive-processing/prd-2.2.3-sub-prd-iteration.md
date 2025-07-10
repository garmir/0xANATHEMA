# PRD-2.2.3: Sub-PRD Iteration System

## Objective
Implement systematic iteration through generated sub-PRDs, applying recursive processing logic with proper file handling and atomic task detection.

## Requirements

### Functional Requirements
1. **File Discovery and Iteration**
   - Scan output directories for generated markdown files
   - Filter files using `.md` extension pattern
   - Handle empty directories gracefully
   - Process files in deterministic order

2. **File Validation**
   - Verify file existence and readability
   - Validate markdown format structure
   - Check for minimum content requirements
   - Detect and skip corrupted or invalid files

3. **Recursive Decision Logic**
   - Apply atomic task detection to each sub-PRD
   - Route atomic tasks to completion processing
   - Route non-atomic tasks to further decomposition
   - Maintain processing state for error recovery

4. **Subdirectory Management**
   - Generate subdirectory names from PRD filenames
   - Handle naming conflicts and special characters
   - Create subdirectories for further decomposition
   - Track subdirectory relationships for navigation

### Implementation
```bash
process_sub_prds() {
    local output_dir="$1"
    local current_depth="$2"
    local processed_count=0
    local atomic_count=0
    
    echo "Processing sub-PRDs in: $output_dir"
    
    # Check if directory exists and is readable
    if [ ! -d "$output_dir" ]; then
        echo "WARNING: Output directory does not exist: $output_dir"
        return 1
    fi
    
    # Iterate through markdown files
    for sub_prd in "$output_dir"/*.md; do
        # Check if glob matched any files
        if [ ! -f "$sub_prd" ]; then
            echo "INFO: No markdown files found in $output_dir"
            continue
        fi
        
        processed_count=$((processed_count + 1))
        echo "Processing sub-PRD: $(basename "$sub_prd")"
        
        # Validate file format
        if ! validate_prd_format "$sub_prd"; then
            echo "WARNING: Invalid PRD format, skipping: $sub_prd"
            continue
        fi
        
        # Apply atomic task detection
        if is_atomic_task "$sub_prd"; then
            echo "ATOMIC: $sub_prd"
            atomic_count=$((atomic_count + 1))
            add_to_atomic_list "$sub_prd"
        else
            echo "DECOMPOSE: $sub_prd"
            # Generate subdirectory name
            sub_dir="${sub_prd%.md}"
            
            # Recurse into subdirectory
            if process_prd_recursive "$sub_prd" "$sub_dir" $((current_depth + 1)); then
                echo "SUCCESS: Completed recursion for $sub_prd"
            else
                echo "ERROR: Failed recursion for $sub_prd"
            fi
        fi
    done
    
    echo "SUMMARY: Processed $processed_count files, $atomic_count atomic tasks"
    return 0
}

validate_prd_format() {
    local prd_file="$1"
    
    # Check file is readable
    if [ ! -r "$prd_file" ]; then
        return 1
    fi
    
    # Check for required sections
    if ! grep -q "^# PRD-" "$prd_file"; then
        return 1
    fi
    
    if ! grep -q "## Objective" "$prd_file"; then
        return 1
    fi
    
    return 0
}

add_to_atomic_list() {
    local atomic_prd="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $atomic_prd" >> "$TASKMASTER_HOME/atomic-tasks.log"
}
```

## Acceptance Criteria
- [ ] All generated sub-PRDs are discovered and processed
- [ ] File validation correctly identifies valid PRD format
- [ ] Atomic detection routes tasks appropriately
- [ ] Subdirectory creation works for non-atomic tasks
- [ ] Processing statistics are accurate and logged
- [ ] Error handling manages file access issues gracefully

## Dependencies
- PRD-2.2.1: Depth Management (completed)
- PRD-2.2.2: Directory Creation (completed)  
- PRD-2.3: Atomic Detection System
- File system with glob pattern support

## Success Metrics
- 100% discovery rate for generated markdown files
- Accurate atomic vs non-atomic classification
- Proper subdirectory creation for all decomposition
- Zero file processing errors for valid inputs
- Complete processing statistics and logging

## Error Handling
- Skip corrupted or unreadable files with warning
- Continue processing despite individual file failures
- Log all errors with sufficient context for debugging
- Provide summary statistics of success/failure rates
- Implement graceful degradation for permission issues