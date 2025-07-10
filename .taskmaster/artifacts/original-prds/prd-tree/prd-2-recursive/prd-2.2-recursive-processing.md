# PRD-2.2: Recursive Processing Implementation

## Objective
Implement the core recursive function that processes PRDs iteratively, creating hierarchical decomposition with proper depth tracking and termination conditions.

## Requirements

### Functional Requirements

1. **Core Recursive Function**
   - Function signature: `process_prd_recursive(input_prd, output_dir, depth)`
   - Default depth parameter initialization to 0
   - Maximum depth limit enforcement (5 levels)
   - Proper error handling and logging

2. **Depth Management**
   - Track current recursion depth accurately
   - Implement boundary checking before recursion
   - Prevent infinite loops through depth limits
   - Log depth progression for debugging

3. **Directory Management**
   - Create output directories automatically
   - Handle directory creation failures
   - Maintain hierarchical naming structure
   - Ensure proper permissions on created directories

4. **Sub-PRD Processing**
   - Iterate through all generated markdown files
   - Filter for valid PRD files using pattern matching
   - Apply recursive processing to non-atomic PRDs
   - Skip processing for atomic tasks

### Non-Functional Requirements
- Function must be reentrant and safe for concurrent calls
- Memory usage must remain bounded during recursion
- Each recursion level must complete within 90 seconds
- Directory structure must remain consistent across runs

## Acceptance Criteria
- [ ] Recursive function correctly limits depth to 5 levels
- [ ] Directory creation works for all recursion levels
- [ ] Atomic task detection prevents unnecessary processing
- [ ] Function handles errors without corrupting state
- [ ] Logging provides clear recursion progress tracking
- [ ] Output directory structure matches expected hierarchy

## Implementation Details

### Function Structure
```bash
process_prd_recursive() {
    local input_prd="$1"
    local output_dir="$2"
    local depth="${3:-0}"
    local max_depth=5
    
    # Depth boundary check
    if [ "$depth" -ge "$max_depth" ]; then
        echo "Max depth reached for $input_prd"
        return 0
    fi
    
    # Create output directory
    mkdir -p "$output_dir" || {
        echo "Failed to create directory: $output_dir"
        return 1
    }
    
    # Process current PRD
    echo "Processing: $input_prd (depth: $depth)"
    task-master research \
        --input "$input_prd" \
        --output "$output_dir" \
        --depth "$depth" || {
        echo "Research command failed for $input_prd"
        return 1
    }
    
    # Process generated sub-PRDs
    for sub_prd in "$output_dir"/*.md; do
        if [ -f "$sub_prd" ]; then
            if is_atomic_task "$sub_prd"; then
                echo "Atomic task reached: $sub_prd"
            else
                sub_dir="${sub_prd%.md}"
                process_prd_recursive "$sub_prd" "$sub_dir" $((depth + 1))
            fi
        fi
    done
    
    return 0
}
```

### Atomic Task Detection
```bash
is_atomic_task() {
    local prd_file="$1"
    
    # Check for single responsibility indicators
    # Check for specific action verbs
    # Check for measurable outcomes
    # Return 0 if atomic, 1 if needs decomposition
    
    task-master analyze-atomicity "$prd_file"
    return $?
}
```

### Main Processing Loop
```bash
# Process all initial PRDs
for prd in "$TASKMASTER_DOCS"/prd-*.md; do
    if [ -f "$prd" ]; then
        prd_dir="${prd%.md}"
        echo "Starting recursive processing for: $prd"
        process_prd_recursive "$prd" "$prd_dir" 1
    fi
done
```

## Dependencies
- PRD-2.1: First-Level Generation (completed)
- Initial PRDs generated and available
- task-master CLI with research and analyze capabilities
- Bash shell with function definition support

## Success Metrics
- Processes 6+ initial PRDs successfully
- Creates 3-4 levels of decomposition hierarchy
- Generates 40+ total PRD files across all levels
- Zero infinite recursion or stack overflow errors
- All directory structures navigable and consistent

## Directory Structure Output
```
.taskmaster/docs/
├── prd-1.md
├── prd-1/
│   ├── prd-1.1.md (Directory Creation)
│   ├── prd-1.2.md (Environment Variables)
│   ├── prd-1.3.md (Logging Configuration)
│   ├── prd-1.1/
│   │   ├── prd-1.1.1.md (mkdir Command)
│   │   └── prd-1.1.2.md (Permission Setting)
│   └── prd-1.2/
│       ├── prd-1.2.1.md (TASKMASTER_HOME)
│       └── prd-1.2.2.md (PATH Export)
├── prd-2.md
└── prd-2/
    ├── prd-2.1.md
    ├── prd-2.2.md  
    └── prd-2.3.md
```

## Error Handling
- Validate input PRD file exists and is readable
- Handle task-master command failures with appropriate logging
- Implement graceful degradation for directory creation failures
- Provide detailed error context for debugging
- Include recovery mechanisms for partial failures

## Performance Considerations
- Limit concurrent recursion to prevent resource exhaustion
- Implement progress tracking for long-running operations
- Use efficient file system operations
- Monitor memory usage during deep recursion
- Include timeout mechanisms for individual operations