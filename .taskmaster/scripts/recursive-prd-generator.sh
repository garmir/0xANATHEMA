#!/bin/bash

# Task-Master PRD Recursive Generation and Optimization System
# Recursive PRD processor with depth tracking and optimization

set -euo pipefail

# Configuration
TASKMASTER_HOME="${TASKMASTER_HOME:-$(pwd)/.taskmaster}"
TASKMASTER_DOCS="${TASKMASTER_DOCS:-$TASKMASTER_HOME/docs}"
TASKMASTER_LOGS="${TASKMASTER_LOGS:-$TASKMASTER_HOME/logs}"
TASKMASTER_OPTIMIZATION="${TASKMASTER_OPTIMIZATION:-$TASKMASTER_HOME/optimization}"
TASKMASTER_CATALYTIC="${TASKMASTER_CATALYTIC:-$TASKMASTER_HOME/catalytic}"

# Create missing directories
mkdir -p "$TASKMASTER_DOCS" "$TASKMASTER_LOGS" "$TASKMASTER_OPTIMIZATION" "$TASKMASTER_CATALYTIC"

# Logging setup
LOG_FILE="$TASKMASTER_LOGS/recursive-prd-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=== Task-Master PRD Recursive Generation Started at $(date) ==="
echo "Working directory: $(pwd)"
echo "TASKMASTER_HOME: $TASKMASTER_HOME"
echo "Log file: $LOG_FILE"
echo ""

# Recursive PRD processor with depth tracking
process_prd_recursive() {
    local input_prd="$1"
    local output_dir="$2"
    local depth="${3:-0}"
    local max_depth=5
    
    echo "Processing PRD: $input_prd (depth: $depth)"
    
    # Check depth limit
    if [ "$depth" -ge "$max_depth" ]; then
        echo "  ⚠️  Max depth ($max_depth) reached for $input_prd"
        return 0
    fi
    
    # Verify input file exists
    if [ ! -f "$input_prd" ]; then
        echo "  ❌ Input PRD file not found: $input_prd"
        return 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    echo "  📁 Created output directory: $output_dir"
    
    # Check if task-master supports research command
    if ! task-master research --help >/dev/null 2>&1; then
        echo "  ⚠️  task-master research command not available, skipping research-based decomposition"
        return 0
    fi
    
    # Generate sub-PRDs using task-master research command
    echo "  🔄 Generating sub-PRDs from: $input_prd"
    
    # Create research prompt for PRD decomposition
    local research_prompt="Analyze the following PRD and decompose it into 3-5 smaller, focused PRDs. Each sub-PRD should cover a specific aspect or component of the original project. Format each as a complete PRD with clear objectives, requirements, and acceptance criteria."
    
    # Use task-master research with the PRD content
    if task-master research "$research_prompt" -f "$input_prd" -s "$output_dir/sub-prds.md" -d detailed; then
        echo "  ✅ Research-based decomposition completed"
        
        # Parse the research output into individual PRD files
        if [ -f "$output_dir/sub-prds.md" ]; then
            echo "  📝 Parsing research output into individual PRD files"
            parse_sub_prds "$output_dir/sub-prds.md" "$output_dir" "$depth"
        fi
    else
        echo "  ⚠️  Research-based decomposition failed, attempting manual parsing"
        manual_prd_decomposition "$input_prd" "$output_dir" "$depth"
    fi
    
    # Process each generated sub-PRD recursively
    local sub_prd_count=0
    for sub_prd in "$output_dir"/prd-*.md; do
        if [ -f "$sub_prd" ]; then
            sub_prd_count=$((sub_prd_count + 1))
            
            # Check if further decomposition needed (file size heuristic)
            local file_size=$(wc -l < "$sub_prd")
            if [ "$file_size" -lt 50 ]; then
                echo "  ⚡ Sub-PRD is atomic (< 50 lines): $sub_prd"
                continue
            fi
            
            # Create subdirectory and recurse
            local sub_dir="${sub_prd%.md}"
            echo "  🔄 Recursively processing: $sub_prd"
            process_prd_recursive "$sub_prd" "$sub_dir" $((depth + 1))
        fi
    done
    
    echo "  ✅ Completed processing at depth $depth. Generated $sub_prd_count sub-PRDs"
}

# Parse research output into individual PRD files
parse_sub_prds() {
    local research_file="$1"
    local output_dir="$2"
    local depth="$3"
    
    echo "    📄 Parsing research output: $research_file"
    
    # Simple parser for markdown sections
    local prd_counter=1
    local current_prd=""
    local in_prd=false
    
    while IFS= read -r line; do
        # Check for PRD section headers
        if [[ "$line" =~ ^#+.*PRD.*$ ]] || [[ "$line" =~ ^#+.*Product.*Requirements.*$ ]]; then
            # Save previous PRD if exists
            if [ "$in_prd" = true ] && [ -n "$current_prd" ]; then
                echo "$current_prd" > "$output_dir/prd-${depth}.${prd_counter}.md"
                echo "    ✅ Created: $output_dir/prd-${depth}.${prd_counter}.md"
                prd_counter=$((prd_counter + 1))
            fi
            
            # Start new PRD
            current_prd="$line"
            in_prd=true
        elif [ "$in_prd" = true ]; then
            current_prd="$current_prd"$'\n'"$line"
        fi
    done < "$research_file"
    
    # Save final PRD
    if [ "$in_prd" = true ] && [ -n "$current_prd" ]; then
        echo "$current_prd" > "$output_dir/prd-${depth}.${prd_counter}.md"
        echo "    ✅ Created: $output_dir/prd-${depth}.${prd_counter}.md"
    fi
}

# Manual PRD decomposition fallback
manual_prd_decomposition() {
    local input_prd="$1"
    local output_dir="$2"
    local depth="$3"
    
    echo "    🔧 Manual PRD decomposition for: $input_prd"
    
    # Create simplified sub-PRDs based on sections
    local section_counter=1
    local current_section=""
    local section_title=""
    
    while IFS= read -r line; do
        if [[ "$line" =~ ^##[[:space:]]*[0-9]+\.[[:space:]]*(.*)$ ]]; then
            # Save previous section as PRD
            if [ -n "$current_section" ] && [ -n "$section_title" ]; then
                create_sub_prd "$output_dir/prd-${depth}.${section_counter}.md" "$section_title" "$current_section"
                section_counter=$((section_counter + 1))
            fi
            
            # Start new section
            section_title="${BASH_REMATCH[1]}"
            current_section="# PRD: $section_title"$'\n\n'"$line"
        elif [ -n "$section_title" ]; then
            current_section="$current_section"$'\n'"$line"
        fi
    done < "$input_prd"
    
    # Save final section
    if [ -n "$current_section" ] && [ -n "$section_title" ]; then
        create_sub_prd "$output_dir/prd-${depth}.${section_counter}.md" "$section_title" "$current_section"
    fi
}

# Create a sub-PRD file
create_sub_prd() {
    local output_file="$1"
    local title="$2"
    local content="$3"
    
    cat > "$output_file" << EOF
$content

## Implementation Notes
- Derived from parent PRD at depth $depth
- Generated at: $(date)
- Part of recursive decomposition process

## Success Criteria
- All components implemented according to specifications
- Comprehensive testing completed
- Documentation updated
- Integration with parent system verified
EOF
    
    echo "    ✅ Created sub-PRD: $output_file"
}

# Main execution function
main() {
    echo "🚀 Starting recursive PRD generation process"
    
    # Check for required commands
    local required_commands=("task-master" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "❌ Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Find all PRD files in docs directory
    local prd_files=()
    while IFS= read -r -d '' file; do
        prd_files+=("$file")
    done < <(find "$TASKMASTER_DOCS" -name "*.txt" -o -name "*.md" | grep -i prd | head -10 | tr '\n' '\0')
    
    if [ ${#prd_files[@]} -eq 0 ]; then
        echo "⚠️  No PRD files found in $TASKMASTER_DOCS"
        echo "Looking for files with 'prd' in the name..."
        find "$TASKMASTER_DOCS" -type f | head -5
        exit 1
    fi
    
    echo "📄 Found ${#prd_files[@]} PRD file(s) to process:"
    printf '  - %s\n' "${prd_files[@]}"
    echo ""
    
    # Process each PRD file recursively
    for prd_file in "${prd_files[@]}"; do
        echo "🔄 Processing PRD: $prd_file"
        
        # Create output directory based on PRD filename
        local prd_basename=$(basename "$prd_file" | sed 's/\.[^.]*$//')
        local output_dir="$TASKMASTER_DOCS/${prd_basename}-decomposed"
        
        # Start recursive processing
        process_prd_recursive "$prd_file" "$output_dir" 1
        
        echo "✅ Completed processing: $prd_file"
        echo ""
    done
    
    echo "🎉 Recursive PRD generation completed successfully!"
    echo "📊 Generated directory structure:"
    find "$TASKMASTER_DOCS" -type d | head -20 | sort
    echo ""
    echo "📄 Generated PRD files:"
    find "$TASKMASTER_DOCS" -name "prd-*.md" | head -20 | sort
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi