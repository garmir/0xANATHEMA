# PRD-2.3: Atomic Task Detection System

## Objective
Implement intelligent detection system to identify when PRDs represent atomic tasks that cannot be further decomposed, preventing unnecessary recursion.

## Requirements

### Functional Requirements

1. **Atomicity Analysis Algorithm**
   - Analyze PRD content for single responsibility principle
   - Detect specific, actionable verbs vs. vague terminology
   - Identify measurable outcomes and success criteria
   - Calculate complexity score for decomposition decision

2. **Content Pattern Recognition**
   - Parse markdown structure for requirement patterns
   - Identify implementation-specific details
   - Detect step-by-step execution instructions
   - Recognize completion criteria specificity

3. **Decision Logic Implementation**
   - Apply configurable atomicity thresholds
   - Use weighted scoring for multiple factors
   - Provide binary decision: atomic vs. decomposable
   - Include confidence level in decision output

4. **Integration with Recursive System**
   - Interface with `process_prd_recursive()` function
   - Return appropriate exit codes for shell integration
   - Provide detailed logging for decision rationale
   - Handle edge cases and ambiguous content

### Non-Functional Requirements
- Analysis must complete within 5 seconds per PRD
- Decision accuracy must exceed 90% for known patterns
- Algorithm must be deterministic and repeatable
- Memory usage must remain minimal during analysis

## Acceptance Criteria
- [ ] Correctly identifies atomic tasks with 90%+ accuracy
- [ ] Prevents unnecessary decomposition of detailed PRDs
- [ ] Integrates seamlessly with recursive processing
- [ ] Provides clear logging of decision rationale
- [ ] Handles edge cases without errors
- [ ] Maintains consistent decision criteria across PRDs

## Implementation Components

### Main Analysis Function
```bash
is_atomic_task() {
    local prd_file="$1"
    local threshold="${2:-0.7}"
    
    if [ ! -f "$prd_file" ]; then
        echo "ERROR: PRD file not found: $prd_file"
        return 2
    fi
    
    # Calculate atomicity score
    local atomicity_score
    atomicity_score=$(calculate_atomicity_score "$prd_file")
    
    # Apply threshold decision
    if (( $(echo "$atomicity_score >= $threshold" | bc -l) )); then
        echo "ATOMIC: $prd_file (score: $atomicity_score)"
        return 0  # Atomic task
    else
        echo "DECOMPOSE: $prd_file (score: $atomicity_score)"
        return 1  # Needs decomposition
    fi
}
```

### Atomicity Score Calculation
```bash
calculate_atomicity_score() {
    local prd_file="$1"
    local score=0
    
    # Factor 1: Single responsibility (0.3 weight)
    local responsibility_score
    responsibility_score=$(analyze_responsibility "$prd_file")
    score=$(echo "$score + $responsibility_score * 0.3" | bc -l)
    
    # Factor 2: Specific action verbs (0.2 weight)
    local verb_score
    verb_score=$(analyze_action_verbs "$prd_file")
    score=$(echo "$score + $verb_score * 0.2" | bc -l)
    
    # Factor 3: Measurable outcomes (0.25 weight)
    local outcome_score
    outcome_score=$(analyze_measurable_outcomes "$prd_file")
    score=$(echo "$score + $outcome_score * 0.25" | bc -l)
    
    # Factor 4: Implementation specificity (0.25 weight)
    local specificity_score
    specificity_score=$(analyze_implementation_specificity "$prd_file")
    score=$(echo "$score + $specificity_score * 0.25" | bc -l)
    
    echo "$score"
}
```

### Responsibility Analysis
```bash
analyze_responsibility() {
    local prd_file="$1"
    local objective_count
    
    # Count distinct objectives in PRD
    objective_count=$(grep -c "^##.*Objective\|^###.*Goal\|^-.*implement\|^-.*create" "$prd_file")
    
    if [ "$objective_count" -eq 1 ]; then
        echo "1.0"  # Single clear objective
    elif [ "$objective_count" -eq 2 ]; then
        echo "0.6"  # Borderline, might be atomic
    else
        echo "0.2"  # Multiple objectives, likely needs decomposition
    fi
}
```

### Action Verb Analysis
```bash
analyze_action_verbs() {
    local prd_file="$1"
    local specific_verbs="create|implement|generate|execute|validate|configure|install|deploy"
    local vague_verbs="handle|manage|deal|work|process|setup|organize"
    
    local specific_count
    local vague_count
    
    specific_count=$(grep -cE "$specific_verbs" "$prd_file")
    vague_count=$(grep -cE "$vague_verbs" "$prd_file")
    
    if [ "$specific_count" -gt 0 ] && [ "$vague_count" -eq 0 ]; then
        echo "1.0"  # Only specific verbs
    elif [ "$specific_count" -gt "$vague_count" ]; then
        echo "0.7"  # Mostly specific
    elif [ "$specific_count" -eq "$vague_count" ]; then
        echo "0.5"  # Mixed
    else
        echo "0.3"  # Mostly vague
    fi
}
```

### Measurable Outcomes Analysis
```bash
analyze_measurable_outcomes() {
    local prd_file="$1"
    local measurable_patterns="[0-9]+\s*(minutes|seconds|MB|GB|%|files?|tasks?)"
    local success_criteria_count
    
    success_criteria_count=$(grep -c "Success Criteria\|Acceptance Criteria" "$prd_file")
    local measurable_count
    measurable_count=$(grep -cE "$measurable_patterns" "$prd_file")
    
    if [ "$success_criteria_count" -gt 0 ] && [ "$measurable_count" -gt 2 ]; then
        echo "1.0"  # Clear measurable criteria
    elif [ "$measurable_count" -gt 0 ]; then
        echo "0.6"  # Some measurable elements
    else
        echo "0.2"  # Vague or missing criteria
    fi
}
```

### Implementation Specificity Analysis
```bash
analyze_implementation_specificity() {
    local prd_file="$1"
    local code_blocks
    local command_examples
    local technical_details
    
    code_blocks=$(grep -c "^```" "$prd_file")
    command_examples=$(grep -c "task-master\|bash\|mkdir\|export" "$prd_file")
    technical_details=$(grep -c "algorithm\|function\|parameter\|configuration" "$prd_file")
    
    local specificity_score
    specificity_score=$((code_blocks + command_examples + technical_details))
    
    if [ "$specificity_score" -gt 5 ]; then
        echo "1.0"  # Highly specific implementation
    elif [ "$specificity_score" -gt 2 ]; then
        echo "0.7"  # Moderately specific
    elif [ "$specificity_score" -gt 0 ]; then
        echo "0.4"  # Some specificity
    else
        echo "0.1"  # Very abstract
    fi
}
```

## Integration Interface
```bash
# Usage in recursive processing
for sub_prd in "$output_dir"/*.md; do
    if [ -f "$sub_prd" ]; then
        if is_atomic_task "$sub_prd" 0.75; then
            echo "Atomic task reached: $sub_prd"
            # Add to atomic task list
            echo "$sub_prd" >> "$TASKMASTER_HOME/atomic-tasks.txt"
        else
            echo "Further decomposition needed: $sub_prd"
            sub_dir="${sub_prd%.md}"
            process_prd_recursive "$sub_prd" "$sub_dir" $((depth + 1))
        fi
    fi
done
```

## Dependencies
- PRD-2.2: Recursive Processing (completed)
- Generated PRD files available for analysis
- Basic text processing utilities (grep, bc)
- Bash shell with mathematical operations support

## Success Metrics
- Correctly classifies 90%+ of test PRDs
- Prevents infinite recursion in 100% of cases
- Analysis completes within 5 seconds per PRD
- Decision rationale is logged and traceable
- No false negatives leading to missed decomposition

## Atomicity Indicators
**Atomic Task Characteristics:**
- Single, specific action verb
- Measurable completion criteria
- Implementation details present
- Clear input/output specification
- Execution time under 30 minutes

**Non-Atomic Task Characteristics:**
- Multiple objectives or responsibilities
- Vague or abstract language
- Missing implementation details
- Complex interdependencies
- Estimated execution time over 1 hour

## Error Handling
- Validate PRD file format before analysis
- Handle missing or corrupted content gracefully
- Provide fallback decisions for edge cases
- Log all analysis steps for debugging
- Include confidence levels in decision output