# Check autonomy score

## Overview

# Check autonomy score
        autonomy_score=$(jq -r '.autonomy_score' "metrics-v$iteration.json")
        echo "Autonomy score: $autonomy_score"
        
        if (( $(echo "$autonomy_score >= $convergence_threshold" | bc -l) )); then
            echo "Achieved autonomous execution capability!"
            cp "execution-plan-v$iteration.sh" final-execution.sh
            break
        fi

## Requirements

Generated from project plan section: Check autonomy score

## Implementation Notes

This PRD was auto-generated from the project plan and may require further decomposition.

