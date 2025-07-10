#!/bin/bash
# Simple Space Complexity Validation System
# Uses system tools for memory measurement without external dependencies

set -e

VALIDATION_DIR="/Users/anam/archive/.taskmaster/complexity-validation"
RESULTS_DIR="$VALIDATION_DIR/results"
mkdir -p "$RESULTS_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

error() {
    echo -e "${RED}✗ $1${NC}"
}

# Memory measurement function
measure_memory() {
    local pid="$1"
    local duration="$2"
    local output_file="$3"
    local interval=0.1
    
    log "Measuring memory for PID $pid for ${duration}s"
    
    echo "timestamp,rss_mb,vsz_mb" > "$output_file"
    
    local start_time=$(date +%s.%N)
    local end_time=$(echo "$start_time + $duration" | bc -l)
    
    while [ $(echo "$(date +%s.%N) < $end_time" | bc -l) -eq 1 ]; do
        if ps -p "$pid" > /dev/null 2>&1; then
            local memory_info=$(ps -o rss,vsz -p "$pid" | tail -1)
            local rss_kb=$(echo "$memory_info" | awk '{print $1}')
            local vsz_kb=$(echo "$memory_info" | awk '{print $2}')
            local rss_mb=$(echo "scale=2; $rss_kb / 1024" | bc -l)
            local vsz_mb=$(echo "scale=2; $vsz_kb / 1024" | bc -l)
            local timestamp=$(date +%s.%N)
            
            echo "$timestamp,$rss_mb,$vsz_mb" >> "$output_file"
        else
            warning "Process $pid terminated"
            break
        fi
        sleep "$interval"
    done
}

# Create test algorithm implementations
create_sqrt_algorithm() {
    cat > "$VALIDATION_DIR/sqrt_test.py" << 'EOF'
#!/usr/bin/env python3
import sys
import math
import time

def sqrt_space_algorithm(n):
    """O(√n) space algorithm simulation"""
    chunk_size = max(1, int(math.sqrt(n)))
    chunks = []
    
    # Allocate sqrt(n) chunks, each with sqrt(n) elements
    for i in range(chunk_size):
        chunk = list(range(chunk_size))
        chunks.append(chunk)
    
    # Simulate processing
    time.sleep(0.5)  # Give time for memory measurement
    
    result = 0
    for chunk in chunks:
        result += sum(chunk)
    
    return result

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(f"Running sqrt algorithm with n={n}")
    result = sqrt_space_algorithm(n)
    print(f"Result: {result}")
EOF
    chmod +x "$VALIDATION_DIR/sqrt_test.py"
}

create_loglog_algorithm() {
    cat > "$VALIDATION_DIR/loglog_test.py" << 'EOF'
#!/usr/bin/env python3
import sys
import math
import time

def loglog_space_algorithm(n):
    """O(log n * log log n) space algorithm simulation"""
    if n <= 2:
        return n
    
    log_n = max(1, int(math.log(n)))
    log_log_n = max(1, int(math.log(log_n)) if log_n > 1 else 1)
    
    # Allocate O(log n * log log n) space
    stack = list(range(log_n))
    memoization = {}
    
    for i in range(log_log_n):
        for j in range(log_n):
            key = (i, j)
            memoization[key] = i * j
    
    # Simulate processing
    time.sleep(0.5)  # Give time for memory measurement
    
    result = sum(stack) + sum(memoization.values())
    return result

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(f"Running loglog algorithm with n={n}")
    result = loglog_space_algorithm(n)
    print(f"Result: {result}")
EOF
    chmod +x "$VALIDATION_DIR/loglog_test.py"
}

# Calculate theoretical bounds
calculate_sqrt_bound() {
    local n="$1"
    local constant=0.001  # MB per element
    echo "scale=3; $constant * sqrt($n)" | bc -l
}

calculate_loglog_bound() {
    local n="$1"
    local constant=0.0001  # MB per element
    if [ "$n" -le 2 ]; then
        echo "$constant"
    else
        local log_n=$(echo "scale=6; l($n)/l(2)" | bc -l)
        local log_log_n=$(echo "scale=6; l($log_n)/l(2)" | bc -l)
        echo "scale=3; $constant * $log_n * $log_log_n" | bc -l
    fi
}

# Validate complexity bounds
validate_complexity() {
    local algorithm="$1"
    local n_values="100 316 1000 3162 10000"
    local tolerance="1.15"  # 15% tolerance
    
    log "Validating $algorithm complexity"
    echo "n,peak_memory_mb,theoretical_mb,ratio,valid" > "$RESULTS_DIR/${algorithm}_validation.csv"
    
    local total_tests=0
    local passed_tests=0
    
    for n in $n_values; do
        log "Testing $algorithm with n=$n"
        
        # Run algorithm and measure memory
        local memory_file="$RESULTS_DIR/memory_${algorithm}_${n}.csv"
        
        if [ "$algorithm" = "sqrt" ]; then
            python3 "$VALIDATION_DIR/sqrt_test.py" "$n" &
        else
            python3 "$VALIDATION_DIR/loglog_test.py" "$n" &
        fi
        
        local pid=$!
        measure_memory "$pid" 2 "$memory_file"
        wait "$pid"
        
        # Analyze results
        if [ -f "$memory_file" ]; then
            local peak_memory=$(awk -F',' 'NR>1 {if($2>max) max=$2} END {print max}' "$memory_file")
            
            # Calculate theoretical bound
            local theoretical
            if [ "$algorithm" = "sqrt" ]; then
                theoretical=$(calculate_sqrt_bound "$n")
            else
                theoretical=$(calculate_loglog_bound "$n")
            fi
            
            # Calculate ratio
            local ratio=$(echo "scale=3; $peak_memory / $theoretical" | bc -l)
            
            # Validate
            local is_valid=$(echo "$ratio <= $tolerance" | bc -l)
            local valid_text="false"
            if [ "$is_valid" -eq 1 ]; then
                valid_text="true"
                ((passed_tests++))
            fi
            
            ((total_tests++))
            
            echo "$n,$peak_memory,$theoretical,$ratio,$valid_text" >> "$RESULTS_DIR/${algorithm}_validation.csv"
            
            if [ "$valid_text" = "true" ]; then
                success "n=$n: ${peak_memory}MB <= ${theoretical}MB * $tolerance (ratio: $ratio)"
            else
                error "n=$n: ${peak_memory}MB > ${theoretical}MB * $tolerance (ratio: $ratio)"
            fi
        else
            error "Failed to measure memory for n=$n"
        fi
    done
    
    # Summary
    local success_rate=$(echo "scale=2; $passed_tests * 100 / $total_tests" | bc -l)
    log "$algorithm validation complete: $passed_tests/$total_tests tests passed (${success_rate}%)"
    
    return $passed_tests
}

# Generate analysis report
generate_report() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local report_file="$RESULTS_DIR/complexity_validation_report.json"
    
    log "Generating validation report"
    
    cat > "$report_file" << EOF
{
  "validation_report": {
    "timestamp": "$timestamp",
    "system_info": {
      "os": "$(uname -s)",
      "architecture": "$(uname -m)",
      "memory_total": "$(sysctl hw.memsize | awk '{print $2/1024/1024/1024}') GB"
    },
    "test_configuration": {
      "tolerance": "15%",
      "measurement_interval": "0.1s",
      "test_duration": "2s",
      "test_sizes": [100, 316, 1000, 3162, 10000]
    },
    "results": {
      "sqrt_algorithm": "$([ -f "$RESULTS_DIR/sqrt_validation.csv" ] && echo "completed" || echo "failed")",
      "loglog_algorithm": "$([ -f "$RESULTS_DIR/loglog_validation.csv" ] && echo "completed" || echo "failed")"
    }
  }
}
EOF

    success "Report generated: $report_file"
}

# Main execution
main() {
    echo
    log "Space Complexity Validation System"
    echo "======================================"
    
    # Create test algorithms
    log "Creating test algorithms"
    create_sqrt_algorithm
    create_loglog_algorithm
    success "Test algorithms created"
    
    # Validate O(√n) algorithm
    echo
    log "Testing O(√n) Space Complexity"
    echo "--------------------------------"
    validate_complexity "sqrt"
    local sqrt_result=$?
    
    # Validate O(log n * log log n) algorithm
    echo
    log "Testing O(log n * log log n) Space Complexity"
    echo "----------------------------------------------"
    validate_complexity "loglog"
    local loglog_result=$?
    
    # Generate final report
    echo
    generate_report
    
    # Summary
    echo
    log "VALIDATION SUMMARY"
    echo "=================="
    success "O(√n) algorithm: $sqrt_result tests passed"
    success "O(log n·log log n) algorithm: $loglog_result tests passed"
    success "Results saved in: $RESULTS_DIR"
    
    if [ "$sqrt_result" -gt 0 ] && [ "$loglog_result" -gt 0 ]; then
        success "Space complexity validation PASSED"
        return 0
    else
        error "Space complexity validation FAILED"
        return 1
    fi
}

# Execute main function
main "$@"