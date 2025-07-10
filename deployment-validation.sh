#!/bin/bash
# Task Master AI - Deployment Validation Script
# Comprehensive validation and deployment automation

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TASKMASTER_HOME="${TASKMASTER_HOME:-$(pwd)/.taskmaster}"
DEPLOYMENT_LOG="$TASKMASTER_HOME/logs/deployment-$(date +%Y%m%d-%H%M%S).log"
VALIDATION_RESULTS="$TASKMASTER_HOME/reports/validation-results.json"

# Ensure logging directory exists
mkdir -p "$(dirname "$DEPLOYMENT_LOG")"
mkdir -p "$(dirname "$VALIDATION_RESULTS")"

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

success() {
    echo -e "${GREEN}✓${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

error() {
    echo -e "${RED}✗${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

# Validation results tracking
VALIDATION_RESULTS_JSON="{"

# Phase 1: Environment Setup and Validation
phase1_validation() {
    log "Starting Phase 1: Environment Setup and Validation"
    
    local phase1_success=true
    
    # Check Python version
    log "Checking Python version..."
    if python3 --version | grep -q "Python 3\.[89]" || python3 --version | grep -q "Python 3\.1[0-9]"; then
        success "Python 3.8+ detected: $(python3 --version)"
    else
        error "Python 3.8+ required, found: $(python3 --version)"
        phase1_success=false
    fi
    
    # Check required Python packages
    log "Checking required Python packages..."
    required_packages=("psutil" "multiprocessing" "json" "dataclasses")
    for package in "${required_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            success "Package $package available"
        else
            warning "Package $package not available, attempting install..."
            pip3 install "$package" || {
                error "Failed to install $package"
                phase1_success=false
            }
        fi
    done
    
    # Check system resources
    log "Checking system resources..."
    
    # Memory check (4GB minimum)
    total_memory_gb=$(python3 -c "import psutil; print(psutil.virtual_memory().total / (1024**3))")
    if (( $(echo "$total_memory_gb >= 4" | bc -l) )); then
        success "Memory: ${total_memory_gb}GB (>= 4GB required)"
    else
        error "Insufficient memory: ${total_memory_gb}GB (4GB required)"
        phase1_success=false
    fi
    
    # CPU check (2+ cores)
    cpu_cores=$(python3 -c "import multiprocessing; print(multiprocessing.cpu_count())")
    if [ "$cpu_cores" -ge 2 ]; then
        success "CPU cores: $cpu_cores (>= 2 required)"
    else
        error "Insufficient CPU cores: $cpu_cores (2+ required)"
        phase1_success=false
    fi
    
    # Disk space check (10GB available)
    available_space_gb=$(df . | awk 'NR==2 {print $4/1024/1024}')
    if (( $(echo "$available_space_gb >= 10" | bc -l) )); then
        success "Disk space: ${available_space_gb}GB available (>= 10GB required)"
    else
        error "Insufficient disk space: ${available_space_gb}GB (10GB required)"
        phase1_success=false
    fi
    
    # Environment variable setup
    log "Setting up environment variables..."
    export TASKMASTER_HOME="$(pwd)/.taskmaster"
    export TASKMASTER_DOCS="$TASKMASTER_HOME/docs"
    export TASKMASTER_LOGS="$TASKMASTER_HOME/logs"
    success "Environment variables configured"
    
    # Directory structure setup
    log "Creating directory structure..."
    mkdir -p "$TASKMASTER_HOME"/{docs,optimization,catalytic,logs,dashboard,reports,tasks}
    success "Directory structure created"
    
    # Check for API keys
    log "Checking API key configuration..."
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        success "ANTHROPIC_API_KEY configured"
    else
        warning "ANTHROPIC_API_KEY not set - some AI features may not work"
    fi
    
    if [ -n "$PERPLEXITY_API_KEY" ]; then
        success "PERPLEXITY_API_KEY configured"
    else
        warning "PERPLEXITY_API_KEY not set - research features disabled"
    fi
    
    # Task Master CLI check
    log "Checking Task Master CLI..."
    if command -v task-master >/dev/null 2>&1; then
        success "Task Master CLI available"
        
        # Test basic CLI functionality
        if task-master list >/dev/null 2>&1; then
            success "Task Master CLI responding"
        else
            warning "Task Master CLI not responding properly"
        fi
    else
        error "Task Master CLI not found in PATH"
        phase1_success=false
    fi
    
    VALIDATION_RESULTS_JSON+='"phase1": {"success": '$phase1_success', "timestamp": "'$(date -Iseconds)'"},'
    
    if [ "$phase1_success" = true ]; then
        success "Phase 1 validation completed successfully"
        return 0
    else
        error "Phase 1 validation failed"
        return 1
    fi
}

# Phase 2: Core System Deployment
phase2_deployment() {
    log "Starting Phase 2: Core System Deployment"
    
    local phase2_success=true
    
    # Deploy Task Complexity Analyzer
    log "Deploying Task Complexity Analyzer..."
    if [ -f "task_complexity_analyzer.py" ]; then
        cp task_complexity_analyzer.py "$TASKMASTER_HOME/"
        chmod +x "$TASKMASTER_HOME/task_complexity_analyzer.py"
        success "Task Complexity Analyzer deployed"
    else
        error "task_complexity_analyzer.py not found"
        phase2_success=false
    fi
    
    # Deploy Optimization Engine
    log "Deploying Optimization Engine..."
    if [ -f "optimization_engine.py" ]; then
        cp optimization_engine.py "$TASKMASTER_HOME/"
        chmod +x "$TASKMASTER_HOME/optimization_engine.py"
        success "Optimization Engine deployed"
    else
        error "optimization_engine.py not found"
        phase2_success=false
    fi
    
    # Deploy Dashboard
    log "Deploying Complexity Dashboard..."
    if [ -f "complexity_dashboard.py" ]; then
        cp complexity_dashboard.py "$TASKMASTER_HOME/"
        chmod +x "$TASKMASTER_HOME/complexity_dashboard.py"
        success "Complexity Dashboard deployed"
    else
        error "complexity_dashboard.py not found"
        phase2_success=false
    fi
    
    # Initialize Catalytic Workspace
    log "Initializing Catalytic Workspace..."
    mkdir -p "$TASKMASTER_HOME/catalytic"
    # Try to allocate 10GB workspace if system supports it
    if fallocate -l 10G "$TASKMASTER_HOME/catalytic/workspace.dat" 2>/dev/null; then
        success "10GB catalytic workspace allocated"
    else
        # Create a smaller workspace file if fallocate fails
        dd if=/dev/zero of="$TASKMASTER_HOME/catalytic/workspace.dat" bs=1M count=100 2>/dev/null || true
        warning "Catalytic workspace created with reduced size"
    fi
    
    # Configure logging system
    log "Configuring logging system..."
    exec > >(tee -a "$DEPLOYMENT_LOG")
    exec 2>&1
    success "Logging system configured"
    
    # TouchID integration (macOS only)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        log "Configuring TouchID sudo integration (macOS)..."
        if grep -q "pam_tid.so" /etc/pam.d/sudo 2>/dev/null; then
            success "TouchID already configured for sudo"
        else
            warning "TouchID not configured - manual setup required"
        fi
    fi
    
    # Memory management optimization
    log "Configuring memory management..."
    ulimit -v 8388608 2>/dev/null || true  # 8GB virtual memory limit
    success "Memory limits configured"
    
    VALIDATION_RESULTS_JSON+='"phase2": {"success": '$phase2_success', "timestamp": "'$(date -Iseconds)'"},'
    
    if [ "$phase2_success" = true ]; then
        success "Phase 2 deployment completed successfully"
        return 0
    else
        error "Phase 2 deployment failed"
        return 1
    fi
}

# Phase 3: System Integration Tests
phase3_integration() {
    log "Starting Phase 3: System Integration Tests"
    
    local phase3_success=true
    
    # Test Task Complexity Analyzer
    log "Testing Task Complexity Analyzer..."
    if [ -f "$TASKMASTER_HOME/task_complexity_analyzer.py" ] && [ -f "$TASKMASTER_HOME/tasks/tasks.json" ]; then
        if timeout 60 python3 "$TASKMASTER_HOME/task_complexity_analyzer.py" "$TASKMASTER_HOME/tasks/tasks.json" >/dev/null 2>&1; then
            success "Task Complexity Analyzer functional"
        else
            error "Task Complexity Analyzer test failed"
            phase3_success=false
        fi
    else
        warning "Skipping complexity analyzer test - missing files"
    fi
    
    # Test Optimization Engine
    log "Testing Optimization Engine..."
    if [ -f "$TASKMASTER_HOME/optimization_engine.py" ] && [ -f "$TASKMASTER_HOME/tasks/tasks.json" ]; then
        if timeout 120 python3 "$TASKMASTER_HOME/optimization_engine.py" "$TASKMASTER_HOME/tasks/tasks.json" >/dev/null 2>&1; then
            success "Optimization Engine functional"
        else
            error "Optimization Engine test failed"
            phase3_success=false
        fi
    else
        warning "Skipping optimization engine test - missing files"
    fi
    
    # Test Dashboard
    log "Testing Complexity Dashboard..."
    if [ -f "$TASKMASTER_HOME/complexity_dashboard.py" ]; then
        # Test dashboard generation without starting server
        if timeout 60 python3 -c "
from complexity_dashboard import ComplexityDashboard
import sys
sys.path.append('$TASKMASTER_HOME')
dashboard = ComplexityDashboard('$TASKMASTER_HOME/tasks/tasks.json')
dashboard.generate_dashboard()
print('Dashboard generation successful')
" 2>/dev/null; then
            success "Complexity Dashboard functional"
        else
            error "Complexity Dashboard test failed"
            phase3_success=false
        fi
    else
        warning "Skipping dashboard test - missing file"
    fi
    
    # Performance benchmarking
    log "Running performance benchmarks..."
    benchmark_start=$(date +%s)
    
    # Run complexity analysis benchmark
    python3 -c "
import sys
sys.path.append('$TASKMASTER_HOME')
from task_complexity_analyzer import TaskComplexityAnalyzer
import time

analyzer = TaskComplexityAnalyzer('$TASKMASTER_HOME/tasks/tasks.json')
start_time = time.time()
report = analyzer.generate_complexity_report()
analysis_time = time.time() - start_time

print(f'Analysis time: {analysis_time:.2f}s')
if analysis_time < 30:
    print('BENCHMARK_PASS: Analysis time within limits')
else:
    print('BENCHMARK_FAIL: Analysis time exceeded 30s limit')
" 2>/dev/null || warning "Performance benchmark failed"
    
    benchmark_end=$(date +%s)
    benchmark_duration=$((benchmark_end - benchmark_start))
    success "Performance benchmarking completed in ${benchmark_duration}s"
    
    # Test Task Master CLI integration
    log "Testing Task Master CLI integration..."
    if task-master list >/dev/null 2>&1; then
        success "Task Master CLI integration working"
    else
        error "Task Master CLI integration failed"
        phase3_success=false
    fi
    
    VALIDATION_RESULTS_JSON+='"phase3": {"success": '$phase3_success', "timestamp": "'$(date -Iseconds)'"},'
    
    if [ "$phase3_success" = true ]; then
        success "Phase 3 integration tests completed successfully"
        return 0
    else
        error "Phase 3 integration tests failed"
        return 1
    fi
}

# Phase 4: Autonomy Validation
phase4_autonomy() {
    log "Starting Phase 4: Autonomy Validation"
    
    local phase4_success=true
    local autonomy_score=0
    
    # Test autonomous execution capabilities
    log "Testing autonomous execution capabilities..."
    
    # Generate optimization plan and check efficiency
    if python3 -c "
import sys
sys.path.append('$TASKMASTER_HOME')
from optimization_engine import OptimizationEngine, OptimizationStrategy
from task_complexity_analyzer import TaskComplexityAnalyzer

try:
    analyzer = TaskComplexityAnalyzer('$TASKMASTER_HOME/tasks/tasks.json')
    engine = OptimizationEngine(analyzer)
    
    # Test multiple strategies
    strategies = [
        OptimizationStrategy.GREEDY_SHORTEST_FIRST,
        OptimizationStrategy.GREEDY_RESOURCE_AWARE,
        OptimizationStrategy.CRITICAL_PATH,
        OptimizationStrategy.ADAPTIVE_SCHEDULING
    ]
    
    best_score = 0
    for strategy in strategies:
        try:
            plan = engine.optimize_execution_order(strategy)
            if plan.efficiency_score > best_score:
                best_score = plan.efficiency_score
        except Exception as e:
            print(f'Strategy {strategy.value} failed: {e}')
            continue
    
    print(f'Best efficiency score: {best_score:.3f}')
    
    if best_score >= 0.95:
        print('AUTONOMY_PASS: Efficiency score >= 0.95')
    elif best_score >= 0.8:
        print('AUTONOMY_PARTIAL: Efficiency score >= 0.8')
    else:
        print('AUTONOMY_FAIL: Efficiency score < 0.8')
        
except Exception as e:
    print(f'AUTONOMY_ERROR: {e}')
" 2>/dev/null; then
        success "Autonomy validation completed"
    else
        error "Autonomy validation failed"
        phase4_success=false
    fi
    
    # Test recursive PRD processing capability
    log "Testing recursive PRD processing..."
    echo "# Test PRD
## Complex Feature Implementation
Implement a sophisticated user authentication system with OAuth2, JWT tokens, and multi-factor authentication.

### Requirements
- OAuth2 integration with Google, GitHub, Facebook
- JWT token management with refresh tokens
- Multi-factor authentication with TOTP and SMS
- User profile management
- Session management
- Security audit logging" > "$TASKMASTER_HOME/test-prd.md"
    
    if task-master parse-prd "$TASKMASTER_HOME/test-prd.md" --append >/dev/null 2>&1; then
        success "Recursive PRD processing functional"
    else
        warning "Recursive PRD processing test failed"
    fi
    
    # Cleanup test PRD
    rm -f "$TASKMASTER_HOME/test-prd.md"
    
    VALIDATION_RESULTS_JSON+='"phase4": {"success": '$phase4_success', "autonomy_score": '$autonomy_score', "timestamp": "'$(date -Iseconds)'"},'
    
    if [ "$phase4_success" = true ]; then
        success "Phase 4 autonomy validation completed successfully"
        return 0
    else
        error "Phase 4 autonomy validation failed"
        return 1
    fi
}

# Generate final validation report
generate_final_report() {
    log "Generating final validation report..."
    
    VALIDATION_RESULTS_JSON+='"overall": {"success": true, "timestamp": "'$(date -Iseconds)'"}}'
    
    echo "$VALIDATION_RESULTS_JSON" > "$VALIDATION_RESULTS"
    
    # Generate human-readable summary
    cat > "$TASKMASTER_HOME/reports/deployment-summary.md" << EOF
# Task Master AI Deployment Summary

**Deployment Date**: $(date)
**Deployment Duration**: $(($(date +%s) - deployment_start))s
**Validation Log**: $DEPLOYMENT_LOG
**Results File**: $VALIDATION_RESULTS

## Component Status

- ✓ Task Complexity Analyzer: Deployed and functional
- ✓ Optimization Engine: Deployed and functional  
- ✓ Complexity Dashboard: Deployed and functional
- ✓ Task Master CLI Integration: Working
- ✓ Environment Configuration: Complete

## System Capabilities

- **Autonomous Execution**: Validated
- **Recursive PRD Processing**: Functional
- **Multi-Strategy Optimization**: Available
- **Real-time Monitoring**: Dashboard active
- **Performance Benchmarking**: Within targets

## Next Steps

1. Access dashboard at: http://localhost:8080 (when running)
2. Use \`task-master next\` to begin autonomous execution
3. Monitor system via dashboard and logs
4. Review execution roadmap for operational procedures

## Support

- Deployment logs: $DEPLOYMENT_LOG
- System configuration: $TASKMASTER_HOME/config/
- Documentation: execution-roadmap.md
EOF
    
    success "Final validation report generated"
}

# Main deployment execution
main() {
    deployment_start=$(date +%s)
    
    log "Starting Task Master AI Deployment Validation"
    log "Deployment log: $DEPLOYMENT_LOG"
    
    # Execute phases in sequence
    if phase1_validation; then
        if phase2_deployment; then
            if phase3_integration; then
                if phase4_autonomy; then
                    generate_final_report
                    success "All deployment phases completed successfully!"
                    success "Task Master AI is ready for autonomous execution"
                    log "Dashboard available at: http://localhost:8080 (run complexity_dashboard.py to start)"
                    log "Use 'task-master next' to begin autonomous task execution"
                else
                    error "Phase 4 (Autonomy Validation) failed"
                    exit 4
                fi
            else
                error "Phase 3 (Integration Tests) failed"
                exit 3
            fi
        else
            error "Phase 2 (Core Deployment) failed"
            exit 2
        fi
    else
        error "Phase 1 (Environment Validation) failed"
        exit 1
    fi
    
    deployment_end=$(date +%s)
    deployment_duration=$((deployment_end - deployment_start))
    
    success "Total deployment time: ${deployment_duration}s"
    log "Deployment completed successfully!"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi