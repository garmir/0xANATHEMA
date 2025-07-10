#!/bin/bash
"""
Deployment Validation Script for Task-Master System
Validates all components for production deployment readiness
"""

# Set error handling
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOG_FILE=".taskmaster/logs/deployment-validation-$(date +%Y%m%d_%H%M%S).log"

# Ensure logs directory exists
mkdir -p .taskmaster/logs

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "PASS")
            echo -e "${GREEN}✅ $message${NC}" | tee -a "$LOG_FILE"
            ;;
        "FAIL")
            echo -e "${RED}❌ $message${NC}" | tee -a "$LOG_FILE"
            ;;
        "WARN")
            echo -e "${YELLOW}⚠️  $message${NC}" | tee -a "$LOG_FILE"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ️  $message${NC}" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Initialize validation report
VALIDATION_REPORT=".taskmaster/reports/deployment-validation-$(date +%Y%m%d_%H%M%S).json"
mkdir -p .taskmaster/reports

cat > "$VALIDATION_REPORT" << 'EOF'
{
  "timestamp": "TIMESTAMP_PLACEHOLDER",
  "validation_type": "deployment_readiness",
  "version": "1.0.0",
  "phases": {
    "system_integrity": {},
    "component_validation": {},
    "performance_testing": {},
    "integration_testing": {},
    "security_validation": {}
  },
  "summary": {
    "total_checks": 0,
    "passed": 0,
    "failed": 0,
    "warnings": 0,
    "deployment_ready": false
  }
}
EOF

# Replace timestamp
sed -i "" "s/TIMESTAMP_PLACEHOLDER/$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)/" "$VALIDATION_REPORT"

log "Starting Task-Master deployment validation"
print_status "INFO" "Task-Master Deployment Validation Started"
print_status "INFO" "Validation report: $VALIDATION_REPORT"

# Counter variables
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

echo ""
print_status "INFO" "═══════════════════════════════════════════════════════════"
print_status "INFO" "PHASE 1: SYSTEM INTEGRITY VALIDATION"
print_status "INFO" "═══════════════════════════════════════════════════════════"

# 1. Validate Core Task Completion
print_status "INFO" "1.1 Validating Core Task Completion Status"
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

if command -v task-master >/dev/null 2>&1; then
    TASK_STATUS=$(task-master list 2>/dev/null | grep "Tasks Progress" | grep -o "100%" || echo "incomplete")
    if [[ "$TASK_STATUS" == "100%" ]]; then
        print_status "PASS" "All core tasks completed (100%)"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        print_status "FAIL" "Core tasks not 100% complete: $TASK_STATUS"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
else
    print_status "FAIL" "task-master command not available"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# 2. Validate Critical Component Files
print_status "INFO" "1.2 Validating Critical Component Files"
CRITICAL_FILES=(
    ".taskmaster/scripts/space-complexity-validator.py"
    ".taskmaster/scripts/task-complexity-analyzer.py"
    ".taskmaster/scripts/file-structure-validator.sh"
    ".taskmaster/docs/execution-roadmap.md"
    ".taskmaster/tasks/tasks.json"
)

for file in "${CRITICAL_FILES[@]}"; do
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [[ -f "$file" ]]; then
        print_status "PASS" "Critical file exists: $file"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        print_status "FAIL" "Critical file missing: $file"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
done

# 3. Validate Configuration Files
print_status "INFO" "1.3 Validating Configuration Files"
CONFIG_FILES=(
    ".taskmaster/config.json"
    ".taskmaster/scripts/requirements.txt"
)

for file in "${CONFIG_FILES[@]}"; do
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [[ -f "$file" ]]; then
        print_status "PASS" "Configuration file exists: $file"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        print_status "WARN" "Configuration file missing: $file"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
    fi
done

echo ""
print_status "INFO" "═══════════════════════════════════════════════════════════"
print_status "INFO" "PHASE 2: COMPONENT VALIDATION"
print_status "INFO" "═══════════════════════════════════════════════════════════"

# 4. Test Space Complexity Validator
print_status "INFO" "2.1 Testing Space Complexity Validator"
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

if [[ -f ".taskmaster/scripts/space-complexity-validator.py" ]]; then
    # Check if script is executable
    if [[ -x ".taskmaster/scripts/space-complexity-validator.py" ]]; then
        print_status "PASS" "Space complexity validator is executable"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        print_status "WARN" "Space complexity validator not executable, fixing..."
        chmod +x .taskmaster/scripts/space-complexity-validator.py
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
    fi
else
    print_status "FAIL" "Space complexity validator missing"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# 5. Test Task Complexity Analyzer
print_status "INFO" "2.2 Testing Task Complexity Analyzer"
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

if [[ -f ".taskmaster/scripts/task-complexity-analyzer.py" ]]; then
    if [[ -x ".taskmaster/scripts/task-complexity-analyzer.py" ]]; then
        print_status "PASS" "Task complexity analyzer is executable"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        print_status "WARN" "Task complexity analyzer not executable, fixing..."
        chmod +x .taskmaster/scripts/task-complexity-analyzer.py
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
    fi
else
    print_status "FAIL" "Task complexity analyzer missing"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# 6. Test File Structure Validator
print_status "INFO" "2.3 Testing File Structure Validator"
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

if [[ -f ".taskmaster/scripts/file-structure-validator.sh" ]]; then
    if [[ -x ".taskmaster/scripts/file-structure-validator.sh" ]]; then
        print_status "PASS" "File structure validator is executable"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        print_status "WARN" "File structure validator not executable, fixing..."
        chmod +x .taskmaster/scripts/file-structure-validator.sh
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
    fi
else
    print_status "FAIL" "File structure validator missing"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

echo ""
print_status "INFO" "═══════════════════════════════════════════════════════════"
print_status "INFO" "PHASE 3: PERFORMANCE TESTING"
print_status "INFO" "═══════════════════════════════════════════════════════════"

# 7. Test Python Dependencies
print_status "INFO" "3.1 Testing Python Dependencies"
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

if [[ -d ".venv" ]]; then
    print_status "PASS" "Python virtual environment exists"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    
    # Test if dependencies are installed
    if source .venv/bin/activate && python -c "import numpy, matplotlib, scipy, psutil" 2>/dev/null; then
        print_status "PASS" "All Python dependencies available"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        print_status "WARN" "Some Python dependencies missing"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
else
    print_status "WARN" "Python virtual environment not found"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
fi

# 8. Test System Resources
print_status "INFO" "3.2 Testing System Resources"
TOTAL_CHECKS=$((TOTAL_CHECKS + 3))

# Check available memory
AVAILABLE_MEMORY=$(python3 -c "import psutil; print(int(psutil.virtual_memory().available / 1024 / 1024))" 2>/dev/null || echo "0")
if [[ $AVAILABLE_MEMORY -gt 1000 ]]; then
    print_status "PASS" "Sufficient memory available: ${AVAILABLE_MEMORY}MB"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    print_status "WARN" "Limited memory available: ${AVAILABLE_MEMORY}MB"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
fi

# Check CPU cores
CPU_CORES=$(python3 -c "import psutil; print(psutil.cpu_count())" 2>/dev/null || echo "1")
if [[ $CPU_CORES -ge 2 ]]; then
    print_status "PASS" "Sufficient CPU cores: $CPU_CORES"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    print_status "WARN" "Limited CPU cores: $CPU_CORES"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
fi

# Check disk space
DISK_SPACE=$(df . | tail -1 | awk '{print $4}' | sed 's/K$//' 2>/dev/null || echo "0")
DISK_SPACE_MB=$((DISK_SPACE / 1024))
if [[ $DISK_SPACE_MB -gt 1000 ]]; then
    print_status "PASS" "Sufficient disk space: ${DISK_SPACE_MB}MB"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    print_status "WARN" "Limited disk space: ${DISK_SPACE_MB}MB"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
fi

echo ""
print_status "INFO" "═══════════════════════════════════════════════════════════"
print_status "INFO" "PHASE 4: INTEGRATION TESTING"
print_status "INFO" "═══════════════════════════════════════════════════════════"

# 9. Test Task-Master Integration
print_status "INFO" "4.1 Testing Task-Master CLI Integration"
TOTAL_CHECKS=$((TOTAL_CHECKS + 2))

# Test task-master list command
if task-master list >/dev/null 2>&1; then
    print_status "PASS" "task-master list command functional"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    print_status "FAIL" "task-master list command failed"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Test task-master next command
if task-master next >/dev/null 2>&1; then
    print_status "PASS" "task-master next command functional"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    print_status "WARN" "task-master next command issues (may be expected if all tasks complete)"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
fi

echo ""
print_status "INFO" "═══════════════════════════════════════════════════════════"
print_status "INFO" "PHASE 5: SECURITY VALIDATION"
print_status "INFO" "═══════════════════════════════════════════════════════════"

# 10. Test File Permissions
print_status "INFO" "5.1 Testing File Permissions and Security"
TOTAL_CHECKS=$((TOTAL_CHECKS + 2))

# Check .taskmaster directory permissions
if [[ -d ".taskmaster" ]] && [[ -r ".taskmaster" ]] && [[ -w ".taskmaster" ]]; then
    print_status "PASS" ".taskmaster directory has correct permissions"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    print_status "FAIL" ".taskmaster directory permission issues"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check for secure file practices
INSECURE_FILES=$(find .taskmaster -name "*.py" -o -name "*.sh" | xargs grep -l "password\|secret\|key" 2>/dev/null | wc -l || echo "0")
if [[ $INSECURE_FILES -eq 0 ]]; then
    print_status "PASS" "No obvious security issues in scripts"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    print_status "WARN" "Potential security issues found in $INSECURE_FILES files"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
fi

# Generate Final Summary
print_status "INFO" "Generating Final Validation Summary"

# Calculate deployment readiness
DEPLOYMENT_READY=false
SUCCESS_RATE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
if [[ $SUCCESS_RATE -ge 85 ]] && [[ $FAILED_CHECKS -eq 0 ]]; then
    DEPLOYMENT_READY=true
fi

# Update summary in report
if command -v jq >/dev/null 2>&1; then
    jq --argjson total "$TOTAL_CHECKS" \
       --argjson passed "$PASSED_CHECKS" \
       --argjson failed "$FAILED_CHECKS" \
       --argjson warnings "$WARNING_CHECKS" \
       --argjson ready "$DEPLOYMENT_READY" \
       '.summary.total_checks = $total | .summary.passed = $passed | .summary.failed = $failed | .summary.warnings = $warnings | .summary.deployment_ready = $ready' \
       "$VALIDATION_REPORT" > "${VALIDATION_REPORT}.tmp" && mv "${VALIDATION_REPORT}.tmp" "$VALIDATION_REPORT"
fi

# Final Report
echo ""
print_status "INFO" "═══════════════════════════════════════════════════════════"
print_status "INFO" "DEPLOYMENT VALIDATION COMPLETE"
print_status "INFO" "═══════════════════════════════════════════════════════════"

echo -e "${BLUE}📊 VALIDATION SUMMARY:${NC}"
echo "   Total Checks: $TOTAL_CHECKS"
echo "   ✅ Passed: $PASSED_CHECKS"
echo "   ❌ Failed: $FAILED_CHECKS"
echo "   ⚠️  Warnings: $WARNING_CHECKS"
echo "   📈 Success Rate: ${SUCCESS_RATE}%"

if [[ $DEPLOYMENT_READY == true ]]; then
    print_status "PASS" "DEPLOYMENT STATUS: READY FOR PRODUCTION ✅"
    echo ""
    print_status "INFO" "System meets all deployment criteria and is ready for production use."
    EXIT_CODE=0
else
    print_status "FAIL" "DEPLOYMENT STATUS: NOT READY ❌"
    echo ""
    print_status "WARN" "System has issues that must be resolved before production deployment."
    EXIT_CODE=1
fi

echo ""
echo -e "${BLUE}📄 REPORTS GENERATED:${NC}"
echo "   📊 Validation Report: $VALIDATION_REPORT"
echo "   📝 Detailed Log: $LOG_FILE"

log "Deployment validation completed with exit code: $EXIT_CODE"
exit $EXIT_CODE