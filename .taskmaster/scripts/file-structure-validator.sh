#!/bin/bash
"""
File Structure Conformance Validator
Validates and fixes file structure according to task-master-instructions.md requirements
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
LOG_FILE=".taskmaster/logs/file-structure-validation-$(date +%Y%m%d_%H%M%S).log"

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

# Initialize compliance report
COMPLIANCE_REPORT=".taskmaster/reports/file-structure-compliance-$(date +%Y%m%d_%H%M%S).json"
mkdir -p .taskmaster/reports

cat > "$COMPLIANCE_REPORT" << 'EOF'
{
  "timestamp": "TIMESTAMP_PLACEHOLDER",
  "validation_type": "file_structure_conformance",
  "requirements_source": "task-master-instructions.md",
  "checks": {
    "environment_variables": {},
    "catalytic_workspace": {},
    "logging_setup": {},
    "touchid_sudo": {},
    "directory_structure": {},
    "file_permissions": {}
  },
  "summary": {
    "total_checks": 0,
    "passed": 0,
    "failed": 0,
    "warnings": 0,
    "overall_status": "unknown"
  },
  "auto_fixes_applied": []
}
EOF

# Replace timestamp
sed -i "" "s/TIMESTAMP_PLACEHOLDER/$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)/" "$COMPLIANCE_REPORT"

log "Starting file structure conformance validation"
print_status "INFO" "File Structure Conformance Validator Started"
print_status "INFO" "Compliance report: $COMPLIANCE_REPORT"

# Counter variables
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0
AUTO_FIXES=()

# Function to update compliance report
update_report() {
    local section=$1
    local check_name=$2
    local status=$3
    local details=$4
    
    # Use jq to update the JSON report if available, otherwise use sed
    if command -v jq >/dev/null 2>&1; then
        jq --arg section "$section" --arg check "$check_name" --arg status "$status" --arg details "$details" \
           '.checks[$section][$check] = {"status": $status, "details": $details}' \
           "$COMPLIANCE_REPORT" > "${COMPLIANCE_REPORT}.tmp" && mv "${COMPLIANCE_REPORT}.tmp" "$COMPLIANCE_REPORT"
    else
        # Fallback to manual JSON construction
        log "jq not available, using manual JSON updates"
    fi
}

# 1. Validate Environment Variables
print_status "INFO" "1. Validating Environment Variables"
TOTAL_CHECKS=$((TOTAL_CHECKS + 3))

# Check TASKMASTER_HOME
if [[ -n "${TASKMASTER_HOME:-}" ]]; then
    if [[ -d "$TASKMASTER_HOME" ]]; then
        print_status "PASS" "TASKMASTER_HOME is set and directory exists: $TASKMASTER_HOME"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        update_report "environment_variables" "TASKMASTER_HOME" "pass" "$TASKMASTER_HOME"
    else
        print_status "FAIL" "TASKMASTER_HOME directory does not exist: $TASKMASTER_HOME"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        update_report "environment_variables" "TASKMASTER_HOME" "fail" "Directory not found"
    fi
else
    print_status "WARN" "TASKMASTER_HOME not set, setting to current directory"
    export TASKMASTER_HOME="$(pwd)"
    echo "export TASKMASTER_HOME=\"$(pwd)\"" >> ~/.zshrc 2>/dev/null || echo "export TASKMASTER_HOME=\"$(pwd)\"" >> ~/.bashrc 2>/dev/null || true
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
    AUTO_FIXES+=("Set TASKMASTER_HOME to $(pwd)")
    update_report "environment_variables" "TASKMASTER_HOME" "auto_fixed" "$(pwd)"
fi

# Check TASKMASTER_DOCS
if [[ -n "${TASKMASTER_DOCS:-}" ]]; then
    if [[ -d "$TASKMASTER_DOCS" ]]; then
        print_status "PASS" "TASKMASTER_DOCS is set and directory exists: $TASKMASTER_DOCS"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        update_report "environment_variables" "TASKMASTER_DOCS" "pass" "$TASKMASTER_DOCS"
    else
        print_status "FAIL" "TASKMASTER_DOCS directory does not exist: $TASKMASTER_DOCS"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        update_report "environment_variables" "TASKMASTER_DOCS" "fail" "Directory not found"
    fi
else
    DOCS_DIR="${TASKMASTER_HOME:-$(pwd)}/.taskmaster/docs"
    mkdir -p "$DOCS_DIR"
    export TASKMASTER_DOCS="$DOCS_DIR"
    echo "export TASKMASTER_DOCS=\"$DOCS_DIR\"" >> ~/.zshrc 2>/dev/null || echo "export TASKMASTER_DOCS=\"$DOCS_DIR\"" >> ~/.bashrc 2>/dev/null || true
    print_status "WARN" "TASKMASTER_DOCS not set, created and set to: $DOCS_DIR"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
    AUTO_FIXES+=("Set TASKMASTER_DOCS to $DOCS_DIR")
    update_report "environment_variables" "TASKMASTER_DOCS" "auto_fixed" "$DOCS_DIR"
fi

# Check TASKMASTER_LOGS
if [[ -n "${TASKMASTER_LOGS:-}" ]]; then
    if [[ -d "$TASKMASTER_LOGS" ]]; then
        print_status "PASS" "TASKMASTER_LOGS is set and directory exists: $TASKMASTER_LOGS"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        update_report "environment_variables" "TASKMASTER_LOGS" "pass" "$TASKMASTER_LOGS"
    else
        print_status "FAIL" "TASKMASTER_LOGS directory does not exist: $TASKMASTER_LOGS"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        update_report "environment_variables" "TASKMASTER_LOGS" "fail" "Directory not found"
    fi
else
    LOGS_DIR="${TASKMASTER_HOME:-$(pwd)}/.taskmaster/logs"
    mkdir -p "$LOGS_DIR"
    export TASKMASTER_LOGS="$LOGS_DIR"
    echo "export TASKMASTER_LOGS=\"$LOGS_DIR\"" >> ~/.zshrc 2>/dev/null || echo "export TASKMASTER_LOGS=\"$LOGS_DIR\"" >> ~/.bashrc 2>/dev/null || true
    print_status "WARN" "TASKMASTER_LOGS not set, created and set to: $LOGS_DIR"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
    AUTO_FIXES+=("Set TASKMASTER_LOGS to $LOGS_DIR")
    update_report "environment_variables" "TASKMASTER_LOGS" "auto_fixed" "$LOGS_DIR"
fi

# 2. Verify Catalytic Workspace
print_status "INFO" "2. Validating Catalytic Workspace"
TOTAL_CHECKS=$((TOTAL_CHECKS + 2))

CATALYTIC_DIR=".taskmaster/catalytic"
if [[ -d "$CATALYTIC_DIR" ]]; then
    print_status "PASS" "Catalytic workspace directory exists: $CATALYTIC_DIR"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    
    # Check approximate size (should be around 10GB capacity)
    if [[ -f "$CATALYTIC_DIR/workspace.dat" ]]; then
        SIZE=$(du -m "$CATALYTIC_DIR" 2>/dev/null | cut -f1 || echo "0")
        if [[ $SIZE -gt 1000 ]]; then  # At least 1GB allocated
            print_status "PASS" "Catalytic workspace has adequate allocation: ${SIZE}MB"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
            update_report "catalytic_workspace" "size_allocation" "pass" "${SIZE}MB"
        else
            print_status "WARN" "Catalytic workspace size seems small: ${SIZE}MB"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
            update_report "catalytic_workspace" "size_allocation" "warning" "${SIZE}MB"
        fi
    else
        print_status "WARN" "Catalytic workspace data file not found, creating placeholder"
        # Create a placeholder file to indicate workspace initialization
        echo "# Catalytic Workspace Initialized $(date)" > "$CATALYTIC_DIR/workspace.dat"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
        AUTO_FIXES+=("Created catalytic workspace placeholder")
        update_report "catalytic_workspace" "size_allocation" "auto_fixed" "Placeholder created"
    fi
    update_report "catalytic_workspace" "directory_exists" "pass" "$CATALYTIC_DIR"
else
    print_status "FAIL" "Catalytic workspace directory missing: $CATALYTIC_DIR"
    mkdir -p "$CATALYTIC_DIR"
    echo "# Catalytic Workspace Initialized $(date)" > "$CATALYTIC_DIR/workspace.dat"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    AUTO_FIXES+=("Created catalytic workspace directory")
    update_report "catalytic_workspace" "directory_exists" "auto_fixed" "Created $CATALYTIC_DIR"
fi

# 3. Audit Logging Setup
print_status "INFO" "3. Validating Comprehensive Logging Setup"
TOTAL_CHECKS=$((TOTAL_CHECKS + 3))

# Check if logs directory exists and has recent files
LOGS_DIR="${TASKMASTER_LOGS:-$(pwd)/.taskmaster/logs}"
if [[ -d "$LOGS_DIR" ]]; then
    print_status "PASS" "Logs directory exists: $LOGS_DIR"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    update_report "logging_setup" "directory_exists" "pass" "$LOGS_DIR"
    
    # Check for recent log files
    LOG_COUNT=$(find "$LOGS_DIR" -name "*.log" -mtime -1 2>/dev/null | wc -l || echo "0")
    if [[ $LOG_COUNT -gt 0 ]]; then
        print_status "PASS" "Recent log files found: $LOG_COUNT files"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        update_report "logging_setup" "recent_logs" "pass" "$LOG_COUNT files"
    else
        print_status "WARN" "No recent log files found"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
        update_report "logging_setup" "recent_logs" "warning" "No recent logs"
    fi
    
    # Check log rotation setup
    if [[ -f "$LOGS_DIR/logrotate.conf" ]] || [[ -f "/etc/logrotate.d/taskmaster" ]]; then
        print_status "PASS" "Log rotation configured"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        update_report "logging_setup" "rotation_configured" "pass" "Configured"
    else
        print_status "WARN" "Log rotation not explicitly configured"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
        update_report "logging_setup" "rotation_configured" "warning" "Not configured"
    fi
else
    print_status "FAIL" "Logs directory missing: $LOGS_DIR"
    mkdir -p "$LOGS_DIR"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    AUTO_FIXES+=("Created logs directory")
    update_report "logging_setup" "directory_exists" "auto_fixed" "Created $LOGS_DIR"
fi

# 4. Configure TouchID Sudo Authentication
print_status "INFO" "4. Validating TouchID Sudo Configuration"
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

# Check if TouchID sudo is configured
if grep -q "pam_tid.so" /etc/pam.d/sudo 2>/dev/null; then
    print_status "PASS" "TouchID sudo authentication is configured"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    update_report "touchid_sudo" "configured" "pass" "TouchID enabled in /etc/pam.d/sudo"
else
    print_status "WARN" "TouchID sudo authentication not configured"
    print_status "INFO" "To enable TouchID sudo, run: sudo sed -i '' '2i\\
auth       sufficient     pam_tid.so\\
' /etc/pam.d/sudo"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
    update_report "touchid_sudo" "configured" "warning" "Manual configuration required"
fi

# 5. Validate Directory Structure
print_status "INFO" "5. Validating Directory Structure Hierarchy"
TOTAL_CHECKS=$((TOTAL_CHECKS + 8))

REQUIRED_DIRS=(
    ".taskmaster"
    ".taskmaster/docs"
    ".taskmaster/tasks"
    ".taskmaster/logs" 
    ".taskmaster/reports"
    ".taskmaster/scripts"
    ".taskmaster/catalytic"
    ".taskmaster/optimization"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        print_status "PASS" "Required directory exists: $dir"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        print_status "FAIL" "Required directory missing: $dir"
        mkdir -p "$dir"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        AUTO_FIXES+=("Created directory $dir")
    fi
done

# 6. Validate File Permissions
print_status "INFO" "6. Validating File Permissions"
TOTAL_CHECKS=$((TOTAL_CHECKS + 2))

# Check script permissions
SCRIPT_COUNT=0
EXECUTABLE_COUNT=0
for script in .taskmaster/scripts/*.sh .taskmaster/scripts/*.py; do
    if [[ -f "$script" ]]; then
        SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
        if [[ -x "$script" ]]; then
            EXECUTABLE_COUNT=$((EXECUTABLE_COUNT + 1))
        else
            chmod +x "$script"
            AUTO_FIXES+=("Made $script executable")
        fi
    fi
done

if [[ $SCRIPT_COUNT -eq $EXECUTABLE_COUNT ]] && [[ $SCRIPT_COUNT -gt 0 ]]; then
    print_status "PASS" "All scripts are executable ($EXECUTABLE_COUNT/$SCRIPT_COUNT)"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    update_report "file_permissions" "scripts_executable" "pass" "$EXECUTABLE_COUNT/$SCRIPT_COUNT"
else
    print_status "WARN" "Some scripts needed permission fixes ($EXECUTABLE_COUNT/$SCRIPT_COUNT)"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
    update_report "file_permissions" "scripts_executable" "auto_fixed" "$EXECUTABLE_COUNT/$SCRIPT_COUNT"
fi

# Check directory permissions
if [[ -w ".taskmaster" ]] && [[ -r ".taskmaster" ]]; then
    print_status "PASS" "Taskmaster directory has correct permissions"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    update_report "file_permissions" "directory_access" "pass" "Read/write access confirmed"
else
    print_status "FAIL" "Taskmaster directory permission issues"
    chmod 755 .taskmaster
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    AUTO_FIXES+=("Fixed .taskmaster directory permissions")
    update_report "file_permissions" "directory_access" "auto_fixed" "Permissions corrected"
fi

# Generate Final Summary
print_status "INFO" "Generating Final Compliance Summary"

# Update summary in report
if command -v jq >/dev/null 2>&1; then
    jq --argjson total "$TOTAL_CHECKS" \
       --argjson passed "$PASSED_CHECKS" \
       --argjson failed "$FAILED_CHECKS" \
       --argjson warnings "$WARNING_CHECKS" \
       --arg overall_status "$(if [[ $FAILED_CHECKS -eq 0 ]]; then echo "compliant"; else echo "non_compliant"; fi)" \
       '.summary.total_checks = $total | .summary.passed = $passed | .summary.failed = $failed | .summary.warnings = $warnings | .summary.overall_status = $overall_status' \
       "$COMPLIANCE_REPORT" > "${COMPLIANCE_REPORT}.tmp" && mv "${COMPLIANCE_REPORT}.tmp" "$COMPLIANCE_REPORT"
fi

# Add auto-fixes to report
if [[ ${#AUTO_FIXES[@]} -gt 0 ]]; then
    printf -v auto_fixes_json '%s\n' "${AUTO_FIXES[@]}" | jq -R . | jq -s . > /tmp/auto_fixes.json
    if command -v jq >/dev/null 2>&1; then
        jq --slurpfile fixes /tmp/auto_fixes.json '.auto_fixes_applied = $fixes[0]' \
           "$COMPLIANCE_REPORT" > "${COMPLIANCE_REPORT}.tmp" && mv "${COMPLIANCE_REPORT}.tmp" "$COMPLIANCE_REPORT"
    fi
    rm -f /tmp/auto_fixes.json
fi

# Final Report
echo ""
print_status "INFO" "═══════════════════════════════════════════════════════════"
print_status "INFO" "FILE STRUCTURE CONFORMANCE VALIDATION COMPLETE"
print_status "INFO" "═══════════════════════════════════════════════════════════"

echo -e "${BLUE}📊 VALIDATION SUMMARY:${NC}"
echo "   Total Checks: $TOTAL_CHECKS"
echo "   ✅ Passed: $PASSED_CHECKS"
echo "   ❌ Failed: $FAILED_CHECKS"
echo "   ⚠️  Warnings: $WARNING_CHECKS"
echo "   🔧 Auto-fixes Applied: ${#AUTO_FIXES[@]}"

if [[ $FAILED_CHECKS -eq 0 ]]; then
    print_status "PASS" "OVERALL STATUS: COMPLIANT ✅"
    echo ""
    print_status "INFO" "All file structure requirements are met or have been auto-fixed."
    EXIT_CODE=0
else
    print_status "FAIL" "OVERALL STATUS: NON-COMPLIANT ❌"
    echo ""
    print_status "WARN" "Some issues require manual intervention."
    EXIT_CODE=1
fi

echo ""
echo -e "${BLUE}📋 AUTO-FIXES APPLIED:${NC}"
for fix in "${AUTO_FIXES[@]}"; do
    echo "   🔧 $fix"
done

echo ""
echo -e "${BLUE}📄 REPORTS GENERATED:${NC}"
echo "   📊 Compliance Report: $COMPLIANCE_REPORT"
echo "   📝 Detailed Log: $LOG_FILE"

log "File structure conformance validation completed with exit code: $EXIT_CODE"
exit $EXIT_CODE