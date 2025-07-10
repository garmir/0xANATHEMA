# LABRYS Operability Report

**Generated:** 2025-07-09T18:22:10.383906
**Status:** OPERATIONAL
**Score:** 85.7%

## Summary
- **Total Checks:** 7
- **Passed:** 6
- **Failed:** 1
- **Critical Failures:** 0

## Check Results
- **python_environment**: ✅ PASS - Python 3.13 OK
- **labrys_imports**: ✅ PASS - LABRYS core modules imported successfully
- **system_initialization**: ✅ PASS - System initialized successfully
- **basic_functionality**: ✅ PASS - Basic functionality working
- **self_test_capability**: ✅ PASS - Self-test capability available
- **introspection_capability**: ✅ PASS - Introspection capability available
- **api_integration**: ❌ FAIL - API key not configured

## Detailed Results
```json
{
  "operational_status": "OPERATIONAL",
  "operability_score": 85.71428571428571,
  "total_checks": 7,
  "passed_checks": 6,
  "failed_checks": 1,
  "critical_failures": 0,
  "check_results": [
    {
      "check_name": "python_environment",
      "passed": true,
      "message": "Python 3.13 OK",
      "severity": "critical",
      "fix_attempted": false,
      "fix_successful": false,
      "timestamp": "2025-07-09T18:22:10.325745"
    },
    {
      "check_name": "labrys_imports",
      "passed": true,
      "message": "LABRYS core modules imported successfully",
      "severity": "critical",
      "fix_attempted": false,
      "fix_successful": false,
      "timestamp": "2025-07-09T18:22:10.341320"
    },
    {
      "check_name": "system_initialization",
      "passed": true,
      "message": "System initialized successfully",
      "severity": "high",
      "fix_attempted": false,
      "fix_successful": false,
      "timestamp": "2025-07-09T18:22:10.362649"
    },
    {
      "check_name": "basic_functionality",
      "passed": true,
      "message": "Basic functionality working",
      "severity": "high",
      "fix_attempted": false,
      "fix_successful": false,
      "timestamp": "2025-07-09T18:22:10.381156"
    },
    {
      "check_name": "self_test_capability",
      "passed": true,
      "message": "Self-test capability available",
      "severity": "medium",
      "fix_attempted": false,
      "fix_successful": false,
      "timestamp": "2025-07-09T18:22:10.382794"
    },
    {
      "check_name": "introspection_capability",
      "passed": true,
      "message": "Introspection capability available",
      "severity": "medium",
      "fix_attempted": false,
      "fix_successful": false,
      "timestamp": "2025-07-09T18:22:10.383891"
    },
    {
      "check_name": "api_integration",
      "passed": false,
      "message": "API key not configured",
      "severity": "low",
      "fix_attempted": false,
      "fix_successful": false,
      "timestamp": "2025-07-09T18:22:10.383897"
    }
  ],
  "timestamp": "2025-07-09T18:22:10.383906"
}
```
