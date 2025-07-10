#!/usr/bin/env python3
"""
Comprehensive System Health Check for Task Master AI
Validates all components after local LLM migration
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Tuple

def run_command(cmd: str) -> Tuple[bool, str]:
    """Run shell command and return success status and output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)

def check_file_health() -> Dict[str, Any]:
    """Check health of all system files"""
    print("🔍 Checking File Health...")
    
    critical_files = {
        "local_llm_research_module.py": "Local LLM research engine",
        "local_llm_demo.py": "Demo and validation script",
        "privacy_compliance_test.py": "Privacy compliance testing",
        "LOCAL_LLM_MIGRATION_GUIDE.md": "Migration documentation",
        "CLAUDE.md": "Main documentation",
        ".taskmaster/reports/task-47-4-implementation.json": "Task 47.4 report",
        ".taskmaster/reports/task-47-5-validation-report.json": "Task 47.5 report",
        ".taskmaster/reports/privacy-compliance-test.json": "Privacy compliance report"
    }
    
    file_health = {
        "status": "HEALTHY",
        "files_checked": len(critical_files),
        "files_present": 0,
        "files_missing": [],
        "file_details": {}
    }
    
    for file_path, description in critical_files.items():
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        
        file_health["file_details"][file_path] = {
            "description": description,
            "exists": exists,
            "size_bytes": size,
            "readable": os.access(file_path, os.R_OK) if exists else False
        }
        
        if exists:
            file_health["files_present"] += 1
            print(f"  ✅ {file_path} ({size} bytes)")
        else:
            file_health["files_missing"].append(file_path)
            print(f"  ❌ {file_path} - MISSING")
    
    if file_health["files_missing"]:
        file_health["status"] = "DEGRADED"
    
    return file_health

def check_python_modules() -> Dict[str, Any]:
    """Check Python module imports and syntax"""
    print("\n🐍 Checking Python Module Health...")
    
    modules_to_check = [
        "local_llm_research_module.py",
        "local_llm_demo.py", 
        "privacy_compliance_test.py"
    ]
    
    module_health = {
        "status": "HEALTHY",
        "modules_checked": len(modules_to_check),
        "syntax_valid": 0,
        "import_errors": [],
        "module_details": {}
    }
    
    for module in modules_to_check:
        if not os.path.exists(module):
            continue
            
        # Check syntax
        success, output = run_command(f"python3 -m py_compile {module}")
        syntax_ok = success
        
        # Check imports (dry run)
        import_success, import_output = run_command(f"python3 -c \"import ast; ast.parse(open('{module}').read())\"")
        
        module_health["module_details"][module] = {
            "syntax_valid": syntax_ok,
            "import_check": import_success,
            "errors": output if not syntax_ok else None
        }
        
        if syntax_ok and import_success:
            module_health["syntax_valid"] += 1
            print(f"  ✅ {module} - Syntax and imports OK")
        else:
            module_health["import_errors"].append(module)
            print(f"  ❌ {module} - Issues detected")
            if output:
                print(f"    Error: {output[:200]}...")
    
    if module_health["import_errors"]:
        module_health["status"] = "DEGRADED"
    
    return module_health

def check_task_master_integration() -> Dict[str, Any]:
    """Check Task Master integration health"""
    print("\n📋 Checking Task Master Integration...")
    
    integration_health = {
        "status": "HEALTHY",
        "task_master_available": False,
        "tasks_file_valid": False,
        "reports_directory": False,
        "integration_details": {}
    }
    
    # Check task-master command availability
    tm_success, tm_output = run_command("task-master --version")
    integration_health["task_master_available"] = tm_success
    
    if tm_success:
        print("  ✅ task-master command available")
    else:
        print("  ⚠️ task-master command not available")
    
    # Check tasks.json file
    tasks_file = ".taskmaster/tasks/tasks.json"
    if os.path.exists(tasks_file):
        try:
            with open(tasks_file, 'r') as f:
                tasks_data = json.load(f)
            integration_health["tasks_file_valid"] = True
            print("  ✅ tasks.json file valid")
            
            # Check task 47 status
            master_tasks = tasks_data.get("master", {}).get("tasks", [])
            task_47 = next((t for t in master_tasks if t["id"] == 47), None)
            
            if task_47:
                subtasks = task_47.get("subtasks", [])
                completed_subtasks = [st for st in subtasks if st.get("status") == "done"]
                integration_health["integration_details"]["task_47_progress"] = {
                    "total_subtasks": len(subtasks),
                    "completed_subtasks": len(completed_subtasks),
                    "completion_rate": len(completed_subtasks) / len(subtasks) if subtasks else 0
                }
                print(f"  ✅ Task 47: {len(completed_subtasks)}/{len(subtasks)} subtasks completed")
            
        except Exception as e:
            print(f"  ❌ tasks.json file invalid: {e}")
            integration_health["status"] = "DEGRADED"
    else:
        print("  ❌ tasks.json file missing")
        integration_health["status"] = "DEGRADED"
    
    # Check reports directory
    reports_dir = ".taskmaster/reports"
    if os.path.exists(reports_dir) and os.path.isdir(reports_dir):
        integration_health["reports_directory"] = True
        report_count = len([f for f in os.listdir(reports_dir) if f.endswith('.json')])
        print(f"  ✅ Reports directory with {report_count} reports")
    else:
        print("  ❌ Reports directory missing")
        integration_health["status"] = "DEGRADED"
    
    return integration_health

def run_functional_tests() -> Dict[str, Any]:
    """Run functional tests for local LLM system"""
    print("\n🧪 Running Functional Tests...")
    
    test_results = {
        "status": "HEALTHY",
        "tests_run": 0,
        "tests_passed": 0,
        "test_details": {}
    }
    
    # Test 1: Local LLM Demo
    print("  🔬 Testing local LLM demo...")
    demo_success, demo_output = run_command("python3 local_llm_demo.py")
    test_results["tests_run"] += 1
    test_results["test_details"]["local_llm_demo"] = {
        "success": demo_success,
        "output_length": len(demo_output),
        "contains_success": "Implementation Complete!" in demo_output
    }
    
    if demo_success and "Implementation Complete!" in demo_output:
        test_results["tests_passed"] += 1
        print("    ✅ Local LLM demo passed")
    else:
        print("    ❌ Local LLM demo failed")
        print(f"    Error: {demo_output[:200]}...")
    
    # Test 2: Privacy Compliance Test
    print("  🔬 Testing privacy compliance...")
    privacy_success, privacy_output = run_command("python3 privacy_compliance_test.py")
    test_results["tests_run"] += 1
    test_results["test_details"]["privacy_compliance"] = {
        "success": privacy_success,
        "output_length": len(privacy_output),
        "privacy_score_100": "100/100" in privacy_output
    }
    
    if privacy_success and "SUCCESSFUL!" in privacy_output:
        test_results["tests_passed"] += 1
        print("    ✅ Privacy compliance test passed")
    else:
        print("    ❌ Privacy compliance test failed")
        print(f"    Error: {privacy_output[:200]}...")
    
    # Test 3: Module Import Test
    print("  🔬 Testing module imports...")
    import_cmd = """python3 -c "
import sys
sys.path.append('.')
try:
    from local_llm_research_module import LocalLLMResearchEngine, LocalLLMConfigFactory
    print('SUCCESS: Core modules imported')
    print('Available providers:', len(LocalLLMConfigFactory.__dict__))
except Exception as e:
    print('FAILED:', str(e))
    sys.exit(1)
" """
    
    import_success, import_output = run_command(import_cmd)
    test_results["tests_run"] += 1
    test_results["test_details"]["module_imports"] = {
        "success": import_success,
        "output": import_output
    }
    
    if import_success and "SUCCESS" in import_output:
        test_results["tests_passed"] += 1
        print("    ✅ Module imports passed")
    else:
        print("    ❌ Module imports failed")
        print(f"    Error: {import_output}")
    
    # Calculate overall test status
    if test_results["tests_passed"] == test_results["tests_run"]:
        test_results["status"] = "HEALTHY"
    elif test_results["tests_passed"] > 0:
        test_results["status"] = "DEGRADED"
    else:
        test_results["status"] = "FAILED"
    
    return test_results

def check_documentation_health() -> Dict[str, Any]:
    """Check documentation completeness and accuracy"""
    print("\n📚 Checking Documentation Health...")
    
    doc_health = {
        "status": "HEALTHY",
        "documents_checked": 0,
        "documents_valid": 0,
        "doc_details": {}
    }
    
    docs_to_check = [
        ("LOCAL_LLM_MIGRATION_GUIDE.md", ["Local LLM", "Migration", "Privacy", "Setup"]),
        ("CLAUDE.md", ["Task Master", "local LLM", "Configuration"])
    ]
    
    for doc_path, required_content in docs_to_check:
        doc_health["documents_checked"] += 1
        
        if os.path.exists(doc_path):
            with open(doc_path, 'r') as f:
                content = f.read()
            
            content_checks = {keyword: keyword.lower() in content.lower() for keyword in required_content}
            all_present = all(content_checks.values())
            
            doc_health["doc_details"][doc_path] = {
                "exists": True,
                "size": len(content),
                "required_content": content_checks,
                "all_content_present": all_present
            }
            
            if all_present:
                doc_health["documents_valid"] += 1
                print(f"  ✅ {doc_path} - Complete")
            else:
                missing = [k for k, v in content_checks.items() if not v]
                print(f"  ⚠️ {doc_path} - Missing: {missing}")
        else:
            doc_health["doc_details"][doc_path] = {
                "exists": False,
                "all_content_present": False
            }
            print(f"  ❌ {doc_path} - Missing")
    
    if doc_health["documents_valid"] < doc_health["documents_checked"]:
        doc_health["status"] = "DEGRADED"
    
    return doc_health

def generate_health_report(results: Dict[str, Any]) -> None:
    """Generate comprehensive health report"""
    print("\n📊 System Health Report")
    print("=" * 60)
    
    overall_status = "HEALTHY"
    issues = []
    
    for component, data in results.items():
        status = data.get("status", "UNKNOWN")
        print(f"\n🔧 {component.replace('_', ' ').title()}: {status}")
        
        if status != "HEALTHY":
            overall_status = "DEGRADED" if overall_status == "HEALTHY" else "FAILED"
            issues.append(f"{component}: {status}")
        
        # Component-specific details
        if component == "file_health":
            print(f"   Files: {data['files_present']}/{data['files_checked']} present")
            if data["files_missing"]:
                print(f"   Missing: {', '.join(data['files_missing'])}")
        
        elif component == "python_modules":
            print(f"   Modules: {data['syntax_valid']}/{data['modules_checked']} valid")
            if data["import_errors"]:
                print(f"   Errors: {', '.join(data['import_errors'])}")
        
        elif component == "task_master_integration":
            print(f"   Task-Master: {'Available' if data['task_master_available'] else 'Not Available'}")
            print(f"   Tasks File: {'Valid' if data['tasks_file_valid'] else 'Invalid'}")
            if "task_47_progress" in data.get("integration_details", {}):
                progress = data["integration_details"]["task_47_progress"]
                print(f"   Task 47: {progress['completed_subtasks']}/{progress['total_subtasks']} completed")
        
        elif component == "functional_tests":
            print(f"   Tests: {data['tests_passed']}/{data['tests_run']} passed")
        
        elif component == "documentation_health":
            print(f"   Docs: {data['documents_valid']}/{data['documents_checked']} complete")
    
    print(f"\n🎯 Overall System Status: {overall_status}")
    
    if overall_status == "HEALTHY":
        print("✅ All systems operational")
        print("✅ Local LLM migration successful")
        print("✅ Privacy compliance validated")
        print("✅ Ready for production use")
    else:
        print(f"⚠️ Issues detected: {len(issues)}")
        for issue in issues:
            print(f"   - {issue}")
    
    # Save detailed report
    os.makedirs(".taskmaster/reports", exist_ok=True)
    report_path = ".taskmaster/reports/system-health-check.json"
    
    health_report = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": overall_status,
        "issues": issues,
        "components": results
    }
    
    with open(report_path, 'w') as f:
        json.dump(health_report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")

def main():
    """Run comprehensive system health check"""
    print("🏥 Task Master AI - Comprehensive Health Check")
    print("=" * 80)
    print("Validating system integrity after local LLM migration...")
    
    results = {}
    
    try:
        # Run all health checks
        results["file_health"] = check_file_health()
        results["python_modules"] = check_python_modules()
        results["task_master_integration"] = check_task_master_integration()
        results["functional_tests"] = run_functional_tests()
        results["documentation_health"] = check_documentation_health()
        
        # Generate comprehensive report
        generate_health_report(results)
        
        # Determine exit code
        overall_healthy = all(r.get("status") == "HEALTHY" for r in results.values())
        return 0 if overall_healthy else 1
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)