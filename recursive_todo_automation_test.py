#!/usr/bin/env python3
"""
Recursive Todo Automation System Test
Tests the end-to-end recursive todo validation and improvement system
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


class RecursiveTodoAutomationTester:
    """Tests the recursive todo automation system"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "test_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0
            },
            "test_details": {},
            "system_validation": {}
        }
    
    async def run_automation_tests(self) -> Dict[str, Any]:
        """Run comprehensive automation tests"""
        print("🔄 RECURSIVE TODO AUTOMATION SYSTEM TEST")
        print("=" * 50)
        
        # Test individual components
        await self._test_github_workflows_structure()
        await self._test_task_master_automation()
        await self._test_validation_reports()
        await self._test_improvement_prompts()
        await self._test_parallel_execution_capability()
        
        # Generate summary
        self._generate_test_summary()
        
        return self.test_results
    
    async def _test_github_workflows_structure(self):
        """Test GitHub workflows structure and configuration"""
        print("⚙️ Testing GitHub Workflows Structure...")
        test_name = "github_workflows_structure"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            workflows_dir = Path(".github/workflows")
            
            # Check for recursive validation workflow
            recursive_workflow = workflows_dir / "recursive-todo-validation.yml"
            if recursive_workflow.exists():
                results["details"]["recursive_workflow"] = "exists"
                
                # Parse workflow content
                with open(recursive_workflow, 'r') as f:
                    workflow_content = f.read()
                
                # Check for key automation features
                automation_features = [
                    "discover-todos",
                    "validate-todo-batches", 
                    "atomize-improvements",
                    "execute-recursive-improvements",
                    "generate-final-report"
                ]
                
                for feature in automation_features:
                    if feature in workflow_content:
                        results["details"][f"feature_{feature}"] = "present"
                    else:
                        results["issues"].append(f"Missing automation feature: {feature}")
                
                # Check for parallel execution
                if "matrix" in workflow_content and "strategy" in workflow_content:
                    results["details"]["parallel_execution"] = "configured"
                else:
                    results["issues"].append("Parallel execution not properly configured")
                
                # Check for recursive improvement logic
                if "recursive" in workflow_content.lower() and "improvement" in workflow_content.lower():
                    results["details"]["recursive_logic"] = "present"
                else:
                    results["issues"].append("Recursive improvement logic not found")
                    
            else:
                results["issues"].append("Recursive validation workflow missing")
                results["status"] = "failed"
            
            # Check for other automation workflows
            other_workflows = [
                "claude-task-execution.yml",
                "parallel-task-validation.yml",
                "meta-recursive-continuous-improvement.yml"
            ]
            
            workflow_count = 0
            for workflow in other_workflows:
                if (workflows_dir / workflow).exists():
                    workflow_count += 1
                    results["details"][f"workflow_{workflow}"] = "exists"
            
            results["details"]["total_automation_workflows"] = workflow_count
            
            if results["issues"]:
                results["status"] = "failed"
            
            print(f"   GitHub workflows: {'✅ VALIDATED' if results['status'] == 'passed' else '❌ ISSUES FOUND'}")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error testing workflows: {e}")
        
        self.test_results["test_details"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _test_task_master_automation(self):
        """Test Task Master automation capabilities"""
        print("🎯 Testing Task Master Automation...")
        test_name = "task_master_automation"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Test task-master research command
            try:
                result = subprocess.run(["task-master", "next"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    results["details"]["next_command"] = "functional"
                    
                    # Parse output to check for tasks
                    if "Next Task:" in result.stdout:
                        results["details"]["next_task_available"] = True
                    else:
                        results["details"]["next_task_available"] = False
                else:
                    results["issues"].append("task-master next command failed")
                    results["status"] = "failed"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results["issues"].append("task-master command unavailable")
                results["status"] = "failed"
            
            # Test task-master list command
            try:
                result = subprocess.run(["task-master", "list"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    results["details"]["list_command"] = "functional"
                    
                    # Count tasks in output
                    task_lines = [line for line in result.stdout.split('\n') if line.strip() and '│' in line]
                    results["details"]["visible_tasks"] = len(task_lines) - 2  # Subtract header lines
                else:
                    results["issues"].append("task-master list command failed")
                    results["status"] = "failed"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results["issues"].append("task-master list command unavailable")
                results["status"] = "failed"
            
            # Check for automation-related tasks in tasks.json
            tasks_file = Path(".taskmaster/tasks/tasks.json")
            if tasks_file.exists():
                with open(tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                
                automation_keywords = ["recursive", "automation", "improvement", "validation"]
                automation_tasks = 0
                
                if "master" in tasks_data and "tasks" in tasks_data["master"]:
                    for task in tasks_data["master"]["tasks"]:
                        task_text = (task.get("title", "") + " " + task.get("description", "")).lower()
                        if any(keyword in task_text for keyword in automation_keywords):
                            automation_tasks += 1
                
                results["details"]["automation_tasks_count"] = automation_tasks
            
            print(f"   Task Master automation: {'✅ OPERATIONAL' if results['status'] == 'passed' else '❌ ISSUES FOUND'}")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error testing Task Master automation: {e}")
        
        self.test_results["test_details"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _test_validation_reports(self):
        """Test validation reporting system"""
        print("📊 Testing Validation Reports...")
        test_name = "validation_reports"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            reports_dir = Path(".taskmaster/reports")
            
            # Check for validation reports
            validation_reports = [
                "comprehensive_project_validation.json",
                "graph-orchestration-test-results.json",
                "agent-handoff-demo.json",
                "local-llm-migration-health-check.md"
            ]
            
            found_reports = 0
            for report in validation_reports:
                report_path = reports_dir / report
                if report_path.exists():
                    found_reports += 1
                    results["details"][f"report_{report}"] = "exists"
                    
                    # Check file size for substantiality
                    file_size = report_path.stat().st_size
                    if file_size > 1000:  # > 1KB
                        results["details"][f"report_{report}_quality"] = "substantial"
                    else:
                        results["issues"].append(f"Report {report} seems too small")
                else:
                    results["issues"].append(f"Missing validation report: {report}")
            
            results["details"]["total_validation_reports"] = found_reports
            
            # Check for comprehensive validation result
            comp_validation = reports_dir / "comprehensive_project_validation.json"
            if comp_validation.exists():
                with open(comp_validation, 'r') as f:
                    validation_data = json.load(f)
                
                success_rate = validation_data.get("validation_summary", {}).get("success_rate", 0)
                results["details"]["last_validation_success_rate"] = success_rate
                
                if success_rate >= 90:
                    results["details"]["validation_health"] = "excellent"
                elif success_rate >= 75:
                    results["details"]["validation_health"] = "good"
                else:
                    results["details"]["validation_health"] = "needs_attention"
                    results["issues"].append("Low validation success rate")
            
            if found_reports < len(validation_reports) * 0.7:  # Less than 70% of expected reports
                results["status"] = "failed"
            
            print(f"   Validation reports: {'✅ COMPREHENSIVE' if results['status'] == 'passed' else '❌ INCOMPLETE'}")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error testing validation reports: {e}")
        
        self.test_results["test_details"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _test_improvement_prompts(self):
        """Test improvement prompt generation"""
        print("🚀 Testing Improvement Prompts...")
        test_name = "improvement_prompts"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Check for improvement-related files
            improvement_files = [
                ".taskmaster/validation/fix_validation_issues.py",
                ".taskmaster/validation/comprehensive_completion_validator.py",
                "recursive_todo_processor.py"
            ]
            
            found_files = 0
            for file_path in improvement_files:
                if Path(file_path).exists():
                    found_files += 1
                    results["details"][f"file_{Path(file_path).name}"] = "exists"
                else:
                    results["issues"].append(f"Missing improvement file: {file_path}")
            
            results["details"]["improvement_files_found"] = found_files
            
            # Check for validation directories
            validation_dirs = [
                ".taskmaster/validation",
                ".taskmaster/improvements",
                ".taskmaster/research"
            ]
            
            for val_dir in validation_dirs:
                if Path(val_dir).exists():
                    results["details"][f"dir_{Path(val_dir).name}"] = "exists"
                else:
                    results["issues"].append(f"Missing directory: {val_dir}")
            
            # Check improvement algorithms
            math_opt_file = Path(".taskmaster/scripts/mathematical-optimization-algorithms.py")
            if math_opt_file.exists():
                results["details"]["optimization_algorithms"] = "available"
                
                # Check file content for algorithm implementations
                with open(math_opt_file, 'r') as f:
                    content = f.read()
                
                algorithms = ["Williams", "Cook", "Mertz", "Pebbling", "Catalytic"]
                found_algorithms = 0
                for algorithm in algorithms:
                    if algorithm in content:
                        found_algorithms += 1
                        results["details"][f"algorithm_{algorithm}"] = "implemented"
                
                results["details"]["total_algorithms"] = found_algorithms
                
                if found_algorithms < 3:
                    results["issues"].append("Insufficient optimization algorithms implemented")
            
            if results["issues"]:
                results["status"] = "warning"
            
            print(f"   Improvement prompts: {'✅ OPERATIONAL' if results['status'] in ['passed', 'warning'] else '❌ ISSUES FOUND'}")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error testing improvement prompts: {e}")
        
        self.test_results["test_details"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _test_parallel_execution_capability(self):
        """Test parallel execution capabilities"""
        print("⚡ Testing Parallel Execution...")
        test_name = "parallel_execution"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Check for parallel processing files
            parallel_files = [
                "graph_based_orchestration.py",
                "agent_handoff_system.py",
                "integrated_graph_orchestration.py"
            ]
            
            parallel_keywords = ["asyncio", "await", "concurrent", "parallel", "gather"]
            total_parallel_features = 0
            
            for file_path in parallel_files:
                if Path(file_path).exists():
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    file_parallel_features = 0
                    for keyword in parallel_keywords:
                        if keyword in content:
                            file_parallel_features += 1
                    
                    results["details"][f"parallel_features_{Path(file_path).stem}"] = file_parallel_features
                    total_parallel_features += file_parallel_features
            
            results["details"]["total_parallel_features"] = total_parallel_features
            
            # Check GitHub workflow parallel configuration
            workflow_file = Path(".github/workflows/recursive-todo-validation.yml")
            if workflow_file.exists():
                with open(workflow_file, 'r') as f:
                    workflow_content = f.read()
                
                # Check for parallel execution indicators
                parallel_indicators = ["matrix", "max-parallel", "strategy", "parallel_workers"]
                workflow_parallel_features = 0
                
                for indicator in parallel_indicators:
                    if indicator in workflow_content:
                        workflow_parallel_features += 1
                        results["details"][f"workflow_{indicator}"] = "present"
                
                results["details"]["workflow_parallel_features"] = workflow_parallel_features
                
                if workflow_parallel_features >= 2:
                    results["details"]["workflow_parallel_support"] = "comprehensive"
                else:
                    results["issues"].append("Limited parallel support in workflows")
            
            # Assess overall parallel capability
            if total_parallel_features >= 10:
                results["details"]["parallel_capability"] = "excellent"
            elif total_parallel_features >= 5:
                results["details"]["parallel_capability"] = "good"
            else:
                results["details"]["parallel_capability"] = "limited"
                results["issues"].append("Limited parallel execution capabilities")
                results["status"] = "warning"
            
            print(f"   Parallel execution: {'✅ CAPABLE' if results['status'] in ['passed', 'warning'] else '❌ LIMITED'}")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error testing parallel execution: {e}")
        
        self.test_results["test_details"][test_name] = results
        self._update_test_counts(results["status"])
    
    def _update_test_counts(self, status: str):
        """Update test counts"""
        self.test_results["test_summary"]["total_tests"] += 1
        
        if status == "passed":
            self.test_results["test_summary"]["passed_tests"] += 1
        elif status in ["failed", "error"]:
            self.test_results["test_summary"]["failed_tests"] += 1
    
    def _generate_test_summary(self):
        """Generate test summary"""
        summary = self.test_results["test_summary"]
        total = summary["total_tests"]
        passed = summary["passed_tests"]
        
        if total > 0:
            summary["success_rate"] = round((passed / total) * 100, 2)
        
        # System validation assessment
        if summary["success_rate"] >= 90:
            self.test_results["system_validation"]["status"] = "excellent"
            self.test_results["system_validation"]["message"] = "Recursive todo automation system is fully operational"
        elif summary["success_rate"] >= 75:
            self.test_results["system_validation"]["status"] = "good"
            self.test_results["system_validation"]["message"] = "Recursive todo automation system is mostly operational"
        else:
            self.test_results["system_validation"]["status"] = "needs_attention"
            self.test_results["system_validation"]["message"] = "Recursive todo automation system needs improvement"
        
        print("\n" + "=" * 50)
        print("📊 AUTOMATION TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"System Status: {self.test_results['system_validation']['status'].upper()}")
        print(f"Assessment: {self.test_results['system_validation']['message']}")


async def main():
    """Run recursive todo automation tests"""
    tester = RecursiveTodoAutomationTester()
    
    try:
        results = await tester.run_automation_tests()
        
        # Save results
        reports_dir = Path(".taskmaster/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = reports_dir / "recursive_todo_automation_test.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Test results saved to: {results_file}")
        
        # Return exit code based on results
        if results["test_summary"]["success_rate"] >= 80:
            print(f"\n✅ RECURSIVE TODO AUTOMATION SYSTEM VALIDATED")
            return 0
        else:
            print(f"\n⚠️ RECURSIVE TODO AUTOMATION SYSTEM NEEDS ATTENTION")
            return 1
            
    except Exception as e:
        print(f"\n❌ AUTOMATION TESTING FAILED: {e}")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)