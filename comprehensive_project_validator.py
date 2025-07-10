#!/usr/bin/env python3
"""
Comprehensive Project Validation Framework
Performs end-to-end testing and validation of the entire project
"""

import asyncio
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import importlib.util
import os


class ProjectValidator:
    """Comprehensive project validation system"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "validation_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0,
                "success_rate": 0.0
            },
            "component_results": {},
            "critical_issues": [],
            "recommendations": [],
            "performance_metrics": {}
        }
        
        # Critical components to validate
        self.critical_components = {
            "task_master": {
                "files": [".taskmaster/tasks/tasks.json", ".taskmaster/config.json"],
                "commands": ["task-master list", "task-master next"]
            },
            "graph_orchestration": {
                "files": ["graph_based_orchestration.py", "agent_handoff_system.py", 
                         "integrated_graph_orchestration.py"],
                "test_file": "test_graph_orchestration.py"
            },
            "multi_agent": {
                "files": ["multi_agent_orchestration.py"],
                "reports": [".taskmaster/reports/enhanced-multi-agent-orchestration-demo.json"]
            },
            "github_workflows": {
                "files": [".github/workflows/recursive-todo-validation.yml"],
                "directories": [".github/workflows"]
            },
            "local_llm": {
                "files": ["local_research_module.py", ".taskmaster/research/local_llm_research_engine.py"],
                "migration_reports": [".taskmaster/reports/local-llm-migration-health-check.md"]
            },
            "optimization": {
                "files": [".taskmaster/scripts/mathematical-optimization-algorithms.py"],
                "validation_reports": [".taskmaster/validation/comprehensive_completion_validator.py"]
            }
        }
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete project validation"""
        print("🔍 COMPREHENSIVE PROJECT VALIDATION")
        print("=" * 60)
        print(f"Project Root: {self.project_root}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run all validation components
        await self._validate_project_structure()
        await self._validate_task_master_integration()
        await self._validate_graph_orchestration()
        await self._validate_multi_agent_framework()
        await self._validate_github_workflows()
        await self._validate_local_llm_integration()
        await self._validate_optimization_systems()
        await self._validate_documentation()
        await self._run_performance_tests()
        
        # Generate final summary
        self._generate_validation_summary()
        
        return self.validation_results
    
    async def _validate_project_structure(self):
        """Validate basic project structure"""
        print("📁 Validating Project Structure...")
        test_name = "project_structure"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Check critical directories
            required_dirs = [
                ".taskmaster",
                ".taskmaster/tasks", 
                ".taskmaster/reports",
                ".taskmaster/scripts",
                ".github/workflows"
            ]
            
            for dir_path in required_dirs:
                if Path(dir_path).exists():
                    results["details"][dir_path] = "exists"
                else:
                    results["details"][dir_path] = "missing"
                    results["issues"].append(f"Missing directory: {dir_path}")
            
            # Check critical files
            required_files = [
                "CLAUDE.md",
                ".taskmaster/tasks/tasks.json",
                "multi_agent_orchestration.py",
                "graph_based_orchestration.py"
            ]
            
            for file_path in required_files:
                if Path(file_path).exists():
                    results["details"][file_path] = "exists"
                else:
                    results["details"][file_path] = "missing"
                    results["issues"].append(f"Missing file: {file_path}")
            
            if results["issues"]:
                results["status"] = "failed"
            
            print(f"   Project structure: {'✅ VALID' if results['status'] == 'passed' else '❌ ISSUES FOUND'}")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error validating project structure: {e}")
        
        self.validation_results["component_results"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _validate_task_master_integration(self):
        """Validate Task Master AI integration"""
        print("🎯 Validating Task Master Integration...")
        test_name = "task_master_integration"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Test task-master command availability
            try:
                result = subprocess.run(["task-master", "list"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    results["details"]["command_available"] = True
                    results["details"]["task_count"] = len(result.stdout.split('\n')) - 2
                else:
                    results["issues"].append("task-master command failed")
                    results["status"] = "failed"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results["issues"].append("task-master command not available")
                results["status"] = "failed"
            
            # Validate tasks.json structure
            tasks_file = Path(".taskmaster/tasks/tasks.json")
            if tasks_file.exists():
                try:
                    with open(tasks_file, 'r') as f:
                        tasks_data = json.load(f)
                    
                    if "master" in tasks_data and "tasks" in tasks_data["master"]:
                        results["details"]["tasks_structure"] = "valid"
                        results["details"]["total_tasks"] = len(tasks_data["master"]["tasks"])
                        
                        # Count task statuses
                        status_counts = {}
                        for task in tasks_data["master"]["tasks"]:
                            status = task.get("status", "unknown")
                            status_counts[status] = status_counts.get(status, 0) + 1
                        results["details"]["task_statuses"] = status_counts
                    else:
                        results["issues"].append("Invalid tasks.json structure")
                        results["status"] = "failed"
                        
                except json.JSONDecodeError:
                    results["issues"].append("tasks.json is not valid JSON")
                    results["status"] = "failed"
            else:
                results["issues"].append("tasks.json file missing")
                results["status"] = "failed"
            
            print(f"   Task Master integration: {'✅ OPERATIONAL' if results['status'] == 'passed' else '❌ ISSUES FOUND'}")
            if results["status"] == "passed":
                print(f"   Total tasks: {results['details'].get('total_tasks', 0)}")
                
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error validating Task Master: {e}")
        
        self.validation_results["component_results"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _validate_graph_orchestration(self):
        """Validate graph-based orchestration system"""
        print("🔄 Validating Graph Orchestration System...")
        test_name = "graph_orchestration"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Test core graph orchestration file
            graph_file = Path("graph_based_orchestration.py")
            if graph_file.exists():
                results["details"]["core_file"] = "exists"
                
                # Test import
                try:
                    spec = importlib.util.spec_from_file_location("graph_based_orchestration", graph_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    results["details"]["import_test"] = "passed"
                    
                    # Check for key classes
                    required_classes = [
                        "GraphBasedOrchestrator", "ExecutableWorkflowGraph", 
                        "GraphNode", "GraphState"
                    ]
                    for class_name in required_classes:
                        if hasattr(module, class_name):
                            results["details"][f"class_{class_name}"] = "available"
                        else:
                            results["issues"].append(f"Missing class: {class_name}")
                            results["status"] = "failed"
                            
                except Exception as e:
                    results["issues"].append(f"Import failed: {str(e)}")
                    results["status"] = "failed"
            else:
                results["issues"].append("graph_based_orchestration.py missing")
                results["status"] = "failed"
            
            # Test handoff system
            handoff_file = Path("agent_handoff_system.py")
            if handoff_file.exists():
                results["details"]["handoff_file"] = "exists"
            else:
                results["issues"].append("agent_handoff_system.py missing")
                results["status"] = "failed"
            
            # Check test results
            test_results_file = Path(".taskmaster/reports/graph-orchestration-test-results.json")
            if test_results_file.exists():
                try:
                    with open(test_results_file, 'r') as f:
                        test_data = json.load(f)
                    results["details"]["test_results"] = test_data["summary"]
                except Exception as e:
                    results["issues"].append(f"Could not read test results: {e}")
            
            print(f"   Graph orchestration: {'✅ VALIDATED' if results['status'] == 'passed' else '❌ ISSUES FOUND'}")
            
        except Exception as e:
            results["status"] = "error" 
            results["error"] = str(e)
            print(f"   ❌ Error validating graph orchestration: {e}")
        
        self.validation_results["component_results"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _validate_multi_agent_framework(self):
        """Validate multi-agent orchestration framework"""
        print("🤖 Validating Multi-Agent Framework...")
        test_name = "multi_agent_framework"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Test multi-agent orchestration file
            multi_agent_file = Path("multi_agent_orchestration.py")
            if multi_agent_file.exists():
                results["details"]["core_file"] = "exists"
                
                # Test file size (should be substantial)
                file_size = multi_agent_file.stat().st_size
                results["details"]["file_size_kb"] = round(file_size / 1024, 2)
                
                if file_size > 10000:  # > 10KB indicates substantial implementation
                    results["details"]["implementation"] = "substantial"
                else:
                    results["issues"].append("Multi-agent file seems too small")
                    results["status"] = "failed"
                
                # Test import
                try:
                    spec = importlib.util.spec_from_file_location("multi_agent_orchestration", multi_agent_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    results["details"]["import_test"] = "passed"
                    
                    # Check for key classes
                    required_classes = [
                        "AgentOrchestrator", "BaseAgent", "ResearchAgent", 
                        "PlanningAgent", "ExecutionAgent", "ValidationAgent"
                    ]
                    for class_name in required_classes:
                        if hasattr(module, class_name):
                            results["details"][f"class_{class_name}"] = "available"
                        else:
                            results["issues"].append(f"Missing class: {class_name}")
                            results["status"] = "failed"
                            
                except Exception as e:
                    results["issues"].append(f"Import failed: {str(e)}")
                    results["status"] = "failed"
            else:
                results["issues"].append("multi_agent_orchestration.py missing")
                results["status"] = "failed"
            
            # Check for demo results
            demo_file = Path(".taskmaster/reports/enhanced-multi-agent-orchestration-demo.json")
            if demo_file.exists():
                results["details"]["demo_results"] = "available"
            else:
                results["issues"].append("Multi-agent demo results missing")
            
            print(f"   Multi-agent framework: {'✅ VALIDATED' if results['status'] == 'passed' else '❌ ISSUES FOUND'}")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error validating multi-agent framework: {e}")
        
        self.validation_results["component_results"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _validate_github_workflows(self):
        """Validate GitHub Actions workflows"""
        print("⚙️ Validating GitHub Workflows...")
        test_name = "github_workflows"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            workflows_dir = Path(".github/workflows")
            if workflows_dir.exists():
                results["details"]["workflows_directory"] = "exists"
                
                # Count workflow files
                workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
                results["details"]["workflow_count"] = len(workflow_files)
                results["details"]["workflow_files"] = [f.name for f in workflow_files]
                
                # Check for key workflows
                key_workflows = [
                    "recursive-todo-validation.yml",
                    "claude-task-execution.yml"
                ]
                
                for workflow in key_workflows:
                    workflow_path = workflows_dir / workflow
                    if workflow_path.exists():
                        results["details"][f"workflow_{workflow}"] = "exists"
                        
                        # Validate YAML syntax
                        try:
                            import yaml
                            with open(workflow_path, 'r') as f:
                                yaml.safe_load(f)
                            results["details"][f"yaml_valid_{workflow}"] = True
                        except ImportError:
                            # yaml not available, skip validation
                            results["details"][f"yaml_valid_{workflow}"] = "skipped"
                        except Exception as e:
                            results["issues"].append(f"Invalid YAML in {workflow}: {e}")
                            results["status"] = "failed"
                    else:
                        results["issues"].append(f"Missing workflow: {workflow}")
                
                if len(workflow_files) == 0:
                    results["issues"].append("No workflow files found")
                    results["status"] = "failed"
                    
            else:
                results["issues"].append("GitHub workflows directory missing")
                results["status"] = "failed"
            
            print(f"   GitHub workflows: {'✅ VALIDATED' if results['status'] == 'passed' else '❌ ISSUES FOUND'}")
            if results["status"] == "passed":
                print(f"   Workflow files: {results['details']['workflow_count']}")
                
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error validating GitHub workflows: {e}")
        
        self.validation_results["component_results"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _validate_local_llm_integration(self):
        """Validate local LLM integration"""
        print("🧠 Validating Local LLM Integration...")
        test_name = "local_llm_integration"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Check local research module
            local_module = Path("local_research_module.py")
            if local_module.exists():
                results["details"]["local_module"] = "exists"
                
                # Test import
                try:
                    spec = importlib.util.spec_from_file_location("local_research_module", local_module)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    results["details"]["import_test"] = "passed"
                    
                    # Check for key functions
                    required_functions = ["research_with_local_llm", "autonomous_stuck_handler_local"]
                    for func_name in required_functions:
                        if hasattr(module, func_name):
                            results["details"][f"function_{func_name}"] = "available"
                        else:
                            results["issues"].append(f"Missing function: {func_name}")
                            
                except Exception as e:
                    results["issues"].append(f"Import failed: {str(e)}")
                    results["status"] = "failed"
            else:
                results["issues"].append("local_research_module.py missing")
                results["status"] = "failed"
            
            # Check LLM research engine
            llm_engine = Path(".taskmaster/research/local_llm_research_engine.py")
            if llm_engine.exists():
                results["details"]["llm_engine"] = "exists"
            else:
                results["issues"].append("Local LLM research engine missing")
            
            # Check migration health report
            health_report = Path(".taskmaster/reports/local-llm-migration-health-check.md")
            if health_report.exists():
                results["details"]["health_report"] = "available"
            else:
                results["issues"].append("LLM migration health report missing")
            
            print(f"   Local LLM integration: {'✅ VALIDATED' if results['status'] == 'passed' else '❌ ISSUES FOUND'}")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error validating local LLM integration: {e}")
        
        self.validation_results["component_results"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _validate_optimization_systems(self):
        """Validate optimization systems"""
        print("🚀 Validating Optimization Systems...")
        test_name = "optimization_systems"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Check mathematical optimization algorithms
            math_opt = Path(".taskmaster/scripts/mathematical-optimization-algorithms.py")
            if math_opt.exists():
                results["details"]["math_optimization"] = "exists"
                
                # Check file size (should be substantial)
                file_size = math_opt.stat().st_size
                results["details"]["math_opt_size_kb"] = round(file_size / 1024, 2)
                
                if file_size > 5000:  # > 5KB indicates implementation
                    results["details"]["math_implementation"] = "substantial"
                else:
                    results["issues"].append("Math optimization file seems incomplete")
            else:
                results["issues"].append("Mathematical optimization algorithms missing")
                results["status"] = "failed"
            
            # Check validation framework
            validator = Path(".taskmaster/validation/comprehensive_completion_validator.py")
            if validator.exists():
                results["details"]["validator"] = "exists"
            else:
                results["issues"].append("Comprehensive validator missing")
            
            # Check for optimization reports
            reports_dir = Path(".taskmaster/reports")
            if reports_dir.exists():
                optimization_reports = list(reports_dir.glob("*optimization*"))
                results["details"]["optimization_reports"] = len(optimization_reports)
            
            print(f"   Optimization systems: {'✅ VALIDATED' if results['status'] == 'passed' else '❌ ISSUES FOUND'}")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error validating optimization systems: {e}")
        
        self.validation_results["component_results"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _validate_documentation(self):
        """Validate project documentation"""
        print("📚 Validating Documentation...")
        test_name = "documentation"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Check core documentation files
            doc_files = {
                "CLAUDE.md": "Main project instructions",
                "README.md": "Project readme",
                "GRAPH_ORCHESTRATION_IMPLEMENTATION_COMPLETE.md": "Graph orchestration docs"
            }
            
            for doc_file, description in doc_files.items():
                if Path(doc_file).exists():
                    results["details"][doc_file] = "exists"
                    
                    # Check file size
                    file_size = Path(doc_file).stat().st_size
                    if file_size > 1000:  # > 1KB
                        results["details"][f"{doc_file}_quality"] = "substantial"
                    else:
                        results["issues"].append(f"{doc_file} seems too brief")
                else:
                    results["issues"].append(f"Missing documentation: {doc_file}")
            
            # Check reports directory
            reports_dir = Path(".taskmaster/reports")
            if reports_dir.exists():
                report_files = list(reports_dir.glob("*.json")) + list(reports_dir.glob("*.md"))
                results["details"]["report_count"] = len(report_files)
                results["details"]["reports"] = [f.name for f in report_files[:10]]  # First 10
            else:
                results["issues"].append("Reports directory missing")
                results["status"] = "failed"
            
            print(f"   Documentation: {'✅ VALIDATED' if results['status'] == 'passed' else '❌ ISSUES FOUND'}")
            if results["status"] == "passed":
                print(f"   Report files: {results['details'].get('report_count', 0)}")
                
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error validating documentation: {e}")
        
        self.validation_results["component_results"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _run_performance_tests(self):
        """Run performance tests"""
        print("⚡ Running Performance Tests...")
        test_name = "performance_tests"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            start_time = datetime.now()
            
            # Test file system performance
            test_file = Path(".taskmaster/.performance_test")
            with open(test_file, 'w') as f:
                f.write("performance test data" * 1000)
            
            # Read test
            with open(test_file, 'r') as f:
                data = f.read()
            
            test_file.unlink()  # Clean up
            
            file_test_time = (datetime.now() - start_time).total_seconds()
            results["details"]["file_io_time_seconds"] = file_test_time
            
            # Test Python import performance
            import_start = datetime.now()
            try:
                import json
                import asyncio
                import subprocess
            except ImportError:
                pass
            
            import_time = (datetime.now() - import_start).total_seconds()
            results["details"]["import_time_seconds"] = import_time
            
            # Memory usage check (basic)
            try:
                import psutil
                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                results["details"]["memory_usage_mb"] = round(memory_usage, 2)
            except ImportError:
                results["details"]["memory_usage_mb"] = "unavailable (psutil not installed)"
            
            # Overall performance assessment
            if file_test_time > 1.0:
                results["issues"].append("File I/O performance seems slow")
                results["status"] = "warning"
            
            print(f"   Performance tests: {'✅ COMPLETED' if results['status'] in ['passed', 'warning'] else '❌ FAILED'}")
            print(f"   File I/O: {file_test_time:.3f}s")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"   ❌ Error running performance tests: {e}")
        
        self.validation_results["component_results"][test_name] = results
        self.validation_results["performance_metrics"] = results["details"]
        self._update_test_counts(results["status"])
    
    def _update_test_counts(self, status: str):
        """Update test counts based on status"""
        self.validation_results["validation_summary"]["total_tests"] += 1
        
        if status == "passed":
            self.validation_results["validation_summary"]["passed_tests"] += 1
        elif status in ["failed", "error"]:
            self.validation_results["validation_summary"]["failed_tests"] += 1
        else:
            self.validation_results["validation_summary"]["skipped_tests"] += 1
    
    def _generate_validation_summary(self):
        """Generate final validation summary"""
        summary = self.validation_results["validation_summary"]
        total = summary["total_tests"]
        passed = summary["passed_tests"]
        
        if total > 0:
            summary["success_rate"] = round((passed / total) * 100, 2)
        
        # Collect critical issues
        critical_issues = []
        for component, results in self.validation_results["component_results"].items():
            if results["status"] in ["failed", "error"]:
                critical_issues.append(f"{component}: {results.get('error', 'validation failed')}")
                for issue in results.get("issues", []):
                    critical_issues.append(f"{component}: {issue}")
        
        self.validation_results["critical_issues"] = critical_issues
        
        # Generate recommendations
        recommendations = []
        if summary["success_rate"] < 80:
            recommendations.append("Project has significant issues that need attention")
        elif summary["success_rate"] < 95:
            recommendations.append("Project is mostly functional but has some issues to resolve")
        else:
            recommendations.append("Project is in excellent condition")
        
        if critical_issues:
            recommendations.append("Address critical issues before deployment")
        
        self.validation_results["recommendations"] = recommendations
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Skipped: {summary['skipped_tests']}")
        print(f"Success Rate: {summary['success_rate']}%")
        
        if critical_issues:
            print(f"\n⚠️ CRITICAL ISSUES ({len(critical_issues)}):")
            for issue in critical_issues[:5]:  # Show first 5
                print(f"  • {issue}")
            if len(critical_issues) > 5:
                print(f"  ... and {len(critical_issues) - 5} more issues")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  • {rec}")


async def main():
    """Run comprehensive project validation"""
    validator = ProjectValidator()
    
    try:
        results = await validator.run_comprehensive_validation()
        
        # Save results
        reports_dir = Path(".taskmaster/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = reports_dir / "comprehensive_project_validation.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Validation results saved to: {results_file}")
        
        # Generate summary report
        summary_file = reports_dir / "PROJECT_VALIDATION_SUMMARY.md"
        with open(summary_file, 'w') as f:
            f.write(f"# Project Validation Summary\n\n")
            f.write(f"**Validation Date**: {results['timestamp']}\n")
            f.write(f"**Project Root**: {results['project_root']}\n\n")
            
            summary = results['validation_summary']
            f.write(f"## Overall Results\n\n")
            f.write(f"- **Total Tests**: {summary['total_tests']}\n")
            f.write(f"- **Passed**: {summary['passed_tests']}\n")
            f.write(f"- **Failed**: {summary['failed_tests']}\n")
            f.write(f"- **Success Rate**: {summary['success_rate']}%\n\n")
            
            if results['critical_issues']:
                f.write(f"## Critical Issues\n\n")
                for issue in results['critical_issues']:
                    f.write(f"- {issue}\n")
                f.write(f"\n")
            
            f.write(f"## Recommendations\n\n")
            for rec in results['recommendations']:
                f.write(f"- {rec}\n")
            
            f.write(f"\n## Component Details\n\n")
            for component, details in results['component_results'].items():
                status_emoji = "✅" if details['status'] == 'passed' else "❌" if details['status'] in ['failed', 'error'] else "⚠️"
                f.write(f"### {status_emoji} {component.replace('_', ' ').title()}\n")
                f.write(f"**Status**: {details['status']}\n\n")
                if details.get('issues'):
                    f.write(f"**Issues**:\n")
                    for issue in details['issues']:
                        f.write(f"- {issue}\n")
                    f.write(f"\n")
        
        print(f"📄 Summary report saved to: {summary_file}")
        
        # Return appropriate exit code
        if results['validation_summary']['success_rate'] >= 80:
            print(f"\n✅ PROJECT VALIDATION COMPLETED SUCCESSFULLY")
            return 0
        else:
            print(f"\n⚠️ PROJECT VALIDATION COMPLETED WITH ISSUES")
            return 1
            
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)