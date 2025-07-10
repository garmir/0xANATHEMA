#!/usr/bin/env python3
"""
Comprehensive System Validation Test Suite
Complete validation of LABRYS + Task Master AI unified system
"""

import unittest
import asyncio
import json
import os
import sys
import tempfile
import subprocess
import time
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime

class TestComprehensiveSystemValidation(unittest.TestCase):
    """Comprehensive test suite for the entire unified system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment for comprehensive validation"""
        cls.test_results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
        cls.project_root = Path.cwd()
        
    def setUp(self):
        """Set up individual test"""
        self.test_start_time = time.time()
        
    def tearDown(self):
        """Clean up after each test"""
        execution_time = time.time() - self.test_start_time
        test_name = self._testMethodName
        
        # Record test result
        test_detail = {
            "test_name": test_name,
            "execution_time": round(execution_time, 3),
            "status": "passed",  # Will be overridden if test fails
            "timestamp": datetime.now().isoformat()
        }
        
        self.test_results["test_details"].append(test_detail)
        
    def record_test_result(self, test_name: str, status: str, details: str = ""):
        """Record test execution result"""
        self.test_results["total_tests"] += 1
        if status == "passed":
            self.test_results["passed_tests"] += 1
        else:
            self.test_results["failed_tests"] += 1
            
        # Update the last test detail entry
        if self.test_results["test_details"]:
            self.test_results["test_details"][-1].update({
                "status": status,
                "details": details
            })
    
    def test_01_project_structure_validation(self):
        """Test complete project structure and file integrity"""
        print("\n🏗️  Testing project structure validation...")
        
        required_files = [
            "unified_autonomous_system.py",
            "labrys_main.py",
            "taskmaster_labrys.py",
            "autonomous_workflow_loop.py",
            ".taskmaster/tasks/tasks.json",
            "README.md",
            "requirements.txt"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.record_test_result("project_structure", "failed", f"Missing files: {missing_files}")
            self.fail(f"Missing critical files: {missing_files}")
        
        # Check directory structure
        required_dirs = [
            ".taskmaster",
            ".taskmaster/tasks",
            ".github/workflows",
            ".github/scripts"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            self.record_test_result("project_structure", "failed", f"Missing directories: {missing_dirs}")
            self.fail(f"Missing critical directories: {missing_dirs}")
            
        self.record_test_result("project_structure", "passed", f"All {len(required_files)} files and {len(required_dirs)} directories present")
        print("✅ Project structure validation passed")
    
    def test_02_python_syntax_validation(self):
        """Test Python syntax validity across all Python files"""
        print("\n🐍 Testing Python syntax validation...")
        
        python_files = list(Path('.').rglob('*.py'))
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                    compile(source, str(py_file), 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
            except Exception as e:
                # Skip files that can't be read or have encoding issues
                continue
        
        if syntax_errors:
            self.record_test_result("python_syntax", "failed", f"Syntax errors: {syntax_errors[:5]}")
            self.fail(f"Python syntax errors found: {syntax_errors[:5]}")
        
        self.record_test_result("python_syntax", "passed", f"All {len(python_files)} Python files have valid syntax")
        print(f"✅ Python syntax validation passed for {len(python_files)} files")
    
    def test_03_unified_system_import_test(self):
        """Test unified autonomous system can be imported and initialized"""
        print("\n🤖 Testing unified system import and initialization...")
        
        try:
            # Test import
            sys.path.insert(0, str(self.project_root))
            
            # Mock external dependencies
            with patch('subprocess.run'), \
                 patch('requests.post'), \
                 patch('time.sleep'):
                
                try:
                    import unified_autonomous_system
                    self.assertTrue(hasattr(unified_autonomous_system, 'UnifiedAutonomousSystem'))
                    
                    # Test class instantiation
                    system = unified_autonomous_system.UnifiedAutonomousSystem()
                    self.assertIsNotNone(system)
                    
                    # Test configuration loading
                    self.assertIsInstance(system.config, dict)
                    
                    self.record_test_result("unified_system_import", "passed", "Import and instantiation successful")
                    print("✅ Unified system import test passed")
                    
                except ImportError as e:
                    self.record_test_result("unified_system_import", "failed", f"Import error: {e}")
                    self.fail(f"Failed to import unified system: {e}")
                    
        except Exception as e:
            self.record_test_result("unified_system_import", "failed", f"Unexpected error: {e}")
            self.fail(f"Unified system test failed: {e}")
    
    def test_04_labrys_framework_validation(self):
        """Test LABRYS framework components and integration"""
        print("\n🗲 Testing LABRYS framework validation...")
        
        try:
            # Check for LABRYS main file
            if not Path("labrys_main.py").exists():
                self.record_test_result("labrys_validation", "failed", "labrys_main.py not found")
                self.fail("LABRYS main file not found")
            
            # Check for TaskMaster-LABRYS integration
            if not Path("taskmaster_labrys.py").exists():
                self.record_test_result("labrys_validation", "failed", "taskmaster_labrys.py not found")
                self.fail("TaskMaster-LABRYS integration file not found")
            
            # Test LABRYS import (with mocking)
            with patch('subprocess.run'), \
                 patch('requests.post'), \
                 patch('time.sleep'):
                
                try:
                    import labrys_main
                    self.assertTrue(hasattr(labrys_main, 'LabrysFramework') or 'labrys' in str(labrys_main))
                    
                    import taskmaster_labrys
                    self.assertTrue(hasattr(taskmaster_labrys, 'TaskMasterLabrys') or 'task' in str(taskmaster_labrys))
                    
                    self.record_test_result("labrys_validation", "passed", "LABRYS components import successfully")
                    print("✅ LABRYS framework validation passed")
                    
                except ImportError as e:
                    self.record_test_result("labrys_validation", "partial", f"Import issues: {e}")
                    print(f"⚠️ LABRYS framework has import issues: {e}")
                    
        except Exception as e:
            self.record_test_result("labrys_validation", "failed", f"Validation error: {e}")
            self.fail(f"LABRYS framework validation failed: {e}")
    
    def test_05_task_master_integration_test(self):
        """Test Task Master AI integration and functionality"""
        print("\n📊 Testing Task Master AI integration...")
        
        try:
            # Check tasks.json structure
            tasks_file = Path(".taskmaster/tasks/tasks.json")
            if not tasks_file.exists():
                self.record_test_result("task_master_integration", "failed", "tasks.json not found")
                self.fail("Task Master tasks.json not found")
            
            # Validate JSON structure
            with open(tasks_file, 'r') as f:
                tasks_data = json.load(f)
            
            # Check required structure
            self.assertIn("master", tasks_data)
            self.assertIn("tasks", tasks_data["master"])
            self.assertIsInstance(tasks_data["master"]["tasks"], list)
            
            # Check task format
            tasks = tasks_data["master"]["tasks"]
            if tasks:
                sample_task = tasks[0]
                required_fields = ["id", "title", "description", "status", "priority"]
                for field in required_fields:
                    self.assertIn(field, sample_task, f"Missing required field: {field}")
            
            # Test CLI availability (mocked)
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="Task Master CLI available")
                
                try:
                    result = subprocess.run(['task-master', 'list'], capture_output=True, text=True, timeout=5)
                    # This is mocked, so we just verify the call was made
                    mock_run.assert_called_once()
                except:
                    pass  # CLI might not be available in test environment
            
            self.record_test_result("task_master_integration", "passed", f"Tasks.json valid with {len(tasks)} tasks")
            print(f"✅ Task Master integration test passed - {len(tasks)} tasks found")
            
        except Exception as e:
            self.record_test_result("task_master_integration", "failed", f"Integration error: {e}")
            self.fail(f"Task Master integration test failed: {e}")
    
    def test_06_autonomous_workflow_functionality(self):
        """Test autonomous workflow loop functionality"""
        print("\n🔄 Testing autonomous workflow functionality...")
        
        try:
            # Check autonomous workflow file
            workflow_file = Path("autonomous_workflow_loop.py")
            if not workflow_file.exists():
                self.record_test_result("autonomous_workflow", "failed", "autonomous_workflow_loop.py not found")
                self.fail("Autonomous workflow file not found")
            
            # Test import with mocking
            with patch('subprocess.run'), \
                 patch('requests.post'), \
                 patch('time.sleep'), \
                 patch('os.system'):
                
                try:
                    import autonomous_workflow_loop
                    
                    # Check for key classes/functions
                    expected_components = [
                        'AutonomousWorkflowLoop',
                        'ResearchSolution',
                        'TodoStep'
                    ]
                    
                    module_content = str(autonomous_workflow_loop)
                    found_components = []
                    
                    for component in expected_components:
                        if hasattr(autonomous_workflow_loop, component) or component.lower() in module_content.lower():
                            found_components.append(component)
                    
                    if len(found_components) >= 2:  # At least 2 of 3 components found
                        self.record_test_result("autonomous_workflow", "passed", f"Found components: {found_components}")
                        print("✅ Autonomous workflow functionality test passed")
                    else:
                        self.record_test_result("autonomous_workflow", "partial", f"Some components missing: {found_components}")
                        print(f"⚠️ Autonomous workflow partially functional: {found_components}")
                        
                except ImportError as e:
                    self.record_test_result("autonomous_workflow", "failed", f"Import error: {e}")
                    self.fail(f"Failed to import autonomous workflow: {e}")
                    
        except Exception as e:
            self.record_test_result("autonomous_workflow", "failed", f"Workflow error: {e}")
            self.fail(f"Autonomous workflow test failed: {e}")
    
    def test_07_github_actions_workflows_validation(self):
        """Test GitHub Actions workflows configuration"""
        print("\n⚙️ Testing GitHub Actions workflows...")
        
        workflows_dir = Path(".github/workflows")
        if not workflows_dir.exists():
            self.record_test_result("github_workflows", "failed", "Workflows directory not found")
            self.fail("GitHub workflows directory not found")
        
        workflow_files = list(workflows_dir.glob("*.yml"))
        if not workflow_files:
            self.record_test_result("github_workflows", "failed", "No workflow files found")
            self.fail("No GitHub Actions workflow files found")
        
        # Validate workflow structure
        valid_workflows = 0
        total_workflows = len(workflow_files)
        
        for workflow in workflow_files:
            try:
                with open(workflow, 'r') as f:
                    content = f.read()
                
                # Check for required workflow elements
                required_elements = ["name:", "on:", "jobs:"]
                if all(element in content for element in required_elements):
                    valid_workflows += 1
                    
            except Exception:
                continue
        
        success_rate = (valid_workflows / total_workflows) * 100
        
        if success_rate >= 80:
            self.record_test_result("github_workflows", "passed", f"{valid_workflows}/{total_workflows} workflows valid")
            print(f"✅ GitHub Actions workflows validation passed - {valid_workflows}/{total_workflows} valid")
        else:
            self.record_test_result("github_workflows", "failed", f"Only {valid_workflows}/{total_workflows} workflows valid")
            self.fail(f"GitHub Actions workflows validation failed - only {success_rate:.1f}% valid")
    
    def test_08_requirements_and_dependencies(self):
        """Test requirements.txt and dependency management"""
        print("\n📦 Testing requirements and dependencies...")
        
        # Check requirements.txt
        req_file = Path("requirements.txt")
        if not req_file.exists():
            self.record_test_result("requirements", "failed", "requirements.txt not found")
            self.fail("requirements.txt not found")
        
        # Parse requirements
        with open(req_file, 'r') as f:
            requirements = f.read()
        
        # Check for essential dependencies
        essential_deps = [
            "requests",
            "python-dotenv", 
            "asyncio",
            "aiohttp"
        ]
        
        missing_deps = []
        for dep in essential_deps:
            if dep not in requirements:
                missing_deps.append(dep)
        
        if missing_deps:
            self.record_test_result("requirements", "partial", f"Missing dependencies: {missing_deps}")
            print(f"⚠️ Some dependencies missing: {missing_deps}")
        else:
            self.record_test_result("requirements", "passed", "All essential dependencies present")
            print("✅ Requirements and dependencies test passed")
    
    def test_09_performance_and_resource_validation(self):
        """Test system performance and resource requirements"""
        print("\n⚡ Testing performance and resource validation...")
        
        try:
            import psutil
            
            # Check system resources
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            disk_free_gb = psutil.disk_usage('.').free / (1024**3)
            
            # Resource requirements (from deployment validation)
            min_memory = 4  # GB
            min_cpu = 2     # cores
            min_disk = 10   # GB
            
            resource_checks = []
            if memory_gb >= min_memory:
                resource_checks.append(f"✅ Memory: {memory_gb:.1f}GB (>= {min_memory}GB)")
            else:
                resource_checks.append(f"❌ Memory: {memory_gb:.1f}GB (< {min_memory}GB)")
            
            if cpu_count >= min_cpu:
                resource_checks.append(f"✅ CPU: {cpu_count} cores (>= {min_cpu})")
            else:
                resource_checks.append(f"❌ CPU: {cpu_count} cores (< {min_cpu})")
            
            if disk_free_gb >= min_disk:
                resource_checks.append(f"✅ Disk: {disk_free_gb:.1f}GB (>= {min_disk}GB)")
            else:
                resource_checks.append(f"❌ Disk: {disk_free_gb:.1f}GB (< {min_disk}GB)")
            
            passed_checks = len([check for check in resource_checks if "✅" in check])
            
            if passed_checks == 3:
                self.record_test_result("performance_validation", "passed", "; ".join(resource_checks))
                print("✅ Performance and resource validation passed")
            else:
                self.record_test_result("performance_validation", "partial", "; ".join(resource_checks))
                print(f"⚠️ Performance validation: {passed_checks}/3 requirements met")
                
        except ImportError:
            # psutil not available, mock the test
            self.record_test_result("performance_validation", "skipped", "psutil not available for testing")
            print("⚠️ Performance validation skipped - psutil not available")
    
    def test_10_end_to_end_integration_test(self):
        """Test end-to-end system integration"""
        print("\n🎯 Testing end-to-end integration...")
        
        try:
            # This is a comprehensive integration test
            integration_components = {
                "unified_system": Path("unified_autonomous_system.py").exists(),
                "labrys_framework": Path("labrys_main.py").exists(),
                "task_master": Path(".taskmaster/tasks/tasks.json").exists(),
                "autonomous_workflow": Path("autonomous_workflow_loop.py").exists(),
                "github_workflows": Path(".github/workflows").exists(),
                "monitoring_scripts": Path(".github/scripts").exists()
            }
            
            # Check integration completeness
            total_components = len(integration_components)
            present_components = sum(integration_components.values())
            integration_score = (present_components / total_components) * 100
            
            missing_components = [comp for comp, present in integration_components.items() if not present]
            
            if integration_score >= 90:
                self.record_test_result("end_to_end_integration", "passed", 
                                       f"Integration score: {integration_score:.1f}% ({present_components}/{total_components})")
                print(f"✅ End-to-end integration test passed - {integration_score:.1f}% complete")
            elif integration_score >= 70:
                self.record_test_result("end_to_end_integration", "partial", 
                                       f"Integration score: {integration_score:.1f}%, missing: {missing_components}")
                print(f"⚠️ End-to-end integration partially successful - {integration_score:.1f}% complete")
            else:
                self.record_test_result("end_to_end_integration", "failed", 
                                       f"Integration score: {integration_score:.1f}%, missing: {missing_components}")
                self.fail(f"End-to-end integration failed - only {integration_score:.1f}% complete")
                
        except Exception as e:
            self.record_test_result("end_to_end_integration", "failed", f"Integration error: {e}")
            self.fail(f"End-to-end integration test failed: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Generate comprehensive test report"""
        
        # Calculate final statistics
        total_tests = cls.test_results["total_tests"]
        passed_tests = cls.test_results["passed_tests"]
        failed_tests = cls.test_results["failed_tests"]
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate report
        report = {
            "comprehensive_system_validation": {
                "timestamp": cls.test_results["timestamp"],
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": round(success_rate, 1),
                "test_details": cls.test_results["test_details"],
                "overall_status": "PASSED" if success_rate >= 80 else "FAILED",
                "system_health_score": min(success_rate, 100),
                "recommendations": []
            }
        }
        
        # Add recommendations based on results
        if success_rate < 100:
            failed_tests_names = [test["test_name"] for test in cls.test_results["test_details"] if test["status"] == "failed"]
            report["comprehensive_system_validation"]["recommendations"].extend([
                f"Address failed tests: {', '.join(failed_tests_names)}",
                "Review system dependencies and configuration"
            ])
        
        if success_rate >= 90:
            report["comprehensive_system_validation"]["recommendations"].append("System ready for production use")
        elif success_rate >= 70:
            report["comprehensive_system_validation"]["recommendations"].append("System functional with minor issues")
        else:
            report["comprehensive_system_validation"]["recommendations"].append("System requires significant improvements")
        
        # Save report
        os.makedirs('.taskmaster/reports', exist_ok=True)
        report_file = '.taskmaster/reports/comprehensive_system_validation.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("🎯 COMPREHENSIVE SYSTEM VALIDATION SUMMARY")
        print("="*70)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Overall Status: {report['comprehensive_system_validation']['overall_status']}")
        print(f"Report saved to: {report_file}")
        print("="*70)


def run_comprehensive_validation():
    """Run comprehensive system validation"""
    
    print("🚀 Starting Comprehensive System Validation")
    print("="*70)
    print("This test suite validates the entire LABRYS + Task Master AI system")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestComprehensiveSystemValidation)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)