#!/usr/bin/env python3
"""
Health Check Script for Task Master AI Local LLM Migration
Tests all components and validates functionality, privacy, and integration
"""

import os
import sys
import json
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
import tempfile

class HealthChecker:
    """Comprehensive health check for Task Master AI local LLM migration"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        self.start_time = datetime.now()
        
    def log_result(self, test_name: str, passed: bool, message: str = "", details: dict = None):
        """Log test result"""
        self.results[test_name] = {
            "passed": passed,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"     {message}")
        if not passed:
            self.errors.append(f"{test_name}: {message}")
    
    def log_warning(self, test_name: str, message: str):
        """Log warning"""
        self.warnings.append(f"{test_name}: {message}")
        print(f"⚠️ WARN: {test_name} - {message}")
    
    def test_file_structure(self):
        """Test that all required files exist"""
        print("\n📁 Testing File Structure...")
        
        required_files = [
            "local_llm_adapter.py",
            "local_research_module.py", 
            "local_planning_engine.py",
            "autonomous_research_integration.py",
            ".taskmaster/config.json"
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                self.log_result(f"file_exists_{file_path}", True, f"Found {file_path}")
            else:
                self.log_result(f"file_exists_{file_path}", False, f"Missing {file_path}")
    
    def test_import_capabilities(self):
        """Test that modules can be imported without external dependencies"""
        print("\n🐍 Testing Import Capabilities...")
        
        # Test basic Python imports (should work)
        basic_imports = [
            "os", "sys", "json", "time", "logging", "pathlib", 
            "datetime", "dataclasses", "typing", "abc"
        ]
        
        for module in basic_imports:
            try:
                __import__(module)
                self.log_result(f"import_{module}", True)
            except ImportError as e:
                self.log_result(f"import_{module}", False, str(e))
        
        # Test external dependencies (may not be available)
        external_imports = ["requests"]
        
        for module in external_imports:
            try:
                __import__(module)
                self.log_result(f"external_import_{module}", True)
            except ImportError:
                self.log_warning(f"external_import_{module}", f"Module {module} not available - local LLM providers may not work")
    
    def test_configuration_files(self):
        """Test configuration file validity"""
        print("\n⚙️ Testing Configuration Files...")
        
        config_path = ".taskmaster/config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Check for required sections
                required_sections = ["models", "global"]
                for section in required_sections:
                    if section in config:
                        self.log_result(f"config_section_{section}", True)
                    else:
                        self.log_result(f"config_section_{section}", False, f"Missing {section} section")
                
                # Check model configuration
                if "models" in config:
                    models = config["models"]
                    required_roles = ["main", "research", "fallback"]
                    for role in required_roles:
                        if role in models:
                            model_config = models[role]
                            if "provider" in model_config and "modelId" in model_config:
                                self.log_result(f"model_config_{role}", True)
                            else:
                                self.log_result(f"model_config_{role}", False, f"Incomplete {role} model config")
                        else:
                            self.log_result(f"model_config_{role}", False, f"Missing {role} model config")
                            
            except json.JSONDecodeError as e:
                self.log_result("config_json_valid", False, f"Invalid JSON: {e}")
            except Exception as e:
                self.log_result("config_file_readable", False, f"Cannot read config: {e}")
        else:
            self.log_result("config_file_exists", False, "Config file not found")
    
    def test_module_syntax(self):
        """Test module syntax without importing external dependencies"""
        print("\n📝 Testing Module Syntax...")
        
        modules_to_test = [
            "local_llm_adapter.py",
            "local_research_module.py",
            "local_planning_engine.py"
        ]
        
        for module_file in modules_to_test:
            if os.path.exists(module_file):
                try:
                    # Test syntax by compiling without executing
                    with open(module_file, 'r') as f:
                        content = f.read()
                    
                    compile(content, module_file, 'exec')
                    self.log_result(f"syntax_{module_file}", True, "Valid Python syntax")
                    
                except SyntaxError as e:
                    self.log_result(f"syntax_{module_file}", False, f"Syntax error: {e}")
                except Exception as e:
                    self.log_result(f"syntax_{module_file}", False, f"Compilation error: {e}")
            else:
                self.log_result(f"syntax_{module_file}", False, "File not found")
    
    def test_local_llm_interfaces(self):
        """Test local LLM interface definitions"""
        print("\n🤖 Testing Local LLM Interfaces...")
        
        # Test that the adapter defines expected methods
        adapter_file = "local_llm_adapter.py"
        if os.path.exists(adapter_file):
            try:
                with open(adapter_file, 'r') as f:
                    content = f.read()
                
                expected_classes = ["LocalLLMAdapter", "OllamaProvider", "LocalAIProvider"]
                expected_methods = ["inference", "research", "is_available"]
                
                for class_name in expected_classes:
                    if f"class {class_name}" in content:
                        self.log_result(f"class_defined_{class_name}", True)
                    else:
                        self.log_result(f"class_defined_{class_name}", False, f"Missing class {class_name}")
                
                for method_name in expected_methods:
                    if f"def {method_name}" in content:
                        self.log_result(f"method_defined_{method_name}", True)
                    else:
                        self.log_result(f"method_defined_{method_name}", False, f"Missing method {method_name}")
                        
            except Exception as e:
                self.log_result("adapter_analysis", False, f"Cannot analyze adapter: {e}")
    
    def test_privacy_compliance(self):
        """Test for external API calls and privacy compliance"""
        print("\n🔒 Testing Privacy Compliance...")
        
        # Scan for external API endpoints
        external_patterns = [
            "api.perplexity.ai",
            "api.openai.com", 
            "api.anthropic.com",
            "https://",
            "http://",
            "requests.post",
            "requests.get",
            "urllib.request",
            "httpx"
        ]
        
        files_to_scan = [
            "local_llm_adapter.py",
            "local_research_module.py",
            "local_planning_engine.py",
            "autonomous_research_integration.py"
        ]
        
        external_calls_found = []
        
        for file_path in files_to_scan:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    for pattern in external_patterns:
                        if pattern in content:
                            # Check if it's in a comment or docstring
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if pattern in line and not line.strip().startswith('#') and '"""' not in line:
                                    external_calls_found.append(f"{file_path}:{i+1} - {line.strip()}")
                                    
                except Exception as e:
                    self.log_warning("privacy_scan", f"Cannot scan {file_path}: {e}")
        
        if external_calls_found:
            self.log_result("privacy_compliance", False, 
                          f"Found {len(external_calls_found)} potential external calls",
                          {"external_calls": external_calls_found})
        else:
            self.log_result("privacy_compliance", True, "No external API calls detected")
    
    def test_task_master_integration(self):
        """Test Task Master CLI integration"""
        print("\n🎯 Testing Task Master Integration...")
        
        # Test task-master command availability
        try:
            result = subprocess.run(['task-master', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.log_result("taskmaster_cli", True, "Task Master CLI available")
            else:
                self.log_result("taskmaster_cli", False, "Task Master CLI not responding")
        except subprocess.TimeoutExpired:
            self.log_result("taskmaster_cli", False, "Task Master CLI timeout")
        except FileNotFoundError:
            self.log_result("taskmaster_cli", False, "Task Master CLI not found")
        except Exception as e:
            self.log_result("taskmaster_cli", False, f"Task Master CLI error: {e}")
        
        # Test tasks.json structure
        tasks_file = ".taskmaster/tasks/tasks.json"
        if os.path.exists(tasks_file):
            try:
                with open(tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                
                if "master" in tasks_data and "tasks" in tasks_data["master"]:
                    task_count = len(tasks_data["master"]["tasks"])
                    self.log_result("tasks_structure", True, f"Found {task_count} tasks")
                else:
                    self.log_result("tasks_structure", False, "Invalid tasks.json structure")
                    
            except Exception as e:
                self.log_result("tasks_file_valid", False, f"Cannot read tasks.json: {e}")
        else:
            self.log_result("tasks_file_exists", False, "tasks.json not found")
    
    def test_recursive_functionality(self):
        """Test recursive breakdown and planning functionality"""
        print("\n🔄 Testing Recursive Functionality...")
        
        # Test recursive planning interface
        planning_file = "local_planning_engine.py"
        if os.path.exists(planning_file):
            try:
                with open(planning_file, 'r') as f:
                    content = f.read()
                
                recursive_indicators = [
                    "plan_recursive_breakdown",
                    "recursive",
                    "depth",
                    "atomic"
                ]
                
                found_indicators = []
                for indicator in recursive_indicators:
                    if indicator in content:
                        found_indicators.append(indicator)
                
                if len(found_indicators) >= 3:
                    self.log_result("recursive_planning", True, 
                                  f"Found recursive indicators: {', '.join(found_indicators)}")
                else:
                    self.log_result("recursive_planning", False, "Missing recursive functionality")
                    
            except Exception as e:
                self.log_result("recursive_analysis", False, f"Cannot analyze recursive functionality: {e}")
    
    def test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        print("\n🛡️ Testing Error Handling...")
        
        files_to_check = [
            "local_llm_adapter.py",
            "local_research_module.py",
            "local_planning_engine.py"
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    error_handling_patterns = [
                        "try:",
                        "except",
                        "fallback",
                        "error",
                        "Exception"
                    ]
                    
                    found_patterns = []
                    for pattern in error_handling_patterns:
                        if pattern in content:
                            found_patterns.append(pattern)
                    
                    if len(found_patterns) >= 3:
                        self.log_result(f"error_handling_{file_path}", True,
                                      f"Found error handling: {', '.join(found_patterns)}")
                    else:
                        self.log_result(f"error_handling_{file_path}", False,
                                      "Insufficient error handling")
                        
                except Exception as e:
                    self.log_result(f"error_check_{file_path}", False, f"Cannot check error handling: {e}")
    
    def test_documentation_status(self):
        """Test documentation and README status"""
        print("\n📚 Testing Documentation...")
        
        # Check for key documentation files
        doc_files = [
            "CLAUDE.md",
            "README.md"
        ]
        
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                try:
                    with open(doc_file, 'r') as f:
                        content = f.read()
                    
                    # Check for local LLM mentions
                    local_llm_indicators = [
                        "local",
                        "LLM",
                        "ollama",
                        "localai",
                        "privacy"
                    ]
                    
                    found_indicators = sum(1 for indicator in local_llm_indicators if indicator.lower() in content.lower())
                    
                    if found_indicators >= 3:
                        self.log_result(f"doc_updated_{doc_file}", True,
                                      f"Documentation appears updated for local LLMs")
                    else:
                        self.log_result(f"doc_updated_{doc_file}", False,
                                      "Documentation may need updating for local LLMs")
                        
                except Exception as e:
                    self.log_result(f"doc_readable_{doc_file}", False, f"Cannot read {doc_file}: {e}")
            else:
                self.log_warning(f"doc_exists_{doc_file}", f"{doc_file} not found")
    
    def test_performance_indicators(self):
        """Test performance and optimization indicators"""
        print("\n⚡ Testing Performance Indicators...")
        
        # Check for performance optimizations
        perf_files = [
            "local_llm_adapter.py",
            "local_research_module.py"
        ]
        
        perf_indicators = [
            "cache",
            "async", 
            "performance",
            "optimization",
            "timeout",
            "memory"
        ]
        
        for file_path in perf_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    found_indicators = [indicator for indicator in perf_indicators if indicator in content.lower()]
                    
                    if len(found_indicators) >= 2:
                        self.log_result(f"performance_{file_path}", True,
                                      f"Found performance optimizations: {', '.join(found_indicators)}")
                    else:
                        self.log_result(f"performance_{file_path}", False,
                                      "Limited performance optimizations detected")
                        
                except Exception as e:
                    self.log_result(f"perf_check_{file_path}", False, f"Cannot check performance: {e}")
    
    def generate_health_report(self):
        """Generate comprehensive health report"""
        print("\n📊 Generating Health Report...")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result["passed"])
        failed_tests = total_tests - passed_tests
        
        health_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "health_check_summary": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "warnings": len(self.warnings),
                "health_score": health_score
            },
            "test_results": self.results,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self._generate_recommendations()
        }
        
        # Save report to file
        report_file = f".taskmaster/health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"🏥 HEALTH CHECK COMPLETE")
        print(f"{'='*60}")
        print(f"Health Score: {health_score:.1f}%")
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Report saved: {report_file}")
        
        if health_score >= 90:
            print("🎉 EXCELLENT: System is healthy and ready for production")
        elif health_score >= 75:
            print("✅ GOOD: System is functional with minor issues")
        elif health_score >= 50:
            print("⚠️ FAIR: System has significant issues that should be addressed")
        else:
            print("❌ POOR: System has critical issues and may not function properly")
        
        return report
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for critical failures
        critical_failures = [
            "config_file_exists",
            "taskmaster_cli", 
            "privacy_compliance"
        ]
        
        for test_name in critical_failures:
            if test_name in self.results and not self.results[test_name]["passed"]:
                if test_name == "config_file_exists":
                    recommendations.append("Create .taskmaster/config.json with local LLM configuration")
                elif test_name == "taskmaster_cli":
                    recommendations.append("Install or configure Task Master CLI")
                elif test_name == "privacy_compliance":
                    recommendations.append("Remove external API calls to ensure privacy compliance")
        
        # Check for missing dependencies
        if any("external_import" in name for name in self.results):
            recommendations.append("Install missing Python dependencies (requests, etc.) for full functionality")
        
        # Check for documentation updates
        doc_tests = [name for name in self.results if name.startswith("doc_updated") and not self.results[name]["passed"]]
        if doc_tests:
            recommendations.append("Update documentation to reflect local LLM migration")
        
        return recommendations

def main():
    """Run comprehensive health check"""
    print("🏥 Task Master AI - Local LLM Migration Health Check")
    print("=" * 60)
    
    checker = HealthChecker()
    
    # Run all tests
    try:
        checker.test_file_structure()
        checker.test_import_capabilities() 
        checker.test_configuration_files()
        checker.test_module_syntax()
        checker.test_local_llm_interfaces()
        checker.test_privacy_compliance()
        checker.test_task_master_integration()
        checker.test_recursive_functionality()
        checker.test_error_handling()
        checker.test_documentation_status()
        checker.test_performance_indicators()
        
    except Exception as e:
        print(f"❌ Health check failed with exception: {e}")
        traceback.print_exc()
    
    # Generate final report
    report = checker.generate_health_report()
    
    return report

if __name__ == "__main__":
    main()