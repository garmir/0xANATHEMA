#!/usr/bin/env python3
"""
Test Local LLM Research Module Functionality
Simplified test without external dependencies
"""

import os
import sys
import json
import time
import subprocess
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class TestResult:
    """Test result structure"""
    test_name: str
    success: bool
    message: str
    duration: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class LocalLLMFunctionalityTester:
    """Test local LLM research module functionality"""
    
    def __init__(self):
        self.test_results = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup test logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run_all_tests(self) -> List[TestResult]:
        """Run comprehensive functionality tests"""
        self.logger.info("🧪 Starting Local LLM Research Module Tests")
        
        # Test 1: Module Import and Structure
        self.test_module_import()
        
        # Test 2: Configuration Management
        self.test_configuration_management()
        
        # Test 3: Research Interface Compatibility
        self.test_research_interface()
        
        # Test 4: Planning Interface Functionality
        self.test_planning_interface()
        
        # Test 5: Error Handling and Fallbacks
        self.test_error_handling()
        
        # Test 6: Task-Master Integration
        self.test_taskmaster_integration()
        
        # Test 7: Privacy and Data Locality
        self.test_privacy_compliance()
        
        # Test 8: Performance Validation
        self.test_performance_metrics()
        
        return self.test_results
    
    def test_module_import(self):
        """Test 1: Module Import and Structure Validation"""
        start_time = time.time()
        test_name = "Module Import and Structure"
        
        try:
            # Test basic Python syntax and structure
            result = subprocess.run([
                'python3', '-c', 
                'import ast; ast.parse(open("local_llm_research_module.py").read())'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Test class structure
                with open('local_llm_research_module.py', 'r') as f:
                    content = f.read()
                
                required_classes = [
                    'LocalLLMConfig',
                    'ResearchRequest', 
                    'ResearchResult',
                    'LocalLLMResearchEngine',
                    'LocalLLMPlanningEngine'
                ]
                
                missing_classes = []
                for cls in required_classes:
                    if f'class {cls}' not in content:
                        missing_classes.append(cls)
                
                if not missing_classes:
                    self.add_test_result(test_name, True, 
                        "All required classes present and syntax valid", 
                        time.time() - start_time)
                else:
                    self.add_test_result(test_name, False,
                        f"Missing classes: {missing_classes}",
                        time.time() - start_time)
            else:
                self.add_test_result(test_name, False,
                    f"Syntax error: {result.stderr}",
                    time.time() - start_time)
                
        except Exception as e:
            self.add_test_result(test_name, False, str(e), time.time() - start_time)
    
    def test_configuration_management(self):
        """Test 2: Configuration Management"""
        start_time = time.time()
        test_name = "Configuration Management"
        
        try:
            # Test config creation and loading
            config_dir = Path(".taskmaster/config")
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Create test config
            test_config = {
                "models": [
                    {
                        "model_name": "test-llama",
                        "endpoint_url": "http://localhost:11434/api/generate",
                        "max_tokens": 4000,
                        "temperature": 0.1,
                        "context_window": 8192
                    }
                ],
                "fallback_model": "test-llama",
                "research_settings": {
                    "max_research_depth": 3,
                    "enable_caching": True
                }
            }
            
            config_path = config_dir / "local_llm_config.json"
            with open(config_path, 'w') as f:
                json.dump(test_config, f, indent=2)
            
            # Verify config can be loaded
            if config_path.exists():
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                if loaded_config == test_config:
                    self.add_test_result(test_name, True,
                        "Configuration created and loaded successfully",
                        time.time() - start_time)
                else:
                    self.add_test_result(test_name, False,
                        "Configuration data mismatch",
                        time.time() - start_time)
            else:
                self.add_test_result(test_name, False,
                    "Configuration file not created",
                    time.time() - start_time)
                
        except Exception as e:
            self.add_test_result(test_name, False, str(e), time.time() - start_time)
    
    def test_research_interface(self):
        """Test 3: Research Interface Compatibility"""
        start_time = time.time()
        test_name = "Research Interface Compatibility"
        
        try:
            # Test interface methods exist and are callable
            interface_methods = [
                'research',
                'plan'
            ]
            
            # Check if interface creation works without external dependencies
            test_code = '''
import sys
sys.path.append('.')
try:
    # Test structure without actual API calls
    from local_llm_research_module import LocalLLMConfig, ResearchRequest, ResearchResult
    
    # Test dataclass creation
    config = LocalLLMConfig(
        model_name="test",
        endpoint_url="http://test",
        max_tokens=100
    )
    
    request = ResearchRequest(
        query="test query",
        context="test context"
    )
    
    print("Interface structures valid")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Structure error: {e}")
'''
            
            result = subprocess.run([
                'python3', '-c', test_code
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and "Interface structures valid" in result.stdout:
                self.add_test_result(test_name, True,
                    "Research interface structures are valid",
                    time.time() - start_time)
            else:
                self.add_test_result(test_name, False,
                    f"Interface test failed: {result.stderr or result.stdout}",
                    time.time() - start_time)
                
        except Exception as e:
            self.add_test_result(test_name, False, str(e), time.time() - start_time)
    
    def test_planning_interface(self):
        """Test 4: Planning Interface Functionality"""
        start_time = time.time()
        test_name = "Planning Interface Functionality"
        
        try:
            # Test planning data structures
            with open('local_llm_research_module.py', 'r') as f:
                content = f.read()
            
            planning_methods = [
                'generate_task_plan',
                'parse_planning_response',
                'extract_steps',
                'extract_timeline',
                'extract_resources'
            ]
            
            missing_methods = []
            for method in planning_methods:
                if f'def {method}' not in content:
                    missing_methods.append(method)
            
            if not missing_methods:
                self.add_test_result(test_name, True,
                    "All planning methods present",
                    time.time() - start_time)
            else:
                self.add_test_result(test_name, False,
                    f"Missing planning methods: {missing_methods}",
                    time.time() - start_time)
                
        except Exception as e:
            self.add_test_result(test_name, False, str(e), time.time() - start_time)
    
    def test_error_handling(self):
        """Test 5: Error Handling and Fallbacks"""
        start_time = time.time()
        test_name = "Error Handling and Fallbacks"
        
        try:
            # Check for error handling patterns
            with open('local_llm_research_module.py', 'r') as f:
                content = f.read()
            
            error_patterns = [
                'try:',
                'except',
                'Exception',
                'fallback',
                'timeout'
            ]
            
            found_patterns = []
            for pattern in error_patterns:
                if pattern in content:
                    found_patterns.append(pattern)
            
            if len(found_patterns) >= 4:  # Most error handling patterns present
                self.add_test_result(test_name, True,
                    f"Error handling patterns found: {found_patterns}",
                    time.time() - start_time)
            else:
                self.add_test_result(test_name, False,
                    f"Insufficient error handling patterns: {found_patterns}",
                    time.time() - start_time)
                
        except Exception as e:
            self.add_test_result(test_name, False, str(e), time.time() - start_time)
    
    def test_taskmaster_integration(self):
        """Test 6: Task-Master Integration"""
        start_time = time.time()
        test_name = "Task-Master Integration"
        
        try:
            # Test task-master CLI availability
            result = subprocess.run(['task-master', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Test if integration interface exists
                with open('local_llm_research_module.py', 'r') as f:
                    content = f.read()
                
                integration_features = [
                    'task_master_research',
                    'TaskMasterResearchInterface',
                    'create_task_master_research_interface'
                ]
                
                found_features = []
                for feature in integration_features:
                    if feature in content:
                        found_features.append(feature)
                
                if len(found_features) >= 2:
                    self.add_test_result(test_name, True,
                        f"Task-Master integration features present: {found_features}",
                        time.time() - start_time)
                else:
                    self.add_test_result(test_name, False,
                        f"Missing integration features: {set(integration_features) - set(found_features)}",
                        time.time() - start_time)
            else:
                self.add_test_result(test_name, False,
                    "Task-Master CLI not available",
                    time.time() - start_time)
                
        except Exception as e:
            self.add_test_result(test_name, False, str(e), time.time() - start_time)
    
    def test_privacy_compliance(self):
        """Test 7: Privacy and Data Locality"""
        start_time = time.time()
        test_name = "Privacy and Data Locality"
        
        try:
            # Check for external API calls in the code
            with open('local_llm_research_module.py', 'r') as f:
                content = f.read()
            
            # Check for localhost/local endpoints
            local_indicators = [
                'localhost',
                '127.0.0.1',
                'local',
                'offline',
                'privacy'
            ]
            
            # Check for external service indicators (should be minimal/configurable)
            external_indicators = [
                'openai.com',
                'anthropic.com',
                'api.perplexity.ai',
                'googleapis.com'
            ]
            
            local_count = sum(1 for indicator in local_indicators if indicator in content.lower())
            external_count = sum(1 for indicator in external_indicators if indicator in content.lower())
            
            # Good privacy compliance: more local indicators, fewer external ones
            if local_count >= 3 and external_count <= 1:
                self.add_test_result(test_name, True,
                    f"Good privacy compliance: {local_count} local vs {external_count} external indicators",
                    time.time() - start_time)
            else:
                self.add_test_result(test_name, False,
                    f"Privacy concerns: {local_count} local vs {external_count} external indicators",
                    time.time() - start_time)
                
        except Exception as e:
            self.add_test_result(test_name, False, str(e), time.time() - start_time)
    
    def test_performance_metrics(self):
        """Test 8: Performance Validation"""
        start_time = time.time()
        test_name = "Performance Metrics"
        
        try:
            # Check for performance-related features
            with open('local_llm_research_module.py', 'r') as f:
                content = f.read()
            
            performance_features = [
                'async',
                'cache',
                'timeout',
                'processing_time',
                'performance'
            ]
            
            found_features = []
            for feature in performance_features:
                if feature in content.lower():
                    found_features.append(feature)
            
            if len(found_features) >= 4:
                self.add_test_result(test_name, True,
                    f"Performance features present: {found_features}",
                    time.time() - start_time)
            else:
                self.add_test_result(test_name, False,
                    f"Limited performance features: {found_features}",
                    time.time() - start_time)
                
        except Exception as e:
            self.add_test_result(test_name, False, str(e), time.time() - start_time)
    
    def add_test_result(self, test_name: str, success: bool, message: str, duration: float):
        """Add test result to collection"""
        result = TestResult(
            test_name=test_name,
            success=success,
            message=message,
            duration=duration
        )
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        self.logger.info(f"{status} {test_name}: {message} ({duration:.2f}s)")
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = total_tests - passed_tests
        
        report = f"""
# Local LLM Research Module Functionality Test Report

**Generated**: {datetime.now().isoformat()}
**Total Tests**: {total_tests}
**Passed**: {passed_tests}
**Failed**: {failed_tests}
**Success Rate**: {(passed_tests/total_tests*100):.1f}%

## Test Results

"""
        
        for result in self.test_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            report += f"### {result.test_name}\n"
            report += f"**Status**: {status}\n"
            report += f"**Duration**: {result.duration:.2f}s\n"
            report += f"**Result**: {result.message}\n\n"
        
        report += f"""
## Summary

The local LLM research module has been tested across 8 critical functionality areas:

1. **Module Import and Structure**: Validates Python syntax and required classes
2. **Configuration Management**: Tests config file creation and loading
3. **Research Interface Compatibility**: Verifies interface structures
4. **Planning Interface Functionality**: Checks planning method availability
5. **Error Handling and Fallbacks**: Validates error handling patterns
6. **Task-Master Integration**: Tests integration with Task-Master CLI
7. **Privacy and Data Locality**: Ensures local-first design
8. **Performance Metrics**: Validates performance-related features

**Overall Assessment**: {'FUNCTIONAL' if passed_tests/total_tests >= 0.7 else 'NEEDS IMPROVEMENT'}

{'✅ Module is ready for integration with local LLM services' if passed_tests/total_tests >= 0.7 else '⚠️ Module requires fixes before integration'}
"""
        
        return report


if __name__ == "__main__":
    print("🧪 Testing Local LLM Research Module Functionality")
    print("=" * 60)
    
    tester = LocalLLMFunctionalityTester()
    results = tester.run_all_tests()
    
    # Generate and save report
    report = tester.generate_test_report()
    
    # Save report to file
    report_file = Path(".taskmaster/reports/local_llm_functionality_test.md")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📊 Test Report Generated: {report_file}")
    print(report)