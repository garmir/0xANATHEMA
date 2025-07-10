#!/usr/bin/env python3
import unittest
import sys
import os
from pathlib import Path

# Setup test environment
try:
    from test_config import setup_test_environment, RobustTestCase
    setup_test_environment()
except:
    pass

class ImprovedTestRunner:
    """Improved test runner with better error handling"""
    
    def __init__(self):
        self.test_results = []
        
    def run_tests_safely(self, test_modules):
        """Run tests with improved error handling"""
        
        for module_name in test_modules:
            try:
                # Import test module
                module = __import__(module_name)
                
                # Load and run tests
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromModule(module)
                
                runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
                result = runner.run(suite)
                
                self.test_results.append({
                    "module": module_name,
                    "tests_run": result.testsRun,
                    "failures": len(result.failures),
                    "errors": len(result.errors),
                    "success": result.wasSuccessful()
                })
                
            except Exception as e:
                print(f"⚠️ Could not run tests for {module_name}: {e}")
                self.test_results.append({
                    "module": module_name,
                    "error": str(e),
                    "success": False
                })
        
        return self.test_results

if __name__ == "__main__":
    runner = ImprovedTestRunner()
    # Add test modules here
    results = runner.run_tests_safely(['test_comprehensive_system_validation'])
    
    # Print summary
    total_modules = len(results)
    successful_modules = len([r for r in results if r.get('success', False)])
    
    print(f"\n📊 Test Summary: {successful_modules}/{total_modules} modules successful")
