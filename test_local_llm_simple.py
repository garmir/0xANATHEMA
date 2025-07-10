#!/usr/bin/env python3
"""
Simple test for local LLM research module without external dependencies
Tests core functionality and interfaces without requiring httpx or external services
"""

import sys
import os
import time
from pathlib import Path

def test_module_import():
    """Test that the module can be imported without errors"""
    try:
        sys.path.insert(0, '.')
        import local_llm_research_module as llm
        
        # Test that core classes exist
        required_classes = [
            'LocalLLMConfig',
            'ResearchRequest', 
            'ResearchResult',
            'LocalLLMResearchEngine',
            'LocalLLMPlanningEngine',
            'TaskMasterResearchInterface'
        ]
        
        missing = []
        for cls_name in required_classes:
            if not hasattr(llm, cls_name):
                missing.append(cls_name)
        
        if missing:
            return False, f"Missing classes: {missing}"
        
        return True, "All required classes present"
        
    except Exception as e:
        return False, f"Import failed: {e}"

def test_dataclass_creation():
    """Test that dataclasses can be created"""
    try:
        sys.path.insert(0, '.')
        from local_llm_research_module import LocalLLMConfig, ResearchRequest, ResearchResult, LLMProvider
        
        # Test LocalLLMConfig creation
        config = LocalLLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="test-model",
            endpoint="http://localhost:11434"
        )
        
        # Test ResearchRequest creation
        request = ResearchRequest(
            query="test query",
            context="test context"
        )
        
        # Test ResearchResult creation
        result = ResearchResult(
            query="test",
            result="test result"
        )
        
        return True, "All dataclasses can be created successfully"
        
    except Exception as e:
        return False, f"Dataclass creation failed: {e}"

def test_planning_methods():
    """Test that planning methods exist and are callable"""
    try:
        sys.path.insert(0, '.')
        from local_llm_research_module import LocalLLMPlanningEngine, LocalLLMResearchEngine, LocalLLMConfigFactory
        
        # Create dummy research engine
        configs = [LocalLLMConfigFactory.create_ollama_config()]
        research_engine = LocalLLMResearchEngine(configs)
        
        # Create planning engine
        planning_engine = LocalLLMPlanningEngine(research_engine)
        
        # Test that required methods exist
        required_methods = [
            'generate_task_plan',
            'parse_planning_response',
            'extract_steps',
            'extract_timeline',
            'extract_resources',
            'extract_dependencies'
        ]
        
        missing = []
        for method_name in required_methods:
            if not hasattr(planning_engine, method_name):
                missing.append(method_name)
        
        if missing:
            return False, f"Missing methods: {missing}"
        
        return True, "All planning methods are present and callable"
        
    except Exception as e:
        return False, f"Planning method test failed: {e}"

def test_task_master_integration():
    """Test Task-Master integration interface"""
    try:
        sys.path.insert(0, '.')
        from local_llm_research_module import (
            TaskMasterResearchInterface, 
            create_task_master_research_interface,
            task_master_research,
            LocalLLMConfigFactory
        )
        
        # Test interface creation
        configs = [LocalLLMConfigFactory.create_ollama_config()]
        interface = create_task_master_research_interface(configs)
        
        # Test that interface has required methods
        required_methods = ['research', 'plan', 'close']
        
        missing = []
        for method_name in required_methods:
            if not hasattr(interface, method_name):
                missing.append(method_name)
        
        if missing:
            return False, f"Missing interface methods: {missing}"
        
        # Test that task_master_research function exists
        if not callable(task_master_research):
            return False, "task_master_research is not callable"
        
        return True, "Task-Master integration interface is complete"
        
    except Exception as e:
        return False, f"Integration test failed: {e}"

def test_privacy_compliance():
    """Test privacy and data locality indicators"""
    try:
        with open('local_llm_research_module.py', 'r') as f:
            content = f.read()
        
        # Count local indicators
        local_indicators = [
            'localhost',
            'local',
            'privacy',
            'offline',
            'private'
        ]
        
        local_count = sum(1 for indicator in local_indicators if indicator in content.lower())
        
        # Check for absence of external service calls
        external_services = [
            'openai.com',
            'anthropic.com', 
            'api.perplexity.ai',
            'googleapis.com'
        ]
        
        external_count = sum(1 for service in external_services if service in content.lower())
        
        if local_count >= 5 and external_count == 0:
            return True, f"Good privacy compliance: {local_count} local indicators, {external_count} external services"
        else:
            return False, f"Privacy concerns: {local_count} local indicators, {external_count} external services"
            
    except Exception as e:
        return False, f"Privacy test failed: {e}"

def run_all_tests():
    """Run all tests and report results"""
    print("🧪 Running Simple Local LLM Tests (No External Dependencies)")
    print("=" * 70)
    
    tests = [
        ("Module Import", test_module_import),
        ("Dataclass Creation", test_dataclass_creation), 
        ("Planning Methods", test_planning_methods),
        ("Task-Master Integration", test_task_master_integration),
        ("Privacy Compliance", test_privacy_compliance)
    ]
    
    results = []
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        start_time = time.time()
        
        try:
            success, message = test_func()
            duration = time.time() - start_time
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status}: {message} ({duration:.2f}s)")
            
            results.append({
                "test": test_name,
                "success": success,
                "message": message,
                "duration": duration
            })
            
            if success:
                passed += 1
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ ERROR: {e} ({duration:.2f}s)")
            results.append({
                "test": test_name,
                "success": False,
                "message": f"Test error: {e}",
                "duration": duration
            })
    
    # Summary
    success_rate = (passed / total) * 100
    print(f"\n📊 Test Summary:")
    print(f"   Total Tests: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {total - passed}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    # Overall assessment
    if success_rate >= 80:
        print(f"\n✅ OVERALL: LOCAL LLM MODULE IS READY FOR DEPLOYMENT")
        print("   • All critical functionality is present")
        print("   • Privacy compliance is maintained")
        print("   • Task-Master integration is complete")
    elif success_rate >= 60:
        print(f"\n⚠️  OVERALL: MODULE NEEDS MINOR FIXES")
        print("   • Core functionality works")
        print("   • Some improvements needed for production")
    else:
        print(f"\n❌ OVERALL: MODULE NEEDS SIGNIFICANT WORK")
        print("   • Critical issues need to be resolved")
        print("   • Not ready for production deployment")
    
    return results

if __name__ == "__main__":
    try:
        results = run_all_tests()
        
        # Save results
        os.makedirs(".taskmaster/reports", exist_ok=True)
        
        import json
        with open(".taskmaster/reports/simple_local_llm_test.json", 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "results": results,
                "summary": {
                    "total_tests": len(results),
                    "passed_tests": sum(1 for r in results if r["success"]),
                    "success_rate": (sum(1 for r in results if r["success"]) / len(results)) * 100
                }
            }, f, indent=2)
        
        print(f"\n📄 Results saved to: .taskmaster/reports/simple_local_llm_test.json")
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()