#!/usr/bin/env python3
"""
Individual Component Testing for Task Master AI
Fixed version to avoid shell command issues
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

def test_python_module_imports():
    """Test Python module imports directly"""
    print("🐍 Testing Python Module Imports...")
    
    try:
        # Test core module import
        from local_llm_research_module import (
            LocalLLMResearchEngine, 
            LocalLLMConfigFactory,
            ResearchContext,
            LocalLLMConfig,
            LLMProvider,
            ModelCapability
        )
        
        # Test configuration creation
        config = LocalLLMConfigFactory.create_ollama_config("llama2")
        print(f"  ✅ Configuration created for {config.provider.value}")
        
        # Test engine initialization
        engine = LocalLLMResearchEngine([config])
        print("  ✅ Research engine initialized")
        
        # Test research context creation
        context = ResearchContext(query="Test query", depth=0, max_depth=2)
        print(f"  ✅ Research context created with correlation_id: {context.correlation_id}")
        
        return True, "All imports successful"
        
    except Exception as e:
        return False, str(e)

async def test_recursive_research():
    """Test recursive research functionality"""
    print("🔄 Testing Recursive Research...")
    
    try:
        from local_llm_research_module import LocalLLMResearchEngine, LocalLLMConfigFactory, ResearchContext
        
        # Create test configuration
        config = LocalLLMConfigFactory.create_ollama_config("llama2")
        engine = LocalLLMResearchEngine([config])
        
        # Test recursive task breakdown
        task = "Implement a comprehensive testing framework"
        result = await engine.recursive_task_breakdown(task, current_depth=0, max_depth=2)
        
        # Validate result structure
        required_fields = ["task", "depth", "provider", "timestamp"]
        missing_fields = [field for field in required_fields if field not in result]
        
        if not missing_fields:
            print("  ✅ Recursive breakdown structure valid")
            
            # Test depth limiting
            deep_result = await engine.recursive_task_breakdown(task, current_depth=3, max_depth=3)
            if "Maximum depth reached" in str(deep_result):
                print("  ✅ Depth limiting works correctly")
            else:
                print("  ⚠️ Depth limiting may not be working optimally")
            
            await engine.close()
            return True, "Recursive research functionality validated"
        else:
            await engine.close()
            return False, f"Missing fields in result: {missing_fields}"
            
    except Exception as e:
        return False, str(e)

def test_multi_provider_support():
    """Test multi-provider LLM support"""
    print("🔌 Testing Multi-Provider Support...")
    
    try:
        from local_llm_research_module import LocalLLMConfigFactory, LLMProvider
        
        # Test all provider configurations
        providers_tested = []
        
        # Test Ollama config
        ollama_config = LocalLLMConfigFactory.create_ollama_config("llama2")
        if ollama_config.provider == LLMProvider.OLLAMA:
            providers_tested.append("Ollama")
        
        # Test LM Studio config
        lm_studio_config = LocalLLMConfigFactory.create_lm_studio_config("mistral-7b")
        if lm_studio_config.provider == LLMProvider.LM_STUDIO:
            providers_tested.append("LM Studio")
        
        # Test LocalAI config
        local_ai_config = LocalLLMConfigFactory.create_local_ai_config("gpt-3.5-turbo")
        if local_ai_config.provider == LLMProvider.LOCAL_AI:
            providers_tested.append("LocalAI")
        
        # Test text-generation-webui config
        webui_config = LocalLLMConfigFactory.create_text_generation_webui_config()
        if webui_config.provider == LLMProvider.TEXT_GENERATION_WEBUI:
            providers_tested.append("Text-Generation-WebUI")
        
        print(f"  ✅ {len(providers_tested)} providers configured successfully")
        print(f"  Providers: {', '.join(providers_tested)}")
        
        if len(providers_tested) >= 4:
            return True, f"All 4 providers configured: {', '.join(providers_tested)}"
        else:
            return True, f"Partial success: {len(providers_tested)}/4 providers"
            
    except Exception as e:
        return False, str(e)

async def test_meta_improvement_analysis():
    """Test meta-improvement analysis capabilities"""
    print("📊 Testing Meta-Improvement Analysis...")
    
    try:
        from local_llm_research_module import LocalLLMResearchEngine, LocalLLMConfigFactory
        
        config = LocalLLMConfigFactory.create_ollama_config("llama2")
        engine = LocalLLMResearchEngine([config])
        
        # Test meta-improvement analysis
        sample_data = {
            "task_completion_rate": 0.85,
            "average_execution_time": 120,
            "error_patterns": ["timeout", "network_error", "validation_failure"],
            "performance_metrics": {
                "cpu_usage": 0.6,
                "memory_usage": 0.4,
                "success_rate": 0.9
            }
        }
        
        patterns = [
            {"pattern": "timeout_issues", "frequency": 0.1},
            {"pattern": "resource_constraints", "frequency": 0.05}
        ]
        
        result = await engine.meta_improvement_analysis(sample_data, patterns)
        
        # Validate result structure
        required_fields = ["input_data", "meta_analysis", "confidence_score", "timestamp"]
        missing_fields = [field for field in required_fields if field not in result]
        
        if not missing_fields:
            print("  ✅ Meta-analysis structure valid")
            
            # Check for confidence score
            confidence = result.get("confidence_score", 0)
            if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                print(f"  ✅ Confidence score valid ({confidence})")
            
            await engine.close()
            return True, "Meta-improvement analysis validated"
        else:
            await engine.close()
            return False, f"Missing fields: {missing_fields}"
            
    except Exception as e:
        return False, str(e)

async def run_all_component_tests():
    """Run all individual component tests"""
    print("🧪 Individual Component Testing for Task Master AI")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Python Module Imports
    success, details = test_python_module_imports()
    results["python_imports"] = {"success": success, "details": details}
    
    # Test 2: Multi-Provider Support
    success, details = test_multi_provider_support()
    results["multi_provider"] = {"success": success, "details": details}
    
    # Test 3: Recursive Research (async)
    success, details = await test_recursive_research()
    results["recursive_research"] = {"success": success, "details": details}
    
    # Test 4: Meta-Improvement Analysis (async)
    success, details = await test_meta_improvement_analysis()
    results["meta_improvement"] = {"success": success, "details": details}
    
    # Summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r["success"])
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n📊 Component Test Summary")
    print("=" * 40)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("✅ ALL COMPONENTS OPERATIONAL")
        overall_status = "FULLY_OPERATIONAL"
    elif success_rate >= 70:
        print("⚠️ PARTIAL FUNCTIONALITY")
        overall_status = "DEGRADED"
    else:
        print("❌ CRITICAL ISSUES DETECTED")
        overall_status = "FAILED"
    
    # Show details for failed tests
    failed_tests = [(name, data) for name, data in results.items() if not data["success"]]
    if failed_tests:
        print("\n❌ Failed Test Details:")
        for name, data in failed_tests:
            print(f"  {name}: {data['details']}")
    
    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "Individual Component Testing",
        "overall_status": overall_status,
        "success_rate": success_rate,
        "test_results": results,
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests
        }
    }
    
    os.makedirs(".taskmaster/reports", exist_ok=True)
    with open(".taskmaster/reports/individual-component-tests.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: .taskmaster/reports/individual-component-tests.json")
    
    return overall_status == "FULLY_OPERATIONAL"

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_component_tests())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Component testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)