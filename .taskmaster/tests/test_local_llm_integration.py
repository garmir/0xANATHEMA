#!/usr/bin/env python3
"""
Comprehensive Integration Test for Local LLM Integration
Tests all refactored modules work together with Local LLM Adapter
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ai.local_llm_adapter import LocalLLMAdapter
from research.autonomous_research_workflow import AutonomousResearchWorkflow
from ai.recursive_self_improvement_engine import RecursiveSelfImprovementEngine
from ai.meta_learning_engine import MetaLearningEngine

def test_local_llm_adapter():
    """Test LocalLLMAdapter functionality"""
    print("🔧 Testing LocalLLMAdapter...")
    
    adapter = LocalLLMAdapter()
    
    # Test adapter status
    status = adapter.get_adapter_status()
    assert isinstance(status, dict)
    assert 'available_models' in status
    assert 'total_models' in status
    
    # Test research query (should fallback gracefully)
    result = adapter.research_query("Test research query about optimization")
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test reasoning request
    result = adapter.reasoning_request("Analyze this problem and provide recommendations")
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test code generation
    result = adapter.code_generation_request("Create a simple Python function")
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test planning request
    result = adapter.planning_request("Create a plan for system optimization")
    assert isinstance(result, str)
    assert len(result) > 0
    
    adapter.shutdown()
    print("✅ LocalLLMAdapter tests passed")

def test_autonomous_research_workflow():
    """Test AutonomousResearchWorkflow with Local LLM integration"""
    print("🔬 Testing AutonomousResearchWorkflow...")
    
    workflow = AutonomousResearchWorkflow(use_local_llm=True)
    
    # Test research cycle
    trigger_context = {
        'execution_time': 25,
        'memory_usage': 0.7,
        'user_count': 500,
        'error_rate': 0.05
    }
    
    cycle_results = workflow.run_autonomous_research_cycle(trigger_context)
    
    # Validate cycle results
    assert isinstance(cycle_results, dict)
    assert 'cycle_id' in cycle_results
    assert 'hypotheses_generated' in cycle_results
    assert 'experiments_designed' in cycle_results
    assert 'findings_discovered' in cycle_results
    assert 'knowledge_synthesized' in cycle_results
    assert cycle_results['hypotheses_generated'] > 0
    
    # Test research status
    status = workflow.get_research_status()
    assert isinstance(status, dict)
    assert 'active_hypotheses' in status
    assert 'knowledge_artifacts' in status
    
    workflow.shutdown()
    print("✅ AutonomousResearchWorkflow tests passed")

def test_recursive_self_improvement_engine():
    """Test RecursiveSelfImprovementEngine with Local LLM integration"""
    print("🔄 Testing RecursiveSelfImprovementEngine...")
    
    rsi_engine = RecursiveSelfImprovementEngine(use_local_llm=True)
    
    # Test improvement session
    cycles = rsi_engine.run_recursive_improvement_session(
        target_components=["test_component"],
        max_cycles=2
    )
    
    # Validate cycles
    assert isinstance(cycles, list)
    assert len(cycles) > 0
    
    for cycle in cycles:
        assert hasattr(cycle, 'cycle_id')
        assert hasattr(cycle, 'candidates_generated')
        assert hasattr(cycle, 'candidates_tested')
        assert hasattr(cycle, 'improvements_applied')
        assert cycle.candidates_generated >= 0
    
    # Test RSI status
    status = rsi_engine.get_rsi_status()
    assert isinstance(status, dict)
    assert 'cycles_completed' in status
    assert 'local_llm_enabled' in status
    assert status['local_llm_enabled'] == True
    
    rsi_engine.shutdown()
    print("✅ RecursiveSelfImprovementEngine tests passed")

def test_meta_learning_engine():
    """Test MetaLearningEngine with Local LLM integration"""
    print("🧠 Testing MetaLearningEngine...")
    
    meta_engine = MetaLearningEngine(use_local_llm=True)
    
    # Test strategy recommendations
    context = {
        'strategy_type': 'code_optimization',
        'performance_metrics': {'execution_time': 15.0}
    }
    
    recommendations = meta_engine.recommend_strategies(context)
    
    # Validate recommendations
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    
    for strategy, score in recommendations:
        assert isinstance(strategy, str)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1
    
    # Test learning summary
    summary = meta_engine.get_learning_summary()
    assert isinstance(summary, dict)
    assert 'current_phase' in summary
    assert 'local_llm_enabled' in summary
    assert summary['local_llm_enabled'] == True
    
    meta_engine.shutdown()
    print("✅ MetaLearningEngine tests passed")

def test_integrated_workflow():
    """Test integrated workflow using all components together"""
    print("🔗 Testing integrated workflow...")
    
    # Initialize all components
    adapter = LocalLLMAdapter()
    workflow = AutonomousResearchWorkflow(use_local_llm=True)
    rsi_engine = RecursiveSelfImprovementEngine(use_local_llm=True)
    meta_engine = MetaLearningEngine(use_local_llm=True)
    
    # Test that all components can work together
    start_time = time.time()
    
    # Research workflow generates hypotheses
    research_context = {'execution_time': 20, 'memory_usage': 0.6}
    research_results = workflow.run_autonomous_research_cycle(research_context)
    
    # RSI engine processes improvements
    rsi_cycles = rsi_engine.run_recursive_improvement_session(max_cycles=1)
    
    # Meta-learning provides strategy recommendations
    meta_context = {'strategy_type': 'process_improvement'}
    recommendations = meta_engine.recommend_strategies(meta_context)
    
    execution_time = time.time() - start_time
    
    # Validate integrated results
    assert research_results['hypotheses_generated'] > 0
    assert len(rsi_cycles) > 0
    assert len(recommendations) > 0
    assert execution_time < 30  # Should complete within 30 seconds
    
    # Clean shutdown
    workflow.shutdown()
    rsi_engine.shutdown()
    meta_engine.shutdown()
    adapter.shutdown()
    
    print(f"✅ Integrated workflow tests passed (execution time: {execution_time:.1f}s)")

def main():
    """Run comprehensive integration tests"""
    print("Local LLM Integration Test Suite")
    print("=" * 50)
    
    test_results = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0
    }
    
    tests = [
        ("LocalLLMAdapter", test_local_llm_adapter),
        ("AutonomousResearchWorkflow", test_autonomous_research_workflow),
        ("RecursiveSelfImprovementEngine", test_recursive_self_improvement_engine),
        ("MetaLearningEngine", test_meta_learning_engine),
        ("IntegratedWorkflow", test_integrated_workflow)
    ]
    
    for test_name, test_func in tests:
        test_results['total_tests'] += 1
        
        try:
            print(f"\n--- Testing {test_name} ---")
            test_func()
            test_results['passed_tests'] += 1
            print(f"✅ {test_name} tests PASSED")
            
        except Exception as e:
            test_results['failed_tests'] += 1
            print(f"❌ {test_name} tests FAILED: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Results Summary:")
    print(f"  Total tests: {test_results['total_tests']}")
    print(f"  Passed: {test_results['passed_tests']}")
    print(f"  Failed: {test_results['failed_tests']}")
    print(f"  Success rate: {(test_results['passed_tests']/test_results['total_tests'])*100:.1f}%")
    
    if test_results['failed_tests'] == 0:
        print("\n🎉 All tests passed! Local LLM integration is working correctly.")
        return True
    else:
        print(f"\n⚠️ {test_results['failed_tests']} tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)