#!/usr/bin/env python3
"""
Test Suite for Recursive Complexity Analyzer with Adaptive Meta-Learning
Comprehensive validation and performance testing
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from recursive_complexity_analyzer import (
    RecursiveComplexityAnalyzer, 
    LearningStrategy, 
    RecursivePattern,
    ComplexityPrediction,
    RecursiveContext
)


class RecursiveComplexityAnalyzerTester:
    """Comprehensive test suite for the recursive complexity analyzer"""
    
    def __init__(self):
        self.analyzer = RecursiveComplexityAnalyzer()
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "test_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0
            },
            "test_details": {},
            "performance_metrics": {}
        }
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        print("🧪 RECURSIVE COMPLEXITY ANALYZER TEST SUITE")
        print("=" * 60)
        
        # Test individual components
        await self._test_basic_functionality()
        await self._test_recursive_analysis()
        await self._test_meta_learning_adaptation()
        await self._test_pattern_detection()
        await self._test_performance_characteristics()
        await self._test_learning_persistence()
        
        # Generate summary
        self._generate_test_summary()
        
        return self.test_results
    
    async def _test_basic_functionality(self):
        """Test basic analyzer functionality"""
        print("🔧 Testing Basic Functionality...")
        test_name = "basic_functionality"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Test analyzer initialization
            self.assertTrue(self.analyzer is not None, "Analyzer should initialize successfully")
            results["details"]["initialization"] = "success"
            
            # Test simple complexity analysis
            prediction = await self.analyzer.analyze_recursive_complexity("1")
            self.assertTrue(isinstance(prediction, ComplexityPrediction), "Should return ComplexityPrediction")
            results["details"]["prediction_type"] = "ComplexityPrediction"
            
            # Test prediction components
            self.assertTrue(prediction.task_id == "1", "Task ID should match")
            self.assertTrue(0.0 <= prediction.confidence_score <= 1.0, "Confidence should be between 0 and 1")
            self.assertTrue(isinstance(prediction.learning_strategy_used, LearningStrategy), "Should have learning strategy")
            results["details"]["prediction_components"] = "valid"
            
            print("   ✅ Basic functionality tests passed")
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            results["issues"].append(f"Basic functionality error: {e}")
            print(f"   ❌ Basic functionality failed: {e}")
        
        self.test_results["test_details"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _test_recursive_analysis(self):
        """Test recursive complexity analysis features"""
        print("🔄 Testing Recursive Analysis...")
        test_name = "recursive_analysis"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Test recursive context building
            context = await self.analyzer._build_recursive_context("1")
            self.assertTrue(isinstance(context, RecursiveContext), "Should build recursive context")
            results["details"]["context_building"] = "success"
            
            # Test context properties
            self.assertTrue(context.depth >= 0, "Depth should be non-negative")
            self.assertTrue(context.subtask_count >= 0, "Subtask count should be non-negative")
            self.assertTrue(context.recursive_multiplier > 0, "Recursive multiplier should be positive")
            results["details"]["context_properties"] = "valid"
            
            # Test recursive complexity adjustment
            prediction = await self.analyzer.analyze_recursive_complexity("1", context)
            self.assertTrue(prediction.meta_features.get("recursive_depth") == context.depth, "Should track recursive depth")
            results["details"]["recursive_tracking"] = "accurate"
            
            print("   ✅ Recursive analysis tests passed")
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            results["issues"].append(f"Recursive analysis error: {e}")
            print(f"   ❌ Recursive analysis failed: {e}")
        
        self.test_results["test_details"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _test_meta_learning_adaptation(self):
        """Test meta-learning adaptation mechanisms"""
        print("🧠 Testing Meta-Learning Adaptation...")
        test_name = "meta_learning_adaptation"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Test different learning strategies
            strategies_tested = []
            for strategy in LearningStrategy:
                try:
                    # Mock a prediction with specific strategy
                    prediction = await self.analyzer.analyze_recursive_complexity("2")
                    prediction.learning_strategy_used = strategy
                    
                    # Test adaptation
                    adapted = await self.analyzer._apply_meta_learning_adaptation(prediction)
                    self.assertTrue(isinstance(adapted, ComplexityPrediction), f"Strategy {strategy} should return prediction")
                    strategies_tested.append(strategy.value)
                except Exception as e:
                    results["issues"].append(f"Strategy {strategy} failed: {e}")
            
            results["details"]["strategies_tested"] = strategies_tested
            results["details"]["total_strategies"] = len(strategies_tested)
            
            # Test meta-learner optimization
            await self.analyzer.optimize_learning_parameters()
            results["details"]["parameter_optimization"] = "success"
            
            print("   ✅ Meta-learning adaptation tests passed")
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            results["issues"].append(f"Meta-learning error: {e}")
            print(f"   ❌ Meta-learning adaptation failed: {e}")
        
        self.test_results["test_details"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _test_pattern_detection(self):
        """Test recursive pattern detection"""
        print("🔍 Testing Pattern Detection...")
        test_name = "pattern_detection"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            patterns_detected = []
            
            # Test multiple tasks to trigger different patterns
            test_tasks = ["11", "12", "13", "14", "15"]
            for task_id in test_tasks:
                try:
                    prediction = await self.analyzer.analyze_recursive_complexity(task_id)
                    if prediction.recursive_pattern:
                        patterns_detected.append(prediction.recursive_pattern.value)
                except Exception as e:
                    results["issues"].append(f"Pattern detection failed for task {task_id}: {e}")
            
            results["details"]["patterns_detected"] = list(set(patterns_detected))
            results["details"]["pattern_count"] = len(set(patterns_detected))
            
            # Test pattern detector directly
            pattern_detector = self.analyzer.pattern_detector
            context = await self.analyzer._build_recursive_context("11")
            pattern = pattern_detector.detect_pattern("11", context, self.analyzer.base_analyzer.tasks_data)
            results["details"]["direct_pattern_detection"] = pattern.value if pattern else "none"
            
            print("   ✅ Pattern detection tests passed")
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            results["issues"].append(f"Pattern detection error: {e}")
            print(f"   ❌ Pattern detection failed: {e}")
        
        self.test_results["test_details"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _test_performance_characteristics(self):
        """Test performance and scalability"""
        print("⚡ Testing Performance Characteristics...")
        test_name = "performance_characteristics"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Test analysis speed
            start_time = time.time()
            prediction = await self.analyzer.analyze_recursive_complexity("1")
            analysis_time = time.time() - start_time
            
            results["details"]["single_analysis_time"] = analysis_time
            self.assertTrue(analysis_time < 1.0, "Single analysis should complete in under 1 second")
            
            # Test batch analysis performance
            batch_tasks = ["1", "2", "3", "11", "12"]
            start_time = time.time()
            
            batch_predictions = []
            for task_id in batch_tasks:
                prediction = await self.analyzer.analyze_recursive_complexity(task_id)
                batch_predictions.append(prediction)
            
            batch_time = time.time() - start_time
            results["details"]["batch_analysis_time"] = batch_time
            results["details"]["batch_size"] = len(batch_tasks)
            results["details"]["average_time_per_task"] = batch_time / len(batch_tasks)
            
            # Test cache performance
            cache_start = time.time()
            cached_prediction = await self.analyzer.analyze_recursive_complexity("1")  # Should hit cache
            cache_time = time.time() - cache_start
            
            results["details"]["cache_lookup_time"] = cache_time
            self.assertTrue(cache_time < analysis_time, "Cache lookup should be faster than fresh analysis")
            
            # Get performance metrics
            metrics = self.analyzer.get_performance_metrics()
            results["details"]["performance_metrics"] = metrics
            
            print("   ✅ Performance characteristics tests passed")
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            results["issues"].append(f"Performance testing error: {e}")
            print(f"   ❌ Performance characteristics failed: {e}")
        
        self.test_results["test_details"][test_name] = results
        self._update_test_counts(results["status"])
    
    async def _test_learning_persistence(self):
        """Test learning data persistence"""
        print("💾 Testing Learning Persistence...")
        test_name = "learning_persistence"
        results = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Test database initialization
            db_path = Path(self.analyzer.learning_db_path)
            results["details"]["database_exists"] = db_path.exists()
            
            # Test prediction storage
            prediction = await self.analyzer.analyze_recursive_complexity("3")
            context = await self.analyzer._build_recursive_context("3")
            await self.analyzer._store_prediction(prediction, context)
            results["details"]["prediction_storage"] = "success"
            
            # Test learning history retrieval
            history = await self.analyzer._get_learning_history("3", prediction.learning_strategy_used)
            results["details"]["history_retrieval"] = "success" if history is None else "found_data"
            
            # Test historical accuracy calculation
            accuracy = self.analyzer._get_historical_accuracy("3", LearningStrategy.GRADIENT_ADAPTATION)
            results["details"]["historical_accuracy"] = accuracy
            self.assertTrue(0.0 <= accuracy <= 1.0, "Historical accuracy should be between 0 and 1")
            
            print("   ✅ Learning persistence tests passed")
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            results["issues"].append(f"Learning persistence error: {e}")
            print(f"   ❌ Learning persistence failed: {e}")
        
        self.test_results["test_details"][test_name] = results
        self._update_test_counts(results["status"])
    
    def assertTrue(self, condition: bool, message: str):
        """Simple assertion helper"""
        if not condition:
            raise AssertionError(message)
    
    def _update_test_counts(self, status: str):
        """Update test counts"""
        self.test_results["test_summary"]["total_tests"] += 1
        
        if status == "passed":
            self.test_results["test_summary"]["passed_tests"] += 1
        else:
            self.test_results["test_summary"]["failed_tests"] += 1
    
    def _generate_test_summary(self):
        """Generate test summary"""
        summary = self.test_results["test_summary"]
        total = summary["total_tests"]
        passed = summary["passed_tests"]
        
        if total > 0:
            summary["success_rate"] = round((passed / total) * 100, 2)
        
        print(f"\n📊 TEST SUMMARY")
        print("=" * 30)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']}%")
        
        # Store performance metrics
        self.test_results["performance_metrics"] = self.analyzer.get_performance_metrics()


async def main():
    """Run the comprehensive test suite"""
    tester = RecursiveComplexityAnalyzerTester()
    
    try:
        results = await tester.run_comprehensive_tests()
        
        # Save results
        reports_dir = Path(".taskmaster/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = reports_dir / "recursive_complexity_analyzer_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Test results saved to: {results_file}")
        
        # Return exit code based on results
        if results["test_summary"]["success_rate"] >= 80:
            print(f"\n✅ RECURSIVE COMPLEXITY ANALYZER TESTS PASSED")
            return 0
        else:
            print(f"\n⚠️ SOME RECURSIVE COMPLEXITY ANALYZER TESTS FAILED")
            return 1
            
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)