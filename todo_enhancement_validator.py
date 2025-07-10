#!/usr/bin/env python3
"""
Todo Enhancement Engine Validation and Optimization Suite
Atomic Task 51.5: Validate and Optimize Recursive Enhancement Engine

This module provides comprehensive validation, testing, and optimization
capabilities for the recursive todo enhancement engine, ensuring correctness,
performance, and scalability.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import statistics
import unittest
import traceback
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
import random
import string

# Import our enhancement components
from recursive_todo_enhancer import RecursiveTodoEnhancer, EnhancementQualityMetrics
from todo_enhancement_integration import TodoEnhancementService, TodoEnhancementRequest


@dataclass
class ValidationTestCase:
    """Individual test case for validation"""
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    input_text: str = ""
    expected_improvement_min: float = 0.0
    expected_strategies: List[str] = field(default_factory=list)
    max_processing_time: float = 5.0
    context: Dict[str, Any] = field(default_factory=dict)
    test_category: str = "general"


@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_case: ValidationTestCase
    success: bool = False
    actual_improvement: float = 0.0
    actual_strategies: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    enhanced_text: str = ""
    error_message: str = ""
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization analysis"""
    avg_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    total_tests: int = 0
    successful_tests: int = 0


class TodoEnhancementValidator:
    """Comprehensive validation and testing framework"""
    
    def __init__(self):
        self.enhancer = RecursiveTodoEnhancer(max_depth=3, timeout_seconds=30.0)
        self.service = TodoEnhancementService()
        self.logger = logging.getLogger("TodoEnhancementValidator")
        
        # Test data
        self.test_cases: List[ValidationTestCase] = []
        self.validation_results: List[ValidationResult] = []
        
        # Performance tracking
        self.performance_data: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
        # Initialize test cases
        self._initialize_test_cases()
    
    def _initialize_test_cases(self):
        """Initialize comprehensive test case suite"""
        
        # Basic clarity enhancement tests
        clarity_tests = [
            ValidationTestCase(
                name="Basic Bug Fix",
                input_text="fix bug",
                expected_improvement_min=0.2,
                expected_strategies=["clarity_enhancement"],
                test_category="clarity"
            ),
            ValidationTestCase(
                name="Ambiguous Task",
                input_text="maybe add some features",
                expected_improvement_min=0.1,
                expected_strategies=["clarity_enhancement"],
                test_category="clarity"
            ),
            ValidationTestCase(
                name="Vague Implementation",
                input_text="implement stuff",
                expected_improvement_min=0.3,
                expected_strategies=["clarity_enhancement"],
                test_category="clarity"
            )
        ]
        
        # Complex decomposition tests
        decomposition_tests = [
            ValidationTestCase(
                name="Complex System Implementation",
                input_text="implement comprehensive user authentication system with oauth jwt and social login integration",
                expected_improvement_min=0.05,
                expected_strategies=["atomic_decomposition", "priority_optimization"],
                max_processing_time=10.0,
                test_category="decomposition"
            ),
            ValidationTestCase(
                name="Large Architecture Task",
                input_text="design and implement microservices architecture with service discovery load balancing and monitoring",
                expected_improvement_min=0.05,
                expected_strategies=["atomic_decomposition"],
                max_processing_time=15.0,
                test_category="decomposition"
            )
        ]
        
        # Context enrichment tests
        context_tests = [
            ValidationTestCase(
                name="API Development",
                input_text="create api endpoints",
                expected_improvement_min=0.1,
                expected_strategies=["context_enrichment"],
                test_category="context"
            ),
            ValidationTestCase(
                name="Database Task",
                input_text="optimize database queries",
                expected_improvement_min=0.1,
                expected_strategies=["context_enrichment"],
                test_category="context"
            )
        ]
        
        # Priority optimization tests
        priority_tests = [
            ValidationTestCase(
                name="Bug Fix Priority",
                input_text="fix critical authentication vulnerability",
                expected_improvement_min=0.05,
                expected_strategies=["priority_optimization"],
                test_category="priority"
            ),
            ValidationTestCase(
                name="Performance Issue",
                input_text="optimize slow database queries",
                expected_improvement_min=0.1,
                expected_strategies=["priority_optimization"],
                test_category="priority"
            )
        ]
        
        # Implementation guidance tests
        guidance_tests = [
            ValidationTestCase(
                name="Complex Implementation",
                input_text="implement machine learning recommendation engine",
                expected_improvement_min=0.05,
                expected_strategies=["implementation_guidance"],
                test_category="guidance"
            ),
            ValidationTestCase(
                name="Deployment Task",
                input_text="deploy application to production",
                expected_improvement_min=0.1,
                expected_strategies=["implementation_guidance"],
                test_category="guidance"
            )
        ]
        
        # Edge cases and stress tests
        edge_tests = [
            ValidationTestCase(
                name="Empty Text",
                input_text="",
                expected_improvement_min=0.0,
                expected_strategies=[],
                test_category="edge_cases"
            ),
            ValidationTestCase(
                name="Very Long Text",
                input_text=" ".join(["implement"] * 50),
                expected_improvement_min=0.0,
                max_processing_time=20.0,
                test_category="edge_cases"
            ),
            ValidationTestCase(
                name="Special Characters",
                input_text="fix bug with @#$%^&*() characters",
                expected_improvement_min=0.1,
                test_category="edge_cases"
            ),
            ValidationTestCase(
                name="Unicode Text",
                input_text="implement 国际化 support",
                expected_improvement_min=0.1,
                test_category="edge_cases"
            )
        ]
        
        # Performance stress tests
        stress_tests = []
        for i in range(10):
            stress_tests.append(ValidationTestCase(
                name=f"Stress Test {i+1}",
                input_text=f"implement feature {i+1} with complex requirements",
                expected_improvement_min=0.05,
                max_processing_time=5.0,
                test_category="stress"
            ))
        
        # Combine all test cases
        self.test_cases = (clarity_tests + decomposition_tests + context_tests + 
                          priority_tests + guidance_tests + edge_tests + stress_tests)
        
        self.logger.info(f"Initialized {len(self.test_cases)} validation test cases")
    
    async def run_validation_suite(self) -> Dict[str, Any]:
        """Run complete validation test suite"""
        self.logger.info("Starting comprehensive validation suite")
        start_time = time.time()
        
        # Run all test cases
        for test_case in self.test_cases:
            result = await self._run_single_test(test_case)
            self.validation_results.append(result)
        
        # Analyze results
        total_time = time.time() - start_time
        analysis = self._analyze_validation_results()
        analysis["total_validation_time"] = total_time
        
        self.logger.info(f"Validation suite completed in {total_time:.2f}s")
        return analysis
    
    async def _run_single_test(self, test_case: ValidationTestCase) -> ValidationResult:
        """Run a single validation test case"""
        self.logger.debug(f"Running test: {test_case.name}")
        
        start_time = time.time()
        
        try:
            # Create enhancement request
            request = TodoEnhancementRequest(
                todo_text=test_case.input_text,
                context=test_case.context,
                max_depth=3,
                timeout_seconds=test_case.max_processing_time
            )
            
            # Run enhancement
            response = await self.service.enhance_single_todo(request)
            
            processing_time = time.time() - start_time
            
            # Validate results
            success = (
                response.success and
                response.improvement_score >= test_case.expected_improvement_min and
                processing_time <= test_case.max_processing_time
            )
            
            # Check strategy usage if specified
            if test_case.expected_strategies:
                strategy_match = response.strategy_used in test_case.expected_strategies
                success = success and strategy_match
            
            result = ValidationResult(
                test_case=test_case,
                success=success,
                actual_improvement=response.improvement_score,
                actual_strategies=[response.strategy_used] if response.strategy_used else [],
                processing_time=processing_time,
                enhanced_text=response.enhanced_text,
                quality_metrics=response.quality_metrics
            )
            
            if not success:
                result.error_message = f"Test failed: improvement={response.improvement_score:.3f}, time={processing_time:.3f}s"
            
        except Exception as e:
            processing_time = time.time() - start_time
            result = ValidationResult(
                test_case=test_case,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
            
        return result
    
    def _analyze_validation_results(self) -> Dict[str, Any]:
        """Analyze validation results and generate report"""
        if not self.validation_results:
            return {"status": "no_tests_run"}
        
        successful_tests = [r for r in self.validation_results if r.success]
        failed_tests = [r for r in self.validation_results if not r.success]
        
        # Overall statistics
        analysis = {
            "overall_success_rate": len(successful_tests) / len(self.validation_results),
            "total_tests": len(self.validation_results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "average_improvement": statistics.mean(r.actual_improvement for r in successful_tests) if successful_tests else 0.0,
            "average_processing_time": statistics.mean(r.processing_time for r in self.validation_results),
            "max_processing_time": max(r.processing_time for r in self.validation_results),
            "min_processing_time": min(r.processing_time for r in self.validation_results)
        }
        
        # Category analysis
        category_stats = {}
        categories = set(r.test_case.test_category for r in self.validation_results)
        
        for category in categories:
            category_results = [r for r in self.validation_results if r.test_case.test_category == category]
            category_successful = [r for r in category_results if r.success]
            
            category_stats[category] = {
                "total": len(category_results),
                "successful": len(category_successful),
                "success_rate": len(category_successful) / len(category_results),
                "avg_improvement": statistics.mean(r.actual_improvement for r in category_successful) if category_successful else 0.0,
                "avg_processing_time": statistics.mean(r.processing_time for r in category_results)
            }
        
        analysis["category_statistics"] = category_stats
        
        # Failed test analysis
        if failed_tests:
            failure_reasons = {}
            for test in failed_tests:
                reason = test.error_message or "Performance criteria not met"
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            analysis["failure_analysis"] = {
                "failure_reasons": failure_reasons,
                "failed_test_details": [
                    {
                        "name": r.test_case.name,
                        "category": r.test_case.test_category,
                        "error": r.error_message,
                        "expected_improvement": r.test_case.expected_improvement_min,
                        "actual_improvement": r.actual_improvement
                    } for r in failed_tests[:10]  # Top 10 failures
                ]
            }
        
        return analysis
    
    async def run_performance_benchmarks(self, num_tests: int = 100) -> PerformanceMetrics:
        """Run performance benchmarks and optimization analysis"""
        self.logger.info(f"Starting performance benchmarks with {num_tests} tests")
        
        # Generate random test data
        test_data = []
        for i in range(num_tests):
            # Generate random todo text
            actions = ["implement", "fix", "create", "design", "optimize", "refactor"]
            subjects = ["api", "database", "frontend", "backend", "service", "component"]
            complexity = random.choice(["simple", "complex", "comprehensive"])
            
            todo_text = f"{random.choice(actions)} {complexity} {random.choice(subjects)} system"
            test_data.append(todo_text)
        
        # Record system metrics before test
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        processing_times = []
        successful_tests = 0
        errors = 0
        
        # Run benchmark tests
        for i, todo_text in enumerate(test_data):
            if i % 20 == 0:
                self.logger.debug(f"Benchmark progress: {i+1}/{num_tests}")
            
            try:
                test_start = time.time()
                result = await self.enhancer.enhance_recursive(todo_text)
                test_time = time.time() - test_start
                
                processing_times.append(test_time)
                if result["improvement_score"] > 0:
                    successful_tests += 1
                    
            except Exception as e:
                errors += 1
                self.logger.debug(f"Benchmark error: {e}")
        
        total_time = time.time() - start_time
        
        # Record system metrics after test
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = process.cpu_percent()
        
        # Get enhancer statistics
        enhancer_stats = self.enhancer.get_performance_statistics()
        
        # Calculate performance metrics
        metrics = PerformanceMetrics(
            avg_processing_time=statistics.mean(processing_times) if processing_times else 0.0,
            max_processing_time=max(processing_times) if processing_times else 0.0,
            min_processing_time=min(processing_times) if processing_times else 0.0,
            throughput_per_second=num_tests / total_time,
            memory_usage_mb=final_memory - initial_memory,
            cpu_usage_percent=(final_cpu + initial_cpu) / 2,
            cache_hit_rate=enhancer_stats.get("cache_hit_rate", 0.0),
            error_rate=errors / num_tests,
            total_tests=num_tests,
            successful_tests=successful_tests
        )
        
        self.logger.info(f"Performance benchmarks completed: {metrics.throughput_per_second:.2f} tests/sec")
        return metrics
    
    async def run_scalability_tests(self) -> Dict[str, Any]:
        """Test system scalability with varying loads"""
        self.logger.info("Running scalability tests")
        
        test_sizes = [10, 50, 100, 500, 1000]
        scalability_results = {}
        
        for size in test_sizes:
            self.logger.info(f"Testing scalability with {size} todos")
            
            try:
                # Generate test data
                test_todos = [f"implement feature {i}" for i in range(size)]
                
                start_time = time.time()
                results = await self.enhancer.enhance_batch(test_todos)
                total_time = time.time() - start_time
                
                successful = sum(1 for r in results if r.get("improvement_score", 0) > 0)
                
                scalability_results[size] = {
                    "total_time": total_time,
                    "throughput": size / total_time,
                    "success_rate": successful / size,
                    "avg_processing_time": total_time / size,
                    "successful_enhancements": successful
                }
                
            except Exception as e:
                scalability_results[size] = {
                    "error": str(e),
                    "total_time": 0,
                    "throughput": 0,
                    "success_rate": 0
                }
        
        # Analyze scalability patterns
        successful_results = {k: v for k, v in scalability_results.items() if "error" not in v}
        
        if len(successful_results) >= 2:
            sizes = list(successful_results.keys())
            throughputs = [successful_results[s]["throughput"] for s in sizes]
            
            # Calculate scalability efficiency (throughput stability)
            throughput_variance = statistics.variance(throughputs) if len(throughputs) > 1 else 0
            scalability_score = 1.0 - min(1.0, throughput_variance / max(throughputs))
            
            analysis = {
                "results": scalability_results,
                "scalability_score": scalability_score,
                "max_tested_size": max(sizes),
                "throughput_stability": 1.0 - (throughput_variance / max(throughputs)) if max(throughputs) > 0 else 0,
                "recommendations": self._generate_scalability_recommendations(successful_results)
            }
        else:
            analysis = {
                "results": scalability_results,
                "scalability_score": 0.0,
                "error": "Insufficient successful tests for analysis"
            }
        
        return analysis
    
    def _generate_scalability_recommendations(self, results: Dict[int, Dict[str, Any]]) -> List[str]:
        """Generate scalability optimization recommendations"""
        recommendations = []
        
        # Analyze throughput trends
        sizes = sorted(results.keys())
        throughputs = [results[s]["throughput"] for s in sizes]
        
        # Check for throughput degradation
        if len(throughputs) >= 3:
            recent_trend = throughputs[-1] / throughputs[-3] if throughputs[-3] > 0 else 1.0
            if recent_trend < 0.8:
                recommendations.append("Throughput degrades significantly with scale - consider batch optimization")
        
        # Check memory usage patterns
        max_size = max(sizes)
        if max_size >= 500:
            recommendations.append("For large batches (500+), consider implementing streaming processing")
        
        # Check processing time variance
        processing_times = [results[s]["avg_processing_time"] for s in sizes]
        if len(processing_times) >= 3:
            time_variance = statistics.variance(processing_times)
            if time_variance > 0.1:
                recommendations.append("High processing time variance detected - investigate caching optimization")
        
        return recommendations
    
    def generate_optimization_report(self, performance_metrics: PerformanceMetrics,
                                   scalability_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive optimization recommendations"""
        
        recommendations = {
            "performance_optimizations": [],
            "scalability_optimizations": [],
            "resource_optimizations": [],
            "algorithmic_optimizations": []
        }
        
        # Performance optimizations
        if performance_metrics.avg_processing_time > 1.0:
            recommendations["performance_optimizations"].append(
                "Average processing time >1s - consider strategy optimization"
            )
        
        if performance_metrics.cache_hit_rate < 0.5:
            recommendations["performance_optimizations"].append(
                "Low cache hit rate - review caching strategy and cache size"
            )
        
        if performance_metrics.error_rate > 0.05:
            recommendations["performance_optimizations"].append(
                "Error rate >5% - investigate timeout and recursion settings"
            )
        
        # Scalability optimizations
        if scalability_analysis.get("scalability_score", 0) < 0.7:
            recommendations["scalability_optimizations"].append(
                "Poor scalability score - implement batch processing optimizations"
            )
        
        recommendations["scalability_optimizations"].extend(
            scalability_analysis.get("recommendations", [])
        )
        
        # Resource optimizations
        if performance_metrics.memory_usage_mb > 100:
            recommendations["resource_optimizations"].append(
                "High memory usage - implement memory pooling or streaming"
            )
        
        if performance_metrics.cpu_usage_percent > 80:
            recommendations["resource_optimizations"].append(
                "High CPU usage - consider async optimization or load balancing"
            )
        
        # Algorithmic optimizations
        if performance_metrics.throughput_per_second < 50:
            recommendations["algorithmic_optimizations"].append(
                "Low throughput - review enhancement strategy algorithms"
            )
        
        # Calculate overall optimization priority
        total_recommendations = sum(len(recs) for recs in recommendations.values())
        priority = "high" if total_recommendations > 8 else "medium" if total_recommendations > 4 else "low"
        
        return {
            "optimization_priority": priority,
            "total_recommendations": total_recommendations,
            "recommendations": recommendations,
            "performance_summary": {
                "throughput": performance_metrics.throughput_per_second,
                "avg_processing_time": performance_metrics.avg_processing_time,
                "success_rate": (performance_metrics.successful_tests / performance_metrics.total_tests) if performance_metrics.total_tests > 0 else 0,
                "scalability_score": scalability_analysis.get("scalability_score", 0)
            }
        }
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation and optimization suite"""
        self.logger.info("Starting comprehensive validation and optimization")
        
        # Run validation tests
        validation_analysis = await self.run_validation_suite()
        
        # Run performance benchmarks
        performance_metrics = await self.run_performance_benchmarks()
        
        # Run scalability tests
        scalability_analysis = await self.run_scalability_tests()
        
        # Generate optimization report
        optimization_report = self.generate_optimization_report(
            performance_metrics, scalability_analysis
        )
        
        # Combine all results
        comprehensive_report = {
            "validation_summary": {
                "total_tests": validation_analysis["total_tests"],
                "success_rate": validation_analysis["overall_success_rate"],
                "average_improvement": validation_analysis["average_improvement"],
                "test_categories": list(validation_analysis["category_statistics"].keys())
            },
            "performance_summary": {
                "throughput_per_second": performance_metrics.throughput_per_second,
                "avg_processing_time": performance_metrics.avg_processing_time,
                "memory_usage_mb": performance_metrics.memory_usage_mb,
                "error_rate": performance_metrics.error_rate
            },
            "scalability_summary": {
                "scalability_score": scalability_analysis.get("scalability_score", 0),
                "max_tested_size": scalability_analysis.get("max_tested_size", 0),
                "throughput_stability": scalability_analysis.get("throughput_stability", 0)
            },
            "optimization_recommendations": optimization_report,
            "detailed_results": {
                "validation_analysis": validation_analysis,
                "performance_metrics": performance_metrics.__dict__,
                "scalability_analysis": scalability_analysis
            },
            "overall_system_health": self._calculate_system_health(
                validation_analysis, performance_metrics, scalability_analysis
            )
        }
        
        self.logger.info("Comprehensive validation completed")
        return comprehensive_report
    
    def _calculate_system_health(self, validation_analysis: Dict[str, Any],
                               performance_metrics: PerformanceMetrics,
                               scalability_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health score"""
        
        # Validation health (40% weight)
        validation_score = validation_analysis["overall_success_rate"] * 0.4
        
        # Performance health (35% weight)
        performance_score = 0
        if performance_metrics.throughput_per_second > 50:
            performance_score += 0.1
        if performance_metrics.avg_processing_time < 1.0:
            performance_score += 0.1
        if performance_metrics.error_rate < 0.05:
            performance_score += 0.1
        if performance_metrics.cache_hit_rate > 0.3:
            performance_score += 0.05
        performance_score *= 0.35
        
        # Scalability health (25% weight)
        scalability_score = scalability_analysis.get("scalability_score", 0) * 0.25
        
        # Overall health
        overall_score = validation_score + performance_score + scalability_score
        
        # Health rating
        if overall_score >= 0.8:
            health_rating = "excellent"
        elif overall_score >= 0.6:
            health_rating = "good"
        elif overall_score >= 0.4:
            health_rating = "fair"
        else:
            health_rating = "poor"
        
        return {
            "overall_score": overall_score,
            "health_rating": health_rating,
            "component_scores": {
                "validation": validation_score,
                "performance": performance_score,
                "scalability": scalability_score
            },
            "recommendation": self._get_health_recommendation(health_rating, overall_score)
        }
    
    def _get_health_recommendation(self, health_rating: str, score: float) -> str:
        """Get health-based recommendation"""
        if health_rating == "excellent":
            return "System is performing excellently. Ready for production deployment."
        elif health_rating == "good":
            return "System is performing well. Minor optimizations recommended before production."
        elif health_rating == "fair":
            return "System performance is acceptable but requires optimization before production."
        else:
            return "System requires significant optimization and testing before production deployment."


# Export key classes
__all__ = [
    "ValidationTestCase", "ValidationResult", "PerformanceMetrics", "TodoEnhancementValidator"
]


if __name__ == "__main__":
    # Demo comprehensive validation
    async def demo():
        logging.basicConfig(level=logging.INFO)
        
        print("🔍 Todo Enhancement Engine Validation & Optimization Suite")
        print("=" * 75)
        
        # Create validator
        validator = TodoEnhancementValidator()
        
        print(f"📋 Initialized validator with {len(validator.test_cases)} test cases")
        
        # Run comprehensive validation
        print("\n🚀 Running comprehensive validation and optimization...")
        report = await validator.run_comprehensive_validation()
        
        # Display results
        print(f"\n📊 Validation Summary:")
        val_summary = report["validation_summary"]
        print(f"   Total Tests: {val_summary['total_tests']}")
        print(f"   Success Rate: {val_summary['success_rate']:.1%}")
        print(f"   Average Improvement: {val_summary['average_improvement']:.3f}")
        print(f"   Test Categories: {len(val_summary['test_categories'])}")
        
        print(f"\n⚡ Performance Summary:")
        perf_summary = report["performance_summary"]
        print(f"   Throughput: {perf_summary['throughput_per_second']:.1f} tests/sec")
        print(f"   Avg Processing Time: {perf_summary['avg_processing_time']:.3f}s")
        print(f"   Memory Usage: {perf_summary['memory_usage_mb']:.1f} MB")
        print(f"   Error Rate: {perf_summary['error_rate']:.1%}")
        
        print(f"\n📈 Scalability Summary:")
        scale_summary = report["scalability_summary"]
        print(f"   Scalability Score: {scale_summary['scalability_score']:.3f}")
        print(f"   Max Tested Size: {scale_summary['max_tested_size']}")
        print(f"   Throughput Stability: {scale_summary['throughput_stability']:.3f}")
        
        print(f"\n🎯 System Health:")
        health = report["overall_system_health"]
        print(f"   Overall Score: {health['overall_score']:.3f}")
        print(f"   Health Rating: {health['health_rating'].upper()}")
        print(f"   Recommendation: {health['recommendation']}")
        
        print(f"\n💡 Optimization Recommendations:")
        opt_recs = report["optimization_recommendations"]
        print(f"   Priority: {opt_recs['optimization_priority'].upper()}")
        print(f"   Total Recommendations: {opt_recs['total_recommendations']}")
        
        for category, recommendations in opt_recs["recommendations"].items():
            if recommendations:
                print(f"\n   {category.replace('_', ' ').title()}:")
                for rec in recommendations[:3]:  # Show top 3
                    print(f"     • {rec}")
        
        print(f"\n✅ Validation and optimization suite completed!")
        print(f"   System is {health['health_rating']} and ready for {('production' if health['health_rating'] in ['excellent', 'good'] else 'further optimization')}")
    
    # Run demo
    asyncio.run(demo())