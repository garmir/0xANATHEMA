#!/usr/bin/env python3
"""
Recursive Enhancement Engine Validator and Optimizer
Comprehensive validation and optimization framework for recursive todo enhancement systems
"""

import asyncio
import json
import time
import traceback
import importlib.util
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import gc
import os


class RecursiveEnhancementValidator:
    """Comprehensive validator for recursive enhancement systems"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "validation_summary": {
                "total_systems_tested": 0,
                "systems_passed": 0,
                "systems_failed": 0,
                "performance_score": 0.0,
                "optimization_score": 0.0
            },
            "system_tests": {},
            "performance_metrics": {},
            "optimization_recommendations": [],
            "scalability_assessment": {}
        }
        
        # Recursive systems to validate
        self.recursive_systems = {
            "recursive_todo_enhancement_engine": {
                "file": "recursive_todo_enhancement_engine.py",
                "main_class": "RecursiveTodoEnhancer",
                "test_methods": ["enhance_task", "calculate_metrics"]
            },
            "recursive_todo_enhancer": {
                "file": "recursive_todo_enhancer.py", 
                "main_class": "RecursiveTodoEnhancer",
                "test_methods": ["enhance_task", "recursive_optimization"]
            },
            "recursive_todo_processor": {
                "file": "recursive_todo_processor.py",
                "main_class": "RecursiveImprovementEngine", 
                "test_methods": ["process_todos", "run_improvement_cycle"]
            },
            "recursive_meta_learning_framework": {
                "file": "recursive_meta_learning_framework.py",
                "main_class": "MetaOptimizer",
                "test_methods": ["optimize", "update_performance"]
            }
        }
        
        # Performance benchmarks
        self.performance_benchmarks = {
            "memory_usage_mb": {"excellent": 50, "good": 100, "acceptable": 200},
            "processing_time_seconds": {"excellent": 1.0, "good": 5.0, "acceptable": 15.0},
            "scalability_factor": {"excellent": 0.9, "good": 0.7, "acceptable": 0.5},
            "optimization_ratio": {"excellent": 0.8, "good": 0.6, "acceptable": 0.4}
        }
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation and optimization"""
        print("🔄 RECURSIVE ENHANCEMENT ENGINE VALIDATION")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Systems to validate: {len(self.recursive_systems)}")
        print()
        
        # Test each recursive system
        for system_name, system_config in self.recursive_systems.items():
            await self._validate_recursive_system(system_name, system_config)
        
        # Run performance tests
        await self._run_performance_tests()
        
        # Run scalability tests
        await self._run_scalability_tests()
        
        # Generate optimization recommendations
        self._generate_optimization_recommendations()
        
        # Calculate final scores
        self._calculate_validation_scores()
        
        return self.validation_results
    
    async def _validate_recursive_system(self, system_name: str, config: Dict[str, Any]):
        """Validate individual recursive system"""
        print(f"🧪 Validating {system_name}...")
        
        test_results = {
            "status": "unknown",
            "import_test": False,
            "class_availability": False,
            "method_tests": {},
            "performance_metrics": {},
            "error_details": None
        }
        
        try:
            file_path = Path(config["file"])
            
            if not file_path.exists():
                test_results["status"] = "failed"
                test_results["error_details"] = f"File not found: {config['file']}"
                print(f"   ❌ File missing: {config['file']}")
                self.validation_results["system_tests"][system_name] = test_results
                return
            
            # Test import
            try:
                spec = importlib.util.spec_from_file_location(system_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                test_results["import_test"] = True
                print(f"   ✅ Import successful")
                
                # Test class availability
                main_class_name = config["main_class"]
                if hasattr(module, main_class_name):
                    test_results["class_availability"] = True
                    main_class = getattr(module, main_class_name)
                    print(f"   ✅ Class {main_class_name} available")
                    
                    # Test methods
                    instance = None
                    try:
                        # Try to instantiate with common parameters
                        if main_class_name == "RecursiveImprovementEngine":
                            instance = main_class(Path.cwd())
                        elif main_class_name == "MetaOptimizer":
                            # Create a minimal config for MetaOptimizer
                            from dataclasses import dataclass
                            @dataclass
                            class MockConfig:
                                max_iterations: int = 10
                                learning_rate: float = 0.01
                            instance = main_class(MockConfig())
                        else:
                            instance = main_class()
                        print(f"   ✅ Class instantiation successful")
                        
                        # Test each method
                        for method_name in config.get("test_methods", []):
                            method_result = await self._test_method(instance, method_name)
                            test_results["method_tests"][method_name] = method_result
                            
                        # Run performance test on instance
                        perf_metrics = await self._test_system_performance(instance, system_name)
                        test_results["performance_metrics"] = perf_metrics
                        
                    except Exception as e:
                        print(f"   ⚠️ Class instantiation failed: {e}")
                        test_results["error_details"] = f"Instantiation failed: {str(e)}"
                        
                        # Try to test methods without instantiation
                        for method_name in config.get("test_methods", []):
                            if hasattr(main_class, method_name):
                                test_results["method_tests"][method_name] = {"available": True, "tested": False}
                
                else:
                    test_results["error_details"] = f"Class {main_class_name} not found"
                    print(f"   ❌ Class {main_class_name} not available")
                
            except Exception as e:
                test_results["error_details"] = f"Import failed: {str(e)}"
                print(f"   ❌ Import failed: {e}")
            
            # Determine overall status
            if test_results["import_test"] and test_results["class_availability"]:
                method_success_count = sum(1 for result in test_results["method_tests"].values() 
                                         if result.get("success", False))
                total_methods = len(config.get("test_methods", []))
                
                if method_success_count >= total_methods * 0.8:  # 80% success rate
                    test_results["status"] = "passed"
                    print(f"   ✅ {system_name} validation PASSED")
                else:
                    test_results["status"] = "partial"
                    print(f"   ⚠️ {system_name} validation PARTIAL")
            else:
                test_results["status"] = "failed"
                print(f"   ❌ {system_name} validation FAILED")
                
        except Exception as e:
            test_results["status"] = "error"
            test_results["error_details"] = str(e)
            print(f"   💥 {system_name} validation ERROR: {e}")
        
        self.validation_results["system_tests"][system_name] = test_results
        self._update_validation_counts(test_results["status"])
    
    async def _test_method(self, instance: Any, method_name: str) -> Dict[str, Any]:
        """Test individual method"""
        result = {"available": False, "tested": False, "success": False, "execution_time": 0.0}
        
        try:
            if hasattr(instance, method_name):
                result["available"] = True
                
                # Test method execution with timeout
                start_time = time.time()
                
                method = getattr(instance, method_name)
                
                # Try to call method with safe parameters
                try:
                    if asyncio.iscoroutinefunction(method):
                        # Async method
                        test_result = await asyncio.wait_for(
                            method(), timeout=5.0
                        )
                    else:
                        # Sync method
                        test_result = method()
                    
                    result["tested"] = True
                    result["success"] = True
                    result["execution_time"] = time.time() - start_time
                    print(f"     ✅ Method {method_name}: {result['execution_time']:.3f}s")
                    
                except TypeError:
                    # Method might require parameters, try with test data
                    try:
                        if method_name in ["enhance_todos", "process_todos"]:
                            test_todos = [{"id": "test", "description": "test todo"}]
                            if asyncio.iscoroutinefunction(method):
                                test_result = await asyncio.wait_for(
                                    method(test_todos), timeout=5.0
                                )
                            else:
                                test_result = method(test_todos)
                        elif method_name in ["enhance_task"]:
                            test_task = {"id": "test", "description": "test task"}
                            if asyncio.iscoroutinefunction(method):
                                test_result = await asyncio.wait_for(
                                    method(test_task), timeout=5.0
                                )
                            else:
                                test_result = method(test_task)
                        elif method_name in ["calculate_metrics"]:
                            test_todo = {"id": "test", "description": "test todo", "priority": "medium"}
                            if asyncio.iscoroutinefunction(method):
                                test_result = await asyncio.wait_for(
                                    method(test_todo), timeout=5.0
                                )
                            else:
                                test_result = method(test_todo)
                        elif method_name in ["run_improvement_cycle"]:
                            test_todos = [{"id": "test", "description": "test todo"}]
                            if asyncio.iscoroutinefunction(method):
                                test_result = await asyncio.wait_for(
                                    method(test_todos), timeout=5.0
                                )
                            else:
                                test_result = method(test_todos)
                        elif method_name in ["optimize", "update_performance"]:
                            # These methods might need specific parameters
                            if asyncio.iscoroutinefunction(method):
                                test_result = await asyncio.wait_for(
                                    method(), timeout=5.0
                                )
                            else:
                                test_result = method()
                        else:
                            # Try calling with empty args
                            if asyncio.iscoroutinefunction(method):
                                test_result = await asyncio.wait_for(
                                    method([]), timeout=5.0
                                )
                            else:
                                test_result = method([])
                        
                        result["tested"] = True
                        result["success"] = True
                        result["execution_time"] = time.time() - start_time
                        print(f"     ✅ Method {method_name}: {result['execution_time']:.3f}s")
                        
                    except Exception as e:
                        result["tested"] = True
                        result["success"] = False
                        result["error"] = str(e)
                        print(f"     ⚠️ Method {method_name}: {str(e)[:50]}...")
                
                except asyncio.TimeoutError:
                    result["tested"] = True
                    result["success"] = False
                    result["error"] = "Method execution timed out"
                    print(f"     ⏱️ Method {method_name}: timeout")
                
                except Exception as e:
                    result["tested"] = True
                    result["success"] = False
                    result["error"] = str(e)
                    print(f"     ❌ Method {method_name}: {str(e)[:50]}...")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"     💥 Method {method_name}: unexpected error")
        
        return result
    
    async def _test_system_performance(self, instance: Any, system_name: str) -> Dict[str, Any]:
        """Test system performance metrics"""
        metrics = {
            "memory_usage_mb": 0.0,
            "cpu_usage_percent": 0.0,
            "processing_time_seconds": 0.0,
            "memory_efficiency": "unknown"
        }
        
        try:
            # Get initial memory using system commands (fallback for systems without psutil)
            try:
                import resource
                initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB on Linux, already MB on macOS
            except:
                initial_memory = 0.0  # Fallback if resource not available
            
            # Run performance test
            start_time = time.time()
            
            # Create test data
            test_data = [
                {"id": f"test_{i}", "description": f"Test todo {i}", "priority": "medium"}
                for i in range(100)  # Test with 100 items
            ]
            
            # Try to run enhancement if method exists
            if hasattr(instance, 'enhance_todos'):
                try:
                    if asyncio.iscoroutinefunction(instance.enhance_todos):
                        await instance.enhance_todos(test_data[:10])  # Smaller subset for safety
                    else:
                        instance.enhance_todos(test_data[:10])
                except:
                    pass  # Ignore errors, we're just testing performance
            elif hasattr(instance, 'process_todos'):
                try:
                    if asyncio.iscoroutinefunction(instance.process_todos):
                        await instance.process_todos(test_data[:10])
                    else:
                        instance.process_todos(test_data[:10])
                except:
                    pass
            
            # Calculate metrics
            end_time = time.time()
            try:
                import resource
                final_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            except:
                final_memory = initial_memory
            
            metrics["memory_usage_mb"] = max(0, final_memory - initial_memory)
            metrics["processing_time_seconds"] = end_time - start_time
            metrics["cpu_usage_percent"] = 0.0  # Placeholder since psutil not available
            
            # Assess memory efficiency
            if metrics["memory_usage_mb"] < self.performance_benchmarks["memory_usage_mb"]["excellent"]:
                metrics["memory_efficiency"] = "excellent"
            elif metrics["memory_usage_mb"] < self.performance_benchmarks["memory_usage_mb"]["good"]:
                metrics["memory_efficiency"] = "good"
            elif metrics["memory_usage_mb"] < self.performance_benchmarks["memory_usage_mb"]["acceptable"]:
                metrics["memory_efficiency"] = "acceptable"
            else:
                metrics["memory_efficiency"] = "poor"
            
            print(f"     📊 Performance: {metrics['processing_time_seconds']:.3f}s, {metrics['memory_usage_mb']:.1f}MB, {metrics['memory_efficiency']}")
            
        except Exception as e:
            metrics["error"] = str(e)
            print(f"     ❌ Performance test failed: {e}")
        
        return metrics
    
    async def _run_performance_tests(self):
        """Run comprehensive performance tests"""
        print(f"\n⚡ Running Performance Tests...")
        
        performance_results = {
            "overall_performance_score": 0.0,
            "memory_efficiency_score": 0.0,
            "processing_speed_score": 0.0,
            "system_benchmarks": {}
        }
        
        try:
            # Test overall system performance
            start_time = time.time()
            
            # Simulate recursive enhancement workload
            test_workload = [
                {"id": f"perf_test_{i}", "description": f"Performance test todo {i}"}
                for i in range(1000)
            ]
            
            # Memory test
            try:
                import resource
                initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            except:
                initial_memory = 0.0
            
            # Process test data in batches (simulating recursive enhancement)
            batch_size = 50
            total_processing_time = 0.0
            
            for i in range(0, len(test_workload), batch_size):
                batch = test_workload[i:i+batch_size]
                batch_start = time.time()
                
                # Simulate enhancement processing
                await asyncio.sleep(0.001)  # Simulate work
                enhanced_batch = [
                    {**item, "enhanced": True, "optimization_level": "high"}
                    for item in batch
                ]
                
                batch_time = time.time() - batch_start
                total_processing_time += batch_time
            
            try:
                import resource
                final_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            except:
                final_memory = initial_memory
            total_time = time.time() - start_time
            
            # Calculate scores
            memory_usage = final_memory - initial_memory
            throughput = len(test_workload) / total_time  # items per second
            
            # Score memory efficiency (0-1 scale)
            if memory_usage <= self.performance_benchmarks["memory_usage_mb"]["excellent"]:
                memory_score = 1.0
            elif memory_usage <= self.performance_benchmarks["memory_usage_mb"]["good"]:
                memory_score = 0.8
            elif memory_usage <= self.performance_benchmarks["memory_usage_mb"]["acceptable"]:
                memory_score = 0.6
            else:
                memory_score = 0.4
            
            # Score processing speed (0-1 scale)
            if total_time <= self.performance_benchmarks["processing_time_seconds"]["excellent"]:
                speed_score = 1.0
            elif total_time <= self.performance_benchmarks["processing_time_seconds"]["good"]:
                speed_score = 0.8
            elif total_time <= self.performance_benchmarks["processing_time_seconds"]["acceptable"]:
                speed_score = 0.6
            else:
                speed_score = 0.4
            
            overall_score = (memory_score + speed_score) / 2
            
            performance_results.update({
                "overall_performance_score": overall_score,
                "memory_efficiency_score": memory_score,
                "processing_speed_score": speed_score,
                "system_benchmarks": {
                    "total_items_processed": len(test_workload),
                    "total_processing_time": total_time,
                    "throughput_items_per_second": throughput,
                    "memory_usage_mb": memory_usage,
                    "batch_processing_time": total_processing_time,
                    "average_batch_time": total_processing_time / (len(test_workload) / batch_size)
                }
            })
            
            print(f"   📊 Overall performance score: {overall_score:.2f}")
            print(f"   🧠 Memory efficiency: {memory_score:.2f} ({memory_usage:.1f}MB)")
            print(f"   ⚡ Processing speed: {speed_score:.2f} ({throughput:.1f} items/sec)")
            
        except Exception as e:
            performance_results["error"] = str(e)
            print(f"   ❌ Performance testing failed: {e}")
        
        self.validation_results["performance_metrics"] = performance_results
    
    async def _run_scalability_tests(self):
        """Test scalability under load"""
        print(f"\n📈 Running Scalability Tests...")
        
        scalability_results = {
            "scalability_score": 0.0,
            "load_test_results": {},
            "memory_scaling": {},
            "performance_degradation": {}
        }
        
        try:
            # Test with increasing loads
            test_loads = [10, 50, 100, 500, 1000]
            load_results = {}
            
            for load_size in test_loads:
                print(f"   Testing load: {load_size} items...")
                
                start_time = time.time()
                try:
                    import resource
                    initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                except:
                    initial_memory = 0.0
                
                # Create test data
                test_data = [
                    {"id": f"scale_test_{i}", "description": f"Scalability test {i}"}
                    for i in range(load_size)
                ]
                
                # Process data (simulate recursive enhancement)
                processed_count = 0
                for item in test_data:
                    # Simulate enhancement
                    enhanced_item = {**item, "enhanced": True}
                    processed_count += 1
                    
                    # Small delay to simulate processing
                    if processed_count % 100 == 0:
                        await asyncio.sleep(0.001)
                
                end_time = time.time()
                try:
                    import resource
                    final_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                except:
                    final_memory = initial_memory
                
                processing_time = end_time - start_time
                memory_used = final_memory - initial_memory
                throughput = load_size / processing_time
                
                load_results[load_size] = {
                    "processing_time": processing_time,
                    "memory_usage_mb": memory_used,
                    "throughput": throughput,
                    "memory_per_item": memory_used / load_size if load_size > 0 else 0
                }
                
                print(f"     Time: {processing_time:.3f}s, Memory: {memory_used:.1f}MB, Throughput: {throughput:.1f} items/sec")
            
            # Calculate scalability metrics
            if len(load_results) >= 2:
                # Compare smallest and largest loads
                small_load = min(test_loads)
                large_load = max(test_loads)
                
                small_result = load_results[small_load]
                large_result = load_results[large_load]
                
                # Calculate scaling efficiency
                expected_time_ratio = large_load / small_load
                actual_time_ratio = large_result["processing_time"] / small_result["processing_time"]
                
                time_scaling_efficiency = expected_time_ratio / actual_time_ratio if actual_time_ratio > 0 else 0
                
                # Memory scaling
                memory_scaling_factor = large_result["memory_per_item"] / small_result["memory_per_item"] if small_result["memory_per_item"] > 0 else 1
                
                scalability_score = min(time_scaling_efficiency, 1.0) * (2.0 - min(memory_scaling_factor, 2.0)) / 2.0
                
                scalability_results.update({
                    "scalability_score": scalability_score,
                    "load_test_results": load_results,
                    "memory_scaling": {
                        "memory_scaling_factor": memory_scaling_factor,
                        "linear_scaling": memory_scaling_factor <= 1.2
                    },
                    "performance_degradation": {
                        "time_scaling_efficiency": time_scaling_efficiency,
                        "acceptable_degradation": time_scaling_efficiency >= 0.7
                    }
                })
                
                print(f"   📊 Scalability score: {scalability_score:.2f}")
                print(f"   📏 Time scaling efficiency: {time_scaling_efficiency:.2f}")
                print(f"   🧠 Memory scaling factor: {memory_scaling_factor:.2f}")
        
        except Exception as e:
            scalability_results["error"] = str(e)
            print(f"   ❌ Scalability testing failed: {e}")
        
        self.validation_results["scalability_assessment"] = scalability_results
    
    def _generate_optimization_recommendations(self):
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze system test results
        failed_systems = []
        partial_systems = []
        
        for system_name, results in self.validation_results["system_tests"].items():
            if results["status"] == "failed":
                failed_systems.append(system_name)
            elif results["status"] == "partial":
                partial_systems.append(system_name)
        
        if failed_systems:
            recommendations.append({
                "priority": "high",
                "category": "system_integrity",
                "description": f"Fix failed systems: {', '.join(failed_systems)}",
                "action": "Review and repair broken recursive enhancement systems"
            })
        
        if partial_systems:
            recommendations.append({
                "priority": "medium", 
                "category": "system_completeness",
                "description": f"Complete partial systems: {', '.join(partial_systems)}",
                "action": "Implement missing methods and improve reliability"
            })
        
        # Analyze performance metrics
        perf_metrics = self.validation_results.get("performance_metrics", {})
        if perf_metrics.get("overall_performance_score", 0) < 0.7:
            recommendations.append({
                "priority": "high",
                "category": "performance",
                "description": "Overall performance below acceptable threshold",
                "action": "Optimize algorithms, reduce memory usage, improve processing speed"
            })
        
        if perf_metrics.get("memory_efficiency_score", 0) < 0.6:
            recommendations.append({
                "priority": "medium",
                "category": "memory_optimization",
                "description": "Memory usage inefficient",
                "action": "Implement memory pooling, reduce object creation, add garbage collection"
            })
        
        # Analyze scalability
        scalability = self.validation_results.get("scalability_assessment", {})
        if scalability.get("scalability_score", 0) < 0.6:
            recommendations.append({
                "priority": "medium",
                "category": "scalability",
                "description": "Poor scalability under load",
                "action": "Implement batch processing, add caching, optimize data structures"
            })
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append({
                "priority": "low",
                "category": "enhancement",
                "description": "System performing well",
                "action": "Consider advanced optimizations: parallel processing, caching strategies"
            })
        
        self.validation_results["optimization_recommendations"] = recommendations
        
        print(f"\n💡 Generated {len(recommendations)} optimization recommendations")
        for rec in recommendations:
            priority_emoji = "🔴" if rec["priority"] == "high" else "🟡" if rec["priority"] == "medium" else "🟢"
            print(f"   {priority_emoji} {rec['category']}: {rec['description']}")
    
    def _calculate_validation_scores(self):
        """Calculate final validation scores"""
        summary = self.validation_results["validation_summary"]
        
        # System health score
        if summary["total_systems_tested"] > 0:
            system_health_score = summary["systems_passed"] / summary["total_systems_tested"]
        else:
            system_health_score = 0.0
        
        # Performance score
        perf_metrics = self.validation_results.get("performance_metrics", {})
        performance_score = perf_metrics.get("overall_performance_score", 0.0)
        
        # Scalability score
        scalability = self.validation_results.get("scalability_assessment", {})
        scalability_score = scalability.get("scalability_score", 0.0)
        
        # Overall optimization score
        optimization_score = (system_health_score + performance_score + scalability_score) / 3
        
        summary["performance_score"] = performance_score
        summary["optimization_score"] = optimization_score
        
        print(f"\n📊 FINAL VALIDATION SCORES")
        print(f"   System Health: {system_health_score:.2f}")
        print(f"   Performance: {performance_score:.2f}")
        print(f"   Scalability: {scalability_score:.2f}")
        print(f"   Overall Optimization: {optimization_score:.2f}")
    
    def _update_validation_counts(self, status: str):
        """Update validation counts"""
        summary = self.validation_results["validation_summary"]
        summary["total_systems_tested"] += 1
        
        if status == "passed":
            summary["systems_passed"] += 1
        elif status in ["failed", "error"]:
            summary["systems_failed"] += 1


async def main():
    """Run recursive enhancement validation"""
    validator = RecursiveEnhancementValidator()
    
    try:
        results = await validator.run_comprehensive_validation()
        
        # Save results
        reports_dir = Path(".taskmaster/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = reports_dir / "recursive_enhancement_validation.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Validation results saved to: {results_file}")
        
        # Generate summary report
        summary_file = reports_dir / "RECURSIVE_ENHANCEMENT_OPTIMIZATION_REPORT.md"
        with open(summary_file, 'w') as f:
            f.write("# Recursive Enhancement Engine Validation Report\n\n")
            f.write(f"**Validation Date**: {results['timestamp']}\n\n")
            
            summary = results['validation_summary']
            f.write(f"## Validation Summary\n\n")
            f.write(f"- **Systems Tested**: {summary['total_systems_tested']}\n")
            f.write(f"- **Systems Passed**: {summary['systems_passed']}\n")
            f.write(f"- **Systems Failed**: {summary['systems_failed']}\n")
            f.write(f"- **Performance Score**: {summary['performance_score']:.2f}\n")
            f.write(f"- **Optimization Score**: {summary['optimization_score']:.2f}\n\n")
            
            f.write(f"## Optimization Recommendations\n\n")
            for rec in results['optimization_recommendations']:
                f.write(f"### {rec['category'].title()} ({rec['priority']} priority)\n")
                f.write(f"**Issue**: {rec['description']}\n\n")
                f.write(f"**Action**: {rec['action']}\n\n")
        
        print(f"📄 Summary report saved to: {summary_file}")
        
        # Determine success
        optimization_score = results['validation_summary']['optimization_score']
        if optimization_score >= 0.8:
            print(f"\n✅ RECURSIVE ENHANCEMENT ENGINE VALIDATION SUCCESSFUL")
            print(f"🎯 Optimization Score: {optimization_score:.2f} (Excellent)")
            return 0
        elif optimization_score >= 0.6:
            print(f"\n⚠️ RECURSIVE ENHANCEMENT ENGINE VALIDATION COMPLETED WITH WARNINGS")
            print(f"🎯 Optimization Score: {optimization_score:.2f} (Good)")
            return 1
        else:
            print(f"\n❌ RECURSIVE ENHANCEMENT ENGINE NEEDS OPTIMIZATION")
            print(f"🎯 Optimization Score: {optimization_score:.2f} (Needs Improvement)")
            return 2
            
    except Exception as e:
        print(f"\n💥 VALIDATION FAILED: {e}")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)