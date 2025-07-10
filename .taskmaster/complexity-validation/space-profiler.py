#!/usr/bin/env python3
"""
Space Complexity Measurement and Validation System
Validates O(√n) and O(log n · log log n) complexity bounds
"""

import time
import math
import json
import psutil
import gc
import tracemalloc
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a specific point in time"""
    timestamp: float
    total_memory: int  # RSS in bytes
    heap_size: int     # Python heap size
    allocation_size: int
    context: str
    n_value: int


@dataclass
class ComplexityResult:
    """Result of complexity validation test"""
    algorithm_name: str
    complexity_type: str
    n_value: int
    peak_memory: int
    theoretical_bound: float
    actual_ratio: float
    is_valid: bool
    confidence_interval: tuple
    execution_time: float


class SpaceComplexityProfiler:
    """Comprehensive space complexity profiler and validator"""
    
    def __init__(self, algorithm_name: str, expected_complexity: str):
        self.algorithm_name = algorithm_name
        self.expected_complexity = expected_complexity
        self.memory_snapshots: List[MemorySnapshot] = []
        self.peak_memory = 0
        self.start_memory = 0
        self.allocation_patterns = {}
        self.start_time = 0
        self.execution_time = 0
        self.constants = {"sqrt": 1000, "loglog": 100}  # Empirical constants
        
    def __enter__(self):
        """Context manager entry"""
        tracemalloc.start()
        self.start_memory = psutil.Process().memory_info().rss
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.execution_time = time.time() - self.start_time
        tracemalloc.stop()
        
    def track_allocation(self, size: int, context: str, n_value: int = 0):
        """Track memory allocation at specific execution point"""
        current_memory = psutil.Process().memory_info().rss
        heap_stats = gc.get_stats()
        heap_size = heap_stats[0]['size'] if heap_stats else 0
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            total_memory=current_memory,
            heap_size=heap_size,
            allocation_size=size,
            context=context,
            n_value=n_value
        )
        
        self.memory_snapshots.append(snapshot)
        self.peak_memory = max(self.peak_memory, current_memory - self.start_memory)
        
    def calculate_theoretical_bound(self, n: int) -> float:
        """Calculate theoretical memory bound for given n"""
        if self.expected_complexity == "O(sqrt(n))":
            return self.constants["sqrt"] * math.sqrt(max(1, n))
        elif self.expected_complexity == "O(log(n) * log(log(n)))":
            if n <= 2:
                return self.constants["loglog"]
            log_n = math.log(n)
            log_log_n = math.log(log_n) if log_n > 1 else 1
            return self.constants["loglog"] * log_n * log_log_n
        else:
            raise ValueError(f"Unsupported complexity type: {self.expected_complexity}")
    
    def validate_complexity_bound(self, n: int, tolerance: float = 0.15) -> bool:
        """Validate that measured memory usage respects complexity bound"""
        theoretical = self.calculate_theoretical_bound(n)
        return self.peak_memory <= theoretical * (1 + tolerance)
    
    def get_complexity_ratio(self, n: int) -> float:
        """Get ratio of actual memory usage to theoretical bound"""
        theoretical = self.calculate_theoretical_bound(n)
        return self.peak_memory / theoretical if theoretical > 0 else float('inf')


class ComplexityValidator:
    """Automated complexity validation framework"""
    
    def __init__(self):
        self.test_sizes = [100, 316, 1000, 3162, 10000, 31623, 100000]
        self.results: Dict[int, ComplexityResult] = {}
        self.regression_threshold = 0.05  # 5% performance degradation threshold
        
    def run_complexity_tests(self, algorithm_func: Callable[[int], Any], 
                           complexity_type: str, iterations: int = 3) -> Dict[str, Any]:
        """Run comprehensive complexity validation tests"""
        print(f"Running complexity tests for {algorithm_func.__name__}")
        print(f"Expected complexity: {complexity_type}")
        print(f"Test iterations per size: {iterations}")
        
        for n in self.test_sizes:
            print(f"\nTesting with n={n}")
            iteration_results = []
            
            for i in range(iterations):
                profiler = SpaceComplexityProfiler(algorithm_func.__name__, complexity_type)
                
                try:
                    with profiler:
                        # Instrument the algorithm execution
                        profiler.track_allocation(0, "algorithm_start", n)
                        result = algorithm_func(n)
                        profiler.track_allocation(0, "algorithm_end", n)
                    
                    # Calculate validation metrics
                    theoretical_bound = profiler.calculate_theoretical_bound(n)
                    is_valid = profiler.validate_complexity_bound(n)
                    ratio = profiler.get_complexity_ratio(n)
                    
                    iteration_result = ComplexityResult(
                        algorithm_name=algorithm_func.__name__,
                        complexity_type=complexity_type,
                        n_value=n,
                        peak_memory=profiler.peak_memory,
                        theoretical_bound=theoretical_bound,
                        actual_ratio=ratio,
                        is_valid=is_valid,
                        confidence_interval=(0, 0),  # Will calculate after all iterations
                        execution_time=profiler.execution_time
                    )
                    
                    iteration_results.append(iteration_result)
                    print(f"  Iteration {i+1}: {profiler.peak_memory:,} bytes, "
                          f"ratio: {ratio:.3f}, valid: {is_valid}")
                    
                except Exception as e:
                    print(f"  Iteration {i+1} failed: {e}")
                    continue
            
            if iteration_results:
                # Calculate statistics across iterations
                peak_memories = [r.peak_memory for r in iteration_results]
                ratios = [r.actual_ratio for r in iteration_results]
                
                # Use median for robust central tendency
                median_memory = statistics.median(peak_memories)
                median_ratio = statistics.median(ratios)
                
                # Calculate confidence interval
                if len(peak_memories) > 1:
                    confidence_interval = stats.t.interval(
                        0.95, len(peak_memories)-1,
                        loc=statistics.mean(peak_memories),
                        scale=stats.sem(peak_memories)
                    )
                else:
                    confidence_interval = (median_memory, median_memory)
                
                # Store aggregated result
                self.results[n] = ComplexityResult(
                    algorithm_name=algorithm_func.__name__,
                    complexity_type=complexity_type,
                    n_value=n,
                    peak_memory=median_memory,
                    theoretical_bound=iteration_results[0].theoretical_bound,
                    actual_ratio=median_ratio,
                    is_valid=median_ratio <= 1.15,  # 15% tolerance
                    confidence_interval=confidence_interval,
                    execution_time=statistics.median([r.execution_time for r in iteration_results])
                )
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze complexity test results and generate report"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Calculate statistics
        valid_tests = sum(1 for r in self.results.values() if r.is_valid)
        total_tests = len(self.results)
        success_rate = valid_tests / total_tests if total_tests > 0 else 0
        
        # Regression analysis
        n_values = [r.n_value for r in self.results.values()]
        ratios = [r.actual_ratio for r in self.results.values()]
        
        # Fit complexity curve
        if len(n_values) >= 3:
            complexity_type = list(self.results.values())[0].complexity_type
            if "sqrt" in complexity_type:
                theoretical_curve = [math.sqrt(n) for n in n_values]
            else:  # log n * log log n
                theoretical_curve = [
                    math.log(n) * math.log(math.log(n)) if n > 2 else 1 
                    for n in n_values
                ]
            
            correlation = stats.pearsonr(theoretical_curve, ratios)[0]
        else:
            correlation = 0.0
        
        # Generate analysis report
        analysis = {
            "summary": {
                "total_tests": total_tests,
                "valid_tests": valid_tests,
                "success_rate": success_rate,
                "complexity_correlation": correlation,
                "algorithm_name": list(self.results.values())[0].algorithm_name,
                "complexity_type": list(self.results.values())[0].complexity_type
            },
            "detailed_results": {
                str(k): {
                    "n_value": v.n_value,
                    "peak_memory_mb": v.peak_memory / (1024 * 1024),
                    "theoretical_bound_mb": v.theoretical_bound / (1024 * 1024),
                    "ratio": v.actual_ratio,
                    "is_valid": v.is_valid,
                    "execution_time_ms": v.execution_time * 1000,
                    "confidence_interval": v.confidence_interval
                }
                for k, v in self.results.items()
            },
            "performance_metrics": {
                "max_ratio": max(ratios) if ratios else 0,
                "min_ratio": min(ratios) if ratios else 0,
                "avg_ratio": statistics.mean(ratios) if ratios else 0,
                "ratio_std_dev": statistics.stdev(ratios) if len(ratios) > 1 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis
    
    def generate_visualization(self, output_path: str = "complexity_analysis.png"):
        """Generate complexity analysis visualization"""
        if not self.results:
            print("No results to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        n_values = [r.n_value for r in self.results.values()]
        memory_values = [r.peak_memory / (1024 * 1024) for r in self.results.values()]  # Convert to MB
        ratios = [r.actual_ratio for r in self.results.values()]
        
        # Plot 1: Memory usage vs input size
        ax1.loglog(n_values, memory_values, 'bo-', label='Measured Memory')
        
        # Plot theoretical curves
        complexity_type = list(self.results.values())[0].complexity_type
        if "sqrt" in complexity_type:
            theoretical = [math.sqrt(n) * 1000 / (1024 * 1024) for n in n_values]  # Convert to MB
            ax1.loglog(n_values, theoretical, 'r--', label='O(√n) Theoretical')
        else:
            theoretical = [
                (math.log(n) * math.log(math.log(n)) if n > 2 else 1) * 100 / (1024 * 1024)
                for n in n_values
            ]
            ax1.loglog(n_values, theoretical, 'r--', label='O(log n · log log n) Theoretical')
        
        ax1.set_xlabel('Input Size (n)')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Space Complexity Validation')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Ratio analysis
        ax2.semilogx(n_values, ratios, 'go-', label='Actual/Theoretical Ratio')
        ax2.axhline(y=1.0, color='r', linestyle='--', label='Theoretical Bound')
        ax2.axhline(y=1.15, color='orange', linestyle='--', label='15% Tolerance')
        ax2.set_xlabel('Input Size (n)')
        ax2.set_ylabel('Memory Ratio')
        ax2.set_title('Complexity Bound Validation')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")


# Example algorithm implementations for testing
def sqrt_space_algorithm(n: int) -> int:
    """Example O(√n) space algorithm - simulated task decomposition"""
    # Simulate sqrt(n) space usage
    chunk_size = max(1, int(math.sqrt(n)))
    chunks = []
    
    for i in range(chunk_size):
        # Each chunk stores sqrt(n) elements
        chunk = list(range(chunk_size))
        chunks.append(chunk)
    
    # Process chunks (simulated work)
    result = 0
    for chunk in chunks:
        result += sum(chunk)
    
    return result


def loglog_space_algorithm(n: int) -> int:
    """Example O(log n · log log n) space algorithm - tree evaluation"""
    if n <= 2:
        return n
    
    # Simulate tree evaluation with O(log n · log log n) auxiliary space
    log_n = int(math.log(n))
    log_log_n = max(1, int(math.log(log_n)) if log_n > 1 else 1)
    
    # Auxiliary data structures
    stack = list(range(log_n))
    memoization = {}
    
    for i in range(log_log_n):
        for j in range(log_n):
            key = (i, j)
            memoization[key] = i * j
    
    # Simulated tree evaluation work
    result = sum(stack) + sum(memoization.values())
    return result


def run_benchmarks():
    """Run comprehensive benchmarks on optimization algorithms"""
    validator = ComplexityValidator()
    
    print("=" * 60)
    print("SPACE COMPLEXITY VALIDATION SYSTEM")
    print("=" * 60)
    
    # Test O(√n) algorithm
    print("\n1. Testing O(√n) Space Algorithm")
    print("-" * 40)
    sqrt_results = validator.run_complexity_tests(
        sqrt_space_algorithm, "O(sqrt(n))", iterations=3
    )
    
    # Reset for next test
    validator.results.clear()
    
    # Test O(log n · log log n) algorithm  
    print("\n2. Testing O(log n · log log n) Space Algorithm")
    print("-" * 50)
    loglog_results = validator.run_complexity_tests(
        loglog_space_algorithm, "O(log(n) * log(log(n)))", iterations=3
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"sqrt_complexity_report_{timestamp}.json", "w") as f:
        json.dump(sqrt_results, f, indent=2)
    
    with open(f"loglog_complexity_report_{timestamp}.json", "w") as f:
        json.dump(loglog_results, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"O(√n) Success Rate: {sqrt_results['summary']['success_rate']:.1%}")
    print(f"O(log n·log log n) Success Rate: {loglog_results['summary']['success_rate']:.1%}")
    print(f"Reports saved with timestamp: {timestamp}")


if __name__ == "__main__":
    run_benchmarks()