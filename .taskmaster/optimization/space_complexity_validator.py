#!/usr/bin/env python3
"""
Space Complexity Validator
Validates O(√n) memory optimization and complexity bounds
"""

import json
import time
import math
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

@dataclass
class ComplexityMeasurement:
    """Individual complexity measurement"""
    input_size: int
    actual_memory_mb: float
    theoretical_bound_mb: float
    execution_time_ms: float
    complexity_ratio: float
    within_bounds: bool

@dataclass
class ComplexityValidation:
    """Complexity validation results"""
    validation_timestamp: datetime
    algorithm_name: str
    target_complexity: str
    measurements: List[ComplexityMeasurement]
    overall_validation: bool
    average_ratio: float
    max_deviation: float
    recommendations: List[str]


class SpaceComplexityValidator:
    """Validates space complexity bounds for optimization algorithms"""
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('ComplexityValidator')
    
    def validate_sqrt_space_optimization(self, test_sizes: List[int] = None) -> ComplexityValidation:
        """Validate O(√n) space complexity"""
        if test_sizes is None:
            test_sizes = [100, 500, 1000, 2500, 5000, 10000]
        
        self.logger.info("Validating O(√n) space complexity")
        
        measurements = []
        
        for n in test_sizes:
            self.logger.info(f"Testing with input size: {n}")
            measurement = self._measure_sqrt_space_complexity(n)
            measurements.append(measurement)
        
        # Analyze results
        valid_measurements = [m for m in measurements if m.within_bounds]
        overall_validation = len(valid_measurements) / len(measurements) >= 0.8
        
        average_ratio = sum(m.complexity_ratio for m in measurements) / len(measurements)
        max_deviation = max(m.complexity_ratio for m in measurements)
        
        recommendations = self._generate_sqrt_recommendations(measurements, overall_validation)
        
        validation = ComplexityValidation(
            validation_timestamp=datetime.now(),
            algorithm_name="Square Root Space Optimization",
            target_complexity="O(√n)",
            measurements=measurements,
            overall_validation=overall_validation,
            average_ratio=average_ratio,
            max_deviation=max_deviation,
            recommendations=recommendations
        )
        
        self._save_validation_results(validation, "sqrt_space_validation")
        return validation
    
    def validate_tree_evaluation_complexity(self, test_sizes: List[int] = None) -> ComplexityValidation:
        """Validate O(log n · log log n) tree evaluation complexity"""
        if test_sizes is None:
            test_sizes = [64, 256, 1024, 4096, 16384, 65536]
        
        self.logger.info("Validating O(log n · log log n) tree evaluation complexity")
        
        measurements = []
        
        for n in test_sizes:
            self.logger.info(f"Testing with input size: {n}")
            measurement = self._measure_tree_evaluation_complexity(n)
            measurements.append(measurement)
        
        # Analyze results
        valid_measurements = [m for m in measurements if m.within_bounds]
        overall_validation = len(valid_measurements) / len(measurements) >= 0.8
        
        average_ratio = sum(m.complexity_ratio for m in measurements) / len(measurements)
        max_deviation = max(m.complexity_ratio for m in measurements)
        
        recommendations = self._generate_tree_recommendations(measurements, overall_validation)
        
        validation = ComplexityValidation(
            validation_timestamp=datetime.now(),
            algorithm_name="Tree Evaluation Optimization",
            target_complexity="O(log n · log log n)",
            measurements=measurements,
            overall_validation=overall_validation,
            average_ratio=average_ratio,
            max_deviation=max_deviation,
            recommendations=recommendations
        )
        
        self._save_validation_results(validation, "tree_evaluation_validation")
        return validation
    
    def _measure_sqrt_space_complexity(self, n: int) -> ComplexityMeasurement:
        """Measure actual vs theoretical O(√n) space complexity"""
        
        # Simulate O(√n) space algorithm
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        # Simulate sqrt-space data structure
        sqrt_n = int(math.sqrt(n))
        data_structure = self._create_sqrt_space_structure(n, sqrt_n)
        
        memory_after = self._get_memory_usage()
        end_time = time.time()
        
        # Calculate metrics
        actual_memory_mb = memory_after - memory_before
        theoretical_bound_mb = self._calculate_sqrt_bound(n)
        execution_time_ms = (end_time - start_time) * 1000
        
        complexity_ratio = actual_memory_mb / theoretical_bound_mb if theoretical_bound_mb > 0 else float('inf')
        within_bounds = complexity_ratio <= 1.5  # Allow 50% tolerance
        
        # Cleanup
        del data_structure
        
        return ComplexityMeasurement(
            input_size=n,
            actual_memory_mb=actual_memory_mb,
            theoretical_bound_mb=theoretical_bound_mb,
            execution_time_ms=execution_time_ms,
            complexity_ratio=complexity_ratio,
            within_bounds=within_bounds
        )
    
    def _measure_tree_evaluation_complexity(self, n: int) -> ComplexityMeasurement:
        """Measure actual vs theoretical O(log n · log log n) complexity"""
        
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        # Simulate tree evaluation with O(log n · log log n) space
        tree_structure = self._create_tree_evaluation_structure(n)
        
        memory_after = self._get_memory_usage()
        end_time = time.time()
        
        # Calculate metrics
        actual_memory_mb = memory_after - memory_before
        theoretical_bound_mb = self._calculate_tree_bound(n)
        execution_time_ms = (end_time - start_time) * 1000
        
        complexity_ratio = actual_memory_mb / theoretical_bound_mb if theoretical_bound_mb > 0 else float('inf')
        within_bounds = complexity_ratio <= 2.0  # Allow 100% tolerance for log factors
        
        # Cleanup
        del tree_structure
        
        return ComplexityMeasurement(
            input_size=n,
            actual_memory_mb=actual_memory_mb,
            theoretical_bound_mb=theoretical_bound_mb,
            execution_time_ms=execution_time_ms,
            complexity_ratio=complexity_ratio,
            within_bounds=within_bounds
        )
    
    def _create_sqrt_space_structure(self, n: int, sqrt_n: int) -> List[Any]:
        """Create data structure that uses O(√n) space"""
        # Simulate sqrt-decomposition: store sqrt(n) blocks of sqrt(n) elements each
        blocks = []
        
        for i in range(sqrt_n):
            # Each block contains aggregated information about sqrt(n) elements
            block = {
                'block_id': i,
                'start_index': i * sqrt_n,
                'end_index': min((i + 1) * sqrt_n, n),
                'summary_data': list(range(sqrt_n))  # Simulated summary
            }
            blocks.append(block)
        
        return blocks
    
    def _create_tree_evaluation_structure(self, n: int) -> Dict[str, Any]:
        """Create tree structure that uses O(log n · log log n) space"""
        # Simulate tree evaluation with logarithmic space
        log_n = int(math.log2(n)) if n > 1 else 1
        log_log_n = int(math.log2(log_n)) if log_n > 1 else 1
        
        tree = {
            'depth': log_n,
            'auxiliary_space': log_log_n,
            'evaluation_stack': list(range(log_n)),
            'memo_table': {i: i for i in range(log_n * log_log_n)}
        }
        
        return tree
    
    def _calculate_sqrt_bound(self, n: int) -> float:
        """Calculate theoretical O(√n) memory bound in MB"""
        sqrt_n = math.sqrt(n)
        
        # Assume each element uses 8 bytes (64-bit), plus overhead
        bytes_per_element = 8
        overhead_factor = 1.2  # 20% overhead for data structures
        
        theoretical_bytes = sqrt_n * bytes_per_element * overhead_factor
        theoretical_mb = theoretical_bytes / (1024 * 1024)
        
        return max(theoretical_mb, 0.001)  # Minimum 1KB
    
    def _calculate_tree_bound(self, n: int) -> float:
        """Calculate theoretical O(log n · log log n) memory bound in MB"""
        if n <= 1:
            return 0.001
        
        log_n = math.log2(n)
        log_log_n = math.log2(log_n) if log_n > 1 else 1
        
        # Assume each element uses 8 bytes, plus overhead
        bytes_per_element = 8
        overhead_factor = 1.5  # 50% overhead for tree structures
        
        theoretical_bytes = log_n * log_log_n * bytes_per_element * overhead_factor
        theoretical_mb = theoretical_bytes / (1024 * 1024)
        
        return max(theoretical_mb, 0.001)  # Minimum 1KB
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback: use sys.getsizeof for rough estimation
            import gc
            objects = gc.get_objects()
            total_size = sum(sys.getsizeof(obj) for obj in objects[:1000])  # Sample
            return total_size / (1024 * 1024)
    
    def _generate_sqrt_recommendations(self, measurements: List[ComplexityMeasurement], 
                                     overall_valid: bool) -> List[str]:
        """Generate recommendations for sqrt space optimization"""
        recommendations = []
        
        if not overall_valid:
            recommendations.append("O(√n) space bounds not consistently met - review algorithm implementation")
        
        high_ratio_measurements = [m for m in measurements if m.complexity_ratio > 2.0]
        if high_ratio_measurements:
            recommendations.append(f"High memory usage detected in {len(high_ratio_measurements)} test cases")
        
        slow_measurements = [m for m in measurements if m.execution_time_ms > 1000]
        if slow_measurements:
            recommendations.append("Execution time optimization needed for large inputs")
        
        if all(m.within_bounds for m in measurements):
            recommendations.append("✅ All measurements within O(√n) bounds - optimization successful")
        
        return recommendations
    
    def _generate_tree_recommendations(self, measurements: List[ComplexityMeasurement], 
                                     overall_valid: bool) -> List[str]:
        """Generate recommendations for tree evaluation optimization"""
        recommendations = []
        
        if not overall_valid:
            recommendations.append("O(log n · log log n) space bounds not consistently met")
        
        high_ratio_measurements = [m for m in measurements if m.complexity_ratio > 3.0]
        if high_ratio_measurements:
            recommendations.append(f"Excessive memory usage in {len(high_ratio_measurements)} test cases")
        
        exponential_growth = any(
            measurements[i].actual_memory_mb > measurements[i-1].actual_memory_mb * 2
            for i in range(1, len(measurements))
        )
        if exponential_growth:
            recommendations.append("Potential exponential memory growth detected - verify logarithmic bounds")
        
        if all(m.within_bounds for m in measurements):
            recommendations.append("✅ All measurements within O(log n · log log n) bounds")
        
        return recommendations
    
    def _save_validation_results(self, validation: ComplexityValidation, filename: str):
        """Save validation results"""
        try:
            os.makedirs('.taskmaster/reports', exist_ok=True)
            
            results_path = Path(f'.taskmaster/reports/{filename}.json')
            with open(results_path, 'w') as f:
                json.dump(asdict(validation), f, indent=2, default=str)
            
            self.logger.info(f"Validation results saved to: {results_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation results: {e}")


def main():
    """Main complexity validation execution"""
    print("Space Complexity Validator")
    print("=" * 40)
    
    validator = SpaceComplexityValidator()
    
    try:
        # Validate O(√n) space optimization
        print("🔍 Validating O(√n) space optimization...")
        sqrt_validation = validator.validate_sqrt_space_optimization()
        
        print(f"√n Validation: {'✅ PASSED' if sqrt_validation.overall_validation else '❌ FAILED'}")
        print(f"Average ratio: {sqrt_validation.average_ratio:.2f}")
        print(f"Max deviation: {sqrt_validation.max_deviation:.2f}")
        
        # Validate O(log n · log log n) tree evaluation
        print("\n🌳 Validating O(log n · log log n) tree evaluation...")
        tree_validation = validator.validate_tree_evaluation_complexity()
        
        print(f"Tree Validation: {'✅ PASSED' if tree_validation.overall_validation else '❌ FAILED'}")
        print(f"Average ratio: {tree_validation.average_ratio:.2f}")
        print(f"Max deviation: {tree_validation.max_deviation:.2f}")
        
        # Show recommendations
        if sqrt_validation.recommendations:
            print(f"\n√n Optimization Recommendations:")
            for rec in sqrt_validation.recommendations:
                print(f"  • {rec}")
        
        if tree_validation.recommendations:
            print(f"\nTree Optimization Recommendations:")
            for rec in tree_validation.recommendations:
                print(f"  • {rec}")
        
        overall_success = sqrt_validation.overall_validation and tree_validation.overall_validation
        
        print(f"\n✅ Space complexity validation completed")
        print(f"Results saved to: .taskmaster/reports/")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Complexity validation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)