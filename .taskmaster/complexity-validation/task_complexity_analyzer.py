#!/usr/bin/env python3
"""
Advanced Task Complexity Analysis and Optimization Engine
Implements sophisticated computational complexity analysis and optimization strategies.
"""

import json
import psutil
import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import concurrent.futures
import threading
import logging

@dataclass
class TaskComplexity:
    """Represents computational complexity metrics for a task"""
    task_id: str
    time_complexity: str  # e.g., "O(n)", "O(log n)", "O(n^2)"
    space_complexity: str  # e.g., "O(1)", "O(n)", "O(log n)"
    time_coefficient: float  # Actual multiplier for complexity
    space_coefficient: float  # Actual memory requirement in MB
    io_requirements: Dict[str, float]  # disk_read_mb, disk_write_mb, network_mb
    parallelization_factor: float  # 0.0 (serial) to 1.0 (fully parallel)
    cpu_intensity: float  # 0.0 (low) to 1.0 (high)
    memory_access_pattern: str  # "sequential", "random", "mixed"
    estimated_duration: float  # seconds
    
class TaskComplexityAnalyzer:
    """Analyzes computational complexity of tasks"""
    
    def __init__(self, system_info: Optional[Dict] = None):
        self.system_info = system_info or self._get_system_info()
        self.complexity_patterns = self._load_complexity_patterns()
        self.logger = logging.getLogger(__name__)
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system resource information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_io_counters': psutil.disk_io_counters(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
    
    def _load_complexity_patterns(self) -> Dict[str, Dict]:
        """Load known complexity patterns for different task types"""
        return {
            'prd_generation': {
                'base_time_complexity': 'O(n)',
                'base_space_complexity': 'O(n)',
                'time_coefficient': 0.1,
                'space_coefficient': 2.0,
                'io_factor': 0.5,
                'cpu_intensity': 0.3
            },
            'dependency_analysis': {
                'base_time_complexity': 'O(n + e)',
                'base_space_complexity': 'O(n)',
                'time_coefficient': 0.05,
                'space_coefficient': 1.0,
                'io_factor': 0.2,
                'cpu_intensity': 0.4
            },
            'optimization': {
                'base_time_complexity': 'O(n log n)',
                'base_space_complexity': 'O(n)',
                'time_coefficient': 0.2,
                'space_coefficient': 3.0,
                'io_factor': 0.3,
                'cpu_intensity': 0.8
            },
            'validation': {
                'base_time_complexity': 'O(n)',
                'base_space_complexity': 'O(1)',
                'time_coefficient': 0.02,
                'space_coefficient': 0.5,
                'io_factor': 0.1,
                'cpu_intensity': 0.2
            },
            'monitoring': {
                'base_time_complexity': 'O(1)',
                'base_space_complexity': 'O(1)',
                'time_coefficient': 0.01,
                'space_coefficient': 0.1,
                'io_factor': 0.05,
                'cpu_intensity': 0.1
            }
        }
    
    def analyze_task(self, task: Dict[str, Any]) -> TaskComplexity:
        """Analyze computational complexity of a single task"""
        task_id = task.get('id', 'unknown')
        title = task.get('title', '').lower()
        description = task.get('description', '').lower()
        details = task.get('details', '').lower()
        
        # Determine task type from content analysis
        task_type = self._classify_task_type(title, description, details)
        pattern = self.complexity_patterns.get(task_type, self.complexity_patterns['validation'])
        
        # Estimate task size (n) from content analysis
        task_size = self._estimate_task_size(task)
        
        # Calculate complexity metrics
        time_complexity = pattern['base_time_complexity']
        space_complexity = pattern['base_space_complexity']
        time_coefficient = pattern['time_coefficient'] * task_size
        space_coefficient = pattern['space_coefficient'] * task_size
        
        # Analyze I/O requirements
        io_requirements = self._analyze_io_requirements(task, pattern['io_factor'])
        
        # Determine parallelization potential
        parallelization_factor = self._calculate_parallelization_factor(task, task_type)
        
        # Estimate duration based on system capabilities
        estimated_duration = self._estimate_duration(
            time_coefficient, task_size, parallelization_factor
        )
        
        return TaskComplexity(
            task_id=str(task_id),
            time_complexity=time_complexity,
            space_complexity=space_complexity,
            time_coefficient=time_coefficient,
            space_coefficient=space_coefficient,
            io_requirements=io_requirements,
            parallelization_factor=parallelization_factor,
            cpu_intensity=pattern['cpu_intensity'],
            memory_access_pattern=self._determine_memory_pattern(task_type),
            estimated_duration=estimated_duration
        )
    
    def _classify_task_type(self, title: str, description: str, details: str) -> str:
        """Classify task type based on content analysis"""
        content = f"{title} {description} {details}"
        
        if any(keyword in content for keyword in ['prd', 'generate', 'create', 'recursive']):
            return 'prd_generation'
        elif any(keyword in content for keyword in ['dependency', 'analyze', 'graph', 'tree']):
            return 'dependency_analysis'
        elif any(keyword in content for keyword in ['optimize', 'algorithm', 'complexity', 'sqrt']):
            return 'optimization'
        elif any(keyword in content for keyword in ['validate', 'check', 'verify', 'conform']):
            return 'validation'
        elif any(keyword in content for keyword in ['monitor', 'dashboard', 'log', 'track']):
            return 'monitoring'
        else:
            return 'validation'  # Default fallback
    
    def _estimate_task_size(self, task: Dict[str, Any]) -> float:
        """Estimate task size (n) based on content and complexity indicators"""
        details = task.get('details', '')
        description = task.get('description', '')
        
        # Base size from content length
        content_length = len(details) + len(description)
        base_size = max(1.0, content_length / 100)  # Normalize to reasonable range
        
        # Adjust based on complexity indicators
        complexity_indicators = [
            'comprehensive', 'advanced', 'sophisticated', 'complex',
            'multiple', 'extensive', 'detailed', 'complete'
        ]
        
        content = f"{details} {description}".lower()
        complexity_multiplier = 1.0
        for indicator in complexity_indicators:
            if indicator in content:
                complexity_multiplier += 0.5
        
        # Check for numeric indicators
        if 'thousand' in content or '1000' in content:
            complexity_multiplier += 2.0
        if 'million' in content:
            complexity_multiplier += 5.0
        
        return base_size * complexity_multiplier
    
    def _analyze_io_requirements(self, task: Dict[str, Any], io_factor: float) -> Dict[str, float]:
        """Analyze I/O requirements for a task"""
        content = f"{task.get('description', '')} {task.get('details', '')}".lower()
        
        # Base I/O requirements
        disk_read = io_factor * 10  # MB
        disk_write = io_factor * 5  # MB
        network = 0.0  # MB
        
        # Adjust based on content analysis
        if any(keyword in content for keyword in ['file', 'directory', 'log', 'report']):
            disk_read *= 2
            disk_write *= 2
        
        if any(keyword in content for keyword in ['api', 'network', 'download', 'fetch']):
            network = io_factor * 50
        
        if any(keyword in content for keyword in ['large', 'big', 'extensive', 'comprehensive']):
            disk_read *= 3
            disk_write *= 3
        
        return {
            'disk_read_mb': disk_read,
            'disk_write_mb': disk_write,
            'network_mb': network
        }
    
    def _calculate_parallelization_factor(self, task: Dict[str, Any], task_type: str) -> float:
        """Calculate how much a task can benefit from parallelization"""
        content = f"{task.get('description', '')} {task.get('details', '')}".lower()
        
        # Base parallelization potential by task type
        base_factors = {
            'prd_generation': 0.7,  # Can parallelize multiple PRDs
            'dependency_analysis': 0.3,  # Graph analysis has sequential components
            'optimization': 0.5,  # Some algorithms can parallelize
            'validation': 0.8,  # Can validate multiple items in parallel
            'monitoring': 0.2   # Often requires sequential access
        }
        
        base_factor = base_factors.get(task_type, 0.5)
        
        # Adjust based on content
        if any(keyword in content for keyword in ['parallel', 'concurrent', 'multiple']):
            base_factor = min(1.0, base_factor + 0.3)
        
        if any(keyword in content for keyword in ['sequential', 'serial', 'dependent']):
            base_factor = max(0.1, base_factor - 0.3)
        
        return base_factor
    
    def _determine_memory_pattern(self, task_type: str) -> str:
        """Determine memory access pattern for a task type"""
        patterns = {
            'prd_generation': 'sequential',
            'dependency_analysis': 'random',
            'optimization': 'mixed',
            'validation': 'sequential',
            'monitoring': 'sequential'
        }
        return patterns.get(task_type, 'mixed')
    
    def _estimate_duration(self, time_coefficient: float, task_size: float, 
                          parallelization_factor: float) -> float:
        """Estimate task execution duration in seconds"""
        # Base calculation considering system capabilities
        cpu_factor = 1.0 / max(1, self.system_info['cpu_count'])
        memory_factor = min(1.0, self.system_info['memory_available'] / (1024**3))  # GB
        
        # Parallel efficiency
        parallel_efficiency = 1.0 - (parallelization_factor * (1.0 - cpu_factor))
        
        # Base duration calculation
        base_duration = time_coefficient * task_size * parallel_efficiency
        
        # Adjust for memory pressure
        if memory_factor < 0.5:  # Less than 512MB available
            base_duration *= 2.0
        elif memory_factor < 0.25:  # Less than 256MB available
            base_duration *= 4.0
        
        return max(1.0, base_duration)  # Minimum 1 second

class OptimizationEngine:
    """Optimizes task execution order based on complexity analysis"""
    
    def __init__(self, analyzer: TaskComplexityAnalyzer):
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)
        
    def optimize_execution_order(self, tasks: List[Dict[str, Any]], 
                                dependencies: Dict[str, List[str]],
                                strategy: str = 'adaptive') -> List[str]:
        """Optimize task execution order using specified strategy"""
        
        # Analyze complexity for all tasks
        task_complexities = {}
        for task in tasks:
            complexity = self.analyzer.analyze_task(task)
            task_complexities[complexity.task_id] = complexity
        
        if strategy == 'greedy':
            return self._greedy_scheduling(task_complexities, dependencies)
        elif strategy == 'dynamic_programming':
            return self._dynamic_programming_scheduling(task_complexities, dependencies)
        elif strategy == 'machine_learning':
            return self._ml_based_scheduling(task_complexities, dependencies)
        else:  # adaptive
            return self._adaptive_scheduling(task_complexities, dependencies)
    
    def _greedy_scheduling(self, complexities: Dict[str, TaskComplexity],
                          dependencies: Dict[str, List[str]]) -> List[str]:
        """Greedy scheduling: always pick the task with highest efficiency ratio"""
        scheduled = []
        remaining = set(complexities.keys())
        
        while remaining:
            # Find tasks with satisfied dependencies
            available = []
            for task_id in remaining:
                deps = dependencies.get(task_id, [])
                if all(dep in scheduled for dep in deps):
                    available.append(task_id)
            
            if not available:
                # Handle circular dependencies by picking lowest ID
                available = [min(remaining)]
            
            # Calculate efficiency ratio (inverse of duration/parallelization)
            best_task = None
            best_efficiency = -1
            
            for task_id in available:
                complexity = complexities[task_id]
                efficiency = (complexity.parallelization_factor + 0.1) / (complexity.estimated_duration + 0.1)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_task = task_id
            
            scheduled.append(best_task)
            remaining.remove(best_task)
        
        return scheduled
    
    def _dynamic_programming_scheduling(self, complexities: Dict[str, TaskComplexity],
                                       dependencies: Dict[str, List[str]]) -> List[str]:
        """Dynamic programming approach for optimal scheduling"""
        # Simplified DP approach: minimize total execution time
        tasks = list(complexities.keys())
        n = len(tasks)
        
        # Build dependency matrix
        dep_matrix = {}
        for i, task in enumerate(tasks):
            dep_matrix[task] = dependencies.get(task, [])
        
        # Use topological sort with optimization
        in_degree = {task: 0 for task in tasks}
        for task in tasks:
            for dep in dep_matrix[task]:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Priority queue with duration as priority
        available = [(complexities[task].estimated_duration, task) 
                    for task in tasks if in_degree[task] == 0]
        available.sort()
        
        scheduled = []
        while available:
            _, task = available.pop(0)
            scheduled.append(task)
            
            # Update dependencies
            for dependent in tasks:
                if task in dep_matrix[dependent]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        duration = complexities[dependent].estimated_duration
                        available.append((duration, dependent))
                        available.sort()
        
        return scheduled
    
    def _ml_based_scheduling(self, complexities: Dict[str, TaskComplexity],
                            dependencies: Dict[str, List[str]]) -> List[str]:
        """Machine learning-based scheduling (simplified heuristic)"""
        # For this implementation, use a weighted scoring system
        # In production, this would use actual ML models
        
        tasks = list(complexities.keys())
        task_scores = {}
        
        for task_id in tasks:
            complexity = complexities[task_id]
            
            # Feature vector for ML scoring
            features = [
                complexity.time_coefficient,
                complexity.space_coefficient,
                complexity.parallelization_factor,
                complexity.cpu_intensity,
                len(dependencies.get(task_id, [])),  # dependency count
                complexity.estimated_duration
            ]
            
            # Simplified ML scoring (weighted sum)
            weights = [0.2, 0.15, 0.3, 0.1, 0.1, 0.15]
            score = sum(f * w for f, w in zip(features, weights))
            task_scores[task_id] = score
        
        # Sort by score and respect dependencies
        return self._topological_sort_with_scores(task_scores, dependencies)
    
    def _adaptive_scheduling(self, complexities: Dict[str, TaskComplexity],
                            dependencies: Dict[str, List[str]]) -> List[str]:
        """Adaptive scheduling that adjusts based on current system state"""
        system_state = self.analyzer._get_system_info()
        
        # Adjust strategy based on system resources
        memory_pressure = 1.0 - (system_state['memory_available'] / system_state['memory_total'])
        cpu_load = psutil.cpu_percent(interval=1) / 100.0
        
        if memory_pressure > 0.8:  # High memory pressure
            # Prioritize low memory tasks
            return self._memory_conscious_scheduling(complexities, dependencies)
        elif cpu_load > 0.8:  # High CPU load
            # Prioritize parallelizable tasks
            return self._cpu_conscious_scheduling(complexities, dependencies)
        else:
            # Use balanced approach
            return self._greedy_scheduling(complexities, dependencies)
    
    def _memory_conscious_scheduling(self, complexities: Dict[str, TaskComplexity],
                                    dependencies: Dict[str, List[str]]) -> List[str]:
        """Schedule tasks prioritizing low memory usage"""
        tasks = list(complexities.keys())
        
        # Sort by memory requirement (ascending)
        memory_sorted = sorted(tasks, 
                              key=lambda t: complexities[t].space_coefficient)
        
        return self._topological_sort_with_preference(memory_sorted, dependencies)
    
    def _cpu_conscious_scheduling(self, complexities: Dict[str, TaskComplexity],
                                 dependencies: Dict[str, List[str]]) -> List[str]:
        """Schedule tasks prioritizing parallelizable tasks"""
        tasks = list(complexities.keys())
        
        # Sort by parallelization factor (descending)
        parallel_sorted = sorted(tasks, 
                                key=lambda t: complexities[t].parallelization_factor,
                                reverse=True)
        
        return self._topological_sort_with_preference(parallel_sorted, dependencies)
    
    def _topological_sort_with_preference(self, preferred_order: List[str],
                                         dependencies: Dict[str, List[str]]) -> List[str]:
        """Topological sort that respects preference order when possible"""
        scheduled = []
        remaining = set(preferred_order)
        
        while remaining:
            # Find available tasks (dependencies satisfied)
            available = []
            for task_id in preferred_order:
                if task_id in remaining:
                    deps = dependencies.get(task_id, [])
                    if all(dep in scheduled for dep in deps):
                        available.append(task_id)
            
            if not available:
                # Handle circular dependencies
                available = [min(remaining)]
            
            # Pick first available task from preferred order
            next_task = available[0]
            scheduled.append(next_task)
            remaining.remove(next_task)
        
        return scheduled
    
    def _topological_sort_with_scores(self, task_scores: Dict[str, float],
                                     dependencies: Dict[str, List[str]]) -> List[str]:
        """Topological sort using task scores as tie-breaker"""
        scheduled = []
        remaining = set(task_scores.keys())
        
        while remaining:
            # Find available tasks
            available = []
            for task_id in remaining:
                deps = dependencies.get(task_id, [])
                if all(dep in scheduled for dep in deps):
                    available.append(task_id)
            
            if not available:
                available = [min(remaining)]
            
            # Pick task with best score
            best_task = max(available, key=lambda t: task_scores[t])
            scheduled.append(best_task)
            remaining.remove(best_task)
        
        return scheduled

def generate_complexity_report(tasks: List[Dict[str, Any]], 
                              dependencies: Dict[str, List[str]],
                              output_file: str = None) -> Dict[str, Any]:
    """Generate comprehensive complexity analysis report"""
    
    analyzer = TaskComplexityAnalyzer()
    optimizer = OptimizationEngine(analyzer)
    
    # Analyze all tasks
    task_complexities = {}
    total_time = 0
    total_memory = 0
    
    for task in tasks:
        complexity = analyzer.analyze_task(task)
        task_complexities[complexity.task_id] = complexity
        total_time += complexity.estimated_duration
        total_memory += complexity.space_coefficient
    
    # Generate optimized execution orders
    strategies = ['greedy', 'dynamic_programming', 'adaptive']
    optimized_orders = {}
    
    for strategy in strategies:
        try:
            order = optimizer.optimize_execution_order(tasks, dependencies, strategy)
            optimized_orders[strategy] = order
        except Exception as e:
            optimized_orders[strategy] = f"Error: {str(e)}"
    
    # Generate report
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'system_info': analyzer.system_info,
        'task_count': len(tasks),
        'total_estimated_time': total_time,
        'total_memory_requirement': total_memory,
        'complexity_analysis': {
            task_id: asdict(complexity) 
            for task_id, complexity in task_complexities.items()
        },
        'optimization_strategies': optimized_orders,
        'bottlenecks': _identify_bottlenecks(task_complexities),
        'recommendations': _generate_recommendations(task_complexities, analyzer.system_info)
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report

def _identify_bottlenecks(complexities: Dict[str, TaskComplexity]) -> List[Dict[str, Any]]:
    """Identify potential bottlenecks in task execution"""
    bottlenecks = []
    
    # Find tasks with high duration
    durations = [c.estimated_duration for c in complexities.values()]
    if durations:
        avg_duration = sum(durations) / len(durations)
        std_duration = np.std(durations)
        threshold = avg_duration + 2 * std_duration
        
        for task_id, complexity in complexities.items():
            if complexity.estimated_duration > threshold:
                bottlenecks.append({
                    'task_id': task_id,
                    'type': 'high_duration',
                    'value': complexity.estimated_duration,
                    'threshold': threshold
                })
    
    # Find tasks with high memory usage
    memory_reqs = [c.space_coefficient for c in complexities.values()]
    if memory_reqs:
        avg_memory = sum(memory_reqs) / len(memory_reqs)
        std_memory = np.std(memory_reqs)
        memory_threshold = avg_memory + 2 * std_memory
        
        for task_id, complexity in complexities.items():
            if complexity.space_coefficient > memory_threshold:
                bottlenecks.append({
                    'task_id': task_id,
                    'type': 'high_memory',
                    'value': complexity.space_coefficient,
                    'threshold': memory_threshold
                })
    
    return bottlenecks

def _generate_recommendations(complexities: Dict[str, TaskComplexity], 
                             system_info: Dict[str, Any]) -> List[str]:
    """Generate optimization recommendations"""
    recommendations = []
    
    # Check memory pressure
    total_memory_req = sum(c.space_coefficient for c in complexities.values())
    available_memory_gb = system_info['memory_available'] / (1024**3)
    
    if total_memory_req > available_memory_gb * 1000:  # Convert to MB
        recommendations.append(
            "Consider breaking down high-memory tasks or increasing system memory"
        )
    
    # Check parallelization opportunities
    highly_parallel = sum(1 for c in complexities.values() 
                         if c.parallelization_factor > 0.7)
    cpu_count = system_info['cpu_count']
    
    if highly_parallel > cpu_count:
        recommendations.append(
            f"Consider increasing CPU cores or scheduling parallel tasks separately "
            f"({highly_parallel} highly parallel tasks, {cpu_count} cores available)"
        )
    
    # Check for I/O bottlenecks
    high_io_tasks = sum(1 for c in complexities.values() 
                       if c.io_requirements['disk_read_mb'] > 100)
    
    if high_io_tasks > 3:
        recommendations.append(
            "Consider using SSD storage or implementing I/O caching for better performance"
        )
    
    return recommendations

if __name__ == "__main__":
    # Example usage
    sample_tasks = [
        {
            'id': 1,
            'title': 'Initialize Project Environment',
            'description': 'Set up working environment with directories and environment variables',
            'details': 'Create directory structure, set environment variables, enable logging'
        },
        {
            'id': 2,
            'title': 'Implement Advanced Optimization Engine',
            'description': 'Create sophisticated optimization system with multiple algorithms',
            'details': 'Implement greedy scheduling, dynamic programming, and machine learning approaches'
        }
    ]
    
    dependencies = {'2': ['1']}
    
    report = generate_complexity_report(sample_tasks, dependencies)
    print(json.dumps(report, indent=2))