#!/usr/bin/env python3
"""
Task Complexity Analyzer - Advanced computational complexity analysis for Task Master AI

This module provides sophisticated analysis of task computational requirements,
evaluating time complexity, space complexity, I/O requirements, and parallelization potential.
"""

import json
import time
import psutil
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import multiprocessing


class ComplexityClass(Enum):
    """Computational complexity classifications"""
    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n²)"
    CUBIC = "O(n³)"
    EXPONENTIAL = "O(2^n)"
    FACTORIAL = "O(n!)"


@dataclass
class TaskComplexity:
    """Data structure for task complexity metrics"""
    task_id: str
    time_complexity: ComplexityClass
    space_complexity: ComplexityClass
    io_operations: int
    parallelization_potential: float  # 0.0 to 1.0
    cpu_intensive: bool
    memory_intensive: bool
    network_dependent: bool
    file_operations: int
    estimated_runtime_seconds: float
    resource_requirements: Dict[str, Any]
    

@dataclass
class SystemResources:
    """Current system resource availability"""
    cpu_cores: int
    available_memory_gb: float
    available_disk_gb: float
    network_bandwidth_mbps: float
    cpu_usage_percent: float
    memory_usage_percent: float
    

class TaskComplexityAnalyzer:
    """
    Advanced task complexity analysis engine that evaluates computational
    requirements and resource dependencies for task optimization.
    """
    
    def __init__(self, tasks_file: str = ".taskmaster/tasks/tasks.json"):
        """Initialize the analyzer with task data"""
        self.tasks_file = tasks_file
        self.tasks_data = self._load_tasks()
        self.system_resources = self._get_system_resources()
        self.complexity_cache = {}
        
    def _load_tasks(self) -> Dict:
        """Load tasks from JSON file"""
        try:
            with open(self.tasks_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Tasks file not found: {self.tasks_file}")
            return {"tags": {"master": {"tasks": []}}}
    
    def _get_system_resources(self) -> SystemResources:
        """Get current system resource availability"""
        try:
            cpu_count = multiprocessing.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return SystemResources(
                cpu_cores=cpu_count,
                available_memory_gb=memory.available / (1024**3),
                available_disk_gb=disk.free / (1024**3),
                network_bandwidth_mbps=100.0,  # Estimated, would need network test
                cpu_usage_percent=psutil.cpu_percent(interval=1),
                memory_usage_percent=memory.percent
            )
        except Exception as e:
            # Fallback for when psutil is not available or mocked
            return SystemResources(
                cpu_cores=multiprocessing.cpu_count(),
                available_memory_gb=8.0,  # Default 8GB
                available_disk_gb=100.0,  # Default 100GB
                network_bandwidth_mbps=100.0,
                cpu_usage_percent=25.0,
                memory_usage_percent=45.0
            )
    
    def analyze_task_complexity(self, task: Dict) -> TaskComplexity:
        """
        Analyze computational complexity of a single task
        
        Args:
            task: Task dictionary from tasks.json
            
        Returns:
            TaskComplexity object with detailed analysis
        """
        task_id = str(task.get('id', ''))
        
        # Check cache first
        if task_id in self.complexity_cache:
            return self.complexity_cache[task_id]
        
        # Analyze task details and description for complexity indicators
        details = task.get('details', '')
        description = task.get('description', '')
        title = task.get('title', '')
        
        # Combine all text for analysis
        task_text = f"{title} {description} {details}".lower()
        
        # Determine time complexity based on task characteristics
        time_complexity = self._determine_time_complexity(task_text, task)
        
        # Determine space complexity
        space_complexity = self._determine_space_complexity(task_text, task)
        
        # Count I/O operations
        io_operations = self._count_io_operations(task_text)
        
        # Assess parallelization potential
        parallelization_potential = self._assess_parallelization(task_text, task)
        
        # Determine resource intensiveness
        cpu_intensive = self._is_cpu_intensive(task_text)
        memory_intensive = self._is_memory_intensive(task_text)
        network_dependent = self._is_network_dependent(task_text)
        
        # Count file operations
        file_operations = self._count_file_operations(task_text)
        
        # Estimate runtime
        estimated_runtime = self._estimate_runtime(task, time_complexity, io_operations)
        
        # Resource requirements
        resource_requirements = self._calculate_resource_requirements(
            task, time_complexity, space_complexity, cpu_intensive, memory_intensive
        )
        
        complexity = TaskComplexity(
            task_id=task_id,
            time_complexity=time_complexity,
            space_complexity=space_complexity,
            io_operations=io_operations,
            parallelization_potential=parallelization_potential,
            cpu_intensive=cpu_intensive,
            memory_intensive=memory_intensive,
            network_dependent=network_dependent,
            file_operations=file_operations,
            estimated_runtime_seconds=estimated_runtime,
            resource_requirements=resource_requirements
        )
        
        # Cache the result
        self.complexity_cache[task_id] = complexity
        
        return complexity
    
    def _determine_time_complexity(self, task_text: str, task: Dict) -> ComplexityClass:
        """Determine time complexity based on task characteristics"""
        
        # Check for exponential complexity indicators
        if any(keyword in task_text for keyword in [
            'recursive', 'permutation', 'combination', 'backtrack', 'exponential',
            'all possible', 'brute force', 'factorial'
        ]):
            return ComplexityClass.EXPONENTIAL
        
        # Check for quadratic complexity indicators  
        if any(keyword in task_text for keyword in [
            'nested loop', 'quadratic', 'o(n²)', 'double iteration',
            'matrix', 'cross-reference', 'all pairs'
        ]):
            return ComplexityClass.QUADRATIC
        
        # Check for linearithmic complexity indicators
        if any(keyword in task_text for keyword in [
            'sort', 'merge', 'divide and conquer', 'tree traversal',
            'o(n log n)', 'hierarchical', 'search tree'
        ]):
            return ComplexityClass.LINEARITHMIC
        
        # Check for linear complexity indicators
        if any(keyword in task_text for keyword in [
            'iterate', 'scan', 'process each', 'linear', 'o(n)',
            'loop through', 'traverse', 'single pass'
        ]):
            return ComplexityClass.LINEAR
        
        # Check for logarithmic complexity indicators
        if any(keyword in task_text for keyword in [
            'binary search', 'log', 'divide', 'tree depth',
            'logarithmic', 'o(log n)'
        ]):
            return ComplexityClass.LOGARITHMIC
        
        # Default to linear for most tasks
        return ComplexityClass.LINEAR
    
    def _determine_space_complexity(self, task_text: str, task: Dict) -> ComplexityClass:
        """Determine space complexity based on task characteristics"""
        
        # Check for high space complexity indicators
        if any(keyword in task_text for keyword in [
            'cache', 'store all', 'memory intensive', 'large data',
            'matrix', 'full dataset', 'in-memory'
        ]):
            return ComplexityClass.LINEAR
        
        # Check for recursive space complexity
        if any(keyword in task_text for keyword in [
            'recursive', 'call stack', 'depth'
        ]):
            return ComplexityClass.LOGARITHMIC
        
        # Default to constant space
        return ComplexityClass.CONSTANT
    
    def _count_io_operations(self, task_text: str) -> int:
        """Count estimated I/O operations based on task description"""
        io_count = 0
        
        io_indicators = [
            'file', 'read', 'write', 'save', 'load', 'database',
            'api', 'network', 'download', 'upload', 'fetch'
        ]
        
        for indicator in io_indicators:
            io_count += task_text.count(indicator)
        
        return max(1, io_count)  # Minimum 1 I/O operation
    
    def _assess_parallelization(self, task_text: str, task: Dict) -> float:
        """Assess parallelization potential (0.0 to 1.0)"""
        
        # High parallelization potential
        if any(keyword in task_text for keyword in [
            'parallel', 'concurrent', 'independent', 'batch',
            'multiple', 'async', 'simultaneous'
        ]):
            return 0.9
        
        # Medium parallelization potential
        if any(keyword in task_text for keyword in [
            'process', 'analyze', 'generate', 'compute'
        ]):
            return 0.6
        
        # Low parallelization potential
        if any(keyword in task_text for keyword in [
            'sequential', 'ordered', 'dependent', 'serial',
            'step-by-step', 'consecutive'
        ]):
            return 0.2
        
        # Default medium potential
        return 0.5
    
    def _is_cpu_intensive(self, task_text: str) -> bool:
        """Determine if task is CPU intensive"""
        cpu_indicators = [
            'compute', 'calculate', 'algorithm', 'optimization',
            'analysis', 'processing', 'complex', 'intensive'
        ]
        return any(indicator in task_text for indicator in cpu_indicators)
    
    def _is_memory_intensive(self, task_text: str) -> bool:
        """Determine if task is memory intensive"""
        memory_indicators = [
            'large', 'cache', 'store', 'memory', 'dataset',
            'matrix', 'buffer', 'in-memory'
        ]
        return any(indicator in task_text for indicator in memory_indicators)
    
    def _is_network_dependent(self, task_text: str) -> bool:
        """Determine if task depends on network operations"""
        network_indicators = [
            'api', 'network', 'download', 'upload', 'fetch',
            'request', 'remote', 'online', 'web'
        ]
        return any(indicator in task_text for indicator in network_indicators)
    
    def _count_file_operations(self, task_text: str) -> int:
        """Count estimated file operations"""
        file_indicators = ['file', 'read', 'write', 'save', 'load']
        return sum(task_text.count(indicator) for indicator in file_indicators)
    
    def _estimate_runtime(self, task: Dict, time_complexity: ComplexityClass, 
                         io_operations: int) -> float:
        """Estimate task runtime in seconds"""
        
        # Base time estimates by complexity class
        base_times = {
            ComplexityClass.CONSTANT: 1.0,
            ComplexityClass.LOGARITHMIC: 2.0,
            ComplexityClass.LINEAR: 10.0,
            ComplexityClass.LINEARITHMIC: 30.0,
            ComplexityClass.QUADRATIC: 120.0,
            ComplexityClass.CUBIC: 600.0,
            ComplexityClass.EXPONENTIAL: 3600.0,
            ComplexityClass.FACTORIAL: 7200.0
        }
        
        base_time = base_times.get(time_complexity, 10.0)
        
        # Add I/O overhead (0.1 seconds per I/O operation)
        io_overhead = io_operations * 0.1
        
        return base_time + io_overhead
    
    def _calculate_resource_requirements(self, task: Dict, time_complexity: ComplexityClass,
                                       space_complexity: ComplexityClass, cpu_intensive: bool,
                                       memory_intensive: bool) -> Dict[str, Any]:
        """Calculate resource requirements for the task"""
        
        # CPU requirements (cores needed)
        cpu_cores = 1
        if cpu_intensive:
            cpu_cores = min(4, self.system_resources.cpu_cores)
        
        # Memory requirements (GB)
        memory_gb = 0.5  # Base requirement
        if memory_intensive:
            memory_gb = 2.0
        if space_complexity in [ComplexityClass.LINEAR, ComplexityClass.QUADRATIC]:
            memory_gb *= 2
        
        # Disk space requirements (GB)
        disk_gb = 0.1  # Base requirement
        if 'file' in task.get('details', '').lower():
            disk_gb = 1.0
        
        return {
            'cpu_cores': cpu_cores,
            'memory_gb': memory_gb,
            'disk_gb': disk_gb,
            'estimated_duration_seconds': self._estimate_runtime(
                task, time_complexity, self._count_io_operations(
                    f"{task.get('title', '')} {task.get('description', '')} {task.get('details', '')}"
                )
            )
        }
    
    def analyze_all_tasks(self) -> List[TaskComplexity]:
        """Analyze complexity for all tasks in the project"""
        complexities = []
        
        # Get tasks from the master tag
        tasks = self.tasks_data.get('tags', {}).get('master', {}).get('tasks', [])
        
        for task in tasks:
            complexity = self.analyze_task_complexity(task)
            complexities.append(complexity)
        
        return complexities
    
    def generate_complexity_report(self) -> Dict[str, Any]:
        """Generate comprehensive complexity analysis report"""
        complexities = self.analyze_all_tasks()
        
        if not complexities:
            return {"error": "No tasks found for analysis"}
        
        # Aggregate statistics
        total_tasks = len(complexities)
        cpu_intensive_count = sum(1 for c in complexities if c.cpu_intensive)
        memory_intensive_count = sum(1 for c in complexities if c.memory_intensive)
        network_dependent_count = sum(1 for c in complexities if c.network_dependent)
        
        # Complexity distribution
        complexity_distribution = {}
        for complexity in complexities:
            tc = complexity.time_complexity.value
            complexity_distribution[tc] = complexity_distribution.get(tc, 0) + 1
        
        # Resource requirements summary
        total_cpu_cores = sum(c.resource_requirements.get('cpu_cores', 1) for c in complexities)
        total_memory_gb = sum(c.resource_requirements.get('memory_gb', 0.5) for c in complexities)
        total_estimated_time = sum(c.estimated_runtime_seconds for c in complexities)
        
        # Parallelization opportunities
        highly_parallelizable = [c for c in complexities if c.parallelization_potential > 0.7]
        
        report = {
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_resources": asdict(self.system_resources),
            "summary": {
                "total_tasks": total_tasks,
                "cpu_intensive_tasks": cpu_intensive_count,
                "memory_intensive_tasks": memory_intensive_count,
                "network_dependent_tasks": network_dependent_count,
                "highly_parallelizable_tasks": len(highly_parallelizable)
            },
            "complexity_distribution": complexity_distribution,
            "resource_requirements": {
                "total_cpu_cores_needed": total_cpu_cores,
                "total_memory_gb_needed": total_memory_gb,
                "estimated_total_runtime_hours": total_estimated_time / 3600
            },
            "optimization_opportunities": {
                "parallelizable_tasks": [c.task_id for c in highly_parallelizable],
                "resource_bottlenecks": self._identify_bottlenecks(complexities),
                "optimization_recommendations": self._generate_recommendations(complexities)
            },
            "detailed_analysis": [asdict(c) for c in complexities]
        }
        
        return report
    
    def _identify_bottlenecks(self, complexities: List[TaskComplexity]) -> List[str]:
        """Identify potential resource bottlenecks"""
        bottlenecks = []
        
        # Memory bottleneck
        memory_intensive_tasks = [c for c in complexities if c.memory_intensive]
        if memory_intensive_tasks:
            total_memory_needed = sum(
                c.resource_requirements.get('memory_gb', 0.5) 
                for c in memory_intensive_tasks
            )
            if total_memory_needed > self.system_resources.available_memory_gb:
                bottlenecks.append(f"Memory bottleneck: {total_memory_needed:.1f}GB needed, "
                                 f"{self.system_resources.available_memory_gb:.1f}GB available")
        
        # CPU bottleneck
        cpu_intensive_tasks = [c for c in complexities if c.cpu_intensive]
        if len(cpu_intensive_tasks) > self.system_resources.cpu_cores:
            bottlenecks.append(f"CPU bottleneck: {len(cpu_intensive_tasks)} CPU-intensive tasks, "
                             f"{self.system_resources.cpu_cores} cores available")
        
        # High complexity tasks
        high_complexity_tasks = [
            c for c in complexities 
            if c.time_complexity in [ComplexityClass.QUADRATIC, ComplexityClass.EXPONENTIAL]
        ]
        if high_complexity_tasks:
            bottlenecks.append(f"Complexity bottleneck: {len(high_complexity_tasks)} "
                             f"high-complexity tasks may dominate execution time")
        
        return bottlenecks
    
    def _generate_recommendations(self, complexities: List[TaskComplexity]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Parallelization recommendations
        parallelizable = [c for c in complexities if c.parallelization_potential > 0.6]
        if parallelizable:
            recommendations.append(
                f"Consider parallel execution for {len(parallelizable)} tasks "
                f"with high parallelization potential"
            )
        
        # Resource management recommendations
        memory_intensive = [c for c in complexities if c.memory_intensive]
        if memory_intensive:
            recommendations.append(
                f"Schedule {len(memory_intensive)} memory-intensive tasks "
                f"during low system memory usage periods"
            )
        
        # Complexity optimization recommendations
        high_complexity = [
            c for c in complexities 
            if c.time_complexity in [ComplexityClass.QUADRATIC, ComplexityClass.EXPONENTIAL]
        ]
        if high_complexity:
            recommendations.append(
                f"Review {len(high_complexity)} high-complexity tasks for "
                f"algorithmic optimization opportunities"
            )
        
        return recommendations
    
    def save_report(self, output_file: str = ".taskmaster/reports/complexity-analysis.json"):
        """Save complexity analysis report to file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        report = self.generate_complexity_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Complexity analysis report saved to: {output_file}")
        return output_file


def main():
    """Main function for command-line usage"""
    if len(sys.argv) > 1:
        tasks_file = sys.argv[1]
    else:
        tasks_file = ".taskmaster/tasks/tasks.json"
    
    analyzer = TaskComplexityAnalyzer(tasks_file)
    
    print("Analyzing task complexity...")
    report_file = analyzer.save_report()
    
    # Print summary
    report = analyzer.generate_complexity_report()
    print("\n" + "="*80)
    print("TASK COMPLEXITY ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total tasks analyzed: {report['summary']['total_tasks']}")
    print(f"CPU-intensive tasks: {report['summary']['cpu_intensive_tasks']}")
    print(f"Memory-intensive tasks: {report['summary']['memory_intensive_tasks']}")
    print(f"Network-dependent tasks: {report['summary']['network_dependent_tasks']}")
    print(f"Highly parallelizable tasks: {report['summary']['highly_parallelizable_tasks']}")
    print(f"\nTotal estimated runtime: {report['resource_requirements']['estimated_total_runtime_hours']:.2f} hours")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()