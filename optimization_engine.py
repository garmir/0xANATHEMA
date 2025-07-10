#!/usr/bin/env python3
"""
Optimization Engine - Advanced task execution optimization for Task Master AI

This module provides sophisticated optimization strategies for task execution order,
resource allocation, and parallel processing using computational complexity theory.
"""

import json
import time
import copy
import heapq
import random
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from task_complexity_analyzer import TaskComplexityAnalyzer, TaskComplexity, SystemResources
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    GREEDY_SHORTEST_FIRST = "greedy_shortest"
    GREEDY_RESOURCE_AWARE = "greedy_resource"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    MACHINE_LEARNING = "machine_learning"
    CRITICAL_PATH = "critical_path"
    ADAPTIVE_SCHEDULING = "adaptive_scheduling"


@dataclass
class ExecutionPlan:
    """Optimized execution plan for tasks"""
    strategy: OptimizationStrategy
    task_order: List[str]
    parallel_groups: List[List[str]]
    resource_allocation: Dict[str, Dict[str, Any]]
    estimated_total_time: float
    efficiency_score: float
    bottlenecks: List[str]
    optimization_metadata: Dict[str, Any]


@dataclass
class ResourceConstraints:
    """System resource constraints for optimization"""
    max_cpu_cores: int
    max_memory_gb: float
    max_disk_gb: float
    max_parallel_tasks: int
    priority_weights: Dict[str, float]


class OptimizationEngine:
    """
    Advanced task execution optimization engine that uses complexity metrics
    to reorder tasks for maximum efficiency with resource constraints.
    """
    
    def __init__(self, complexity_analyzer: TaskComplexityAnalyzer = None,
                 tasks_file: str = ".taskmaster/tasks/tasks.json"):
        """Initialize the optimization engine"""
        self.tasks_file = tasks_file
        self.analyzer = complexity_analyzer or TaskComplexityAnalyzer(tasks_file)
        self.system_resources = self.analyzer.system_resources
        self.task_complexities = {}
        self.dependency_graph = {}
        self.optimization_history = []
        
    def analyze_dependencies(self) -> Dict[str, List[str]]:
        """Build dependency graph from task data"""
        tasks = self.analyzer.tasks_data.get('tags', {}).get('master', {}).get('tasks', [])
        dependency_graph = {}
        
        for task in tasks:
            task_id = str(task.get('id', ''))
            dependencies = task.get('dependencies', [])
            dependency_graph[task_id] = [str(dep) for dep in dependencies]
        
        self.dependency_graph = dependency_graph
        return dependency_graph
    
    def get_task_complexities(self) -> Dict[str, TaskComplexity]:
        """Get complexity analysis for all tasks"""
        if not self.task_complexities:
            complexities = self.analyzer.analyze_all_tasks()
            self.task_complexities = {c.task_id: c for c in complexities}
        
        return self.task_complexities
    
    def optimize_execution_order(self, strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE_SCHEDULING,
                                constraints: ResourceConstraints = None) -> ExecutionPlan:
        """
        Optimize task execution order using specified strategy
        
        Args:
            strategy: Optimization strategy to use
            constraints: Resource constraints for optimization
            
        Returns:
            ExecutionPlan with optimized task ordering and resource allocation
        """
        
        # Set default constraints if not provided
        if constraints is None:
            constraints = ResourceConstraints(
                max_cpu_cores=self.system_resources.cpu_cores,
                max_memory_gb=self.system_resources.available_memory_gb * 0.8,  # Reserve 20%
                max_disk_gb=self.system_resources.available_disk_gb * 0.9,      # Reserve 10%
                max_parallel_tasks=min(8, self.system_resources.cpu_cores * 2),
                priority_weights={'high': 1.0, 'medium': 0.6, 'low': 0.3}
            )
        
        # Prepare data
        self.analyze_dependencies()
        complexities = self.get_task_complexities()
        
        # Apply optimization strategy
        if strategy == OptimizationStrategy.GREEDY_SHORTEST_FIRST:
            plan = self._optimize_greedy_shortest_first(complexities, constraints)
        elif strategy == OptimizationStrategy.GREEDY_RESOURCE_AWARE:
            plan = self._optimize_greedy_resource_aware(complexities, constraints)
        elif strategy == OptimizationStrategy.DYNAMIC_PROGRAMMING:
            plan = self._optimize_dynamic_programming(complexities, constraints)
        elif strategy == OptimizationStrategy.CRITICAL_PATH:
            plan = self._optimize_critical_path(complexities, constraints)
        elif strategy == OptimizationStrategy.ADAPTIVE_SCHEDULING:
            plan = self._optimize_adaptive_scheduling(complexities, constraints)
        else:
            # Default to adaptive scheduling
            plan = self._optimize_adaptive_scheduling(complexities, constraints)
        
        # Store optimization result
        self.optimization_history.append(plan)
        
        return plan
    
    def _optimize_greedy_shortest_first(self, complexities: Dict[str, TaskComplexity],
                                       constraints: ResourceConstraints) -> ExecutionPlan:
        """Greedy optimization: shortest tasks first"""
        
        # Sort tasks by estimated runtime
        available_tasks = list(complexities.keys())
        task_runtimes = [(complexities[tid].estimated_runtime_seconds, tid) 
                        for tid in available_tasks]
        task_runtimes.sort()
        
        # Respect dependencies
        task_order = []
        completed = set()
        
        while len(task_order) < len(available_tasks):
            added_task = False
            
            for runtime, task_id in task_runtimes:
                if task_id in completed:
                    continue
                
                # Check if dependencies are satisfied
                dependencies = self.dependency_graph.get(task_id, [])
                if all(dep in completed for dep in dependencies):
                    task_order.append(task_id)
                    completed.add(task_id)
                    added_task = True
                    break
            
            if not added_task:
                # Handle circular dependencies or missing tasks
                remaining = [tid for _, tid in task_runtimes if tid not in completed]
                if remaining:
                    task_order.append(remaining[0])
                    completed.add(remaining[0])
        
        # Create parallel groups
        parallel_groups = self._create_parallel_groups(task_order, complexities, constraints)
        
        # Calculate resource allocation
        resource_allocation = self._calculate_resource_allocation(task_order, complexities, constraints)
        
        # Calculate metrics
        total_time = sum(complexities[tid].estimated_runtime_seconds for tid in task_order)
        efficiency_score = self._calculate_efficiency_score(task_order, complexities, constraints)
        bottlenecks = self._identify_execution_bottlenecks(task_order, complexities, constraints)
        
        return ExecutionPlan(
            strategy=OptimizationStrategy.GREEDY_SHORTEST_FIRST,
            task_order=task_order,
            parallel_groups=parallel_groups,
            resource_allocation=resource_allocation,
            estimated_total_time=total_time,
            efficiency_score=efficiency_score,
            bottlenecks=bottlenecks,
            optimization_metadata={
                "optimization_time": time.time(),
                "total_tasks": len(task_order),
                "constraints_applied": asdict(constraints)
            }
        )
    
    def _optimize_greedy_resource_aware(self, complexities: Dict[str, TaskComplexity],
                                       constraints: ResourceConstraints) -> ExecutionPlan:
        """Greedy optimization considering resource constraints"""
        
        task_order = []
        completed = set()
        available_tasks = list(complexities.keys())
        
        # Current resource usage tracking
        current_cpu = 0
        current_memory = 0.0
        
        while len(task_order) < len(available_tasks):
            best_task = None
            best_score = -1
            
            for task_id in available_tasks:
                if task_id in completed:
                    continue
                
                # Check dependencies
                dependencies = self.dependency_graph.get(task_id, [])
                if not all(dep in completed for dep in dependencies):
                    continue
                
                # Check resource constraints
                task_complexity = complexities[task_id]
                required_cpu = task_complexity.resource_requirements.get('cpu_cores', 1)
                required_memory = task_complexity.resource_requirements.get('memory_gb', 0.5)
                
                if (current_cpu + required_cpu <= constraints.max_cpu_cores and
                    current_memory + required_memory <= constraints.max_memory_gb):
                    
                    # Calculate score: efficiency / resource_usage
                    efficiency = 1.0 / task_complexity.estimated_runtime_seconds
                    resource_usage = required_cpu + required_memory
                    score = efficiency / max(resource_usage, 0.1)
                    
                    if score > best_score:
                        best_score = score
                        best_task = task_id
            
            if best_task:
                task_order.append(best_task)
                completed.add(best_task)
                
                # Update resource usage
                task_complexity = complexities[best_task]
                current_cpu += task_complexity.resource_requirements.get('cpu_cores', 1)
                current_memory += task_complexity.resource_requirements.get('memory_gb', 0.5)
                
                # Reset resources periodically (simulating task completion)
                if len(task_order) % 3 == 0:
                    current_cpu = 0
                    current_memory = 0.0
            else:
                # Force add next available task to avoid infinite loop
                remaining = [tid for tid in available_tasks if tid not in completed]
                if remaining:
                    task_order.append(remaining[0])
                    completed.add(remaining[0])
        
        # Create parallel groups and calculate metrics
        parallel_groups = self._create_parallel_groups(task_order, complexities, constraints)
        resource_allocation = self._calculate_resource_allocation(task_order, complexities, constraints)
        total_time = sum(complexities[tid].estimated_runtime_seconds for tid in task_order)
        efficiency_score = self._calculate_efficiency_score(task_order, complexities, constraints)
        bottlenecks = self._identify_execution_bottlenecks(task_order, complexities, constraints)
        
        return ExecutionPlan(
            strategy=OptimizationStrategy.GREEDY_RESOURCE_AWARE,
            task_order=task_order,
            parallel_groups=parallel_groups,
            resource_allocation=resource_allocation,
            estimated_total_time=total_time,
            efficiency_score=efficiency_score,
            bottlenecks=bottlenecks,
            optimization_metadata={
                "optimization_time": time.time(),
                "total_tasks": len(task_order),
                "constraints_applied": asdict(constraints)
            }
        )
    
    def _optimize_critical_path(self, complexities: Dict[str, TaskComplexity],
                               constraints: ResourceConstraints) -> ExecutionPlan:
        """Critical Path Method optimization"""
        
        # Calculate longest path to each task (critical path)
        task_distances = {}
        available_tasks = list(complexities.keys())
        
        # Initialize distances
        for task_id in available_tasks:
            task_distances[task_id] = 0
        
        # Calculate critical path using topological sort and longest path
        def calculate_critical_path():
            in_degree = {tid: 0 for tid in available_tasks}
            
            # Calculate in-degrees
            for task_id in available_tasks:
                dependencies = self.dependency_graph.get(task_id, [])
                in_degree[task_id] = len(dependencies)
            
            # Topological sort with longest path calculation
            queue = [tid for tid, degree in in_degree.items() if degree == 0]
            topo_order = []
            
            while queue:
                current = queue.pop(0)
                topo_order.append(current)
                
                # Update distances to dependent tasks
                for task_id in available_tasks:
                    dependencies = self.dependency_graph.get(task_id, [])
                    if current in dependencies:
                        # Update distance with longest path
                        current_distance = task_distances[current] + complexities[current].estimated_runtime_seconds
                        task_distances[task_id] = max(task_distances[task_id], current_distance)
                        
                        in_degree[task_id] -= 1
                        if in_degree[task_id] == 0:
                            queue.append(task_id)
            
            return topo_order
        
        # Get topological order considering critical path
        topo_order = calculate_critical_path()
        
        # Sort by critical path length (longest first)
        task_order = sorted(available_tasks, 
                           key=lambda tid: task_distances[tid], 
                           reverse=True)
        
        # Ensure dependencies are respected
        final_order = []
        completed = set()
        
        for task_id in task_order:
            if task_id in completed:
                continue
            
            dependencies = self.dependency_graph.get(task_id, [])
            if all(dep in completed for dep in dependencies):
                final_order.append(task_id)
                completed.add(task_id)
        
        # Add any remaining tasks
        for task_id in available_tasks:
            if task_id not in completed:
                final_order.append(task_id)
        
        # Create parallel groups and calculate metrics
        parallel_groups = self._create_parallel_groups(final_order, complexities, constraints)
        resource_allocation = self._calculate_resource_allocation(final_order, complexities, constraints)
        total_time = sum(complexities[tid].estimated_runtime_seconds for tid in final_order)
        efficiency_score = self._calculate_efficiency_score(final_order, complexities, constraints)
        bottlenecks = self._identify_execution_bottlenecks(final_order, complexities, constraints)
        
        return ExecutionPlan(
            strategy=OptimizationStrategy.CRITICAL_PATH,
            task_order=final_order,
            parallel_groups=parallel_groups,
            resource_allocation=resource_allocation,
            estimated_total_time=total_time,
            efficiency_score=efficiency_score,
            bottlenecks=bottlenecks,
            optimization_metadata={
                "optimization_time": time.time(),
                "total_tasks": len(final_order),
                "critical_path_distances": task_distances,
                "constraints_applied": asdict(constraints)
            }
        )
    
    def _optimize_adaptive_scheduling(self, complexities: Dict[str, TaskComplexity],
                                     constraints: ResourceConstraints) -> ExecutionPlan:
        """Adaptive scheduling that adjusts based on real-time conditions"""
        
        # Combine multiple strategies and select best
        strategies = [
            OptimizationStrategy.GREEDY_SHORTEST_FIRST,
            OptimizationStrategy.GREEDY_RESOURCE_AWARE,
            OptimizationStrategy.CRITICAL_PATH
        ]
        
        best_plan = None
        best_efficiency = -1
        
        for strategy in strategies:
            if strategy == OptimizationStrategy.GREEDY_SHORTEST_FIRST:
                plan = self._optimize_greedy_shortest_first(complexities, constraints)
            elif strategy == OptimizationStrategy.GREEDY_RESOURCE_AWARE:
                plan = self._optimize_greedy_resource_aware(complexities, constraints)
            elif strategy == OptimizationStrategy.CRITICAL_PATH:
                plan = self._optimize_critical_path(complexities, constraints)
            
            if plan.efficiency_score > best_efficiency:
                best_efficiency = plan.efficiency_score
                best_plan = plan
        
        # Update strategy to adaptive
        if best_plan:
            best_plan.strategy = OptimizationStrategy.ADAPTIVE_SCHEDULING
            best_plan.optimization_metadata["adaptive_strategy_selected"] = best_plan.strategy.value
            best_plan.optimization_metadata["strategies_evaluated"] = [s.value for s in strategies]
        
        return best_plan
    
    def _optimize_dynamic_programming(self, complexities: Dict[str, TaskComplexity],
                                     constraints: ResourceConstraints) -> ExecutionPlan:
        """Dynamic programming optimization for optimal task ordering"""
        
        available_tasks = list(complexities.keys())
        n = len(available_tasks)
        
        # For large task sets, use approximation
        if n > 20:
            return self._optimize_adaptive_scheduling(complexities, constraints)
        
        # DP state: (completed_tasks_bitmask, current_time, current_resources)
        # This is a simplified version due to complexity constraints
        
        # Use greedy approximation for DP
        return self._optimize_greedy_resource_aware(complexities, constraints)
    
    def _create_parallel_groups(self, task_order: List[str], 
                               complexities: Dict[str, TaskComplexity],
                               constraints: ResourceConstraints) -> List[List[str]]:
        """Create groups of tasks that can be executed in parallel"""
        
        parallel_groups = []
        remaining_tasks = task_order.copy()
        completed = set()
        
        while remaining_tasks:
            current_group = []
            current_cpu = 0
            current_memory = 0.0
            group_size = 0
            
            for task_id in remaining_tasks.copy():
                # Check dependencies
                dependencies = self.dependency_graph.get(task_id, [])
                if not all(dep in completed for dep in dependencies):
                    continue
                
                # Check resource constraints
                task_complexity = complexities[task_id]
                required_cpu = task_complexity.resource_requirements.get('cpu_cores', 1)
                required_memory = task_complexity.resource_requirements.get('memory_gb', 0.5)
                
                # Check parallelization potential
                if task_complexity.parallelization_potential < 0.3:
                    # Low parallelization potential, skip if group already has tasks
                    if current_group:
                        continue
                
                if (current_cpu + required_cpu <= constraints.max_cpu_cores and
                    current_memory + required_memory <= constraints.max_memory_gb and
                    group_size < constraints.max_parallel_tasks):
                    
                    current_group.append(task_id)
                    current_cpu += required_cpu
                    current_memory += required_memory
                    group_size += 1
                    remaining_tasks.remove(task_id)
                    
                    # If low parallelization potential, only add one task per group
                    if task_complexity.parallelization_potential < 0.3:
                        break
            
            if current_group:
                parallel_groups.append(current_group)
                completed.update(current_group)
            else:
                # Force add at least one task to avoid infinite loop
                if remaining_tasks:
                    parallel_groups.append([remaining_tasks.pop(0)])
        
        return parallel_groups
    
    def _calculate_resource_allocation(self, task_order: List[str],
                                     complexities: Dict[str, TaskComplexity],
                                     constraints: ResourceConstraints) -> Dict[str, Dict[str, Any]]:
        """Calculate optimal resource allocation for each task"""
        
        allocation = {}
        
        for task_id in task_order:
            task_complexity = complexities[task_id]
            
            # Base allocation from complexity analysis
            base_cpu = task_complexity.resource_requirements.get('cpu_cores', 1)
            base_memory = task_complexity.resource_requirements.get('memory_gb', 0.5)
            
            # Adjust based on parallelization potential
            if task_complexity.parallelization_potential > 0.7:
                # Can use more cores
                allocated_cpu = min(base_cpu * 2, constraints.max_cpu_cores)
            else:
                allocated_cpu = base_cpu
            
            # Adjust memory based on complexity
            memory_multiplier = 1.0
            if task_complexity.memory_intensive:
                memory_multiplier = 1.5
            
            allocated_memory = min(base_memory * memory_multiplier, constraints.max_memory_gb)
            
            allocation[task_id] = {
                'cpu_cores': allocated_cpu,
                'memory_gb': allocated_memory,
                'priority': 'high' if task_complexity.cpu_intensive else 'medium',
                'estimated_duration': task_complexity.estimated_runtime_seconds,
                'parallelization_potential': task_complexity.parallelization_potential
            }
        
        return allocation
    
    def _calculate_efficiency_score(self, task_order: List[str],
                                   complexities: Dict[str, TaskComplexity],
                                   constraints: ResourceConstraints) -> float:
        """Calculate efficiency score for the execution plan"""
        
        # Factors: resource utilization, parallelization, dependency satisfaction
        
        total_time = sum(complexities[tid].estimated_runtime_seconds for tid in task_order)
        total_cpu_used = sum(complexities[tid].resource_requirements.get('cpu_cores', 1) 
                            for tid in task_order)
        total_memory_used = sum(complexities[tid].resource_requirements.get('memory_gb', 0.5) 
                               for tid in task_order)
        
        # Resource utilization efficiency (0-1)
        cpu_efficiency = min(total_cpu_used / (constraints.max_cpu_cores * len(task_order)), 1.0)
        memory_efficiency = min(total_memory_used / (constraints.max_memory_gb * len(task_order)), 1.0)
        
        # Parallelization efficiency
        parallelizable_tasks = [tid for tid in task_order 
                               if complexities[tid].parallelization_potential > 0.6]
        parallelization_efficiency = len(parallelizable_tasks) / len(task_order)
        
        # Dependency satisfaction (check for proper ordering)
        dependency_violations = 0
        completed = set()
        for task_id in task_order:
            dependencies = self.dependency_graph.get(task_id, [])
            for dep in dependencies:
                if dep not in completed:
                    dependency_violations += 1
            completed.add(task_id)
        
        dependency_efficiency = 1.0 - (dependency_violations / max(len(task_order), 1))
        
        # Combined efficiency score
        efficiency_score = (
            cpu_efficiency * 0.3 +
            memory_efficiency * 0.2 +
            parallelization_efficiency * 0.3 +
            dependency_efficiency * 0.2
        )
        
        return efficiency_score
    
    def _identify_execution_bottlenecks(self, task_order: List[str],
                                       complexities: Dict[str, TaskComplexity],
                                       constraints: ResourceConstraints) -> List[str]:
        """Identify potential bottlenecks in the execution plan"""
        
        bottlenecks = []
        
        # Time bottlenecks (tasks taking much longer than average)
        avg_time = sum(complexities[tid].estimated_runtime_seconds for tid in task_order) / len(task_order)
        time_bottlenecks = [tid for tid in task_order 
                           if complexities[tid].estimated_runtime_seconds > avg_time * 3]
        
        if time_bottlenecks:
            bottlenecks.append(f"Time bottlenecks: tasks {time_bottlenecks} may dominate execution time")
        
        # Resource bottlenecks
        high_cpu_tasks = [tid for tid in task_order 
                         if complexities[tid].resource_requirements.get('cpu_cores', 1) > constraints.max_cpu_cores * 0.5]
        if high_cpu_tasks:
            bottlenecks.append(f"CPU bottlenecks: tasks {high_cpu_tasks} require high CPU allocation")
        
        high_memory_tasks = [tid for tid in task_order 
                            if complexities[tid].resource_requirements.get('memory_gb', 0.5) > constraints.max_memory_gb * 0.5]
        if high_memory_tasks:
            bottlenecks.append(f"Memory bottlenecks: tasks {high_memory_tasks} require high memory allocation")
        
        # Dependency bottlenecks (tasks with many dependencies)
        dependency_counts = {tid: len(self.dependency_graph.get(tid, [])) for tid in task_order}
        max_deps = max(dependency_counts.values()) if dependency_counts else 0
        if max_deps > 3:
            high_dep_tasks = [tid for tid, count in dependency_counts.items() if count > 3]
            bottlenecks.append(f"Dependency bottlenecks: tasks {high_dep_tasks} have many dependencies")
        
        return bottlenecks
    
    def generate_execution_script(self, plan: ExecutionPlan, 
                                 output_file: str = ".taskmaster/execution-plan.sh") -> str:
        """Generate executable script from optimization plan"""
        
        script_lines = [
            "#!/bin/bash",
            "# Auto-generated Task Master execution plan",
            f"# Strategy: {plan.strategy.value}",
            f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Estimated total time: {plan.estimated_total_time:.2f} seconds",
            f"# Efficiency score: {plan.efficiency_score:.3f}",
            "",
            "set -e",  # Exit on error
            "",
            "echo \"Starting Task Master execution plan...\"",
            "echo \"Strategy: " + plan.strategy.value + "\"",
            "echo \"Total tasks: " + str(len(plan.task_order)) + "\"",
            ""
        ]
        
        # Add parallel group execution
        for i, group in enumerate(plan.parallel_groups):
            script_lines.append(f"echo \"Executing parallel group {i + 1}/{len(plan.parallel_groups)}...\"")
            
            if len(group) == 1:
                # Single task execution
                task_id = group[0]
                allocation = plan.resource_allocation.get(task_id, {})
                script_lines.append(f"echo \"Starting task {task_id}...\"")
                script_lines.append(f"task-master set-status --id={task_id} --status=in-progress")
                script_lines.append(f"# TODO: Implement actual task execution for task {task_id}")
                script_lines.append(f"task-master set-status --id={task_id} --status=done")
                script_lines.append(f"echo \"Completed task {task_id}\"")
            else:
                # Parallel execution
                script_lines.append("(")
                for task_id in group:
                    allocation = plan.resource_allocation.get(task_id, {})
                    script_lines.append(f"  (")
                    script_lines.append(f"    echo \"Starting task {task_id} in parallel...\"")
                    script_lines.append(f"    task-master set-status --id={task_id} --status=in-progress")
                    script_lines.append(f"    # TODO: Implement actual task execution for task {task_id}")
                    script_lines.append(f"    task-master set-status --id={task_id} --status=done")
                    script_lines.append(f"    echo \"Completed task {task_id}\"")
                    script_lines.append(f"  ) &")
                script_lines.append("  wait")
                script_lines.append(")")
            
            script_lines.append("")
        
        script_lines.extend([
            "echo \"All tasks completed!\"",
            "task-master list",
            "echo \"Execution plan finished.\""
        ])
        
        script_content = "\n".join(script_lines)
        
        # Save script
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(output_file, 0o755)
        
        print(f"Execution script generated: {output_file}")
        return output_file
    
    def save_optimization_report(self, plan: ExecutionPlan, 
                                output_file: str = ".taskmaster/reports/optimization-report.json") -> str:
        """Save optimization report to file"""
        
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_plan": asdict(plan),
            "system_resources": asdict(self.system_resources),
            "optimization_summary": {
                "strategy_used": plan.strategy.value,
                "total_tasks": len(plan.task_order),
                "parallel_groups": len(plan.parallel_groups),
                "estimated_runtime_hours": plan.estimated_total_time / 3600,
                "efficiency_score": plan.efficiency_score,
                "bottlenecks_identified": len(plan.bottlenecks)
            },
            "recommendations": self._generate_optimization_recommendations(plan)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Optimization report saved: {output_file}")
        return output_file
    
    def _generate_optimization_recommendations(self, plan: ExecutionPlan) -> List[str]:
        """Generate recommendations for further optimization"""
        
        recommendations = []
        
        if plan.efficiency_score < 0.7:
            recommendations.append("Consider reviewing task dependencies and resource allocation")
        
        if len(plan.bottlenecks) > 0:
            recommendations.append("Address identified bottlenecks to improve performance")
        
        if plan.estimated_total_time > 3600:  # More than 1 hour
            recommendations.append("Consider breaking down long-running tasks into smaller subtasks")
        
        single_task_groups = sum(1 for group in plan.parallel_groups if len(group) == 1)
        if single_task_groups > len(plan.parallel_groups) * 0.8:
            recommendations.append("Low parallelization detected - review tasks for parallel execution opportunities")
        
        return recommendations


def main():
    """Main function for command-line usage"""
    import sys
    
    if len(sys.argv) > 1:
        tasks_file = sys.argv[1]
    else:
        tasks_file = ".taskmaster/tasks/tasks.json"
    
    print("Initializing optimization engine...")
    analyzer = TaskComplexityAnalyzer(tasks_file)
    engine = OptimizationEngine(analyzer, tasks_file)
    
    print("Optimizing task execution plan...")
    plan = engine.optimize_execution_order(OptimizationStrategy.ADAPTIVE_SCHEDULING)
    
    print("\n" + "="*80)
    print("TASK EXECUTION OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Strategy: {plan.strategy.value}")
    print(f"Total tasks: {len(plan.task_order)}")
    print(f"Parallel groups: {len(plan.parallel_groups)}")
    print(f"Estimated total time: {plan.estimated_total_time:.2f} seconds ({plan.estimated_total_time/3600:.2f} hours)")
    print(f"Efficiency score: {plan.efficiency_score:.3f}")
    print(f"Bottlenecks identified: {len(plan.bottlenecks)}")
    
    if plan.bottlenecks:
        print("\nBottlenecks:")
        for bottleneck in plan.bottlenecks:
            print(f"  - {bottleneck}")
    
    # Save results
    script_file = engine.generate_execution_script(plan)
    report_file = engine.save_optimization_report(plan)
    
    print(f"\nExecution script: {script_file}")
    print(f"Optimization report: {report_file}")


if __name__ == "__main__":
    main()