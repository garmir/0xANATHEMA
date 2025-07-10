#!/usr/bin/env python3
"""
Self-Improving Architecture with Recursive Meta-Learning and NAS
Atomic Task 50.1: Define Target Tasks and Performance Metrics

This module establishes the foundational metrics and task definitions for 
a self-improving AI architecture system that continuously optimizes itself
through neural architecture search and meta-learning.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from pathlib import Path
import statistics
import numpy as np


class TaskType(Enum):
    """Types of tasks for self-improving architecture"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OPTIMIZATION = "optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    UNSUPERVISED_LEARNING = "unsupervised_learning"
    META_LEARNING = "meta_learning"
    ARCHITECTURE_SEARCH = "architecture_search"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    TRANSFER_LEARNING = "transfer_learning"
    CONTINUAL_LEARNING = "continual_learning"


class PerformanceMetricType(Enum):
    """Types of performance metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    LOSS = "loss"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ENERGY_CONSUMPTION = "energy_consumption"
    CONVERGENCE_TIME = "convergence_time"
    ADAPTATION_SPEED = "adaptation_speed"
    GENERALIZATION_ERROR = "generalization_error"
    ROBUSTNESS_SCORE = "robustness_score"
    ARCHITECTURAL_COMPLEXITY = "architectural_complexity"
    SEARCH_EFFICIENCY = "search_efficiency"
    META_LEARNING_SPEED = "meta_learning_speed"


class OptimizationObjective(Enum):
    """Optimization objectives for self-improvement"""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    STABILIZE = "stabilize"
    BALANCE = "balance"


@dataclass
class PerformanceMetric:
    """Definition of a performance metric"""
    name: str
    metric_type: PerformanceMetricType
    objective: OptimizationObjective
    weight: float = 1.0
    threshold: Optional[float] = None
    baseline: Optional[float] = None
    target: Optional[float] = None
    measurement_function: Optional[Callable] = None
    aggregation_method: str = "mean"  # mean, median, max, min, std
    time_window: Optional[timedelta] = None
    description: str = ""
    
    def __post_init__(self):
        if self.time_window is None:
            self.time_window = timedelta(hours=24)
    
    def is_improvement(self, old_value: float, new_value: float) -> bool:
        """Check if new value represents improvement"""
        if self.objective == OptimizationObjective.MAXIMIZE:
            return new_value > old_value
        elif self.objective == OptimizationObjective.MINIMIZE:
            return new_value < old_value
        elif self.objective == OptimizationObjective.STABILIZE:
            # Improvement means getting closer to target
            if self.target is not None:
                return abs(new_value - self.target) < abs(old_value - self.target)
            return False
        else:  # BALANCE
            # Custom logic for balanced objectives
            return True
    
    def calculate_improvement_score(self, old_value: float, new_value: float) -> float:
        """Calculate improvement score (0-1, higher is better)"""
        if self.objective == OptimizationObjective.MAXIMIZE:
            if old_value == 0:
                return 1.0 if new_value > 0 else 0.0
            return max(0, (new_value - old_value) / abs(old_value))
        elif self.objective == OptimizationObjective.MINIMIZE:
            if old_value == 0:
                return 1.0 if new_value < 0 else 0.0
            return max(0, (old_value - new_value) / abs(old_value))
        elif self.objective == OptimizationObjective.STABILIZE:
            if self.target is not None:
                old_distance = abs(old_value - self.target)
                new_distance = abs(new_value - self.target)
                if old_distance == 0:
                    return 1.0 if new_distance == 0 else 0.0
                return max(0, (old_distance - new_distance) / old_distance)
        return 0.0


@dataclass
class TaskDefinition:
    """Definition of a target task for the self-improving architecture"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    task_type: TaskType = TaskType.CLASSIFICATION
    description: str = ""
    input_spec: Dict[str, Any] = field(default_factory=dict)
    output_spec: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: List[PerformanceMetric] = field(default_factory=list)
    evaluation_protocol: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0
    difficulty_level: int = 1  # 1-10 scale
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    expected_training_time: Optional[timedelta] = None
    adaptation_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def add_metric(self, metric: PerformanceMetric):
        """Add performance metric to task"""
        self.performance_metrics.append(metric)
    
    def get_primary_metric(self) -> Optional[PerformanceMetric]:
        """Get primary performance metric (highest weight)"""
        if not self.performance_metrics:
            return None
        return max(self.performance_metrics, key=lambda m: m.weight)
    
    def calculate_task_score(self, metric_values: Dict[str, float]) -> float:
        """Calculate overall task performance score"""
        if not self.performance_metrics:
            return 0.0
        
        weighted_scores = []
        total_weight = 0
        
        for metric in self.performance_metrics:
            if metric.name in metric_values:
                value = metric_values[metric.name]
                
                # Normalize value to 0-1 scale
                if metric.baseline is not None and metric.target is not None:
                    if metric.objective == OptimizationObjective.MAXIMIZE:
                        normalized = (value - metric.baseline) / (metric.target - metric.baseline)
                    else:  # MINIMIZE
                        normalized = (metric.baseline - value) / (metric.baseline - metric.target)
                    normalized = max(0, min(1, normalized))
                else:
                    normalized = value  # Assume already normalized
                
                weighted_scores.append(normalized * metric.weight)
                total_weight += metric.weight
        
        if total_weight == 0:
            return 0.0
        
        return sum(weighted_scores) / total_weight


@dataclass
class ArchitectureObjective:
    """High-level objectives for the self-improving architecture"""
    name: str
    description: str
    target_tasks: List[str]  # Task IDs
    optimization_horizon: timedelta = field(default_factory=lambda: timedelta(days=30))
    resource_budget: Dict[str, float] = field(default_factory=dict)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    meta_objectives: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.meta_objectives:
            self.meta_objectives = [
                "continuous_improvement",
                "adaptation_speed",
                "resource_efficiency",
                "robustness",
                "generalization"
            ]


class TaskRegistry:
    """Registry for managing target tasks and their definitions"""
    
    def __init__(self):
        self.tasks: Dict[str, TaskDefinition] = {}
        self.task_groups: Dict[str, Set[str]] = {}
        self.objectives: Dict[str, ArchitectureObjective] = {}
        self.logger = logging.getLogger("TaskRegistry")
        
        # Initialize with default tasks
        self._initialize_default_tasks()
    
    def _initialize_default_tasks(self):
        """Initialize registry with common ML tasks"""
        
        # 1. Image Classification Task
        img_classification = TaskDefinition(
            name="Image Classification",
            task_type=TaskType.CLASSIFICATION,
            description="Multi-class image classification on CIFAR-10/ImageNet",
            input_spec={"shape": [224, 224, 3], "type": "image"},
            output_spec={"classes": 1000, "type": "categorical"},
            dataset_info={"name": "ImageNet", "size": 1000000, "classes": 1000},
            constraints={"max_params": 50000000, "max_latency_ms": 100},
            priority=1.0,
            difficulty_level=7,
            resource_requirements={"memory_gb": 8, "compute_hours": 24}
        )
        
        # Add metrics for image classification
        img_classification.add_metric(PerformanceMetric(
            name="top1_accuracy",
            metric_type=PerformanceMetricType.ACCURACY,
            objective=OptimizationObjective.MAXIMIZE,
            weight=1.0,
            baseline=0.7,
            target=0.85,
            description="Top-1 classification accuracy"
        ))
        
        img_classification.add_metric(PerformanceMetric(
            name="inference_latency",
            metric_type=PerformanceMetricType.LATENCY,
            objective=OptimizationObjective.MINIMIZE,
            weight=0.5,
            baseline=200.0,
            target=50.0,
            description="Inference latency in milliseconds"
        ))
        
        img_classification.add_metric(PerformanceMetric(
            name="model_size",
            metric_type=PerformanceMetricType.MEMORY_USAGE,
            objective=OptimizationObjective.MINIMIZE,
            weight=0.3,
            baseline=100.0,
            target=20.0,
            description="Model size in MB"
        ))
        
        self.register_task(img_classification)
        
        # 2. Natural Language Understanding
        nlu_task = TaskDefinition(
            name="Natural Language Understanding",
            task_type=TaskType.CLASSIFICATION,
            description="Sentiment analysis and text classification",
            input_spec={"max_length": 512, "type": "text"},
            output_spec={"classes": 3, "type": "categorical"},
            dataset_info={"name": "GLUE", "size": 100000, "classes": 3},
            constraints={"max_params": 110000000, "max_latency_ms": 200},
            priority=0.8,
            difficulty_level=8,
            resource_requirements={"memory_gb": 16, "compute_hours": 48}
        )
        
        nlu_task.add_metric(PerformanceMetric(
            name="f1_score",
            metric_type=PerformanceMetricType.F1_SCORE,
            objective=OptimizationObjective.MAXIMIZE,
            weight=1.0,
            baseline=0.8,
            target=0.92,
            description="Macro F1 score"
        ))
        
        nlu_task.add_metric(PerformanceMetric(
            name="inference_speed",
            metric_type=PerformanceMetricType.THROUGHPUT,
            objective=OptimizationObjective.MAXIMIZE,
            weight=0.4,
            baseline=100.0,
            target=500.0,
            description="Throughput in samples per second"
        ))
        
        self.register_task(nlu_task)
        
        # 3. Reinforcement Learning Task
        rl_task = TaskDefinition(
            name="Reinforcement Learning Control",
            task_type=TaskType.REINFORCEMENT_LEARNING,
            description="Continuous control in OpenAI Gym environments",
            input_spec={"observation_space": "continuous", "action_space": "continuous"},
            output_spec={"action_dim": 4, "type": "continuous"},
            dataset_info={"environment": "MuJoCo", "episodes": 10000},
            constraints={"max_episodes": 50000, "max_wall_time_hours": 12},
            priority=0.9,
            difficulty_level=9,
            resource_requirements={"memory_gb": 4, "compute_hours": 72}
        )
        
        rl_task.add_metric(PerformanceMetric(
            name="episode_reward",
            metric_type=PerformanceMetricType.ACCURACY,  # Using as proxy for reward
            objective=OptimizationObjective.MAXIMIZE,
            weight=1.0,
            baseline=100.0,
            target=2000.0,
            description="Average episode reward"
        ))
        
        rl_task.add_metric(PerformanceMetric(
            name="sample_efficiency",
            metric_type=PerformanceMetricType.CONVERGENCE_TIME,
            objective=OptimizationObjective.MINIMIZE,
            weight=0.6,
            baseline=1000000.0,
            target=100000.0,
            description="Samples needed to reach threshold performance"
        ))
        
        self.register_task(rl_task)
        
        # 4. Meta-Learning Task
        meta_task = TaskDefinition(
            name="Few-Shot Learning",
            task_type=TaskType.META_LEARNING,
            description="Few-shot classification across multiple domains",
            input_spec={"support_shots": 5, "query_shots": 15, "ways": 5},
            output_spec={"classes": 5, "type": "categorical"},
            dataset_info={"name": "Omniglot", "tasks": 1000, "classes_per_task": 5},
            constraints={"max_support_shots": 10, "max_adaptation_steps": 5},
            priority=1.2,
            difficulty_level=10,
            resource_requirements={"memory_gb": 12, "compute_hours": 96}
        )
        
        meta_task.add_metric(PerformanceMetric(
            name="adaptation_accuracy",
            metric_type=PerformanceMetricType.ACCURACY,
            objective=OptimizationObjective.MAXIMIZE,
            weight=1.0,
            baseline=0.6,
            target=0.9,
            description="Accuracy after few-shot adaptation"
        ))
        
        meta_task.add_metric(PerformanceMetric(
            name="adaptation_speed",
            metric_type=PerformanceMetricType.ADAPTATION_SPEED,
            objective=OptimizationObjective.MAXIMIZE,
            weight=0.7,
            baseline=0.1,
            target=0.8,
            description="Rate of improvement per adaptation step"
        ))
        
        self.register_task(meta_task)
        
        # 5. Architecture Search Task
        nas_task = TaskDefinition(
            name="Neural Architecture Search",
            task_type=TaskType.ARCHITECTURE_SEARCH,
            description="Automated architecture discovery for image classification",
            input_spec={"search_space": "MobileNet-based", "operations": 20},
            output_spec={"architecture": "genotype", "type": "discrete"},
            dataset_info={"proxy_dataset": "CIFAR-10", "full_dataset": "ImageNet"},
            constraints={"max_search_time_hours": 24, "max_architecture_params": 10000000},
            priority=1.5,
            difficulty_level=10,
            resource_requirements={"memory_gb": 32, "compute_hours": 200}
        )
        
        nas_task.add_metric(PerformanceMetric(
            name="discovered_accuracy",
            metric_type=PerformanceMetricType.ACCURACY,
            objective=OptimizationObjective.MAXIMIZE,
            weight=1.0,
            baseline=0.75,
            target=0.88,
            description="Accuracy of discovered architecture"
        ))
        
        nas_task.add_metric(PerformanceMetric(
            name="search_efficiency",
            metric_type=PerformanceMetricType.SEARCH_EFFICIENCY,
            objective=OptimizationObjective.MAXIMIZE,
            weight=0.8,
            baseline=0.3,
            target=0.9,
            description="Ratio of good architectures found to total searched"
        ))
        
        self.register_task(nas_task)
        
        # Create task groups
        self.task_groups["core_ml"] = {img_classification.id, nlu_task.id}
        self.task_groups["advanced_ml"] = {rl_task.id, meta_task.id}
        self.task_groups["meta_optimization"] = {nas_task.id}
        
        # Define architecture objectives
        self.objectives["general_purpose"] = ArchitectureObjective(
            name="General Purpose AI System",
            description="Optimize for performance across diverse ML tasks",
            target_tasks=list(self.tasks.keys()),
            optimization_horizon=timedelta(days=30),
            resource_budget={"compute_hours": 1000, "memory_gb": 64},
            success_criteria={"average_improvement": 0.15, "tasks_improved": 0.8}
        )
        
        self.objectives["efficient_mobile"] = ArchitectureObjective(
            name="Efficient Mobile Deployment",
            description="Optimize for mobile/edge deployment constraints",
            target_tasks=[img_classification.id, nlu_task.id],
            optimization_horizon=timedelta(days=14),
            resource_budget={"compute_hours": 200, "memory_gb": 8},
            success_criteria={"latency_improvement": 0.3, "size_reduction": 0.5}
        )
    
    def register_task(self, task: TaskDefinition):
        """Register a new task"""
        self.tasks[task.id] = task
        self.logger.info(f"Registered task: {task.name} ({task.id})")
    
    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_tasks_by_type(self, task_type: TaskType) -> List[TaskDefinition]:
        """Get all tasks of specific type"""
        return [task for task in self.tasks.values() if task.task_type == task_type]
    
    def get_tasks_by_group(self, group_name: str) -> List[TaskDefinition]:
        """Get tasks in a specific group"""
        task_ids = self.task_groups.get(group_name, set())
        return [self.tasks[tid] for tid in task_ids if tid in self.tasks]
    
    def get_high_priority_tasks(self, threshold: float = 1.0) -> List[TaskDefinition]:
        """Get tasks with priority above threshold"""
        return [task for task in self.tasks.values() if task.priority >= threshold]
    
    def calculate_task_portfolio_score(self, metric_values: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate performance scores for all tasks"""
        scores = {}
        for task_id, task in self.tasks.items():
            if task_id in metric_values:
                scores[task_id] = task.calculate_task_score(metric_values[task_id])
            else:
                scores[task_id] = 0.0
        return scores
    
    def get_improvement_opportunities(self, scores: Dict[str, float], 
                                    min_improvement_potential: float = 0.2) -> List[TaskDefinition]:
        """Identify tasks with high improvement potential"""
        opportunities = []
        for task_id, score in scores.items():
            if task_id in self.tasks and score < (1.0 - min_improvement_potential):
                opportunities.append(self.tasks[task_id])
        
        # Sort by priority and improvement potential
        return sorted(opportunities, 
                     key=lambda t: (t.priority, 1.0 - scores.get(t.id, 0.0)), 
                     reverse=True)
    
    def export_task_definitions(self) -> Dict[str, Any]:
        """Export all task definitions"""
        return {
            "tasks": {tid: asdict(task) for tid, task in self.tasks.items()},
            "task_groups": {name: list(tasks) for name, tasks in self.task_groups.items()},
            "objectives": {name: asdict(obj) for name, obj in self.objectives.items()}
        }
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered tasks"""
        task_types = {}
        difficulty_distribution = {}
        priority_distribution = {}
        
        for task in self.tasks.values():
            # Task type distribution
            task_type = task.task_type.value
            task_types[task_type] = task_types.get(task_type, 0) + 1
            
            # Difficulty distribution
            difficulty = task.difficulty_level
            difficulty_distribution[difficulty] = difficulty_distribution.get(difficulty, 0) + 1
            
            # Priority distribution
            priority_bucket = f"{task.priority:.1f}"
            priority_distribution[priority_bucket] = priority_distribution.get(priority_bucket, 0) + 1
        
        total_metrics = sum(len(task.performance_metrics) for task in self.tasks.values())
        avg_metrics_per_task = total_metrics / len(self.tasks) if self.tasks else 0
        
        return {
            "total_tasks": len(self.tasks),
            "task_groups": len(self.task_groups),
            "objectives": len(self.objectives),
            "task_type_distribution": task_types,
            "difficulty_distribution": difficulty_distribution,
            "priority_distribution": priority_distribution,
            "total_metrics": total_metrics,
            "avg_metrics_per_task": avg_metrics_per_task,
            "metric_types": list(set(m.metric_type.value for task in self.tasks.values() 
                                   for m in task.performance_metrics))
        }


# Export key classes
__all__ = [
    "TaskType", "PerformanceMetricType", "OptimizationObjective", 
    "PerformanceMetric", "TaskDefinition", "ArchitectureObjective", "TaskRegistry"
]


if __name__ == "__main__":
    # Demo usage
    def demo():
        logging.basicConfig(level=logging.INFO)
        
        # Create task registry
        registry = TaskRegistry()
        
        print("🎯 Self-Improving Architecture: Task Definitions & Performance Metrics")
        print("=" * 80)
        
        # Show task statistics
        stats = registry.get_task_statistics()
        print(f"\n📊 Task Registry Statistics:")
        print(f"  Total Tasks: {stats['total_tasks']}")
        print(f"  Task Groups: {stats['task_groups']}")
        print(f"  Architecture Objectives: {stats['objectives']}")
        print(f"  Average Metrics per Task: {stats['avg_metrics_per_task']:.1f}")
        print(f"  Task Types: {list(stats['task_type_distribution'].keys())}")
        print(f"  Metric Types: {len(stats['metric_types'])} unique types")
        
        # Show high priority tasks
        high_priority = registry.get_high_priority_tasks(threshold=1.0)
        print(f"\n🔥 High Priority Tasks ({len(high_priority)}):")
        for task in high_priority[:3]:  # Show top 3
            primary_metric = task.get_primary_metric()
            print(f"  • {task.name} (Priority: {task.priority})")
            print(f"    Type: {task.task_type.value}, Difficulty: {task.difficulty_level}/10")
            if primary_metric:
                print(f"    Primary Metric: {primary_metric.name} ({primary_metric.objective.value})")
            print(f"    Metrics: {len(task.performance_metrics)} defined")
        
        # Simulate performance evaluation
        print(f"\n📈 Performance Evaluation Simulation:")
        simulated_metrics = {}
        for task_id, task in registry.tasks.items():
            task_metrics = {}
            for metric in task.performance_metrics:
                # Simulate current performance between baseline and target
                if metric.baseline is not None and metric.target is not None:
                    progress = 0.6  # 60% of way to target
                    current_value = metric.baseline + (metric.target - metric.baseline) * progress
                    task_metrics[metric.name] = current_value
            simulated_metrics[task_id] = task_metrics
        
        # Calculate task scores
        scores = registry.calculate_task_portfolio_score(simulated_metrics)
        print(f"  Current Task Scores:")
        for task_id, score in scores.items():
            task_name = registry.tasks[task_id].name
            print(f"    {task_name}: {score:.3f}")
        
        # Identify improvement opportunities
        opportunities = registry.get_improvement_opportunities(scores, 0.2)
        print(f"\n🎯 Top Improvement Opportunities ({len(opportunities)}):")
        for task in opportunities[:3]:
            score = scores.get(task.id, 0.0)
            potential = 1.0 - score
            print(f"  • {task.name}")
            print(f"    Current Score: {score:.3f}, Potential: {potential:.3f}")
            print(f"    Priority: {task.priority}, Difficulty: {task.difficulty_level}")
        
        # Show architecture objectives
        print(f"\n🏗️ Architecture Objectives:")
        for name, objective in registry.objectives.items():
            print(f"  • {objective.name}")
            print(f"    Target Tasks: {len(objective.target_tasks)}")
            print(f"    Optimization Horizon: {objective.optimization_horizon.days} days")
            print(f"    Success Criteria: {objective.success_criteria}")
        
        print(f"\n✅ Task definitions and metrics framework ready for self-improving architecture!")
    
    # Run demo
    demo()