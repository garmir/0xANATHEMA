#!/usr/bin/env python3
"""
Intelligent Task Prioritization System

Dynamic task prioritization using:
- Complexity analysis and dependency resolution
- Machine learning-based impact prediction
- Resource optimization and autonomous execution enhancement
- Adaptive priority adjustment based on execution feedback
"""

import os
import sys
import json
import time
import math
import statistics
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum
import logging
import pickle
from collections import defaultdict, deque
import heapq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    DEFERRED = 5

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskComplexity(Enum):
    """Task complexity levels"""
    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    EXTREME = 5

@dataclass
class TaskMetrics:
    """Comprehensive task metrics for prioritization"""
    estimated_duration_hours: float = 0.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    complexity_score: float = 0.0
    impact_score: float = 0.0
    urgency_score: float = 0.0
    dependency_weight: float = 0.0
    risk_factor: float = 0.0
    autonomy_contribution: float = 0.0
    historical_success_rate: float = 1.0
    blocking_factor: float = 0.0  # How many tasks this blocks

@dataclass
class Task:
    """Enhanced task representation with prioritization data"""
    task_id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    tags: List[str] = field(default_factory=list)
    created_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    assignee: Optional[str] = None
    category: str = "general"
    
    # Calculated fields
    priority_score: float = 0.0
    urgency_decay_factor: float = 1.0
    predicted_success_probability: float = 1.0

@dataclass 
class PrioritizationConfig:
    """Configuration for prioritization algorithm"""
    # Scoring weights
    complexity_weight: float = 0.15
    impact_weight: float = 0.25
    urgency_weight: float = 0.20
    dependency_weight: float = 0.15
    autonomy_weight: float = 0.20
    risk_weight: float = 0.05
    
    # Time-based factors
    urgency_decay_rate: float = 0.1  # Per day
    deadline_boost_factor: float = 2.0
    
    # Machine learning parameters
    ml_prediction_enabled: bool = True
    feedback_learning_rate: float = 0.1
    historical_weight: float = 0.3
    
    # Resource optimization
    resource_balancing_enabled: bool = True
    parallel_execution_factor: float = 1.5
    
    # Adaptive parameters
    priority_recalculation_interval: int = 300  # seconds
    dynamic_adjustment_enabled: bool = True

class TaskDependencyGraph:
    """Directed acyclic graph for task dependencies"""
    
    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self.tasks: Dict[str, Task] = {}
    
    def add_task(self, task: Task):
        """Add task to dependency graph"""
        self.tasks[task.task_id] = task
        
        # Add dependencies
        for dep_id in task.dependencies:
            self.graph[dep_id].add(task.task_id)
            self.reverse_graph[task.task_id].add(dep_id)
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks with no unmet dependencies"""
        ready_tasks = []
        
        for task_id, task in self.tasks.items():
            if task.status in [TaskStatus.PENDING]:
                unmet_deps = [
                    dep_id for dep_id in task.dependencies
                    if self.tasks.get(dep_id, Task("")).status != TaskStatus.COMPLETED
                ]
                if not unmet_deps:
                    ready_tasks.append(task_id)
        
        return ready_tasks
    
    def calculate_blocking_factor(self, task_id: str) -> float:
        """Calculate how many tasks this task blocks"""
        blocked_count = 0
        visited = set()
        
        def count_blocked(tid: str):
            if tid in visited:
                return 0
            visited.add(tid)
            
            count = len(self.graph[tid])
            for dependent in self.graph[tid]:
                count += count_blocked(dependent)
            return count
        
        return count_blocked(task_id)
    
    def get_critical_path_length(self, task_id: str) -> float:
        """Calculate critical path length from this task"""
        memo = {}
        
        def longest_path(tid: str) -> float:
            if tid in memo:
                return memo[tid]
            
            if tid not in self.graph or not self.graph[tid]:
                memo[tid] = self.tasks.get(tid, Task("")).metrics.estimated_duration_hours
                return memo[tid]
            
            max_downstream = max(
                longest_path(dep) for dep in self.graph[tid]
            )
            
            memo[tid] = self.tasks.get(tid, Task("")).metrics.estimated_duration_hours + max_downstream
            return memo[tid]
        
        return longest_path(task_id)
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect dependency cycles"""
        white = set(self.tasks.keys())
        gray = set()
        black = set()
        cycles = []
        
        def dfs(node: str, path: List[str]):
            if node in gray:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in black:
                return
            
            white.discard(node)
            gray.add(node)
            path.append(node)
            
            for neighbor in self.graph[node]:
                dfs(neighbor, path[:])
            
            gray.discard(node)
            black.add(node)
        
        while white:
            dfs(white.pop(), [])
        
        return cycles

class TaskComplexityAnalyzer:
    """Analyze task complexity using multiple heuristics"""
    
    def __init__(self):
        self.complexity_cache = {}
        self.feature_weights = {
            'description_length': 0.1,
            'dependency_count': 0.2,
            'estimated_duration': 0.3,
            'technology_complexity': 0.2,
            'integration_complexity': 0.2
        }
    
    def analyze_complexity(self, task: Task) -> float:
        """Comprehensive complexity analysis"""
        cache_key = self._generate_cache_key(task)
        if cache_key in self.complexity_cache:
            return self.complexity_cache[cache_key]
        
        complexity_factors = {
            'description_length': self._analyze_description_complexity(task),
            'dependency_count': self._analyze_dependency_complexity(task),
            'estimated_duration': self._analyze_duration_complexity(task),
            'technology_complexity': self._analyze_technology_complexity(task),
            'integration_complexity': self._analyze_integration_complexity(task)
        }
        
        # Weighted complexity score
        complexity_score = sum(
            factor_score * self.feature_weights[factor]
            for factor, factor_score in complexity_factors.items()
        )
        
        # Normalize to 0-1 range and apply logarithmic scaling
        normalized_score = min(1.0, complexity_score)
        final_score = math.log(1 + 9 * normalized_score) / math.log(10)
        
        self.complexity_cache[cache_key] = final_score
        return final_score
    
    def _generate_cache_key(self, task: Task) -> str:
        """Generate cache key for task complexity"""
        key_data = f"{task.title}_{task.description}_{len(task.dependencies)}_{task.metrics.estimated_duration_hours}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _analyze_description_complexity(self, task: Task) -> float:
        """Analyze complexity based on description"""
        description = task.description.lower()
        
        # Complexity indicators
        complex_keywords = [
            'integrate', 'optimize', 'refactor', 'migrate', 'design',
            'architecture', 'algorithm', 'machine learning', 'ai',
            'performance', 'scalability', 'security', 'distributed',
            'concurrent', 'parallel', 'recursive', 'meta'
        ]
        
        simple_keywords = [
            'fix', 'update', 'add', 'remove', 'copy', 'move',
            'rename', 'format', 'clean', 'simple', 'basic'
        ]
        
        complex_count = sum(1 for keyword in complex_keywords if keyword in description)
        simple_count = sum(1 for keyword in simple_keywords if keyword in description)
        
        # Description length factor
        length_factor = min(1.0, len(description) / 1000)
        
        # Complexity score
        complexity_score = (complex_count * 0.3 - simple_count * 0.1 + length_factor * 0.2)
        return max(0.0, min(1.0, complexity_score))
    
    def _analyze_dependency_complexity(self, task: Task) -> float:
        """Analyze complexity based on dependencies"""
        dep_count = len(task.dependencies)
        
        # Logarithmic scaling for dependencies
        if dep_count == 0:
            return 0.1
        elif dep_count <= 2:
            return 0.3
        elif dep_count <= 5:
            return 0.6
        else:
            return min(1.0, 0.6 + 0.1 * (dep_count - 5))
    
    def _analyze_duration_complexity(self, task: Task) -> float:
        """Analyze complexity based on estimated duration"""
        duration = task.metrics.estimated_duration_hours
        
        if duration <= 1:
            return 0.2
        elif duration <= 4:
            return 0.4
        elif duration <= 8:
            return 0.6
        elif duration <= 24:
            return 0.8
        else:
            return 1.0
    
    def _analyze_technology_complexity(self, task: Task) -> float:
        """Analyze complexity based on technology requirements"""
        description = task.description.lower()
        title = task.title.lower()
        
        # Technology complexity indicators
        high_complexity_tech = [
            'kubernetes', 'microservices', 'blockchain', 'quantum',
            'neural network', 'deep learning', 'tensorflow', 'pytorch',
            'distributed system', 'consensus algorithm', 'cryptography'
        ]
        
        medium_complexity_tech = [
            'database', 'api', 'rest', 'graphql', 'docker',
            'ci/cd', 'monitoring', 'logging', 'caching'
        ]
        
        text = f"{title} {description}"
        
        high_count = sum(1 for tech in high_complexity_tech if tech in text)
        medium_count = sum(1 for tech in medium_complexity_tech if tech in text)
        
        return min(1.0, high_count * 0.4 + medium_count * 0.2)
    
    def _analyze_integration_complexity(self, task: Task) -> float:
        """Analyze complexity based on integration requirements"""
        integration_indicators = [
            'integrate', 'connect', 'sync', 'interface', 'protocol',
            'external', 'third-party', 'legacy', 'migration'
        ]
        
        text = f"{task.title} {task.description}".lower()
        integration_count = sum(1 for indicator in integration_indicators if indicator in text)
        
        return min(1.0, integration_count * 0.3)

class MachineLearningPredictor:
    """ML-based task success and impact prediction"""
    
    def __init__(self):
        self.prediction_model = None
        self.feature_history = []
        self.outcome_history = []
        self.model_accuracy = 0.7  # Starting accuracy
    
    def predict_success_probability(self, task: Task, historical_data: List[Dict]) -> float:
        """Predict task success probability using historical data"""
        if len(historical_data) < 10:
            # Insufficient data, use heuristic
            return self._heuristic_success_prediction(task)
        
        # Feature extraction
        features = self._extract_features(task)
        
        # Simple ML model simulation (would use actual ML in production)
        success_probability = self._simulate_ml_prediction(features, historical_data)
        
        return max(0.1, min(1.0, success_probability))
    
    def predict_impact_score(self, task: Task, project_context: Dict) -> float:
        """Predict task impact on project goals"""
        # Impact factors
        impact_factors = {
            'autonomy_contribution': self._calculate_autonomy_impact(task),
            'critical_path_impact': self._calculate_critical_path_impact(task, project_context),
            'resource_optimization_impact': self._calculate_resource_impact(task),
            'risk_mitigation_impact': self._calculate_risk_impact(task),
            'stakeholder_value_impact': self._calculate_stakeholder_impact(task)
        }
        
        # Weighted impact score
        weights = {
            'autonomy_contribution': 0.3,
            'critical_path_impact': 0.25,
            'resource_optimization_impact': 0.2,
            'risk_mitigation_impact': 0.15,
            'stakeholder_value_impact': 0.1
        }
        
        impact_score = sum(
            impact_factors[factor] * weights[factor]
            for factor in impact_factors
        )
        
        return min(1.0, impact_score)
    
    def _extract_features(self, task: Task) -> List[float]:
        """Extract numerical features from task"""
        return [
            task.metrics.complexity_score,
            len(task.dependencies),
            task.metrics.estimated_duration_hours,
            1.0 if task.deadline else 0.0,
            len(task.tags),
            task.metrics.risk_factor
        ]
    
    def _simulate_ml_prediction(self, features: List[float], historical_data: List[Dict]) -> float:
        """Simulate ML model prediction"""
        # Simple weighted average based on similar tasks
        similar_tasks = self._find_similar_tasks(features, historical_data)
        
        if not similar_tasks:
            return 0.7  # Default probability
        
        success_rates = [task.get('success_rate', 0.7) for task in similar_tasks]
        return statistics.mean(success_rates)
    
    def _find_similar_tasks(self, features: List[float], historical_data: List[Dict]) -> List[Dict]:
        """Find similar tasks in historical data"""
        similar_tasks = []
        
        for task_data in historical_data[-50:]:  # Consider recent tasks
            if 'features' not in task_data:
                continue
            
            # Calculate feature similarity
            similarity = self._calculate_similarity(features, task_data['features'])
            
            if similarity > 0.7:  # Similarity threshold
                similar_tasks.append(task_data)
        
        return similar_tasks[:10]  # Top 10 similar tasks
    
    def _calculate_similarity(self, features1: List[float], features2: List[float]) -> float:
        """Calculate cosine similarity between feature vectors"""
        if len(features1) != len(features2):
            return 0.0
        
        dot_product = sum(f1 * f2 for f1, f2 in zip(features1, features2))
        magnitude1 = math.sqrt(sum(f ** 2 for f in features1))
        magnitude2 = math.sqrt(sum(f ** 2 for f in features2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _heuristic_success_prediction(self, task: Task) -> float:
        """Heuristic-based success prediction when ML data insufficient"""
        base_probability = 0.8
        
        # Adjust based on complexity
        complexity_penalty = task.metrics.complexity_score * 0.3
        
        # Adjust based on dependencies
        dependency_penalty = min(0.2, len(task.dependencies) * 0.05)
        
        # Adjust based on risk
        risk_penalty = task.metrics.risk_factor * 0.25
        
        success_probability = base_probability - complexity_penalty - dependency_penalty - risk_penalty
        
        return max(0.2, min(1.0, success_probability))
    
    def _calculate_autonomy_impact(self, task: Task) -> float:
        """Calculate task's contribution to system autonomy"""
        autonomy_keywords = [
            'autonomous', 'automated', 'self-healing', 'optimization',
            'machine learning', 'ai', 'intelligent', 'adaptive'
        ]
        
        text = f"{task.title} {task.description}".lower()
        keyword_count = sum(1 for keyword in autonomy_keywords if keyword in text)
        
        return min(1.0, keyword_count * 0.3 + task.metrics.autonomy_contribution)
    
    def _calculate_critical_path_impact(self, task: Task, project_context: Dict) -> float:
        """Calculate impact on project critical path"""
        # Simplified critical path impact calculation
        blocking_factor = task.metrics.blocking_factor
        duration_factor = min(1.0, task.metrics.estimated_duration_hours / 8)
        
        return min(1.0, blocking_factor * 0.6 + duration_factor * 0.4)
    
    def _calculate_resource_impact(self, task: Task) -> float:
        """Calculate impact on resource optimization"""
        resource_keywords = ['performance', 'optimization', 'efficiency', 'memory', 'cpu']
        text = f"{task.title} {task.description}".lower()
        
        keyword_count = sum(1 for keyword in resource_keywords if keyword in text)
        return min(1.0, keyword_count * 0.25)
    
    def _calculate_risk_impact(self, task: Task) -> float:
        """Calculate impact on project risk mitigation"""
        return max(0.0, 1.0 - task.metrics.risk_factor)
    
    def _calculate_stakeholder_impact(self, task: Task) -> float:
        """Calculate impact on stakeholder value"""
        value_keywords = ['user', 'customer', 'feature', 'functionality', 'experience']
        text = f"{task.title} {task.description}".lower()
        
        keyword_count = sum(1 for keyword in value_keywords if keyword in text)
        return min(1.0, keyword_count * 0.2)

class IntelligentTaskPrioritizer:
    """Main intelligent task prioritization system"""
    
    def __init__(self, config: PrioritizationConfig = None):
        self.config = config or PrioritizationConfig()
        self.dependency_graph = TaskDependencyGraph()
        self.complexity_analyzer = TaskComplexityAnalyzer()
        self.ml_predictor = MachineLearningPredictor()
        
        # State tracking
        self.historical_data = []
        self.last_prioritization_time = 0
        self.execution_feedback = []
        
        # Priority queues for different categories
        self.priority_queues = {
            TaskPriority.CRITICAL: [],
            TaskPriority.HIGH: [],
            TaskPriority.MEDIUM: [],
            TaskPriority.LOW: [],
            TaskPriority.DEFERRED: []
        }
    
    def add_tasks(self, tasks: List[Task]):
        """Add multiple tasks to the prioritization system"""
        for task in tasks:
            self.add_task(task)
    
    def add_task(self, task: Task):
        """Add single task to the prioritization system"""
        # Analyze task complexity
        task.metrics.complexity_score = self.complexity_analyzer.analyze_complexity(task)
        
        # Add to dependency graph
        self.dependency_graph.add_task(task)
        
        # Calculate initial metrics
        self._calculate_task_metrics(task)
        
        # Calculate priority score
        self._calculate_priority_score(task)
        
        # Add to appropriate priority queue
        heapq.heappush(self.priority_queues[task.priority], (-task.priority_score, task.task_id, task))
        
        logger.info(f"Task added: {task.task_id} (Priority: {task.priority.name}, Score: {task.priority_score:.3f})")
    
    def get_next_tasks(self, count: int = 1, consider_resources: bool = True) -> List[Task]:
        """Get next highest priority tasks for execution"""
        if time.time() - self.last_prioritization_time > self.config.priority_recalculation_interval:
            self._recalculate_all_priorities()
        
        # Get ready tasks (no unmet dependencies)
        ready_task_ids = self.dependency_graph.get_ready_tasks()
        ready_tasks = [
            self.dependency_graph.tasks[task_id] 
            for task_id in ready_task_ids
            if self.dependency_graph.tasks[task_id].status == TaskStatus.PENDING
        ]
        
        if not ready_tasks:
            return []
        
        # Sort by priority score
        ready_tasks.sort(key=lambda t: t.priority_score, reverse=True)
        
        # Apply resource balancing if enabled
        if consider_resources and self.config.resource_balancing_enabled:
            ready_tasks = self._apply_resource_balancing(ready_tasks)
        
        # Apply parallel execution optimization
        selected_tasks = self._select_parallel_executable_tasks(ready_tasks, count)
        
        return selected_tasks[:count]
    
    def update_task_status(self, task_id: str, new_status: TaskStatus, execution_metrics: Dict[str, Any] = None):
        """Update task status and record execution feedback"""
        if task_id not in self.dependency_graph.tasks:
            logger.warning(f"Task {task_id} not found")
            return
        
        task = self.dependency_graph.tasks[task_id]
        old_status = task.status
        task.status = new_status
        task.last_updated = time.time()
        
        # Record execution feedback for ML learning
        if execution_metrics:
            feedback = {
                'task_id': task_id,
                'predicted_success': task.predicted_success_probability,
                'actual_success': new_status == TaskStatus.COMPLETED,
                'predicted_duration': task.metrics.estimated_duration_hours,
                'actual_duration': execution_metrics.get('actual_duration_hours', 0),
                'complexity_score': task.metrics.complexity_score,
                'priority_score': task.priority_score,
                'timestamp': time.time()
            }
            self.execution_feedback.append(feedback)
        
        # Update dependent tasks if this task is completed
        if new_status == TaskStatus.COMPLETED:
            self._update_dependent_tasks_priorities(task_id)
        
        logger.info(f"Task {task_id} status updated: {old_status.value} -> {new_status.value}")
    
    def _calculate_task_metrics(self, task: Task):
        """Calculate comprehensive task metrics"""
        # Dependency weight
        task.metrics.dependency_weight = len(task.dependencies) * 0.1
        
        # Blocking factor
        task.metrics.blocking_factor = self.dependency_graph.calculate_blocking_factor(task.task_id)
        
        # Urgency score based on deadline
        if task.deadline:
            time_to_deadline = task.deadline - time.time()
            if time_to_deadline > 0:
                days_to_deadline = time_to_deadline / (24 * 3600)
                task.metrics.urgency_score = max(0.1, 1.0 - (days_to_deadline / 30))
            else:
                task.metrics.urgency_score = 1.0  # Overdue
        else:
            # Time-based urgency decay
            task_age_days = (time.time() - task.created_time) / (24 * 3600)
            task.metrics.urgency_score = min(1.0, 0.1 + task_age_days * self.config.urgency_decay_rate)
        
        # Impact score using ML prediction
        project_context = {'total_tasks': len(self.dependency_graph.tasks)}
        task.metrics.impact_score = self.ml_predictor.predict_impact_score(task, project_context)
        
        # Success probability prediction
        task.predicted_success_probability = self.ml_predictor.predict_success_probability(
            task, self.historical_data
        )
        
        # Risk factor based on complexity and unknowns
        task.metrics.risk_factor = min(1.0, task.metrics.complexity_score * 0.7 + 
                                      (1.0 - task.predicted_success_probability) * 0.3)
        
        # Autonomy contribution
        task.metrics.autonomy_contribution = self.ml_predictor._calculate_autonomy_impact(task)
    
    def _calculate_priority_score(self, task: Task):
        """Calculate comprehensive priority score"""
        config = self.config
        metrics = task.metrics
        
        # Base score components
        complexity_component = metrics.complexity_score * config.complexity_weight
        impact_component = metrics.impact_score * config.impact_weight
        urgency_component = metrics.urgency_score * config.urgency_weight
        dependency_component = (metrics.blocking_factor + metrics.dependency_weight) * config.dependency_weight
        autonomy_component = metrics.autonomy_contribution * config.autonomy_weight
        risk_component = (1.0 - metrics.risk_factor) * config.risk_weight  # Lower risk = higher priority
        
        # Success probability adjustment
        success_adjustment = task.predicted_success_probability * 0.1
        
        # Deadline boost
        deadline_boost = 0.0
        if task.deadline and task.deadline - time.time() < 7 * 24 * 3600:  # Within a week
            deadline_boost = config.deadline_boost_factor * 0.1
        
        # Calculate final priority score
        priority_score = (
            complexity_component + impact_component + urgency_component +
            dependency_component + autonomy_component + risk_component +
            success_adjustment + deadline_boost
        )
        
        # Normalize to 0-1 range
        task.priority_score = min(1.0, max(0.0, priority_score))
        
        # Assign priority level
        if task.priority_score >= 0.8:
            task.priority = TaskPriority.CRITICAL
        elif task.priority_score >= 0.6:
            task.priority = TaskPriority.HIGH
        elif task.priority_score >= 0.4:
            task.priority = TaskPriority.MEDIUM
        elif task.priority_score >= 0.2:
            task.priority = TaskPriority.LOW
        else:
            task.priority = TaskPriority.DEFERRED
    
    def _recalculate_all_priorities(self):
        """Recalculate priorities for all pending tasks"""
        logger.info("Recalculating task priorities")
        
        # Clear priority queues
        for priority in TaskPriority:
            self.priority_queues[priority] = []
        
        # Recalculate for all pending tasks
        for task in self.dependency_graph.tasks.values():
            if task.status == TaskStatus.PENDING:
                self._calculate_task_metrics(task)
                self._calculate_priority_score(task)
                heapq.heappush(
                    self.priority_queues[task.priority], 
                    (-task.priority_score, task.task_id, task)
                )
        
        self.last_prioritization_time = time.time()
    
    def _update_dependent_tasks_priorities(self, completed_task_id: str):
        """Update priorities of tasks that depended on completed task"""
        if completed_task_id in self.dependency_graph.graph:
            for dependent_task_id in self.dependency_graph.graph[completed_task_id]:
                dependent_task = self.dependency_graph.tasks[dependent_task_id]
                if dependent_task.status == TaskStatus.PENDING:
                    # Recalculate metrics and priority
                    self._calculate_task_metrics(dependent_task)
                    self._calculate_priority_score(dependent_task)
                    
                    # Update in priority queue
                    heapq.heappush(
                        self.priority_queues[dependent_task.priority],
                        (-dependent_task.priority_score, dependent_task.task_id, dependent_task)
                    )
    
    def _apply_resource_balancing(self, tasks: List[Task]) -> List[Task]:
        """Apply resource balancing to task selection"""
        # Group tasks by resource requirements
        resource_groups = defaultdict(list)
        
        for task in tasks:
            primary_resource = self._get_primary_resource_type(task)
            resource_groups[primary_resource].append(task)
        
        # Balance across resource types
        balanced_tasks = []
        max_per_resource = max(1, len(tasks) // len(resource_groups))
        
        for resource_type, resource_tasks in resource_groups.items():
            # Sort by priority within resource group
            resource_tasks.sort(key=lambda t: t.priority_score, reverse=True)
            balanced_tasks.extend(resource_tasks[:max_per_resource])
        
        # Add remaining highest priority tasks
        remaining_tasks = [t for t in tasks if t not in balanced_tasks]
        remaining_tasks.sort(key=lambda t: t.priority_score, reverse=True)
        balanced_tasks.extend(remaining_tasks[:len(tasks) - len(balanced_tasks)])
        
        return balanced_tasks
    
    def _get_primary_resource_type(self, task: Task) -> str:
        """Determine primary resource type for task"""
        # Analyze task content for resource hints
        text = f"{task.title} {task.description}".lower()
        
        if any(keyword in text for keyword in ['cpu', 'compute', 'algorithm', 'calculation']):
            return 'cpu'
        elif any(keyword in text for keyword in ['memory', 'data', 'storage', 'cache']):
            return 'memory'
        elif any(keyword in text for keyword in ['network', 'api', 'external', 'download']):
            return 'network'
        elif any(keyword in text for keyword in ['disk', 'file', 'database', 'write']):
            return 'io'
        else:
            return 'general'
    
    def _select_parallel_executable_tasks(self, tasks: List[Task], max_count: int) -> List[Task]:
        """Select tasks that can be executed in parallel"""
        if not self.config.resource_balancing_enabled:
            return tasks[:max_count]
        
        selected = []
        resource_usage = defaultdict(float)
        
        for task in tasks:
            if len(selected) >= max_count:
                break
            
            # Check if adding this task would exceed resource limits
            primary_resource = self._get_primary_resource_type(task)
            required_capacity = task.metrics.resource_requirements.get(primary_resource, 1.0)
            
            # Simple resource limit check (would be more sophisticated in practice)
            if resource_usage[primary_resource] + required_capacity <= 4.0:  # Max 4 units per resource
                selected.append(task)
                resource_usage[primary_resource] += required_capacity
        
        return selected
    
    def generate_prioritization_report(self) -> Dict[str, Any]:
        """Generate comprehensive prioritization report"""
        report = {
            'timestamp': time.time(),
            'total_tasks': len(self.dependency_graph.tasks),
            'task_status_breakdown': defaultdict(int),
            'priority_breakdown': defaultdict(int),
            'complexity_analysis': {},
            'dependency_analysis': {},
            'prediction_accuracy': {},
            'resource_analysis': {},
            'recommendations': []
        }
        
        # Task status breakdown
        for task in self.dependency_graph.tasks.values():
            report['task_status_breakdown'][task.status.value] += 1
            report['priority_breakdown'][task.priority.name] += 1
        
        # Complexity analysis
        complexity_scores = [task.metrics.complexity_score for task in self.dependency_graph.tasks.values()]
        if complexity_scores:
            report['complexity_analysis'] = {
                'average': statistics.mean(complexity_scores),
                'median': statistics.median(complexity_scores),
                'std_dev': statistics.stdev(complexity_scores) if len(complexity_scores) > 1 else 0,
                'max': max(complexity_scores),
                'min': min(complexity_scores)
            }
        
        # Dependency analysis
        dependency_cycles = self.dependency_graph.detect_cycles()
        report['dependency_analysis'] = {
            'cycles_detected': len(dependency_cycles),
            'cycles': dependency_cycles,
            'average_dependencies_per_task': statistics.mean([
                len(task.dependencies) for task in self.dependency_graph.tasks.values()
            ]) if self.dependency_graph.tasks else 0
        }
        
        # Prediction accuracy
        if self.execution_feedback:
            successful_predictions = sum(
                1 for feedback in self.execution_feedback
                if (feedback['predicted_success'] > 0.5) == feedback['actual_success']
            )
            report['prediction_accuracy'] = {
                'overall_accuracy': successful_predictions / len(self.execution_feedback),
                'total_predictions': len(self.execution_feedback)
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check for dependency cycles
        cycles = self.dependency_graph.detect_cycles()
        if cycles:
            recommendations.append(f"Resolve {len(cycles)} dependency cycles to unblock execution")
        
        # Check for high-risk tasks
        high_risk_tasks = [
            task for task in self.dependency_graph.tasks.values()
            if task.metrics.risk_factor > 0.7 and task.status == TaskStatus.PENDING
        ]
        if high_risk_tasks:
            recommendations.append(f"Review {len(high_risk_tasks)} high-risk tasks for risk mitigation")
        
        # Check for overdue tasks
        overdue_tasks = [
            task for task in self.dependency_graph.tasks.values()
            if task.deadline and task.deadline < time.time() and task.status == TaskStatus.PENDING
        ]
        if overdue_tasks:
            recommendations.append(f"Prioritize {len(overdue_tasks)} overdue tasks")
        
        # Check for resource imbalance
        resource_distribution = defaultdict(int)
        for task in self.dependency_graph.tasks.values():
            if task.status == TaskStatus.PENDING:
                resource_distribution[self._get_primary_resource_type(task)] += 1
        
        if resource_distribution:
            max_resource = max(resource_distribution.values())
            min_resource = min(resource_distribution.values())
            if max_resource > 2 * min_resource:
                recommendations.append("Consider task redistribution to balance resource usage")
        
        return recommendations


def main():
    """Demo and testing function"""
    # Create sample tasks
    tasks = [
        Task(
            task_id="T001",
            title="Implement Authentication System",
            description="Design and implement JWT-based authentication with role-based access control",
            metrics=TaskMetrics(
                estimated_duration_hours=8.0,
                resource_requirements={'cpu': 2.0, 'memory': 1.0}
            ),
            deadline=time.time() + 7 * 24 * 3600,  # 1 week
            category="security"
        ),
        Task(
            task_id="T002", 
            title="Optimize Database Queries",
            description="Analyze and optimize slow database queries for better performance",
            dependencies=["T001"],
            metrics=TaskMetrics(
                estimated_duration_hours=4.0,
                resource_requirements={'memory': 3.0, 'io': 2.0}
            ),
            category="performance"
        ),
        Task(
            task_id="T003",
            title="Fix Critical Bug in Payment Processing",
            description="Urgent fix for payment gateway integration causing transaction failures",
            metrics=TaskMetrics(
                estimated_duration_hours=2.0,
                urgency_score=0.9,
                impact_score=0.95,
                resource_requirements={'cpu': 1.0}
            ),
            deadline=time.time() + 24 * 3600,  # 1 day
            category="bugfix"
        ),
        Task(
            task_id="T004",
            title="Implement Machine Learning Model",
            description="Develop and train recommendation system using collaborative filtering",
            dependencies=["T002"],
            metrics=TaskMetrics(
                estimated_duration_hours=16.0,
                autonomy_contribution=0.8,
                resource_requirements={'cpu': 4.0, 'memory': 8.0}
            ),
            category="ai"
        ),
        Task(
            task_id="T005",
            title="Update Documentation",
            description="Update API documentation and user guides",
            metrics=TaskMetrics(
                estimated_duration_hours=3.0,
                resource_requirements={'general': 1.0}
            ),
            category="documentation"
        )
    ]
    
    # Initialize prioritizer
    config = PrioritizationConfig(
        complexity_weight=0.2,
        impact_weight=0.3,
        urgency_weight=0.25,
        dependency_weight=0.15,
        autonomy_weight=0.1
    )
    
    prioritizer = IntelligentTaskPrioritizer(config)
    
    # Add tasks
    print("Adding tasks to prioritizer...")
    prioritizer.add_tasks(tasks)
    
    # Get prioritized tasks
    print("\nGetting next tasks to execute:")
    next_tasks = prioritizer.get_next_tasks(count=3)
    
    for i, task in enumerate(next_tasks, 1):
        print(f"\n{i}. Task: {task.title}")
        print(f"   Priority: {task.priority.name} (Score: {task.priority_score:.3f})")
        print(f"   Complexity: {task.metrics.complexity_score:.3f}")
        print(f"   Impact: {task.metrics.impact_score:.3f}")
        print(f"   Dependencies: {task.dependencies}")
        print(f"   Predicted Success: {task.predicted_success_probability:.3f}")
    
    # Simulate task completion
    if next_tasks:
        completed_task = next_tasks[0]
        print(f"\nSimulating completion of task: {completed_task.title}")
        prioritizer.update_task_status(
            completed_task.task_id,
            TaskStatus.COMPLETED,
            {'actual_duration_hours': 1.5, 'success': True}
        )
    
    # Generate report
    print("\nGenerating prioritization report...")
    report = prioritizer.generate_prioritization_report()
    
    print(f"\nPrioritization Report:")
    print(f"Total Tasks: {report['total_tasks']}")
    print(f"Status Breakdown: {dict(report['task_status_breakdown'])}")
    print(f"Priority Breakdown: {dict(report['priority_breakdown'])}")
    print(f"Average Complexity: {report['complexity_analysis'].get('average', 0):.3f}")
    print(f"Recommendations: {len(report['recommendations'])}")
    for rec in report['recommendations']:
        print(f"  - {rec}")


if __name__ == "__main__":
    main()