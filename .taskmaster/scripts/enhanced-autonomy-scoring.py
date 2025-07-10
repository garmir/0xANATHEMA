#!/usr/bin/env python3
"""
Enhanced Autonomy Scoring and Validation System

Comprehensive autonomy assessment featuring:
- Multi-dimensional autonomy scoring
- Real-time validation checkpoints
- Adaptive scoring with machine learning
- Detailed autonomy analytics and reporting
- Historical trend analysis and prediction
"""

import os
import sys
import time
import json
import math
import statistics
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import defaultdict, deque
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutonomyDimension(Enum):
    """Autonomy scoring dimensions"""
    TASK_COMPLETION = "task_completion"
    ERROR_RECOVERY = "error_recovery"
    DECISION_MAKING = "decision_making"
    LEARNING_ADAPTATION = "learning_adaptation"
    RESOURCE_MANAGEMENT = "resource_management"
    SELF_OPTIMIZATION = "self_optimization"
    PROBLEM_SOLVING = "problem_solving"
    COMMUNICATION = "communication"

class ValidationLevel(Enum):
    """Validation checkpoint levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    COMPREHENSIVE = "comprehensive"

class AutonomyTrend(Enum):
    """Autonomy score trends"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"

@dataclass
class AutonomyMetrics:
    """Detailed autonomy metrics for a single dimension"""
    dimension: AutonomyDimension
    score: float = 0.0
    confidence: float = 0.0
    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)
    trend: AutonomyTrend = AutonomyTrend.STABLE
    improvement_rate: float = 0.0
    
    # Detailed sub-metrics
    sub_metrics: Dict[str, float] = field(default_factory=dict)
    historical_scores: List[float] = field(default_factory=list)
    
    # Validation data
    validation_count: int = 0
    validation_passed: int = 0
    last_validation_time: float = 0.0

@dataclass
class AutonomyScore:
    """Comprehensive autonomy score assessment"""
    overall_score: float = 0.0
    confidence_level: float = 0.0
    assessment_timestamp: float = field(default_factory=time.time)
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Dimensional scores
    dimensional_scores: Dict[AutonomyDimension, AutonomyMetrics] = field(default_factory=dict)
    
    # Aggregated statistics
    score_variance: float = 0.0
    score_stability: float = 0.0
    improvement_velocity: float = 0.0
    
    # Validation results
    validation_checkpoints_passed: int = 0
    validation_checkpoints_total: int = 0
    validation_score: float = 0.0
    
    # Context information
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationCheckpoint:
    """Autonomy validation checkpoint definition"""
    checkpoint_id: str
    name: str
    description: str
    level: ValidationLevel
    required_dimensions: List[AutonomyDimension]
    min_score_threshold: float
    validation_function: Optional[Callable] = None
    weight: float = 1.0
    timeout_seconds: int = 60
    
    # Checkpoint history
    execution_count: int = 0
    success_count: int = 0
    last_execution_time: float = 0.0
    average_execution_time: float = 0.0

@dataclass
class ValidationResult:
    """Result of a validation checkpoint execution"""
    checkpoint_id: str
    passed: bool
    score: float
    confidence: float
    execution_time: float
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

class AutonomyDimensionEvaluator:
    """Evaluates specific autonomy dimensions"""
    
    def __init__(self):
        self.evaluation_cache = {}
        self.historical_data = defaultdict(list)
    
    def evaluate_task_completion(self, context: Dict[str, Any]) -> AutonomyMetrics:
        """Evaluate task completion autonomy"""
        metrics = AutonomyMetrics(dimension=AutonomyDimension.TASK_COMPLETION)
        
        # Get task execution data
        total_tasks = context.get('total_tasks', 0)
        completed_tasks = context.get('completed_tasks', 0)
        failed_tasks = context.get('failed_tasks', 0)
        autonomous_completions = context.get('autonomous_completions', 0)
        
        if total_tasks > 0:
            # Base completion rate
            completion_rate = completed_tasks / total_tasks
            
            # Autonomy factor (tasks completed without human intervention)
            autonomy_factor = autonomous_completions / max(completed_tasks, 1)
            
            # Quality factor (successful vs failed)
            quality_factor = completed_tasks / max(completed_tasks + failed_tasks, 1)
            
            # Combine factors
            base_score = completion_rate * 0.4 + autonomy_factor * 0.4 + quality_factor * 0.2
            
            # Apply confidence based on sample size
            confidence = min(1.0, total_tasks / 50.0)  # Full confidence at 50+ tasks
            
            metrics.score = base_score
            metrics.confidence = confidence
            metrics.sample_count = total_tasks
            
            # Sub-metrics
            metrics.sub_metrics = {
                'completion_rate': completion_rate,
                'autonomy_factor': autonomy_factor,
                'quality_factor': quality_factor,
                'avg_task_complexity': context.get('avg_task_complexity', 0.5)
            }
        
        self._update_historical_data(metrics)
        return metrics
    
    def evaluate_error_recovery(self, context: Dict[str, Any]) -> AutonomyMetrics:
        """Evaluate error recovery autonomy"""
        metrics = AutonomyMetrics(dimension=AutonomyDimension.ERROR_RECOVERY)
        
        total_errors = context.get('total_errors', 0)
        recovered_errors = context.get('recovered_errors', 0)
        autonomous_recoveries = context.get('autonomous_recoveries', 0)
        recovery_time_avg = context.get('recovery_time_avg_seconds', 0)
        
        if total_errors > 0:
            # Recovery rate
            recovery_rate = recovered_errors / total_errors
            
            # Autonomy in recovery
            autonomy_factor = autonomous_recoveries / max(recovered_errors, 1)
            
            # Speed factor (faster recovery is better)
            speed_factor = max(0.1, 1.0 - min(recovery_time_avg / 300.0, 0.9))  # 5 min baseline
            
            # Learning factor (improving over time)
            learning_factor = self._calculate_learning_factor(
                AutonomyDimension.ERROR_RECOVERY, recovery_rate
            )
            
            base_score = (recovery_rate * 0.4 + autonomy_factor * 0.3 + 
                         speed_factor * 0.2 + learning_factor * 0.1)
            
            confidence = min(1.0, total_errors / 20.0)  # Full confidence at 20+ errors
            
            metrics.score = base_score
            metrics.confidence = confidence
            metrics.sample_count = total_errors
            
            metrics.sub_metrics = {
                'recovery_rate': recovery_rate,
                'autonomy_factor': autonomy_factor,
                'speed_factor': speed_factor,
                'learning_factor': learning_factor,
                'avg_recovery_time': recovery_time_avg
            }
        
        self._update_historical_data(metrics)
        return metrics
    
    def evaluate_decision_making(self, context: Dict[str, Any]) -> AutonomyMetrics:
        """Evaluate decision making autonomy"""
        metrics = AutonomyMetrics(dimension=AutonomyDimension.DECISION_MAKING)
        
        total_decisions = context.get('total_decisions', 0)
        correct_decisions = context.get('correct_decisions', 0)
        autonomous_decisions = context.get('autonomous_decisions', 0)
        decision_confidence_avg = context.get('decision_confidence_avg', 0)
        
        if total_decisions > 0:
            # Decision accuracy
            accuracy = correct_decisions / total_decisions
            
            # Autonomy factor
            autonomy_factor = autonomous_decisions / total_decisions
            
            # Confidence factor
            confidence_factor = decision_confidence_avg
            
            # Complexity factor (harder decisions score higher)
            complexity_factor = context.get('avg_decision_complexity', 0.5)
            
            base_score = (accuracy * 0.4 + autonomy_factor * 0.3 + 
                         confidence_factor * 0.2 + complexity_factor * 0.1)
            
            confidence = min(1.0, total_decisions / 30.0)
            
            metrics.score = base_score
            metrics.confidence = confidence
            metrics.sample_count = total_decisions
            
            metrics.sub_metrics = {
                'accuracy': accuracy,
                'autonomy_factor': autonomy_factor,
                'confidence_factor': confidence_factor,
                'complexity_factor': complexity_factor
            }
        
        self._update_historical_data(metrics)
        return metrics
    
    def evaluate_learning_adaptation(self, context: Dict[str, Any]) -> AutonomyMetrics:
        """Evaluate learning and adaptation autonomy"""
        metrics = AutonomyMetrics(dimension=AutonomyDimension.LEARNING_ADAPTATION)
        
        learning_events = context.get('learning_events', 0)
        successful_adaptations = context.get('successful_adaptations', 0)
        knowledge_retention = context.get('knowledge_retention_rate', 0)
        adaptation_speed = context.get('adaptation_speed_factor', 0)
        
        if learning_events > 0:
            # Adaptation success rate
            adaptation_rate = successful_adaptations / learning_events
            
            # Knowledge retention (how well learned concepts are retained)
            retention_factor = knowledge_retention
            
            # Speed of adaptation
            speed_factor = adaptation_speed
            
            # Transfer learning (applying knowledge to new situations)
            transfer_factor = context.get('transfer_learning_success', 0.5)
            
            base_score = (adaptation_rate * 0.3 + retention_factor * 0.3 + 
                         speed_factor * 0.2 + transfer_factor * 0.2)
            
            confidence = min(1.0, learning_events / 15.0)
            
            metrics.score = base_score
            metrics.confidence = confidence
            metrics.sample_count = learning_events
            
            metrics.sub_metrics = {
                'adaptation_rate': adaptation_rate,
                'retention_factor': retention_factor,
                'speed_factor': speed_factor,
                'transfer_factor': transfer_factor
            }
        
        self._update_historical_data(metrics)
        return metrics
    
    def evaluate_resource_management(self, context: Dict[str, Any]) -> AutonomyMetrics:
        """Evaluate resource management autonomy"""
        metrics = AutonomyMetrics(dimension=AutonomyDimension.RESOURCE_MANAGEMENT)
        
        # Resource utilization efficiency
        cpu_efficiency = context.get('cpu_efficiency', 0.5)
        memory_efficiency = context.get('memory_efficiency', 0.5)
        io_efficiency = context.get('io_efficiency', 0.5)
        
        # Resource optimization actions
        optimization_actions = context.get('optimization_actions', 0)
        successful_optimizations = context.get('successful_optimizations', 0)
        
        # Resource allocation decisions
        allocation_decisions = context.get('allocation_decisions', 0)
        optimal_allocations = context.get('optimal_allocations', 0)
        
        # Calculate efficiency score
        efficiency_score = (cpu_efficiency + memory_efficiency + io_efficiency) / 3
        
        # Optimization success rate
        optimization_rate = (successful_optimizations / max(optimization_actions, 1) 
                           if optimization_actions > 0 else 0.5)
        
        # Allocation accuracy
        allocation_accuracy = (optimal_allocations / max(allocation_decisions, 1)
                             if allocation_decisions > 0 else 0.5)
        
        # Autonomous management factor
        autonomous_factor = context.get('autonomous_resource_actions', 0.5)
        
        base_score = (efficiency_score * 0.4 + optimization_rate * 0.3 + 
                     allocation_accuracy * 0.2 + autonomous_factor * 0.1)
        
        confidence = min(1.0, (optimization_actions + allocation_decisions) / 25.0)
        
        metrics.score = base_score
        metrics.confidence = confidence
        metrics.sample_count = optimization_actions + allocation_decisions
        
        metrics.sub_metrics = {
            'efficiency_score': efficiency_score,
            'optimization_rate': optimization_rate,
            'allocation_accuracy': allocation_accuracy,
            'autonomous_factor': autonomous_factor
        }
        
        self._update_historical_data(metrics)
        return metrics
    
    def evaluate_self_optimization(self, context: Dict[str, Any]) -> AutonomyMetrics:
        """Evaluate self-optimization autonomy"""
        metrics = AutonomyMetrics(dimension=AutonomyDimension.SELF_OPTIMIZATION)
        
        # Self-improvement actions
        self_improvement_actions = context.get('self_improvement_actions', 0)
        successful_improvements = context.get('successful_improvements', 0)
        
        # Performance improvements over time
        performance_improvement = context.get('performance_improvement_rate', 0)
        
        # Parameter optimization
        parameter_optimizations = context.get('parameter_optimizations', 0)
        optimal_parameters = context.get('optimal_parameters', 0)
        
        # Meta-learning capabilities
        meta_learning_score = context.get('meta_learning_score', 0.5)
        
        if self_improvement_actions > 0:
            improvement_rate = successful_improvements / self_improvement_actions
        else:
            improvement_rate = 0.5
        
        if parameter_optimizations > 0:
            parameter_accuracy = optimal_parameters / parameter_optimizations
        else:
            parameter_accuracy = 0.5
        
        base_score = (improvement_rate * 0.3 + performance_improvement * 0.3 + 
                     parameter_accuracy * 0.2 + meta_learning_score * 0.2)
        
        confidence = min(1.0, self_improvement_actions / 10.0)
        
        metrics.score = base_score
        metrics.confidence = confidence
        metrics.sample_count = self_improvement_actions
        
        metrics.sub_metrics = {
            'improvement_rate': improvement_rate,
            'performance_improvement': performance_improvement,
            'parameter_accuracy': parameter_accuracy,
            'meta_learning_score': meta_learning_score
        }
        
        self._update_historical_data(metrics)
        return metrics
    
    def evaluate_problem_solving(self, context: Dict[str, Any]) -> AutonomyMetrics:
        """Evaluate problem solving autonomy"""
        metrics = AutonomyMetrics(dimension=AutonomyDimension.PROBLEM_SOLVING)
        
        problems_encountered = context.get('problems_encountered', 0)
        problems_solved = context.get('problems_solved', 0)
        autonomous_solutions = context.get('autonomous_solutions', 0)
        novel_problems_solved = context.get('novel_problems_solved', 0)
        
        if problems_encountered > 0:
            # Problem solving rate
            solving_rate = problems_solved / problems_encountered
            
            # Autonomy in problem solving
            autonomy_factor = autonomous_solutions / max(problems_solved, 1)
            
            # Novel problem handling
            novelty_factor = novel_problems_solved / max(problems_encountered, 1)
            
            # Solution quality
            solution_quality = context.get('avg_solution_quality', 0.7)
            
            base_score = (solving_rate * 0.4 + autonomy_factor * 0.3 + 
                         novelty_factor * 0.2 + solution_quality * 0.1)
            
            confidence = min(1.0, problems_encountered / 20.0)
            
            metrics.score = base_score
            metrics.confidence = confidence
            metrics.sample_count = problems_encountered
            
            metrics.sub_metrics = {
                'solving_rate': solving_rate,
                'autonomy_factor': autonomy_factor,
                'novelty_factor': novelty_factor,
                'solution_quality': solution_quality
            }
        
        self._update_historical_data(metrics)
        return metrics
    
    def evaluate_communication(self, context: Dict[str, Any]) -> AutonomyMetrics:
        """Evaluate communication autonomy"""
        metrics = AutonomyMetrics(dimension=AutonomyDimension.COMMUNICATION)
        
        communication_events = context.get('communication_events', 0)
        successful_communications = context.get('successful_communications', 0)
        autonomous_communications = context.get('autonomous_communications', 0)
        information_clarity = context.get('information_clarity_score', 0.7)
        
        if communication_events > 0:
            # Communication success rate
            success_rate = successful_communications / communication_events
            
            # Autonomy in communication
            autonomy_factor = autonomous_communications / communication_events
            
            # Clarity and effectiveness
            clarity_factor = information_clarity
            
            # Adaptive communication (adjusting style/content)
            adaptivity_factor = context.get('communication_adaptivity', 0.5)
            
            base_score = (success_rate * 0.4 + autonomy_factor * 0.3 + 
                         clarity_factor * 0.2 + adaptivity_factor * 0.1)
            
            confidence = min(1.0, communication_events / 15.0)
            
            metrics.score = base_score
            metrics.confidence = confidence
            metrics.sample_count = communication_events
            
            metrics.sub_metrics = {
                'success_rate': success_rate,
                'autonomy_factor': autonomy_factor,
                'clarity_factor': clarity_factor,
                'adaptivity_factor': adaptivity_factor
            }
        
        self._update_historical_data(metrics)
        return metrics
    
    def _calculate_learning_factor(self, dimension: AutonomyDimension, current_score: float) -> float:
        """Calculate learning improvement factor"""
        historical_scores = self.historical_data[dimension]
        
        if len(historical_scores) < 3:
            return 0.5  # Neutral when insufficient data
        
        # Calculate trend
        recent_scores = historical_scores[-5:]  # Last 5 scores
        older_scores = historical_scores[-10:-5] if len(historical_scores) >= 10 else historical_scores[:-5]
        
        if not older_scores:
            return 0.5
        
        recent_avg = statistics.mean(recent_scores)
        older_avg = statistics.mean(older_scores)
        
        improvement = (recent_avg - older_avg) / max(older_avg, 0.01)
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, 0.5 + improvement))
    
    def _update_historical_data(self, metrics: AutonomyMetrics):
        """Update historical data for trend analysis"""
        dimension = metrics.dimension
        self.historical_data[dimension].append(metrics.score)
        
        # Keep only recent history (last 100 scores)
        if len(self.historical_data[dimension]) > 100:
            self.historical_data[dimension] = self.historical_data[dimension][-100:]
        
        # Update trend
        metrics.trend = self._calculate_trend(dimension)
        metrics.improvement_rate = self._calculate_improvement_rate(dimension)
    
    def _calculate_trend(self, dimension: AutonomyDimension) -> AutonomyTrend:
        """Calculate trend for a dimension"""
        scores = self.historical_data[dimension]
        
        if len(scores) < 5:
            return AutonomyTrend.STABLE
        
        recent_scores = scores[-5:]
        
        # Calculate linear regression slope
        x = list(range(len(recent_scores)))
        y = recent_scores
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x_squared = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x)
        
        # Calculate variance to detect volatility
        variance = statistics.variance(recent_scores)
        
        if variance > 0.05:  # High variance threshold
            return AutonomyTrend.VOLATILE
        elif slope > 0.01:  # Positive trend threshold
            return AutonomyTrend.IMPROVING
        elif slope < -0.01:  # Negative trend threshold
            return AutonomyTrend.DECLINING
        else:
            return AutonomyTrend.STABLE
    
    def _calculate_improvement_rate(self, dimension: AutonomyDimension) -> float:
        """Calculate improvement rate per unit time"""
        scores = self.historical_data[dimension]
        
        if len(scores) < 2:
            return 0.0
        
        # Simple improvement rate: (current - initial) / time_periods
        initial_score = scores[0]
        current_score = scores[-1]
        time_periods = len(scores) - 1
        
        return (current_score - initial_score) / time_periods

class AutonomyValidator:
    """Validates autonomy through comprehensive checkpoints"""
    
    def __init__(self):
        self.checkpoints = {}
        self.validation_history = []
        self._initialize_default_checkpoints()
    
    def _initialize_default_checkpoints(self):
        """Initialize default validation checkpoints"""
        
        # Basic level checkpoints
        self.register_checkpoint(ValidationCheckpoint(
            checkpoint_id="basic_task_completion",
            name="Basic Task Completion",
            description="Validate basic task completion capabilities",
            level=ValidationLevel.BASIC,
            required_dimensions=[AutonomyDimension.TASK_COMPLETION],
            min_score_threshold=0.7,
            validation_function=self._validate_basic_task_completion
        ))
        
        self.register_checkpoint(ValidationCheckpoint(
            checkpoint_id="basic_error_handling",
            name="Basic Error Handling",
            description="Validate basic error detection and recovery",
            level=ValidationLevel.BASIC,
            required_dimensions=[AutonomyDimension.ERROR_RECOVERY],
            min_score_threshold=0.6,
            validation_function=self._validate_basic_error_handling
        ))
        
        # Intermediate level checkpoints
        self.register_checkpoint(ValidationCheckpoint(
            checkpoint_id="intermediate_decision_making",
            name="Intermediate Decision Making",
            description="Validate autonomous decision making under uncertainty",
            level=ValidationLevel.INTERMEDIATE,
            required_dimensions=[AutonomyDimension.DECISION_MAKING, AutonomyDimension.PROBLEM_SOLVING],
            min_score_threshold=0.75,
            validation_function=self._validate_intermediate_decision_making
        ))
        
        self.register_checkpoint(ValidationCheckpoint(
            checkpoint_id="intermediate_learning",
            name="Intermediate Learning",
            description="Validate learning and adaptation capabilities",
            level=ValidationLevel.INTERMEDIATE,
            required_dimensions=[AutonomyDimension.LEARNING_ADAPTATION],
            min_score_threshold=0.7,
            validation_function=self._validate_intermediate_learning
        ))
        
        # Advanced level checkpoints
        self.register_checkpoint(ValidationCheckpoint(
            checkpoint_id="advanced_optimization",
            name="Advanced Self-Optimization",
            description="Validate advanced self-optimization and meta-learning",
            level=ValidationLevel.ADVANCED,
            required_dimensions=[
                AutonomyDimension.SELF_OPTIMIZATION,
                AutonomyDimension.RESOURCE_MANAGEMENT
            ],
            min_score_threshold=0.8,
            validation_function=self._validate_advanced_optimization
        ))
        
        # Comprehensive checkpoint
        self.register_checkpoint(ValidationCheckpoint(
            checkpoint_id="comprehensive_autonomy",
            name="Comprehensive Autonomy Assessment",
            description="Validate overall autonomous capabilities across all dimensions",
            level=ValidationLevel.COMPREHENSIVE,
            required_dimensions=list(AutonomyDimension),
            min_score_threshold=0.85,
            validation_function=self._validate_comprehensive_autonomy,
            weight=2.0
        ))
    
    def register_checkpoint(self, checkpoint: ValidationCheckpoint):
        """Register a validation checkpoint"""
        self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        logger.info(f"Validation checkpoint registered: {checkpoint.name}")
    
    def validate_autonomy(self, autonomy_score: AutonomyScore, 
                         level: ValidationLevel = ValidationLevel.COMPREHENSIVE) -> List[ValidationResult]:
        """Validate autonomy score against checkpoints"""
        results = []
        
        # Filter checkpoints by level
        applicable_checkpoints = [
            cp for cp in self.checkpoints.values()
            if cp.level.value <= level.value or level == ValidationLevel.COMPREHENSIVE
        ]
        
        for checkpoint in applicable_checkpoints:
            result = self._execute_checkpoint(checkpoint, autonomy_score)
            results.append(result)
            self.validation_history.append(result)
        
        logger.info(f"Autonomy validation completed: {len(results)} checkpoints executed")
        return results
    
    def _execute_checkpoint(self, checkpoint: ValidationCheckpoint, 
                           autonomy_score: AutonomyScore) -> ValidationResult:
        """Execute a single validation checkpoint"""
        start_time = time.time()
        
        try:
            # Check if required dimensions are present
            missing_dimensions = []
            for dimension in checkpoint.required_dimensions:
                if dimension not in autonomy_score.dimensional_scores:
                    missing_dimensions.append(dimension.value)
            
            if missing_dimensions:
                return ValidationResult(
                    checkpoint_id=checkpoint.checkpoint_id,
                    passed=False,
                    score=0.0,
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    error_message=f"Missing required dimensions: {missing_dimensions}"
                )
            
            # Execute validation function if provided
            if checkpoint.validation_function:
                validation_result = checkpoint.validation_function(autonomy_score, checkpoint)
            else:
                validation_result = self._default_validation(autonomy_score, checkpoint)
            
            # Update checkpoint statistics
            checkpoint.execution_count += 1
            checkpoint.last_execution_time = time.time()
            if validation_result.passed:
                checkpoint.success_count += 1
            
            execution_time = time.time() - start_time
            checkpoint.average_execution_time = (
                (checkpoint.average_execution_time * (checkpoint.execution_count - 1) + execution_time)
                / checkpoint.execution_count
            )
            
            return validation_result
            
        except Exception as e:
            return ValidationResult(
                checkpoint_id=checkpoint.checkpoint_id,
                passed=False,
                score=0.0,
                confidence=0.0,
                execution_time=time.time() - start_time,
                error_message=f"Validation error: {str(e)}"
            )
    
    def _default_validation(self, autonomy_score: AutonomyScore, 
                           checkpoint: ValidationCheckpoint) -> ValidationResult:
        """Default validation logic"""
        
        # Calculate average score across required dimensions
        dimension_scores = []
        total_confidence = 0.0
        
        for dimension in checkpoint.required_dimensions:
            metrics = autonomy_score.dimensional_scores[dimension]
            dimension_scores.append(metrics.score)
            total_confidence += metrics.confidence
        
        average_score = statistics.mean(dimension_scores) if dimension_scores else 0.0
        average_confidence = total_confidence / len(checkpoint.required_dimensions)
        
        # Check against threshold
        passed = average_score >= checkpoint.min_score_threshold
        
        return ValidationResult(
            checkpoint_id=checkpoint.checkpoint_id,
            passed=passed,
            score=average_score,
            confidence=average_confidence,
            execution_time=time.time() - time.time(),
            details={
                'threshold': checkpoint.min_score_threshold,
                'dimension_scores': {
                    dim.value: autonomy_score.dimensional_scores[dim].score 
                    for dim in checkpoint.required_dimensions
                }
            }
        )
    
    def _validate_basic_task_completion(self, autonomy_score: AutonomyScore, 
                                       checkpoint: ValidationCheckpoint) -> ValidationResult:
        """Validate basic task completion capabilities"""
        task_metrics = autonomy_score.dimensional_scores[AutonomyDimension.TASK_COMPLETION]
        
        # Check sub-metrics for more detailed validation
        completion_rate = task_metrics.sub_metrics.get('completion_rate', 0)
        autonomy_factor = task_metrics.sub_metrics.get('autonomy_factor', 0)
        quality_factor = task_metrics.sub_metrics.get('quality_factor', 0)
        
        # Specific criteria for basic task completion
        criteria_met = {
            'completion_rate_sufficient': completion_rate >= 0.7,
            'autonomy_sufficient': autonomy_factor >= 0.6,
            'quality_sufficient': quality_factor >= 0.8,
            'sample_size_adequate': task_metrics.sample_count >= 10
        }
        
        passed = all(criteria_met.values())
        confidence = task_metrics.confidence
        
        return ValidationResult(
            checkpoint_id=checkpoint.checkpoint_id,
            passed=passed,
            score=task_metrics.score,
            confidence=confidence,
            execution_time=0.1,
            details={
                'criteria_met': criteria_met,
                'sub_metrics': task_metrics.sub_metrics
            }
        )
    
    def _validate_basic_error_handling(self, autonomy_score: AutonomyScore, 
                                      checkpoint: ValidationCheckpoint) -> ValidationResult:
        """Validate basic error handling capabilities"""
        error_metrics = autonomy_score.dimensional_scores[AutonomyDimension.ERROR_RECOVERY]
        
        recovery_rate = error_metrics.sub_metrics.get('recovery_rate', 0)
        autonomy_factor = error_metrics.sub_metrics.get('autonomy_factor', 0)
        speed_factor = error_metrics.sub_metrics.get('speed_factor', 0)
        
        criteria_met = {
            'recovery_rate_sufficient': recovery_rate >= 0.6,
            'autonomous_recovery': autonomy_factor >= 0.5,
            'recovery_speed_acceptable': speed_factor >= 0.4,
            'has_error_experience': error_metrics.sample_count >= 5
        }
        
        passed = all(criteria_met.values())
        
        return ValidationResult(
            checkpoint_id=checkpoint.checkpoint_id,
            passed=passed,
            score=error_metrics.score,
            confidence=error_metrics.confidence,
            execution_time=0.1,
            details={'criteria_met': criteria_met}
        )
    
    def _validate_intermediate_decision_making(self, autonomy_score: AutonomyScore, 
                                              checkpoint: ValidationCheckpoint) -> ValidationResult:
        """Validate intermediate decision making capabilities"""
        decision_metrics = autonomy_score.dimensional_scores[AutonomyDimension.DECISION_MAKING]
        problem_metrics = autonomy_score.dimensional_scores[AutonomyDimension.PROBLEM_SOLVING]
        
        # Combined score for decision making and problem solving
        combined_score = (decision_metrics.score + problem_metrics.score) / 2
        combined_confidence = (decision_metrics.confidence + problem_metrics.confidence) / 2
        
        accuracy = decision_metrics.sub_metrics.get('accuracy', 0)
        autonomy_factor = decision_metrics.sub_metrics.get('autonomy_factor', 0)
        solving_rate = problem_metrics.sub_metrics.get('solving_rate', 0)
        
        criteria_met = {
            'decision_accuracy_good': accuracy >= 0.75,
            'autonomous_decisions': autonomy_factor >= 0.7,
            'problem_solving_capable': solving_rate >= 0.65,
            'sufficient_experience': (decision_metrics.sample_count + problem_metrics.sample_count) >= 20
        }
        
        passed = all(criteria_met.values())
        
        return ValidationResult(
            checkpoint_id=checkpoint.checkpoint_id,
            passed=passed,
            score=combined_score,
            confidence=combined_confidence,
            execution_time=0.15,
            details={'criteria_met': criteria_met}
        )
    
    def _validate_intermediate_learning(self, autonomy_score: AutonomyScore, 
                                       checkpoint: ValidationCheckpoint) -> ValidationResult:
        """Validate intermediate learning capabilities"""
        learning_metrics = autonomy_score.dimensional_scores[AutonomyDimension.LEARNING_ADAPTATION]
        
        adaptation_rate = learning_metrics.sub_metrics.get('adaptation_rate', 0)
        retention_factor = learning_metrics.sub_metrics.get('retention_factor', 0)
        transfer_factor = learning_metrics.sub_metrics.get('transfer_factor', 0)
        
        criteria_met = {
            'adaptation_successful': adaptation_rate >= 0.7,
            'knowledge_retained': retention_factor >= 0.6,
            'transfer_learning_works': transfer_factor >= 0.5,
            'learning_trend_positive': learning_metrics.trend in [AutonomyTrend.IMPROVING, AutonomyTrend.STABLE]
        }
        
        passed = all(criteria_met.values())
        
        return ValidationResult(
            checkpoint_id=checkpoint.checkpoint_id,
            passed=passed,
            score=learning_metrics.score,
            confidence=learning_metrics.confidence,
            execution_time=0.12,
            details={'criteria_met': criteria_met}
        )
    
    def _validate_advanced_optimization(self, autonomy_score: AutonomyScore, 
                                       checkpoint: ValidationCheckpoint) -> ValidationResult:
        """Validate advanced optimization capabilities"""
        optimization_metrics = autonomy_score.dimensional_scores[AutonomyDimension.SELF_OPTIMIZATION]
        resource_metrics = autonomy_score.dimensional_scores[AutonomyDimension.RESOURCE_MANAGEMENT]
        
        combined_score = (optimization_metrics.score + resource_metrics.score) / 2
        combined_confidence = (optimization_metrics.confidence + resource_metrics.confidence) / 2
        
        improvement_rate = optimization_metrics.sub_metrics.get('improvement_rate', 0)
        performance_improvement = optimization_metrics.sub_metrics.get('performance_improvement', 0)
        efficiency_score = resource_metrics.sub_metrics.get('efficiency_score', 0)
        optimization_rate = resource_metrics.sub_metrics.get('optimization_rate', 0)
        
        criteria_met = {
            'self_improvement_active': improvement_rate >= 0.8,
            'performance_improving': performance_improvement >= 0.1,
            'resource_efficiency_high': efficiency_score >= 0.75,
            'optimization_successful': optimization_rate >= 0.7,
            'meta_learning_capable': optimization_metrics.sub_metrics.get('meta_learning_score', 0) >= 0.6
        }
        
        passed = all(criteria_met.values())
        
        return ValidationResult(
            checkpoint_id=checkpoint.checkpoint_id,
            passed=passed,
            score=combined_score,
            confidence=combined_confidence,
            execution_time=0.2,
            details={'criteria_met': criteria_met}
        )
    
    def _validate_comprehensive_autonomy(self, autonomy_score: AutonomyScore, 
                                        checkpoint: ValidationCheckpoint) -> ValidationResult:
        """Validate comprehensive autonomy across all dimensions"""
        
        # Check that all dimensions meet minimum thresholds
        dimension_results = {}
        all_passed = True
        total_score = 0.0
        total_confidence = 0.0
        
        for dimension in AutonomyDimension:
            if dimension in autonomy_score.dimensional_scores:
                metrics = autonomy_score.dimensional_scores[dimension]
                dimension_passed = metrics.score >= 0.7  # Minimum for comprehensive
                dimension_results[dimension.value] = {
                    'score': metrics.score,
                    'passed': dimension_passed,
                    'confidence': metrics.confidence
                }
                
                if not dimension_passed:
                    all_passed = False
                
                total_score += metrics.score
                total_confidence += metrics.confidence
        
        average_score = total_score / len(AutonomyDimension)
        average_confidence = total_confidence / len(AutonomyDimension)
        
        # Additional comprehensive criteria
        comprehensive_criteria = {
            'all_dimensions_adequate': all_passed,
            'overall_score_high': average_score >= checkpoint.min_score_threshold,
            'score_stability_good': autonomy_score.score_stability >= 0.8,
            'improvement_trend_positive': autonomy_score.improvement_velocity >= 0.0,
            'high_confidence': average_confidence >= 0.7
        }
        
        passed = all(comprehensive_criteria.values())
        
        return ValidationResult(
            checkpoint_id=checkpoint.checkpoint_id,
            passed=passed,
            score=average_score,
            confidence=average_confidence,
            execution_time=0.3,
            details={
                'dimension_results': dimension_results,
                'comprehensive_criteria': comprehensive_criteria
            }
        )

class EnhancedAutonomyScorer:
    """Main enhanced autonomy scoring system"""
    
    def __init__(self, workspace_path: str = ".taskmaster"):
        self.workspace_path = Path(workspace_path)
        self.evaluator = AutonomyDimensionEvaluator()
        self.validator = AutonomyValidator()
        
        # Scoring history
        self.scoring_history = []
        self.validation_history = []
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 60  # seconds
        
        # Callbacks
        self.score_callbacks = []
        self.validation_callbacks = []
        
        # Initialize workspace
        self._initialize_workspace()
    
    def _initialize_workspace(self):
        """Initialize autonomy scoring workspace"""
        autonomy_dir = self.workspace_path / "autonomy"
        autonomy_dir.mkdir(parents=True, exist_ok=True)
        
        subdirs = ["scores", "validations", "reports", "checkpoints"]
        for subdir in subdirs:
            (autonomy_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"Autonomy scoring workspace initialized: {autonomy_dir}")
    
    def register_score_callback(self, callback: Callable[[AutonomyScore], None]):
        """Register callback for score updates"""
        self.score_callbacks.append(callback)
    
    def register_validation_callback(self, callback: Callable[[List[ValidationResult]], None]):
        """Register callback for validation results"""
        self.validation_callbacks.append(callback)
    
    def calculate_autonomy_score(self, context: Dict[str, Any]) -> AutonomyScore:
        """Calculate comprehensive autonomy score"""
        
        # Evaluate each dimension
        dimensional_scores = {}
        
        dimensional_scores[AutonomyDimension.TASK_COMPLETION] = (
            self.evaluator.evaluate_task_completion(context)
        )
        dimensional_scores[AutonomyDimension.ERROR_RECOVERY] = (
            self.evaluator.evaluate_error_recovery(context)
        )
        dimensional_scores[AutonomyDimension.DECISION_MAKING] = (
            self.evaluator.evaluate_decision_making(context)
        )
        dimensional_scores[AutonomyDimension.LEARNING_ADAPTATION] = (
            self.evaluator.evaluate_learning_adaptation(context)
        )
        dimensional_scores[AutonomyDimension.RESOURCE_MANAGEMENT] = (
            self.evaluator.evaluate_resource_management(context)
        )
        dimensional_scores[AutonomyDimension.SELF_OPTIMIZATION] = (
            self.evaluator.evaluate_self_optimization(context)
        )
        dimensional_scores[AutonomyDimension.PROBLEM_SOLVING] = (
            self.evaluator.evaluate_problem_solving(context)
        )
        dimensional_scores[AutonomyDimension.COMMUNICATION] = (
            self.evaluator.evaluate_communication(context)
        )
        
        # Calculate overall score
        scores = [metrics.score for metrics in dimensional_scores.values()]
        confidences = [metrics.confidence for metrics in dimensional_scores.values()]
        
        overall_score = statistics.mean(scores)
        confidence_level = statistics.mean(confidences)
        
        # Calculate additional statistics
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        score_stability = max(0.0, 1.0 - score_variance)
        
        # Calculate improvement velocity
        improvement_velocity = self._calculate_improvement_velocity(dimensional_scores)
        
        # Create autonomy score object
        autonomy_score = AutonomyScore(
            overall_score=overall_score,
            confidence_level=confidence_level,
            dimensional_scores=dimensional_scores,
            score_variance=score_variance,
            score_stability=score_stability,
            improvement_velocity=improvement_velocity,
            context=context
        )
        
        # Store in history
        self.scoring_history.append(autonomy_score)
        
        # Keep only recent history (last 100 scores)
        if len(self.scoring_history) > 100:
            self.scoring_history = self.scoring_history[-100:]
        
        # Notify callbacks
        for callback in self.score_callbacks:
            try:
                callback(autonomy_score)
            except Exception as e:
                logger.error(f"Score callback failed: {e}")
        
        logger.info(f"Autonomy score calculated: {overall_score:.3f} (confidence: {confidence_level:.3f})")
        
        return autonomy_score
    
    def validate_autonomy_score(self, autonomy_score: AutonomyScore, 
                               level: ValidationLevel = ValidationLevel.COMPREHENSIVE) -> List[ValidationResult]:
        """Validate autonomy score with comprehensive checkpoints"""
        
        validation_results = self.validator.validate_autonomy(autonomy_score, level)
        
        # Update autonomy score with validation results
        autonomy_score.validation_checkpoints_total = len(validation_results)
        autonomy_score.validation_checkpoints_passed = sum(1 for r in validation_results if r.passed)
        autonomy_score.validation_score = (
            autonomy_score.validation_checkpoints_passed / autonomy_score.validation_checkpoints_total
            if autonomy_score.validation_checkpoints_total > 0 else 0.0
        )
        
        # Store validation history
        self.validation_history.extend(validation_results)
        
        # Notify callbacks
        for callback in self.validation_callbacks:
            try:
                callback(validation_results)
            except Exception as e:
                logger.error(f"Validation callback failed: {e}")
        
        logger.info(f"Autonomy validation completed: {autonomy_score.validation_checkpoints_passed}/"
                   f"{autonomy_score.validation_checkpoints_total} checkpoints passed")
        
        return validation_results
    
    def start_real_time_monitoring(self, context_provider: Callable[[], Dict[str, Any]]):
        """Start real-time autonomy monitoring"""
        if self.monitoring_active:
            logger.warning("Real-time monitoring already active")
            return
        
        self.monitoring_active = True
        self.context_provider = context_provider
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Real-time autonomy monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop real-time autonomy monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Real-time autonomy monitoring stopped")
    
    def _monitoring_loop(self):
        """Real-time monitoring loop"""
        while self.monitoring_active:
            try:
                # Get current context
                context = self.context_provider()
                
                # Calculate autonomy score
                autonomy_score = self.calculate_autonomy_score(context)
                
                # Validate if score is concerning
                if autonomy_score.overall_score < 0.8:
                    validation_results = self.validate_autonomy_score(
                        autonomy_score, ValidationLevel.INTERMEDIATE
                    )
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Real-time monitoring error: {e}")
                time.sleep(self.monitor_interval)
    
    def _calculate_improvement_velocity(self, dimensional_scores: Dict[AutonomyDimension, AutonomyMetrics]) -> float:
        """Calculate overall improvement velocity"""
        improvement_rates = []
        
        for metrics in dimensional_scores.values():
            if metrics.improvement_rate is not None:
                improvement_rates.append(metrics.improvement_rate)
        
        if improvement_rates:
            return statistics.mean(improvement_rates)
        else:
            return 0.0
    
    def generate_autonomy_report(self, include_history: bool = True) -> Dict[str, Any]:
        """Generate comprehensive autonomy report"""
        
        if not self.scoring_history:
            return {"error": "No autonomy scores available"}
        
        latest_score = self.scoring_history[-1]
        
        report = {
            "timestamp": time.time(),
            "latest_score": asdict(latest_score),
            "dimensional_analysis": {},
            "trend_analysis": {},
            "validation_summary": {},
            "recommendations": []
        }
        
        # Dimensional analysis
        for dimension, metrics in latest_score.dimensional_scores.items():
            report["dimensional_analysis"][dimension.value] = {
                "score": metrics.score,
                "confidence": metrics.confidence,
                "trend": metrics.trend.value,
                "improvement_rate": metrics.improvement_rate,
                "sub_metrics": metrics.sub_metrics
            }
        
        # Trend analysis
        if len(self.scoring_history) >= 5:
            recent_scores = [score.overall_score for score in self.scoring_history[-10:]]
            report["trend_analysis"] = {
                "recent_average": statistics.mean(recent_scores),
                "score_variance": statistics.variance(recent_scores),
                "improvement_trend": recent_scores[-1] > recent_scores[0],
                "stability": 1.0 - min(1.0, statistics.variance(recent_scores))
            }
        
        # Validation summary
        if self.validation_history:
            recent_validations = self.validation_history[-20:]  # Last 20 validations
            passed_count = sum(1 for v in recent_validations if v.passed)
            
            report["validation_summary"] = {
                "recent_validation_count": len(recent_validations),
                "recent_pass_rate": passed_count / len(recent_validations),
                "last_validation_time": max(v.timestamp for v in recent_validations),
                "checkpoint_performance": self._analyze_checkpoint_performance()
            }
        
        # Generate recommendations
        report["recommendations"] = self._generate_autonomy_recommendations(latest_score)
        
        # Save report
        if include_history:
            report["scoring_history"] = [asdict(score) for score in self.scoring_history[-10:]]
            report["validation_history"] = [asdict(result) for result in self.validation_history[-20:]]
        
        # Save to file
        report_path = self.workspace_path / "autonomy" / "reports" / f"autonomy_report_{int(time.time())}.json"
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Autonomy report saved: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to save autonomy report: {e}")
        
        return report
    
    def _analyze_checkpoint_performance(self) -> Dict[str, Any]:
        """Analyze performance of validation checkpoints"""
        checkpoint_stats = {}
        
        for checkpoint_id, checkpoint in self.validator.checkpoints.items():
            if checkpoint.execution_count > 0:
                success_rate = checkpoint.success_count / checkpoint.execution_count
                checkpoint_stats[checkpoint_id] = {
                    "name": checkpoint.name,
                    "execution_count": checkpoint.execution_count,
                    "success_rate": success_rate,
                    "average_execution_time": checkpoint.average_execution_time,
                    "level": checkpoint.level.value
                }
        
        return checkpoint_stats
    
    def _generate_autonomy_recommendations(self, autonomy_score: AutonomyScore) -> List[str]:
        """Generate actionable recommendations for improving autonomy"""
        recommendations = []
        
        # Check each dimension for improvement opportunities
        for dimension, metrics in autonomy_score.dimensional_scores.items():
            if metrics.score < 0.7:
                recommendations.append(
                    f"Improve {dimension.value.replace('_', ' ')}: current score {metrics.score:.2f} is below 0.7"
                )
            
            if metrics.trend == AutonomyTrend.DECLINING:
                recommendations.append(
                    f"Address declining trend in {dimension.value.replace('_', ' ')}"
                )
            
            if metrics.confidence < 0.5:
                recommendations.append(
                    f"Increase sample size for {dimension.value.replace('_', ' ')} to improve confidence"
                )
        
        # Overall recommendations
        if autonomy_score.overall_score < 0.8:
            recommendations.append("Overall autonomy score is below 0.8 - focus on comprehensive improvements")
        
        if autonomy_score.score_stability < 0.7:
            recommendations.append("Autonomy scores are unstable - investigate sources of variance")
        
        if autonomy_score.improvement_velocity < 0:
            recommendations.append("Autonomy is declining overall - implement corrective measures")
        
        # Validation-based recommendations
        if autonomy_score.validation_score < 0.8:
            recommendations.append("Validation checkpoint pass rate is low - review failed checkpoints")
        
        return recommendations


def main():
    """Demo autonomy scoring system"""
    # Initialize scorer
    scorer = EnhancedAutonomyScorer()
    
    # Register callbacks
    def score_callback(score: AutonomyScore):
        print(f"📊 New autonomy score: {score.overall_score:.3f} (confidence: {score.confidence_level:.3f})")
    
    def validation_callback(results: List[ValidationResult]):
        passed = sum(1 for r in results if r.passed)
        print(f"✅ Validation: {passed}/{len(results)} checkpoints passed")
    
    scorer.register_score_callback(score_callback)
    scorer.register_validation_callback(validation_callback)
    
    # Create sample context data
    context = {
        'total_tasks': 50,
        'completed_tasks': 42,
        'failed_tasks': 3,
        'autonomous_completions': 38,
        'total_errors': 8,
        'recovered_errors': 6,
        'autonomous_recoveries': 5,
        'recovery_time_avg_seconds': 120,
        'total_decisions': 25,
        'correct_decisions': 22,
        'autonomous_decisions': 20,
        'decision_confidence_avg': 0.85,
        'learning_events': 12,
        'successful_adaptations': 10,
        'knowledge_retention_rate': 0.8,
        'adaptation_speed_factor': 0.7,
        'cpu_efficiency': 0.75,
        'memory_efficiency': 0.8,
        'io_efficiency': 0.7,
        'optimization_actions': 15,
        'successful_optimizations': 12,
        'problems_encountered': 18,
        'problems_solved': 16,
        'autonomous_solutions': 14,
        'communication_events': 20,
        'successful_communications': 18,
        'autonomous_communications': 16
    }
    
    print("🚀 Starting autonomy scoring demo...")
    
    # Calculate autonomy score
    autonomy_score = scorer.calculate_autonomy_score(context)
    
    print(f"\n📈 Autonomy Score Results:")
    print(f"Overall Score: {autonomy_score.overall_score:.3f}")
    print(f"Confidence: {autonomy_score.confidence_level:.3f}")
    print(f"Score Stability: {autonomy_score.score_stability:.3f}")
    print(f"Improvement Velocity: {autonomy_score.improvement_velocity:.3f}")
    
    print(f"\n📊 Dimensional Scores:")
    for dimension, metrics in autonomy_score.dimensional_scores.items():
        print(f"  {dimension.value.replace('_', ' ').title()}: {metrics.score:.3f} "
              f"({metrics.trend.value}, confidence: {metrics.confidence:.2f})")
    
    # Validate autonomy score
    print(f"\n🔍 Running validation checkpoints...")
    validation_results = scorer.validate_autonomy_score(autonomy_score)
    
    print(f"\n✅ Validation Results:")
    for result in validation_results:
        status = "PASS" if result.passed else "FAIL"
        checkpoint = scorer.validator.checkpoints[result.checkpoint_id]
        print(f"  {checkpoint.name}: {status} (score: {result.score:.3f})")
    
    # Generate report
    print(f"\n📄 Generating autonomy report...")
    report = scorer.generate_autonomy_report()
    
    print(f"\n📋 Report Summary:")
    print(f"Recent Average Score: {report['trend_analysis'].get('recent_average', 0):.3f}")
    print(f"Validation Pass Rate: {report['validation_summary'].get('recent_pass_rate', 0):.3f}")
    print(f"Recommendations: {len(report['recommendations'])}")
    
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n🎯 Demo completed successfully!")


if __name__ == "__main__":
    main()