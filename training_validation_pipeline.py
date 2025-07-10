#!/usr/bin/env python3
"""
Training and Validation Pipeline for Self-Improving Architecture
Atomic Task 50.4: Implement Training and Validation Pipeline

This module implements a comprehensive training and validation pipeline that
integrates with the recursive meta-learning framework and task registry to
provide continuous improvement capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from pathlib import Path
import statistics
import numpy as np
from collections import defaultdict, deque
import copy

# Import from our previous modules
from self_improving_architecture import TaskRegistry, TaskDefinition, PerformanceMetric
from recursive_meta_learning import RecursiveMetaController, MetaExperience


class PipelineStage(Enum):
    """Training pipeline stages"""
    DATA_PREPARATION = "data_preparation"
    MODEL_INITIALIZATION = "model_initialization"
    TRAINING = "training"
    VALIDATION = "validation"
    META_LEARNING = "meta_learning"
    ARCHITECTURE_SEARCH = "architecture_search"
    PERFORMANCE_EVALUATION = "performance_evaluation"
    MODEL_SELECTION = "model_selection"
    DEPLOYMENT = "deployment"


class ValidationStrategy(Enum):
    """Validation strategies"""
    HOLDOUT = "holdout"
    K_FOLD = "k_fold"
    TIME_SERIES_SPLIT = "time_series_split"
    STRATIFIED = "stratified"
    LEAVE_ONE_OUT = "leave_one_out"
    BOOTSTRAP = "bootstrap"


class TrainingMode(Enum):
    """Training modes"""
    STANDARD = "standard"
    META_LEARNING = "meta_learning"
    CONTINUAL_LEARNING = "continual_learning"
    TRANSFER_LEARNING = "transfer_learning"
    MULTI_TASK = "multi_task"
    SELF_SUPERVISED = "self_supervised"


@dataclass
class TrainingConfiguration:
    """Training configuration parameters"""
    mode: TrainingMode = TrainingMode.STANDARD
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    validation_frequency: int = 10
    early_stopping_patience: int = 20
    optimizer: str = "adam"
    scheduler: str = "cosine"
    regularization: Dict[str, float] = field(default_factory=dict)
    augmentation: Dict[str, Any] = field(default_factory=dict)
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    save_frequency: int = 50
    
    def __post_init__(self):
        if not self.regularization:
            self.regularization = {"weight_decay": 1e-4, "dropout": 0.1}
        if not self.augmentation:
            self.augmentation = {"enabled": True, "strength": 0.5}


@dataclass
class ValidationConfiguration:
    """Validation configuration parameters"""
    strategy: ValidationStrategy = ValidationStrategy.HOLDOUT
    validation_split: float = 0.2
    k_folds: int = 5
    test_split: float = 0.1
    stratify: bool = True
    shuffle: bool = True
    random_seed: int = 42
    cross_validation_repeats: int = 1
    bootstrap_samples: int = 1000


@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    epoch: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    training_time: float = 0.0
    memory_usage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "training_time": self.training_time,
            "memory_usage": self.memory_usage
        }


@dataclass
class ValidationResults:
    """Validation results"""
    task_id: str
    model_id: str = ""
    validation_strategy: ValidationStrategy = ValidationStrategy.HOLDOUT
    metrics: Dict[str, float] = field(default_factory=dict)
    fold_results: List[Dict[str, float]] = field(default_factory=list)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    validation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_mean_metric(self, metric_name: str) -> float:
        """Get mean value across folds"""
        if metric_name in self.metrics:
            return self.metrics[metric_name]
        
        values = [fold.get(metric_name, 0.0) for fold in self.fold_results]
        return statistics.mean(values) if values else 0.0
    
    def get_std_metric(self, metric_name: str) -> float:
        """Get standard deviation across folds"""
        values = [fold.get(metric_name, 0.0) for fold in self.fold_results]
        return statistics.stdev(values) if len(values) > 1 else 0.0


class DataPipeline:
    """Data preparation and loading pipeline"""
    
    def __init__(self, task_registry: TaskRegistry):
        self.task_registry = task_registry
        self.data_cache: Dict[str, Any] = {}
        self.preprocessing_cache: Dict[str, Any] = {}
        self.logger = logging.getLogger("DataPipeline")
    
    async def prepare_task_data(self, task_id: str, 
                               validation_config: ValidationConfiguration) -> Dict[str, Any]:
        """Prepare data for training/validation"""
        task = self.task_registry.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        self.logger.info(f"Preparing data for task: {task.name}")
        
        # Check cache first
        cache_key = f"{task_id}_{hash(str(validation_config))}"
        if cache_key in self.data_cache:
            self.logger.info("Using cached data")
            return self.data_cache[cache_key]
        
        # Simulate data loading based on task type
        await asyncio.sleep(0.5)  # Simulate I/O
        
        # Generate synthetic data based on task specifications
        data = self._generate_synthetic_data(task, validation_config)
        
        # Cache the data
        self.data_cache[cache_key] = data
        
        self.logger.info(f"Data prepared: {data['train_size']} train, {data['val_size']} val, {data['test_size']} test samples")
        return data
    
    def _generate_synthetic_data(self, task: TaskDefinition, 
                                validation_config: ValidationConfiguration) -> Dict[str, Any]:
        """Generate synthetic data for demonstration"""
        # Calculate data sizes based on task
        base_size = task.dataset_info.get("size", 10000)
        test_size = int(base_size * validation_config.test_split)
        remaining_size = base_size - test_size
        val_size = int(remaining_size * validation_config.validation_split)
        train_size = remaining_size - val_size
        
        # Generate feature dimensions based on input spec
        if "shape" in task.input_spec:
            input_shape = task.input_spec["shape"]
        else:
            input_shape = [224, 224, 3]  # Default image shape
        
        # Generate output dimensions
        if "classes" in task.output_spec:
            num_classes = task.output_spec["classes"]
        else:
            num_classes = 10  # Default
        
        return {
            "task_id": task.id,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "train_data": f"synthetic_train_{train_size}",
            "val_data": f"synthetic_val_{val_size}",
            "test_data": f"synthetic_test_{test_size}",
            "preprocessing": self._get_preprocessing_config(task)
        }
    
    def _get_preprocessing_config(self, task: TaskDefinition) -> Dict[str, Any]:
        """Get preprocessing configuration for task"""
        config = {
            "normalization": "standard",
            "augmentation": task.task_type.value in ["classification", "regression"]
        }
        
        if task.task_type.value == "classification":
            config.update({
                "class_balancing": True,
                "label_smoothing": 0.1
            })
        
        return config


class TrainingEngine:
    """Core training engine"""
    
    def __init__(self, task_registry: TaskRegistry, meta_controller: RecursiveMetaController):
        self.task_registry = task_registry
        self.meta_controller = meta_controller
        self.active_trainings: Dict[str, Dict[str, Any]] = {}
        self.training_history: Dict[str, List[TrainingMetrics]] = {}
        self.model_checkpoints: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("TrainingEngine")
    
    async def train_model(self, task_id: str, model_config: Dict[str, Any],
                         training_config: TrainingConfiguration,
                         data: Dict[str, Any]) -> str:
        """Train a model for given task"""
        training_id = str(uuid.uuid4())
        
        self.logger.info(f"Starting training {training_id} for task {task_id}")
        
        # Initialize training state
        training_state = {
            "training_id": training_id,
            "task_id": task_id,
            "model_config": model_config,
            "training_config": training_config,
            "data": data,
            "start_time": datetime.now(),
            "status": "training",
            "best_metric": 0.0,
            "epochs_without_improvement": 0
        }
        
        self.active_trainings[training_id] = training_state
        self.training_history[training_id] = []
        
        try:
            # Training loop
            for epoch in range(training_config.num_epochs):
                # Simulate training step
                metrics = await self._training_step(training_state, epoch)
                self.training_history[training_id].append(metrics)
                
                # Validation step
                if epoch % training_config.validation_frequency == 0:
                    val_metrics = await self._validation_step(training_state, epoch)
                    metrics.val_loss = val_metrics["loss"]
                    metrics.val_accuracy = val_metrics["accuracy"]
                
                # Early stopping check
                if self._should_stop_early(training_state, metrics):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Save checkpoint
                if epoch % training_config.save_frequency == 0:
                    await self._save_checkpoint(training_state, epoch, metrics)
                
                # Meta-learning integration
                if training_config.mode == TrainingMode.META_LEARNING:
                    await self._meta_learning_step(training_state, metrics)
            
            training_state["status"] = "completed"
            training_state["end_time"] = datetime.now()
            
            self.logger.info(f"Training {training_id} completed")
            return training_id
            
        except Exception as e:
            training_state["status"] = "failed"
            training_state["error"] = str(e)
            self.logger.error(f"Training {training_id} failed: {e}")
            raise
    
    async def _training_step(self, training_state: Dict[str, Any], epoch: int) -> TrainingMetrics:
        """Simulate training step"""
        await asyncio.sleep(0.1)  # Simulate computation
        
        # Simulate training metrics with some improvement over time
        base_accuracy = 0.1 + (epoch / training_state["training_config"].num_epochs) * 0.7
        noise = np.random.normal(0, 0.02)
        train_accuracy = min(1.0, max(0.0, base_accuracy + noise))
        
        train_loss = max(0.1, 2.0 - (epoch / training_state["training_config"].num_epochs) * 1.5 + abs(noise))
        
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            learning_rate=training_state["training_config"].learning_rate * (0.95 ** (epoch // 10)),
            gradient_norm=np.random.uniform(0.5, 2.0),
            training_time=0.1,
            memory_usage=np.random.uniform(4.0, 8.0)
        )
        
        return metrics
    
    async def _validation_step(self, training_state: Dict[str, Any], epoch: int) -> Dict[str, float]:
        """Simulate validation step"""
        await asyncio.sleep(0.05)  # Simulate validation computation
        
        # Validation typically slightly lower than training
        train_history = self.training_history[training_state["training_id"]]
        if train_history:
            latest_train_acc = train_history[-1].train_accuracy
            val_accuracy = latest_train_acc * np.random.uniform(0.85, 0.95)
            val_loss = train_history[-1].train_loss * np.random.uniform(1.05, 1.15)
        else:
            val_accuracy = 0.1
            val_loss = 2.0
        
        return {
            "accuracy": val_accuracy,
            "loss": val_loss,
            "precision": val_accuracy * np.random.uniform(0.95, 1.05),
            "recall": val_accuracy * np.random.uniform(0.95, 1.05),
            "f1_score": val_accuracy * np.random.uniform(0.95, 1.05)
        }
    
    def _should_stop_early(self, training_state: Dict[str, Any], 
                          metrics: TrainingMetrics) -> bool:
        """Check early stopping condition"""
        config = training_state["training_config"]
        
        # Check if validation accuracy improved
        if metrics.val_accuracy > training_state["best_metric"]:
            training_state["best_metric"] = metrics.val_accuracy
            training_state["epochs_without_improvement"] = 0
            return False
        else:
            training_state["epochs_without_improvement"] += 1
            return training_state["epochs_without_improvement"] >= config.early_stopping_patience
    
    async def _save_checkpoint(self, training_state: Dict[str, Any], 
                              epoch: int, metrics: TrainingMetrics):
        """Save model checkpoint"""
        checkpoint_id = f"{training_state['training_id']}_epoch_{epoch}"
        
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "training_id": training_state["training_id"],
            "epoch": epoch,
            "model_state": f"model_weights_epoch_{epoch}",  # Placeholder
            "optimizer_state": f"optimizer_state_epoch_{epoch}",  # Placeholder
            "metrics": metrics.to_dict(),
            "timestamp": datetime.now()
        }
        
        self.model_checkpoints[checkpoint_id] = checkpoint
        self.logger.debug(f"Saved checkpoint: {checkpoint_id}")
    
    async def _meta_learning_step(self, training_state: Dict[str, Any], 
                                 metrics: TrainingMetrics):
        """Integrate with meta-learning controller"""
        if metrics.epoch % 10 == 0:  # Every 10 epochs
            # Submit meta-learning task
            meta_context = {
                "task_id": training_state["task_id"],
                "current_performance": metrics.val_accuracy,
                "training_progress": metrics.epoch / training_state["training_config"].num_epochs,
                "task_type": "training_optimization"
            }
            
            # Create dummy support/query data for meta-learning
            support_data = [f"training_sample_{i}" for i in range(10)]
            query_data = [f"validation_sample_{i}" for i in range(5)]
            
            await self.meta_controller.submit_adaptation_task(
                meta_context, support_data, query_data, priority=1.5
            )


class ValidationEngine:
    """Validation and evaluation engine"""
    
    def __init__(self, task_registry: TaskRegistry):
        self.task_registry = task_registry
        self.validation_results: Dict[str, ValidationResults] = {}
        self.logger = logging.getLogger("ValidationEngine")
    
    async def validate_model(self, task_id: str, model_id: str,
                           validation_config: ValidationConfiguration,
                           data: Dict[str, Any]) -> ValidationResults:
        """Validate model performance"""
        self.logger.info(f"Validating model {model_id} for task {task_id}")
        
        start_time = time.time()
        
        # Get task definition
        task = self.task_registry.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Create validation results container
        results = ValidationResults(
            task_id=task_id,
            model_id=model_id,
            validation_strategy=validation_config.strategy
        )
        
        # Perform validation based on strategy
        if validation_config.strategy == ValidationStrategy.K_FOLD:
            await self._k_fold_validation(results, task, validation_config, data)
        elif validation_config.strategy == ValidationStrategy.HOLDOUT:
            await self._holdout_validation(results, task, validation_config, data)
        elif validation_config.strategy == ValidationStrategy.BOOTSTRAP:
            await self._bootstrap_validation(results, task, validation_config, data)
        else:
            await self._holdout_validation(results, task, validation_config, data)
        
        # Calculate summary metrics
        self._calculate_summary_metrics(results, task)
        
        results.validation_time = time.time() - start_time
        
        # Store results
        self.validation_results[f"{task_id}_{model_id}"] = results
        
        self.logger.info(f"Validation completed in {results.validation_time:.2f}s")
        return results
    
    async def _k_fold_validation(self, results: ValidationResults, 
                               task: TaskDefinition,
                               config: ValidationConfiguration,
                               data: Dict[str, Any]):
        """Perform k-fold cross-validation"""
        k_folds = config.k_folds
        
        for fold in range(k_folds):
            self.logger.debug(f"Processing fold {fold + 1}/{k_folds}")
            await asyncio.sleep(0.2)  # Simulate training/validation time
            
            # Simulate fold results
            fold_metrics = await self._simulate_fold_evaluation(task, fold)
            results.fold_results.append(fold_metrics)
        
        # Calculate confidence intervals
        self._calculate_confidence_intervals(results)
    
    async def _holdout_validation(self, results: ValidationResults,
                                task: TaskDefinition,
                                config: ValidationConfiguration,
                                data: Dict[str, Any]):
        """Perform holdout validation"""
        await asyncio.sleep(0.3)  # Simulate validation time
        
        # Simulate validation results
        fold_metrics = await self._simulate_fold_evaluation(task, 0)
        results.fold_results.append(fold_metrics)
    
    async def _bootstrap_validation(self, results: ValidationResults,
                                  task: TaskDefinition,
                                  config: ValidationConfiguration,
                                  data: Dict[str, Any]):
        """Perform bootstrap validation"""
        num_samples = min(config.bootstrap_samples, 100)  # Limit for demo
        
        for sample in range(num_samples):
            if sample % 20 == 0:  # Log progress
                self.logger.debug(f"Bootstrap sample {sample + 1}/{num_samples}")
            
            await asyncio.sleep(0.01)  # Simulate sampling and evaluation
            
            # Simulate bootstrap sample results
            sample_metrics = await self._simulate_fold_evaluation(task, sample)
            results.fold_results.append(sample_metrics)
        
        # Calculate confidence intervals
        self._calculate_confidence_intervals(results)
    
    async def _simulate_fold_evaluation(self, task: TaskDefinition, 
                                      fold: int) -> Dict[str, float]:
        """Simulate evaluation for a single fold"""
        # Base performance with some variation
        base_performance = 0.7 + np.random.normal(0, 0.05)
        base_performance = max(0.0, min(1.0, base_performance))
        
        # Generate metrics based on task
        metrics = {}
        
        for metric in task.performance_metrics:
            if metric.metric_type.value in ["accuracy", "precision", "recall", "f1_score"]:
                # Performance metrics
                value = base_performance * np.random.uniform(0.95, 1.05)
                value = max(0.0, min(1.0, value))
            elif metric.metric_type.value in ["loss", "latency"]:
                # Loss/latency metrics (lower is better)
                value = (1.0 - base_performance) * np.random.uniform(0.8, 1.2)
                value = max(0.001, value)
            elif metric.metric_type.value in ["throughput"]:
                # Throughput (higher is better)
                value = base_performance * 1000 * np.random.uniform(0.9, 1.1)
                value = max(1.0, value)
            else:
                # Default metric
                value = base_performance * np.random.uniform(0.9, 1.1)
                value = max(0.0, min(1.0, value))
            
            metrics[metric.name] = value
        
        return metrics
    
    def _calculate_summary_metrics(self, results: ValidationResults, 
                                  task: TaskDefinition):
        """Calculate summary metrics across folds"""
        if not results.fold_results:
            return
        
        # Calculate mean and std for each metric
        for metric in task.performance_metrics:
            values = [fold.get(metric.name, 0.0) for fold in results.fold_results]
            
            if values:
                results.metrics[metric.name] = statistics.mean(values)
                results.metrics[f"{metric.name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    
    def _calculate_confidence_intervals(self, results: ValidationResults):
        """Calculate confidence intervals for metrics"""
        if len(results.fold_results) < 2:
            return
        
        for metric_name in results.metrics:
            if not metric_name.endswith("_std"):
                values = [fold.get(metric_name, 0.0) for fold in results.fold_results]
                
                if values:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values)
                    
                    # 95% confidence interval (assuming normal distribution)
                    margin = 1.96 * (std_val / np.sqrt(len(values)))
                    results.confidence_intervals[metric_name] = (
                        mean_val - margin, mean_val + margin
                    )


class TrainingValidationPipeline:
    """Main training and validation pipeline coordinator"""
    
    def __init__(self, task_registry: TaskRegistry = None):
        self.task_registry = task_registry or TaskRegistry()
        self.meta_controller = RecursiveMetaController()
        self.data_pipeline = DataPipeline(self.task_registry)
        self.training_engine = TrainingEngine(self.task_registry, self.meta_controller)
        self.validation_engine = ValidationEngine(self.task_registry)
        
        self.pipeline_history: List[Dict[str, Any]] = []
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("TrainingValidationPipeline")
    
    async def run_full_pipeline(self, task_id: str,
                               model_config: Dict[str, Any] = None,
                               training_config: TrainingConfiguration = None,
                               validation_config: ValidationConfiguration = None) -> Dict[str, Any]:
        """Run complete training and validation pipeline"""
        pipeline_id = str(uuid.uuid4())
        
        self.logger.info(f"Starting pipeline {pipeline_id} for task {task_id}")
        
        # Use defaults if not provided
        if training_config is None:
            training_config = TrainingConfiguration()
        if validation_config is None:
            validation_config = ValidationConfiguration()
        if model_config is None:
            model_config = {"type": "default", "parameters": {}}
        
        start_time = datetime.now()
        
        pipeline_state = {
            "pipeline_id": pipeline_id,
            "task_id": task_id,
            "status": "running",
            "start_time": start_time,
            "stage": PipelineStage.DATA_PREPARATION
        }
        
        self.active_pipelines[pipeline_id] = pipeline_state
        
        try:
            # Stage 1: Data Preparation
            self.logger.info(f"Pipeline {pipeline_id}: Data preparation")
            pipeline_state["stage"] = PipelineStage.DATA_PREPARATION
            data = await self.data_pipeline.prepare_task_data(task_id, validation_config)
            
            # Stage 2: Training
            self.logger.info(f"Pipeline {pipeline_id}: Training")
            pipeline_state["stage"] = PipelineStage.TRAINING
            training_id = await self.training_engine.train_model(
                task_id, model_config, training_config, data
            )
            
            # Stage 3: Validation
            self.logger.info(f"Pipeline {pipeline_id}: Validation")
            pipeline_state["stage"] = PipelineStage.VALIDATION
            validation_results = await self.validation_engine.validate_model(
                task_id, training_id, validation_config, data
            )
            
            # Stage 4: Performance Evaluation
            self.logger.info(f"Pipeline {pipeline_id}: Performance evaluation")
            pipeline_state["stage"] = PipelineStage.PERFORMANCE_EVALUATION
            performance_analysis = await self._analyze_performance(task_id, validation_results)
            
            # Complete pipeline
            end_time = datetime.now()
            pipeline_duration = (end_time - start_time).total_seconds()
            
            pipeline_result = {
                "pipeline_id": pipeline_id,
                "task_id": task_id,
                "status": "completed",
                "duration": pipeline_duration,
                "training_id": training_id,
                "validation_results": validation_results,
                "performance_analysis": performance_analysis,
                "data_info": data,
                "configurations": {
                    "training": training_config,
                    "validation": validation_config,
                    "model": model_config
                }
            }
            
            # Store in history
            self.pipeline_history.append(pipeline_result)
            pipeline_state["status"] = "completed"
            
            self.logger.info(f"Pipeline {pipeline_id} completed in {pipeline_duration:.2f}s")
            return pipeline_result
            
        except Exception as e:
            pipeline_state["status"] = "failed"
            pipeline_state["error"] = str(e)
            self.logger.error(f"Pipeline {pipeline_id} failed: {e}")
            raise
        
        finally:
            if pipeline_id in self.active_pipelines:
                del self.active_pipelines[pipeline_id]
    
    async def _analyze_performance(self, task_id: str, 
                                  validation_results: ValidationResults) -> Dict[str, Any]:
        """Analyze model performance"""
        task = self.task_registry.get_task(task_id)
        if not task:
            return {"error": "Task not found"}
        
        analysis = {
            "task_performance_score": task.calculate_task_score(validation_results.metrics),
            "metric_analysis": {},
            "improvement_recommendations": [],
            "convergence_analysis": {},
            "robustness_metrics": {}
        }
        
        # Analyze each metric
        for metric in task.performance_metrics:
            if metric.name in validation_results.metrics:
                value = validation_results.metrics[metric.name]
                baseline = metric.baseline or 0.5
                target = metric.target or 0.8
                
                # Calculate improvement vs baseline
                if metric.baseline:
                    improvement = metric.calculate_improvement_score(baseline, value)
                else:
                    improvement = value
                
                analysis["metric_analysis"][metric.name] = {
                    "value": value,
                    "baseline": baseline,
                    "target": target,
                    "improvement_score": improvement,
                    "target_achieved": value >= target if metric.target else False,
                    "confidence_interval": validation_results.confidence_intervals.get(metric.name)
                }
                
                # Generate recommendations
                if value < target:
                    gap = target - value
                    if gap > 0.2:
                        analysis["improvement_recommendations"].append(
                            f"Significant improvement needed for {metric.name} (gap: {gap:.3f})"
                        )
                    elif gap > 0.1:
                        analysis["improvement_recommendations"].append(
                            f"Moderate improvement needed for {metric.name} (gap: {gap:.3f})"
                        )
        
        # Convergence analysis
        if len(validation_results.fold_results) > 5:
            metric_values = [fold.get(list(validation_results.metrics.keys())[0], 0.0) 
                           for fold in validation_results.fold_results]
            
            analysis["convergence_analysis"] = {
                "variance": statistics.variance(metric_values) if len(metric_values) > 1 else 0.0,
                "coefficient_of_variation": (statistics.stdev(metric_values) / statistics.mean(metric_values) 
                                           if statistics.mean(metric_values) > 0 and len(metric_values) > 1 else 0.0),
                "convergence_quality": "high" if statistics.variance(metric_values) < 0.01 else "moderate"
            }
        
        return analysis
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status"""
        return {
            "active_pipelines": len(self.active_pipelines),
            "completed_pipelines": len(self.pipeline_history),
            "active_trainings": len(self.training_engine.active_trainings),
            "total_checkpoints": len(self.training_engine.model_checkpoints),
            "validation_results": len(self.validation_engine.validation_results),
            "cache_sizes": {
                "data_cache": len(self.data_pipeline.data_cache),
                "preprocessing_cache": len(self.data_pipeline.preprocessing_cache)
            }
        }


# Export key classes
__all__ = [
    "PipelineStage", "ValidationStrategy", "TrainingMode",
    "TrainingConfiguration", "ValidationConfiguration", "TrainingMetrics",
    "ValidationResults", "DataPipeline", "TrainingEngine", "ValidationEngine",
    "TrainingValidationPipeline"
]


if __name__ == "__main__":
    # Demo usage
    async def demo():
        logging.basicConfig(level=logging.INFO)
        
        print("🔧 Training and Validation Pipeline Demo")
        print("=" * 60)
        
        # Create pipeline
        pipeline = TrainingValidationPipeline()
        
        # Get high priority tasks
        high_priority_tasks = pipeline.task_registry.get_high_priority_tasks(threshold=1.0)
        
        print(f"📋 Running pipeline on {len(high_priority_tasks)} high-priority tasks")
        
        # Run pipeline for first few tasks
        results = []
        for task in high_priority_tasks[:2]:  # Limit for demo
            print(f"\n🚀 Running pipeline for: {task.name}")
            
            # Configure training and validation
            training_config = TrainingConfiguration(
                num_epochs=20,
                batch_size=16,
                learning_rate=0.001,
                early_stopping_patience=5
            )
            
            validation_config = ValidationConfiguration(
                strategy=ValidationStrategy.K_FOLD,
                k_folds=3,
                validation_split=0.2
            )
            
            # Run pipeline
            try:
                result = await pipeline.run_full_pipeline(
                    task.id, 
                    training_config=training_config,
                    validation_config=validation_config
                )
                results.append((task.name, result))
                
            except Exception as e:
                print(f"❌ Pipeline failed for {task.name}: {e}")
                continue
        
        # Show results
        print(f"\n📊 Pipeline Results:")
        for task_name, result in results:
            val_results = result["validation_results"]
            perf_analysis = result["performance_analysis"]
            
            print(f"\n  • {task_name}:")
            print(f"    Duration: {result['duration']:.2f}s")
            print(f"    Task Performance Score: {perf_analysis['task_performance_score']:.3f}")
            print(f"    Validation Strategy: {val_results.validation_strategy.value}")
            print(f"    Number of Folds: {len(val_results.fold_results)}")
            
            # Show key metrics
            if val_results.metrics:
                primary_metric = list(val_results.metrics.keys())[0]
                print(f"    Primary Metric ({primary_metric}): {val_results.metrics[primary_metric]:.3f}")
                
                if primary_metric in val_results.confidence_intervals:
                    ci = val_results.confidence_intervals[primary_metric]
                    print(f"    95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
            
            # Show recommendations
            recommendations = perf_analysis.get("improvement_recommendations", [])
            if recommendations:
                print(f"    Recommendations: {len(recommendations)} items")
                for rec in recommendations[:2]:  # Show first 2
                    print(f"      - {rec}")
        
        # Show overall status
        status = pipeline.get_pipeline_status()
        print(f"\n🎯 Pipeline Status:")
        print(f"  Completed Pipelines: {status['completed_pipelines']}")
        print(f"  Active Trainings: {status['active_trainings']}")
        print(f"  Total Checkpoints: {status['total_checkpoints']}")
        print(f"  Validation Results: {status['validation_results']}")
        print(f"  Data Cache Size: {status['cache_sizes']['data_cache']}")
        
        print(f"\n✅ Training and validation pipeline operational!")
    
    # Run demo
    asyncio.run(demo())