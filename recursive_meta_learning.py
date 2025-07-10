#!/usr/bin/env python3
"""
Recursive Meta-Learning Framework
Atomic Task 50.2: Design Recursive Meta-Learning Framework

This module implements a recursive meta-learning system that enables models to
learn from sequential decision points and adapt to new tasks using recursive
feedback loops and meta-optimization strategies.
"""

import asyncio
import json
import logging
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple, NamedTuple
from pathlib import Path
import statistics
import numpy as np
from collections import defaultdict, deque
import copy


class MetaLearningStrategy(Enum):
    """Types of meta-learning strategies"""
    MAML = "model_agnostic_meta_learning"
    REPTILE = "reptile"
    PROTOTYPICAL = "prototypical_networks"
    MATCHING = "matching_networks"
    RELATION = "relation_networks"
    GRADIENT_BASED = "gradient_based"
    MEMORY_AUGMENTED = "memory_augmented"
    OPTIMIZATION_BASED = "optimization_based"
    METRIC_BASED = "metric_based"
    RECURSIVE_SELF_IMPROVEMENT = "recursive_self_improvement"


class AdaptationMode(Enum):
    """Modes of adaptation in meta-learning"""
    FINE_TUNING = "fine_tuning"
    FAST_WEIGHTS = "fast_weights"
    CONTEXT_MODULATION = "context_modulation"
    HYPERNETWORK = "hypernetwork"
    MEMORY_RETRIEVAL = "memory_retrieval"
    RECURSIVE_REFLECTION = "recursive_reflection"


class MetaObjective(Enum):
    """Meta-learning optimization objectives"""
    FAST_ADAPTATION = "fast_adaptation"
    SAMPLE_EFFICIENCY = "sample_efficiency"
    GENERALIZATION = "generalization"
    TRANSFER_CAPABILITY = "transfer_capability"
    CATASTROPHIC_FORGETTING_RESISTANCE = "catastrophic_forgetting_resistance"
    RECURSIVE_IMPROVEMENT = "recursive_improvement"


@dataclass
class MetaExperience:
    """Single meta-learning experience"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_context: Dict[str, Any] = field(default_factory=dict)
    support_data: List[Any] = field(default_factory=list)
    query_data: List[Any] = field(default_factory=list)
    adaptation_steps: List[Dict[str, Any]] = field(default_factory=list)
    performance_trajectory: List[float] = field(default_factory=list)
    meta_gradient: Optional[Dict[str, Any]] = None
    adaptation_success: bool = False
    adaptation_time: float = 0.0
    final_performance: float = 0.0
    context_similarity: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_adaptation_speed(self) -> float:
        """Calculate adaptation speed metric"""
        if len(self.performance_trajectory) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(self.performance_trajectory)):
            improvement = self.performance_trajectory[i] - self.performance_trajectory[i-1]
            improvements.append(max(0, improvement))
        
        return statistics.mean(improvements) if improvements else 0.0
    
    def extract_adaptation_pattern(self) -> Dict[str, Any]:
        """Extract reusable adaptation patterns"""
        return {
            "context_features": self._extract_context_features(),
            "adaptation_trajectory": self.performance_trajectory,
            "successful_steps": [step for step in self.adaptation_steps if step.get("success", False)],
            "convergence_pattern": self._analyze_convergence(),
            "meta_insights": self._extract_meta_insights()
        }
    
    def _extract_context_features(self) -> Dict[str, Any]:
        """Extract key features from task context"""
        features = {}
        if "task_type" in self.task_context:
            features["task_type"] = self.task_context["task_type"]
        if "data_size" in self.task_context:
            features["data_size"] = self.task_context["data_size"]
        if "complexity" in self.task_context:
            features["complexity"] = self.task_context["complexity"]
        return features
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence patterns"""
        if len(self.performance_trajectory) < 3:
            return {"pattern": "insufficient_data"}
        
        # Calculate convergence characteristics
        trajectory = np.array(self.performance_trajectory)
        differences = np.diff(trajectory)
        
        return {
            "converged": len(differences) > 0 and abs(differences[-1]) < 0.01,
            "convergence_step": len(differences),
            "convergence_rate": float(np.mean(differences)) if len(differences) > 0 else 0.0,
            "stability": float(np.std(differences[-5:])) if len(differences) >= 5 else 1.0
        }
    
    def _extract_meta_insights(self) -> List[str]:
        """Extract meta-learning insights"""
        insights = []
        
        if self.adaptation_success:
            insights.append("successful_adaptation")
        
        if self.adaptation_time < 5.0:
            insights.append("fast_adaptation")
        
        if len(self.performance_trajectory) > 0 and self.performance_trajectory[-1] > 0.8:
            insights.append("high_performance_achieved")
        
        return insights


@dataclass
class MetaStrategy:
    """Meta-learning strategy configuration"""
    name: str
    strategy_type: MetaLearningStrategy
    adaptation_mode: AdaptationMode
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)
    success_rate: float = 0.0
    average_adaptation_time: float = 0.0
    contexts_handled: Set[str] = field(default_factory=set)
    recursive_depth: int = 1
    meta_optimizer_state: Optional[Dict[str, Any]] = None
    
    def update_performance(self, performance: float, adaptation_time: float, 
                         context: str, success: bool):
        """Update strategy performance metrics"""
        self.performance_history.append(performance)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # Update success rate
        recent_successes = sum(1 for p in self.performance_history[-20:] if p > 0.7)
        self.success_rate = recent_successes / min(20, len(self.performance_history))
        
        # Update average adaptation time
        if self.average_adaptation_time == 0:
            self.average_adaptation_time = adaptation_time
        else:
            self.average_adaptation_time = 0.9 * self.average_adaptation_time + 0.1 * adaptation_time
        
        self.contexts_handled.add(context)
    
    def get_effectiveness_score(self) -> float:
        """Calculate overall effectiveness score"""
        if not self.performance_history:
            return 0.0
        
        # Combine multiple factors
        performance_score = statistics.mean(self.performance_history[-10:])
        speed_score = max(0, 1.0 - (self.average_adaptation_time / 60.0))  # Normalize to 60s
        consistency_score = 1.0 - (statistics.stdev(self.performance_history[-10:]) 
                                 if len(self.performance_history) >= 2 else 0.5)
        versatility_score = min(1.0, len(self.contexts_handled) / 10.0)
        
        return (performance_score * 0.4 + speed_score * 0.2 + 
                consistency_score * 0.2 + versatility_score * 0.2)


class MetaLearningEngine:
    """Core meta-learning engine with recursive capabilities"""
    
    def __init__(self, memory_size: int = 10000):
        self.memory_size = memory_size
        self.experience_memory: deque = deque(maxlen=memory_size)
        self.meta_strategies: Dict[str, MetaStrategy] = {}
        self.active_strategy: Optional[str] = None
        self.context_similarity_threshold = 0.8
        self.adaptation_budget = 50  # Max adaptation steps
        self.recursive_depth_limit = 5
        self.logger = logging.getLogger("MetaLearningEngine")
        
        # Performance tracking
        self.adaptation_history: List[Dict[str, Any]] = []
        self.meta_learning_stats = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "average_adaptation_time": 0.0,
            "best_performance": 0.0,
            "strategy_switches": 0
        }
        
        # Initialize default strategies
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default meta-learning strategies"""
        
        # MAML-based strategy
        maml_strategy = MetaStrategy(
            name="MAML_Recursive",
            strategy_type=MetaLearningStrategy.MAML,
            adaptation_mode=AdaptationMode.FINE_TUNING,
            hyperparameters={
                "inner_lr": 0.01,
                "outer_lr": 0.001,
                "inner_steps": 5,
                "meta_batch_size": 4
            },
            recursive_depth=2
        )
        self.register_strategy(maml_strategy)
        
        # Prototypical networks strategy
        proto_strategy = MetaStrategy(
            name="Prototypical_Memory",
            strategy_type=MetaLearningStrategy.PROTOTYPICAL,
            adaptation_mode=AdaptationMode.MEMORY_RETRIEVAL,
            hyperparameters={
                "embedding_dim": 128,
                "distance_metric": "euclidean",
                "temperature": 1.0
            },
            recursive_depth=1
        )
        self.register_strategy(proto_strategy)
        
        # Recursive self-improvement strategy
        recursive_strategy = MetaStrategy(
            name="Recursive_Self_Improvement",
            strategy_type=MetaLearningStrategy.RECURSIVE_SELF_IMPROVEMENT,
            adaptation_mode=AdaptationMode.RECURSIVE_REFLECTION,
            hyperparameters={
                "reflection_depth": 3,
                "improvement_threshold": 0.05,
                "meta_meta_lr": 0.0001
            },
            recursive_depth=3
        )
        self.register_strategy(recursive_strategy)
        
        # Set default active strategy
        self.active_strategy = "MAML_Recursive"
    
    def register_strategy(self, strategy: MetaStrategy):
        """Register a new meta-learning strategy"""
        self.meta_strategies[strategy.name] = strategy
        self.logger.info(f"Registered meta-learning strategy: {strategy.name}")
    
    async def adapt_to_task(self, task_context: Dict[str, Any], 
                           support_data: List[Any], 
                           query_data: List[Any],
                           recursive_depth: int = 0) -> MetaExperience:
        """Adapt to a new task using meta-learning"""
        
        if recursive_depth >= self.recursive_depth_limit:
            self.logger.warning(f"Maximum recursive depth {self.recursive_depth_limit} reached")
            return self._create_fallback_experience(task_context, support_data, query_data)
        
        start_time = time.time()
        self.logger.info(f"Starting adaptation to task (depth {recursive_depth}): {task_context.get('name', 'unknown')}")
        
        # Select best strategy for this context
        strategy_name = await self._select_strategy(task_context, recursive_depth)
        strategy = self.meta_strategies[strategy_name]
        
        # Create experience container
        experience = MetaExperience(
            task_context=task_context,
            support_data=support_data,
            query_data=query_data
        )
        
        try:
            # Execute adaptation based on strategy
            if strategy.strategy_type == MetaLearningStrategy.MAML:
                await self._adapt_with_maml(experience, strategy, recursive_depth)
            elif strategy.strategy_type == MetaLearningStrategy.PROTOTYPICAL:
                await self._adapt_with_prototypical(experience, strategy, recursive_depth)
            elif strategy.strategy_type == MetaLearningStrategy.RECURSIVE_SELF_IMPROVEMENT:
                await self._adapt_with_recursive_improvement(experience, strategy, recursive_depth)
            else:
                await self._adapt_with_generic(experience, strategy, recursive_depth)
            
            # Calculate final metrics
            adaptation_time = time.time() - start_time
            experience.adaptation_time = adaptation_time
            experience.adaptation_success = len(experience.performance_trajectory) > 0 and experience.performance_trajectory[-1] > 0.5
            experience.final_performance = experience.performance_trajectory[-1] if experience.performance_trajectory else 0.0
            
            # Update strategy performance
            strategy.update_performance(
                experience.final_performance,
                adaptation_time,
                task_context.get("task_type", "unknown"),
                experience.adaptation_success
            )
            
            # Store experience in memory
            self.experience_memory.append(experience)
            
            # Update meta-learning stats
            self._update_meta_stats(experience)
            
            # Recursive improvement: If performance is below threshold, try recursive adaptation
            if (experience.final_performance < 0.7 and 
                recursive_depth < self.recursive_depth_limit - 1 and
                strategy.recursive_depth > recursive_depth):
                
                self.logger.info(f"Performance {experience.final_performance:.3f} below threshold, attempting recursive improvement")
                
                # Create enhanced context with current experience
                enhanced_context = task_context.copy()
                enhanced_context["previous_experience"] = experience.extract_adaptation_pattern()
                enhanced_context["recursive_attempt"] = recursive_depth + 1
                
                # Recursive adaptation with enhanced context
                recursive_experience = await self.adapt_to_task(
                    enhanced_context, support_data, query_data, recursive_depth + 1
                )
                
                # Use better of the two experiences
                if recursive_experience.final_performance > experience.final_performance:
                    experience = recursive_experience
            
            self.logger.info(f"Adaptation completed: {experience.final_performance:.3f} performance in {adaptation_time:.2f}s")
            return experience
            
        except Exception as e:
            self.logger.error(f"Error during adaptation: {e}")
            experience.adaptation_success = False
            experience.final_performance = 0.0
            return experience
    
    async def _select_strategy(self, task_context: Dict[str, Any], recursive_depth: int) -> str:
        """Select best meta-learning strategy for given context"""
        
        # Find similar past experiences
        similar_experiences = self._find_similar_experiences(task_context)
        
        if similar_experiences:
            # Use strategy that worked best for similar contexts
            strategy_performance = defaultdict(list)
            for exp in similar_experiences[-20:]:  # Last 20 similar experiences
                for strategy_name, strategy in self.meta_strategies.items():
                    if task_context.get("task_type") in strategy.contexts_handled:
                        strategy_performance[strategy_name].append(exp.final_performance)
            
            if strategy_performance:
                best_strategy = max(strategy_performance.items(), 
                                  key=lambda x: statistics.mean(x[1]))[0]
                
                # Consider recursive depth
                if (recursive_depth > 0 and 
                    self.meta_strategies[best_strategy].recursive_depth <= recursive_depth):
                    # Switch to recursive strategy for deeper levels
                    recursive_strategies = [name for name, s in self.meta_strategies.items() 
                                          if s.recursive_depth > recursive_depth]
                    if recursive_strategies:
                        best_strategy = max(recursive_strategies, 
                                          key=lambda x: self.meta_strategies[x].get_effectiveness_score())
                
                self.logger.info(f"Selected strategy based on similar experiences: {best_strategy}")
                return best_strategy
        
        # Fallback: Use strategy with highest effectiveness score
        if not self.meta_strategies:
            return "MAML_Recursive"
        
        best_strategy = max(self.meta_strategies.items(), 
                          key=lambda x: x[1].get_effectiveness_score())[0]
        
        self.logger.info(f"Selected strategy based on effectiveness: {best_strategy}")
        return best_strategy
    
    async def _adapt_with_maml(self, experience: MetaExperience, 
                              strategy: MetaStrategy, recursive_depth: int):
        """Implement MAML-based adaptation"""
        inner_lr = strategy.hyperparameters.get("inner_lr", 0.01)
        inner_steps = strategy.hyperparameters.get("inner_steps", 5)
        
        # Simulate MAML adaptation process
        base_performance = 0.3  # Starting performance
        experience.performance_trajectory.append(base_performance)
        
        for step in range(inner_steps):
            # Simulate inner loop gradient update
            await asyncio.sleep(0.1)  # Simulate computation time
            
            # Calculate improvement based on support data
            improvement = min(0.1, len(experience.support_data) * 0.02)
            new_performance = experience.performance_trajectory[-1] + improvement * inner_lr
            
            # Add some noise and diminishing returns
            noise = np.random.normal(0, 0.02)
            new_performance += noise
            new_performance = min(1.0, max(0.0, new_performance))
            
            experience.performance_trajectory.append(new_performance)
            experience.adaptation_steps.append({
                "step": step,
                "learning_rate": inner_lr,
                "performance": new_performance,
                "success": new_performance > experience.performance_trajectory[-2]
            })
            
            # Early stopping if converged
            if step > 0 and abs(new_performance - experience.performance_trajectory[-2]) < 0.01:
                break
    
    async def _adapt_with_prototypical(self, experience: MetaExperience, 
                                      strategy: MetaStrategy, recursive_depth: int):
        """Implement prototypical networks adaptation"""
        embedding_dim = strategy.hyperparameters.get("embedding_dim", 128)
        
        # Simulate prototype learning
        base_performance = 0.4
        experience.performance_trajectory.append(base_performance)
        
        # Build prototypes from support data
        num_prototypes = min(len(experience.support_data), 10)
        for proto_step in range(num_prototypes):
            await asyncio.sleep(0.05)
            
            # Simulate prototype-based classification improvement
            improvement = 0.05 * (1.0 - proto_step / num_prototypes)
            new_performance = experience.performance_trajectory[-1] + improvement
            new_performance = min(1.0, max(0.0, new_performance))
            
            experience.performance_trajectory.append(new_performance)
            experience.adaptation_steps.append({
                "step": proto_step,
                "prototype_count": proto_step + 1,
                "performance": new_performance,
                "success": True
            })
    
    async def _adapt_with_recursive_improvement(self, experience: MetaExperience, 
                                               strategy: MetaStrategy, recursive_depth: int):
        """Implement recursive self-improvement adaptation"""
        reflection_depth = strategy.hyperparameters.get("reflection_depth", 3)
        improvement_threshold = strategy.hyperparameters.get("improvement_threshold", 0.05)
        
        base_performance = 0.35
        experience.performance_trajectory.append(base_performance)
        
        for reflection_level in range(reflection_depth):
            await asyncio.sleep(0.2)  # Deeper reflection takes more time
            
            # Simulate recursive self-reflection
            if reflection_level == 0:
                # First-order reflection: analyze current performance
                reflection_gain = 0.08
            elif reflection_level == 1:
                # Second-order reflection: reflect on reflection process
                reflection_gain = 0.06
            else:
                # Higher-order reflection: meta-meta learning
                reflection_gain = 0.04
            
            # Apply recursive depth bonus
            depth_bonus = min(0.05, recursive_depth * 0.02)
            total_improvement = reflection_gain + depth_bonus
            
            new_performance = experience.performance_trajectory[-1] + total_improvement
            new_performance = min(1.0, max(0.0, new_performance))
            
            experience.performance_trajectory.append(new_performance)
            experience.adaptation_steps.append({
                "step": reflection_level,
                "reflection_depth": reflection_level + 1,
                "recursive_depth": recursive_depth,
                "performance": new_performance,
                "improvement": total_improvement,
                "success": total_improvement > improvement_threshold
            })
            
            # Stop if improvement is below threshold
            if total_improvement < improvement_threshold:
                break
    
    async def _adapt_with_generic(self, experience: MetaExperience, 
                                 strategy: MetaStrategy, recursive_depth: int):
        """Generic adaptation fallback"""
        # Simple linear improvement
        base_performance = 0.25
        experience.performance_trajectory.append(base_performance)
        
        for step in range(5):
            await asyncio.sleep(0.1)
            new_performance = base_performance + (step + 1) * 0.1
            new_performance = min(1.0, max(0.0, new_performance))
            experience.performance_trajectory.append(new_performance)
            experience.adaptation_steps.append({
                "step": step,
                "performance": new_performance,
                "success": True
            })
    
    def _find_similar_experiences(self, task_context: Dict[str, Any]) -> List[MetaExperience]:
        """Find experiences with similar task contexts"""
        similar = []
        target_type = task_context.get("task_type", "")
        
        for exp in self.experience_memory:
            similarity = self._calculate_context_similarity(task_context, exp.task_context)
            if similarity >= self.context_similarity_threshold:
                exp.context_similarity = similarity
                similar.append(exp)
        
        # Sort by similarity and recency
        similar.sort(key=lambda x: (x.context_similarity, x.timestamp), reverse=True)
        return similar[:20]  # Return top 20 most similar
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                    context2: Dict[str, Any]) -> float:
        """Calculate similarity between task contexts"""
        if not context1 or not context2:
            return 0.0
        
        # Simple similarity based on common keys and values
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        matching_values = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                matching_values += 1
        
        return matching_values / len(common_keys)
    
    def _create_fallback_experience(self, task_context: Dict[str, Any],
                                   support_data: List[Any],
                                   query_data: List[Any]) -> MetaExperience:
        """Create fallback experience when adaptation fails"""
        experience = MetaExperience(
            task_context=task_context,
            support_data=support_data,
            query_data=query_data,
            performance_trajectory=[0.3],  # Basic fallback performance
            adaptation_success=False,
            final_performance=0.3
        )
        return experience
    
    def _update_meta_stats(self, experience: MetaExperience):
        """Update meta-learning statistics"""
        self.meta_learning_stats["total_adaptations"] += 1
        
        if experience.adaptation_success:
            self.meta_learning_stats["successful_adaptations"] += 1
        
        # Update average adaptation time
        total = self.meta_learning_stats["total_adaptations"]
        current_avg = self.meta_learning_stats["average_adaptation_time"]
        new_avg = ((current_avg * (total - 1)) + experience.adaptation_time) / total
        self.meta_learning_stats["average_adaptation_time"] = new_avg
        
        # Update best performance
        if experience.final_performance > self.meta_learning_stats["best_performance"]:
            self.meta_learning_stats["best_performance"] = experience.final_performance
    
    def get_meta_learning_insights(self) -> Dict[str, Any]:
        """Get insights about meta-learning performance"""
        if not self.experience_memory:
            return {"status": "no_experiences"}
        
        recent_experiences = list(self.experience_memory)[-50:]
        
        # Calculate success rate
        success_rate = sum(1 for exp in recent_experiences if exp.adaptation_success) / len(recent_experiences)
        
        # Calculate average performance
        avg_performance = statistics.mean(exp.final_performance for exp in recent_experiences)
        
        # Analyze adaptation patterns
        adaptation_speeds = [exp.calculate_adaptation_speed() for exp in recent_experiences]
        avg_adaptation_speed = statistics.mean(adaptation_speeds) if adaptation_speeds else 0.0
        
        # Strategy effectiveness
        strategy_stats = {}
        for strategy_name, strategy in self.meta_strategies.items():
            strategy_stats[strategy_name] = {
                "effectiveness_score": strategy.get_effectiveness_score(),
                "success_rate": strategy.success_rate,
                "avg_adaptation_time": strategy.average_adaptation_time,
                "contexts_handled": len(strategy.contexts_handled)
            }
        
        return {
            "total_experiences": len(self.experience_memory),
            "recent_success_rate": success_rate,
            "average_performance": avg_performance,
            "average_adaptation_speed": avg_adaptation_speed,
            "strategy_statistics": strategy_stats,
            "meta_learning_stats": self.meta_learning_stats,
            "best_strategy": max(self.meta_strategies.items(), 
                               key=lambda x: x[1].get_effectiveness_score())[0] if self.meta_strategies else None
        }


class RecursiveMetaController:
    """High-level controller for recursive meta-learning system"""
    
    def __init__(self):
        self.meta_engine = MetaLearningEngine()
        self.task_queue: deque = deque()
        self.active_adaptations: Dict[str, asyncio.Task] = {}
        self.adaptation_results: Dict[str, MetaExperience] = {}
        self.recursive_improvement_enabled = True
        self.max_concurrent_adaptations = 5
        self.logger = logging.getLogger("RecursiveMetaController")
        
        # Performance tracking
        self.controller_stats = {
            "tasks_processed": 0,
            "average_improvement": 0.0,
            "recursive_calls": 0,
            "strategy_switches": 0
        }
    
    async def submit_adaptation_task(self, task_context: Dict[str, Any],
                                   support_data: List[Any],
                                   query_data: List[Any],
                                   priority: float = 1.0) -> str:
        """Submit a new adaptation task"""
        task_id = str(uuid.uuid4())
        
        task_info = {
            "task_id": task_id,
            "context": task_context,
            "support_data": support_data,
            "query_data": query_data,
            "priority": priority,
            "submitted_at": datetime.now()
        }
        
        self.task_queue.append(task_info)
        self.logger.info(f"Submitted adaptation task: {task_id}")
        
        # Start processing if we have capacity
        if len(self.active_adaptations) < self.max_concurrent_adaptations:
            await self._process_next_task()
        
        return task_id
    
    async def _process_next_task(self):
        """Process next task in queue"""
        if not self.task_queue or len(self.active_adaptations) >= self.max_concurrent_adaptations:
            return
        
        # Get highest priority task
        task_info = max(self.task_queue, key=lambda x: x["priority"])
        self.task_queue.remove(task_info)
        
        task_id = task_info["task_id"]
        
        # Start adaptation
        adaptation_task = asyncio.create_task(
            self._execute_adaptation(task_info)
        )
        self.active_adaptations[task_id] = adaptation_task
        
        self.logger.info(f"Started processing adaptation task: {task_id}")
    
    async def _execute_adaptation(self, task_info: Dict[str, Any]):
        """Execute adaptation task"""
        task_id = task_info["task_id"]
        
        try:
            experience = await self.meta_engine.adapt_to_task(
                task_info["context"],
                task_info["support_data"],
                task_info["query_data"]
            )
            
            self.adaptation_results[task_id] = experience
            self.controller_stats["tasks_processed"] += 1
            
            # Update average improvement
            if experience.adaptation_success:
                improvement = experience.final_performance - (experience.performance_trajectory[0] if experience.performance_trajectory else 0.3)
                current_avg = self.controller_stats["average_improvement"]
                total_tasks = self.controller_stats["tasks_processed"]
                new_avg = ((current_avg * (total_tasks - 1)) + improvement) / total_tasks
                self.controller_stats["average_improvement"] = new_avg
            
            self.logger.info(f"Completed adaptation task {task_id}: {experience.final_performance:.3f} performance")
            
        except Exception as e:
            self.logger.error(f"Error processing adaptation task {task_id}: {e}")
        
        finally:
            # Clean up
            if task_id in self.active_adaptations:
                del self.active_adaptations[task_id]
            
            # Process next task if available
            await self._process_next_task()
    
    async def get_adaptation_result(self, task_id: str) -> Optional[MetaExperience]:
        """Get adaptation result for task"""
        # Wait for completion if still active
        if task_id in self.active_adaptations:
            try:
                await self.active_adaptations[task_id]
            except:
                pass
        
        return self.adaptation_results.get(task_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get recursive meta-learning system status"""
        meta_insights = self.meta_engine.get_meta_learning_insights()
        
        return {
            "controller_stats": self.controller_stats,
            "meta_engine_insights": meta_insights,
            "queue_size": len(self.task_queue),
            "active_adaptations": len(self.active_adaptations),
            "total_results": len(self.adaptation_results),
            "recursive_improvement_enabled": self.recursive_improvement_enabled
        }


# Export key classes
__all__ = [
    "MetaLearningStrategy", "AdaptationMode", "MetaObjective",
    "MetaExperience", "MetaStrategy", "MetaLearningEngine", "RecursiveMetaController"
]


if __name__ == "__main__":
    # Demo usage
    async def demo():
        logging.basicConfig(level=logging.INFO)
        
        print("🧠 Recursive Meta-Learning Framework Demo")
        print("=" * 60)
        
        # Create recursive meta-learning controller
        controller = RecursiveMetaController()
        
        # Simulate different types of tasks
        tasks = [
            {
                "name": "Image Classification",
                "context": {"task_type": "classification", "domain": "vision", "complexity": 7},
                "support_size": 50,
                "query_size": 20
            },
            {
                "name": "Few-Shot NLP",
                "context": {"task_type": "classification", "domain": "nlp", "complexity": 8},
                "support_size": 10,
                "query_size": 15
            },
            {
                "name": "Reinforcement Learning",
                "context": {"task_type": "rl", "domain": "control", "complexity": 9},
                "support_size": 100,
                "query_size": 50
            },
            {
                "name": "Meta-Learning Task",
                "context": {"task_type": "meta_learning", "domain": "general", "complexity": 10},
                "support_size": 25,
                "query_size": 10
            }
        ]
        
        print(f"📝 Submitting {len(tasks)} adaptation tasks...")
        
        # Submit all tasks
        task_ids = []
        for i, task in enumerate(tasks):
            # Generate dummy data
            support_data = [f"support_sample_{j}" for j in range(task["support_size"])]
            query_data = [f"query_sample_{j}" for j in range(task["query_size"])]
            
            task_id = await controller.submit_adaptation_task(
                task["context"],
                support_data,
                query_data,
                priority=1.0 + i * 0.1
            )
            task_ids.append((task_id, task["name"]))
        
        print(f"⏳ Waiting for adaptations to complete...")
        
        # Wait for all tasks to complete
        results = []
        for task_id, task_name in task_ids:
            result = await controller.get_adaptation_result(task_id)
            if result:
                results.append((task_name, result))
        
        print(f"\n📊 Adaptation Results:")
        for task_name, result in results:
            print(f"  • {task_name}:")
            print(f"    Final Performance: {result.final_performance:.3f}")
            print(f"    Adaptation Time: {result.adaptation_time:.2f}s")
            print(f"    Adaptation Steps: {len(result.adaptation_steps)}")
            print(f"    Success: {'✅' if result.adaptation_success else '❌'}")
            print(f"    Adaptation Speed: {result.calculate_adaptation_speed():.3f}")
        
        # Show system status
        status = controller.get_system_status()
        print(f"\n🎯 System Status:")
        print(f"  Tasks Processed: {status['controller_stats']['tasks_processed']}")
        print(f"  Average Improvement: {status['controller_stats']['average_improvement']:.3f}")
        print(f"  Total Experiences: {status['meta_engine_insights'].get('total_experiences', 0)}")
        print(f"  Recent Success Rate: {status['meta_engine_insights'].get('recent_success_rate', 0):.1%}")
        print(f"  Best Strategy: {status['meta_engine_insights'].get('best_strategy', 'N/A')}")
        
        # Show strategy performance
        strategy_stats = status['meta_engine_insights'].get('strategy_statistics', {})
        if strategy_stats:
            print(f"\n🔧 Strategy Performance:")
            for strategy_name, stats in strategy_stats.items():
                print(f"  • {strategy_name}:")
                print(f"    Effectiveness: {stats['effectiveness_score']:.3f}")
                print(f"    Success Rate: {stats['success_rate']:.1%}")
                print(f"    Avg Time: {stats['avg_adaptation_time']:.2f}s")
        
        print(f"\n✅ Recursive meta-learning framework operational!")
    
    # Run demo
    asyncio.run(demo())