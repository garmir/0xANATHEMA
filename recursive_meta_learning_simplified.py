#!/usr/bin/env python3
"""
Simplified Recursive Meta-Learning Framework for Task-Master
Task 50.2: Design Recursive Meta-Learning Framework (Simplified Version)

This module implements a recursive meta-learning system without external dependencies,
using only Python standard library components.
"""

import json
import logging
import time
import random
import statistics
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetaLearningState:
    """Represents the state of meta-learning at any decision point"""
    task_id: str
    decision_level: int
    context: Dict[str, Any]
    performance_history: List[float]
    adaptation_parameters: Dict[str, float]
    recursive_feedback: List[Dict[str, Any]]
    timestamp: float
    meta_gradient: Optional[float] = None
    
class DecisionPoint:
    """Represents a decision point in the recursive learning process"""
    
    def __init__(self, point_id: str, level: int, parent_id: Optional[str] = None):
        self.point_id = point_id
        self.level = level
        self.parent_id = parent_id
        self.children: List[str] = []
        self.feedback_buffer = deque(maxlen=100)
        self.adaptation_weight = 1.0
        self.performance_score = 0.0
        
    def add_feedback(self, feedback: Dict[str, Any]):
        """Add feedback from subsequent decision points"""
        self.feedback_buffer.append({
            'timestamp': time.time(),
            'feedback': feedback,
            'source_level': feedback.get('level', self.level + 1)
        })
        
    def get_recursive_context(self) -> Dict[str, Any]:
        """Extract context from recursive feedback"""
        if not self.feedback_buffer:
            return {}
            
        recent_feedback = list(self.feedback_buffer)[-10:]  # Last 10 feedback items
        performances = [f['feedback'].get('performance', 0) for f in recent_feedback]
        
        context = {
            'feedback_count': len(recent_feedback),
            'avg_performance': statistics.mean(performances) if performances else 0,
            'trend': self._calculate_trend(recent_feedback),
            'adaptation_suggestions': [f['feedback'].get('adaptation', {}) for f in recent_feedback]
        }
        return context

    def _calculate_trend(self, feedback_list: List[Dict]) -> str:
        """Calculate performance trend from feedback"""
        if len(feedback_list) < 2:
            return "insufficient_data"
            
        performances = [f['feedback'].get('performance', 0) for f in feedback_list]
        if len(performances) < 2:
            return "no_trend"
            
        # Simple slope calculation
        x_vals = list(range(len(performances)))
        n = len(performances)
        sum_x = sum(x_vals)
        sum_y = sum(performances)
        sum_xy = sum(x * y for x, y in zip(x_vals, performances))
        sum_x2 = sum(x * x for x in x_vals)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"

class MetaLearner(ABC):
    """Abstract base class for meta-learning algorithms"""
    
    @abstractmethod
    def learn(self, state: MetaLearningState) -> Dict[str, Any]:
        """Learn from current state and return adaptation parameters"""
        pass
    
    @abstractmethod
    def adapt(self, parameters: Dict[str, Any]) -> None:
        """Apply adaptation parameters to the model"""
        pass

class SimpleMetaLearner(MetaLearner):
    """Simple meta-learning implementation using basic statistics"""
    
    def __init__(self, learning_rate: float = 0.01, adaptation_steps: int = 5):
        self.learning_rate = learning_rate
        self.adaptation_steps = adaptation_steps
        self.adaptation_history = []
        
    def learn(self, state: MetaLearningState) -> Dict[str, Any]:
        """Implement simple meta-learning"""
        # Calculate performance gradient
        if len(state.performance_history) >= 2:
            recent_performances = state.performance_history[-5:]  # Last 5 performances
            if len(recent_performances) >= 2:
                performance_gradient = recent_performances[-1] - recent_performances[0]
            else:
                performance_gradient = 0.0
        else:
            performance_gradient = 0.0
            
        # Compute adaptation parameters
        adaptation_params = {
            'learning_rate_multiplier': 1.0 + performance_gradient * 0.1,
            'regularization_strength': max(0.01, 0.1 - abs(performance_gradient) * 0.05),
            'exploration_factor': 0.1 + max(0, -performance_gradient) * 0.2,
            'performance_gradient': performance_gradient,
            'confidence': min(1.0, len(state.performance_history) / 10.0)
        }
        
        self.adaptation_history.append(adaptation_params)
        logger.info(f"Simple meta-learning adaptation for task {state.task_id}: {adaptation_params}")
        return adaptation_params
    
    def adapt(self, parameters: Dict[str, Any]) -> None:
        """Apply meta-learned parameters"""
        self.learning_rate *= parameters.get('learning_rate_multiplier', 1.0)
        logger.info(f"Adapted learning rate to: {self.learning_rate}")

class RecursiveMetaLearningFramework:
    """Main framework for recursive meta-learning"""
    
    def __init__(self, max_recursion_depth: int = 5, convergence_threshold: float = 0.01):
        self.max_recursion_depth = max_recursion_depth
        self.convergence_threshold = convergence_threshold
        self.decision_points: Dict[str, DecisionPoint] = {}
        self.meta_learners: Dict[str, MetaLearner] = {}
        self.global_state = MetaLearningState(
            task_id="global",
            decision_level=0,
            context={},
            performance_history=[],
            adaptation_parameters={},
            recursive_feedback=[],
            timestamp=time.time()
        )
        self.performance_tracker = []
        self.adaptation_history = []
        
    def register_meta_learner(self, name: str, learner: MetaLearner):
        """Register a meta-learning algorithm"""
        self.meta_learners[name] = learner
        logger.info(f"Registered meta-learner: {name}")
    
    def create_decision_point(self, point_id: str, level: int, parent_id: Optional[str] = None) -> DecisionPoint:
        """Create a new decision point in the hierarchy"""
        if level > self.max_recursion_depth:
            raise ValueError(f"Recursion depth {level} exceeds maximum {self.max_recursion_depth}")
            
        decision_point = DecisionPoint(point_id, level, parent_id)
        self.decision_points[point_id] = decision_point
        
        if parent_id and parent_id in self.decision_points:
            self.decision_points[parent_id].children.append(point_id)
            
        logger.info(f"Created decision point {point_id} at level {level}")
        return decision_point
    
    def execute_recursive_learning(self, task_id: str, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recursive meta-learning for a task"""
        logger.info(f"Starting recursive learning for task: {task_id}")
        
        # Create root decision point
        root_point = self.create_decision_point(f"{task_id}_root", 0)
        
        # Initialize task state
        task_state = MetaLearningState(
            task_id=task_id,
            decision_level=0,
            context=initial_context,
            performance_history=[],
            adaptation_parameters={},
            recursive_feedback=[],
            timestamp=time.time()
        )
        
        # Execute recursive learning cycles
        results = self._recursive_learning_cycle(task_state, root_point, 0)
        
        # Update global state
        self.global_state.performance_history.extend(task_state.performance_history)
        self.global_state.recursive_feedback.append(results)
        
        logger.info(f"Completed recursive learning for task: {task_id}")
        return results
    
    def _recursive_learning_cycle(self, state: MetaLearningState, decision_point: DecisionPoint, depth: int) -> Dict[str, Any]:
        """Execute a single recursive learning cycle"""
        if depth >= self.max_recursion_depth:
            return {"status": "max_depth_reached", "depth": depth, "performance": decision_point.performance_score}
        
        logger.info(f"Recursive cycle at depth {depth} for decision point {decision_point.point_id}")
        
        # Get recursive context from feedback
        recursive_context = decision_point.get_recursive_context()
        state.context.update(recursive_context)
        
        # Apply meta-learning
        adaptation_results = {}
        for name, learner in self.meta_learners.items():
            try:
                adaptation_params = learner.learn(state)
                learner.adapt(adaptation_params)
                adaptation_results[name] = adaptation_params
                logger.info(f"Applied meta-learning with {name}")
            except Exception as e:
                logger.error(f"Meta-learning failed for {name}: {e}")
                adaptation_results[name] = {"error": str(e)}
        
        # Simulate task execution and performance measurement
        performance = self._simulate_task_execution(state, adaptation_results)
        state.performance_history.append(performance)
        decision_point.performance_score = performance
        
        # Generate feedback for parent decision points
        feedback = {
            "performance": performance,
            "level": depth,
            "adaptation_params": adaptation_results,
            "context": state.context.copy(),
            "timestamp": time.time()
        }
        
        # Propagate feedback to parent
        if decision_point.parent_id and decision_point.parent_id in self.decision_points:
            self.decision_points[decision_point.parent_id].add_feedback(feedback)
        
        # Create child decision points for further recursion
        child_results = []
        if depth < self.max_recursion_depth - 1:
            for i in range(min(2, self.max_recursion_depth - depth)):  # Max 2 children per level
                child_id = f"{decision_point.point_id}_child_{i}"
                child_point = self.create_decision_point(child_id, depth + 1, decision_point.point_id)
                
                # Create child state with inherited context
                child_state = MetaLearningState(
                    task_id=f"{state.task_id}_child_{i}",
                    decision_level=depth + 1,
                    context=state.context.copy(),
                    performance_history=state.performance_history.copy(),
                    adaptation_parameters=state.adaptation_parameters.copy(),
                    recursive_feedback=[],
                    timestamp=time.time()
                )
                
                child_result = self._recursive_learning_cycle(child_state, child_point, depth + 1)
                child_results.append(child_result)
                
                # Propagate child feedback back up
                child_feedback = {
                    "child_performance": child_result.get("performance", 0),
                    "child_adaptations": child_result.get("adaptations", {}),
                    "source_child": child_id
                }
                decision_point.add_feedback(child_feedback)
        
        # Check for convergence
        converged = self._check_convergence(state, decision_point)
        
        # Compile results
        results = {
            "task_id": state.task_id,
            "decision_point": decision_point.point_id,
            "depth": depth,
            "performance": performance,
            "adaptations": adaptation_results,
            "context": state.context,
            "converged": converged,
            "child_results": child_results,
            "recursive_feedback_count": len(decision_point.feedback_buffer),
            "timestamp": time.time()
        }
        
        self.adaptation_history.append(results)
        return results
    
    def _simulate_task_execution(self, state: MetaLearningState, adaptations: Dict[str, Any]) -> float:
        """Simulate task execution and return performance score"""
        # Simulate performance based on adaptations and context
        base_performance = 0.7  # Base performance
        
        # Factor in adaptation quality
        adaptation_bonus = 0.0
        for learner_name, params in adaptations.items():
            if isinstance(params, dict) and "error" not in params:
                # Simulate performance improvement from good adaptations
                gradient = params.get('performance_gradient', 0)
                confidence = params.get('confidence', 0.5)
                adaptation_bonus += (gradient * confidence * 0.1)
        
        # Factor in recursive context
        context_bonus = 0.0
        if "avg_performance" in state.context:
            context_bonus = min(0.2, state.context["avg_performance"] * 0.1)
        
        # Add some randomness to simulate real-world variability
        noise = random.gauss(0, 0.05)
        
        performance = base_performance + adaptation_bonus + context_bonus + noise
        performance = max(0.0, min(1.0, performance))  # Clamp to [0, 1]
        
        logger.info(f"Simulated task performance: {performance:.3f}")
        return performance
    
    def _check_convergence(self, state: MetaLearningState, decision_point: DecisionPoint) -> bool:
        """Check if the learning process has converged"""
        if len(state.performance_history) < 5:
            return False
        
        recent_performance = state.performance_history[-5:]
        performance_variance = statistics.variance(recent_performance) if len(recent_performance) > 1 else 1.0
        
        converged = performance_variance < self.convergence_threshold
        if converged:
            logger.info(f"Convergence detected for {decision_point.point_id}: variance={performance_variance:.4f}")
        
        return converged
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get current status of the framework"""
        avg_performance = statistics.mean(self.global_state.performance_history) if self.global_state.performance_history else 0.0
        
        return {
            "total_decision_points": len(self.decision_points),
            "registered_meta_learners": list(self.meta_learners.keys()),
            "global_performance_history": self.global_state.performance_history,
            "adaptation_cycles": len(self.adaptation_history),
            "average_performance": avg_performance,
            "framework_uptime": time.time() - self.global_state.timestamp
        }
    
    def save_state(self, filepath: str):
        """Save framework state to file"""
        # Convert decision points to serializable format
        decision_points_data = {}
        for point_id, point in self.decision_points.items():
            decision_points_data[point_id] = {
                'point_id': point.point_id,
                'level': point.level,
                'parent_id': point.parent_id,
                'children': point.children,
                'performance_score': point.performance_score,
                'feedback_buffer': list(point.feedback_buffer)
            }
        
        state_data = {
            "decision_points": decision_points_data,
            "global_state": asdict(self.global_state),
            "adaptation_history": self.adaptation_history,
            "performance_tracker": self.performance_tracker,
            "framework_config": {
                "max_recursion_depth": self.max_recursion_depth,
                "convergence_threshold": self.convergence_threshold
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        logger.info(f"Framework state saved to {filepath}")

def create_task_master_meta_learning_system() -> RecursiveMetaLearningFramework:
    """Factory function to create a configured meta-learning system for Task-Master"""
    framework = RecursiveMetaLearningFramework(max_recursion_depth=4, convergence_threshold=0.005)
    
    # Register meta-learners
    simple_learner = SimpleMetaLearner(learning_rate=0.01, adaptation_steps=5)
    framework.register_meta_learner("simple_meta_learner", simple_learner)
    
    logger.info("Task-Master recursive meta-learning system created and configured")
    return framework

def test_framework():
    """Test the recursive meta-learning framework"""
    print("Testing Recursive Meta-Learning Framework...")
    
    # Create and test the framework
    framework = create_task_master_meta_learning_system()
    
    # Test with a sample task
    test_context = {
        "task_type": "code_optimization",
        "complexity": "medium",
        "previous_performance": 0.75,
        "resource_constraints": {"memory": "limited", "time": "moderate"}
    }
    
    logger.info("Starting recursive meta-learning test...")
    results = framework.execute_recursive_learning("test_task_001", test_context)
    
    # Display results
    print("\nRecursive Meta-Learning Results:")
    print(f"Task ID: {results['task_id']}")
    print(f"Final Performance: {results['performance']:.3f}")
    print(f"Recursion Depth: {results['depth']}")
    print(f"Converged: {results['converged']}")
    print(f"Child Results: {len(results['child_results'])}")
    
    # Show framework status
    status = framework.get_framework_status()
    print(f"\nFramework Status:")
    print(f"Decision Points: {status['total_decision_points']}")
    print(f"Meta-Learners: {status['registered_meta_learners']}")
    print(f"Average Performance: {status['average_performance']:.3f}")
    print(f"Adaptation Cycles: {status['adaptation_cycles']}")
    
    # Save state for persistence
    framework.save_state(".taskmaster/meta_learning_state.json")
    print("\nFramework state saved successfully")
    
    return framework

# Example usage and testing
if __name__ == "__main__":
    test_framework()