#!/usr/bin/env python3
"""
Integrated Meta-Learning and Neural Architecture Search System
Task 50.3: Integration of recursive meta-learning with NAS module

This module combines the recursive meta-learning framework with the Neural Architecture Search
system to create a unified self-improving architecture for Task-Master.
"""

import json
import logging
import time
import random
import statistics
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import copy

# Import our framework components
from recursive_meta_learning_simplified import (
    RecursiveMetaLearningFramework, 
    MetaLearningState, 
    DecisionPoint,
    create_task_master_meta_learning_system
)
from neural_architecture_search import (
    NeuralArchitectureSearch, 
    Architecture, 
    LayerConfig, 
    LayerType, 
    ActivationType,
    create_task_master_nas_system
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntegratedSearchState:
    """State for integrated meta-learning and NAS search"""
    task_id: str
    meta_learning_state: MetaLearningState
    nas_search_context: Dict[str, Any]
    discovered_architectures: List[Architecture]
    performance_history: List[float]
    optimization_cycle: int
    convergence_threshold: float = 0.01
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class MetaNASOptimizer:
    """Optimizer that uses meta-learning to guide NAS decisions"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.optimization_history = []
        self.architecture_preferences = {}
        self.performance_predictions = {}
        
    def optimize_search_strategy(self, search_state: IntegratedSearchState) -> Dict[str, Any]:
        """Use meta-learning insights to optimize NAS search strategy"""
        
        # Analyze meta-learning performance trends
        meta_performance = search_state.meta_learning_state.performance_history
        if len(meta_performance) < 2:
            return self._default_search_strategy()
        
        # Calculate performance trend
        recent_trend = statistics.mean(meta_performance[-3:]) - statistics.mean(meta_performance[-6:-3]) if len(meta_performance) >= 6 else 0
        
        # Adapt search strategy based on meta-learning insights
        strategy = {
            "population_size": 15,
            "max_generations": 8,
            "mutation_rate": 0.3,
            "exploration_factor": 0.5,
            "complexity_penalty": 0.1
        }
        
        # Adjust based on performance trend
        if recent_trend > 0.1:
            # Performance improving - be more aggressive
            strategy["mutation_rate"] = 0.4
            strategy["exploration_factor"] = 0.7
            strategy["max_generations"] = 10
        elif recent_trend < -0.1:
            # Performance declining - be more conservative
            strategy["mutation_rate"] = 0.2
            strategy["exploration_factor"] = 0.3
            strategy["complexity_penalty"] = 0.2
        
        # Learn from successful architectures
        if search_state.discovered_architectures:
            best_arch = max(search_state.discovered_architectures, key=lambda a: a.performance_score)
            self._update_architecture_preferences(best_arch)
            strategy.update(self._extract_preferences_as_constraints())
        
        logger.info(f"Optimized search strategy: {strategy}")
        return strategy
    
    def _default_search_strategy(self) -> Dict[str, Any]:
        """Default search strategy when no meta-learning history is available"""
        return {
            "population_size": 15,
            "max_generations": 8,
            "mutation_rate": 0.3,
            "exploration_factor": 0.5,
            "complexity_penalty": 0.1
        }
    
    def _update_architecture_preferences(self, architecture: Architecture):
        """Update architecture preferences based on successful architectures"""
        # Track successful layer types
        for layer in architecture.layers:
            layer_type = layer.layer_type.value
            if layer_type not in self.architecture_preferences:
                self.architecture_preferences[layer_type] = {"count": 0, "performance_sum": 0.0}
            
            self.architecture_preferences[layer_type]["count"] += 1
            self.architecture_preferences[layer_type]["performance_sum"] += architecture.performance_score
        
        # Track successful complexity ranges
        complexity = architecture.complexity_score
        if "complexity_range" not in self.architecture_preferences:
            self.architecture_preferences["complexity_range"] = []
        self.architecture_preferences["complexity_range"].append(complexity)
    
    def _extract_preferences_as_constraints(self) -> Dict[str, Any]:
        """Extract learned preferences as search constraints"""
        constraints = {}
        
        # Preferred layer types based on success rate
        if self.architecture_preferences:
            layer_preferences = {}
            for layer_type, stats in self.architecture_preferences.items():
                if layer_type != "complexity_range" and stats["count"] > 0:
                    avg_performance = stats["performance_sum"] / stats["count"]
                    layer_preferences[layer_type] = avg_performance
            
            # Sort by performance and get top performers
            if layer_preferences:
                sorted_layers = sorted(layer_preferences.items(), key=lambda x: x[1], reverse=True)
                top_layers = [layer for layer, _ in sorted_layers[:5]]  # Top 5 layer types
                constraints["preferred_layer_types"] = top_layers
        
        # Preferred complexity range
        if "complexity_range" in self.architecture_preferences:
            complexities = self.architecture_preferences["complexity_range"]
            if complexities:
                avg_complexity = statistics.mean(complexities)
                std_complexity = statistics.stdev(complexities) if len(complexities) > 1 else 0.1
                constraints["preferred_complexity_range"] = (
                    max(0.1, avg_complexity - std_complexity),
                    avg_complexity + std_complexity
                )
        
        return constraints

class IntegratedMetaNASSystem:
    """Main system integrating recursive meta-learning with Neural Architecture Search"""
    
    def __init__(self, max_optimization_cycles: int = 5):
        self.max_optimization_cycles = max_optimization_cycles
        
        # Initialize core components
        self.meta_learning_framework = create_task_master_meta_learning_system()
        self.nas_system = create_task_master_nas_system()
        self.meta_nas_optimizer = MetaNASOptimizer()
        
        # System state
        self.search_history = []
        self.integration_statistics = {
            "total_cycles": 0,
            "architectures_discovered": 0,
            "performance_improvements": 0,
            "convergence_achieved": False
        }
        
    def optimize_task_architecture(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Main optimization loop integrating meta-learning and NAS"""
        logger.info(f"Starting integrated optimization for task: {task_context.get('task_type', 'unknown')}")
        
        # Initialize integrated search state
        initial_meta_state = MetaLearningState(
            task_id=task_context.get("task_id", "integrated_task"),
            decision_level=0,
            context=task_context,
            performance_history=[],
            adaptation_parameters={},
            recursive_feedback=[],
            timestamp=time.time()
        )
        
        search_state = IntegratedSearchState(
            task_id=task_context.get("task_id", "integrated_task"),
            meta_learning_state=initial_meta_state,
            nas_search_context=task_context,
            discovered_architectures=[],
            performance_history=[],
            optimization_cycle=0
        )
        
        best_overall_architecture = None
        best_overall_performance = 0.0
        
        # Main optimization loop
        for cycle in range(self.max_optimization_cycles):
            search_state.optimization_cycle = cycle
            logger.info(f"Optimization cycle {cycle + 1}/{self.max_optimization_cycles}")
            
            # Step 1: Run meta-learning to understand task characteristics
            meta_results = self.meta_learning_framework.execute_recursive_learning(
                task_id=f"{search_state.task_id}_cycle_{cycle}",
                initial_context=task_context
            )
            
            # Update meta-learning state
            search_state.meta_learning_state.performance_history.extend(
                [meta_results.get("performance", 0.0)]
            )
            search_state.meta_learning_state.recursive_feedback.append(meta_results)
            
            # Step 2: Use meta-learning insights to optimize NAS strategy
            optimized_strategy = self.meta_nas_optimizer.optimize_search_strategy(search_state)
            
            # Step 3: Execute NAS with optimized strategy
            nas_constraints = self._convert_strategy_to_constraints(optimized_strategy, task_context)
            cycle_best_architecture = self.nas_system.search_architecture(task_context, nas_constraints)
            
            # Step 4: Evaluate integrated performance
            integrated_performance = self._evaluate_integrated_performance(
                cycle_best_architecture, meta_results, task_context
            )
            
            # Update search state
            search_state.discovered_architectures.append(cycle_best_architecture)
            search_state.performance_history.append(integrated_performance)
            
            # Track best overall performance
            if integrated_performance > best_overall_performance:
                best_overall_performance = integrated_performance
                best_overall_architecture = cycle_best_architecture
            
            logger.info(f"Cycle {cycle + 1} integrated performance: {integrated_performance:.3f}")
            
            # Step 5: Check for convergence
            if self._check_integration_convergence(search_state):
                logger.info(f"Integration converged after {cycle + 1} cycles")
                self.integration_statistics["convergence_achieved"] = True
                break
            
            # Step 6: Adapt for next cycle
            self._adapt_for_next_cycle(search_state, meta_results, cycle_best_architecture)
        
        # Compile final results
        final_results = self._compile_final_results(
            search_state, best_overall_architecture, best_overall_performance
        )
        
        # Update statistics
        self.integration_statistics["total_cycles"] = search_state.optimization_cycle + 1
        self.integration_statistics["architectures_discovered"] = len(search_state.discovered_architectures)
        self.integration_statistics["performance_improvements"] = sum(
            1 for i in range(1, len(search_state.performance_history))
            if search_state.performance_history[i] > search_state.performance_history[i-1]
        )
        
        # Store search history
        self.search_history.append(final_results)
        
        logger.info(f"Integrated optimization completed. Best performance: {best_overall_performance:.3f}")
        return final_results
    
    def _convert_strategy_to_constraints(self, strategy: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert optimization strategy to NAS constraints"""
        constraints = {
            "max_complexity": 1.0 - strategy.get("complexity_penalty", 0.1),
            "preferred_layer_types": [LayerType.DENSE, LayerType.LSTM, LayerType.DROPOUT],
            "max_layers": 10
        }
        
        # Add strategy-specific constraints
        if "preferred_layer_types" in strategy:
            # Convert string layer types to LayerType enums
            preferred_types = []
            for layer_str in strategy["preferred_layer_types"]:
                try:
                    layer_type = LayerType(layer_str)
                    preferred_types.append(layer_type)
                except ValueError:
                    continue
            if preferred_types:
                constraints["preferred_layer_types"] = preferred_types
        
        if "preferred_complexity_range" in strategy:
            min_complexity, max_complexity = strategy["preferred_complexity_range"]
            constraints["max_complexity"] = max_complexity
            constraints["min_complexity"] = min_complexity
        
        # Task-specific constraints
        task_type = task_context.get("task_type", "general")
        if task_type == "sequence":
            constraints["preferred_layer_types"] = [LayerType.LSTM, LayerType.GRU, LayerType.ATTENTION, LayerType.DENSE]
        elif task_type == "vision":
            constraints["preferred_layer_types"] = [LayerType.CONV1D, LayerType.CONV2D, LayerType.DENSE]
        
        return constraints
    
    def _evaluate_integrated_performance(self, architecture: Architecture, 
                                       meta_results: Dict[str, Any], 
                                       task_context: Dict[str, Any]) -> float:
        """Evaluate combined performance of architecture and meta-learning"""
        # Base architecture performance
        arch_performance = architecture.performance_score
        
        # Meta-learning performance contribution
        meta_performance = meta_results.get("performance", 0.5)
        
        # Integration bonus based on alignment
        integration_bonus = self._calculate_integration_bonus(architecture, meta_results, task_context)
        
        # Combined performance with weighted components
        integrated_performance = (
            0.6 * arch_performance +    # Architecture performance (primary)
            0.3 * meta_performance +    # Meta-learning performance 
            0.1 * integration_bonus     # Integration synergy
        )
        
        return min(1.0, integrated_performance)
    
    def _calculate_integration_bonus(self, architecture: Architecture, 
                                   meta_results: Dict[str, Any], 
                                   task_context: Dict[str, Any]) -> float:
        """Calculate bonus for good integration between architecture and meta-learning"""
        bonus = 0.0
        
        # Complexity alignment bonus
        meta_adaptations = meta_results.get("adaptations", {})
        arch_complexity = architecture.complexity_score
        
        # Check if architecture complexity matches meta-learning suggestions
        for learner_name, adaptations in meta_adaptations.items():
            if isinstance(adaptations, dict):
                exploration_factor = adaptations.get("exploration_factor", 0.1)
                if exploration_factor > 0.4 and arch_complexity > 0.5:
                    bonus += 0.2  # High exploration suggests complex architecture is good
                elif exploration_factor < 0.2 and arch_complexity < 0.3:
                    bonus += 0.2  # Low exploration suggests simple architecture is good
        
        # Convergence alignment bonus
        meta_converged = meta_results.get("converged", False)
        arch_efficiency = architecture.efficiency_score
        
        if meta_converged and arch_efficiency > 1.0:
            bonus += 0.3  # Both systems performing well
        
        # Task type alignment bonus
        task_type = task_context.get("task_type", "general")
        if task_type == "sequence":
            sequence_layers = sum(1 for layer in architecture.layers 
                                if layer.layer_type in [LayerType.LSTM, LayerType.GRU])
            bonus += min(0.2, sequence_layers * 0.05)
        
        return min(1.0, bonus)
    
    def _check_integration_convergence(self, search_state: IntegratedSearchState) -> bool:
        """Check if the integrated optimization has converged"""
        if len(search_state.performance_history) < 3:
            return False
        
        # Check performance stability
        recent_performances = search_state.performance_history[-3:]
        performance_variance = statistics.variance(recent_performances)
        
        if performance_variance < search_state.convergence_threshold:
            return True
        
        # Check meta-learning convergence
        meta_converged = False
        if search_state.meta_learning_state.recursive_feedback:
            latest_meta = search_state.meta_learning_state.recursive_feedback[-1]
            meta_converged = latest_meta.get("converged", False)
        
        # Check NAS convergence (high performance achieved)
        nas_converged = max(search_state.performance_history) > 0.9
        
        return meta_converged and nas_converged
    
    def _adapt_for_next_cycle(self, search_state: IntegratedSearchState, 
                            meta_results: Dict[str, Any], 
                            architecture: Architecture):
        """Adapt parameters for the next optimization cycle"""
        # Update task context with learned insights
        current_performance = search_state.performance_history[-1]
        
        # Adjust complexity preferences
        if current_performance > 0.8:
            # Good performance, can try more complex architectures
            search_state.nas_search_context["complexity_preference"] = "increase"
        elif current_performance < 0.6:
            # Poor performance, try simpler architectures
            search_state.nas_search_context["complexity_preference"] = "decrease"
        
        # Update convergence threshold based on progress
        if len(search_state.performance_history) >= 2:
            improvement = search_state.performance_history[-1] - search_state.performance_history[-2]
            if improvement < 0.05:
                # Slow improvement, relax convergence threshold
                search_state.convergence_threshold *= 1.2
            else:
                # Good improvement, maintain or tighten threshold
                search_state.convergence_threshold *= 0.9
        
        # Add cycle information to meta-learning context
        search_state.meta_learning_state.context["optimization_cycle"] = search_state.optimization_cycle
        search_state.meta_learning_state.context["architecture_complexity"] = architecture.complexity_score
        search_state.meta_learning_state.context["integration_performance"] = current_performance
    
    def _compile_final_results(self, search_state: IntegratedSearchState, 
                             best_architecture: Architecture, 
                             best_performance: float) -> Dict[str, Any]:
        """Compile comprehensive final results"""
        results = {
            "task_id": search_state.task_id,
            "optimization_cycles": search_state.optimization_cycle + 1,
            "best_architecture": {
                "id": best_architecture.arch_id,
                "performance": best_architecture.performance_score,
                "complexity": best_architecture.complexity_score,
                "efficiency": best_architecture.efficiency_score,
                "layers": [layer.to_dict() for layer in best_architecture.layers]
            },
            "best_integrated_performance": best_performance,
            "performance_history": search_state.performance_history,
            "meta_learning_summary": {
                "total_adaptations": len(search_state.meta_learning_state.recursive_feedback),
                "final_performance": search_state.meta_learning_state.performance_history[-1] if search_state.meta_learning_state.performance_history else 0,
                "convergence_achieved": search_state.meta_learning_state.recursive_feedback[-1].get("converged", False) if search_state.meta_learning_state.recursive_feedback else False
            },
            "nas_summary": {
                "architectures_evaluated": len(search_state.discovered_architectures),
                "average_performance": statistics.mean([arch.performance_score for arch in search_state.discovered_architectures]),
                "complexity_range": (
                    min(arch.complexity_score for arch in search_state.discovered_architectures),
                    max(arch.complexity_score for arch in search_state.discovered_architectures)
                ) if search_state.discovered_architectures else (0, 0)
            },
            "integration_statistics": self.integration_statistics.copy(),
            "optimization_recommendations": self._generate_optimization_recommendations(search_state),
            "timestamp": time.time()
        }
        
        return results
    
    def _generate_optimization_recommendations(self, search_state: IntegratedSearchState) -> List[str]:
        """Generate recommendations for future optimization"""
        recommendations = []
        
        # Performance trend analysis
        if len(search_state.performance_history) >= 2:
            trend = search_state.performance_history[-1] - search_state.performance_history[0]
            if trend > 0.2:
                recommendations.append("Excellent optimization progress - consider increasing complexity for further gains")
            elif trend < 0.1:
                recommendations.append("Limited improvement - consider adjusting search strategy or constraints")
        
        # Architecture diversity analysis
        if search_state.discovered_architectures:
            complexity_values = [arch.complexity_score for arch in search_state.discovered_architectures]
            complexity_variance = statistics.variance(complexity_values) if len(complexity_values) > 1 else 0
            
            if complexity_variance < 0.01:
                recommendations.append("Low architecture diversity - increase exploration parameters")
            elif complexity_variance > 0.5:
                recommendations.append("High architecture diversity - consider focusing search on promising regions")
        
        # Meta-learning integration analysis
        if search_state.meta_learning_state.recursive_feedback:
            latest_meta = search_state.meta_learning_state.recursive_feedback[-1]
            if latest_meta.get("converged", False):
                recommendations.append("Meta-learning converged - architecture search can be more focused")
            else:
                recommendations.append("Meta-learning still adapting - continue exploration-heavy search")
        
        # Convergence analysis
        if not self.integration_statistics["convergence_achieved"]:
            recommendations.append("Consider increasing max_optimization_cycles for better convergence")
        
        return recommendations
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "integration_statistics": self.integration_statistics,
            "meta_learning_status": self.meta_learning_framework.get_framework_status(),
            "nas_status": self.nas_system.get_search_statistics(),
            "search_history_count": len(self.search_history),
            "optimizer_preferences": self.meta_nas_optimizer.architecture_preferences
        }
    
    def save_system_state(self, filepath: str):
        """Save complete system state"""
        system_state = {
            "integration_statistics": self.integration_statistics,
            "search_history": self.search_history,
            "optimizer_preferences": self.meta_nas_optimizer.architecture_preferences,
            "meta_learning_framework_status": self.meta_learning_framework.get_framework_status(),
            "nas_system_status": self.nas_system.get_search_statistics(),
            "timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(system_state, f, indent=2, default=str)
        
        logger.info(f"Integrated system state saved to {filepath}")

def create_integrated_meta_nas_system() -> IntegratedMetaNASSystem:
    """Factory function to create the integrated system"""
    system = IntegratedMetaNASSystem(max_optimization_cycles=4)  # Reasonable for testing
    logger.info("Integrated Meta-Learning and NAS system created")
    return system

def test_integrated_system():
    """Test the integrated meta-learning and NAS system"""
    print("Testing Integrated Meta-Learning and NAS System...")
    
    # Create integrated system
    system = create_integrated_meta_nas_system()
    
    # Define test task
    task_context = {
        "task_id": "integrated_test_001",
        "task_type": "sequence",
        "complexity": "medium",
        "data_size": "large",
        "performance_target": 0.85,
        "resource_constraints": {"memory": "limited", "time": "moderate"}
    }
    
    # Run integrated optimization
    logger.info("Starting integrated optimization test...")
    results = system.optimize_task_architecture(task_context)
    
    # Display results
    print(f"\nIntegrated Optimization Results:")
    print(f"Task ID: {results['task_id']}")
    print(f"Optimization Cycles: {results['optimization_cycles']}")
    print(f"Best Integrated Performance: {results['best_integrated_performance']:.3f}")
    print(f"Best Architecture ID: {results['best_architecture']['id']}")
    print(f"Architecture Performance: {results['best_architecture']['performance']:.3f}")
    print(f"Architecture Complexity: {results['best_architecture']['complexity']:.3f}")
    print(f"Architecture Efficiency: {results['best_architecture']['efficiency']:.3f}")
    
    # Show meta-learning summary
    meta_summary = results['meta_learning_summary']
    print(f"\nMeta-Learning Summary:")
    print(f"Total Adaptations: {meta_summary['total_adaptations']}")
    print(f"Final Performance: {meta_summary['final_performance']:.3f}")
    print(f"Convergence Achieved: {meta_summary['convergence_achieved']}")
    
    # Show NAS summary
    nas_summary = results['nas_summary']
    print(f"\nNAS Summary:")
    print(f"Architectures Evaluated: {nas_summary['architectures_evaluated']}")
    print(f"Average Performance: {nas_summary['average_performance']:.3f}")
    print(f"Complexity Range: {nas_summary['complexity_range'][0]:.3f} - {nas_summary['complexity_range'][1]:.3f}")
    
    # Show recommendations
    print(f"\nOptimization Recommendations:")
    for i, rec in enumerate(results['optimization_recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Show system status
    status = system.get_system_status()
    print(f"\nSystem Status:")
    print(f"Integration Statistics: {status['integration_statistics']}")
    
    # Save system state
    system.save_system_state(".taskmaster/integrated_system_state.json")
    print(f"\nIntegrated system state saved successfully")
    
    return system, results

if __name__ == "__main__":
    test_integrated_system()