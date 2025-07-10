#!/usr/bin/env python3
"""
Neural Architecture Search (NAS) Module for Self-Improving Task-Master
Task 50.3: Integrate Neural Architecture Search (NAS) Module

This module implements a Neural Architecture Search system that automatically
discovers and optimizes neural network architectures within the meta-learning framework.
"""

import json
import logging
import time
import random
import statistics
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayerType(Enum):
    """Available layer types for architecture search"""
    DENSE = "dense"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"
    TRANSFORMER = "transformer"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    RESIDUAL = "residual"

class ActivationType(Enum):
    """Available activation functions"""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    GELU = "gelu"
    SWISH = "swish"
    LEAKY_RELU = "leaky_relu"

@dataclass
class LayerConfig:
    """Configuration for a neural network layer"""
    layer_type: LayerType
    units: Optional[int] = None
    activation: Optional[ActivationType] = None
    dropout_rate: Optional[float] = None
    kernel_size: Optional[int] = None
    filters: Optional[int] = None
    attention_heads: Optional[int] = None
    layer_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "layer_type": self.layer_type.value,
            "units": self.units,
            "activation": self.activation.value if self.activation else None,
            "dropout_rate": self.dropout_rate,
            "kernel_size": self.kernel_size,
            "filters": self.filters,
            "attention_heads": self.attention_heads,
            "layer_id": self.layer_id
        }

@dataclass
class Architecture:
    """Represents a complete neural network architecture"""
    arch_id: str
    layers: List[LayerConfig]
    performance_score: float = 0.0
    complexity_score: float = 0.0
    efficiency_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = None
    
    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
    
    def calculate_complexity(self) -> float:
        """Calculate architecture complexity score"""
        complexity = 0.0
        for layer in self.layers:
            if layer.units:
                complexity += layer.units * 0.001
            if layer.filters:
                complexity += layer.filters * 0.002
            if layer.attention_heads:
                complexity += layer.attention_heads * 0.05
            if layer.layer_type in [LayerType.LSTM, LayerType.GRU]:
                complexity += 0.1
            if layer.layer_type == LayerType.TRANSFORMER:
                complexity += 0.2
        
        self.complexity_score = complexity
        return complexity
    
    def calculate_efficiency(self) -> float:
        """Calculate efficiency score (performance vs complexity)"""
        if self.complexity_score == 0:
            self.calculate_complexity()
        
        if self.complexity_score > 0:
            self.efficiency_score = self.performance_score / self.complexity_score
        else:
            self.efficiency_score = self.performance_score
        
        return self.efficiency_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "arch_id": self.arch_id,
            "layers": [layer.to_dict() for layer in self.layers],
            "performance_score": self.performance_score,
            "complexity_score": self.complexity_score,
            "efficiency_score": self.efficiency_score,
            "generation": self.generation,
            "parent_ids": self.parent_ids
        }

class ArchitectureGenerator(ABC):
    """Abstract base class for architecture generation strategies"""
    
    @abstractmethod
    def generate_architecture(self, constraints: Dict[str, Any]) -> Architecture:
        """Generate a new architecture based on constraints"""
        pass
    
    @abstractmethod
    def mutate_architecture(self, parent: Architecture, mutation_rate: float = 0.3) -> Architecture:
        """Create a mutated version of an existing architecture"""
        pass

class RandomArchitectureGenerator(ArchitectureGenerator):
    """Random architecture generation strategy"""
    
    def __init__(self, max_layers: int = 10, min_layers: int = 2):
        self.max_layers = max_layers
        self.min_layers = min_layers
        self.architecture_counter = 0
    
    def generate_architecture(self, constraints: Dict[str, Any]) -> Architecture:
        """Generate a random architecture"""
        self.architecture_counter += 1
        
        # Determine number of layers
        num_layers = random.randint(self.min_layers, self.max_layers)
        
        # Extract constraints
        max_complexity = constraints.get("max_complexity", 1.0)
        preferred_types = constraints.get("preferred_layer_types", list(LayerType))
        task_type = constraints.get("task_type", "general")
        
        layers = []
        for i in range(num_layers):
            layer = self._generate_random_layer(i, task_type, preferred_types)
            layers.append(layer)
        
        arch = Architecture(
            arch_id=f"random_arch_{self.architecture_counter}",
            layers=layers,
            generation=0
        )
        
        # Ensure complexity constraint is met
        complexity = arch.calculate_complexity()
        if complexity > max_complexity:
            # Simplify architecture
            arch = self._simplify_architecture(arch, max_complexity)
        
        logger.info(f"Generated random architecture {arch.arch_id} with {len(layers)} layers")
        return arch
    
    def _generate_random_layer(self, layer_index: int, task_type: str, preferred_types: List[LayerType]) -> LayerConfig:
        """Generate a random layer configuration"""
        # Choose layer type based on task type and preferences
        if task_type == "sequence" and layer_index < 3:
            # Prefer sequential layers for sequence tasks
            layer_type = random.choice([LayerType.LSTM, LayerType.GRU, LayerType.ATTENTION])
        elif task_type == "vision" and layer_index < 5:
            # Prefer convolutional layers for vision tasks
            layer_type = random.choice([LayerType.CONV1D, LayerType.CONV2D])
        else:
            layer_type = random.choice(preferred_types)
        
        layer_config = LayerConfig(
            layer_type=layer_type,
            layer_id=f"layer_{layer_index}"
        )
        
        # Configure layer-specific parameters
        if layer_type in [LayerType.DENSE, LayerType.LSTM, LayerType.GRU]:
            layer_config.units = random.choice([32, 64, 128, 256, 512])
            layer_config.activation = random.choice(list(ActivationType))
        
        elif layer_type in [LayerType.CONV1D, LayerType.CONV2D]:
            layer_config.filters = random.choice([16, 32, 64, 128])
            layer_config.kernel_size = random.choice([3, 5, 7])
            layer_config.activation = random.choice([ActivationType.RELU, ActivationType.GELU])
        
        elif layer_type == LayerType.ATTENTION:
            layer_config.attention_heads = random.choice([4, 8, 12, 16])
            layer_config.units = random.choice([64, 128, 256])
        
        elif layer_type == LayerType.TRANSFORMER:
            layer_config.attention_heads = random.choice([4, 8, 12])
            layer_config.units = random.choice([128, 256, 512])
        
        elif layer_type == LayerType.DROPOUT:
            layer_config.dropout_rate = random.uniform(0.1, 0.5)
        
        return layer_config
    
    def _simplify_architecture(self, arch: Architecture, max_complexity: float) -> Architecture:
        """Simplify architecture to meet complexity constraints"""
        simplified_layers = []
        current_complexity = 0.0
        
        for layer in arch.layers:
            layer_complexity = self._estimate_layer_complexity(layer)
            if current_complexity + layer_complexity <= max_complexity:
                simplified_layers.append(layer)
                current_complexity += layer_complexity
            else:
                # Try to add a simpler version
                simple_layer = self._simplify_layer(layer)
                simple_complexity = self._estimate_layer_complexity(simple_layer)
                if current_complexity + simple_complexity <= max_complexity:
                    simplified_layers.append(simple_layer)
                    current_complexity += simple_complexity
        
        arch.layers = simplified_layers
        arch.calculate_complexity()
        return arch
    
    def _estimate_layer_complexity(self, layer: LayerConfig) -> float:
        """Estimate the complexity contribution of a single layer"""
        complexity = 0.0
        if layer.units:
            complexity += layer.units * 0.001
        if layer.filters:
            complexity += layer.filters * 0.002
        if layer.attention_heads:
            complexity += layer.attention_heads * 0.05
        if layer.layer_type in [LayerType.LSTM, LayerType.GRU]:
            complexity += 0.1
        if layer.layer_type == LayerType.TRANSFORMER:
            complexity += 0.2
        return complexity
    
    def _simplify_layer(self, layer: LayerConfig) -> LayerConfig:
        """Create a simpler version of a layer"""
        simple_layer = copy.deepcopy(layer)
        
        if simple_layer.units and simple_layer.units > 64:
            simple_layer.units = simple_layer.units // 2
        if simple_layer.filters and simple_layer.filters > 32:
            simple_layer.filters = simple_layer.filters // 2
        if simple_layer.attention_heads and simple_layer.attention_heads > 4:
            simple_layer.attention_heads = simple_layer.attention_heads // 2
        
        return simple_layer
    
    def mutate_architecture(self, parent: Architecture, mutation_rate: float = 0.3) -> Architecture:
        """Create a mutated version of an existing architecture"""
        self.architecture_counter += 1
        
        mutated_layers = []
        for layer in parent.layers:
            if random.random() < mutation_rate:
                # Mutate this layer
                mutated_layer = self._mutate_layer(layer)
                mutated_layers.append(mutated_layer)
            else:
                # Keep original layer
                mutated_layers.append(copy.deepcopy(layer))
        
        # Occasionally add or remove layers
        if random.random() < 0.2 and len(mutated_layers) < self.max_layers:
            # Add a new layer
            new_layer = self._generate_random_layer(len(mutated_layers), "general", list(LayerType))
            insert_pos = random.randint(0, len(mutated_layers))
            mutated_layers.insert(insert_pos, new_layer)
        
        elif random.random() < 0.1 and len(mutated_layers) > self.min_layers:
            # Remove a layer
            remove_pos = random.randint(0, len(mutated_layers) - 1)
            mutated_layers.pop(remove_pos)
        
        mutated_arch = Architecture(
            arch_id=f"mutated_arch_{self.architecture_counter}",
            layers=mutated_layers,
            generation=parent.generation + 1,
            parent_ids=[parent.arch_id]
        )
        
        logger.info(f"Mutated architecture {parent.arch_id} -> {mutated_arch.arch_id}")
        return mutated_arch
    
    def _mutate_layer(self, layer: LayerConfig) -> LayerConfig:
        """Mutate a single layer configuration"""
        mutated = copy.deepcopy(layer)
        
        # Mutate layer type occasionally
        if random.random() < 0.1:
            if layer.layer_type in [LayerType.LSTM, LayerType.GRU]:
                mutated.layer_type = random.choice([LayerType.LSTM, LayerType.GRU])
            elif layer.layer_type in [LayerType.CONV1D, LayerType.CONV2D]:
                mutated.layer_type = random.choice([LayerType.CONV1D, LayerType.CONV2D])
        
        # Mutate parameters
        if mutated.units:
            mutation_factor = random.uniform(0.5, 2.0)
            mutated.units = max(16, int(mutated.units * mutation_factor))
        
        if mutated.filters:
            mutation_factor = random.uniform(0.5, 2.0)
            mutated.filters = max(8, int(mutated.filters * mutation_factor))
        
        if mutated.attention_heads:
            mutated.attention_heads = random.choice([4, 8, 12, 16])
        
        if mutated.activation and random.random() < 0.3:
            mutated.activation = random.choice(list(ActivationType))
        
        if mutated.dropout_rate and random.random() < 0.3:
            mutated.dropout_rate = random.uniform(0.1, 0.5)
        
        return mutated

class PerformanceEvaluator:
    """Evaluates architecture performance for different tasks"""
    
    def __init__(self):
        self.evaluation_cache = {}
        self.evaluation_history = []
    
    def evaluate_architecture(self, architecture: Architecture, task_context: Dict[str, Any]) -> float:
        """Evaluate architecture performance for a given task"""
        # Check cache first
        cache_key = f"{architecture.arch_id}_{hash(str(task_context))}"
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        # Simulate performance evaluation
        performance = self._simulate_performance(architecture, task_context)
        
        # Cache result
        self.evaluation_cache[cache_key] = performance
        self.evaluation_history.append({
            "arch_id": architecture.arch_id,
            "task_context": task_context,
            "performance": performance,
            "timestamp": time.time()
        })
        
        # Update architecture performance
        architecture.performance_score = performance
        architecture.calculate_efficiency()
        
        logger.info(f"Evaluated architecture {architecture.arch_id}: performance={performance:.3f}")
        return performance
    
    def _simulate_performance(self, architecture: Architecture, task_context: Dict[str, Any]) -> float:
        """Simulate architecture performance based on configuration and task"""
        base_performance = 0.5
        
        # Task-specific performance factors
        task_type = task_context.get("task_type", "general")
        complexity = task_context.get("complexity", "medium")
        data_size = task_context.get("data_size", "medium")
        
        # Analyze architecture for task suitability
        performance_score = base_performance
        
        # Task type matching
        sequence_layers = sum(1 for layer in architecture.layers 
                            if layer.layer_type in [LayerType.LSTM, LayerType.GRU, LayerType.ATTENTION])
        conv_layers = sum(1 for layer in architecture.layers 
                         if layer.layer_type in [LayerType.CONV1D, LayerType.CONV2D])
        dense_layers = sum(1 for layer in architecture.layers 
                          if layer.layer_type == LayerType.DENSE)
        
        if task_type == "sequence":
            performance_score += sequence_layers * 0.1
            performance_score -= conv_layers * 0.05
        elif task_type == "vision":
            performance_score += conv_layers * 0.1
            performance_score -= sequence_layers * 0.05
        elif task_type == "tabular":
            performance_score += dense_layers * 0.08
        
        # Complexity matching
        arch_complexity = architecture.calculate_complexity()
        if complexity == "high" and arch_complexity > 0.5:
            performance_score += 0.15
        elif complexity == "low" and arch_complexity < 0.3:
            performance_score += 0.1
        elif complexity == "medium" and 0.2 <= arch_complexity <= 0.6:
            performance_score += 0.12
        
        # Architecture quality factors
        has_regularization = any(layer.layer_type == LayerType.DROPOUT for layer in architecture.layers)
        if has_regularization:
            performance_score += 0.05
        
        has_normalization = any(layer.layer_type == LayerType.BATCH_NORM for layer in architecture.layers)
        if has_normalization:
            performance_score += 0.08
        
        has_residual = any(layer.layer_type == LayerType.RESIDUAL for layer in architecture.layers)
        if has_residual and len(architecture.layers) > 5:
            performance_score += 0.1
        
        # Penalize very complex or very simple architectures
        if arch_complexity > 1.5:
            performance_score -= 0.2
        elif arch_complexity < 0.1:
            performance_score -= 0.15
        
        # Add some randomness to simulate real-world variability
        noise = random.gauss(0, 0.1)
        performance_score += noise
        
        # Clamp to valid range
        performance_score = max(0.0, min(1.0, performance_score))
        
        return performance_score

class NeuralArchitectureSearch:
    """Main Neural Architecture Search system"""
    
    def __init__(self, population_size: int = 20, max_generations: int = 10):
        self.population_size = population_size
        self.max_generations = max_generations
        self.population: List[Architecture] = []
        self.best_architectures: List[Architecture] = []
        self.generation_history: List[Dict[str, Any]] = []
        
        # Components
        self.generator = RandomArchitectureGenerator()
        self.evaluator = PerformanceEvaluator()
        
        # Search state
        self.current_generation = 0
        self.search_started = False
        
    def initialize_population(self, constraints: Dict[str, Any]) -> None:
        """Initialize the population with random architectures"""
        logger.info(f"Initializing NAS population with {self.population_size} architectures")
        
        self.population = []
        for i in range(self.population_size):
            arch = self.generator.generate_architecture(constraints)
            self.population.append(arch)
        
        self.search_started = True
        logger.info("NAS population initialized successfully")
    
    def search_architecture(self, task_context: Dict[str, Any], constraints: Dict[str, Any]) -> Architecture:
        """Execute architecture search for a specific task"""
        logger.info(f"Starting NAS for task: {task_context.get('task_type', 'unknown')}")
        
        if not self.search_started:
            self.initialize_population(constraints)
        
        best_architecture = None
        best_performance = -1.0
        
        for generation in range(self.max_generations):
            self.current_generation = generation
            logger.info(f"NAS Generation {generation + 1}/{self.max_generations}")
            
            # Evaluate all architectures in current population
            generation_performances = []
            for arch in self.population:
                performance = self.evaluator.evaluate_architecture(arch, task_context)
                generation_performances.append(performance)
                
                if performance > best_performance:
                    best_performance = performance
                    best_architecture = arch
            
            # Record generation statistics
            generation_stats = {
                "generation": generation,
                "best_performance": max(generation_performances),
                "avg_performance": statistics.mean(generation_performances),
                "worst_performance": min(generation_performances),
                "population_size": len(self.population),
                "timestamp": time.time()
            }
            self.generation_history.append(generation_stats)
            
            logger.info(f"Generation {generation}: best={generation_stats['best_performance']:.3f}, "
                       f"avg={generation_stats['avg_performance']:.3f}")
            
            # Check for early termination
            if best_performance > 0.95:
                logger.info("Early termination: excellent performance achieved")
                break
            
            # Create next generation (except for last iteration)
            if generation < self.max_generations - 1:
                self.population = self._create_next_generation(task_context)
        
        # Store best architecture
        if best_architecture:
            self.best_architectures.append(best_architecture)
        
        logger.info(f"NAS completed. Best architecture: {best_architecture.arch_id} "
                   f"(performance={best_performance:.3f})")
        
        return best_architecture
    
    def _create_next_generation(self, task_context: Dict[str, Any]) -> List[Architecture]:
        """Create the next generation using evolution strategies"""
        # Sort population by performance
        sorted_population = sorted(self.population, 
                                 key=lambda arch: arch.performance_score, 
                                 reverse=True)
        
        # Selection: keep top 50% as parents
        num_parents = self.population_size // 2
        parents = sorted_population[:num_parents]
        
        # Create next generation
        next_generation = []
        
        # Elitism: keep top 20% unchanged
        num_elite = max(1, self.population_size // 5)
        next_generation.extend(parents[:num_elite])
        
        # Fill rest with mutations and crossovers
        while len(next_generation) < self.population_size:
            if random.random() < 0.7:
                # Mutation
                parent = random.choice(parents)
                mutation_rate = self._adaptive_mutation_rate()
                child = self.generator.mutate_architecture(parent, mutation_rate)
                next_generation.append(child)
            else:
                # Crossover (simplified - just pick better parent's structure)
                parent1, parent2 = random.sample(parents, 2)
                better_parent = parent1 if parent1.performance_score > parent2.performance_score else parent2
                child = self.generator.mutate_architecture(better_parent, 0.1)
                next_generation.append(child)
        
        return next_generation[:self.population_size]
    
    def _adaptive_mutation_rate(self) -> float:
        """Calculate adaptive mutation rate based on search progress"""
        if len(self.generation_history) < 2:
            return 0.3
        
        # Check if performance is stagnating
        recent_best = [gen["best_performance"] for gen in self.generation_history[-3:]]
        if len(recent_best) >= 2:
            improvement = recent_best[-1] - recent_best[0]
            if improvement < 0.01:
                return 0.5  # High mutation for exploration
            else:
                return 0.2  # Low mutation for exploitation
        
        return 0.3
    
    def get_best_architectures(self, top_k: int = 5) -> List[Architecture]:
        """Get the top-k best architectures discovered"""
        all_best = sorted(self.best_architectures, 
                         key=lambda arch: arch.performance_score, 
                         reverse=True)
        return all_best[:top_k]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics"""
        if not self.generation_history:
            return {"status": "not_started"}
        
        all_performances = []
        for gen in self.generation_history:
            all_performances.extend([gen["best_performance"], gen["avg_performance"]])
        
        stats = {
            "generations_completed": len(self.generation_history),
            "population_size": self.population_size,
            "best_overall_performance": max(arch.performance_score for arch in self.best_architectures) if self.best_architectures else 0,
            "avg_performance_trend": [gen["avg_performance"] for gen in self.generation_history],
            "best_performance_trend": [gen["best_performance"] for gen in self.generation_history],
            "total_architectures_evaluated": len(self.evaluator.evaluation_history),
            "search_duration": time.time() - self.generation_history[0]["timestamp"] if self.generation_history else 0,
            "convergence_achieved": len(self.generation_history) < self.max_generations
        }
        
        return stats
    
    def save_search_state(self, filepath: str) -> None:
        """Save the current search state"""
        state_data = {
            "population": [arch.to_dict() for arch in self.population],
            "best_architectures": [arch.to_dict() for arch in self.best_architectures],
            "generation_history": self.generation_history,
            "current_generation": self.current_generation,
            "search_started": self.search_started,
            "evaluator_history": self.evaluator.evaluation_history,
            "config": {
                "population_size": self.population_size,
                "max_generations": self.max_generations
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        logger.info(f"NAS search state saved to {filepath}")

def create_task_master_nas_system() -> NeuralArchitectureSearch:
    """Factory function to create a NAS system optimized for Task-Master"""
    nas = NeuralArchitectureSearch(
        population_size=15,  # Reasonable size for testing
        max_generations=8    # Reasonable for quick iteration
    )
    
    logger.info("Task-Master NAS system created and configured")
    return nas

def test_nas_system():
    """Test the Neural Architecture Search system"""
    print("Testing Neural Architecture Search System...")
    
    # Create NAS system
    nas = create_task_master_nas_system()
    
    # Define test task and constraints
    task_context = {
        "task_type": "sequence",
        "complexity": "medium",
        "data_size": "large",
        "performance_target": 0.85
    }
    
    constraints = {
        "max_complexity": 0.8,
        "preferred_layer_types": [LayerType.LSTM, LayerType.DENSE, LayerType.DROPOUT],
        "max_layers": 8
    }
    
    # Run architecture search
    logger.info("Starting NAS test run...")
    best_arch = nas.search_architecture(task_context, constraints)
    
    # Display results
    print(f"\nNAS Search Results:")
    print(f"Best Architecture ID: {best_arch.arch_id}")
    print(f"Performance Score: {best_arch.performance_score:.3f}")
    print(f"Complexity Score: {best_arch.complexity_score:.3f}")
    print(f"Efficiency Score: {best_arch.efficiency_score:.3f}")
    print(f"Number of Layers: {len(best_arch.layers)}")
    
    # Show architecture details
    print(f"\nArchitecture Details:")
    for i, layer in enumerate(best_arch.layers):
        print(f"  Layer {i}: {layer.layer_type.value} "
              f"(units={layer.units}, activation={layer.activation.value if layer.activation else None})")
    
    # Show search statistics
    stats = nas.get_search_statistics()
    print(f"\nSearch Statistics:")
    print(f"Generations: {stats['generations_completed']}")
    print(f"Total Evaluations: {stats['total_architectures_evaluated']}")
    print(f"Search Duration: {stats['search_duration']:.2f} seconds")
    print(f"Best Performance: {stats['best_overall_performance']:.3f}")
    
    # Save search state
    nas.save_search_state(".taskmaster/nas_search_state.json")
    print(f"\nNAS search state saved successfully")
    
    return nas, best_arch

if __name__ == "__main__":
    test_nas_system()