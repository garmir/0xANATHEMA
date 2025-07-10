#!/usr/bin/env python3
"""
Evolutionary Optimization with Local Models
Replaces external model calls with local LLM evaluation for autonomous improvement
"""

import asyncio
import json
import time
import random
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
import logging
import numpy as np
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.api_abstraction import UnifiedModelAPI, TaskType

logger = logging.getLogger(__name__)

@dataclass
class Individual:
    """Represents an individual in the evolutionary population"""
    id: str
    genome: Dict[str, Any]
    fitness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "genome": self.genome,
            "fitness": self.fitness,
            "metadata": self.metadata,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            genome=data["genome"],
            fitness=data.get("fitness"),
            metadata=data.get("metadata", {}),
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            created_at=data.get("created_at", time.time())
        )

@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization"""
    population_size: int = 20
    max_generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 2
    tournament_size: int = 3
    convergence_threshold: float = 0.001
    max_stagnation: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elite_size": self.elite_size,
            "tournament_size": self.tournament_size,
            "convergence_threshold": self.convergence_threshold,
            "max_stagnation": self.max_stagnation
        }

class FitnessEvaluator(ABC):
    """Abstract base class for fitness evaluation"""
    
    @abstractmethod
    async def evaluate(self, individual: Individual) -> float:
        """Evaluate fitness of an individual"""
        pass
    
    @abstractmethod
    def get_optimization_goal(self) -> str:
        """Return description of optimization goal"""
        pass

class LocalLLMFitnessEvaluator(FitnessEvaluator):
    """Fitness evaluator using local LLM assessment"""
    
    def __init__(self, 
                 api: UnifiedModelAPI,
                 evaluation_criteria: str,
                 optimization_goal: str):
        self.api = api
        self.evaluation_criteria = evaluation_criteria
        self.optimization_goal = optimization_goal
        self.evaluation_cache = {}
        
    async def evaluate(self, individual: Individual) -> float:
        """Evaluate individual using local LLM"""
        # Check cache first
        genome_hash = str(hash(json.dumps(individual.genome, sort_keys=True)))
        if genome_hash in self.evaluation_cache:
            return self.evaluation_cache[genome_hash]
        
        evaluation_prompt = f"""
        Evaluate the fitness of this solution configuration for optimization:
        
        OPTIMIZATION GOAL: {self.optimization_goal}
        
        EVALUATION CRITERIA: {self.evaluation_criteria}
        
        SOLUTION CONFIGURATION:
        {json.dumps(individual.genome, indent=2)}
        
        INDIVIDUAL METADATA:
        - Generation: {individual.generation}
        - ID: {individual.id}
        - Created: {datetime.fromtimestamp(individual.created_at)}
        
        Please evaluate this configuration on a scale of 0.0 to 1.0 based on:
        1. How well it meets the optimization goal
        2. Practical feasibility and implementability
        3. Performance characteristics
        4. Resource efficiency
        5. Robustness and error handling
        6. Scalability potential
        
        Provide your assessment in this format:
        FITNESS_SCORE: [0.0-1.0]
        STRENGTHS: [key strengths of this configuration]
        WEAKNESSES: [areas for improvement]
        REASONING: [detailed explanation of the score]
        """
        
        try:
            response = await self.api.generate(
                evaluation_prompt,
                task_type=TaskType.ANALYSIS,
                temperature=0.2  # Low temperature for consistent evaluation
            )
            
            # Parse fitness score
            fitness = 0.5  # Default
            strengths = []
            weaknesses = []
            reasoning = ""
            
            for line in response.content.split('\n'):
                line = line.strip()
                if line.startswith('FITNESS_SCORE:'):
                    try:
                        fitness = float(line.split(':')[1].strip())
                        fitness = max(0.0, min(1.0, fitness))  # Clamp to [0,1]
                    except:
                        pass
                elif line.startswith('STRENGTHS:'):
                    strengths.append(line.split(':', 1)[1].strip())
                elif line.startswith('WEAKNESSES:'):
                    weaknesses.append(line.split(':', 1)[1].strip())
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            # Update individual metadata
            individual.metadata.update({
                'fitness_evaluation': {
                    'score': fitness,
                    'strengths': strengths,
                    'weaknesses': weaknesses,
                    'reasoning': reasoning,
                    'model_used': response.model_used,
                    'evaluated_at': time.time()
                }
            })
            
            # Cache result
            self.evaluation_cache[genome_hash] = fitness
            
            return fitness
            
        except Exception as e:
            logger.error(f"Fitness evaluation failed for {individual.id}: {e}")
            return 0.1  # Low default fitness for failed evaluations
    
    def get_optimization_goal(self) -> str:
        """Return optimization goal"""
        return self.optimization_goal

class PerformanceFitnessEvaluator(FitnessEvaluator):
    """Fitness evaluator based on actual performance metrics"""
    
    def __init__(self, 
                 performance_function: Callable[[Dict[str, Any]], float],
                 optimization_goal: str = "Maximize performance metrics"):
        self.performance_function = performance_function
        self.optimization_goal = optimization_goal
        self.evaluation_cache = {}
    
    async def evaluate(self, individual: Individual) -> float:
        """Evaluate based on performance function"""
        genome_hash = str(hash(json.dumps(individual.genome, sort_keys=True)))
        if genome_hash in self.evaluation_cache:
            return self.evaluation_cache[genome_hash]
        
        try:
            # Run performance function (could be async)
            if asyncio.iscoroutinefunction(self.performance_function):
                fitness = await self.performance_function(individual.genome)
            else:
                fitness = self.performance_function(individual.genome)
            
            # Normalize to [0,1] range
            fitness = max(0.0, min(1.0, fitness))
            
            # Cache result
            self.evaluation_cache[genome_hash] = fitness
            
            return fitness
            
        except Exception as e:
            logger.error(f"Performance evaluation failed for {individual.id}: {e}")
            return 0.0
    
    def get_optimization_goal(self) -> str:
        return self.optimization_goal

class EvolutionaryOptimizer:
    """
    Evolutionary optimizer using local LLMs for fitness evaluation
    Supports multiple optimization strategies and autonomous parameter tuning
    """
    
    def __init__(self,
                 api: UnifiedModelAPI,
                 fitness_evaluator: FitnessEvaluator,
                 config: EvolutionConfig = None,
                 results_dir: str = ".taskmaster/local_modules/optimization/results"):
        self.api = api
        self.fitness_evaluator = fitness_evaluator
        self.config = config or EvolutionConfig()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Evolution state
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[Tuple[int, float, float, float]] = []  # generation, best, avg, worst
        self.stagnation_counter = 0
        
        # Performance tracking
        self.evolution_stats = {
            "total_evaluations": 0,
            "avg_evaluation_time": 0,
            "convergence_generation": None,
            "optimization_time": 0
        }
        
        # Thread pool for parallel evaluation
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def initialize_population(self, genome_template: Dict[str, Any], 
                            parameter_ranges: Dict[str, Any]) -> List[Individual]:
        """Initialize population with random individuals"""
        population = []
        
        for i in range(self.config.population_size):
            genome = self._generate_random_genome(genome_template, parameter_ranges)
            individual = Individual(
                id=f"gen0_ind{i}",
                genome=genome,
                generation=0
            )
            population.append(individual)
        
        logger.info(f"Initialized population of {len(population)} individuals")
        return population
    
    def _generate_random_genome(self, template: Dict[str, Any], 
                               ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random genome based on template and ranges"""
        genome = {}
        
        for key, value in template.items():
            if key in ranges:
                range_spec = ranges[key]
                
                if isinstance(range_spec, dict):
                    if range_spec.get("type") == "float":
                        genome[key] = random.uniform(range_spec["min"], range_spec["max"])
                    elif range_spec.get("type") == "int":
                        genome[key] = random.randint(range_spec["min"], range_spec["max"])
                    elif range_spec.get("type") == "choice":
                        genome[key] = random.choice(range_spec["choices"])
                    elif range_spec.get("type") == "bool":
                        genome[key] = random.choice([True, False])
                    else:
                        genome[key] = value
                else:
                    genome[key] = value
            else:
                genome[key] = value
        
        return genome
    
    async def evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """Evaluate fitness for entire population"""
        logger.info(f"Evaluating population of {len(population)} individuals")
        start_time = time.time()
        
        # Evaluate in parallel batches
        batch_size = min(4, len(population))  # Limit concurrent evaluations
        evaluated_population = []
        
        for i in range(0, len(population), batch_size):
            batch = population[i:i + batch_size]
            
            # Evaluate batch
            tasks = [self.fitness_evaluator.evaluate(individual) for individual in batch]
            fitnesses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update individuals with fitness scores
            for individual, fitness in zip(batch, fitnesses):
                if isinstance(fitness, Exception):
                    logger.error(f"Evaluation failed for {individual.id}: {fitness}")
                    individual.fitness = 0.0
                else:
                    individual.fitness = fitness
                
                evaluated_population.append(individual)
        
        # Update statistics
        total_time = time.time() - start_time
        self.evolution_stats["total_evaluations"] += len(population)
        self.evolution_stats["avg_evaluation_time"] = (
            (self.evolution_stats["avg_evaluation_time"] * (self.evolution_stats["total_evaluations"] - len(population)) +
             total_time) / self.evolution_stats["total_evaluations"]
        )
        
        logger.info(f"Population evaluation completed in {total_time:.2f}s")
        return evaluated_population
    
    def select_parents(self, population: List[Individual]) -> List[Individual]:
        """Select parents using tournament selection"""
        parents = []
        
        for _ in range(self.config.population_size):
            # Tournament selection
            tournament = random.sample(population, min(self.config.tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.fitness or 0)
            parents.append(winner)
        
        return parents
    
    async def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Create offspring through crossover"""
        if random.random() > self.config.crossover_rate:
            # No crossover, return copies of parents
            child1 = Individual(
                id=f"gen{self.generation + 1}_child_{random.randint(1000, 9999)}",
                genome=parent1.genome.copy(),
                generation=self.generation + 1,
                parent_ids=[parent1.id]
            )
            child2 = Individual(
                id=f"gen{self.generation + 1}_child_{random.randint(1000, 9999)}",
                genome=parent2.genome.copy(),
                generation=self.generation + 1,
                parent_ids=[parent2.id]
            )
            return child1, child2
        
        # Perform crossover using local LLM guidance
        crossover_result = await self._llm_guided_crossover(parent1, parent2)
        
        child1 = Individual(
            id=f"gen{self.generation + 1}_child_{random.randint(1000, 9999)}",
            genome=crossover_result["child1_genome"],
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id],
            metadata={"crossover_method": crossover_result["method"]}
        )
        
        child2 = Individual(
            id=f"gen{self.generation + 1}_child_{random.randint(1000, 9999)}",
            genome=crossover_result["child2_genome"],
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id],
            metadata={"crossover_method": crossover_result["method"]}
        )
        
        return child1, child2
    
    async def _llm_guided_crossover(self, parent1: Individual, parent2: Individual) -> Dict[str, Any]:
        """Use local LLM to guide crossover operation"""
        crossover_prompt = f"""
        Perform intelligent crossover between two parent solutions to create offspring:
        
        OPTIMIZATION GOAL: {self.fitness_evaluator.get_optimization_goal()}
        
        PARENT 1 (Fitness: {parent1.fitness:.3f}):
        {json.dumps(parent1.genome, indent=2)}
        
        PARENT 2 (Fitness: {parent2.fitness:.3f}):
        {json.dumps(parent2.genome, indent=2)}
        
        Create two offspring by combining the best aspects of both parents.
        Consider:
        1. Which parameters from each parent are likely most beneficial
        2. How to balance exploration vs exploitation
        3. Maintaining valid parameter combinations
        
        Provide response in this JSON format:
        {{
            "child1_genome": {{ /* combined parameters */ }},
            "child2_genome": {{ /* alternative combination */ }},
            "method": "Description of crossover strategy used",
            "reasoning": "Why this combination was chosen"
        }}
        """
        
        try:
            response = await self.api.generate(
                crossover_prompt,
                task_type=TaskType.OPTIMIZATION,
                temperature=0.4  # Some randomness for diversity
            )
            
            # Parse JSON response
            try:
                crossover_data = json.loads(response.content)
                return crossover_data
            except json.JSONDecodeError:
                # Fallback to simple crossover
                return self._simple_crossover(parent1, parent2)
                
        except Exception as e:
            logger.error(f"LLM-guided crossover failed: {e}")
            return self._simple_crossover(parent1, parent2)
    
    def _simple_crossover(self, parent1: Individual, parent2: Individual) -> Dict[str, Any]:
        """Simple crossover fallback"""
        genome1 = parent1.genome.copy()
        genome2 = parent2.genome.copy()
        
        # Uniform crossover
        for key in genome1.keys():
            if random.random() < 0.5:
                genome1[key], genome2[key] = genome2[key], genome1[key]
        
        return {
            "child1_genome": genome1,
            "child2_genome": genome2,
            "method": "uniform_crossover",
            "reasoning": "Simple uniform parameter exchange"
        }
    
    async def mutate(self, individual: Individual, 
                   parameter_ranges: Dict[str, Any]) -> Individual:
        """Mutate individual using local LLM guidance"""
        if random.random() > self.config.mutation_rate:
            return individual
        
        mutation_result = await self._llm_guided_mutation(individual, parameter_ranges)
        
        mutated_individual = Individual(
            id=f"gen{self.generation + 1}_mut_{random.randint(1000, 9999)}",
            genome=mutation_result["mutated_genome"],
            generation=self.generation + 1,
            parent_ids=[individual.id],
            metadata={
                "mutation_method": mutation_result["method"],
                "mutation_strength": mutation_result.get("strength", "medium")
            }
        )
        
        return mutated_individual
    
    async def _llm_guided_mutation(self, individual: Individual, 
                                 parameter_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Use local LLM to guide mutation"""
        mutation_prompt = f"""
        Perform intelligent mutation on this solution to create a variant:
        
        OPTIMIZATION GOAL: {self.fitness_evaluator.get_optimization_goal()}
        
        CURRENT SOLUTION (Fitness: {individual.fitness:.3f}):
        {json.dumps(individual.genome, indent=2)}
        
        PARAMETER RANGES:
        {json.dumps(parameter_ranges, indent=2)}
        
        Create a mutated version that:
        1. Explores nearby parameter space
        2. Has potential for improvement
        3. Maintains parameter validity
        4. Introduces reasonable variation
        
        Consider the current fitness level to determine mutation strength:
        - High fitness: small, conservative changes
        - Medium fitness: moderate exploration
        - Low fitness: larger changes for escape
        
        Provide response in this JSON format:
        {{
            "mutated_genome": {{ /* modified parameters */ }},
            "method": "Description of mutation strategy",
            "strength": "small|medium|large",
            "reasoning": "Why these changes were made"
        }}
        """
        
        try:
            response = await self.api.generate(
                mutation_prompt,
                task_type=TaskType.OPTIMIZATION,
                temperature=0.5  # Higher temperature for exploration
            )
            
            # Parse JSON response
            try:
                mutation_data = json.loads(response.content)
                return mutation_data
            except json.JSONDecodeError:
                # Fallback to simple mutation
                return self._simple_mutation(individual, parameter_ranges)
                
        except Exception as e:
            logger.error(f"LLM-guided mutation failed: {e}")
            return self._simple_mutation(individual, parameter_ranges)
    
    def _simple_mutation(self, individual: Individual, 
                        parameter_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Simple mutation fallback"""
        mutated_genome = individual.genome.copy()
        
        # Mutate random parameter
        keys = list(mutated_genome.keys())
        key_to_mutate = random.choice(keys)
        
        if key_to_mutate in parameter_ranges:
            range_spec = parameter_ranges[key_to_mutate]
            
            if isinstance(range_spec, dict):
                if range_spec.get("type") == "float":
                    current = mutated_genome[key_to_mutate]
                    range_size = range_spec["max"] - range_spec["min"]
                    mutation_size = range_size * 0.1  # 10% mutation
                    mutated_genome[key_to_mutate] = max(
                        range_spec["min"],
                        min(range_spec["max"], current + random.uniform(-mutation_size, mutation_size))
                    )
                elif range_spec.get("type") == "int":
                    current = mutated_genome[key_to_mutate]
                    range_size = range_spec["max"] - range_spec["min"]
                    mutation_size = max(1, range_size // 10)
                    mutated_genome[key_to_mutate] = max(
                        range_spec["min"],
                        min(range_spec["max"], current + random.randint(-mutation_size, mutation_size))
                    )
                elif range_spec.get("type") == "choice":
                    mutated_genome[key_to_mutate] = random.choice(range_spec["choices"])
                elif range_spec.get("type") == "bool":
                    mutated_genome[key_to_mutate] = not mutated_genome[key_to_mutate]
        
        return {
            "mutated_genome": mutated_genome,
            "method": "random_parameter_mutation",
            "strength": "medium",
            "reasoning": f"Random mutation of {key_to_mutate}"
        }
    
    async def evolve_generation(self, parameter_ranges: Dict[str, Any]) -> List[Individual]:
        """Evolve one generation"""
        logger.info(f"Evolving generation {self.generation + 1}")
        
        # Selection
        parents = self.select_parents(self.population)
        
        # Create offspring
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = await self.crossover(parent1, parent2)
            
            # Mutation
            child1 = await self.mutate(child1, parameter_ranges)
            child2 = await self.mutate(child2, parameter_ranges)
            
            offspring.extend([child1, child2])
        
        # Combine with elite individuals
        elite = sorted(self.population, key=lambda x: x.fitness or 0, reverse=True)[:self.config.elite_size]
        new_population = elite + offspring[:self.config.population_size - self.config.elite_size]
        
        # Evaluate new population
        self.population = await self.evaluate_population(new_population)
        self.generation += 1
        
        return self.population
    
    def update_statistics(self):
        """Update evolution statistics"""
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        
        if fitnesses:
            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            worst_fitness = min(fitnesses)
            
            self.fitness_history.append((self.generation, best_fitness, avg_fitness, worst_fitness))
            
            # Update best individual
            best_individual = max(self.population, key=lambda x: x.fitness or 0)
            if self.best_individual is None or best_individual.fitness > self.best_individual.fitness:
                self.best_individual = best_individual
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            logger.info(f"Generation {self.generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, Worst={worst_fitness:.4f}")
    
    def check_convergence(self) -> bool:
        """Check if evolution has converged"""
        if len(self.fitness_history) < 5:
            return False
        
        # Check for stagnation
        if self.stagnation_counter >= self.config.max_stagnation:
            logger.info(f"Evolution stagnated after {self.stagnation_counter} generations")
            return True
        
        # Check for fitness plateau
        recent_best = [entry[1] for entry in self.fitness_history[-5:]]
        if max(recent_best) - min(recent_best) < self.config.convergence_threshold:
            logger.info(f"Evolution converged (fitness plateau)")
            self.evolution_stats["convergence_generation"] = self.generation
            return True
        
        return False
    
    async def optimize(self, 
                     genome_template: Dict[str, Any],
                     parameter_ranges: Dict[str, Any],
                     save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete evolutionary optimization
        
        Args:
            genome_template: Template for individual genomes
            parameter_ranges: Valid ranges for each parameter
            save_results: Whether to save optimization results
            
        Returns:
            Optimization results including best individual and statistics
        """
        start_time = time.time()
        logger.info(f"Starting evolutionary optimization with {self.config.population_size} individuals")
        logger.info(f"Optimization goal: {self.fitness_evaluator.get_optimization_goal()}")
        
        # Initialize population
        self.population = self.initialize_population(genome_template, parameter_ranges)
        self.population = await self.evaluate_population(self.population)
        self.update_statistics()
        
        # Evolution loop
        for generation in range(self.config.max_generations):
            await self.evolve_generation(parameter_ranges)
            self.update_statistics()
            
            # Check convergence
            if self.check_convergence():
                break
        
        # Finalize results
        self.evolution_stats["optimization_time"] = time.time() - start_time
        
        optimization_results = {
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
            "final_generation": self.generation,
            "fitness_history": self.fitness_history,
            "evolution_stats": self.evolution_stats,
            "config": self.config.to_dict(),
            "optimization_goal": self.fitness_evaluator.get_optimization_goal(),
            "timestamp": datetime.now().isoformat(),
            "population_final": [ind.to_dict() for ind in self.population]
        }
        
        # Save results
        if save_results:
            results_file = self.results_dir / f"optimization_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(optimization_results, f, indent=2)
            logger.info(f"Optimization results saved to {results_file}")
        
        logger.info(f"Optimization completed in {self.evolution_stats['optimization_time']:.2f}s")
        logger.info(f"Best fitness achieved: {self.best_individual.fitness:.4f}")
        
        return optimization_results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization progress"""
        if not self.fitness_history:
            return {"status": "not_started"}
        
        latest_stats = self.fitness_history[-1]
        return {
            "status": "completed" if self.check_convergence() else "running",
            "current_generation": self.generation,
            "best_fitness": latest_stats[1],
            "avg_fitness": latest_stats[2],
            "population_size": len(self.population),
            "total_evaluations": self.evolution_stats["total_evaluations"],
            "optimization_time": self.evolution_stats.get("optimization_time", time.time()),
            "stagnation_counter": self.stagnation_counter
        }

# Example usage
if __name__ == "__main__":
    async def test_evolutionary_optimization():
        from ..core.api_abstraction import UnifiedModelAPI, ModelConfigFactory
        
        # Initialize API
        api = UnifiedModelAPI()
        api.add_model("ollama-llama2", ModelConfigFactory.create_ollama_config(
            "llama2", capabilities=[TaskType.OPTIMIZATION, TaskType.ANALYSIS]
        ))
        
        # Define optimization problem
        evaluation_criteria = """
        Optimize task scheduling parameters for maximum throughput and minimum latency.
        Consider resource utilization, load balancing, and system stability.
        """
        
        optimization_goal = "Maximize task processing efficiency while maintaining system stability"
        
        # Create fitness evaluator
        fitness_evaluator = LocalLLMFitnessEvaluator(
            api=api,
            evaluation_criteria=evaluation_criteria,
            optimization_goal=optimization_goal
        )
        
        # Create optimizer
        config = EvolutionConfig(
            population_size=10,
            max_generations=20,
            mutation_rate=0.15,
            crossover_rate=0.8
        )
        
        optimizer = EvolutionaryOptimizer(
            api=api,
            fitness_evaluator=fitness_evaluator,
            config=config
        )
        
        # Define genome template and parameter ranges
        genome_template = {
            "batch_size": 32,
            "timeout": 60,
            "max_concurrent": 4,
            "retry_attempts": 3,
            "cache_enabled": True,
            "load_balancing": "round_robin"
        }
        
        parameter_ranges = {
            "batch_size": {"type": "int", "min": 1, "max": 100},
            "timeout": {"type": "int", "min": 10, "max": 300},
            "max_concurrent": {"type": "int", "min": 1, "max": 16},
            "retry_attempts": {"type": "int", "min": 0, "max": 10},
            "cache_enabled": {"type": "bool"},
            "load_balancing": {"type": "choice", "choices": ["round_robin", "least_connections", "random"]}
        }
        
        # Run optimization
        results = await optimizer.optimize(genome_template, parameter_ranges)
        
        print(f"Optimization completed!")
        print(f"Best individual: {json.dumps(results['best_individual'], indent=2)}")
        print(f"Final generation: {results['final_generation']}")
        print(f"Optimization time: {results['evolution_stats']['optimization_time']:.2f}s")
    
    # Run test
    asyncio.run(test_evolutionary_optimization())