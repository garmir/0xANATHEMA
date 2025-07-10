#!/usr/bin/env python3
"""
Recursive Todo Enhancement Engine Architecture
Atomic Task 51.2: Design Recursive Enhancement Architecture

This module defines the architecture and core components for a recursive
todo enhancement system that integrates with Task-Master AI and leverages
the existing self-improving architecture.
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
# Fallback for numpy if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    import statistics

from collections import defaultdict, deque
import copy


class EnhancementType(Enum):
    """Types of todo enhancement operations"""
    CLARITY = "clarity_enhancement"
    ATOMIC_DECOMPOSITION = "atomic_decomposition"
    PRIORITY_OPTIMIZATION = "priority_optimization"
    CONTEXT_ENRICHMENT = "context_enrichment"
    IMPLEMENTATION_GUIDANCE = "implementation_guidance"


class RecursionTrigger(Enum):
    """Conditions that trigger recursive enhancement"""
    COMPLEXITY_THRESHOLD = "complexity_threshold"
    QUALITY_IMPROVEMENT = "quality_improvement"
    USER_REQUEST = "user_request"
    MULTI_DIMENSIONAL = "multi_dimensional"


class EnhancementQuality(Enum):
    """Quality levels for enhanced todos"""
    POOR = "poor"          # <40% improvement
    FAIR = "fair"          # 40-60% improvement
    GOOD = "good"          # 60-80% improvement
    EXCELLENT = "excellent" # >80% improvement


@dataclass
class TodoItem:
    """Core todo item data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    original_content: str = ""
    priority: int = 5  # 1-10 scale
    complexity_score: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    enhancement_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_enhanced_at: Optional[datetime] = None
    
    def calculate_clarity_score(self) -> float:
        """Calculate clarity score based on content analysis"""
        # Simplified scoring based on content characteristics
        clarity_factors = {
            "has_action_verb": len([w for w in self.content.lower().split() 
                                  if w in ["implement", "create", "design", "build", "test"]]) > 0,
            "is_specific": len(self.content.split()) > 3,
            "has_measurable_outcome": any(word in self.content.lower() 
                                        for word in ["complete", "finish", "deliver", "achieve"]),
            "avoids_ambiguity": not any(word in self.content.lower() 
                                      for word in ["maybe", "probably", "might", "could"])
        }
        
        return sum(clarity_factors.values()) / len(clarity_factors)
    
    def needs_enhancement(self, threshold: float = 0.7) -> bool:
        """Check if todo needs enhancement based on quality thresholds"""
        clarity = self.calculate_clarity_score()
        return (clarity < threshold or 
                self.complexity_score > 7.0 or 
                len(self.content.split()) < 3)


@dataclass
class EnhancementResult:
    """Result of a todo enhancement operation"""
    original_todo: TodoItem
    enhanced_todo: TodoItem
    enhancement_type: EnhancementType
    quality_improvement: float
    recursive_depth: int
    processing_time: float
    confidence_score: float
    applied_strategies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_quality_level(self) -> EnhancementQuality:
        """Determine quality level based on improvement score"""
        if self.quality_improvement >= 0.8:
            return EnhancementQuality.EXCELLENT
        elif self.quality_improvement >= 0.6:
            return EnhancementQuality.GOOD
        elif self.quality_improvement >= 0.4:
            return EnhancementQuality.FAIR
        else:
            return EnhancementQuality.POOR


@dataclass
class RecursiveEnhancementContext:
    """Context for recursive enhancement processing"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_depth: int = 3
    current_depth: int = 0
    enhancement_budget: float = 30.0  # seconds
    quality_threshold: float = 0.8
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    project_context: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    
    def should_continue_recursion(self, current_quality: float) -> bool:
        """Determine if recursion should continue"""
        return (self.current_depth < self.max_depth and
                current_quality < self.quality_threshold and
                (datetime.now() - self.started_at).total_seconds() < self.enhancement_budget)


class EnhancementStrategy(ABC):
    """Abstract base class for enhancement strategies"""
    
    @abstractmethod
    async def can_enhance(self, todo: TodoItem, context: RecursiveEnhancementContext) -> bool:
        """Check if this strategy can enhance the given todo"""
        pass
    
    @abstractmethod
    async def enhance(self, todo: TodoItem, context: RecursiveEnhancementContext) -> TodoItem:
        """Apply enhancement to the todo item"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this enhancement strategy"""
        pass


class ClarityEnhancementStrategy(EnhancementStrategy):
    """Strategy for improving todo clarity and specificity"""
    
    async def can_enhance(self, todo: TodoItem, context: RecursiveEnhancementContext) -> bool:
        return todo.calculate_clarity_score() < 0.7
    
    async def enhance(self, todo: TodoItem, context: RecursiveEnhancementContext) -> TodoItem:
        enhanced_todo = copy.deepcopy(todo)
        
        # Simulate clarity enhancement
        content = enhanced_todo.content
        
        # Add action verb if missing
        if not any(verb in content.lower() for verb in ["implement", "create", "design", "build"]):
            if "todo" in content.lower():
                content = content.replace("todo", "implement", 1)
            else:
                content = f"Implement {content}"
        
        # Add specificity
        if len(content.split()) < 5:
            content = f"{content} with clear acceptance criteria and deliverables"
        
        # Remove ambiguous language
        ambiguous_words = {"maybe": "should", "probably": "will", "might": "should"}
        for ambiguous, replacement in ambiguous_words.items():
            content = content.replace(ambiguous, replacement)
        
        enhanced_todo.content = content
        enhanced_todo.last_enhanced_at = datetime.now()
        
        return enhanced_todo
    
    def get_strategy_name(self) -> str:
        return "clarity_enhancement"


class AtomicDecompositionStrategy(EnhancementStrategy):
    """Strategy for breaking complex todos into atomic tasks"""
    
    async def can_enhance(self, todo: TodoItem, context: RecursiveEnhancementContext) -> bool:
        return todo.complexity_score > 7.0 or len(todo.content.split()) > 15
    
    async def enhance(self, todo: TodoItem, context: RecursiveEnhancementContext) -> TodoItem:
        enhanced_todo = copy.deepcopy(todo)
        
        # Simulate atomic decomposition
        original_content = enhanced_todo.content
        
        # Generate atomic subtasks
        if "implement" in original_content.lower():
            atomic_tasks = [
                f"Research requirements for {original_content.lower().replace('implement', '').strip()}",
                f"Design architecture for {original_content.lower().replace('implement', '').strip()}",
                f"Implement core functionality for {original_content.lower().replace('implement', '').strip()}",
                f"Test and validate {original_content.lower().replace('implement', '').strip()}"
            ]
        else:
            atomic_tasks = [
                f"Plan approach for: {original_content}",
                f"Execute primary work: {original_content}",
                f"Validate completion: {original_content}"
            ]
        
        # Update content with decomposed tasks
        enhanced_todo.content = f"{original_content} (Decomposed into {len(atomic_tasks)} atomic tasks)"
        enhanced_todo.metadata["atomic_tasks"] = atomic_tasks
        enhanced_todo.last_enhanced_at = datetime.now()
        
        return enhanced_todo
    
    def get_strategy_name(self) -> str:
        return "atomic_decomposition"


class ContextEnrichmentStrategy(EnhancementStrategy):
    """Strategy for adding relevant context and prerequisites"""
    
    async def can_enhance(self, todo: TodoItem, context: RecursiveEnhancementContext) -> bool:
        return len(todo.context) < 3 or not todo.dependencies
    
    async def enhance(self, todo: TodoItem, context: RecursiveEnhancementContext) -> TodoItem:
        enhanced_todo = copy.deepcopy(todo)
        
        # Add context based on content analysis
        content_lower = enhanced_todo.content.lower()
        
        # Add technical context
        if any(tech in content_lower for tech in ["implement", "code", "develop"]):
            enhanced_todo.context.update({
                "domain": "software_development",
                "skill_level": "intermediate",
                "estimated_effort": "medium"
            })
        
        # Add project context from recursive context
        if context.project_context:
            enhanced_todo.context.update({
                "project_phase": context.project_context.get("phase", "development"),
                "priority_level": context.project_context.get("priority", "medium")
            })
        
        # Add implementation guidance
        enhanced_todo.content = f"{enhanced_todo.content} (Context: {enhanced_todo.context.get('domain', 'general')} task)"
        enhanced_todo.last_enhanced_at = datetime.now()
        
        return enhanced_todo
    
    def get_strategy_name(self) -> str:
        return "context_enrichment"


class RecursiveEnhancementEngine:
    """Core engine for recursive todo enhancement with meta-learning"""
    
    def __init__(self):
        self.strategies: List[EnhancementStrategy] = []
        self.enhancement_history: List[EnhancementResult] = []
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.meta_learning_enabled = True
        self.adaptation_rate = 0.1
        self.meta_parameters = {}
        self.logger = logging.getLogger("RecursiveEnhancementEngine")
        
        # Initialize default strategies
        self._initialize_strategies()
        self._initialize_meta_learning()
    
    def _initialize_strategies(self):
        """Initialize enhancement strategies"""
        self.strategies = [
            ClarityEnhancementStrategy(),
            AtomicDecompositionStrategy(),
            ContextEnrichmentStrategy()
        ]
        
        self.logger.info(f"Initialized {len(self.strategies)} enhancement strategies")
    
    def _initialize_meta_learning(self):
        """Initialize meta-learning parameters"""
        self.meta_parameters = {
            'clarity_threshold': 0.7,
            'complexity_threshold': 7.0,
            'recursion_depth_limit': 3,
            'quality_improvement_threshold': 0.05,
            'strategy_selection_temperature': 0.8
        }
        
        if self.meta_learning_enabled:
            self.logger.info("Meta-learning initialized with adaptive parameter tuning")
    
    async def enhance_todo_recursive(self, todo: TodoItem, 
                                   context: RecursiveEnhancementContext = None) -> EnhancementResult:
        """Recursively enhance a todo item"""
        if context is None:
            context = RecursiveEnhancementContext()
        
        start_time = time.time()
        self.logger.info(f"Starting recursive enhancement (depth {context.current_depth}): {todo.content[:50]}...")
        
        # Base case: check recursion limits
        if not context.should_continue_recursion(todo.calculate_clarity_score()):
            return self._create_result(todo, todo, EnhancementType.CLARITY, 
                                     0.0, context.current_depth, time.time() - start_time)
        
        # Select best strategy for current todo
        selected_strategy = await self._select_strategy(todo, context)
        if not selected_strategy:
            return self._create_result(todo, todo, EnhancementType.CLARITY, 
                                     0.0, context.current_depth, time.time() - start_time)
        
        # Apply enhancement
        enhanced_todo = await selected_strategy.enhance(todo, context)
        
        # Calculate improvement
        original_quality = todo.calculate_clarity_score()
        enhanced_quality = enhanced_todo.calculate_clarity_score()
        improvement = enhanced_quality - original_quality
        
        # Record enhancement
        enhancement_record = {
            "strategy": selected_strategy.get_strategy_name(),
            "depth": context.current_depth,
            "improvement": improvement,
            "timestamp": datetime.now()
        }
        enhanced_todo.enhancement_history.append(enhancement_record)
        
        # Update strategy performance
        self.strategy_performance[selected_strategy.get_strategy_name()].append(improvement)
        
        # Recursive enhancement if improvement threshold not met
        if (improvement > 0.05 and 
            enhanced_quality < context.quality_threshold and 
            context.current_depth < context.max_depth - 1):
            
            # Create new context for deeper recursion
            recursive_context = copy.deepcopy(context)
            recursive_context.current_depth += 1
            
            self.logger.info(f"Applying recursive enhancement at depth {recursive_context.current_depth}")
            
            # Recursive call
            recursive_result = await self.enhance_todo_recursive(enhanced_todo, recursive_context)
            enhanced_todo = recursive_result.enhanced_todo
            improvement += recursive_result.quality_improvement
        
        # Create final result
        processing_time = time.time() - start_time
        result = self._create_result(todo, enhanced_todo, 
                                   EnhancementType.CLARITY, improvement, 
                                   context.current_depth, processing_time)
        
        result.applied_strategies = [strategy.get_strategy_name() for strategy in self.strategies
                                   if await strategy.can_enhance(todo, context)]
        
        # Store result
        self.enhancement_history.append(result)
        
        # Meta-learning update
        if self.meta_learning_enabled:
            await self._update_meta_parameters(result)
        
        self.logger.info(f"Enhancement completed: {improvement:.3f} improvement in {processing_time:.2f}s")
        return result
    
    async def _select_strategy(self, todo: TodoItem, 
                             context: RecursiveEnhancementContext) -> Optional[EnhancementStrategy]:
        """Select the best enhancement strategy for the given todo"""
        applicable_strategies = []
        
        for strategy in self.strategies:
            if await strategy.can_enhance(todo, context):
                # Calculate strategy effectiveness score
                strategy_name = strategy.get_strategy_name()
                recent_performance = self.strategy_performance[strategy_name][-10:]  # Last 10 uses
                effectiveness = statistics.mean(recent_performance) if recent_performance else 0.5
                
                applicable_strategies.append((strategy, effectiveness))
        
        if not applicable_strategies:
            return None
        
        # Sort by effectiveness and return best strategy
        applicable_strategies.sort(key=lambda x: x[1], reverse=True)
        return applicable_strategies[0][0]
    
    def _create_result(self, original: TodoItem, enhanced: TodoItem, 
                      enhancement_type: EnhancementType, improvement: float,
                      depth: int, processing_time: float) -> EnhancementResult:
        """Create enhancement result object"""
        return EnhancementResult(
            original_todo=original,
            enhanced_todo=enhanced,
            enhancement_type=enhancement_type,
            quality_improvement=improvement,
            recursive_depth=depth,
            processing_time=processing_time,
            confidence_score=min(1.0, improvement * 2),  # Simple confidence calculation
            metadata={
                "engine_version": "1.0",
                "strategy_count": len(self.strategies)
            }
        )
    
    async def enhance_todo_batch(self, todos: List[TodoItem], 
                               context: RecursiveEnhancementContext = None) -> List[EnhancementResult]:
        """Enhance multiple todos in batch"""
        if context is None:
            context = RecursiveEnhancementContext()
        
        self.logger.info(f"Starting batch enhancement of {len(todos)} todos")
        
        results = []
        for todo in todos:
            try:
                result = await self.enhance_todo_recursive(todo, context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error enhancing todo {todo.id}: {e}")
        
        self.logger.info(f"Batch enhancement completed: {len(results)} todos processed")
        return results
    
    async def _update_meta_parameters(self, result: EnhancementResult):
        """Update meta-learning parameters based on enhancement results"""
        if not self.meta_learning_enabled:
            return
        
        # Adaptive threshold adjustment based on performance
        improvement = result.quality_improvement
        
        # Update clarity threshold based on performance trend
        recent_improvements = [r.quality_improvement for r in self.enhancement_history[-10:]]
        if len(recent_improvements) >= 5:
            avg_improvement = statistics.mean(recent_improvements)
            
            if avg_improvement > 0.3:  # Good performance, can be more selective
                self.meta_parameters['clarity_threshold'] = min(0.9, 
                    self.meta_parameters['clarity_threshold'] + self.adaptation_rate * 0.1)
            elif avg_improvement < 0.1:  # Poor performance, be less selective
                self.meta_parameters['clarity_threshold'] = max(0.5,
                    self.meta_parameters['clarity_threshold'] - self.adaptation_rate * 0.1)
        
        # Update complexity threshold based on decomposition success
        if result.enhancement_type == EnhancementType.ATOMIC_DECOMPOSITION:
            if improvement > 0.2:
                self.meta_parameters['complexity_threshold'] *= (1 - self.adaptation_rate * 0.05)
            else:
                self.meta_parameters['complexity_threshold'] *= (1 + self.adaptation_rate * 0.05)
        
        # Update recursion depth based on diminishing returns
        if result.recursive_depth > 1 and improvement < 0.1:
            self.meta_parameters['recursion_depth_limit'] = max(1,
                self.meta_parameters['recursion_depth_limit'] - 1)
        elif result.recursive_depth >= self.meta_parameters['recursion_depth_limit'] and improvement > 0.3:
            self.meta_parameters['recursion_depth_limit'] = min(5,
                self.meta_parameters['recursion_depth_limit'] + 1)
        
        # Log significant parameter changes
        if len(self.enhancement_history) % 10 == 0:
            self.logger.info(f"Meta-parameters updated: clarity_threshold={self.meta_parameters['clarity_threshold']:.3f}, "
                           f"complexity_threshold={self.meta_parameters['complexity_threshold']:.1f}, "
                           f"recursion_depth_limit={self.meta_parameters['recursion_depth_limit']}")
    
    def get_meta_learning_status(self) -> Dict[str, Any]:
        """Get current meta-learning parameters and adaptation status"""
        return {
            "enabled": self.meta_learning_enabled,
            "adaptation_rate": self.adaptation_rate,
            "current_parameters": self.meta_parameters.copy(),
            "enhancements_processed": len(self.enhancement_history),
            "parameter_stability": self._calculate_parameter_stability()
        }
    
    def _calculate_parameter_stability(self) -> float:
        """Calculate how stable the meta-parameters are (lower = more stable)"""
        if len(self.enhancement_history) < 20:
            return 1.0  # High instability with few samples
        
        # Simple stability metric based on recent parameter changes
        # In a real implementation, this would track parameter change history
        recent_improvements = [r.quality_improvement for r in self.enhancement_history[-20:]]
        improvement_variance = statistics.variance(recent_improvements) if len(recent_improvements) > 1 else 1.0
        
        # Lower variance indicates more stable performance and parameters
        return min(1.0, improvement_variance * 2.0)
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """Get enhancement engine statistics"""
        if not self.enhancement_history:
            return {"status": "no_enhancements"}
        
        recent_results = self.enhancement_history[-50:]
        
        stats = {
            "total_enhancements": len(self.enhancement_history),
            "average_improvement": statistics.mean(r.quality_improvement for r in recent_results),
            "average_processing_time": statistics.mean(r.processing_time for r in recent_results),
            "strategy_performance": {},
            "quality_distribution": {
                "excellent": sum(1 for r in recent_results if r.get_quality_level() == EnhancementQuality.EXCELLENT),
                "good": sum(1 for r in recent_results if r.get_quality_level() == EnhancementQuality.GOOD),
                "fair": sum(1 for r in recent_results if r.get_quality_level() == EnhancementQuality.FAIR),
                "poor": sum(1 for r in recent_results if r.get_quality_level() == EnhancementQuality.POOR)
            }
        }
        
        # Strategy performance analysis
        for strategy_name, performance_history in self.strategy_performance.items():
            if performance_history:
                stats["strategy_performance"][strategy_name] = {
                    "average_improvement": statistics.mean(performance_history[-20:]),
                    "usage_count": len(performance_history),
                    "effectiveness_score": statistics.mean(performance_history[-10:]) if len(performance_history) >= 10 else 0.5
                }
        
        return stats


class TaskMasterIntegration:
    """Integration layer with Task-Master AI system"""
    
    def __init__(self, enhancement_engine: RecursiveEnhancementEngine):
        self.enhancement_engine = enhancement_engine
        self.logger = logging.getLogger("TaskMasterIntegration")
    
    async def enhance_task_json(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance todos from Task-Master tasks.json format"""
        enhanced_task_data = copy.deepcopy(task_data)
        
        # Extract todos from task structure
        todos = self._extract_todos_from_task_data(task_data)
        
        # Enhance todos
        enhancement_results = await self.enhancement_engine.enhance_todo_batch(todos)
        
        # Update task data with enhanced todos
        self._update_task_data_with_enhancements(enhanced_task_data, enhancement_results)
        
        return enhanced_task_data
    
    def _extract_todos_from_task_data(self, task_data: Dict[str, Any]) -> List[TodoItem]:
        """Extract todo items from task-master data structure"""
        todos = []
        
        # Process main tasks
        for task in task_data.get("master", {}).get("tasks", []):
            todo = TodoItem(
                id=str(task.get("id", "")),
                content=task.get("title", ""),
                original_content=task.get("description", ""),
                priority=self._map_priority(task.get("priority", "medium")),
                complexity_score=self._estimate_complexity(task.get("title", "")),
                context={
                    "task_type": "main_task",
                    "status": task.get("status", "pending"),
                    "test_strategy": task.get("testStrategy", "")
                },
                dependencies=task.get("dependencies", [])
            )
            todos.append(todo)
            
            # Process subtasks
            for subtask in task.get("subtasks", []):
                subtodo = TodoItem(
                    id=str(subtask.get("id", "")),
                    content=subtask.get("title", ""),
                    original_content=subtask.get("description", ""),
                    priority=self._map_priority(subtask.get("priority", "medium")),
                    complexity_score=self._estimate_complexity(subtask.get("title", "")),
                    context={
                        "task_type": "subtask",
                        "parent_task": task.get("id"),
                        "status": subtask.get("status", "pending")
                    },
                    dependencies=subtask.get("dependencies", [])
                )
                todos.append(subtodo)
        
        return todos
    
    def _map_priority(self, priority_str: str) -> int:
        """Map priority string to numeric value"""
        priority_map = {
            "low": 3,
            "medium": 5,
            "high": 8,
            "critical": 10
        }
        return priority_map.get(priority_str.lower(), 5)
    
    def _estimate_complexity(self, content: str) -> float:
        """Estimate complexity based on content analysis"""
        # Simple complexity estimation
        factors = {
            "length": min(10, len(content.split()) / 5),
            "technical_terms": sum(1 for word in ["implement", "design", "architecture", "system"] 
                                 if word in content.lower()),
            "scope_indicators": sum(1 for word in ["comprehensive", "complete", "full", "entire"] 
                                  if word in content.lower())
        }
        
        return min(10.0, sum(factors.values()))
    
    def _update_task_data_with_enhancements(self, task_data: Dict[str, Any], 
                                          enhancement_results: List[EnhancementResult]):
        """Update task data with enhancement results"""
        # Create enhancement lookup
        enhancements_by_id = {result.enhanced_todo.id: result for result in enhancement_results}
        
        # Update main tasks
        for task in task_data.get("master", {}).get("tasks", []):
            task_id = str(task.get("id", ""))
            if task_id in enhancements_by_id:
                enhancement = enhancements_by_id[task_id]
                task["title"] = enhancement.enhanced_todo.content
                task["enhancement_metadata"] = {
                    "improved": True,
                    "improvement_score": enhancement.quality_improvement,
                    "strategies_applied": enhancement.applied_strategies,
                    "enhanced_at": enhancement.timestamp.isoformat()
                }
            
            # Update subtasks
            for subtask in task.get("subtasks", []):
                subtask_id = str(subtask.get("id", ""))
                if subtask_id in enhancements_by_id:
                    enhancement = enhancements_by_id[subtask_id]
                    subtask["title"] = enhancement.enhanced_todo.content
                    subtask["enhancement_metadata"] = {
                        "improved": True,
                        "improvement_score": enhancement.quality_improvement,
                        "strategies_applied": enhancement.applied_strategies,
                        "enhanced_at": enhancement.timestamp.isoformat()
                    }


# Export key classes
__all__ = [
    "EnhancementType", "RecursionTrigger", "EnhancementQuality",
    "TodoItem", "EnhancementResult", "RecursiveEnhancementContext",
    "EnhancementStrategy", "ClarityEnhancementStrategy", "AtomicDecompositionStrategy",
    "ContextEnrichmentStrategy", "RecursiveEnhancementEngine", "TaskMasterIntegration"
]


if __name__ == "__main__":
    # Demo usage
    async def demo():
        logging.basicConfig(level=logging.INFO)
        
        print("🔄 Recursive Todo Enhancement Engine Architecture Demo")
        print("=" * 65)
        
        # Create enhancement engine
        engine = RecursiveEnhancementEngine()
        
        # Create sample todos
        sample_todos = [
            TodoItem(
                content="fix bug",
                complexity_score=3.0
            ),
            TodoItem(
                content="implement comprehensive user authentication system with oauth",
                complexity_score=8.5
            ),
            TodoItem(
                content="maybe add some documentation",
                complexity_score=4.0
            ),
            TodoItem(
                content="research and implement advanced caching mechanisms",
                complexity_score=7.0
            )
        ]
        
        print(f"📝 Enhancing {len(sample_todos)} todo items...")
        
        # Enhance todos
        context = RecursiveEnhancementContext(
            max_depth=2,
            quality_threshold=0.8,
            enhancement_budget=60.0
        )
        
        results = []
        for todo in sample_todos:
            print(f"\n🚀 Enhancing: '{todo.content}'")
            result = await engine.enhance_todo_recursive(todo, context)
            results.append(result)
            
            print(f"  Original clarity: {todo.calculate_clarity_score():.3f}")
            print(f"  Enhanced: '{result.enhanced_todo.content}'")
            print(f"  New clarity: {result.enhanced_todo.calculate_clarity_score():.3f}")
            print(f"  Improvement: {result.quality_improvement:.3f}")
            print(f"  Quality level: {result.get_quality_level().value}")
            print(f"  Processing time: {result.processing_time:.2f}s")
        
        # Show statistics
        stats = engine.get_enhancement_statistics()
        print(f"\n📊 Enhancement Statistics:")
        print(f"  Total enhancements: {stats['total_enhancements']}")
        print(f"  Average improvement: {stats['average_improvement']:.3f}")
        print(f"  Average processing time: {stats['average_processing_time']:.3f}s")
        
        print(f"\n📈 Quality Distribution:")
        for quality, count in stats['quality_distribution'].items():
            print(f"  {quality.capitalize()}: {count}")
        
        print(f"\n🔧 Strategy Performance:")
        for strategy, perf in stats['strategy_performance'].items():
            print(f"  {strategy}: {perf['average_improvement']:.3f} avg improvement")
        
        # Show meta-learning status
        meta_status = engine.get_meta_learning_status()
        print(f"\n🧠 Meta-Learning Status:")
        print(f"  Enabled: {meta_status['enabled']}")
        print(f"  Enhancements processed: {meta_status['enhancements_processed']}")
        print(f"  Parameter stability: {meta_status['parameter_stability']:.3f}")
        print(f"  Current parameters:")
        for param, value in meta_status['current_parameters'].items():
            if isinstance(value, float):
                print(f"    {param}: {value:.3f}")
            else:
                print(f"    {param}: {value}")
        
        # Test Task-Master integration
        print(f"\n🔗 Testing Task-Master Integration...")
        integration = TaskMasterIntegration(engine)
        
        sample_task_data = {
            "master": {
                "tasks": [
                    {
                        "id": 1,
                        "title": "fix authentication bug",
                        "description": "User login is broken",
                        "priority": "high",
                        "status": "pending",
                        "subtasks": [
                            {
                                "id": "1.1",
                                "title": "investigate issue",
                                "priority": "high",
                                "status": "pending"
                            }
                        ]
                    }
                ]
            }
        }
        
        enhanced_task_data = await integration.enhance_task_json(sample_task_data)
        
        print(f"  Original task: {sample_task_data['master']['tasks'][0]['title']}")
        print(f"  Enhanced task: {enhanced_task_data['master']['tasks'][0]['title']}")
        
        print(f"\n✅ Recursive Todo Enhancement Engine architecture operational!")
    
    # Run demo
    asyncio.run(demo())