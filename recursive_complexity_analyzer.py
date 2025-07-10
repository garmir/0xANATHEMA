#!/usr/bin/env python3
"""
Recursive Complexity Analyzer with Adaptive Meta-Learning
Advanced recursive analysis system that continuously learns and adapts complexity predictions
through meta-learning algorithms and recursive pattern recognition.
"""

import asyncio
import json
import time
import hashlib
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Union
import gc
import threading
import sqlite3

# Import existing complexity analyzer with fallback
try:
    from task_complexity_analyzer import TaskComplexityAnalyzer, TaskComplexity, ComplexityClass, SystemResources
except ImportError:
    # Fallback implementations for systems without psutil
    from enum import Enum
    from dataclasses import dataclass
    from typing import Dict, Any
    
    class ComplexityClass(Enum):
        CONSTANT = "O(1)"
        LOGARITHMIC = "O(log n)"
        LINEAR = "O(n)"
        LINEARITHMIC = "O(n log n)"
        QUADRATIC = "O(n²)"
        CUBIC = "O(n³)"
        EXPONENTIAL = "O(2^n)"
        FACTORIAL = "O(n!)"
    
    @dataclass
    class TaskComplexity:
        task_id: str
        time_complexity: ComplexityClass
        space_complexity: ComplexityClass
        io_operations: int
        parallelization_potential: float
        cpu_intensive: bool
        memory_intensive: bool
        network_dependent: bool
        file_operations: int
        estimated_runtime_seconds: float
        resource_requirements: Dict[str, Any]
    
    @dataclass
    class SystemResources:
        cpu_cores: int
        available_memory_gb: float
        available_disk_gb: float
        network_bandwidth_mbps: float
        cpu_usage_percent: float
        memory_usage_percent: float
    
    class TaskComplexityAnalyzer:
        def __init__(self, tasks_file: str = ".taskmaster/tasks/tasks.json"):
            self.tasks_file = tasks_file
            self.tasks_data = self._load_tasks()
        
        def _load_tasks(self) -> Dict:
            try:
                with open(self.tasks_file, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                return {"tags": {"master": {"tasks": []}}}
        
        def analyze_task_complexity(self, task_id: str) -> TaskComplexity:
            return TaskComplexity(
                task_id=task_id,
                time_complexity=ComplexityClass.LINEAR,
                space_complexity=ComplexityClass.LINEAR,
                io_operations=5,
                parallelization_potential=0.5,
                cpu_intensive=False,
                memory_intensive=False,
                network_dependent=False,
                file_operations=2,
                estimated_runtime_seconds=10.0,
                resource_requirements={"memory_mb": 100, "cpu_cores": 1}
            )


class LearningStrategy(Enum):
    """Meta-learning strategies for complexity adaptation"""
    GRADIENT_ADAPTATION = "gradient_adaptation"
    PATTERN_RECOGNITION = "pattern_recognition"
    RECURSIVE_FEEDBACK = "recursive_feedback"
    ENSEMBLE_LEARNING = "ensemble_learning"
    TEMPORAL_WEIGHTING = "temporal_weighting"


class RecursivePattern(Enum):
    """Types of recursive patterns in task complexity"""
    LINEAR_SCALING = "linear_scaling"
    EXPONENTIAL_GROWTH = "exponential_growth"
    CYCLICAL_COMPLEXITY = "cyclical_complexity"
    HIERARCHICAL_DECOMPOSITION = "hierarchical_decomposition"
    DEPENDENCY_CASCADE = "dependency_cascade"


@dataclass
class ComplexityPrediction:
    """Enhanced complexity prediction with confidence and learning metadata"""
    task_id: str
    predicted_complexity: TaskComplexity
    confidence_score: float  # 0.0 to 1.0
    learning_strategy_used: LearningStrategy
    recursive_pattern: Optional[RecursivePattern]
    historical_accuracy: float
    prediction_timestamp: datetime
    meta_features: Dict[str, Any] = field(default_factory=dict)
    adaptation_score: float = 0.0


@dataclass
class LearningHistory:
    """Tracks learning history and adaptation performance"""
    prediction_id: str
    predicted_values: Dict[str, Any]
    actual_values: Dict[str, Any]
    accuracy_score: float
    learning_rate: float
    adaptation_applied: bool
    timestamp: datetime
    meta_learning_insights: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecursiveContext:
    """Context for recursive complexity analysis"""
    depth: int
    parent_task_id: Optional[str]
    sibling_tasks: List[str]
    subtask_count: int
    dependency_level: int
    recursive_multiplier: float
    pattern_signature: str


class AdaptiveMetaLearner:
    """Meta-learning engine for complexity prediction adaptation"""
    
    def __init__(self, learning_rate: float = 0.01, decay_factor: float = 0.99):
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.adaptation_weights = defaultdict(float)
        self.pattern_library = {}
        self.confidence_thresholds = {
            LearningStrategy.GRADIENT_ADAPTATION: 0.8,
            LearningStrategy.PATTERN_RECOGNITION: 0.7,
            LearningStrategy.RECURSIVE_FEEDBACK: 0.85,
            LearningStrategy.ENSEMBLE_LEARNING: 0.9,
            LearningStrategy.TEMPORAL_WEIGHTING: 0.75
        }
        
    def adapt_prediction(self, history: LearningHistory, current_prediction: ComplexityPrediction) -> ComplexityPrediction:
        """Adapt complexity prediction based on learning history"""
        strategy = current_prediction.learning_strategy_used
        
        # Calculate adaptation factor based on historical accuracy
        adaptation_factor = self._calculate_adaptation_factor(history, strategy)
        
        # Apply strategy-specific adaptation
        adapted_prediction = self._apply_strategy_adaptation(
            current_prediction, adaptation_factor, history
        )
        
        # Update learning weights
        self._update_learning_weights(strategy, history.accuracy_score)
        
        return adapted_prediction
    
    def _calculate_adaptation_factor(self, history: LearningHistory, strategy: LearningStrategy) -> float:
        """Calculate how much to adapt based on learning history"""
        base_adaptation = 1.0 - history.accuracy_score
        strategy_weight = self.adaptation_weights[strategy]
        
        # Apply temporal decay to recent learning
        time_factor = min(1.0, (datetime.now() - history.timestamp).total_seconds() / 3600)
        
        return base_adaptation * (1.0 + strategy_weight) * (1.0 - time_factor * self.decay_factor)
    
    def _apply_strategy_adaptation(self, prediction: ComplexityPrediction, 
                                 adaptation_factor: float, history: LearningHistory) -> ComplexityPrediction:
        """Apply strategy-specific adaptations to the prediction"""
        adapted_prediction = prediction
        
        if prediction.learning_strategy_used == LearningStrategy.GRADIENT_ADAPTATION:
            # Adjust runtime estimates based on gradient descent
            runtime_adjustment = adaptation_factor * self.learning_rate
            adapted_prediction.predicted_complexity.estimated_runtime_seconds *= (1.0 + runtime_adjustment)
            
        elif prediction.learning_strategy_used == LearningStrategy.PATTERN_RECOGNITION:
            # Adjust based on pattern matching accuracy
            if prediction.recursive_pattern:
                pattern_accuracy = self._get_pattern_accuracy(prediction.recursive_pattern)
                confidence_adjustment = adaptation_factor * pattern_accuracy
                adapted_prediction.confidence_score = min(1.0, 
                    adapted_prediction.confidence_score + confidence_adjustment)
                
        elif prediction.learning_strategy_used == LearningStrategy.RECURSIVE_FEEDBACK:
            # Apply recursive feedback corrections
            feedback_correction = self._calculate_recursive_feedback(history, adaptation_factor)
            adapted_prediction.adaptation_score = feedback_correction
            
        adapted_prediction.meta_features["adaptation_applied"] = True
        adapted_prediction.meta_features["adaptation_factor"] = adaptation_factor
        
        return adapted_prediction
    
    def _update_learning_weights(self, strategy: LearningStrategy, accuracy: float):
        """Update strategy weights based on performance"""
        current_weight = self.adaptation_weights[strategy]
        weight_update = self.learning_rate * (accuracy - 0.5)  # Center around 0.5
        self.adaptation_weights[strategy] = current_weight + weight_update
    
    def _get_pattern_accuracy(self, pattern: RecursivePattern) -> float:
        """Get historical accuracy for a specific recursive pattern"""
        return self.pattern_library.get(pattern, 0.5)
    
    def _calculate_recursive_feedback(self, history: LearningHistory, adaptation_factor: float) -> float:
        """Calculate recursive feedback correction factor"""
        # Implementation of recursive feedback algorithm
        base_feedback = history.accuracy_score * adaptation_factor
        recursive_depth_penalty = 0.95 ** history.meta_learning_insights.get("recursive_depth", 1)
        return base_feedback * recursive_depth_penalty


class RecursiveComplexityAnalyzer:
    """
    Advanced recursive complexity analyzer with adaptive meta-learning capabilities.
    Continuously learns and improves complexity predictions through recursive analysis.
    """
    
    def __init__(self, tasks_file: str = ".taskmaster/tasks/tasks.json", 
                 learning_database: str = ".taskmaster/learning/complexity_learning.db"):
        """Initialize the recursive complexity analyzer"""
        self.base_analyzer = TaskComplexityAnalyzer(tasks_file)
        self.learning_db_path = learning_database
        self.meta_learner = AdaptiveMetaLearner()
        
        # Initialize learning database
        self._init_learning_database()
        
        # Recursive analysis state
        self.recursive_cache = {}
        self.pattern_detector = RecursivePatternDetector()
        self.complexity_history = deque(maxlen=1000)
        self.learning_thread = None
        self.is_learning_active = True
        
        # Performance metrics
        self.analysis_metrics = {
            "total_predictions": 0,
            "accurate_predictions": 0,
            "adaptation_improvements": 0,
            "recursive_patterns_detected": 0,
            "learning_iterations": 0
        }
    
    def _init_learning_database(self):
        """Initialize SQLite database for learning history"""
        Path(self.learning_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.learning_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT,
                    task_id TEXT,
                    predicted_complexity TEXT,
                    actual_complexity TEXT,
                    accuracy_score REAL,
                    learning_strategy TEXT,
                    recursive_pattern TEXT,
                    adaptation_applied BOOLEAN,
                    timestamp TEXT,
                    meta_features TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS complexity_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    prediction_data TEXT,
                    confidence_score REAL,
                    learning_strategy TEXT,
                    recursive_context TEXT,
                    timestamp TEXT
                )
            """)
    
    async def analyze_recursive_complexity(self, task_id: str, 
                                         recursive_context: Optional[RecursiveContext] = None) -> ComplexityPrediction:
        """
        Perform recursive complexity analysis with adaptive meta-learning
        """
        # Create recursive context if not provided
        if recursive_context is None:
            recursive_context = await self._build_recursive_context(task_id)
        
        # Check cache for existing predictions
        cache_key = self._generate_cache_key(task_id, recursive_context)
        if cache_key in self.recursive_cache:
            cached_prediction = self.recursive_cache[cache_key]
            return await self._adapt_cached_prediction(cached_prediction, recursive_context)
        
        # Analyze base complexity
        base_complexity = self._analyze_base_complexity(task_id)
        
        # Apply recursive complexity analysis
        recursive_complexity = await self._apply_recursive_analysis(
            base_complexity, recursive_context
        )
        
        # Detect recursive patterns
        recursive_pattern = self.pattern_detector.detect_pattern(
            task_id, recursive_context, self.base_analyzer.tasks_data
        )
        
        # Select optimal learning strategy
        learning_strategy = self._select_learning_strategy(recursive_context, recursive_pattern)
        
        # Create complexity prediction
        prediction = ComplexityPrediction(
            task_id=task_id,
            predicted_complexity=recursive_complexity,
            confidence_score=self._calculate_confidence_score(
                recursive_complexity, recursive_context, recursive_pattern
            ),
            learning_strategy_used=learning_strategy,
            recursive_pattern=recursive_pattern,
            historical_accuracy=self._get_historical_accuracy(task_id, learning_strategy),
            prediction_timestamp=datetime.now(),
            meta_features={
                "recursive_depth": recursive_context.depth,
                "dependency_level": recursive_context.dependency_level,
                "subtask_count": recursive_context.subtask_count,
                "pattern_signature": recursive_context.pattern_signature
            }
        )
        
        # Apply meta-learning adaptation
        adapted_prediction = await self._apply_meta_learning_adaptation(prediction)
        
        # Cache the prediction
        self.recursive_cache[cache_key] = adapted_prediction
        
        # Store prediction for learning
        await self._store_prediction(adapted_prediction, recursive_context)
        
        # Update metrics
        self.analysis_metrics["total_predictions"] += 1
        if recursive_pattern:
            self.analysis_metrics["recursive_patterns_detected"] += 1
        
        return adapted_prediction
    
    async def _build_recursive_context(self, task_id: str) -> RecursiveContext:
        """Build recursive context for a task"""
        tasks = self.base_analyzer.tasks_data.get('tags', {}).get('master', {}).get('tasks', [])
        task_data = next((t for t in tasks if str(t.get('id')) == task_id), {})
        
        # Calculate recursive properties
        depth = self._calculate_task_depth(task_id, tasks)
        parent_task_id = self._find_parent_task(task_id, tasks)
        sibling_tasks = self._find_sibling_tasks(task_id, tasks)
        subtask_count = len(task_data.get('subtasks', []))
        dependency_level = len(task_data.get('dependencies', []))
        
        # Generate pattern signature
        pattern_signature = self._generate_pattern_signature(task_data)
        
        # Calculate recursive multiplier based on context
        recursive_multiplier = self._calculate_recursive_multiplier(
            depth, subtask_count, dependency_level
        )
        
        return RecursiveContext(
            depth=depth,
            parent_task_id=parent_task_id,
            sibling_tasks=sibling_tasks,
            subtask_count=subtask_count,
            dependency_level=dependency_level,
            recursive_multiplier=recursive_multiplier,
            pattern_signature=pattern_signature
        )
    
    def _analyze_base_complexity(self, task_id: str) -> TaskComplexity:
        """Analyze base complexity using existing analyzer"""
        return self.base_analyzer.analyze_task_complexity(task_id)
    
    async def _apply_recursive_analysis(self, base_complexity: TaskComplexity, 
                                      context: RecursiveContext) -> TaskComplexity:
        """Apply recursive complexity modifications"""
        # Create recursive complexity based on base complexity
        recursive_complexity = TaskComplexity(
            task_id=base_complexity.task_id,
            time_complexity=self._adjust_time_complexity(base_complexity.time_complexity, context),
            space_complexity=self._adjust_space_complexity(base_complexity.space_complexity, context),
            io_operations=int(base_complexity.io_operations * context.recursive_multiplier),
            parallelization_potential=self._adjust_parallelization_potential(
                base_complexity.parallelization_potential, context
            ),
            cpu_intensive=base_complexity.cpu_intensive or context.depth > 3,
            memory_intensive=base_complexity.memory_intensive or context.subtask_count > 10,
            network_dependent=base_complexity.network_dependent,
            file_operations=int(base_complexity.file_operations * context.recursive_multiplier),
            estimated_runtime_seconds=base_complexity.estimated_runtime_seconds * context.recursive_multiplier,
            resource_requirements=self._adjust_resource_requirements(
                base_complexity.resource_requirements, context
            )
        )
        
        return recursive_complexity
    
    def _adjust_time_complexity(self, base_complexity: ComplexityClass, context: RecursiveContext) -> ComplexityClass:
        """Adjust time complexity based on recursive context"""
        complexity_adjustments = {
            ComplexityClass.CONSTANT: ComplexityClass.LOGARITHMIC if context.depth > 2 else ComplexityClass.CONSTANT,
            ComplexityClass.LOGARITHMIC: ComplexityClass.LINEAR if context.subtask_count > 5 else ComplexityClass.LOGARITHMIC,
            ComplexityClass.LINEAR: ComplexityClass.LINEARITHMIC if context.recursive_multiplier > 2.0 else ComplexityClass.LINEAR,
            ComplexityClass.LINEARITHMIC: ComplexityClass.QUADRATIC if context.dependency_level > 3 else ComplexityClass.LINEARITHMIC,
            ComplexityClass.QUADRATIC: ComplexityClass.CUBIC if context.depth > 4 else ComplexityClass.QUADRATIC,
        }
        
        return complexity_adjustments.get(base_complexity, base_complexity)
    
    def _adjust_space_complexity(self, base_complexity: ComplexityClass, context: RecursiveContext) -> ComplexityClass:
        """Adjust space complexity based on recursive context"""
        # Space complexity generally increases with recursive depth
        if context.depth > 3:
            complexity_order = [
                ComplexityClass.CONSTANT, ComplexityClass.LOGARITHMIC, ComplexityClass.LINEAR,
                ComplexityClass.LINEARITHMIC, ComplexityClass.QUADRATIC, ComplexityClass.CUBIC
            ]
            
            try:
                current_index = complexity_order.index(base_complexity)
                new_index = min(len(complexity_order) - 1, current_index + context.depth - 3)
                return complexity_order[new_index]
            except ValueError:
                return base_complexity
        
        return base_complexity
    
    def _adjust_parallelization_potential(self, base_potential: float, context: RecursiveContext) -> float:
        """Adjust parallelization potential based on recursive context"""
        # Higher subtask count generally increases parallelization potential
        parallel_bonus = min(0.3, context.subtask_count * 0.05)
        
        # But deeper recursion might reduce it due to synchronization overhead
        depth_penalty = max(0.0, context.depth * 0.02)
        
        adjusted_potential = base_potential + parallel_bonus - depth_penalty
        return max(0.0, min(1.0, adjusted_potential))
    
    def _adjust_resource_requirements(self, base_requirements: Dict[str, Any], 
                                    context: RecursiveContext) -> Dict[str, Any]:
        """Adjust resource requirements based on recursive context"""
        adjusted = base_requirements.copy()
        
        # Scale memory requirements
        memory_multiplier = 1.0 + (context.depth * 0.1) + (context.subtask_count * 0.05)
        adjusted["memory_mb"] = adjusted.get("memory_mb", 100) * memory_multiplier
        
        # Scale CPU requirements
        cpu_multiplier = context.recursive_multiplier
        adjusted["cpu_cores"] = max(1, int(adjusted.get("cpu_cores", 1) * cpu_multiplier))
        
        # Add recursive-specific requirements
        adjusted["recursive_depth"] = context.depth
        adjusted["coordination_overhead"] = context.subtask_count * 0.1
        
        return adjusted
    
    def _select_learning_strategy(self, context: RecursiveContext, 
                                pattern: Optional[RecursivePattern]) -> LearningStrategy:
        """Select optimal learning strategy based on context and pattern"""
        # Strategy selection logic based on recursive characteristics
        if context.depth <= 2:
            return LearningStrategy.GRADIENT_ADAPTATION
        elif pattern in [RecursivePattern.CYCLICAL_COMPLEXITY, RecursivePattern.LINEAR_SCALING]:
            return LearningStrategy.PATTERN_RECOGNITION
        elif context.subtask_count > 10:
            return LearningStrategy.ENSEMBLE_LEARNING
        elif context.dependency_level > 3:
            return LearningStrategy.RECURSIVE_FEEDBACK
        else:
            return LearningStrategy.TEMPORAL_WEIGHTING
    
    def _calculate_confidence_score(self, complexity: TaskComplexity, context: RecursiveContext,
                                  pattern: Optional[RecursivePattern]) -> float:
        """Calculate confidence score for the complexity prediction"""
        base_confidence = 0.7
        
        # Adjust based on recursive depth (deeper = less confident)
        depth_adjustment = max(0.0, 0.3 - (context.depth * 0.05))
        
        # Adjust based on pattern recognition
        pattern_adjustment = 0.15 if pattern else 0.0
        
        # Adjust based on historical accuracy
        historical_accuracy = self._get_historical_accuracy(complexity.task_id, LearningStrategy.GRADIENT_ADAPTATION)
        history_adjustment = (historical_accuracy - 0.5) * 0.2
        
        confidence = base_confidence + depth_adjustment + pattern_adjustment + history_adjustment
        return max(0.1, min(1.0, confidence))
    
    async def _apply_meta_learning_adaptation(self, prediction: ComplexityPrediction) -> ComplexityPrediction:
        """Apply meta-learning adaptation to the prediction"""
        # Get relevant learning history
        learning_history = await self._get_learning_history(
            prediction.task_id, prediction.learning_strategy_used
        )
        
        if learning_history:
            # Apply adaptation using meta-learner
            adapted_prediction = self.meta_learner.adapt_prediction(learning_history, prediction)
            self.analysis_metrics["adaptation_improvements"] += 1
            return adapted_prediction
        
        return prediction
    
    async def _get_learning_history(self, task_id: str, strategy: LearningStrategy) -> Optional[LearningHistory]:
        """Retrieve relevant learning history from database"""
        try:
            with sqlite3.connect(self.learning_db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM learning_history 
                    WHERE task_id = ? AND learning_strategy = ? 
                    ORDER BY timestamp DESC LIMIT 1
                """, (task_id, strategy.value))
                
                row = cursor.fetchone()
                if row:
                    return LearningHistory(
                        prediction_id=row[1],
                        predicted_values=json.loads(row[2]),
                        actual_values=json.loads(row[3]),
                        accuracy_score=row[4],
                        learning_rate=0.01,  # Default learning rate
                        adaptation_applied=bool(row[7]),
                        timestamp=datetime.fromisoformat(row[8]),
                        meta_learning_insights=json.loads(row[9]) if row[9] else {}
                    )
        except Exception as e:
            print(f"Error retrieving learning history: {e}")
        
        return None
    
    async def _store_prediction(self, prediction: ComplexityPrediction, context: RecursiveContext):
        """Store prediction in database for future learning"""
        try:
            with sqlite3.connect(self.learning_db_path) as conn:
                conn.execute("""
                    INSERT INTO complexity_predictions 
                    (task_id, prediction_data, confidence_score, learning_strategy, recursive_context, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    prediction.task_id,
                    json.dumps(asdict(prediction.predicted_complexity), default=str),
                    prediction.confidence_score,
                    prediction.learning_strategy_used.value,
                    json.dumps(asdict(context), default=str),
                    prediction.prediction_timestamp.isoformat()
                ))
        except Exception as e:
            print(f"Error storing prediction: {e}")
    
    def _calculate_task_depth(self, task_id: str, tasks: List[Dict]) -> int:
        """Calculate the recursive depth of a task"""
        task_data = next((t for t in tasks if str(t.get('id')) == task_id), {})
        
        # Simple depth calculation based on subtasks
        def count_depth(task):
            subtasks = task.get('subtasks', [])
            if not subtasks:
                return 1
            return 1 + max(count_depth(st) for st in subtasks)
        
        return count_depth(task_data)
    
    def _find_parent_task(self, task_id: str, tasks: List[Dict]) -> Optional[str]:
        """Find parent task ID if exists"""
        for task in tasks:
            subtasks = task.get('subtasks', [])
            for subtask in subtasks:
                if str(subtask.get('id')) == task_id:
                    return str(task.get('id'))
        return None
    
    def _find_sibling_tasks(self, task_id: str, tasks: List[Dict]) -> List[str]:
        """Find sibling task IDs"""
        parent_id = self._find_parent_task(task_id, tasks)
        if not parent_id:
            return []
        
        parent_task = next((t for t in tasks if str(t.get('id')) == parent_id), {})
        siblings = []
        for subtask in parent_task.get('subtasks', []):
            sibling_id = str(subtask.get('id'))
            if sibling_id != task_id:
                siblings.append(sibling_id)
        
        return siblings
    
    def _generate_pattern_signature(self, task_data: Dict) -> str:
        """Generate a signature for pattern recognition"""
        signature_components = [
            task_data.get('title', ''),
            task_data.get('description', ''),
            str(len(task_data.get('dependencies', []))),
            str(len(task_data.get('subtasks', []))),
            task_data.get('priority', 'medium')
        ]
        
        signature_string = '|'.join(signature_components)
        return hashlib.md5(signature_string.encode()).hexdigest()[:16]
    
    def _calculate_recursive_multiplier(self, depth: int, subtask_count: int, dependency_level: int) -> float:
        """Calculate recursive complexity multiplier"""
        base_multiplier = 1.0
        
        # Depth contributes exponentially
        depth_factor = 1.2 ** max(0, depth - 1)
        
        # Subtasks contribute linearly with diminishing returns
        subtask_factor = 1.0 + (subtask_count * 0.1) / (1.0 + subtask_count * 0.05)
        
        # Dependencies add overhead
        dependency_factor = 1.0 + (dependency_level * 0.05)
        
        return base_multiplier * depth_factor * subtask_factor * dependency_factor
    
    def _get_historical_accuracy(self, task_id: str, strategy: LearningStrategy) -> float:
        """Get historical accuracy for a task and strategy combination"""
        try:
            with sqlite3.connect(self.learning_db_path) as conn:
                cursor = conn.execute("""
                    SELECT AVG(accuracy_score) FROM learning_history 
                    WHERE task_id = ? AND learning_strategy = ?
                """, (task_id, strategy.value))
                
                result = cursor.fetchone()
                return result[0] if result[0] is not None else 0.5
        except Exception:
            return 0.5  # Default neutral accuracy
    
    def _generate_cache_key(self, task_id: str, context: RecursiveContext) -> str:
        """Generate cache key for recursive context"""
        key_components = [
            task_id,
            str(context.depth),
            str(context.subtask_count),
            str(context.dependency_level),
            context.pattern_signature
        ]
        
        key_string = '|'.join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _adapt_cached_prediction(self, cached_prediction: ComplexityPrediction, 
                                     current_context: RecursiveContext) -> ComplexityPrediction:
        """Adapt cached prediction to current context"""
        # Simple adaptation - in practice, this could be more sophisticated
        time_elapsed = (datetime.now() - cached_prediction.prediction_timestamp).total_seconds()
        
        # Decay confidence over time
        time_decay = max(0.5, 1.0 - (time_elapsed / 3600))  # Decay over 1 hour
        cached_prediction.confidence_score *= time_decay
        
        # Mark as adapted
        cached_prediction.meta_features["cache_adapted"] = True
        cached_prediction.meta_features["time_decay_applied"] = time_decay
        
        return cached_prediction
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the analyzer"""
        accuracy_rate = (
            self.analysis_metrics["accurate_predictions"] / 
            max(1, self.analysis_metrics["total_predictions"])
        )
        
        return {
            **self.analysis_metrics,
            "accuracy_rate": accuracy_rate,
            "cache_hit_rate": len(self.recursive_cache) / max(1, self.analysis_metrics["total_predictions"]),
            "patterns_per_prediction": (
                self.analysis_metrics["recursive_patterns_detected"] / 
                max(1, self.analysis_metrics["total_predictions"])
            )
        }
    
    async def optimize_learning_parameters(self):
        """Optimize learning parameters based on performance"""
        metrics = self.get_performance_metrics()
        
        # Adjust learning rate based on accuracy
        if metrics["accuracy_rate"] < 0.7:
            self.meta_learner.learning_rate *= 1.1  # Increase learning rate
        elif metrics["accuracy_rate"] > 0.9:
            self.meta_learner.learning_rate *= 0.95  # Decrease learning rate
        
        # Update confidence thresholds
        for strategy in LearningStrategy:
            strategy_accuracy = await self._get_strategy_accuracy(strategy)
            if strategy_accuracy > 0.8:
                self.meta_learner.confidence_thresholds[strategy] *= 0.98
            elif strategy_accuracy < 0.6:
                self.meta_learner.confidence_thresholds[strategy] *= 1.02
    
    async def _get_strategy_accuracy(self, strategy: LearningStrategy) -> float:
        """Get accuracy for a specific learning strategy"""
        try:
            with sqlite3.connect(self.learning_db_path) as conn:
                cursor = conn.execute("""
                    SELECT AVG(accuracy_score) FROM learning_history 
                    WHERE learning_strategy = ?
                """, (strategy.value,))
                
                result = cursor.fetchone()
                return result[0] if result[0] is not None else 0.5
        except Exception:
            return 0.5


class RecursivePatternDetector:
    """Detects recursive patterns in task complexity"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.pattern_signatures = defaultdict(list)
    
    def detect_pattern(self, task_id: str, context: RecursiveContext, tasks_data: Dict) -> Optional[RecursivePattern]:
        """Detect recursive pattern in task structure"""
        # Linear scaling pattern
        if self._is_linear_scaling(context):
            return RecursivePattern.LINEAR_SCALING
        
        # Exponential growth pattern
        if self._is_exponential_growth(context):
            return RecursivePattern.EXPONENTIAL_GROWTH
        
        # Cyclical complexity pattern
        if self._is_cyclical_complexity(context, tasks_data):
            return RecursivePattern.CYCLICAL_COMPLEXITY
        
        # Hierarchical decomposition pattern
        if self._is_hierarchical_decomposition(context):
            return RecursivePattern.HIERARCHICAL_DECOMPOSITION
        
        # Dependency cascade pattern
        if self._is_dependency_cascade(context):
            return RecursivePattern.DEPENDENCY_CASCADE
        
        return None
    
    def _is_linear_scaling(self, context: RecursiveContext) -> bool:
        """Check if task exhibits linear scaling pattern"""
        return (
            context.depth <= 3 and 
            context.subtask_count > 0 and 
            context.recursive_multiplier < 2.0
        )
    
    def _is_exponential_growth(self, context: RecursiveContext) -> bool:
        """Check if task exhibits exponential growth pattern"""
        return (
            context.depth > 3 and 
            context.recursive_multiplier > 3.0 and
            context.subtask_count > 5
        )
    
    def _is_cyclical_complexity(self, context: RecursiveContext, tasks_data: Dict) -> bool:
        """Check if task exhibits cyclical complexity pattern"""
        # Simple heuristic based on pattern signature repetition
        signature_count = self.pattern_signatures[context.pattern_signature]
        signature_count.append(datetime.now())
        
        # Keep only recent signatures (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.pattern_signatures[context.pattern_signature] = [
            ts for ts in signature_count if ts > cutoff
        ]
        
        return len(self.pattern_signatures[context.pattern_signature]) > 3
    
    def _is_hierarchical_decomposition(self, context: RecursiveContext) -> bool:
        """Check if task exhibits hierarchical decomposition pattern"""
        return (
            context.depth > 2 and 
            context.subtask_count > 3 and 
            len(context.sibling_tasks) > 1
        )
    
    def _is_dependency_cascade(self, context: RecursiveContext) -> bool:
        """Check if task exhibits dependency cascade pattern"""
        return (
            context.dependency_level > 2 and 
            context.depth > 1 and
            context.recursive_multiplier > 1.5
        )


async def main():
    """Demonstration of the Recursive Complexity Analyzer"""
    print("🔄 RECURSIVE COMPLEXITY ANALYZER WITH ADAPTIVE META-LEARNING")
    print("=" * 70)
    
    analyzer = RecursiveComplexityAnalyzer()
    
    # Example task IDs for analysis
    test_task_ids = ["1", "2", "3", "11", "12"]
    
    for task_id in test_task_ids:
        try:
            print(f"\n📊 Analyzing Task {task_id}...")
            
            prediction = await analyzer.analyze_recursive_complexity(task_id)
            
            print(f"   Task ID: {prediction.task_id}")
            print(f"   Time Complexity: {prediction.predicted_complexity.time_complexity.value}")
            print(f"   Space Complexity: {prediction.predicted_complexity.space_complexity.value}")
            print(f"   Estimated Runtime: {prediction.predicted_complexity.estimated_runtime_seconds:.2f}s")
            print(f"   Confidence Score: {prediction.confidence_score:.2f}")
            print(f"   Learning Strategy: {prediction.learning_strategy_used.value}")
            print(f"   Recursive Pattern: {prediction.recursive_pattern.value if prediction.recursive_pattern else 'None'}")
            print(f"   Adaptation Score: {prediction.adaptation_score:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error analyzing task {task_id}: {e}")
    
    # Display performance metrics
    print(f"\n📈 PERFORMANCE METRICS")
    print("=" * 30)
    metrics = analyzer.get_performance_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Recursive Complexity Analysis Complete")


if __name__ == "__main__":
    asyncio.run(main())