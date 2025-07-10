#!/usr/bin/env python3
"""
Adaptive Learning Engine for Self-Improving AI Models
Learns from execution patterns to optimize task generation and execution strategies
"""

import json
import os
import time
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
# import numpy as np  # Not available, will use built-in statistics

@dataclass
class ExecutionPattern:
    """Individual execution pattern data"""
    task_id: str
    task_type: str
    complexity_score: float
    execution_time: float
    success_rate: float
    resource_usage: Dict[str, float]
    context_features: Dict[str, Any]
    timestamp: datetime

@dataclass
class OptimizationStrategy:
    """Optimization strategy learned from patterns"""
    strategy_id: str
    pattern_fingerprint: str
    success_rate: float
    avg_speedup: float
    confidence_score: float
    applicable_conditions: Dict[str, Any]
    optimization_params: Dict[str, Any]
    usage_count: int

@dataclass
class LearningModel:
    """AI learning model state"""
    model_version: str
    training_data_size: int
    accuracy_score: float
    last_training: datetime
    feature_weights: Dict[str, float]
    optimization_strategies: List[OptimizationStrategy]

class AdaptiveLearningEngine:
    """Self-improving AI engine that learns from execution patterns"""
    
    def __init__(self, learning_dir: str = '.taskmaster/ai/learning'):
        self.learning_dir = Path(learning_dir)
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        # Learning data storage
        self.patterns_file = self.learning_dir / 'execution_patterns.json'
        self.model_file = self.learning_dir / 'learning_model.pkl'
        self.strategies_file = self.learning_dir / 'optimization_strategies.json'
        
        # In-memory data structures
        self.execution_patterns = deque(maxlen=10000)  # Recent patterns
        self.pattern_index = defaultdict(list)  # Fast lookup by features
        self.optimization_strategies = {}
        self.current_model = None
        
        # Learning parameters
        self.min_patterns_for_learning = 50
        self.retraining_interval = timedelta(hours=24)
        self.confidence_threshold = 0.75
        
        self.load_existing_data()
        
    def load_existing_data(self):
        """Load existing learning data"""
        try:
            # Load execution patterns
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r') as f:
                    pattern_data = json.load(f)
                    for p in pattern_data:
                        p['timestamp'] = datetime.fromisoformat(p['timestamp'])
                        pattern = ExecutionPattern(**p)
                        self.execution_patterns.append(pattern)
                        self._index_pattern(pattern)
            
            # Load optimization strategies
            if self.strategies_file.exists():
                with open(self.strategies_file, 'r') as f:
                    strategies_data = json.load(f)
                    for s_data in strategies_data:
                        strategy = OptimizationStrategy(**s_data)
                        self.optimization_strategies[strategy.strategy_id] = strategy
            
            # Load learning model
            if self.model_file.exists():
                with open(self.model_file, 'rb') as f:
                    self.current_model = pickle.load(f)
                    
            print(f"✅ Loaded {len(self.execution_patterns)} patterns, "
                  f"{len(self.optimization_strategies)} strategies")
                  
        except Exception as e:
            print(f"⚠️ Error loading existing data: {e}")
    
    def record_execution_pattern(self, task_data: Dict[str, Any], 
                                execution_result: Dict[str, Any]) -> str:
        """Record a new execution pattern for learning"""
        
        # Extract pattern features
        pattern = ExecutionPattern(
            task_id=task_data.get('id', 'unknown'),
            task_type=self._classify_task_type(task_data),
            complexity_score=self._calculate_complexity_score(task_data),
            execution_time=execution_result.get('execution_time', 0),
            success_rate=1.0 if execution_result.get('success', False) else 0.0,
            resource_usage=execution_result.get('resource_usage', {}),
            context_features=self._extract_context_features(task_data),
            timestamp=datetime.now()
        )
        
        # Store pattern
        self.execution_patterns.append(pattern)
        self._index_pattern(pattern)
        
        # Trigger learning if we have enough data
        if len(self.execution_patterns) >= self.min_patterns_for_learning:
            self._trigger_learning_if_needed()
        
        # Save patterns periodically
        self._save_patterns()
        
        return f"pattern_{int(time.time())}"
    
    def get_optimization_recommendations(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get AI-powered optimization recommendations for a task"""
        
        recommendations = []
        
        # Extract task features
        task_features = self._extract_context_features(task_data)
        task_complexity = self._calculate_complexity_score(task_data)
        task_type = self._classify_task_type(task_data)
        
        # Find matching optimization strategies
        for strategy_id, strategy in self.optimization_strategies.items():
            if self._strategy_applies_to_task(strategy, task_features, task_type):
                recommendation = {
                    'strategy_id': strategy_id,
                    'optimization_type': strategy.optimization_params.get('type', 'general'),
                    'expected_speedup': strategy.avg_speedup,
                    'confidence': strategy.confidence_score,
                    'parameters': strategy.optimization_params,
                    'description': self._generate_strategy_description(strategy)
                }
                recommendations.append(recommendation)
        
        # Sort by confidence and expected benefit
        recommendations.sort(key=lambda x: x['confidence'] * x['expected_speedup'], reverse=True)
        
        # Add pattern-based recommendations
        pattern_recommendations = self._get_pattern_based_recommendations(
            task_features, task_complexity, task_type
        )
        recommendations.extend(pattern_recommendations)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def adapt_task_generation_strategy(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt task generation strategy based on learned patterns"""
        
        if not self.current_model:
            return self._get_default_generation_strategy()
        
        # Analyze project context patterns
        context_features = self._extract_project_features(project_context)
        
        # Predict optimal task generation parameters
        strategy = {
            'suggested_task_count': self._predict_optimal_task_count(context_features),
            'priority_distribution': self._predict_priority_distribution(context_features),
            'complexity_targets': self._predict_complexity_targets(context_features),
            'dependency_patterns': self._predict_dependency_patterns(context_features),
            'execution_order_hints': self._predict_execution_order(context_features)
        }
        
        return strategy
    
    def _classify_task_type(self, task_data: Dict[str, Any]) -> str:
        """Classify task type based on content"""
        title = task_data.get('title', '').lower()
        description = task_data.get('description', '').lower()
        details = task_data.get('details', '').lower()
        
        content = f"{title} {description} {details}"
        
        # Classification patterns
        if any(keyword in content for keyword in ['test', 'validation', 'verify']):
            return 'testing'
        elif any(keyword in content for keyword in ['implement', 'create', 'build']):
            return 'implementation'
        elif any(keyword in content for keyword in ['optimize', 'improve', 'enhance']):
            return 'optimization'
        elif any(keyword in content for keyword in ['refactor', 'reorganize', 'cleanup']):
            return 'refactoring'
        elif any(keyword in content for keyword in ['research', 'analyze', 'investigate']):
            return 'research'
        elif any(keyword in content for keyword in ['document', 'readme', 'guide']):
            return 'documentation'
        else:
            return 'general'
    
    def _calculate_complexity_score(self, task_data: Dict[str, Any]) -> float:
        """Calculate task complexity score"""
        score = 0.0
        
        # Length-based complexity
        title_len = len(task_data.get('title', ''))
        details_len = len(task_data.get('details', ''))
        score += min(title_len / 100, 1.0) * 0.2
        score += min(details_len / 1000, 1.0) * 0.3
        
        # Dependency complexity
        dependencies = task_data.get('dependencies', [])
        score += min(len(dependencies) / 5, 1.0) * 0.2
        
        # Subtask complexity
        subtasks = task_data.get('subtasks', [])
        score += min(len(subtasks) / 10, 1.0) * 0.3
        
        return min(score, 1.0)
    
    def _extract_context_features(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual features from task data"""
        return {
            'has_dependencies': len(task_data.get('dependencies', [])) > 0,
            'dependency_count': len(task_data.get('dependencies', [])),
            'has_subtasks': len(task_data.get('subtasks', [])) > 0,
            'subtask_count': len(task_data.get('subtasks', [])),
            'priority': task_data.get('priority', 'medium'),
            'title_length': len(task_data.get('title', '')),
            'details_length': len(task_data.get('details', '')),
            'has_test_strategy': bool(task_data.get('testStrategy')),
            'creation_hour': datetime.now().hour,
            'creation_day': datetime.now().weekday()
        }
    
    def _index_pattern(self, pattern: ExecutionPattern):
        """Index pattern for fast lookup"""
        # Index by task type
        self.pattern_index[f"type:{pattern.task_type}"].append(pattern)
        
        # Index by complexity range
        complexity_range = f"complexity:{int(pattern.complexity_score * 10)}"
        self.pattern_index[complexity_range].append(pattern)
        
        # Index by success
        success_key = f"success:{pattern.success_rate > 0.5}"
        self.pattern_index[success_key].append(pattern)
    
    def _trigger_learning_if_needed(self):
        """Trigger learning process if conditions are met"""
        if not self.current_model:
            self._perform_initial_learning()
        else:
            # Check if retraining is needed
            last_training = self.current_model.last_training
            if datetime.now() - last_training > self.retraining_interval:
                self._perform_incremental_learning()
    
    def _perform_initial_learning(self):
        """Perform initial learning from accumulated patterns"""
        print("🧠 Starting initial AI learning...")
        
        # Analyze patterns and extract insights
        insights = self._analyze_execution_patterns()
        
        # Generate optimization strategies
        strategies = self._generate_optimization_strategies(insights)
        
        # Create initial model
        self.current_model = LearningModel(
            model_version="1.0",
            training_data_size=len(self.execution_patterns),
            accuracy_score=0.0,  # Will be calculated after validation
            last_training=datetime.now(),
            feature_weights=insights.get('feature_weights', {}),
            optimization_strategies=strategies
        )
        
        # Save model and strategies
        self._save_model()
        self._save_strategies()
        
        print(f"✅ Initial learning completed: {len(strategies)} strategies generated")
    
    def _perform_incremental_learning(self):
        """Perform incremental learning with new patterns"""
        print("🔄 Performing incremental learning...")
        
        # Get new patterns since last training
        last_training = self.current_model.last_training
        new_patterns = [p for p in self.execution_patterns if p.timestamp > last_training]
        
        if len(new_patterns) < 10:
            return  # Not enough new data
        
        # Update insights with new data
        new_insights = self._analyze_execution_patterns(new_patterns)
        
        # Update existing strategies or create new ones
        self._update_optimization_strategies(new_insights)
        
        # Update model
        self.current_model.training_data_size += len(new_patterns)
        self.current_model.last_training = datetime.now()
        self.current_model.model_version = f"{self.current_model.model_version}.{int(time.time())}"
        
        # Save updates
        self._save_model()
        self._save_strategies()
        
        print(f"✅ Incremental learning completed: {len(new_patterns)} new patterns processed")
    
    def _analyze_execution_patterns(self, patterns: List[ExecutionPattern] = None) -> Dict[str, Any]:
        """Analyze execution patterns to extract insights"""
        if patterns is None:
            patterns = list(self.execution_patterns)
        
        insights = {
            'total_patterns': len(patterns),
            'success_rate_by_type': {},
            'avg_execution_time_by_type': {},
            'complexity_distribution': {},
            'feature_weights': {},
            'correlation_matrix': {}
        }
        
        # Group patterns by type
        by_type = defaultdict(list)
        for pattern in patterns:
            by_type[pattern.task_type].append(pattern)
        
        # Calculate success rates by type
        for task_type, type_patterns in by_type.items():
            success_rate = sum(p.success_rate for p in type_patterns) / len(type_patterns)
            avg_time = sum(p.execution_time for p in type_patterns) / len(type_patterns)
            
            insights['success_rate_by_type'][task_type] = success_rate
            insights['avg_execution_time_by_type'][task_type] = avg_time
        
        # Analyze complexity distribution
        complexity_scores = [p.complexity_score for p in patterns]
        if complexity_scores:
            n = len(complexity_scores)
            mean_score = sum(complexity_scores) / n
            variance = sum((x - mean_score) ** 2 for x in complexity_scores) / n
            std_score = variance ** 0.5
            sorted_scores = sorted(complexity_scores)
            median_score = sorted_scores[n // 2] if n % 2 else (sorted_scores[n//2-1] + sorted_scores[n//2]) / 2
            
            insights['complexity_distribution'] = {
                'mean': mean_score,
                'std': std_score,
                'median': median_score
            }
        
        return insights
    
    def _generate_optimization_strategies(self, insights: Dict[str, Any]) -> List[OptimizationStrategy]:
        """Generate optimization strategies from insights"""
        strategies = []
        
        # Strategy 1: Task type optimization
        for task_type, success_rate in insights['success_rate_by_type'].items():
            if success_rate < 0.8:  # Low success rate
                strategy = OptimizationStrategy(
                    strategy_id=f"improve_{task_type}_success",
                    pattern_fingerprint=hashlib.md5(f"{task_type}_optimization".encode()).hexdigest()[:8],
                    success_rate=success_rate,
                    avg_speedup=1.2,
                    confidence_score=0.8,
                    applicable_conditions={'task_type': task_type},
                    optimization_params={
                        'type': 'success_improvement',
                        'target_task_type': task_type,
                        'recommended_actions': ['increase_testing', 'add_validation', 'simplify_requirements']
                    },
                    usage_count=0
                )
                strategies.append(strategy)
        
        # Strategy 2: Execution time optimization
        for task_type, avg_time in insights['avg_execution_time_by_type'].items():
            if avg_time > 300:  # Slow execution (5+ minutes)
                strategy = OptimizationStrategy(
                    strategy_id=f"speed_up_{task_type}",
                    pattern_fingerprint=hashlib.md5(f"{task_type}_speed".encode()).hexdigest()[:8],
                    success_rate=0.9,
                    avg_speedup=1.5,
                    confidence_score=0.7,
                    applicable_conditions={'task_type': task_type},
                    optimization_params={
                        'type': 'speed_optimization',
                        'target_task_type': task_type,
                        'recommended_actions': ['parallel_execution', 'caching', 'resource_preallocation']
                    },
                    usage_count=0
                )
                strategies.append(strategy)
        
        return strategies
    
    def _strategy_applies_to_task(self, strategy: OptimizationStrategy, 
                                 task_features: Dict[str, Any], task_type: str) -> bool:
        """Check if optimization strategy applies to given task"""
        conditions = strategy.applicable_conditions
        
        # Check task type
        if 'task_type' in conditions and conditions['task_type'] != task_type:
            return False
        
        # Check other conditions
        for condition, value in conditions.items():
            if condition == 'task_type':
                continue
            if condition in task_features and task_features[condition] != value:
                return False
        
        return True
    
    def _get_pattern_based_recommendations(self, task_features: Dict[str, Any], 
                                         complexity: float, task_type: str) -> List[Dict[str, Any]]:
        """Get recommendations based on similar historical patterns"""
        recommendations = []
        
        # Find similar patterns
        similar_patterns = []
        for pattern in self.execution_patterns:
            if (pattern.task_type == task_type and 
                abs(pattern.complexity_score - complexity) < 0.2):
                similar_patterns.append(pattern)
        
        if not similar_patterns:
            return recommendations
        
        # Analyze successful patterns
        successful_patterns = [p for p in similar_patterns if p.success_rate > 0.8]
        if successful_patterns:
            avg_time = sum(p.execution_time for p in successful_patterns) / len(successful_patterns)
            
            recommendations.append({
                'strategy_id': 'pattern_based_timing',
                'optimization_type': 'timing_prediction',
                'expected_speedup': 1.1,
                'confidence': 0.6,
                'parameters': {'estimated_time': avg_time},
                'description': f'Based on {len(successful_patterns)} similar successful tasks'
            })
        
        return recommendations
    
    def _get_default_generation_strategy(self) -> Dict[str, Any]:
        """Get default task generation strategy"""
        return {
            'suggested_task_count': 5,
            'priority_distribution': {'high': 0.3, 'medium': 0.5, 'low': 0.2},
            'complexity_targets': {'simple': 0.4, 'moderate': 0.4, 'complex': 0.2},
            'dependency_patterns': 'sequential',
            'execution_order_hints': 'dependency_first'
        }
    
    def _extract_project_features(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract project-level features"""
        return {
            'project_type': project_context.get('project_type', 'general'),
            'team_size': project_context.get('team_size', 1),
            'has_deadline': 'deadline' in project_context,
            'complexity_level': project_context.get('complexity', 'medium')
        }
    
    def _predict_optimal_task_count(self, features: Dict[str, Any]) -> int:
        """Predict optimal number of tasks"""
        base_count = 5
        team_size = features.get('team_size', 1)
        return max(3, min(15, base_count * team_size))
    
    def _predict_priority_distribution(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Predict optimal priority distribution"""
        if features.get('has_deadline', False):
            return {'high': 0.4, 'medium': 0.4, 'low': 0.2}
        return {'high': 0.2, 'medium': 0.6, 'low': 0.2}
    
    def _predict_complexity_targets(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Predict complexity distribution targets"""
        return {'simple': 0.4, 'moderate': 0.4, 'complex': 0.2}
    
    def _predict_dependency_patterns(self, features: Dict[str, Any]) -> str:
        """Predict optimal dependency patterns"""
        team_size = features.get('team_size', 1)
        return 'parallel' if team_size > 2 else 'sequential'
    
    def _predict_execution_order(self, features: Dict[str, Any]) -> str:
        """Predict optimal execution order"""
        return 'dependency_first'
    
    def _generate_strategy_description(self, strategy: OptimizationStrategy) -> str:
        """Generate human-readable strategy description"""
        opt_type = strategy.optimization_params.get('type', 'general')
        if opt_type == 'success_improvement':
            return f"Improve success rate for {strategy.optimization_params.get('target_task_type', 'tasks')}"
        elif opt_type == 'speed_optimization':
            return f"Speed up execution for {strategy.optimization_params.get('target_task_type', 'tasks')}"
        return "General optimization strategy"
    
    def _update_optimization_strategies(self, new_insights: Dict[str, Any]):
        """Update existing optimization strategies with new insights"""
        # Simple implementation - can be expanded
        for strategy_id, strategy in self.optimization_strategies.items():
            strategy.usage_count += 1
    
    def _save_patterns(self):
        """Save execution patterns to disk"""
        try:
            pattern_data = []
            for pattern in list(self.execution_patterns)[-1000:]:  # Save last 1000
                data = asdict(pattern)
                data['timestamp'] = pattern.timestamp.isoformat()
                pattern_data.append(data)
            
            with open(self.patterns_file, 'w') as f:
                json.dump(pattern_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving patterns: {e}")
    
    def _save_model(self):
        """Save learning model to disk"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.current_model, f)
        except Exception as e:
            print(f"⚠️ Error saving model: {e}")
    
    def _save_strategies(self):
        """Save optimization strategies to disk"""
        try:
            strategies_data = []
            for strategy in self.optimization_strategies.values():
                strategies_data.append(asdict(strategy))
            
            with open(self.strategies_file, 'w') as f:
                json.dump(strategies_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving strategies: {e}")

def main():
    """Demo of adaptive learning engine"""
    print("Adaptive Learning Engine Demo")
    print("=" * 40)
    
    engine = AdaptiveLearningEngine()
    
    # Demo: Record some execution patterns
    demo_tasks = [
        {
            'id': 'demo-1',
            'title': 'Implement user authentication',
            'description': 'Create JWT-based auth system',
            'details': 'Use bcrypt for hashing, implement login/logout endpoints',
            'priority': 'high',
            'dependencies': [],
            'subtasks': []
        },
        {
            'id': 'demo-2', 
            'title': 'Optimize database queries',
            'description': 'Improve query performance',
            'details': 'Add indexes, optimize JOIN operations, implement caching',
            'priority': 'medium',
            'dependencies': ['demo-1'],
            'subtasks': []
        }
    ]
    
    demo_results = [
        {'success': True, 'execution_time': 240, 'resource_usage': {'memory': 150, 'cpu': 45}},
        {'success': True, 'execution_time': 180, 'resource_usage': {'memory': 120, 'cpu': 35}}
    ]
    
    # Record patterns
    for task, result in zip(demo_tasks, demo_results):
        pattern_id = engine.record_execution_pattern(task, result)
        print(f"✅ Recorded pattern: {pattern_id}")
    
    # Get recommendations
    new_task = {
        'title': 'Create API endpoints',
        'description': 'Build REST API for user management',
        'details': 'Implement CRUD operations with validation',
        'priority': 'high'
    }
    
    recommendations = engine.get_optimization_recommendations(new_task)
    print(f"\n📊 Optimization recommendations for new task:")
    for rec in recommendations:
        print(f"  • {rec['optimization_type']}: {rec['description']}")
        print(f"    Confidence: {rec['confidence']:.1%}, Expected speedup: {rec['expected_speedup']:.1f}x")
    
    # Adapt generation strategy
    project_context = {'project_type': 'web_app', 'team_size': 3, 'deadline': '2 weeks'}
    strategy = engine.adapt_task_generation_strategy(project_context)
    print(f"\n🎯 Adapted task generation strategy:")
    print(f"  • Suggested task count: {strategy.get('suggested_task_count', 'adaptive')}")
    print(f"  • Priority distribution: {strategy.get('priority_distribution', {})}")
    
    print(f"\n✅ Adaptive learning engine demo completed")

if __name__ == "__main__":
    main()