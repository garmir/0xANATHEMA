#!/usr/bin/env python3
"""
Intelligent Task Prediction and Auto-Generation System
AI-powered analysis of project patterns for autonomous task creation
"""

import os
import sys
import json
import time
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import networkx as nx
from collections import defaultdict, Counter
import sqlite3
from flask import Flask, jsonify, request
import threading

@dataclass
class TaskPattern:
    """Represents a discovered task pattern"""
    pattern_id: str
    pattern_type: str  # "sequence", "dependency", "temporal", "semantic"
    confidence: float
    frequency: int
    tasks_involved: List[str]
    triggers: List[str]
    typical_duration: float
    success_rate: float
    description: str

@dataclass
class TaskPrediction:
    """Represents a predicted task"""
    prediction_id: str
    predicted_title: str
    predicted_description: str
    predicted_details: str
    confidence_score: float
    reasoning: str
    suggested_priority: str
    estimated_duration: float
    suggested_dependencies: List[str]
    prediction_timestamp: datetime
    pattern_sources: List[str]
    auto_generated: bool = True

@dataclass
class UserBehavior:
    """Tracks user behavior patterns"""
    user_id: str
    task_completion_patterns: Dict[str, float]
    preferred_priorities: Dict[str, float]
    work_time_patterns: Dict[int, float]  # hour -> activity level
    task_sequence_preferences: List[Tuple[str, str]]
    rejection_patterns: Dict[str, int]
    acceptance_patterns: Dict[str, int]
    coding_patterns: Dict[str, float]

class PatternAnalysisModule:
    """Analyzes task completion patterns and workflows"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.task_sequences = []
        self.dependency_patterns = {}
        self.temporal_patterns = {}
        self.semantic_clusters = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.kmeans_model = None
        
    def analyze_task_history(self, task_history: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Analyze historical tasks to discover patterns"""
        patterns = []
        
        # Temporal patterns
        temporal_patterns = self._find_temporal_patterns(task_history)
        patterns.extend(temporal_patterns)
        
        # Sequence patterns
        sequence_patterns = self._find_sequence_patterns(task_history)
        patterns.extend(sequence_patterns)
        
        # Dependency patterns
        dependency_patterns = self._find_dependency_patterns(task_history)
        patterns.extend(dependency_patterns)
        
        # Semantic patterns
        semantic_patterns = self._find_semantic_patterns(task_history)
        patterns.extend(semantic_patterns)
        
        self.logger.info(f"Discovered {len(patterns)} task patterns")
        return patterns
    
    def _find_temporal_patterns(self, task_history: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Find patterns in task timing and completion"""
        patterns = []
        
        # Group tasks by completion time
        completion_times = defaultdict(list)
        for task in task_history:
            if task.get('status') == 'done' and 'completed_at' in task:
                completed_at = datetime.fromisoformat(task['completed_at'])
                hour = completed_at.hour
                day_of_week = completed_at.weekday()
                completion_times[f"hour_{hour}"].append(task)
                completion_times[f"day_{day_of_week}"].append(task)
        
        # Find significant temporal clusters
        for time_key, tasks in completion_times.items():
            if len(tasks) >= 3:  # Minimum for pattern
                task_types = [self._extract_task_type(task) for task in tasks]
                most_common_type = Counter(task_types).most_common(1)[0]
                
                if most_common_type[1] >= 2:  # At least 2 occurrences
                    pattern = TaskPattern(
                        pattern_id=f"temporal_{time_key}_{most_common_type[0]}",
                        pattern_type="temporal",
                        confidence=most_common_type[1] / len(tasks),
                        frequency=most_common_type[1],
                        tasks_involved=[t['id'] for t in tasks if self._extract_task_type(t) == most_common_type[0]],
                        triggers=[time_key],
                        typical_duration=np.mean([self._estimate_task_duration(t) for t in tasks]),
                        success_rate=1.0,  # All completed tasks
                        description=f"Tasks of type '{most_common_type[0]}' typically completed during {time_key}"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_sequence_patterns(self, task_history: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Find patterns in task sequences"""
        patterns = []
        
        # Sort tasks by completion time
        completed_tasks = [t for t in task_history if t.get('status') == 'done' and 'completed_at' in t]
        completed_tasks.sort(key=lambda x: x.get('completed_at', ''))
        
        # Extract task sequences
        sequences = []
        window_size = 3
        
        for i in range(len(completed_tasks) - window_size + 1):
            sequence = completed_tasks[i:i + window_size]
            task_types = [self._extract_task_type(task) for task in sequence]
            sequences.append(task_types)
        
        # Find common sequences
        sequence_counts = Counter(tuple(seq) for seq in sequences)
        
        for sequence, count in sequence_counts.items():
            if count >= 2:  # Minimum for pattern
                pattern = TaskPattern(
                    pattern_id=f"sequence_{'_'.join(sequence)}",
                    pattern_type="sequence",
                    confidence=count / len(sequences),
                    frequency=count,
                    tasks_involved=list(sequence),
                    triggers=[sequence[0]],  # First task triggers the sequence
                    typical_duration=sum(self._estimate_task_duration_by_type(t) for t in sequence),
                    success_rate=1.0,
                    description=f"Task sequence: {' -> '.join(sequence)}"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_dependency_patterns(self, task_history: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Find patterns in task dependencies"""
        patterns = []
        
        # Build dependency graph
        dependency_graph = nx.DiGraph()
        dependency_pairs = []
        
        for task in task_history:
            task_id = str(task.get('id', ''))
            dependencies = task.get('dependencies', [])
            
            for dep in dependencies:
                dependency_graph.add_edge(str(dep), task_id)
                dependency_pairs.append((str(dep), task_id))
        
        # Find common dependency patterns
        if dependency_pairs:
            pair_counts = Counter(dependency_pairs)
            
            for (dep, task), count in pair_counts.items():
                if count >= 2:
                    pattern = TaskPattern(
                        pattern_id=f"dependency_{dep}_{task}",
                        pattern_type="dependency",
                        confidence=count / len(dependency_pairs),
                        frequency=count,
                        tasks_involved=[dep, task],
                        triggers=[dep],
                        typical_duration=self._estimate_task_duration_by_id(task_history, task),
                        success_rate=1.0,
                        description=f"Task {task} typically depends on {dep}"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_semantic_patterns(self, task_history: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Find patterns in task semantics using NLP"""
        patterns = []
        
        # Extract text features
        task_texts = []
        task_ids = []
        
        for task in task_history:
            text = f"{task.get('title', '')} {task.get('description', '')} {task.get('details', '')}"
            if text.strip():
                task_texts.append(text)
                task_ids.append(str(task.get('id', '')))
        
        if len(task_texts) < 3:
            return patterns
        
        try:
            # Vectorize text
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(task_texts)
            
            # Cluster similar tasks
            n_clusters = min(5, len(task_texts) // 2)
            if n_clusters >= 2:
                self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = self.kmeans_model.fit_predict(tfidf_matrix)
                
                # Create patterns for each cluster
                for cluster_id in range(n_clusters):
                    cluster_tasks = [task_ids[i] for i, c in enumerate(clusters) if c == cluster_id]
                    
                    if len(cluster_tasks) >= 2:
                        # Get representative terms for this cluster
                        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                        cluster_vectors = tfidf_matrix[cluster_indices]
                        centroid = cluster_vectors.mean(axis=0).A1
                        
                        # Get top terms
                        feature_names = self.tfidf_vectorizer.get_feature_names_out()
                        top_terms_indices = centroid.argsort()[-5:][::-1]
                        top_terms = [feature_names[i] for i in top_terms_indices]
                        
                        pattern = TaskPattern(
                            pattern_id=f"semantic_cluster_{cluster_id}",
                            pattern_type="semantic",
                            confidence=len(cluster_tasks) / len(task_texts),
                            frequency=len(cluster_tasks),
                            tasks_involved=cluster_tasks,
                            triggers=top_terms[:2],
                            typical_duration=np.mean([self._estimate_task_duration_by_id(task_history, tid) for tid in cluster_tasks]),
                            success_rate=1.0,
                            description=f"Tasks related to: {', '.join(top_terms[:3])}"
                        )
                        patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error in semantic pattern analysis: {e}")
        
        return patterns
    
    def _extract_task_type(self, task: Dict[str, Any]) -> str:
        """Extract task type from task data"""
        title = task.get('title', '').lower()
        description = task.get('description', '').lower()
        
        # Simple keyword-based classification
        if any(word in title + description for word in ['implement', 'create', 'build', 'develop']):
            return 'implementation'
        elif any(word in title + description for word in ['test', 'validate', 'verify']):
            return 'testing'
        elif any(word in title + description for word in ['fix', 'debug', 'resolve']):
            return 'debugging'
        elif any(word in title + description for word in ['document', 'write', 'update']):
            return 'documentation'
        elif any(word in title + description for word in ['setup', 'configure', 'install']):
            return 'setup'
        else:
            return 'general'
    
    def _estimate_task_duration(self, task: Dict[str, Any]) -> float:
        """Estimate task duration in hours"""
        # Simple heuristic based on task complexity
        details_length = len(task.get('details', ''))
        base_duration = 2.0  # hours
        
        if details_length > 1000:
            return base_duration * 3
        elif details_length > 500:
            return base_duration * 2
        else:
            return base_duration
    
    def _estimate_task_duration_by_type(self, task_type: str) -> float:
        """Estimate duration by task type"""
        durations = {
            'implementation': 4.0,
            'testing': 2.0,
            'debugging': 3.0,
            'documentation': 1.5,
            'setup': 1.0,
            'general': 2.0
        }
        return durations.get(task_type, 2.0)
    
    def _estimate_task_duration_by_id(self, task_history: List[Dict[str, Any]], task_id: str) -> float:
        """Estimate duration for specific task ID"""
        for task in task_history:
            if str(task.get('id', '')) == task_id:
                return self._estimate_task_duration(task)
        return 2.0

class TrajectoryPredictionEngine:
    """Predicts development trajectory using ML models"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.priority_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.duration_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.next_task_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
    
    def train_models(self, task_history: List[Dict[str, Any]], patterns: List[TaskPattern]):
        """Train ML models on historical data"""
        if len(task_history) < 10:
            self.logger.warning("Insufficient data for training trajectory models")
            return False
        
        try:
            # Prepare features and labels
            features, priority_labels, duration_labels, next_task_labels = self._prepare_training_data(task_history)
            
            if len(features) < 5:
                return False
            
            # Encode categorical labels
            self.label_encoders['priority'] = LabelEncoder()
            priority_encoded = self.label_encoders['priority'].fit_transform(priority_labels)
            
            self.label_encoders['next_task'] = LabelEncoder()
            next_task_encoded = self.label_encoders['next_task'].fit_transform(next_task_labels)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train models
            self.priority_classifier.fit(features_scaled, priority_encoded)
            self.duration_regressor.fit(features_scaled, duration_labels)
            
            if len(set(next_task_encoded)) > 1:  # Ensure multiple classes
                self.next_task_classifier.fit(features_scaled, next_task_encoded)
            
            self.is_trained = True
            self.logger.info("Trajectory prediction models trained successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training trajectory models: {e}")
            return False
    
    def predict_next_tasks(self, current_context: Dict[str, Any], patterns: List[TaskPattern]) -> List[TaskPrediction]:
        """Predict next likely tasks"""
        predictions = []
        
        if not self.is_trained:
            return predictions
        
        try:
            # Extract features from current context
            features = self._extract_context_features(current_context)
            features_scaled = self.scaler.transform([features])
            
            # Predict priority
            priority_prob = self.priority_classifier.predict_proba(features_scaled)[0]
            priority_classes = self.label_encoders['priority'].classes_
            predicted_priority = priority_classes[np.argmax(priority_prob)]
            
            # Predict duration
            predicted_duration = max(0.5, self.duration_regressor.predict(features_scaled)[0])
            
            # Generate predictions based on patterns
            for pattern in patterns:
                if pattern.confidence > 0.3:  # Only use high-confidence patterns
                    prediction = self._generate_prediction_from_pattern(
                        pattern, predicted_priority, predicted_duration, current_context
                    )
                    if prediction:
                        predictions.append(prediction)
            
            # Sort by confidence
            predictions.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return predictions[:5]  # Return top 5 predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting next tasks: {e}")
            return predictions
    
    def _prepare_training_data(self, task_history: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[str], List[float], List[str]]:
        """Prepare training data from task history"""
        features = []
        priority_labels = []
        duration_labels = []
        next_task_labels = []
        
        sorted_tasks = sorted(task_history, key=lambda x: x.get('created_at', ''))
        
        for i, task in enumerate(sorted_tasks[:-1]):  # Exclude last task (no next task)
            try:
                # Extract features
                feature_vector = self._extract_task_features(task, sorted_tasks[:i])
                
                # Labels
                priority = task.get('priority', 'medium')
                duration = self._estimate_duration_from_task(task)
                next_task_type = self._extract_task_type_from_dict(sorted_tasks[i + 1]) if i + 1 < len(sorted_tasks) else 'end'
                
                features.append(feature_vector)
                priority_labels.append(priority)
                duration_labels.append(duration)
                next_task_labels.append(next_task_type)
                
            except Exception as e:
                self.logger.warning(f"Error processing task {task.get('id', 'unknown')}: {e}")
                continue
        
        return features, priority_labels, duration_labels, next_task_labels
    
    def _extract_task_features(self, task: Dict[str, Any], previous_tasks: List[Dict[str, Any]]) -> List[float]:
        """Extract numerical features from task"""
        features = []
        
        # Task characteristics
        title_length = len(task.get('title', ''))
        description_length = len(task.get('description', ''))
        details_length = len(task.get('details', ''))
        dependency_count = len(task.get('dependencies', []))
        
        features.extend([title_length, description_length, details_length, dependency_count])
        
        # Task type (one-hot encoded)
        task_type = self._extract_task_type_from_dict(task)
        task_types = ['implementation', 'testing', 'debugging', 'documentation', 'setup', 'general']
        for t_type in task_types:
            features.append(1.0 if task_type == t_type else 0.0)
        
        # Context features
        recent_task_count = len(previous_tasks)
        features.append(min(recent_task_count, 10))  # Cap at 10
        
        # Recent task types
        recent_types = [self._extract_task_type_from_dict(t) for t in previous_tasks[-3:]]
        for t_type in task_types:
            features.append(recent_types.count(t_type))
        
        return features
    
    def _extract_context_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract features from current context"""
        features = []
        
        # Current project state
        total_tasks = context.get('total_tasks', 0)
        completed_tasks = context.get('completed_tasks', 0)
        pending_tasks = context.get('pending_tasks', 0)
        
        features.extend([total_tasks, completed_tasks, pending_tasks])
        
        # Progress ratio
        progress_ratio = completed_tasks / max(total_tasks, 1)
        features.append(progress_ratio)
        
        # Recent activity
        recent_completions = context.get('recent_completions', 0)
        features.append(recent_completions)
        
        # Pad with zeros if needed
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]  # Ensure consistent feature size
    
    def _generate_prediction_from_pattern(self, pattern: TaskPattern, predicted_priority: str, 
                                        predicted_duration: float, context: Dict[str, Any]) -> Optional[TaskPrediction]:
        """Generate task prediction from pattern"""
        try:
            prediction_id = f"pred_{pattern.pattern_id}_{int(time.time())}"
            
            # Generate task based on pattern type
            if pattern.pattern_type == "sequence":
                title = f"Continue {pattern.description.split(' -> ')[-1]} workflow"
                description = f"Next step in {pattern.description} sequence"
                details = f"Based on pattern analysis, this task follows the {pattern.description} sequence with {pattern.confidence:.1%} confidence."
                
            elif pattern.pattern_type == "semantic":
                title = f"Implement {pattern.triggers[0]} functionality"
                description = f"Task related to {', '.join(pattern.triggers)}"
                details = f"Semantic analysis suggests a task involving {', '.join(pattern.triggers)} based on similar completed tasks."
                
            elif pattern.pattern_type == "temporal":
                title = f"Scheduled {pattern.triggers[0]} task"
                description = f"Task typically performed during {pattern.triggers[0]}"
                details = f"Temporal pattern analysis indicates this type of task is usually completed {pattern.triggers[0]}."
                
            else:  # dependency pattern
                title = f"Follow-up task for {pattern.tasks_involved[0]}"
                description = f"Dependent task based on completion of {pattern.tasks_involved[0]}"
                details = f"Dependency analysis suggests this task typically follows {pattern.tasks_involved[0]}."
            
            reasoning = f"Generated from {pattern.pattern_type} pattern with {pattern.frequency} occurrences and {pattern.confidence:.1%} confidence"
            
            prediction = TaskPrediction(
                prediction_id=prediction_id,
                predicted_title=title,
                predicted_description=description,
                predicted_details=details,
                confidence_score=pattern.confidence * 0.8,  # Slight discount for being predicted
                reasoning=reasoning,
                suggested_priority=predicted_priority,
                estimated_duration=predicted_duration,
                suggested_dependencies=[],
                prediction_timestamp=datetime.now(),
                pattern_sources=[pattern.pattern_id]
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error generating prediction from pattern: {e}")
            return None
    
    def _extract_task_type_from_dict(self, task: Dict[str, Any]) -> str:
        """Extract task type from task dictionary"""
        title = task.get('title', '').lower()
        description = task.get('description', '').lower()
        
        if any(word in title + description for word in ['implement', 'create', 'build', 'develop']):
            return 'implementation'
        elif any(word in title + description for word in ['test', 'validate', 'verify']):
            return 'testing'
        elif any(word in title + description for word in ['fix', 'debug', 'resolve']):
            return 'debugging'
        elif any(word in title + description for word in ['document', 'write', 'update']):
            return 'documentation'
        elif any(word in title + description for word in ['setup', 'configure', 'install']):
            return 'setup'
        else:
            return 'general'
    
    def _estimate_duration_from_task(self, task: Dict[str, Any]) -> float:
        """Estimate duration from task data"""
        complexity_indicators = len(task.get('details', ''))
        
        if complexity_indicators > 1000:
            return 6.0  # hours
        elif complexity_indicators > 500:
            return 4.0
        elif complexity_indicators > 200:
            return 2.0
        else:
            return 1.0

class BehavioralLearningSystem:
    """Learns user behavior patterns and preferences"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.user_behaviors: Dict[str, UserBehavior] = {}
        self.feedback_history = []
    
    def track_user_behavior(self, user_id: str, action: str, task_data: Dict[str, Any], context: Dict[str, Any]):
        """Track user behavior for learning"""
        if user_id not in self.user_behaviors:
            self.user_behaviors[user_id] = UserBehavior(
                user_id=user_id,
                task_completion_patterns={},
                preferred_priorities={},
                work_time_patterns={},
                task_sequence_preferences=[],
                rejection_patterns={},
                acceptance_patterns={},
                coding_patterns={}
            )
        
        behavior = self.user_behaviors[user_id]
        
        if action == "task_completed":
            self._update_completion_patterns(behavior, task_data, context)
        elif action == "task_accepted":
            self._update_acceptance_patterns(behavior, task_data)
        elif action == "task_rejected":
            self._update_rejection_patterns(behavior, task_data)
        elif action == "priority_changed":
            self._update_priority_preferences(behavior, task_data)
    
    def _update_completion_patterns(self, behavior: UserBehavior, task_data: Dict[str, Any], context: Dict[str, Any]):
        """Update task completion patterns"""
        task_type = self._extract_task_type(task_data)
        
        # Update completion patterns
        if task_type not in behavior.task_completion_patterns:
            behavior.task_completion_patterns[task_type] = 0.0
        behavior.task_completion_patterns[task_type] += 1.0
        
        # Update work time patterns
        current_hour = datetime.now().hour
        if current_hour not in behavior.work_time_patterns:
            behavior.work_time_patterns[current_hour] = 0.0
        behavior.work_time_patterns[current_hour] += 1.0
    
    def _update_acceptance_patterns(self, behavior: UserBehavior, task_data: Dict[str, Any]):
        """Update task acceptance patterns"""
        task_type = self._extract_task_type(task_data)
        priority = task_data.get('priority', 'medium')
        
        pattern_key = f"{task_type}_{priority}"
        if pattern_key not in behavior.acceptance_patterns:
            behavior.acceptance_patterns[pattern_key] = 0
        behavior.acceptance_patterns[pattern_key] += 1
    
    def _update_rejection_patterns(self, behavior: UserBehavior, task_data: Dict[str, Any]):
        """Update task rejection patterns"""
        task_type = self._extract_task_type(task_data)
        priority = task_data.get('priority', 'medium')
        
        pattern_key = f"{task_type}_{priority}"
        if pattern_key not in behavior.rejection_patterns:
            behavior.rejection_patterns[pattern_key] = 0
        behavior.rejection_patterns[pattern_key] += 1
    
    def _update_priority_preferences(self, behavior: UserBehavior, task_data: Dict[str, Any]):
        """Update priority preferences"""
        priority = task_data.get('priority', 'medium')
        if priority not in behavior.preferred_priorities:
            behavior.preferred_priorities[priority] = 0.0
        behavior.preferred_priorities[priority] += 1.0
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preference summary"""
        if user_id not in self.user_behaviors:
            return {}
        
        behavior = self.user_behaviors[user_id]
        
        # Calculate preferences
        preferred_task_types = sorted(
            behavior.task_completion_patterns.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        preferred_priorities = sorted(
            behavior.preferred_priorities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        most_active_hours = sorted(
            behavior.work_time_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            "preferred_task_types": [t[0] for t in preferred_task_types],
            "preferred_priorities": [p[0] for p in preferred_priorities],
            "most_active_hours": [h[0] for h in most_active_hours],
            "acceptance_rate": self._calculate_acceptance_rate(behavior),
            "task_completion_velocity": sum(behavior.task_completion_patterns.values())
        }
    
    def _calculate_acceptance_rate(self, behavior: UserBehavior) -> float:
        """Calculate overall acceptance rate"""
        total_accepted = sum(behavior.acceptance_patterns.values())
        total_rejected = sum(behavior.rejection_patterns.values())
        total_feedback = total_accepted + total_rejected
        
        if total_feedback == 0:
            return 0.5  # Default
        
        return total_accepted / total_feedback
    
    def _extract_task_type(self, task_data: Dict[str, Any]) -> str:
        """Extract task type from task data"""
        title = task_data.get('title', '').lower()
        description = task_data.get('description', '').lower()
        
        if any(word in title + description for word in ['implement', 'create', 'build']):
            return 'implementation'
        elif any(word in title + description for word in ['test', 'validate']):
            return 'testing'
        elif any(word in title + description for word in ['fix', 'debug']):
            return 'debugging'
        elif any(word in title + description for word in ['document', 'write']):
            return 'documentation'
        else:
            return 'general'

class IntelligentTaskPredictionSystem:
    """Main system orchestrating all prediction components"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.db_path = Path(self.config.get('db_path', '.taskmaster/intelligence/predictions.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pattern_analyzer = PatternAnalysisModule(self.logger)
        self.trajectory_engine = TrajectoryPredictionEngine(self.logger)
        self.behavior_system = BehavioralLearningSystem(self.logger)
        
        # Initialize database
        self._init_database()
        
        # Flask app for API
        self.app = Flask(__name__)
        self._setup_api_routes()
        
        # Load existing data
        self._load_historical_data()
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            "db_path": ".taskmaster/intelligence/predictions.db",
            "api_port": 8081,
            "log_level": "INFO",
            "prediction_threshold": 0.5,
            "max_predictions": 10,
            "auto_generate_enabled": True,
            "learning_enabled": True
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("TaskPredictionSystem")
        logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        log_file = Path(".taskmaster/intelligence/prediction.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _init_database(self):
        """Initialize database for predictions"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS task_predictions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    details TEXT,
                    confidence_score REAL,
                    reasoning TEXT,
                    suggested_priority TEXT,
                    estimated_duration REAL,
                    suggested_dependencies TEXT,
                    prediction_timestamp DATETIME,
                    pattern_sources TEXT,
                    status TEXT DEFAULT 'pending',
                    user_feedback TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT,
                    user_id TEXT,
                    feedback_type TEXT,
                    feedback_data TEXT,
                    timestamp DATETIME
                )
            ''')
    
    def _load_historical_data(self):
        """Load historical task data for training"""
        try:
            tasks_file = Path('.taskmaster/tasks/tasks.json')
            if tasks_file.exists():
                with open(tasks_file, 'r') as f:
                    data = json.load(f)
                    
                tasks = []
                for tag_data in data.values():
                    if 'tasks' in tag_data:
                        tasks.extend(tag_data['tasks'])
                
                # Analyze patterns
                patterns = self.pattern_analyzer.analyze_task_history(tasks)
                
                # Train trajectory models
                self.trajectory_engine.train_models(tasks, patterns)
                
                self.logger.info(f"Loaded {len(tasks)} historical tasks and discovered {len(patterns)} patterns")
                
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
    
    def generate_predictions(self, user_id: str = "default") -> List[TaskPrediction]:
        """Generate task predictions"""
        try:
            # Get current project context
            context = self._get_current_context()
            
            # Get user preferences
            user_prefs = self.behavior_system.get_user_preferences(user_id)
            
            # Get patterns (simplified - would load from analysis)
            patterns = []  # In production, load discovered patterns
            
            # Generate predictions
            predictions = self.trajectory_engine.predict_next_tasks(context, patterns)
            
            # Apply user preferences
            predictions = self._apply_user_preferences(predictions, user_prefs)
            
            # Store predictions
            self._store_predictions(predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return []
    
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current project context"""
        try:
            tasks_file = Path('.taskmaster/tasks/tasks.json')
            if tasks_file.exists():
                with open(tasks_file, 'r') as f:
                    data = json.load(f)
                    
                total_tasks = 0
                completed_tasks = 0
                pending_tasks = 0
                
                for tag_data in data.values():
                    if 'tasks' in tag_data:
                        tasks = tag_data['tasks']
                        total_tasks += len(tasks)
                        completed_tasks += len([t for t in tasks if t.get('status') == 'done'])
                        pending_tasks += len([t for t in tasks if t.get('status') == 'pending'])
                
                return {
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'pending_tasks': pending_tasks,
                    'recent_completions': completed_tasks  # Simplified
                }
        except Exception:
            pass
        
        return {
            'total_tasks': 0,
            'completed_tasks': 0,
            'pending_tasks': 0,
            'recent_completions': 0
        }
    
    def _apply_user_preferences(self, predictions: List[TaskPrediction], user_prefs: Dict[str, Any]) -> List[TaskPrediction]:
        """Apply user preferences to predictions"""
        if not user_prefs:
            return predictions
        
        # Adjust confidence based on user preferences
        preferred_types = user_prefs.get('preferred_task_types', [])
        
        for prediction in predictions:
            # Simple heuristic: boost confidence for preferred task types
            for pref_type in preferred_types:
                if pref_type.lower() in prediction.predicted_title.lower():
                    prediction.confidence_score = min(1.0, prediction.confidence_score * 1.2)
                    break
        
        return sorted(predictions, key=lambda x: x.confidence_score, reverse=True)
    
    def _store_predictions(self, predictions: List[TaskPrediction]):
        """Store predictions in database"""
        with sqlite3.connect(self.db_path) as conn:
            for pred in predictions:
                conn.execute('''
                    INSERT OR REPLACE INTO task_predictions 
                    (id, title, description, details, confidence_score, reasoning,
                     suggested_priority, estimated_duration, suggested_dependencies,
                     prediction_timestamp, pattern_sources)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pred.prediction_id,
                    pred.predicted_title,
                    pred.predicted_description,
                    pred.predicted_details,
                    pred.confidence_score,
                    pred.reasoning,
                    pred.suggested_priority,
                    pred.estimated_duration,
                    json.dumps(pred.suggested_dependencies),
                    pred.prediction_timestamp.isoformat(),
                    json.dumps(pred.pattern_sources)
                ))
    
    def _setup_api_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/api/predictions', methods=['GET'])
        def get_predictions():
            user_id = request.args.get('user_id', 'default')
            predictions = self.generate_predictions(user_id)
            return jsonify([asdict(p) for p in predictions])
        
        @self.app.route('/api/feedback', methods=['POST'])
        def submit_feedback():
            data = request.json
            prediction_id = data.get('prediction_id')
            user_id = data.get('user_id', 'default')
            feedback_type = data.get('feedback_type')  # 'accept', 'reject', 'modify'
            feedback_data = data.get('feedback_data', {})
            
            # Store feedback
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO user_feedback 
                    (prediction_id, user_id, feedback_type, feedback_data, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    prediction_id, user_id, feedback_type,
                    json.dumps(feedback_data), datetime.now().isoformat()
                ))
            
            # Learn from feedback
            if self.config.get('learning_enabled', True):
                self.behavior_system.track_user_behavior(
                    user_id, f"task_{feedback_type}d", feedback_data, {}
                )
            
            return jsonify({"status": "success"})
    
    def start_api_server(self):
        """Start the API server"""
        port = self.config.get('api_port', 8081)
        self.logger.info(f"Starting prediction API on port {port}")
        self.app.run(host='0.0.0.0', port=port, debug=False)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Task Prediction System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--generate", action="store_true", help="Generate predictions and exit")
    parser.add_argument("--user-id", default="default", help="User ID for predictions")
    
    args = parser.parse_args()
    
    system = IntelligentTaskPredictionSystem(args.config)
    
    if args.generate:
        predictions = system.generate_predictions(args.user_id)
        for pred in predictions:
            print(f"Prediction: {pred.predicted_title}")
            print(f"Confidence: {pred.confidence_score:.2%}")
            print(f"Reasoning: {pred.reasoning}")
            print("-" * 50)
    else:
        system.start_api_server()

if __name__ == "__main__":
    main()