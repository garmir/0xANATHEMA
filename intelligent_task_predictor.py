#!/usr/bin/env python3
"""
Intelligent Task Prediction and Auto-Generation System

This module provides AI-powered task prediction and auto-generation capabilities
for Task Master AI, analyzing project patterns and development trajectories to
automatically predict and generate future tasks.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import logging
import statistics
import pickle
import hashlib
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Simplified ML implementation without external dependencies
try:
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Simple fallback implementations
    class TfidfVectorizer:
        def __init__(self, **kwargs):
            self.vocabulary = {}
        
        def fit_transform(self, texts):
            return [[1.0] * len(texts)]
    
    class RandomForestClassifier:
        def __init__(self, **kwargs):
            self.classes_ = ['general', 'authentication', 'api', 'testing', 'deployment', 'ui']
        
        def fit(self, X, y):
            pass
        
        def predict(self, X):
            return ['general'] * len(X)
        
        def predict_proba(self, X):
            return [[1.0/6] * 6 for _ in X]
    
    class KMeans:
        def __init__(self, **kwargs):
            pass
        
        def fit_predict(self, X):
            return [0] * len(X)
    
    class StandardScaler:
        def __init__(self):
            pass
        
        def fit_transform(self, X):
            return X
        
        def transform(self, X):
            return X


@dataclass
class TaskPattern:
    """Task pattern identification result"""
    pattern_id: str
    pattern_type: str
    confidence: float
    frequency: int
    examples: List[str]
    description: str
    metadata: Dict[str, Any]


@dataclass
class PredictionResult:
    """Task prediction result"""
    prediction_id: str
    predicted_task: Dict[str, Any]
    confidence_score: float
    prediction_method: str
    historical_evidence: List[str]
    timestamp: str
    requires_approval: bool = True


@dataclass
class BehavioralInsight:
    """User behavioral insight"""
    insight_id: str
    behavior_type: str
    pattern_description: str
    frequency: int
    confidence: float
    recommendations: List[str]


class PatternAnalysisModule:
    """Analyzes completed tasks and project patterns to identify recurring workflows"""
    
    def __init__(self, tasks_file: str):
        """Initialize pattern analysis module"""
        self.tasks_file = tasks_file
        self.logger = self._setup_logging()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.pattern_database = defaultdict(list)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for pattern analysis"""
        logger = logging.getLogger("pattern_analysis")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_task_data(self) -> List[Dict[str, Any]]:
        """Load task data from tasks.json"""
        try:
            with open(self.tasks_file, 'r') as f:
                data = json.load(f)
                return data.get('master', {}).get('tasks', [])
        except Exception as e:
            self.logger.error(f"Error loading task data: {e}")
            return []
    
    def extract_text_features(self, tasks: List[Dict[str, Any]]) -> List[List[float]]:
        """Extract text features from task descriptions"""
        texts = []
        for task in tasks:
            text_parts = [
                task.get('title', ''),
                task.get('description', ''),
                task.get('details', '')
            ]
            texts.append(' '.join(filter(None, text_parts)))
        
        if texts:
            if SKLEARN_AVAILABLE:
                return self.vectorizer.fit_transform(texts).toarray()
            else:
                # Simple feature extraction - word count and common keywords
                features = []
                for text in texts:
                    words = text.lower().split()
                    feature_vector = [
                        len(words),  # Word count
                        1.0 if 'auth' in text.lower() else 0.0,
                        1.0 if 'api' in text.lower() else 0.0,
                        1.0 if 'test' in text.lower() else 0.0,
                        1.0 if 'deploy' in text.lower() else 0.0,
                        1.0 if 'ui' in text.lower() or 'dashboard' in text.lower() else 0.0
                    ]
                    features.append(feature_vector)
                return features
        return []
    
    def analyze_completion_patterns(self, tasks: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Analyze task completion patterns"""
        patterns = []
        
        # Analyze by status transitions
        status_transitions = defaultdict(int)
        for task in tasks:
            status = task.get('status', 'pending')
            priority = task.get('priority', 'medium')
            status_transitions[f"{priority}->{status}"] += 1
        
        # Identify common patterns
        for transition, count in status_transitions.items():
            if count >= 3:  # Pattern threshold
                patterns.append(TaskPattern(
                    pattern_id=hashlib.md5(transition.encode()).hexdigest()[:8],
                    pattern_type="status_transition",
                    confidence=min(count / 10.0, 1.0),
                    frequency=count,
                    examples=[transition],
                    description=f"Common transition pattern: {transition}",
                    metadata={"transition": transition, "count": count}
                ))
        
        return patterns
    
    def analyze_dependency_patterns(self, tasks: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Analyze task dependency patterns"""
        patterns = []
        dependency_graph = defaultdict(list)
        
        # Build dependency graph
        for task in tasks:
            task_id = task.get('id')
            dependencies = task.get('dependencies', [])
            for dep in dependencies:
                dependency_graph[dep].append(task_id)
        
        # Analyze common dependency structures
        for parent, children in dependency_graph.items():
            if len(children) >= 2:  # Multiple dependent tasks
                patterns.append(TaskPattern(
                    pattern_id=hashlib.md5(f"dep_{parent}".encode()).hexdigest()[:8],
                    pattern_type="dependency_hub",
                    confidence=len(children) / 10.0,
                    frequency=len(children),
                    examples=[f"Task {parent} -> {children}"],
                    description=f"Task {parent} is a dependency hub with {len(children)} dependent tasks",
                    metadata={"parent": parent, "children": children}
                ))
        
        return patterns
    
    def analyze_semantic_clusters(self, tasks: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Analyze semantic clustering of tasks"""
        patterns = []
        
        # Extract text features
        text_features = self.extract_text_features(tasks)
        if not text_features:
            return patterns
        
        # Perform clustering
        n_clusters = min(5, len(tasks) // 3) if len(tasks) > 3 else 1
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(text_features)
            
            # Analyze clusters
            cluster_groups = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                cluster_groups[cluster_id].append(tasks[i])
            
            for cluster_id, cluster_tasks in cluster_groups.items():
                if len(cluster_tasks) >= 2:
                    titles = [task.get('title', '') for task in cluster_tasks]
                    patterns.append(TaskPattern(
                        pattern_id=hashlib.md5(f"cluster_{cluster_id}".encode()).hexdigest()[:8],
                        pattern_type="semantic_cluster",
                        confidence=len(cluster_tasks) / len(tasks),
                        frequency=len(cluster_tasks),
                        examples=titles[:3],
                        description=f"Semantic cluster with {len(cluster_tasks)} related tasks",
                        metadata={"cluster_id": cluster_id, "task_count": len(cluster_tasks)}
                    ))
        
        return patterns
    
    def analyze_temporal_patterns(self, tasks: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Analyze temporal patterns in task creation and completion"""
        patterns = []
        
        # Group tasks by creation patterns (if timestamps available)
        time_patterns = defaultdict(list)
        
        # Analyze task complexity evolution
        complexity_trend = []
        for task in tasks:
            # Estimate complexity based on description length and subtask count
            desc_length = len(task.get('description', ''))
            subtask_count = len(task.get('subtasks', []))
            complexity = desc_length + (subtask_count * 50)
            complexity_trend.append(complexity)
        
        if len(complexity_trend) > 5:
            # Check for increasing complexity trend
            recent_avg = statistics.mean(complexity_trend[-3:])
            early_avg = statistics.mean(complexity_trend[:3])
            
            if recent_avg > early_avg * 1.2:
                patterns.append(TaskPattern(
                    pattern_id=hashlib.md5("complexity_increase".encode()).hexdigest()[:8],
                    pattern_type="complexity_trend",
                    confidence=0.8,
                    frequency=len(complexity_trend),
                    examples=["Increasing task complexity over time"],
                    description="Project tasks are becoming more complex",
                    metadata={"trend": "increasing", "early_avg": early_avg, "recent_avg": recent_avg}
                ))
        
        return patterns
    
    def generate_pattern_report(self) -> Dict[str, Any]:
        """Generate comprehensive pattern analysis report"""
        tasks = self.load_task_data()
        if not tasks:
            return {"error": "No task data available"}
        
        all_patterns = []
        
        # Run all pattern analyses
        all_patterns.extend(self.analyze_completion_patterns(tasks))
        all_patterns.extend(self.analyze_dependency_patterns(tasks))
        all_patterns.extend(self.analyze_semantic_clusters(tasks))
        all_patterns.extend(self.analyze_temporal_patterns(tasks))
        
        # Store patterns in database
        for pattern in all_patterns:
            self.pattern_database[pattern.pattern_type].append(pattern)
        
        # Generate summary
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tasks_analyzed": len(tasks),
            "patterns_identified": len(all_patterns),
            "pattern_types": {
                "status_transition": len([p for p in all_patterns if p.pattern_type == "status_transition"]),
                "dependency_hub": len([p for p in all_patterns if p.pattern_type == "dependency_hub"]),
                "semantic_cluster": len([p for p in all_patterns if p.pattern_type == "semantic_cluster"]),
                "complexity_trend": len([p for p in all_patterns if p.pattern_type == "complexity_trend"])
            },
            "high_confidence_patterns": [
                asdict(p) for p in all_patterns if p.confidence > 0.7
            ],
            "pattern_summary": [asdict(p) for p in all_patterns]
        }
        
        self.logger.info(f"Pattern analysis complete: {len(all_patterns)} patterns identified")
        return report


class TrajectoryPredictionEngine:
    """Machine learning-based trajectory prediction for project development"""
    
    def __init__(self, tasks_file: str):
        """Initialize trajectory prediction engine"""
        self.tasks_file = tasks_file
        self.logger = self._setup_logging()
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for trajectory prediction"""
        logger = logging.getLogger("trajectory_prediction")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def extract_trajectory_features(self, tasks: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[str]]:
        """Extract features for trajectory prediction"""
        features = []
        labels = []
        
        # Sort tasks by ID to maintain temporal order
        sorted_tasks = sorted(tasks, key=lambda x: x.get('id', 0))
        
        for i, task in enumerate(sorted_tasks):
            if i < 2:  # Need at least 2 previous tasks for context
                continue
                
            # Previous tasks context (last 2 tasks)
            prev_tasks = sorted_tasks[max(0, i-2):i]
            
            # Extract features from previous tasks
            feature_vector = []
            
            # Previous task complexity
            for prev_task in prev_tasks:
                desc_len = len(prev_task.get('description', ''))
                subtask_count = len(prev_task.get('subtasks', []))
                dep_count = len(prev_task.get('dependencies', []))
                feature_vector.extend([desc_len, subtask_count, dep_count])
            
            # Pad if necessary
            while len(feature_vector) < 6:  # 2 tasks * 3 features
                feature_vector.append(0)
            
            # Current project state
            total_tasks = i + 1
            completed_tasks = len([t for t in sorted_tasks[:i+1] if t.get('status') == 'done'])
            completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            feature_vector.extend([total_tasks, completed_tasks, completion_rate])
            
            features.append(feature_vector)
            
            # Label: next task type (simplified classification)
            current_task = task
            if 'auth' in current_task.get('title', '').lower():
                labels.append('authentication')
            elif 'api' in current_task.get('title', '').lower():
                labels.append('api')
            elif 'test' in current_task.get('title', '').lower():
                labels.append('testing')
            elif 'deploy' in current_task.get('title', '').lower():
                labels.append('deployment')
            elif 'ui' in current_task.get('title', '').lower() or 'dashboard' in current_task.get('title', '').lower():
                labels.append('ui')
            else:
                labels.append('general')
        
        return features, labels
    
    def train_prediction_model(self, tasks: List[Dict[str, Any]]) -> bool:
        """Train the trajectory prediction model"""
        try:
            features, labels = self.extract_trajectory_features(tasks)
            
            if len(features) < 5:  # Minimum training samples
                self.logger.warning("Insufficient data for training trajectory model")
                return False
            
            # Scale features
            if SKLEARN_AVAILABLE:
                features_scaled = self.scaler.fit_transform(features)
                
                # Split data
                if len(features) > 10:
                    X_train, X_test, y_train, y_test = train_test_split(
                        features_scaled, labels, test_size=0.2, random_state=42
                    )
                    
                    # Train classifier
                    self.classifier.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = self.classifier.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    self.logger.info(f"Model trained with accuracy: {accuracy:.3f}")
                else:
                    # Use all data for training
                    self.classifier.fit(features_scaled, labels)
                    self.logger.info("Model trained with full dataset")
            else:
                # Simple rule-based training for fallback
                self.classifier.fit(features, labels)
                self.logger.info("Model trained with simple rule-based approach")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error training prediction model: {e}")
            return False
    
    def predict_next_task_types(self, project_history: List[Dict[str, Any]], num_predictions: int = 3) -> List[str]:
        """Predict likely next task types based on project trajectory"""
        if not self.is_trained:
            if not self.train_prediction_model(project_history):
                return ["general", "testing", "deployment"]  # Fallback predictions
        
        try:
            # Extract features from recent project state
            if len(project_history) < 2:
                return ["general", "testing", "deployment"]
            
            # Use last 2 tasks as context
            recent_tasks = project_history[-2:]
            feature_vector = []
            
            for task in recent_tasks:
                desc_len = len(task.get('description', ''))
                subtask_count = len(task.get('subtasks', []))
                dep_count = len(task.get('dependencies', []))
                feature_vector.extend([desc_len, subtask_count, dep_count])
            
            # Current project state
            total_tasks = len(project_history)
            completed_tasks = len([t for t in project_history if t.get('status') == 'done'])
            completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            feature_vector.extend([total_tasks, completed_tasks, completion_rate])
            
            # Scale features and predict
            if SKLEARN_AVAILABLE:
                features_scaled = self.scaler.transform([feature_vector])
                
                # Get prediction probabilities
                probabilities = self.classifier.predict_proba(features_scaled)[0]
                classes = self.classifier.classes_
                
                # Sort by probability and return top predictions
                predictions = list(zip(classes, probabilities))
                predictions.sort(key=lambda x: x[1], reverse=True)
                
                return [pred[0] for pred in predictions[:num_predictions]]
            else:
                # Simple rule-based prediction
                return self.classifier.predict([feature_vector])[:num_predictions]
            
        except Exception as e:
            self.logger.error(f"Error predicting task types: {e}")
            return ["general", "testing", "deployment"]


class BehavioralLearningSystem:
    """Tracks user preferences and coding patterns for personalized predictions"""
    
    def __init__(self, tasks_file: str):
        """Initialize behavioral learning system"""
        self.tasks_file = tasks_file
        self.logger = self._setup_logging()
        self.behavioral_db_path = ".taskmaster/behavioral_data.db"
        self._init_database()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for behavioral learning"""
        logger = logging.getLogger("behavioral_learning")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_database(self):
        """Initialize behavioral data database"""
        os.makedirs(os.path.dirname(self.behavioral_db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.behavioral_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preference_type TEXT,
                preference_value TEXT,
                frequency INTEGER,
                confidence REAL,
                last_updated TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                interaction_type TEXT,
                interaction_data TEXT,
                timestamp TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS behavioral_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_type TEXT,
                description TEXT,
                confidence REAL,
                evidence_count INTEGER,
                created_date TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def track_task_interaction(self, task_id: str, interaction_type: str, data: Dict[str, Any]):
        """Track user interaction with tasks"""
        conn = sqlite3.connect(self.behavioral_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO task_interactions (task_id, interaction_type, interaction_data, timestamp)
            VALUES (?, ?, ?, ?)
        """, (task_id, interaction_type, json.dumps(data), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def analyze_completion_preferences(self, tasks: List[Dict[str, Any]]) -> List[BehavioralInsight]:
        """Analyze user preferences in task completion"""
        insights = []
        
        # Analyze priority preferences
        priority_counts = Counter()
        completed_tasks = [t for t in tasks if t.get('status') == 'done']
        
        for task in completed_tasks:
            priority = task.get('priority', 'medium')
            priority_counts[priority] += 1
        
        if priority_counts:
            preferred_priority = priority_counts.most_common(1)[0][0]
            confidence = priority_counts[preferred_priority] / len(completed_tasks)
            
            insights.append(BehavioralInsight(
                insight_id=hashlib.md5(f"priority_{preferred_priority}".encode()).hexdigest()[:8],
                behavior_type="priority_preference",
                pattern_description=f"User prefers completing {preferred_priority} priority tasks",
                frequency=priority_counts[preferred_priority],
                confidence=confidence,
                recommendations=[f"Generate more {preferred_priority} priority tasks"]
            ))
        
        # Analyze task complexity preferences
        complexity_scores = []
        for task in completed_tasks:
            desc_len = len(task.get('description', ''))
            subtask_count = len(task.get('subtasks', []))
            complexity = desc_len + (subtask_count * 50)
            complexity_scores.append(complexity)
        
        if complexity_scores:
            avg_complexity = statistics.mean(complexity_scores)
            
            if avg_complexity > 200:
                complexity_pref = "complex"
            elif avg_complexity > 100:
                complexity_pref = "moderate"
            else:
                complexity_pref = "simple"
            
            insights.append(BehavioralInsight(
                insight_id=hashlib.md5(f"complexity_{complexity_pref}".encode()).hexdigest()[:8],
                behavior_type="complexity_preference",
                pattern_description=f"User prefers {complexity_pref} tasks (avg complexity: {avg_complexity:.0f})",
                frequency=len(complexity_scores),
                confidence=0.8,
                recommendations=[f"Generate tasks with {complexity_pref} complexity levels"]
            ))
        
        return insights
    
    def learn_from_feedback(self, task_id: str, accepted: bool, feedback: Optional[str] = None):
        """Learn from user feedback on generated tasks"""
        interaction_data = {
            "accepted": accepted,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        self.track_task_interaction(task_id, "feedback", interaction_data)
        
        # Update behavioral insights based on feedback
        self._update_insights_from_feedback(task_id, accepted, feedback)
    
    def _update_insights_from_feedback(self, task_id: str, accepted: bool, feedback: Optional[str]):
        """Update behavioral insights based on user feedback"""
        conn = sqlite3.connect(self.behavioral_db_path)
        cursor = conn.cursor()
        
        # Simple feedback learning - track acceptance rates
        insight_type = "task_acceptance"
        description = f"Task acceptance rate tracking"
        
        # Check if insight exists
        cursor.execute("""
            SELECT id, evidence_count, confidence FROM behavioral_insights 
            WHERE insight_type = ?
        """, (insight_type,))
        
        result = cursor.fetchone()
        if result:
            insight_id, evidence_count, current_confidence = result
            new_evidence_count = evidence_count + 1
            
            # Update confidence based on acceptance
            if accepted:
                new_confidence = (current_confidence * evidence_count + 1.0) / new_evidence_count
            else:
                new_confidence = (current_confidence * evidence_count + 0.0) / new_evidence_count
            
            cursor.execute("""
                UPDATE behavioral_insights 
                SET confidence = ?, evidence_count = ?
                WHERE id = ?
            """, (new_confidence, new_evidence_count, insight_id))
        else:
            # Create new insight
            initial_confidence = 1.0 if accepted else 0.0
            cursor.execute("""
                INSERT INTO behavioral_insights 
                (insight_type, description, confidence, evidence_count, created_date)
                VALUES (?, ?, ?, ?, ?)
            """, (insight_type, description, initial_confidence, 1, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_behavioral_insights(self) -> List[BehavioralInsight]:
        """Get current behavioral insights"""
        conn = sqlite3.connect(self.behavioral_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT insight_type, description, confidence, evidence_count 
            FROM behavioral_insights
            ORDER BY confidence DESC
        """)
        
        insights = []
        for row in cursor.fetchall():
            insight_type, description, confidence, evidence_count = row
            insights.append(BehavioralInsight(
                insight_id=hashlib.md5(f"{insight_type}_{description}".encode()).hexdigest()[:8],
                behavior_type=insight_type,
                pattern_description=description,
                frequency=evidence_count,
                confidence=confidence,
                recommendations=[]
            ))
        
        conn.close()
        return insights


class AutoGenerationFramework:
    """Creates tasks automatically based on predictions with confidence scores"""
    
    def __init__(self, tasks_file: str):
        """Initialize auto-generation framework"""
        self.tasks_file = tasks_file
        self.logger = self._setup_logging()
        self.pattern_analyzer = PatternAnalysisModule(tasks_file)
        self.trajectory_engine = TrajectoryPredictionEngine(tasks_file)
        self.behavioral_system = BehavioralLearningSystem(tasks_file)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for auto-generation"""
        logger = logging.getLogger("auto_generation")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_task_suggestions(self, num_suggestions: int = 3) -> List[PredictionResult]:
        """Generate task suggestions based on analysis"""
        suggestions = []
        
        # Load current tasks
        tasks = self.pattern_analyzer.load_task_data()
        if not tasks:
            return suggestions
        
        # Get pattern analysis
        pattern_report = self.pattern_analyzer.generate_pattern_report()
        
        # Get trajectory predictions
        predicted_types = self.trajectory_engine.predict_next_task_types(tasks, num_suggestions)
        
        # Get behavioral insights
        behavioral_insights = self.behavioral_system.get_behavioral_insights()
        
        # Generate suggestions based on predictions
        for i, task_type in enumerate(predicted_types):
            suggestion = self._create_task_suggestion(
                task_type, 
                tasks, 
                pattern_report, 
                behavioral_insights,
                confidence_base=0.9 - (i * 0.1)
            )
            suggestions.append(suggestion)
        
        # Add pattern-based suggestions
        high_confidence_patterns = pattern_report.get('high_confidence_patterns', [])
        for pattern in high_confidence_patterns[:2]:  # Top 2 patterns
            pattern_suggestion = self._create_pattern_based_suggestion(pattern, tasks)
            suggestions.append(pattern_suggestion)
        
        return suggestions[:num_suggestions]
    
    def _create_task_suggestion(self, task_type: str, existing_tasks: List[Dict[str, Any]], 
                               pattern_report: Dict[str, Any], behavioral_insights: List[BehavioralInsight],
                               confidence_base: float) -> PredictionResult:
        """Create a task suggestion based on predicted type"""
        
        # Get next task ID
        max_id = max([task.get('id', 0) for task in existing_tasks]) if existing_tasks else 0
        next_id = max_id + 1
        
        # Task templates based on type
        task_templates = {
            'authentication': {
                'title': 'Implement Advanced User Authentication System',
                'description': 'Create comprehensive user authentication with multi-factor support',
                'details': 'Implement OAuth2 integration, JWT token management, and secure session handling with support for social login providers.',
                'priority': 'high'
            },
            'api': {
                'title': 'Develop RESTful API Endpoints',
                'description': 'Create robust API endpoints with proper validation and error handling',
                'details': 'Design and implement REST API with OpenAPI documentation, input validation, rate limiting, and comprehensive error responses.',
                'priority': 'high'
            },
            'testing': {
                'title': 'Implement Comprehensive Test Suite',
                'description': 'Create unit, integration, and end-to-end tests for system validation',
                'details': 'Develop test suite covering all critical functionality with automated testing pipeline and coverage reporting.',
                'priority': 'medium'
            },
            'deployment': {
                'title': 'Setup Production Deployment Pipeline',
                'description': 'Create automated deployment pipeline for production environments',
                'details': 'Implement CI/CD pipeline with automated testing, security scanning, and deployment to production infrastructure.',
                'priority': 'high'
            },
            'ui': {
                'title': 'Design Interactive User Interface',
                'description': 'Create responsive and intuitive user interface components',
                'details': 'Develop modern UI components with accessibility support, responsive design, and optimal user experience.',
                'priority': 'medium'
            },
            'general': {
                'title': 'Implement Core System Feature',
                'description': 'Develop essential system functionality based on project requirements',
                'details': 'Analyze requirements and implement core feature with proper documentation and testing support.',
                'priority': 'medium'
            }
        }
        
        template = task_templates.get(task_type, task_templates['general'])
        
        # Adjust based on behavioral insights
        preferred_priority = 'medium'  # Default
        for insight in behavioral_insights:
            if insight.behavior_type == 'priority_preference' and insight.confidence > 0.6:
                if 'high' in insight.pattern_description:
                    preferred_priority = 'high'
                elif 'low' in insight.pattern_description:
                    preferred_priority = 'low'
        
        # Create suggested task
        suggested_task = {
            'id': next_id,
            'title': template['title'],
            'description': template['description'],
            'details': template['details'],
            'priority': preferred_priority,
            'status': 'pending',
            'dependencies': [],
            'subtasks': []
        }
        
        # Create prediction result
        prediction = PredictionResult(
            prediction_id=hashlib.md5(f"{task_type}_{next_id}_{time.time()}".encode()).hexdigest()[:8],
            predicted_task=suggested_task,
            confidence_score=confidence_base,
            prediction_method="trajectory_analysis",
            historical_evidence=[f"Predicted based on {task_type} pattern analysis"],
            timestamp=datetime.now().isoformat(),
            requires_approval=True
        )
        
        return prediction
    
    def _create_pattern_based_suggestion(self, pattern: Dict[str, Any], existing_tasks: List[Dict[str, Any]]) -> PredictionResult:
        """Create task suggestion based on identified pattern"""
        
        max_id = max([task.get('id', 0) for task in existing_tasks]) if existing_tasks else 0
        next_id = max_id + 1
        
        pattern_type = pattern.get('pattern_type', 'general')
        
        if pattern_type == 'complexity_trend':
            suggested_task = {
                'id': next_id,
                'title': 'Optimize System Performance and Scalability',
                'description': 'Address increasing complexity with performance optimizations',
                'details': 'Analyze system bottlenecks and implement performance improvements to handle growing complexity while maintaining responsiveness.',
                'priority': 'high',
                'status': 'pending',
                'dependencies': [],
                'subtasks': []
            }
        elif pattern_type == 'dependency_hub':
            suggested_task = {
                'id': next_id,
                'title': 'Refactor Dependency Architecture',
                'description': 'Optimize dependency structure for better maintainability',
                'details': 'Review and refactor dependency relationships to reduce coupling and improve system modularity.',
                'priority': 'medium',
                'status': 'pending',
                'dependencies': [],
                'subtasks': []
            }
        else:
            suggested_task = {
                'id': next_id,
                'title': 'Continue Pattern-Based Development',
                'description': 'Extend existing patterns for consistent development',
                'details': 'Build upon identified patterns to maintain consistency and leverage established workflows.',
                'priority': 'medium',
                'status': 'pending',
                'dependencies': [],
                'subtasks': []
            }
        
        prediction = PredictionResult(
            prediction_id=hashlib.md5(f"pattern_{pattern_type}_{next_id}".encode()).hexdigest()[:8],
            predicted_task=suggested_task,
            confidence_score=pattern.get('confidence', 0.7),
            prediction_method="pattern_analysis",
            historical_evidence=[pattern.get('description', 'Pattern-based prediction')],
            timestamp=datetime.now().isoformat(),
            requires_approval=True
        )
        
        return prediction


class TaskPredictionEngine:
    """Main engine that coordinates all prediction components"""
    
    def __init__(self, tasks_file: str = ".taskmaster/tasks/tasks.json"):
        """Initialize the main prediction engine"""
        self.tasks_file = tasks_file
        self.logger = self._setup_logging()
        
        # Initialize components
        self.pattern_analyzer = PatternAnalysisModule(tasks_file)
        self.trajectory_engine = TrajectoryPredictionEngine(tasks_file)
        self.behavioral_system = BehavioralLearningSystem(tasks_file)
        self.auto_generator = AutoGenerationFramework(tasks_file)
        
        # API integration
        self.api_endpoints = {
            'analyze_patterns': self.analyze_patterns,
            'predict_trajectory': self.predict_trajectory,
            'generate_tasks': self.generate_tasks,
            'learn_from_feedback': self.learn_from_feedback,
            'get_insights': self.get_insights
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for main engine"""
        logger = logging.getLogger("task_prediction_engine")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_patterns(self, historical_tasks: List[Dict] = None) -> Dict:
        """Analyze patterns in historical tasks"""
        try:
            return self.pattern_analyzer.generate_pattern_report()
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return {"error": str(e)}
    
    def predict_trajectory(self, project_history: List[Dict] = None) -> List[str]:
        """Predict project trajectory"""
        try:
            if project_history is None:
                project_history = self.pattern_analyzer.load_task_data()
            
            return self.trajectory_engine.predict_next_task_types(project_history)
        except Exception as e:
            self.logger.error(f"Trajectory prediction failed: {e}")
            return ["general", "testing", "deployment"]
    
    def generate_tasks(self, predictions: Dict = None) -> List[Dict]:
        """Generate new tasks based on predictions"""
        try:
            suggestions = self.auto_generator.generate_task_suggestions()
            return [asdict(suggestion) for suggestion in suggestions]
        except Exception as e:
            self.logger.error(f"Task generation failed: {e}")
            return []
    
    def learn_from_feedback(self, task_id: str, accepted: bool, feedback: str = None) -> bool:
        """Learn from user feedback"""
        try:
            self.behavioral_system.learn_from_feedback(task_id, accepted, feedback)
            return True
        except Exception as e:
            self.logger.error(f"Feedback learning failed: {e}")
            return False
    
    def get_insights(self) -> List[Dict]:
        """Get behavioral insights"""
        try:
            insights = self.behavioral_system.get_behavioral_insights()
            return [asdict(insight) for insight in insights]
        except Exception as e:
            self.logger.error(f"Getting insights failed: {e}")
            return []
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete analysis and prediction pipeline"""
        self.logger.info("Starting full task prediction analysis...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "pattern_analysis": self.analyze_patterns(),
            "trajectory_predictions": self.predict_trajectory(),
            "task_suggestions": self.generate_tasks(),
            "behavioral_insights": self.get_insights(),
            "status": "completed"
        }
        
        # Save results
        output_file = ".taskmaster/reports/prediction_analysis.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Full analysis completed. Results saved to {output_file}")
        return results


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Task Prediction System")
    parser.add_argument("--tasks-file", default=".taskmaster/tasks/tasks.json", help="Path to tasks.json file")
    parser.add_argument("--analyze", action="store_true", help="Run pattern analysis")
    parser.add_argument("--predict", action="store_true", help="Generate trajectory predictions")
    parser.add_argument("--generate", action="store_true", help="Generate task suggestions")
    parser.add_argument("--full", action="store_true", help="Run full analysis pipeline")
    parser.add_argument("--feedback", nargs=3, metavar=("TASK_ID", "ACCEPTED", "FEEDBACK"), help="Provide feedback (task_id accepted feedback)")
    
    args = parser.parse_args()
    
    engine = TaskPredictionEngine(args.tasks_file)
    
    if args.feedback:
        task_id, accepted_str, feedback = args.feedback
        accepted = accepted_str.lower() in ['true', '1', 'yes']
        success = engine.learn_from_feedback(task_id, accepted, feedback)
        print(f"Feedback recorded: {success}")
        
    elif args.analyze:
        results = engine.analyze_patterns()
        print(json.dumps(results, indent=2))
        
    elif args.predict:
        predictions = engine.predict_trajectory()
        print(f"Predicted task types: {predictions}")
        
    elif args.generate:
        suggestions = engine.generate_tasks()
        print(json.dumps(suggestions, indent=2))
        
    elif args.full:
        results = engine.run_full_analysis()
        print("Full analysis completed!")
        print(f"Pattern analysis: {len(results['pattern_analysis'].get('pattern_summary', []))} patterns identified")
        print(f"Task suggestions: {len(results['task_suggestions'])} generated")
        print(f"Behavioral insights: {len(results['behavioral_insights'])} insights")
        
    else:
        # Default: show current insights and suggestions
        print("Intelligent Task Prediction System")
        print("=" * 40)
        
        insights = engine.get_insights()
        if insights:
            print(f"\nBehavioral Insights ({len(insights)}):")
            for insight in insights[:3]:
                print(f"  • {insight['pattern_description']} (confidence: {insight['confidence']:.2f})")
        
        suggestions = engine.generate_tasks()
        if suggestions:
            print(f"\nTask Suggestions ({len(suggestions)}):")
            for suggestion in suggestions[:3]:
                task = suggestion['predicted_task']
                print(f"  • {task['title']} (confidence: {suggestion['confidence_score']:.2f})")


if __name__ == "__main__":
    main()