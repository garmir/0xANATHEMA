#!/usr/bin/env python3
"""
Recursive Todo Enhancement Engine for Task Master AI

A comprehensive system that autonomously analyzes, optimizes, and enhances todo lists
and task structures through recursive analysis and improvement cycles.

Core Features:
- Todo Analysis Engine: Parse and analyze existing todo structures
- Recursive Enhancement Framework: Apply improvement cycles with configurable depth
- Intelligent Task Decomposition: Break down complex todos into manageable subtasks
- Todo Quality Assessment: Score and improve todo quality across multiple dimensions
- Enhancement Automation: Auto-apply improvements and generate missing components
- Task Master Integration: Seamless integration with existing Task Master infrastructure
- Local LLM Integration: Intelligent analysis using local LLM abstraction layer
- Meta-Learning: Learn from previous enhancement outcomes to improve strategies
"""

import json
import os
import re
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import statistics
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TodoStatus(Enum):
    """Task status enumeration matching Task Master conventions"""
    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    DONE = "done"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"

class Priority(Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EnhancementType(Enum):
    """Types of enhancements that can be applied"""
    DECOMPOSITION = "decomposition"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    QUALITY_IMPROVEMENT = "quality_improvement"
    DESCRIPTION_ENHANCEMENT = "description_enhancement"
    TIME_ESTIMATION = "time_estimation"
    RESOURCE_PLANNING = "resource_planning"
    TEST_STRATEGY = "test_strategy"
    VALIDATION_CRITERIA = "validation_criteria"

@dataclass
class QualityMetrics:
    """Quality metrics for todo assessment"""
    clarity_score: float = 0.0
    completeness_score: float = 0.0
    actionability_score: float = 0.0
    specificity_score: float = 0.0
    testability_score: float = 0.0
    feasibility_score: float = 0.0
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'clarity_score': self.clarity_score,
            'completeness_score': self.completeness_score,
            'actionability_score': self.actionability_score,
            'specificity_score': self.specificity_score,
            'testability_score': self.testability_score,
            'feasibility_score': self.feasibility_score,
            'overall_score': self.overall_score
        }

@dataclass
class EnhancementResult:
    """Result of an enhancement operation"""
    enhancement_type: EnhancementType
    original_task: Dict[str, Any]
    enhanced_task: Dict[str, Any]
    quality_improvement: float
    suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Todo:
    """Enhanced todo data structure"""
    id: str
    title: str
    description: str = ""
    status: TodoStatus = TodoStatus.PENDING
    priority: Priority = Priority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    subtasks: List['Todo'] = field(default_factory=list)
    details: str = ""
    test_strategy: str = ""
    time_estimate: Optional[int] = None  # in minutes
    resource_requirements: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    enhancement_history: List[EnhancementResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert todo to Task Master compatible dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'status': self.status.value,
            'priority': self.priority.value,
            'dependencies': self.dependencies,
            'subtasks': [subtask.to_dict() for subtask in self.subtasks],
            'details': self.details,
            'testStrategy': self.test_strategy,
            'timeEstimate': self.time_estimate,
            'resourceRequirements': self.resource_requirements,
            'validationCriteria': self.validation_criteria,
            'createdAt': self.created_at.isoformat(),
            'updatedAt': self.updated_at.isoformat(),
            'qualityMetrics': self.quality_metrics.to_dict(),
            'enhancementHistory': [
                {
                    'type': result.enhancement_type.value,
                    'qualityImprovement': result.quality_improvement,
                    'suggestions': result.suggestions,
                    'timestamp': result.timestamp.isoformat(),
                    'metadata': result.metadata
                } for result in self.enhancement_history
            ],
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Todo':
        """Create Todo from Task Master compatible dictionary"""
        todo = cls(
            id=data.get('id', str(uuid.uuid4())),
            title=data.get('title', ''),
            description=data.get('description', ''),
            status=TodoStatus(data.get('status', 'pending')),
            priority=Priority(data.get('priority', 'medium')),
            dependencies=data.get('dependencies', []),
            details=data.get('details', ''),
            test_strategy=data.get('testStrategy', ''),
            time_estimate=data.get('timeEstimate'),
            resource_requirements=data.get('resourceRequirements', []),
            validation_criteria=data.get('validationCriteria', []),
            metadata=data.get('metadata', {})
        )
        
        # Parse dates
        if 'createdAt' in data:
            todo.created_at = datetime.fromisoformat(data['createdAt'])
        if 'updatedAt' in data:
            todo.updated_at = datetime.fromisoformat(data['updatedAt'])
        
        # Parse quality metrics
        if 'qualityMetrics' in data:
            metrics_data = data['qualityMetrics']
            todo.quality_metrics = QualityMetrics(**metrics_data)
        
        # Parse subtasks
        if 'subtasks' in data:
            todo.subtasks = [cls.from_dict(subtask) for subtask in data['subtasks']]
        
        return todo

class LocalLLMAdapter:
    """Adapter for local LLM integration"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the local LLM model"""
        try:
            # Try to import and use local LLM libraries
            # This is a placeholder - actual implementation would depend on the specific LLM library
            logger.info("Initializing local LLM model...")
            # For now, we'll use a mock implementation
            self.model = "mock_llm_model"
        except Exception as e:
            logger.warning(f"Failed to initialize local LLM: {e}")
            self.model = None
    
    def analyze_text(self, text: str, context: str = "") -> Dict[str, Any]:
        """Analyze text using local LLM"""
        if not self.model:
            return self._fallback_analysis(text, context)
        
        try:
            # Mock LLM analysis - replace with actual LLM call
            analysis = {
                'clarity': self._assess_clarity(text),
                'completeness': self._assess_completeness(text),
                'actionability': self._assess_actionability(text),
                'specificity': self._assess_specificity(text),
                'suggestions': self._generate_suggestions(text, context)
            }
            return analysis
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(text, context)
    
    def _fallback_analysis(self, text: str, context: str = "") -> Dict[str, Any]:
        """Fallback analysis using rule-based methods"""
        return {
            'clarity': self._assess_clarity(text),
            'completeness': self._assess_completeness(text),
            'actionability': self._assess_actionability(text),
            'specificity': self._assess_specificity(text),
            'suggestions': self._generate_basic_suggestions(text)
        }
    
    def _assess_clarity(self, text: str) -> float:
        """Assess text clarity using heuristics"""
        if not text:
            return 0.0
        
        score = 0.5  # Base score
        
        # Check for clear action words
        action_words = ['implement', 'create', 'build', 'design', 'develop', 'test', 'deploy', 'fix', 'update', 'add', 'remove']
        if any(word in text.lower() for word in action_words):
            score += 0.2
        
        # Check for specific technical terms
        if len(re.findall(r'\b[A-Z][a-zA-Z]+\b', text)) > 0:
            score += 0.1
        
        # Penalize overly long sentences
        sentences = text.split('.')
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if avg_length > 20:
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def _assess_completeness(self, text: str) -> float:
        """Assess text completeness using heuristics"""
        if not text:
            return 0.0
        
        score = 0.3  # Base score
        
        # Check for key elements
        if len(text.split()) > 10:
            score += 0.2
        
        # Check for technical details
        if any(keyword in text.lower() for keyword in ['api', 'database', 'function', 'class', 'method', 'endpoint']):
            score += 0.2
        
        # Check for acceptance criteria indicators
        if any(keyword in text.lower() for keyword in ['should', 'must', 'will', 'criteria', 'requirement']):
            score += 0.3
        
        return min(1.0, max(0.0, score))
    
    def _assess_actionability(self, text: str) -> float:
        """Assess text actionability using heuristics"""
        if not text:
            return 0.0
        
        score = 0.2  # Base score
        
        # Check for action verbs
        action_verbs = ['create', 'build', 'implement', 'develop', 'design', 'test', 'deploy', 'configure', 'setup', 'install']
        if any(verb in text.lower() for verb in action_verbs):
            score += 0.4
        
        # Check for specific targets
        if any(keyword in text.lower() for keyword in ['file', 'function', 'class', 'component', 'module', 'system']):
            score += 0.2
        
        # Check for measurable outcomes
        if any(keyword in text.lower() for keyword in ['complete', 'working', 'functional', 'tested', 'deployed']):
            score += 0.2
        
        return min(1.0, max(0.0, score))
    
    def _assess_specificity(self, text: str) -> float:
        """Assess text specificity using heuristics"""
        if not text:
            return 0.0
        
        score = 0.2  # Base score
        
        # Check for specific names/identifiers
        if len(re.findall(r'\b[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b', text)) > 0:
            score += 0.3
        
        # Check for file paths or technical specifications
        if any(pattern in text for pattern in ['/', '.', ':', '-', '_']):
            score += 0.2
        
        # Check for numbers or quantities
        if re.search(r'\d+', text):
            score += 0.2
        
        # Check for technical terms
        tech_terms = ['api', 'database', 'server', 'client', 'frontend', 'backend', 'framework', 'library']
        if any(term in text.lower() for term in tech_terms):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _generate_suggestions(self, text: str, context: str = "") -> List[str]:
        """Generate enhancement suggestions using LLM"""
        # Mock LLM suggestions - replace with actual LLM call
        suggestions = self._generate_basic_suggestions(text)
        
        # Add context-aware suggestions if context is provided
        if context:
            if 'test' in context.lower():
                suggestions.append("Consider adding specific test cases and validation criteria")
            if 'api' in context.lower():
                suggestions.append("Include API endpoint specifications and request/response formats")
            if 'database' in context.lower():
                suggestions.append("Specify database schema changes and migration requirements")
        
        return suggestions
    
    def _generate_basic_suggestions(self, text: str) -> List[str]:
        """Generate basic enhancement suggestions using rule-based methods"""
        suggestions = []
        
        if not text:
            return ["Add a clear task description"]
        
        # Check for missing elements
        if len(text.split()) < 5:
            suggestions.append("Expand task description with more specific details")
        
        if not any(verb in text.lower() for verb in ['create', 'build', 'implement', 'develop', 'test']):
            suggestions.append("Start with a clear action verb (create, build, implement, etc.)")
        
        if not re.search(r'\b(should|must|will|ensure|verify)\b', text.lower()):
            suggestions.append("Add acceptance criteria using 'should', 'must', or 'will'")
        
        if 'test' not in text.lower():
            suggestions.append("Consider adding testing requirements")
        
        if not re.search(r'\d+', text):
            suggestions.append("Add specific quantities, timeframes, or measurable outcomes")
        
        return suggestions

class TodoAnalyzer:
    """Analyzes todo structures for optimization opportunities"""
    
    def __init__(self, llm_adapter: LocalLLMAdapter):
        self.llm_adapter = llm_adapter
    
    def analyze_todo(self, todo: Todo) -> QualityMetrics:
        """Analyze a single todo for quality metrics"""
        text = f"{todo.title} {todo.description} {todo.details}".strip()
        
        if not text:
            return QualityMetrics()
        
        # Use LLM for analysis
        analysis = self.llm_adapter.analyze_text(text)
        
        # Calculate quality metrics
        metrics = QualityMetrics(
            clarity_score=analysis.get('clarity', 0.0),
            completeness_score=analysis.get('completeness', 0.0),
            actionability_score=analysis.get('actionability', 0.0),
            specificity_score=analysis.get('specificity', 0.0),
            testability_score=self._assess_testability(todo),
            feasibility_score=self._assess_feasibility(todo)
        )
        
        # Calculate overall score
        metrics.overall_score = (
            metrics.clarity_score * 0.2 +
            metrics.completeness_score * 0.2 +
            metrics.actionability_score * 0.2 +
            metrics.specificity_score * 0.15 +
            metrics.testability_score * 0.15 +
            metrics.feasibility_score * 0.1
        )
        
        return metrics
    
    def _assess_testability(self, todo: Todo) -> float:
        """Assess how testable a todo is"""
        score = 0.0
        
        # Check for existing test strategy
        if todo.test_strategy:
            score += 0.4
        
        # Check for validation criteria
        if todo.validation_criteria:
            score += 0.3
        
        # Check for measurable outcomes in description
        text = f"{todo.title} {todo.description} {todo.details}".lower()
        if any(keyword in text for keyword in ['test', 'verify', 'validate', 'check', 'ensure']):
            score += 0.2
        
        # Check for specific success criteria
        if any(keyword in text for keyword in ['complete', 'working', 'functional', 'successful']):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_feasibility(self, todo: Todo) -> float:
        """Assess the feasibility of a todo"""
        score = 0.7  # Base feasibility score
        
        # Check for resource requirements
        if todo.resource_requirements:
            score += 0.1
        
        # Check for time estimates
        if todo.time_estimate:
            score += 0.1
        
        # Check for excessive complexity indicators
        text = f"{todo.title} {todo.description} {todo.details}".lower()
        complexity_indicators = ['complex', 'complicated', 'difficult', 'challenging', 'major refactor']
        if any(indicator in text for indicator in complexity_indicators):
            score -= 0.2
        
        # Check for dependencies
        if len(todo.dependencies) > 5:
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def find_optimization_opportunities(self, todos: List[Todo]) -> List[Dict[str, Any]]:
        """Find optimization opportunities in a list of todos"""
        opportunities = []
        
        # Check for redundant tasks
        opportunities.extend(self._find_redundant_tasks(todos))
        
        # Check for missing dependencies
        opportunities.extend(self._find_missing_dependencies(todos))
        
        # Check for tasks that should be broken down
        opportunities.extend(self._find_decomposition_candidates(todos))
        
        # Check for tasks with poor quality
        opportunities.extend(self._find_low_quality_tasks(todos))
        
        return opportunities
    
    def _find_redundant_tasks(self, todos: List[Todo]) -> List[Dict[str, Any]]:
        """Find potentially redundant tasks"""
        opportunities = []
        
        # Simple similarity check based on title keywords
        for i, todo1 in enumerate(todos):
            for j, todo2 in enumerate(todos[i+1:], i+1):
                similarity = self._calculate_similarity(todo1.title, todo2.title)
                if similarity > 0.7:
                    opportunities.append({
                        'type': 'redundant_tasks',
                        'task_ids': [todo1.id, todo2.id],
                        'similarity': similarity,
                        'suggestion': f"Tasks '{todo1.title}' and '{todo2.title}' appear similar. Consider merging or clarifying differences."
                    })
        
        return opportunities
    
    def _find_missing_dependencies(self, todos: List[Todo]) -> List[Dict[str, Any]]:
        """Find potentially missing dependencies"""
        opportunities = []
        
        # Check for logical dependencies based on keywords
        dependency_keywords = {
            'setup': ['config', 'install', 'initialize'],
            'test': ['implement', 'create', 'build'],
            'deploy': ['test', 'build', 'package'],
            'config': ['setup', 'install']
        }
        
        for todo in todos:
            title_lower = todo.title.lower()
            for keyword, prerequisites in dependency_keywords.items():
                if keyword in title_lower:
                    for prereq in prerequisites:
                        prereq_tasks = [t for t in todos if prereq in t.title.lower() and t.id != todo.id]
                        for prereq_task in prereq_tasks:
                            if prereq_task.id not in todo.dependencies:
                                opportunities.append({
                                    'type': 'missing_dependency',
                                    'task_id': todo.id,
                                    'suggested_dependency': prereq_task.id,
                                    'suggestion': f"Task '{todo.title}' may depend on '{prereq_task.title}'"
                                })
        
        return opportunities
    
    def _find_decomposition_candidates(self, todos: List[Todo]) -> List[Dict[str, Any]]:
        """Find tasks that should be broken down into subtasks"""
        opportunities = []
        
        for todo in todos:
            # Check if task is complex enough to warrant decomposition
            complexity_score = self._calculate_complexity_score(todo)
            
            if complexity_score > 0.7 and len(todo.subtasks) == 0:
                opportunities.append({
                    'type': 'decomposition_candidate',
                    'task_id': todo.id,
                    'complexity_score': complexity_score,
                    'suggestion': f"Task '{todo.title}' appears complex and could benefit from decomposition into subtasks"
                })
        
        return opportunities
    
    def _find_low_quality_tasks(self, todos: List[Todo]) -> List[Dict[str, Any]]:
        """Find tasks with poor quality that need improvement"""
        opportunities = []
        
        for todo in todos:
            if todo.quality_metrics.overall_score < 0.5:
                opportunities.append({
                    'type': 'low_quality_task',
                    'task_id': todo.id,
                    'quality_score': todo.quality_metrics.overall_score,
                    'suggestion': f"Task '{todo.title}' has low quality score ({todo.quality_metrics.overall_score:.2f}) and needs improvement"
                })
        
        return opportunities
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_complexity_score(self, todo: Todo) -> float:
        """Calculate complexity score for a todo"""
        score = 0.0
        
        # Check description length
        total_text = f"{todo.title} {todo.description} {todo.details}"
        if len(total_text.split()) > 50:
            score += 0.3
        
        # Check for complexity indicators
        complexity_words = ['complex', 'multiple', 'various', 'several', 'integrate', 'coordinate', 'manage']
        if any(word in total_text.lower() for word in complexity_words):
            score += 0.2
        
        # Check for multiple action verbs
        action_verbs = ['create', 'build', 'implement', 'develop', 'test', 'deploy', 'configure', 'setup']
        verb_count = sum(1 for verb in action_verbs if verb in total_text.lower())
        if verb_count > 2:
            score += 0.3
        
        # Check for multiple technologies/components
        tech_terms = ['api', 'database', 'frontend', 'backend', 'server', 'client', 'framework']
        tech_count = sum(1 for term in tech_terms if term in total_text.lower())
        if tech_count > 2:
            score += 0.2
        
        return min(1.0, score)

class DependencyAnalyzer:
    """Analyzes and resolves task dependencies"""
    
    def __init__(self):
        self.dependency_graph = {}
    
    def build_dependency_graph(self, todos: List[Todo]) -> Dict[str, List[str]]:
        """Build a dependency graph from todos"""
        self.dependency_graph = {}
        
        for todo in todos:
            self.dependency_graph[todo.id] = todo.dependencies.copy()
            
            # Add subtask dependencies
            for subtask in todo.subtasks:
                self.dependency_graph[subtask.id] = subtask.dependencies.copy()
                # Subtasks implicitly depend on their parent being started
                if todo.id not in self.dependency_graph[subtask.id]:
                    self.dependency_graph[subtask.id].append(todo.id)
        
        return self.dependency_graph
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the dependency graph"""
        def dfs(node, visited, rec_stack, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor, visited, rec_stack, path)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            path.pop()
            return None
        
        visited = set()
        cycles = []
        
        for node in self.dependency_graph:
            if node not in visited:
                cycle = dfs(node, visited, set(), [])
                if cycle:
                    cycles.append(cycle)
        
        return cycles
    
    def suggest_dependency_resolution(self, cycles: List[List[str]]) -> List[Dict[str, Any]]:
        """Suggest ways to resolve circular dependencies"""
        suggestions = []
        
        for cycle in cycles:
            # Find the weakest link in the cycle
            weakest_edge = self._find_weakest_edge(cycle)
            
            suggestions.append({
                'type': 'circular_dependency_resolution',
                'cycle': cycle,
                'suggested_removal': weakest_edge,
                'suggestion': f"Remove dependency from {weakest_edge[0]} to {weakest_edge[1]} to break the cycle"
            })
        
        return suggestions
    
    def _find_weakest_edge(self, cycle: List[str]) -> Tuple[str, str]:
        """Find the weakest edge in a dependency cycle"""
        # For now, just return the last edge in the cycle
        # In a more sophisticated implementation, this could analyze
        # the strength of relationships between tasks
        return (cycle[-1], cycle[0])
    
    def optimize_task_order(self, todos: List[Todo]) -> List[str]:
        """Optimize the order of tasks based on dependencies"""
        # Topological sort
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        # Build graph and calculate in-degrees
        for todo in todos:
            for dep in todo.dependencies:
                graph[dep].append(todo.id)
                in_degree[todo.id] += 1
        
        # Initialize queue with tasks that have no dependencies
        queue = deque([todo.id for todo in todos if in_degree[todo.id] == 0])
        ordered_tasks = []
        
        while queue:
            current = queue.popleft()
            ordered_tasks.append(current)
            
            # Reduce in-degree for dependent tasks
            for dependent in graph[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return ordered_tasks
    
    def identify_parallel_opportunities(self, todos: List[Todo]) -> List[List[str]]:
        """Identify tasks that can be worked on in parallel"""
        parallel_groups = []
        
        # Group tasks by their dependency depth
        depth_groups = defaultdict(list)
        
        def calculate_depth(task_id, visited=None):
            if visited is None:
                visited = set()
            
            if task_id in visited:
                return 0  # Circular dependency, treat as depth 0
            
            visited.add(task_id)
            
            dependencies = self.dependency_graph.get(task_id, [])
            if not dependencies:
                return 0
            
            max_depth = max(calculate_depth(dep, visited.copy()) for dep in dependencies)
            return max_depth + 1
        
        for todo in todos:
            depth = calculate_depth(todo.id)
            depth_groups[depth].append(todo.id)
        
        # Tasks at the same depth can potentially be worked on in parallel
        for depth, task_ids in depth_groups.items():
            if len(task_ids) > 1:
                parallel_groups.append(task_ids)
        
        return parallel_groups

class TaskDecomposer:
    """Decomposes complex tasks into manageable subtasks"""
    
    def __init__(self, llm_adapter: LocalLLMAdapter):
        self.llm_adapter = llm_adapter
    
    def decompose_task(self, todo: Todo, max_subtasks: int = 5) -> List[Todo]:
        """Decompose a complex task into subtasks"""
        if len(todo.subtasks) > 0:
            return todo.subtasks  # Already decomposed
        
        # Use LLM to suggest decomposition
        decomposition_prompt = f"""
        Task: {todo.title}
        Description: {todo.description}
        Details: {todo.details}
        
        Context: This is a software development task that needs to be broken down into manageable subtasks.
        """
        
        analysis = self.llm_adapter.analyze_text(decomposition_prompt, "task_decomposition")
        
        # Generate subtasks based on analysis and heuristics
        subtasks = self._generate_subtasks(todo, max_subtasks)
        
        return subtasks
    
    def _generate_subtasks(self, todo: Todo, max_subtasks: int) -> List[Todo]:
        """Generate subtasks using heuristics and patterns"""
        subtasks = []
        
        # Common software development patterns
        patterns = self._identify_patterns(todo)
        
        for i, pattern in enumerate(patterns[:max_subtasks]):
            subtask_id = f"{todo.id}.{i+1}"
            subtask = Todo(
                id=subtask_id,
                title=pattern['title'],
                description=pattern['description'],
                priority=todo.priority,
                details=pattern.get('details', ''),
                test_strategy=pattern.get('test_strategy', ''),
                validation_criteria=pattern.get('validation_criteria', [])
            )
            subtasks.append(subtask)
        
        return subtasks
    
    def _identify_patterns(self, todo: Todo) -> List[Dict[str, Any]]:
        """Identify common development patterns in the task"""
        patterns = []
        text = f"{todo.title} {todo.description} {todo.details}".lower()
        
        # API development pattern
        if any(keyword in text for keyword in ['api', 'endpoint', 'service']):
            patterns.extend([
                {
                    'title': 'Design API specification',
                    'description': 'Define API endpoints, request/response formats, and error handling',
                    'details': 'Create OpenAPI/Swagger documentation',
                    'test_strategy': 'Validate API spec against requirements'
                },
                {
                    'title': 'Implement API endpoints',
                    'description': 'Code the actual API endpoints and business logic',
                    'details': 'Implement request validation, business logic, and response formatting',
                    'test_strategy': 'Unit tests for each endpoint'
                },
                {
                    'title': 'Add API integration tests',
                    'description': 'Create end-to-end tests for API functionality',
                    'details': 'Test complete request/response cycles with real data',
                    'test_strategy': 'Integration tests covering happy path and error scenarios'
                }
            ])
        
        # Database pattern
        if any(keyword in text for keyword in ['database', 'db', 'schema', 'model']):
            patterns.extend([
                {
                    'title': 'Design database schema',
                    'description': 'Create database tables and relationships',
                    'details': 'Define tables, columns, constraints, and indexes',
                    'test_strategy': 'Validate schema against data requirements'
                },
                {
                    'title': 'Implement database models',
                    'description': 'Create ORM models and database access layer',
                    'details': 'Implement models with proper relationships and validation',
                    'test_strategy': 'Unit tests for model methods and validation'
                },
                {
                    'title': 'Create database migrations',
                    'description': 'Implement database migration scripts',
                    'details': 'Create migration scripts for schema changes',
                    'test_strategy': 'Test migrations on sample data'
                }
            ])
        
        # Frontend pattern
        if any(keyword in text for keyword in ['frontend', 'ui', 'component', 'interface']):
            patterns.extend([
                {
                    'title': 'Design UI components',
                    'description': 'Create wireframes and component specifications',
                    'details': 'Define component props, state, and behavior',
                    'test_strategy': 'Review designs with stakeholders'
                },
                {
                    'title': 'Implement UI components',
                    'description': 'Code the actual UI components',
                    'details': 'Implement components with proper styling and behavior',
                    'test_strategy': 'Unit tests for component logic'
                },
                {
                    'title': 'Add UI integration tests',
                    'description': 'Create tests for UI interactions',
                    'details': 'Test user interactions and component integration',
                    'test_strategy': 'End-to-end tests for user workflows'
                }
            ])
        
        # Generic implementation pattern
        if not patterns:
            patterns.extend([
                {
                    'title': 'Plan implementation approach',
                    'description': f'Create detailed implementation plan for {todo.title}',
                    'details': 'Research requirements, design approach, identify dependencies',
                    'test_strategy': 'Review plan with team'
                },
                {
                    'title': 'Implement core functionality',
                    'description': f'Code the main functionality for {todo.title}',
                    'details': 'Implement the primary features and logic',
                    'test_strategy': 'Unit tests for core functionality'
                },
                {
                    'title': 'Add tests and validation',
                    'description': f'Create comprehensive tests for {todo.title}',
                    'details': 'Add unit tests, integration tests, and validation',
                    'test_strategy': 'Achieve target test coverage'
                }
            ])
        
        return patterns[:5]  # Limit to 5 patterns

class EnhancementGenerator:
    """Generates enhancements for todos"""
    
    def __init__(self, llm_adapter: LocalLLMAdapter):
        self.llm_adapter = llm_adapter
    
    def enhance_todo(self, todo: Todo, enhancement_types: List[EnhancementType]) -> Todo:
        """Apply multiple enhancements to a todo"""
        enhanced_todo = todo
        
        for enhancement_type in enhancement_types:
            enhancement_result = self._apply_enhancement(enhanced_todo, enhancement_type)
            enhanced_todo = Todo.from_dict(enhancement_result.enhanced_task)
            enhanced_todo.enhancement_history.append(enhancement_result)
        
        return enhanced_todo
    
    def _apply_enhancement(self, todo: Todo, enhancement_type: EnhancementType) -> EnhancementResult:
        """Apply a specific enhancement to a todo"""
        original_dict = todo.to_dict()
        enhanced_dict = original_dict.copy()
        
        if enhancement_type == EnhancementType.DESCRIPTION_ENHANCEMENT:
            enhanced_dict = self._enhance_description(enhanced_dict)
        elif enhancement_type == EnhancementType.TIME_ESTIMATION:
            enhanced_dict = self._add_time_estimation(enhanced_dict)
        elif enhancement_type == EnhancementType.RESOURCE_PLANNING:
            enhanced_dict = self._add_resource_planning(enhanced_dict)
        elif enhancement_type == EnhancementType.TEST_STRATEGY:
            enhanced_dict = self._add_test_strategy(enhanced_dict)
        elif enhancement_type == EnhancementType.VALIDATION_CRITERIA:
            enhanced_dict = self._add_validation_criteria(enhanced_dict)
        
        # Calculate quality improvement
        original_quality = todo.quality_metrics.overall_score
        enhanced_todo = Todo.from_dict(enhanced_dict)
        enhanced_quality = TodoAnalyzer(self.llm_adapter).analyze_todo(enhanced_todo).overall_score
        
        suggestions = self._generate_enhancement_suggestions(todo, enhancement_type)
        
        return EnhancementResult(
            enhancement_type=enhancement_type,
            original_task=original_dict,
            enhanced_task=enhanced_dict,
            quality_improvement=enhanced_quality - original_quality,
            suggestions=suggestions
        )
    
    def _enhance_description(self, todo_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the description of a todo"""
        current_description = todo_dict.get('description', '')
        title = todo_dict.get('title', '')
        
        if len(current_description) < 20:
            # Generate more detailed description
            enhanced_description = self._generate_detailed_description(title, current_description)
            todo_dict['description'] = enhanced_description
        
        return todo_dict
    
    def _generate_detailed_description(self, title: str, current_description: str) -> str:
        """Generate a more detailed description"""
        # Use LLM analysis to enhance description
        context = f"Title: {title}\nCurrent Description: {current_description}"
        analysis = self.llm_adapter.analyze_text(context, "description_enhancement")
        
        # Fallback to rule-based enhancement
        if not current_description:
            base_description = f"Implement {title.lower()}"
        else:
            base_description = current_description
        
        # Add common enhancement patterns
        enhancements = []
        
        if 'api' in title.lower():
            enhancements.append("Design and implement RESTful API endpoints")
        elif 'database' in title.lower():
            enhancements.append("Design database schema and implement data access layer")
        elif 'test' in title.lower():
            enhancements.append("Create comprehensive test suite with unit and integration tests")
        elif 'ui' in title.lower() or 'frontend' in title.lower():
            enhancements.append("Design and implement user interface components")
        
        if enhancements:
            return f"{base_description}. {'. '.join(enhancements)}."
        
        return base_description
    
    def _add_time_estimation(self, todo_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add time estimation to a todo"""
        if todo_dict.get('timeEstimate') is None:
            # Estimate based on complexity and type
            title = todo_dict.get('title', '').lower()
            description = todo_dict.get('description', '').lower()
            
            base_estimate = 120  # 2 hours base
            
            # Adjust based on keywords
            if any(keyword in title for keyword in ['api', 'service', 'endpoint']):
                base_estimate *= 2
            elif any(keyword in title for keyword in ['database', 'schema', 'migration']):
                base_estimate *= 1.5
            elif any(keyword in title for keyword in ['ui', 'frontend', 'component']):
                base_estimate *= 1.8
            elif any(keyword in title for keyword in ['test', 'testing']):
                base_estimate *= 1.2
            
            # Adjust based on complexity indicators
            if any(keyword in description for keyword in ['complex', 'multiple', 'integrate']):
                base_estimate *= 1.5
            
            todo_dict['timeEstimate'] = int(base_estimate)
        
        return todo_dict
    
    def _add_resource_planning(self, todo_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add resource planning to a todo"""
        if not todo_dict.get('resourceRequirements'):
            resources = []
            
            title = todo_dict.get('title', '').lower()
            description = todo_dict.get('description', '').lower()
            
            # Add common resource requirements
            if any(keyword in title for keyword in ['api', 'backend', 'service']):
                resources.extend(['Backend developer', 'API documentation', 'Testing environment'])
            elif any(keyword in title for keyword in ['database', 'schema']):
                resources.extend(['Database developer', 'Database access', 'Migration tools'])
            elif any(keyword in title for keyword in ['ui', 'frontend']):
                resources.extend(['Frontend developer', 'Design assets', 'Testing devices'])
            elif any(keyword in title for keyword in ['deploy', 'deployment']):
                resources.extend(['DevOps engineer', 'Production environment', 'Monitoring tools'])
            
            # Add generic resources
            if not resources:
                resources.append('Developer time')
            
            todo_dict['resourceRequirements'] = resources
        
        return todo_dict
    
    def _add_test_strategy(self, todo_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add test strategy to a todo"""
        if not todo_dict.get('testStrategy'):
            title = todo_dict.get('title', '').lower()
            
            strategies = []
            
            if any(keyword in title for keyword in ['api', 'service', 'endpoint']):
                strategies.extend([
                    'Unit tests for business logic',
                    'Integration tests for API endpoints',
                    'Contract tests for API specification'
                ])
            elif any(keyword in title for keyword in ['database', 'schema']):
                strategies.extend([
                    'Unit tests for data models',
                    'Integration tests for database operations',
                    'Migration tests with sample data'
                ])
            elif any(keyword in title for keyword in ['ui', 'frontend', 'component']):
                strategies.extend([
                    'Unit tests for component logic',
                    'Integration tests for user interactions',
                    'Visual regression tests'
                ])
            else:
                strategies.append('Unit tests for core functionality')
            
            todo_dict['testStrategy'] = '. '.join(strategies)
        
        return todo_dict
    
    def _add_validation_criteria(self, todo_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add validation criteria to a todo"""
        if not todo_dict.get('validationCriteria'):
            criteria = []
            
            title = todo_dict.get('title', '').lower()
            
            # Add common validation criteria
            if any(keyword in title for keyword in ['implement', 'create', 'build']):
                criteria.extend([
                    'Implementation is complete and functional',
                    'Code follows project standards and conventions',
                    'All tests pass successfully'
                ])
            
            if any(keyword in title for keyword in ['api', 'service']):
                criteria.extend([
                    'API endpoints respond correctly',
                    'API documentation is updated',
                    'Error handling is implemented'
                ])
            
            if any(keyword in title for keyword in ['database', 'schema']):
                criteria.extend([
                    'Database schema is properly defined',
                    'Data migrations run without errors',
                    'Data integrity is maintained'
                ])
            
            if any(keyword in title for keyword in ['ui', 'frontend']):
                criteria.extend([
                    'UI components render correctly',
                    'User interactions work as expected',
                    'UI is responsive and accessible'
                ])
            
            # Add generic criteria if none found
            if not criteria:
                criteria.extend([
                    'Task requirements are met',
                    'Implementation is tested and validated',
                    'Documentation is updated'
                ])
            
            todo_dict['validationCriteria'] = criteria
        
        return todo_dict
    
    def _generate_enhancement_suggestions(self, todo: Todo, enhancement_type: EnhancementType) -> List[str]:
        """Generate suggestions for specific enhancement type"""
        suggestions = []
        
        if enhancement_type == EnhancementType.DESCRIPTION_ENHANCEMENT:
            suggestions.append("Consider adding more specific technical details")
            suggestions.append("Include acceptance criteria and success metrics")
        elif enhancement_type == EnhancementType.TIME_ESTIMATION:
            suggestions.append("Break down time estimates by subtasks")
            suggestions.append("Include buffer time for unexpected issues")
        elif enhancement_type == EnhancementType.RESOURCE_PLANNING:
            suggestions.append("Consider skill requirements and availability")
            suggestions.append("Include external dependencies and tools")
        elif enhancement_type == EnhancementType.TEST_STRATEGY:
            suggestions.append("Include both positive and negative test cases")
            suggestions.append("Consider performance and security testing")
        elif enhancement_type == EnhancementType.VALIDATION_CRITERIA:
            suggestions.append("Make criteria measurable and specific")
            suggestions.append("Include both functional and non-functional requirements")
        
        return suggestions

class QualityScorer:
    """Scores todo quality across multiple dimensions"""
    
    def __init__(self, llm_adapter: LocalLLMAdapter):
        self.llm_adapter = llm_adapter
        self.analyzer = TodoAnalyzer(llm_adapter)
    
    def score_todo(self, todo: Todo) -> QualityMetrics:
        """Score a todo across all quality dimensions"""
        return self.analyzer.analyze_todo(todo)
    
    def score_todo_list(self, todos: List[Todo]) -> Dict[str, Any]:
        """Score a list of todos and provide aggregate metrics"""
        scores = []
        individual_scores = {}
        
        for todo in todos:
            metrics = self.score_todo(todo)
            scores.append(metrics.overall_score)
            individual_scores[todo.id] = metrics.to_dict()
        
        if not scores:
            return {
                'average_score': 0.0,
                'median_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'individual_scores': {},
                'recommendations': []
            }
        
        aggregate_metrics = {
            'average_score': statistics.mean(scores),
            'median_score': statistics.median(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'individual_scores': individual_scores,
            'recommendations': self._generate_recommendations(todos, individual_scores)
        }
        
        return aggregate_metrics
    
    def _generate_recommendations(self, todos: List[Todo], scores: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate recommendations based on scoring results"""
        recommendations = []
        
        # Find low-scoring todos
        low_scoring_todos = [
            todo for todo in todos 
            if scores[todo.id]['overall_score'] < 0.5
        ]
        
        if low_scoring_todos:
            recommendations.append(f"Focus on improving {len(low_scoring_todos)} low-scoring tasks")
        
        # Find common quality issues
        clarity_issues = [
            todo for todo in todos
            if scores[todo.id]['clarity_score'] < 0.5
        ]
        
        if clarity_issues:
            recommendations.append(f"Improve clarity for {len(clarity_issues)} tasks")
        
        completeness_issues = [
            todo for todo in todos
            if scores[todo.id]['completeness_score'] < 0.5
        ]
        
        if completeness_issues:
            recommendations.append(f"Add more details to {len(completeness_issues)} tasks")
        
        testability_issues = [
            todo for todo in todos
            if scores[todo.id]['testability_score'] < 0.5
        ]
        
        if testability_issues:
            recommendations.append(f"Add test strategies to {len(testability_issues)} tasks")
        
        return recommendations

class TaskMasterIntegration:
    """Integration layer for Task Master CLI compatibility"""
    
    def __init__(self, taskmaster_dir: str = ".taskmaster"):
        self.taskmaster_dir = Path(taskmaster_dir)
        self.tasks_file = self.taskmaster_dir / "tasks" / "tasks.json"
        self.config_file = self.taskmaster_dir / "config.json"
    
    def load_tasks(self) -> List[Todo]:
        """Load tasks from Task Master tasks.json file"""
        if not self.tasks_file.exists():
            return []
        
        try:
            with open(self.tasks_file, 'r') as f:
                data = json.load(f)
            
            todos = []
            for task_data in data.get('tasks', []):
                todo = Todo.from_dict(task_data)
                todos.append(todo)
            
            return todos
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")
            return []
    
    def save_tasks(self, todos: List[Todo]) -> bool:
        """Save tasks to Task Master tasks.json file"""
        try:
            # Ensure directory exists
            self.tasks_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert todos to Task Master format
            tasks_data = {
                'tasks': [todo.to_dict() for todo in todos],
                'lastUpdated': datetime.now().isoformat()
            }
            
            with open(self.tasks_file, 'w') as f:
                json.dump(tasks_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")
            return False
    
    def get_task_by_id(self, task_id: str) -> Optional[Todo]:
        """Get a specific task by ID"""
        todos = self.load_tasks()
        for todo in todos:
            if todo.id == task_id:
                return todo
            # Check subtasks
            for subtask in todo.subtasks:
                if subtask.id == task_id:
                    return subtask
        return None
    
    def update_task(self, task_id: str, updated_todo: Todo) -> bool:
        """Update a specific task"""
        todos = self.load_tasks()
        
        for i, todo in enumerate(todos):
            if todo.id == task_id:
                todos[i] = updated_todo
                return self.save_tasks(todos)
            
            # Check subtasks
            for j, subtask in enumerate(todo.subtasks):
                if subtask.id == task_id:
                    todo.subtasks[j] = updated_todo
                    return self.save_tasks(todos)
        
        return False
    
    def export_enhanced_tasks(self, todos: List[Todo], output_file: str) -> bool:
        """Export enhanced tasks to a file"""
        try:
            export_data = {
                'enhanced_tasks': [todo.to_dict() for todo in todos],
                'export_timestamp': datetime.now().isoformat(),
                'enhancement_summary': self._generate_enhancement_summary(todos)
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to export tasks: {e}")
            return False
    
    def _generate_enhancement_summary(self, todos: List[Todo]) -> Dict[str, Any]:
        """Generate a summary of enhancements applied"""
        summary = {
            'total_tasks': len(todos),
            'enhanced_tasks': 0,
            'enhancement_types': defaultdict(int),
            'quality_improvements': []
        }
        
        for todo in todos:
            if todo.enhancement_history:
                summary['enhanced_tasks'] += 1
                for enhancement in todo.enhancement_history:
                    summary['enhancement_types'][enhancement.enhancement_type.value] += 1
                    summary['quality_improvements'].append(enhancement.quality_improvement)
        
        if summary['quality_improvements']:
            summary['average_quality_improvement'] = statistics.mean(summary['quality_improvements'])
        
        return summary

class MetaLearningSystem:
    """Meta-learning system for enhancement strategy optimization"""
    
    def __init__(self, learning_data_file: str = "enhancement_learning.pkl"):
        self.learning_data_file = learning_data_file
        self.enhancement_outcomes = self._load_learning_data()
    
    def _load_learning_data(self) -> Dict[str, List[float]]:
        """Load historical enhancement outcomes"""
        if os.path.exists(self.learning_data_file):
            try:
                with open(self.learning_data_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load learning data: {e}")
        
        return defaultdict(list)
    
    def _save_learning_data(self):
        """Save learning data to file"""
        try:
            with open(self.learning_data_file, 'wb') as f:
                pickle.dump(dict(self.enhancement_outcomes), f)
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def record_enhancement_outcome(self, enhancement_type: EnhancementType, 
                                 quality_improvement: float, 
                                 context: Dict[str, Any]):
        """Record the outcome of an enhancement for learning"""
        key = f"{enhancement_type.value}_{self._context_to_key(context)}"
        self.enhancement_outcomes[key].append(quality_improvement)
        self._save_learning_data()
    
    def get_enhancement_effectiveness(self, enhancement_type: EnhancementType, 
                                   context: Dict[str, Any]) -> float:
        """Get the historical effectiveness of an enhancement type in a given context"""
        key = f"{enhancement_type.value}_{self._context_to_key(context)}"
        outcomes = self.enhancement_outcomes.get(key, [])
        
        if not outcomes:
            return 0.5  # Default effectiveness
        
        return statistics.mean(outcomes)
    
    def recommend_enhancement_strategy(self, todo: Todo) -> List[EnhancementType]:
        """Recommend enhancement strategy based on learning"""
        context = self._extract_context(todo)
        
        # Score all enhancement types
        scored_enhancements = []
        for enhancement_type in EnhancementType:
            effectiveness = self.get_enhancement_effectiveness(enhancement_type, context)
            scored_enhancements.append((enhancement_type, effectiveness))
        
        # Sort by effectiveness and return top recommendations
        scored_enhancements.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 most effective enhancements
        return [enhancement[0] for enhancement in scored_enhancements[:3]]
    
    def _context_to_key(self, context: Dict[str, Any]) -> str:
        """Convert context to a hashable key"""
        # Create a simple hash of the context
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()[:8]
    
    def _extract_context(self, todo: Todo) -> Dict[str, Any]:
        """Extract relevant context from a todo for learning"""
        text = f"{todo.title} {todo.description}".lower()
        
        context = {
            'has_api': 'api' in text,
            'has_database': any(keyword in text for keyword in ['database', 'db', 'schema']),
            'has_ui': any(keyword in text for keyword in ['ui', 'frontend', 'component']),
            'has_test': 'test' in text,
            'complexity_level': 'high' if len(text.split()) > 20 else 'low',
            'has_dependencies': len(todo.dependencies) > 0,
            'has_subtasks': len(todo.subtasks) > 0
        }
        
        return context

class PerformanceMonitor:
    """Monitor and track enhancement performance"""
    
    def __init__(self):
        self.metrics = {
            'enhancement_times': defaultdict(list),
            'quality_improvements': defaultdict(list),
            'success_rates': defaultdict(int),
            'error_counts': defaultdict(int)
        }
    
    def record_enhancement_time(self, enhancement_type: EnhancementType, duration: float):
        """Record time taken for an enhancement"""
        self.metrics['enhancement_times'][enhancement_type.value].append(duration)
    
    def record_quality_improvement(self, enhancement_type: EnhancementType, improvement: float):
        """Record quality improvement from an enhancement"""
        self.metrics['quality_improvements'][enhancement_type.value].append(improvement)
    
    def record_success(self, enhancement_type: EnhancementType):
        """Record successful enhancement"""
        self.metrics['success_rates'][enhancement_type.value] += 1
    
    def record_error(self, enhancement_type: EnhancementType):
        """Record enhancement error"""
        self.metrics['error_counts'][enhancement_type.value] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        report = {
            'enhancement_statistics': {},
            'overall_performance': {}
        }
        
        for enhancement_type in EnhancementType:
            type_name = enhancement_type.value
            
            times = self.metrics['enhancement_times'].get(type_name, [])
            improvements = self.metrics['quality_improvements'].get(type_name, [])
            successes = self.metrics['success_rates'].get(type_name, 0)
            errors = self.metrics['error_counts'].get(type_name, 0)
            
            report['enhancement_statistics'][type_name] = {
                'average_time': statistics.mean(times) if times else 0,
                'average_improvement': statistics.mean(improvements) if improvements else 0,
                'success_count': successes,
                'error_count': errors,
                'success_rate': successes / (successes + errors) if (successes + errors) > 0 else 0
            }
        
        # Overall performance
        all_times = [time for times in self.metrics['enhancement_times'].values() for time in times]
        all_improvements = [imp for imps in self.metrics['quality_improvements'].values() for imp in imps]
        
        report['overall_performance'] = {
            'total_enhancements': len(all_times),
            'average_time': statistics.mean(all_times) if all_times else 0,
            'average_improvement': statistics.mean(all_improvements) if all_improvements else 0,
            'total_errors': sum(self.metrics['error_counts'].values())
        }
        
        return report

class RecursiveTodoEnhancementEngine:
    """Main engine for recursive todo enhancement"""
    
    def __init__(self, 
                 taskmaster_dir: str = ".taskmaster",
                 max_recursion_depth: int = 3,
                 enable_meta_learning: bool = True):
        self.taskmaster_dir = taskmaster_dir
        self.max_recursion_depth = max_recursion_depth
        self.enable_meta_learning = enable_meta_learning
        
        # Initialize components
        self.llm_adapter = LocalLLMAdapter()
        self.todo_analyzer = TodoAnalyzer(self.llm_adapter)
        self.dependency_analyzer = DependencyAnalyzer()
        self.task_decomposer = TaskDecomposer(self.llm_adapter)
        self.enhancement_generator = EnhancementGenerator(self.llm_adapter)
        self.quality_scorer = QualityScorer(self.llm_adapter)
        self.taskmaster_integration = TaskMasterIntegration(taskmaster_dir)
        self.meta_learning = MetaLearningSystem() if enable_meta_learning else None
        self.performance_monitor = PerformanceMonitor()
        
        logger.info("Recursive Todo Enhancement Engine initialized")
    
    def enhance_todos(self, 
                     todos: Optional[List[Todo]] = None,
                     enhancement_types: Optional[List[EnhancementType]] = None,
                     recursive_depth: int = 1) -> List[Todo]:
        """Main method to enhance todos with recursive improvement cycles"""
        if todos is None:
            todos = self.taskmaster_integration.load_tasks()
        
        if not todos:
            logger.warning("No todos to enhance")
            return []
        
        logger.info(f"Starting enhancement of {len(todos)} todos with depth {recursive_depth}")
        
        enhanced_todos = todos.copy()
        
        # Apply recursive enhancement cycles
        for cycle in range(recursive_depth):
            logger.info(f"Starting enhancement cycle {cycle + 1}/{recursive_depth}")
            
            enhanced_todos = self._apply_enhancement_cycle(
                enhanced_todos, 
                enhancement_types
            )
        
        # Save enhanced todos
        self.taskmaster_integration.save_tasks(enhanced_todos)
        
        logger.info(f"Enhancement complete. Enhanced {len(enhanced_todos)} todos")
        return enhanced_todos
    
    def _apply_enhancement_cycle(self, 
                               todos: List[Todo],
                               enhancement_types: Optional[List[EnhancementType]] = None) -> List[Todo]:
        """Apply one cycle of enhancements to todos"""
        enhanced_todos = []
        
        for todo in todos:
            enhanced_todo = self._enhance_single_todo(todo, enhancement_types)
            enhanced_todos.append(enhanced_todo)
        
        return enhanced_todos
    
    def _enhance_single_todo(self, 
                           todo: Todo,
                           enhancement_types: Optional[List[EnhancementType]] = None) -> Todo:
        """Enhance a single todo"""
        start_time = time.time()
        
        try:
            # Analyze current quality
            original_quality = self.quality_scorer.score_todo(todo)
            todo.quality_metrics = original_quality
            
            # Determine enhancement types to apply
            if enhancement_types is None:
                if self.meta_learning:
                    enhancement_types = self.meta_learning.recommend_enhancement_strategy(todo)
                else:
                    enhancement_types = [
                        EnhancementType.DESCRIPTION_ENHANCEMENT,
                        EnhancementType.TIME_ESTIMATION,
                        EnhancementType.TEST_STRATEGY
                    ]
            
            # Apply enhancements
            enhanced_todo = self.enhancement_generator.enhance_todo(todo, enhancement_types)
            
            # Re-analyze quality
            enhanced_quality = self.quality_scorer.score_todo(enhanced_todo)
            enhanced_todo.quality_metrics = enhanced_quality
            
            # Record performance metrics
            duration = time.time() - start_time
            quality_improvement = enhanced_quality.overall_score - original_quality.overall_score
            
            for enhancement_type in enhancement_types:
                self.performance_monitor.record_enhancement_time(enhancement_type, duration)
                self.performance_monitor.record_quality_improvement(enhancement_type, quality_improvement)
                self.performance_monitor.record_success(enhancement_type)
                
                # Record for meta-learning
                if self.meta_learning:
                    context = self.meta_learning._extract_context(todo)
                    self.meta_learning.record_enhancement_outcome(
                        enhancement_type, quality_improvement, context
                    )
            
            # Enhance subtasks recursively
            if enhanced_todo.subtasks:
                enhanced_subtasks = []
                for subtask in enhanced_todo.subtasks:
                    enhanced_subtask = self._enhance_single_todo(subtask, enhancement_types)
                    enhanced_subtasks.append(enhanced_subtask)
                enhanced_todo.subtasks = enhanced_subtasks
            
            return enhanced_todo
            
        except Exception as e:
            logger.error(f"Failed to enhance todo {todo.id}: {e}")
            
            # Record error
            if enhancement_types:
                for enhancement_type in enhancement_types:
                    self.performance_monitor.record_error(enhancement_type)
            
            return todo
    
    def analyze_project_todos(self) -> Dict[str, Any]:
        """Analyze all project todos and provide comprehensive report"""
        todos = self.taskmaster_integration.load_tasks()
        
        if not todos:
            return {"error": "No todos found in project"}
        
        # Build dependency graph
        self.dependency_analyzer.build_dependency_graph(todos)
        
        # Analyze todos
        analysis_report = {
            'project_overview': {
                'total_todos': len(todos),
                'by_status': self._count_by_status(todos),
                'by_priority': self._count_by_priority(todos)
            },
            'quality_analysis': self.quality_scorer.score_todo_list(todos),
            'optimization_opportunities': self.todo_analyzer.find_optimization_opportunities(todos),
            'dependency_analysis': {
                'circular_dependencies': self.dependency_analyzer.detect_circular_dependencies(),
                'parallel_opportunities': self.dependency_analyzer.identify_parallel_opportunities(todos),
                'optimal_order': self.dependency_analyzer.optimize_task_order(todos)
            },
            'decomposition_recommendations': self._find_decomposition_recommendations(todos),
            'performance_metrics': self.performance_monitor.get_performance_report()
        }
        
        return analysis_report
    
    def _count_by_status(self, todos: List[Todo]) -> Dict[str, int]:
        """Count todos by status"""
        counts = defaultdict(int)
        for todo in todos:
            counts[todo.status.value] += 1
        return dict(counts)
    
    def _count_by_priority(self, todos: List[Todo]) -> Dict[str, int]:
        """Count todos by priority"""
        counts = defaultdict(int)
        for todo in todos:
            counts[todo.priority.value] += 1
        return dict(counts)
    
    def _find_decomposition_recommendations(self, todos: List[Todo]) -> List[Dict[str, Any]]:
        """Find todos that should be decomposed"""
        recommendations = []
        
        for todo in todos:
            if len(todo.subtasks) == 0:
                complexity_score = self.todo_analyzer._calculate_complexity_score(todo)
                if complexity_score > 0.6:
                    recommendations.append({
                        'todo_id': todo.id,
                        'todo_title': todo.title,
                        'complexity_score': complexity_score,
                        'recommended_subtasks': len(self.task_decomposer.decompose_task(todo))
                    })
        
        return recommendations
    
    def auto_decompose_complex_todos(self, complexity_threshold: float = 0.6) -> List[Todo]:
        """Automatically decompose complex todos into subtasks"""
        todos = self.taskmaster_integration.load_tasks()
        modified_todos = []
        
        for todo in todos:
            if len(todo.subtasks) == 0:
                complexity_score = self.todo_analyzer._calculate_complexity_score(todo)
                if complexity_score > complexity_threshold:
                    logger.info(f"Decomposing complex todo: {todo.title}")
                    subtasks = self.task_decomposer.decompose_task(todo)
                    todo.subtasks = subtasks
                    modified_todos.append(todo)
        
        if modified_todos:
            self.taskmaster_integration.save_tasks(todos)
            logger.info(f"Decomposed {len(modified_todos)} complex todos")
        
        return todos
    
    def optimize_dependencies(self) -> Dict[str, Any]:
        """Optimize task dependencies and resolve issues"""
        todos = self.taskmaster_integration.load_tasks()
        
        # Build dependency graph
        self.dependency_analyzer.build_dependency_graph(todos)
        
        # Detect and resolve circular dependencies
        circular_deps = self.dependency_analyzer.detect_circular_dependencies()
        resolutions = []
        
        if circular_deps:
            resolutions = self.dependency_analyzer.suggest_dependency_resolution(circular_deps)
            
            # Apply automatic resolutions (remove weakest edges)
            for resolution in resolutions:
                if resolution['type'] == 'circular_dependency_resolution':
                    from_task, to_task = resolution['suggested_removal']
                    
                    # Find and update the task
                    for todo in todos:
                        if todo.id == from_task and to_task in todo.dependencies:
                            todo.dependencies.remove(to_task)
                            logger.info(f"Removed circular dependency: {from_task} -> {to_task}")
        
        # Save optimized todos
        self.taskmaster_integration.save_tasks(todos)
        
        return {
            'circular_dependencies_found': len(circular_deps),
            'resolutions_applied': len(resolutions),
            'optimized_order': self.dependency_analyzer.optimize_task_order(todos),
            'parallel_opportunities': self.dependency_analyzer.identify_parallel_opportunities(todos)
        }
    
    def batch_enhance_by_pattern(self, pattern: str, enhancement_types: List[EnhancementType]) -> List[Todo]:
        """Enhance todos that match a specific pattern"""
        todos = self.taskmaster_integration.load_tasks()
        matching_todos = []
        
        for todo in todos:
            if pattern.lower() in todo.title.lower() or pattern.lower() in todo.description.lower():
                matching_todos.append(todo)
        
        if not matching_todos:
            logger.info(f"No todos found matching pattern: {pattern}")
            return []
        
        logger.info(f"Enhancing {len(matching_todos)} todos matching pattern: {pattern}")
        
        # Apply enhancements
        enhanced_todos = []
        for todo in matching_todos:
            enhanced_todo = self.enhancement_generator.enhance_todo(todo, enhancement_types)
            enhanced_todos.append(enhanced_todo)
        
        # Update todos in the main list
        for enhanced_todo in enhanced_todos:
            for i, todo in enumerate(todos):
                if todo.id == enhanced_todo.id:
                    todos[i] = enhanced_todo
                    break
        
        self.taskmaster_integration.save_tasks(todos)
        return enhanced_todos
    
    def export_enhancement_report(self, output_file: str) -> bool:
        """Export comprehensive enhancement report"""
        try:
            report = self.analyze_project_todos()
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Enhancement report exported to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export enhancement report: {e}")
            return False

# CLI Interface Functions
def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Recursive Todo Enhancement Engine")
    parser.add_argument("--taskmaster-dir", default=".taskmaster", help="Task Master directory")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum recursion depth")
    parser.add_argument("--no-meta-learning", action="store_true", help="Disable meta-learning")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Enhance command
    enhance_parser = subparsers.add_parser("enhance", help="Enhance todos")
    enhance_parser.add_argument("--depth", type=int, default=1, help="Enhancement depth")
    enhance_parser.add_argument("--types", nargs="+", help="Enhancement types to apply")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze project todos")
    analyze_parser.add_argument("--output", help="Output file for analysis report")
    
    # Decompose command
    decompose_parser = subparsers.add_parser("decompose", help="Decompose complex todos")
    decompose_parser.add_argument("--threshold", type=float, default=0.6, help="Complexity threshold")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize dependencies")
    
    # Batch enhance command
    batch_parser = subparsers.add_parser("batch-enhance", help="Batch enhance by pattern")
    batch_parser.add_argument("pattern", help="Pattern to match")
    batch_parser.add_argument("--types", nargs="+", required=True, help="Enhancement types")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = RecursiveTodoEnhancementEngine(
        taskmaster_dir=args.taskmaster_dir,
        max_recursion_depth=args.max_depth,
        enable_meta_learning=not args.no_meta_learning
    )
    
    if args.command == "enhance":
        enhancement_types = None
        if args.types:
            enhancement_types = [EnhancementType(t) for t in args.types]
        
        enhanced_todos = engine.enhance_todos(
            enhancement_types=enhancement_types,
            recursive_depth=args.depth
        )
        
        print(f"Enhanced {len(enhanced_todos)} todos")
        
    elif args.command == "analyze":
        report = engine.analyze_project_todos()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Analysis report saved to: {args.output}")
        else:
            print(json.dumps(report, indent=2, default=str))
            
    elif args.command == "decompose":
        todos = engine.auto_decompose_complex_todos(args.threshold)
        print(f"Processed {len(todos)} todos for decomposition")
        
    elif args.command == "optimize":
        result = engine.optimize_dependencies()
        print(f"Optimization complete: {result}")
        
    elif args.command == "batch-enhance":
        enhancement_types = [EnhancementType(t) for t in args.types]
        enhanced_todos = engine.batch_enhance_by_pattern(args.pattern, enhancement_types)
        print(f"Enhanced {len(enhanced_todos)} todos matching pattern: {args.pattern}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()