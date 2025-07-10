#!/usr/bin/env python3
"""
Recursive Todo Enhancement Engine
A self-improving system that autonomously enhances todo lists and workflows
Built on the recursive meta-learning framework
"""

import json
import os
import re
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class EnhancementType(Enum):
    """Types of enhancements that can be applied"""
    CLARITY_IMPROVEMENT = "clarity_improvement"
    TASK_DECOMPOSITION = "task_decomposition"
    DEPENDENCY_ADDITION = "dependency_addition"
    PRIORITY_ADJUSTMENT = "priority_adjustment"
    DUPLICATE_RESOLUTION = "duplicate_resolution"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    CONTEXT_ENRICHMENT = "context_enrichment"

@dataclass
class EnhancementSuggestion:
    """Represents an enhancement suggestion"""
    id: str
    type: EnhancementType
    todo_id: str
    description: str
    confidence: float
    suggested_change: Dict[str, Any]
    reasoning: str
    applied: bool = False
    user_feedback: Optional[Dict] = None

@dataclass
class TodoQualityMetrics:
    """Quality metrics for a todo item"""
    clarity_score: float
    actionability_score: float
    specificity_score: float
    completeness_score: float
    overall_score: float

class RecursiveTodoEnhancer:
    """Core recursive todo enhancement engine"""
    
    def __init__(self, config_path: str = ".taskmaster/enhancement_config.json"):
        self.config_path = config_path
        self.enhancement_history = []
        self.learning_data = {}
        self.performance_metrics = {}
        self.recursive_depth = 0
        self.max_recursive_depth = 5
        
        # Load or initialize configuration
        self.config = self._load_config()
        
        # Enhancement strategies with learned weights
        self.enhancement_strategies = {
            EnhancementType.CLARITY_IMPROVEMENT: {
                "weight": 1.0,
                "success_rate": 0.8,
                "patterns": []
            },
            EnhancementType.TASK_DECOMPOSITION: {
                "weight": 1.0,
                "success_rate": 0.75,
                "patterns": []
            },
            EnhancementType.DEPENDENCY_ADDITION: {
                "weight": 1.0,
                "success_rate": 0.9,
                "patterns": []
            },
            EnhancementType.PRIORITY_ADJUSTMENT: {
                "weight": 1.0,
                "success_rate": 0.7,
                "patterns": []
            },
            EnhancementType.DUPLICATE_RESOLUTION: {
                "weight": 1.0,
                "success_rate": 0.95,
                "patterns": []
            },
            EnhancementType.WORKFLOW_OPTIMIZATION: {
                "weight": 1.0,
                "success_rate": 0.6,
                "patterns": []
            },
            EnhancementType.CONTEXT_ENRICHMENT: {
                "weight": 1.0,
                "success_rate": 0.8,
                "patterns": []
            }
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load enhancement configuration"""
        default_config = {
            "enhancement_threshold": 0.6,
            "auto_apply_threshold": 0.9,
            "learning_rate": 0.1,
            "feedback_weight": 0.7,
            "recursive_improvement_enabled": True,
            "max_suggestions_per_todo": 3
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    default_config.update(config)
            return default_config
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return default_config
    
    def analyze_todo_quality(self, todo: Dict[str, Any]) -> TodoQualityMetrics:
        """Analyze the quality of a todo item"""
        content = todo.get('content', '')
        
        # Clarity score based on language patterns
        clarity_score = self._calculate_clarity_score(content)
        
        # Actionability score based on action verbs and specificity
        actionability_score = self._calculate_actionability_score(content)
        
        # Specificity score based on concrete details
        specificity_score = self._calculate_specificity_score(content)
        
        # Completeness score based on required information
        completeness_score = self._calculate_completeness_score(todo)
        
        # Overall weighted score
        overall_score = (
            0.3 * clarity_score +
            0.3 * actionability_score +
            0.2 * specificity_score +
            0.2 * completeness_score
        )
        
        return TodoQualityMetrics(
            clarity_score=clarity_score,
            actionability_score=actionability_score,
            specificity_score=specificity_score,
            completeness_score=completeness_score,
            overall_score=overall_score
        )
    
    def _calculate_clarity_score(self, content: str) -> float:
        """Calculate clarity score for todo content"""
        if not content or len(content.strip()) < 3:
            return 0.0
        
        # Factors that improve clarity
        clarity_factors = []
        
        # Check for clear action words
        action_words = ['implement', 'create', 'design', 'build', 'analyze', 'test', 'fix', 'update', 'add', 'remove']
        has_action = any(word in content.lower() for word in action_words)
        clarity_factors.append(0.8 if has_action else 0.3)
        
        # Check for vague terms
        vague_terms = ['something', 'stuff', 'things', 'maybe', 'possibly', 'kind of']
        has_vague = any(term in content.lower() for term in vague_terms)
        clarity_factors.append(0.2 if has_vague else 0.8)
        
        # Check length appropriateness
        length_score = min(1.0, len(content) / 50) if len(content) < 200 else 0.7
        clarity_factors.append(length_score)
        
        # Check for specific details
        has_specifics = bool(re.search(r'\d+|[A-Z][a-z]+\s[A-Z][a-z]+|\w+\.\w+', content))
        clarity_factors.append(0.9 if has_specifics else 0.5)
        
        return statistics.mean(clarity_factors)
    
    def _calculate_actionability_score(self, content: str) -> float:
        """Calculate actionability score for todo content"""
        if not content:
            return 0.0
        
        actionability_factors = []
        
        # Check for imperative mood (starts with verb)
        starts_with_verb = bool(re.match(r'^(implement|create|design|build|analyze|test|fix|update|add|remove|write|read|install|configure|deploy)', content.lower()))
        actionability_factors.append(0.9 if starts_with_verb else 0.4)
        
        # Check for measurable outcomes
        has_outcome = bool(re.search(r'(complete|finish|deliver|submit|publish|deploy|release)', content.lower()))
        actionability_factors.append(0.8 if has_outcome else 0.5)
        
        # Check for concrete objects/targets
        has_targets = bool(re.search(r'(file|function|class|module|component|system|feature|bug|issue)', content.lower()))
        actionability_factors.append(0.8 if has_targets else 0.4)
        
        # Avoid abstract language
        abstract_terms = ['consider', 'think about', 'explore', 'investigate', 'research']
        has_abstract = any(term in content.lower() for term in abstract_terms)
        actionability_factors.append(0.3 if has_abstract else 0.8)
        
        return statistics.mean(actionability_factors)
    
    def _calculate_specificity_score(self, content: str) -> float:
        """Calculate specificity score for todo content"""
        if not content:
            return 0.0
        
        specificity_factors = []
        
        # Check for numbers/quantities
        has_numbers = bool(re.search(r'\d+', content))
        specificity_factors.append(0.8 if has_numbers else 0.4)
        
        # Check for file/path references
        has_paths = bool(re.search(r'[/\\]\w+|\.py|\.js|\.md|\w+\.\w+', content))
        specificity_factors.append(0.9 if has_paths else 0.5)
        
        # Check for technology/tool names
        tech_terms = ['python', 'javascript', 'react', 'node', 'git', 'docker', 'api', 'database', 'sql']
        has_tech = any(term in content.lower() for term in tech_terms)
        specificity_factors.append(0.8 if has_tech else 0.5)
        
        # Check for proper nouns
        has_proper_nouns = bool(re.search(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', content))
        specificity_factors.append(0.7 if has_proper_nouns else 0.4)
        
        return statistics.mean(specificity_factors)
    
    def _calculate_completeness_score(self, todo: Dict[str, Any]) -> float:
        """Calculate completeness score for todo item"""
        completeness_factors = []
        
        # Check for required fields
        has_content = bool(todo.get('content', '').strip())
        completeness_factors.append(1.0 if has_content else 0.0)
        
        has_priority = bool(todo.get('priority'))
        completeness_factors.append(0.8 if has_priority else 0.5)
        
        has_status = bool(todo.get('status'))
        completeness_factors.append(0.9 if has_status else 0.6)
        
        has_id = bool(todo.get('id'))
        completeness_factors.append(0.9 if has_id else 0.7)
        
        return statistics.mean(completeness_factors)
    
    def generate_enhancement_suggestions(self, todos: List[Dict[str, Any]]) -> List[EnhancementSuggestion]:
        """Generate enhancement suggestions for a list of todos"""
        suggestions = []
        
        # Analyze each todo individually
        for todo in todos:
            todo_suggestions = self._analyze_single_todo(todo)
            suggestions.extend(todo_suggestions)
        
        # Analyze todos collectively for patterns and relationships
        collective_suggestions = self._analyze_todo_relationships(todos)
        suggestions.extend(collective_suggestions)
        
        # Filter and rank suggestions
        filtered_suggestions = self._filter_and_rank_suggestions(suggestions)
        
        return filtered_suggestions
    
    def _analyze_single_todo(self, todo: Dict[str, Any]) -> List[EnhancementSuggestion]:
        """Analyze a single todo and generate enhancement suggestions"""
        suggestions = []
        todo_id = todo.get('id', str(uuid.uuid4()))
        quality_metrics = self.analyze_todo_quality(todo)
        
        # Clarity improvement suggestions
        if quality_metrics.clarity_score < 0.7:
            suggestions.append(self._suggest_clarity_improvement(todo, quality_metrics))
        
        # Task decomposition suggestions
        if self._should_decompose_task(todo):
            suggestions.append(self._suggest_task_decomposition(todo, quality_metrics))
        
        # Priority adjustment suggestions
        if self._should_adjust_priority(todo, quality_metrics):
            suggestions.append(self._suggest_priority_adjustment(todo, quality_metrics))
        
        # Context enrichment suggestions
        if quality_metrics.specificity_score < 0.6:
            suggestions.append(self._suggest_context_enrichment(todo, quality_metrics))
        
        return [s for s in suggestions if s is not None]
    
    def _suggest_clarity_improvement(self, todo: Dict[str, Any], metrics: TodoQualityMetrics) -> Optional[EnhancementSuggestion]:
        """Suggest clarity improvements for a todo"""
        content = todo.get('content', '')
        
        # Generate improved content based on patterns
        improved_content = self._improve_content_clarity(content)
        
        if improved_content != content:
            return EnhancementSuggestion(
                id=str(uuid.uuid4()),
                type=EnhancementType.CLARITY_IMPROVEMENT,
                todo_id=todo.get('id', ''),
                description=f"Improve clarity from {metrics.clarity_score:.2f} to estimated 0.85+",
                confidence=0.8,
                suggested_change={"content": improved_content},
                reasoning="Enhanced clarity by adding specific action verbs and removing vague language"
            )
        return None
    
    def _improve_content_clarity(self, content: str) -> str:
        """Improve content clarity using learned patterns"""
        if not content:
            return content
        
        improved = content
        
        # Replace vague verbs with specific ones
        vague_replacements = {
            'handle': 'implement',
            'deal with': 'resolve',
            'work on': 'develop',
            'look at': 'analyze',
            'check': 'validate',
            'fix up': 'refactor',
            'make better': 'optimize'
        }
        
        for vague, specific in vague_replacements.items():
            improved = re.sub(rf'\b{vague}\b', specific, improved, flags=re.IGNORECASE)
        
        # Ensure it starts with an action verb if it doesn't already
        action_verbs = ['implement', 'create', 'design', 'build', 'analyze', 'test', 'fix', 'update', 'add', 'remove']
        if not any(improved.lower().startswith(verb) for verb in action_verbs):
            # Try to infer appropriate action verb
            if 'bug' in improved.lower() or 'error' in improved.lower():
                improved = f"Fix {improved.lower()}"
            elif 'test' in improved.lower():
                improved = f"Implement {improved.lower()}"
            elif 'new' in improved.lower():
                improved = f"Create {improved.lower()}"
            else:
                improved = f"Implement {improved.lower()}"
        
        return improved.strip()
    
    def _should_decompose_task(self, todo: Dict[str, Any]) -> bool:
        """Determine if a task should be decomposed"""
        content = todo.get('content', '')
        
        # Check for complexity indicators
        complexity_indicators = [
            len(content) > 100,  # Long descriptions
            ' and ' in content.lower(),  # Multiple actions
            'implement' in content.lower() and ('system' in content.lower() or 'framework' in content.lower()),
            content.count(',') > 2,  # Multiple comma-separated items
            bool(re.search(r'\d+\s+(steps?|phases?|parts?)', content.lower()))  # Explicit multi-step indicators
        ]
        
        return sum(complexity_indicators) >= 2
    
    def _suggest_task_decomposition(self, todo: Dict[str, Any], metrics: TodoQualityMetrics) -> Optional[EnhancementSuggestion]:
        """Suggest task decomposition for complex todos"""
        content = todo.get('content', '')
        
        # Generate subtask suggestions
        subtasks = self._generate_subtasks(content)
        
        if len(subtasks) > 1:
            return EnhancementSuggestion(
                id=str(uuid.uuid4()),
                type=EnhancementType.TASK_DECOMPOSITION,
                todo_id=todo.get('id', ''),
                description=f"Break down complex task into {len(subtasks)} subtasks",
                confidence=0.7,
                suggested_change={"subtasks": subtasks},
                reasoning="Task appears complex and could benefit from decomposition for better tracking"
            )
        return None
    
    def _generate_subtasks(self, content: str) -> List[str]:
        """Generate subtasks from a complex task description"""
        subtasks = []
        
        # Look for explicit enumeration
        enumeration_patterns = [
            r'(\d+)\.?\s+([^.]+(?:\.|$))',  # Numbered lists
            r'([a-z])\)?\s+([^.]+(?:\.|$))',  # Lettered lists
            r'(?:^|\n)\s*[-*•]\s*([^.\n]+)',  # Bullet points
        ]
        
        for pattern in enumeration_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        subtasks.append(match[1].strip())
                    else:
                        subtasks.append(match.strip())
                break
        
        # If no explicit enumeration, try to infer from content
        if not subtasks:
            # Look for conjunction-separated tasks
            if ' and ' in content.lower():
                parts = re.split(r'\s+and\s+', content, flags=re.IGNORECASE)
                if len(parts) > 1:
                    subtasks = [part.strip() for part in parts]
            
            # Look for common task patterns
            elif any(keyword in content.lower() for keyword in ['implement', 'create', 'design']):
                if 'system' in content.lower() or 'framework' in content.lower():
                    subtasks = [
                        f"Design {content.lower().split()[-1]} architecture",
                        f"Implement core {content.lower().split()[-1]} functionality",
                        f"Test and validate {content.lower().split()[-1]}"
                    ]
        
        return subtasks[:5]  # Limit to 5 subtasks
    
    def _should_adjust_priority(self, todo: Dict[str, Any], metrics: TodoQualityMetrics) -> bool:
        """Determine if priority should be adjusted"""
        content = todo.get('content', '').lower()
        current_priority = todo.get('priority', 'medium')
        
        # High priority indicators
        high_priority_keywords = ['urgent', 'critical', 'blocker', 'asap', 'deadline', 'emergency']
        has_high_priority_indicators = any(keyword in content for keyword in high_priority_keywords)
        
        # Low priority indicators
        low_priority_keywords = ['nice to have', 'optional', 'eventually', 'when time permits', 'future']
        has_low_priority_indicators = any(keyword in content for keyword in low_priority_keywords)
        
        # Suggest adjustment if current priority doesn't match indicators
        if has_high_priority_indicators and current_priority != 'high':
            return True
        if has_low_priority_indicators and current_priority != 'low':
            return True
        
        return False
    
    def _suggest_priority_adjustment(self, todo: Dict[str, Any], metrics: TodoQualityMetrics) -> Optional[EnhancementSuggestion]:
        """Suggest priority adjustment for a todo"""
        content = todo.get('content', '').lower()
        current_priority = todo.get('priority', 'medium')
        
        # Determine suggested priority
        suggested_priority = self._determine_priority(content)
        
        if suggested_priority != current_priority:
            return EnhancementSuggestion(
                id=str(uuid.uuid4()),
                type=EnhancementType.PRIORITY_ADJUSTMENT,
                todo_id=todo.get('id', ''),
                description=f"Adjust priority from {current_priority} to {suggested_priority}",
                confidence=0.6,
                suggested_change={"priority": suggested_priority},
                reasoning=f"Content analysis suggests {suggested_priority} priority based on keywords and context"
            )
        return None
    
    def _determine_priority(self, content: str) -> str:
        """Determine appropriate priority based on content analysis"""
        high_priority_score = 0
        low_priority_score = 0
        
        # High priority indicators
        high_indicators = ['urgent', 'critical', 'blocker', 'asap', 'deadline', 'emergency', 'fix bug', 'production']
        high_priority_score = sum(1 for indicator in high_indicators if indicator in content.lower())
        
        # Low priority indicators
        low_indicators = ['nice to have', 'optional', 'eventually', 'when time permits', 'future', 'enhancement']
        low_priority_score = sum(1 for indicator in low_indicators if indicator in content.lower())
        
        if high_priority_score > low_priority_score and high_priority_score > 0:
            return 'high'
        elif low_priority_score > high_priority_score and low_priority_score > 0:
            return 'low'
        else:
            return 'medium'
    
    def _suggest_context_enrichment(self, todo: Dict[str, Any], metrics: TodoQualityMetrics) -> Optional[EnhancementSuggestion]:
        """Suggest context enrichment for a todo"""
        content = todo.get('content', '')
        
        # Generate enriched content
        enriched_content = self._enrich_content_context(content)
        
        if enriched_content != content:
            return EnhancementSuggestion(
                id=str(uuid.uuid4()),
                type=EnhancementType.CONTEXT_ENRICHMENT,
                todo_id=todo.get('id', ''),
                description="Add context and specificity to improve clarity",
                confidence=0.6,
                suggested_change={"content": enriched_content},
                reasoning="Added context to make the task more specific and actionable"
            )
        return None
    
    def _enrich_content_context(self, content: str) -> str:
        """Enrich content with additional context"""
        if not content:
            return content
        
        enriched = content
        
        # Add file type context if missing
        if any(word in content.lower() for word in ['function', 'method', 'class']) and not re.search(r'\.\w+', content):
            if 'python' in content.lower() or 'py' in content.lower():
                enriched += " (Python implementation)"
            elif 'javascript' in content.lower() or 'js' in content.lower():
                enriched += " (JavaScript implementation)"
        
        # Add testing context if appropriate
        if 'implement' in content.lower() and 'test' not in content.lower():
            enriched += " with unit tests"
        
        # Add documentation context for significant features
        if any(word in content.lower() for word in ['system', 'framework', 'engine', 'architecture']):
            if 'document' not in content.lower():
                enriched += " and documentation"
        
        return enriched
    
    def _analyze_todo_relationships(self, todos: List[Dict[str, Any]]) -> List[EnhancementSuggestion]:
        """Analyze relationships between todos for enhancement opportunities"""
        suggestions = []
        
        # Detect duplicates
        duplicate_suggestions = self._detect_duplicates(todos)
        suggestions.extend(duplicate_suggestions)
        
        # Detect missing dependencies
        dependency_suggestions = self._detect_missing_dependencies(todos)
        suggestions.extend(dependency_suggestions)
        
        # Detect workflow optimization opportunities
        workflow_suggestions = self._detect_workflow_optimizations(todos)
        suggestions.extend(workflow_suggestions)
        
        return suggestions
    
    def _detect_duplicates(self, todos: List[Dict[str, Any]]) -> List[EnhancementSuggestion]:
        """Detect duplicate or very similar todos"""
        suggestions = []
        
        for i, todo1 in enumerate(todos):
            for j, todo2 in enumerate(todos[i+1:], i+1):
                similarity = self._calculate_content_similarity(
                    todo1.get('content', ''), 
                    todo2.get('content', '')
                )
                
                if similarity > 0.8:  # High similarity threshold
                    suggestions.append(EnhancementSuggestion(
                        id=str(uuid.uuid4()),
                        type=EnhancementType.DUPLICATE_RESOLUTION,
                        todo_id=todo1.get('id', ''),
                        description=f"Potential duplicate of todo {todo2.get('id', '')}",
                        confidence=similarity,
                        suggested_change={"merge_with": todo2.get('id', '')},
                        reasoning=f"Content similarity: {similarity:.2f}"
                    ))
        
        return suggestions
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings"""
        if not content1 or not content2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _detect_missing_dependencies(self, todos: List[Dict[str, Any]]) -> List[EnhancementSuggestion]:
        """Detect missing dependencies between todos"""
        suggestions = []
        
        for i, todo in enumerate(todos):
            content = todo.get('content', '').lower()
            todo_id = todo.get('id', '')
            
            # Look for todos that this one might depend on
            for j, potential_dep in enumerate(todos):
                if i == j:
                    continue
                
                dep_content = potential_dep.get('content', '').lower()
                dep_id = potential_dep.get('id', '')
                
                # Check for dependency patterns
                if self._indicates_dependency(content, dep_content):
                    # Check if dependency already exists
                    existing_deps = todo.get('dependencies', [])
                    if dep_id not in existing_deps:
                        suggestions.append(EnhancementSuggestion(
                            id=str(uuid.uuid4()),
                            type=EnhancementType.DEPENDENCY_ADDITION,
                            todo_id=todo_id,
                            description=f"Add dependency on {dep_id}",
                            confidence=0.7,
                            suggested_change={"add_dependency": dep_id},
                            reasoning="Content analysis suggests this task depends on the completion of another"
                        ))
        
        return suggestions
    
    def _indicates_dependency(self, content: str, potential_dep_content: str) -> bool:
        """Check if content indicates a dependency on potential_dep_content"""
        # Simple dependency detection patterns
        dependency_patterns = [
            # Direct references
            any(word in content for word in potential_dep_content.split() if len(word) > 3),
            # Sequential patterns
            'test' in content and any(word in potential_dep_content for word in ['implement', 'create', 'build']),
            'deploy' in content and 'test' in potential_dep_content,
            'document' in content and any(word in potential_dep_content for word in ['implement', 'create']),
        ]
        
        return any(dependency_patterns)
    
    def _detect_workflow_optimizations(self, todos: List[Dict[str, Any]]) -> List[EnhancementSuggestion]:
        """Detect workflow optimization opportunities"""
        suggestions = []
        
        # Analyze priority distribution
        priorities = [todo.get('priority', 'medium') for todo in todos]
        high_priority_count = priorities.count('high')
        total_count = len(priorities)
        
        # Too many high priority items
        if high_priority_count > total_count * 0.5 and total_count > 3:
            suggestions.append(EnhancementSuggestion(
                id=str(uuid.uuid4()),
                type=EnhancementType.WORKFLOW_OPTIMIZATION,
                todo_id="workflow",
                description="Consider redistributing priorities - too many high priority items",
                confidence=0.8,
                suggested_change={"rebalance_priorities": True},
                reasoning=f"{high_priority_count}/{total_count} items marked as high priority"
            ))
        
        # Detect bottlenecks (todos with many dependencies)
        for todo in todos:
            dependencies = todo.get('dependencies', [])
            if len(dependencies) > 3:
                suggestions.append(EnhancementSuggestion(
                    id=str(uuid.uuid4()),
                    type=EnhancementType.WORKFLOW_OPTIMIZATION,
                    todo_id=todo.get('id', ''),
                    description="Consider breaking down this todo - it has many dependencies",
                    confidence=0.6,
                    suggested_change={"review_dependencies": True},
                    reasoning=f"Todo has {len(dependencies)} dependencies, potentially creating a bottleneck"
                ))
        
        return suggestions
    
    def _filter_and_rank_suggestions(self, suggestions: List[EnhancementSuggestion]) -> List[EnhancementSuggestion]:
        """Filter and rank enhancement suggestions"""
        # Filter by confidence threshold
        threshold = self.config.get('enhancement_threshold', 0.6)
        filtered = [s for s in suggestions if s.confidence >= threshold]
        
        # Rank by confidence and strategy success rate
        def rank_score(suggestion: EnhancementSuggestion) -> float:
            strategy = self.enhancement_strategies.get(suggestion.type, {})
            strategy_weight = strategy.get('weight', 1.0)
            strategy_success = strategy.get('success_rate', 0.5)
            
            return suggestion.confidence * strategy_weight * strategy_success
        
        # Sort by rank score (descending)
        filtered.sort(key=rank_score, reverse=True)
        
        # Group by todo_id and limit per todo
        max_per_todo = self.config.get('max_suggestions_per_todo', 3)
        todo_counts = {}
        final_suggestions = []
        
        for suggestion in filtered:
            todo_id = suggestion.todo_id
            current_count = todo_counts.get(todo_id, 0)
            
            if current_count < max_per_todo:
                final_suggestions.append(suggestion)
                todo_counts[todo_id] = current_count + 1
        
        return final_suggestions
    
    def apply_enhancement(self, suggestion: EnhancementSuggestion, todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply an enhancement suggestion to the todos"""
        updated_todos = todos.copy()
        
        # Find the target todo
        target_todo = None
        target_index = None
        
        for i, todo in enumerate(updated_todos):
            if todo.get('id') == suggestion.todo_id:
                target_todo = todo
                target_index = i
                break
        
        if target_todo is None:
            return updated_todos
        
        # Apply the suggested change
        for key, value in suggestion.suggested_change.items():
            if key == "merge_with":
                # Handle duplicate merging
                updated_todos = self._merge_todos(updated_todos, suggestion.todo_id, value)
            elif key == "add_dependency":
                # Add dependency
                dependencies = target_todo.get('dependencies', [])
                if value not in dependencies:
                    dependencies.append(value)
                    updated_todos[target_index]['dependencies'] = dependencies
            elif key == "subtasks":
                # Add subtasks
                updated_todos[target_index]['subtasks'] = value
            else:
                # Direct field update
                updated_todos[target_index][key] = value
        
        # Mark suggestion as applied
        suggestion.applied = True
        self.enhancement_history.append(suggestion)
        
        return updated_todos
    
    def _merge_todos(self, todos: List[Dict[str, Any]], todo_id1: str, todo_id2: str) -> List[Dict[str, Any]]:
        """Merge two similar todos"""
        todo1 = None
        todo2 = None
        
        for todo in todos:
            if todo.get('id') == todo_id1:
                todo1 = todo
            elif todo.get('id') == todo_id2:
                todo2 = todo
        
        if todo1 is None or todo2 is None:
            return todos
        
        # Create merged todo
        merged_content = f"{todo1.get('content', '')} (merged with: {todo2.get('content', '')})"
        merged_todo = todo1.copy()
        merged_todo['content'] = merged_content
        
        # Merge dependencies
        deps1 = set(todo1.get('dependencies', []))
        deps2 = set(todo2.get('dependencies', []))
        merged_todo['dependencies'] = list(deps1.union(deps2))
        
        # Remove the duplicates and add merged
        updated_todos = [todo for todo in todos if todo.get('id') not in [todo_id1, todo_id2]]
        updated_todos.append(merged_todo)
        
        return updated_todos
    
    def recursive_self_improvement(self) -> Dict[str, Any]:
        """Perform recursive self-improvement of enhancement strategies"""
        if self.recursive_depth >= self.max_recursive_depth:
            return {"status": "max_depth_reached", "depth": self.recursive_depth}
        
        self.recursive_depth += 1
        
        improvement_results = {
            "depth": self.recursive_depth,
            "improvements_made": [],
            "performance_gains": {},
            "strategy_updates": {}
        }
        
        # Analyze historical performance
        performance_analysis = self._analyze_enhancement_performance()
        
        # Update strategy weights based on success rates
        for enhancement_type, metrics in performance_analysis.items():
            if enhancement_type in self.enhancement_strategies:
                old_weight = self.enhancement_strategies[enhancement_type]["weight"]
                old_success = self.enhancement_strategies[enhancement_type]["success_rate"]
                
                # Update success rate with learning
                new_success = metrics.get("success_rate", old_success)
                learning_rate = self.config.get("learning_rate", 0.1)
                updated_success = old_success * (1 - learning_rate) + new_success * learning_rate
                
                # Update weight based on performance
                performance_factor = updated_success / max(old_success, 0.1)
                new_weight = old_weight * (0.9 + 0.2 * performance_factor)  # Bounded adjustment
                
                self.enhancement_strategies[enhancement_type]["success_rate"] = updated_success
                self.enhancement_strategies[enhancement_type]["weight"] = new_weight
                
                improvement_results["strategy_updates"][enhancement_type.value] = {
                    "old_weight": old_weight,
                    "new_weight": new_weight,
                    "old_success_rate": old_success,
                    "new_success_rate": updated_success
                }
        
        # Learn new patterns from successful enhancements
        pattern_learning_results = self._learn_enhancement_patterns()
        improvement_results["improvements_made"].extend(pattern_learning_results)
        
        # Evaluate overall improvement
        current_performance = self._calculate_overall_performance()
        improvement_results["performance_gains"]["overall_score"] = current_performance
        
        # Recursive call for further improvement (if beneficial)
        if current_performance > 0.8 and self.recursive_depth < self.max_recursive_depth:
            recursive_results = self.recursive_self_improvement()
            improvement_results["recursive_results"] = recursive_results
        
        self.recursive_depth -= 1
        return improvement_results
    
    def _analyze_enhancement_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance of different enhancement types"""
        performance = {}
        
        for enhancement_type in EnhancementType:
            type_suggestions = [s for s in self.enhancement_history if s.type == enhancement_type]
            
            if type_suggestions:
                applied_count = sum(1 for s in type_suggestions if s.applied)
                total_count = len(type_suggestions)
                success_rate = applied_count / total_count if total_count > 0 else 0
                
                # Calculate user feedback score
                feedback_scores = []
                for s in type_suggestions:
                    if s.user_feedback:
                        feedback_scores.append(s.user_feedback.get('rating', 0.5))
                
                avg_feedback = statistics.mean(feedback_scores) if feedback_scores else 0.5
                
                performance[enhancement_type] = {
                    "success_rate": success_rate,
                    "user_satisfaction": avg_feedback,
                    "total_suggestions": total_count,
                    "applied_count": applied_count
                }
        
        return performance
    
    def _learn_enhancement_patterns(self) -> List[str]:
        """Learn new enhancement patterns from successful applications"""
        improvements = []
        
        # Analyze successful clarity improvements
        successful_clarity = [s for s in self.enhancement_history 
                            if s.type == EnhancementType.CLARITY_IMPROVEMENT and s.applied]
        
        if len(successful_clarity) > 3:
            # Extract common patterns
            patterns = self._extract_improvement_patterns(successful_clarity)
            self.enhancement_strategies[EnhancementType.CLARITY_IMPROVEMENT]["patterns"] = patterns
            improvements.append(f"Learned {len(patterns)} new clarity improvement patterns")
        
        # Analyze successful decompositions
        successful_decomp = [s for s in self.enhancement_history 
                           if s.type == EnhancementType.TASK_DECOMPOSITION and s.applied]
        
        if len(successful_decomp) > 2:
            decomp_patterns = self._extract_decomposition_patterns(successful_decomp)
            self.enhancement_strategies[EnhancementType.TASK_DECOMPOSITION]["patterns"] = decomp_patterns
            improvements.append(f"Learned {len(decomp_patterns)} new task decomposition patterns")
        
        return improvements
    
    def _extract_improvement_patterns(self, successful_suggestions: List[EnhancementSuggestion]) -> List[Dict]:
        """Extract patterns from successful enhancement suggestions"""
        patterns = []
        
        for suggestion in successful_suggestions:
            if "content" in suggestion.suggested_change:
                original_reasoning = suggestion.reasoning
                change_type = self._classify_change_type(suggestion.suggested_change["content"])
                
                patterns.append({
                    "change_type": change_type,
                    "reasoning": original_reasoning,
                    "confidence": suggestion.confidence,
                    "feedback_score": suggestion.user_feedback.get('rating', 0.8) if suggestion.user_feedback else 0.8
                })
        
        return patterns
    
    def _classify_change_type(self, new_content: str) -> str:
        """Classify the type of change made to content"""
        if any(word in new_content.lower() for word in ['implement', 'create', 'build']):
            return "action_verb_addition"
        elif any(word in new_content.lower() for word in ['test', 'validate', 'verify']):
            return "validation_addition"
        elif re.search(r'\.\w+', new_content):
            return "technical_specification"
        else:
            return "general_clarification"
    
    def _extract_decomposition_patterns(self, successful_decompositions: List[EnhancementSuggestion]) -> List[Dict]:
        """Extract patterns from successful task decompositions"""
        patterns = []
        
        for suggestion in successful_decompositions:
            if "subtasks" in suggestion.suggested_change:
                subtasks = suggestion.suggested_change["subtasks"]
                pattern = {
                    "subtask_count": len(subtasks),
                    "decomposition_strategy": self._identify_decomposition_strategy(subtasks),
                    "confidence": suggestion.confidence
                }
                patterns.append(pattern)
        
        return patterns
    
    def _identify_decomposition_strategy(self, subtasks: List[str]) -> str:
        """Identify the strategy used for task decomposition"""
        if any('design' in task.lower() for task in subtasks):
            return "design_implement_test"
        elif any('phase' in task.lower() for task in subtasks):
            return "phased_approach"
        elif len(subtasks) <= 3:
            return "simple_breakdown"
        else:
            return "detailed_breakdown"
    
    def _calculate_overall_performance(self) -> float:
        """Calculate overall enhancement engine performance"""
        if not self.enhancement_history:
            return 0.5
        
        # Calculate metrics
        total_suggestions = len(self.enhancement_history)
        applied_suggestions = sum(1 for s in self.enhancement_history if s.applied)
        application_rate = applied_suggestions / total_suggestions if total_suggestions > 0 else 0
        
        # Average confidence of applied suggestions
        applied_confidences = [s.confidence for s in self.enhancement_history if s.applied]
        avg_confidence = statistics.mean(applied_confidences) if applied_confidences else 0.5
        
        # User feedback scores
        feedback_scores = []
        for s in self.enhancement_history:
            if s.user_feedback and s.applied:
                feedback_scores.append(s.user_feedback.get('rating', 0.5))
        
        avg_feedback = statistics.mean(feedback_scores) if feedback_scores else 0.5
        
        # Weighted overall score
        overall_score = (
            0.4 * application_rate +
            0.3 * avg_confidence +
            0.3 * avg_feedback
        )
        
        return overall_score
    
    def save_enhancement_state(self, filepath: str = ".taskmaster/enhancement_state.json") -> None:
        """Save the current enhancement engine state"""
        state = {
            "config": self.config,
            "enhancement_strategies": {k.value: v for k, v in self.enhancement_strategies.items()},
            "enhancement_history": [asdict(s) for s in self.enhancement_history],
            "performance_metrics": self.performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_enhancement_state(self, filepath: str = ".taskmaster/enhancement_state.json") -> bool:
        """Load enhancement engine state from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                self.config.update(state.get("config", {}))
                
                # Load enhancement strategies
                strategies = state.get("enhancement_strategies", {})
                for type_name, strategy_data in strategies.items():
                    try:
                        enhancement_type = EnhancementType(type_name)
                        self.enhancement_strategies[enhancement_type] = strategy_data
                    except ValueError:
                        continue
                
                # Load enhancement history
                history_data = state.get("enhancement_history", [])
                self.enhancement_history = []
                for item in history_data:
                    try:
                        suggestion = EnhancementSuggestion(**item)
                        self.enhancement_history.append(suggestion)
                    except (TypeError, ValueError):
                        continue
                
                self.performance_metrics = state.get("performance_metrics", {})
                return True
            
        except Exception as e:
            print(f"Warning: Could not load enhancement state from {filepath}: {e}")
        
        return False

def main():
    """Main execution function for testing the enhancement engine"""
    print("Initializing Recursive Todo Enhancement Engine...")
    
    enhancer = RecursiveTodoEnhancer()
    
    # Load any existing state
    enhancer.load_enhancement_state()
    
    # Example todos for testing
    sample_todos = [
        {
            "id": "test_1",
            "content": "fix the thing",
            "status": "pending",
            "priority": "medium"
        },
        {
            "id": "test_2", 
            "content": "implement user authentication system with JWT tokens and password hashing",
            "status": "pending",
            "priority": "high"
        },
        {
            "id": "test_3",
            "content": "write tests",
            "status": "pending",
            "priority": "low"
        }
    ]
    
    # Analyze todo quality
    print("\nAnalyzing todo quality...")
    for todo in sample_todos:
        metrics = enhancer.analyze_todo_quality(todo)
        print(f"Todo '{todo['content'][:30]}...': Overall score {metrics.overall_score:.2f}")
    
    # Generate enhancement suggestions
    print("\nGenerating enhancement suggestions...")
    suggestions = enhancer.generate_enhancement_suggestions(sample_todos)
    
    for suggestion in suggestions:
        print(f"\n- {suggestion.type.value}: {suggestion.description}")
        print(f"  Confidence: {suggestion.confidence:.2f}")
        print(f"  Reasoning: {suggestion.reasoning}")
    
    # Perform recursive self-improvement
    print("\nPerforming recursive self-improvement...")
    improvement_results = enhancer.recursive_self_improvement()
    print(f"Improvement results: {improvement_results}")
    
    # Save state
    enhancer.save_enhancement_state()
    print("\nEnhancement engine state saved.")
    
    return enhancer, suggestions

if __name__ == "__main__":
    main()