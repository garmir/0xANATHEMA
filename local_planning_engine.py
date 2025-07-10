#!/usr/bin/env python3
"""
Local Planning Engine for Task Master AI
Handles task planning, decomposition, and strategy generation using local LLMs
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from local_llm_adapter import LocalLLMAdapter

@dataclass
class PlanningTask:
    """Represents a planning task with metadata"""
    id: str
    title: str
    description: str
    priority: str = "medium"
    dependencies: List[str] = None
    estimated_complexity: int = 1
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class PlanningResult:
    """Result of planning operation"""
    success: bool
    tasks: List[PlanningTask]
    strategy: str
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

class LocalPlanningEngine:
    """Local planning engine using local LLMs for task planning and strategy generation"""
    
    def __init__(self, adapter: Optional[LocalLLMAdapter] = None):
        self.adapter = adapter or LocalLLMAdapter()
        self.logger = self._setup_logging()
        self.planning_templates_path = ".taskmaster/planning_templates"
        self.planning_cache_path = ".taskmaster/planning_cache"
        self._ensure_directories()
    
    def _setup_logging(self):
        """Setup logging"""
        logger = logging.getLogger("local_planning")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs(self.planning_templates_path, exist_ok=True)
        os.makedirs(self.planning_cache_path, exist_ok=True)
    
    def plan_project(self, project_description: str, context: str = "", max_tasks: int = 20) -> PlanningResult:
        """
        Plan a complete project by breaking it down into manageable tasks
        """
        self.logger.info(f"Planning project: {project_description[:100]}...")
        
        try:
            # Generate planning strategy
            strategy = self._generate_planning_strategy(project_description, context)
            
            # Break down into tasks
            tasks = self._decompose_into_tasks(project_description, strategy, max_tasks)
            
            # Analyze dependencies
            tasks_with_deps = self._analyze_dependencies(tasks)
            
            # Estimate complexity
            tasks_with_complexity = self._estimate_complexity(tasks_with_deps)
            
            return PlanningResult(
                success=True,
                tasks=tasks_with_complexity,
                strategy=strategy,
                metadata={
                    "project_description": project_description,
                    "total_tasks": len(tasks_with_complexity),
                    "timestamp": datetime.now().isoformat(),
                    "planning_method": "local_llm_decomposition"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
            return PlanningResult(
                success=False,
                tasks=[],
                strategy="",
                metadata={},
                error_message=str(e)
            )
    
    def _generate_planning_strategy(self, project_description: str, context: str) -> str:
        """Generate high-level planning strategy"""
        
        strategy_prompt = f"""
        Generate a comprehensive planning strategy for this project:
        
        Project Description: {project_description}
        {f"Additional Context: {context}" if context else ""}
        
        Please provide:
        1. High-level approach and methodology
        2. Key phases and milestones
        3. Critical success factors
        4. Risk assessment and mitigation strategies
        5. Resource requirements and constraints
        6. Quality assurance and validation approach
        
        Focus on creating a clear, actionable strategy that can guide task decomposition.
        """
        
        result = self.adapter.inference(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert project planner and strategist. Provide clear, actionable planning guidance."
                },
                {
                    "role": "user",
                    "content": strategy_prompt
                }
            ],
            role="main",
            max_tokens=2000
        )
        
        if "error" not in result:
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            self.logger.warning(f"Strategy generation failed: {result['error']}")
            return "Basic sequential approach with incremental implementation and testing."
    
    def _decompose_into_tasks(self, project_description: str, strategy: str, max_tasks: int) -> List[PlanningTask]:
        """Decompose project into specific tasks"""
        
        decomposition_prompt = f"""
        Break down this project into specific, actionable tasks:
        
        Project: {project_description}
        Strategy: {strategy}
        
        Requirements:
        1. Create {max_tasks} or fewer specific tasks
        2. Each task should be clearly defined and actionable
        3. Tasks should follow the strategy provided
        4. Include both implementation and validation tasks
        5. Consider setup, development, testing, and deployment phases
        
        Format each task as:
        TASK_ID: unique_identifier
        TITLE: Brief descriptive title
        DESCRIPTION: Detailed description of what needs to be done
        PRIORITY: high/medium/low
        
        Example:
        TASK_ID: setup_environment
        TITLE: Setup Development Environment
        DESCRIPTION: Install required dependencies, configure development tools, and prepare workspace
        PRIORITY: high
        """
        
        result = self.adapter.inference(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at breaking complex projects into manageable tasks. Be specific and actionable."
                },
                {
                    "role": "user",
                    "content": decomposition_prompt
                }
            ],
            role="main",
            max_tokens=3000
        )
        
        if "error" not in result:
            task_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return self._parse_tasks_from_text(task_content)
        else:
            self.logger.warning(f"Task decomposition failed: {result['error']}")
            return self._create_fallback_tasks(project_description)
    
    def _parse_tasks_from_text(self, task_text: str) -> List[PlanningTask]:
        """Parse tasks from LLM-generated text"""
        tasks = []
        current_task = {}
        
        lines = task_text.split('\n')
        task_counter = 1
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('TASK_ID:'):
                # Start new task
                if current_task:
                    task = self._create_task_from_dict(current_task, task_counter)
                    if task:
                        tasks.append(task)
                    task_counter += 1
                
                current_task = {'id': line.split(':', 1)[1].strip()}
                
            elif line.startswith('TITLE:') and current_task:
                current_task['title'] = line.split(':', 1)[1].strip()
                
            elif line.startswith('DESCRIPTION:') and current_task:
                current_task['description'] = line.split(':', 1)[1].strip()
                
            elif line.startswith('PRIORITY:') and current_task:
                current_task['priority'] = line.split(':', 1)[1].strip().lower()
        
        # Add final task
        if current_task:
            task = self._create_task_from_dict(current_task, task_counter)
            if task:
                tasks.append(task)
        
        # If parsing failed, try alternative parsing
        if not tasks:
            tasks = self._alternative_task_parsing(task_text)
        
        return tasks
    
    def _create_task_from_dict(self, task_dict: Dict, counter: int) -> Optional[PlanningTask]:
        """Create PlanningTask from dictionary"""
        try:
            task_id = task_dict.get('id', f'task_{counter}')
            title = task_dict.get('title', f'Task {counter}')
            description = task_dict.get('description', 'No description provided')
            priority = task_dict.get('priority', 'medium')
            
            return PlanningTask(
                id=task_id,
                title=title,
                description=description,
                priority=priority
            )
        except Exception as e:
            self.logger.warning(f"Failed to create task from dict: {e}")
            return None
    
    def _alternative_task_parsing(self, task_text: str) -> List[PlanningTask]:
        """Alternative method to parse tasks from text"""
        tasks = []
        
        # Look for numbered lists or bullet points
        task_patterns = [
            r'^\d+\.\s*(.+)',  # 1. Task description
            r'^[-*]\s*(.+)',   # - Task description or * Task description
            r'^Task\s*\d*:\s*(.+)'  # Task: description
        ]
        
        lines = task_text.split('\n')
        task_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in task_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    task_content = match.group(1).strip()
                    
                    if len(task_content) > 10:  # Reasonable task description length
                        task = PlanningTask(
                            id=f'parsed_task_{task_counter}',
                            title=task_content[:50] + '...' if len(task_content) > 50 else task_content,
                            description=task_content,
                            priority='medium'
                        )
                        tasks.append(task)
                        task_counter += 1
                    break
        
        return tasks
    
    def _create_fallback_tasks(self, project_description: str) -> List[PlanningTask]:
        """Create fallback tasks when parsing fails"""
        
        fallback_tasks = [
            PlanningTask(
                id="analyze_requirements",
                title="Analyze Requirements",
                description=f"Analyze and document requirements for: {project_description}",
                priority="high"
            ),
            PlanningTask(
                id="design_architecture",
                title="Design Architecture",
                description="Design system architecture and technical approach",
                priority="high"
            ),
            PlanningTask(
                id="setup_environment",
                title="Setup Development Environment",
                description="Setup development tools, dependencies, and workspace",
                priority="medium"
            ),
            PlanningTask(
                id="implement_core",
                title="Implement Core Functionality",
                description="Implement the core features and functionality",
                priority="high"
            ),
            PlanningTask(
                id="testing",
                title="Testing and Validation",
                description="Create and execute comprehensive tests",
                priority="medium"
            ),
            PlanningTask(
                id="documentation",
                title="Documentation",
                description="Create user and technical documentation",
                priority="low"
            )
        ]
        
        return fallback_tasks
    
    def _analyze_dependencies(self, tasks: List[PlanningTask]) -> List[PlanningTask]:
        """Analyze and set task dependencies"""
        
        if len(tasks) <= 1:
            return tasks
        
        dependency_prompt = f"""
        Analyze these tasks and identify dependencies between them:
        
        {self._format_tasks_for_analysis(tasks)}
        
        For each task, identify which other tasks it depends on (must be completed before it can start).
        
        Format: TASK_ID: [dependency1, dependency2, ...]
        
        Example:
        implement_core: [setup_environment, design_architecture]
        testing: [implement_core]
        """
        
        result = self.adapter.inference(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing task dependencies and project sequencing."
                },
                {
                    "role": "user",
                    "content": dependency_prompt
                }
            ],
            role="main",
            max_tokens=1500
        )
        
        if "error" not in result:
            dep_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return self._apply_dependencies(tasks, dep_content)
        else:
            self.logger.warning(f"Dependency analysis failed: {result['error']}")
            return self._apply_simple_dependencies(tasks)
    
    def _format_tasks_for_analysis(self, tasks: List[PlanningTask]) -> str:
        """Format tasks for dependency analysis"""
        formatted = []
        for task in tasks:
            formatted.append(f"ID: {task.id}\nTitle: {task.title}\nDescription: {task.description}\n")
        return "\n".join(formatted)
    
    def _apply_dependencies(self, tasks: List[PlanningTask], dep_content: str) -> List[PlanningTask]:
        """Apply dependencies based on LLM analysis"""
        
        # Create task ID to task mapping
        task_map = {task.id: task for task in tasks}
        
        # Parse dependencies
        lines = dep_content.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                task_id = parts[0].strip()
                deps_str = parts[1].strip()
                
                if task_id in task_map:
                    # Parse dependencies
                    deps = []
                    if deps_str and deps_str != '[]':
                        # Remove brackets and split
                        deps_clean = deps_str.strip('[]')
                        if deps_clean:
                            deps = [dep.strip().strip(',').strip('"\'') for dep in deps_clean.split(',')]
                            deps = [dep for dep in deps if dep and dep in task_map]
                    
                    task_map[task_id].dependencies = deps
        
        return list(task_map.values())
    
    def _apply_simple_dependencies(self, tasks: List[PlanningTask]) -> List[PlanningTask]:
        """Apply simple sequential dependencies as fallback"""
        
        for i, task in enumerate(tasks):
            if i > 0:
                # Each task depends on the previous one
                task.dependencies = [tasks[i-1].id]
        
        return tasks
    
    def _estimate_complexity(self, tasks: List[PlanningTask]) -> List[PlanningTask]:
        """Estimate complexity for each task"""
        
        for task in tasks:
            # Simple heuristic based on description length and keywords
            desc_lower = task.description.lower()
            
            complexity = 1  # Base complexity
            
            # Add complexity for certain keywords
            high_complexity_keywords = ['implement', 'develop', 'create', 'build', 'design']
            medium_complexity_keywords = ['configure', 'setup', 'install', 'test']
            
            for keyword in high_complexity_keywords:
                if keyword in desc_lower:
                    complexity += 2
                    break
            
            for keyword in medium_complexity_keywords:
                if keyword in desc_lower:
                    complexity += 1
                    break
            
            # Add complexity based on description length
            if len(task.description) > 100:
                complexity += 1
            
            # Cap complexity at 5
            task.estimated_complexity = min(complexity, 5)
        
        return tasks
    
    def plan_recursive_breakdown(self, task_description: str, max_depth: int = 3, current_depth: int = 0) -> PlanningResult:
        """
        Recursively break down a task into smaller subtasks
        Compatible with Task Master's recursive breakdown methodology
        """
        
        if current_depth >= max_depth:
            self.logger.info(f"Max depth {max_depth} reached for recursive breakdown")
            return PlanningResult(
                success=True,
                tasks=[],
                strategy="Max depth reached",
                metadata={"depth": current_depth, "atomic": True}
            )
        
        # Check if task is already atomic
        atomicity_result = self._check_task_atomicity(task_description)
        
        if atomicity_result['is_atomic']:
            self.logger.info(f"Task is atomic at depth {current_depth}")
            return PlanningResult(
                success=True,
                tasks=[],
                strategy="Task is atomic",
                metadata={"depth": current_depth, "atomic": True}
            )
        
        # Break down into subtasks
        breakdown_prompt = f"""
        Break down this task into 3-7 smaller, more specific subtasks:
        
        Task: {task_description}
        Current Depth: {current_depth}
        Max Depth: {max_depth}
        
        Requirements:
        1. Each subtask should be more specific than the parent task
        2. Subtasks should collectively accomplish the parent task
        3. Aim for 3-7 subtasks (optimal for recursive processing)
        4. Make subtasks as atomic as possible
        
        Format each subtask as:
        SUBTASK: Brief description of the subtask
        """
        
        result = self.adapter.inference(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at recursive task decomposition. Break tasks into optimal subtasks for recursive processing."
                },
                {
                    "role": "user",
                    "content": breakdown_prompt
                }
            ],
            role="main",
            max_tokens=2000
        )
        
        if "error" not in result:
            subtask_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            subtasks = self._parse_recursive_subtasks(subtask_content, current_depth)
            
            return PlanningResult(
                success=True,
                tasks=subtasks,
                strategy=f"Recursive breakdown at depth {current_depth}",
                metadata={
                    "depth": current_depth,
                    "atomic": False,
                    "subtask_count": len(subtasks)
                }
            )
        else:
            self.logger.warning(f"Recursive breakdown failed: {result['error']}")
            return PlanningResult(
                success=False,
                tasks=[],
                strategy="",
                metadata={"depth": current_depth},
                error_message=result['error']
            )
    
    def _check_task_atomicity(self, task_description: str) -> Dict[str, Any]:
        """Check if a task is atomic (cannot be meaningfully broken down further)"""
        
        atomicity_prompt = f"""
        Analyze this task to determine if it is atomic (cannot be meaningfully broken down further):
        
        Task: {task_description}
        
        Consider:
        1. Is this a single, specific action?
        2. Can it be completed in one session/sitting?
        3. Does it require only one type of expertise/skill?
        4. Would breaking it down further create tasks that are too granular to be useful?
        
        Respond with:
        ATOMIC: yes/no
        REASON: Brief explanation of why it is or isn't atomic
        """
        
        result = self.adapter.inference(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at determining task atomicity for optimal project planning."
                },
                {
                    "role": "user",
                    "content": atomicity_prompt
                }
            ],
            role="main",
            max_tokens=500
        )
        
        if "error" not in result:
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Parse atomicity response
            is_atomic = False
            reason = "Could not determine atomicity"
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('ATOMIC:'):
                    atomic_value = line.split(':', 1)[1].strip().lower()
                    is_atomic = atomic_value in ['yes', 'true', '1']
                elif line.startswith('REASON:'):
                    reason = line.split(':', 1)[1].strip()
            
            return {
                "is_atomic": is_atomic,
                "reason": reason,
                "confidence": 0.8  # Default confidence
            }
        else:
            # Fallback: assume not atomic if we can't determine
            return {
                "is_atomic": False,
                "reason": "Could not analyze atomicity",
                "confidence": 0.1
            }
    
    def _parse_recursive_subtasks(self, subtask_content: str, current_depth: int) -> List[PlanningTask]:
        """Parse subtasks from recursive breakdown"""
        subtasks = []
        lines = subtask_content.split('\n')
        subtask_counter = 1
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('SUBTASK:'):
                subtask_desc = line.split(':', 1)[1].strip()
                
                if len(subtask_desc) > 5:  # Reasonable subtask description
                    subtask = PlanningTask(
                        id=f'depth_{current_depth}_subtask_{subtask_counter}',
                        title=subtask_desc[:50] + '...' if len(subtask_desc) > 50 else subtask_desc,
                        description=subtask_desc,
                        priority='medium',
                        estimated_complexity=max(1, 3 - current_depth)  # Complexity decreases with depth
                    )
                    subtasks.append(subtask)
                    subtask_counter += 1
        
        return subtasks

def main():
    """Test the local planning engine"""
    print("📋 Testing Local Planning Engine")
    print("=" * 50)
    
    planning_engine = LocalPlanningEngine()
    
    # Test project planning
    test_project = "Build a web-based task management application with user authentication, real-time updates, and mobile responsive design"
    
    print(f"Planning project: {test_project}")
    result = planning_engine.plan_project(test_project, max_tasks=8)
    
    if result.success:
        print(f"✅ Planning successful! Generated {len(result.tasks)} tasks")
        print(f"Strategy: {result.strategy[:200]}...")
        
        print("\nGenerated Tasks:")
        for i, task in enumerate(result.tasks, 1):
            print(f"{i}. {task.title} (Priority: {task.priority}, Complexity: {task.estimated_complexity})")
            if task.dependencies:
                print(f"   Dependencies: {task.dependencies}")
    else:
        print(f"❌ Planning failed: {result.error_message}")
    
    # Test recursive breakdown
    print("\n" + "=" * 50)
    print("Testing Recursive Breakdown")
    
    test_task = "Implement user authentication system with OAuth integration"
    recursive_result = planning_engine.plan_recursive_breakdown(test_task, max_depth=3)
    
    if recursive_result.success:
        print(f"✅ Recursive breakdown successful! Generated {len(recursive_result.tasks)} subtasks")
        for i, subtask in enumerate(recursive_result.tasks, 1):
            print(f"{i}. {subtask.title}")
    else:
        print(f"❌ Recursive breakdown failed: {recursive_result.error_message}")

if __name__ == "__main__":
    main()