#!/usr/bin/env python3
"""
Recursive PRD Processor with Local LLM Integration
Replaces external API dependencies with local inference for task breakdown
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import logging
import hashlib
import uuid

from .api_abstraction import UnifiedModelAPI, TaskType, ModelConfigFactory

logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Represents a single task in the hierarchy"""
    id: str
    title: str
    description: str
    priority: str = "medium"
    status: str = "pending"
    dependencies: List[str] = field(default_factory=list)
    subtasks: List['Task'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "dependencies": self.dependencies,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary representation"""
        task = cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            priority=data.get("priority", "medium"),
            status=data.get("status", "pending"),
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time())
        )
        
        # Recursively create subtasks
        for subtask_data in data.get("subtasks", []):
            subtask = cls.from_dict(subtask_data)
            task.subtasks.append(subtask)
        
        return task

@dataclass
class DecompositionContext:
    """Context for task decomposition"""
    max_depth: int = 3
    current_depth: int = 0
    parent_task: Optional[Task] = None
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    complexity_threshold: float = 0.7
    min_subtask_count: int = 2
    max_subtask_count: int = 8

class RecursivePRDProcessor:
    """
    Recursive PRD processor that uses local LLMs for task breakdown
    Replaces external API dependencies with local inference capabilities
    """
    
    def __init__(self, 
                 api: UnifiedModelAPI,
                 output_dir: str = ".taskmaster/local_modules/tasks",
                 cache_dir: str = ".taskmaster/local_modules/cache"):
        self.api = api
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Task hierarchy storage
        self.tasks: Dict[str, Task] = {}
        self.task_hierarchy: List[Task] = []
        
        # Processing state
        self.processing_context = DecompositionContext()
        self.knowledge_base = self._load_knowledge_base()
        
        # Performance tracking
        self.performance_metrics = {
            "total_decompositions": 0,
            "successful_decompositions": 0,
            "avg_decomposition_time": 0,
            "cache_hit_rate": 0
        }
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load domain knowledge base for better decomposition"""
        knowledge_file = self.cache_dir / "decomposition_knowledge.json"
        
        if knowledge_file.exists():
            try:
                with open(knowledge_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")
        
        # Default knowledge base
        return {
            "task_patterns": {
                "implementation": {
                    "subtasks": ["design", "develop", "test", "deploy"],
                    "complexity_factors": ["integration", "scalability", "security"]
                },
                "research": {
                    "subtasks": ["literature_review", "analysis", "synthesis", "validation"],
                    "complexity_factors": ["scope", "depth", "novelty"]
                },
                "optimization": {
                    "subtasks": ["profile", "identify_bottlenecks", "implement_fixes", "validate"],
                    "complexity_factors": ["performance_impact", "resource_constraints"]
                }
            },
            "complexity_indicators": [
                "multiple components",
                "external dependencies",
                "performance requirements",
                "security considerations",
                "integration complexity",
                "scalability needs"
            ],
            "atomic_task_indicators": [
                "single action",
                "clear outcome",
                "minimal dependencies",
                "time-bounded",
                "measurable result"
            ]
        }
    
    def _save_knowledge_base(self):
        """Save updated knowledge base"""
        knowledge_file = self.cache_dir / "decomposition_knowledge.json"
        try:
            with open(knowledge_file, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save knowledge base: {e}")
    
    def _generate_task_id(self, parent_id: str = None) -> str:
        """Generate unique task ID"""
        if parent_id:
            # Generate subtask ID (e.g., 1.1, 1.2, 2.1.1)
            parent_task = self.tasks.get(parent_id)
            if parent_task:
                subtask_count = len(parent_task.subtasks)
                return f"{parent_id}.{subtask_count + 1}"
        
        # Generate top-level task ID
        top_level_count = len([t for t in self.tasks.values() if '.' not in t.id])
        return str(top_level_count + 1)
    
    async def _assess_task_complexity(self, task_description: str) -> float:
        """Assess task complexity using local LLM"""
        complexity_prompt = f"""
        Analyze the complexity of this task and provide a complexity score between 0.0 and 1.0:
        
        Task: {task_description}
        
        Consider these factors:
        - Number of components involved
        - Integration complexity
        - Technical difficulty
        - Dependencies
        - Time requirements
        - Risk factors
        
        Complexity indicators:
        {', '.join(self.knowledge_base.get('complexity_indicators', []))}
        
        Provide your assessment in this format:
        COMPLEXITY_SCORE: [0.0-1.0]
        REASONING: [brief explanation]
        IS_ATOMIC: [true/false - whether task cannot be meaningfully decomposed]
        """
        
        try:
            response = await self.api.generate(
                complexity_prompt,
                task_type=TaskType.ANALYSIS,
                temperature=0.3
            )
            
            # Parse response
            content = response.content
            complexity_score = 0.5  # Default
            is_atomic = False
            
            for line in content.split('\n'):
                if line.startswith('COMPLEXITY_SCORE:'):
                    try:
                        complexity_score = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('IS_ATOMIC:'):
                    is_atomic = 'true' in line.lower()
            
            return complexity_score, is_atomic
            
        except Exception as e:
            logger.error(f"Failed to assess complexity: {e}")
            return 0.5, False
    
    async def _decompose_task(self, task: Task, context: DecompositionContext) -> List[Task]:
        """Decompose a task into subtasks using local LLM"""
        if context.current_depth >= context.max_depth:
            logger.info(f"Max depth reached for task {task.id}")
            return []
        
        # Check if task is atomic
        complexity_score, is_atomic = await self._assess_task_complexity(task.description)
        
        if is_atomic or complexity_score < context.complexity_threshold:
            logger.info(f"Task {task.id} is atomic or low complexity ({complexity_score:.2f})")
            return []
        
        # Generate decomposition prompt
        decomposition_prompt = f"""
        Decompose this task into {context.min_subtask_count}-{context.max_subtask_count} clear, actionable subtasks:
        
        PARENT TASK: {task.title}
        DESCRIPTION: {task.description}
        
        Context:
        - Current depth: {context.current_depth}
        - Max depth: {context.max_depth}
        - Domain: {task.metadata.get('domain', 'general')}
        
        Task patterns from knowledge base:
        {json.dumps(self.knowledge_base.get('task_patterns', {}), indent=2)}
        
        Requirements:
        1. Each subtask should be specific and actionable
        2. Subtasks should be logically sequenced
        3. Avoid redundancy between subtasks
        4. Consider dependencies between subtasks
        5. Each subtask should be completable independently
        
        Provide response in this JSON format:
        {{
            "subtasks": [
                {{
                    "title": "Subtask title",
                    "description": "Detailed description",
                    "priority": "high|medium|low",
                    "dependencies": ["id1", "id2"],
                    "domain": "technical|research|planning|implementation",
                    "estimated_effort": "low|medium|high"
                }}
            ],
            "reasoning": "Brief explanation of decomposition strategy"
        }}
        """
        
        try:
            response = await self.api.generate(
                decomposition_prompt,
                task_type=TaskType.PLANNING,
                temperature=0.4
            )
            
            # Parse JSON response
            try:
                decomposition_data = json.loads(response.content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                content = response.content
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    decomposition_data = json.loads(content[start_idx:end_idx])
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Create subtasks
            subtasks = []
            for i, subtask_data in enumerate(decomposition_data.get('subtasks', [])):
                subtask_id = self._generate_task_id(task.id)
                
                subtask = Task(
                    id=subtask_id,
                    title=subtask_data.get('title', f'Subtask {i+1}'),
                    description=subtask_data.get('description', ''),
                    priority=subtask_data.get('priority', 'medium'),
                    dependencies=subtask_data.get('dependencies', []),
                    metadata={
                        'domain': subtask_data.get('domain', 'general'),
                        'estimated_effort': subtask_data.get('estimated_effort', 'medium'),
                        'parent_id': task.id,
                        'depth': context.current_depth + 1,
                        'generated_by': 'recursive_prd_processor',
                        'model_used': response.model_used
                    }
                )
                
                subtasks.append(subtask)
                self.tasks[subtask_id] = subtask
            
            logger.info(f"Generated {len(subtasks)} subtasks for {task.id}")
            return subtasks
            
        except Exception as e:
            logger.error(f"Failed to decompose task {task.id}: {e}")
            return []
    
    async def _recursive_decomposition(self, task: Task, context: DecompositionContext) -> Task:
        """Recursively decompose a task"""
        logger.info(f"Processing task {task.id} at depth {context.current_depth}")
        
        # Update context
        new_context = DecompositionContext(
            max_depth=context.max_depth,
            current_depth=context.current_depth + 1,
            parent_task=task,
            domain_knowledge=context.domain_knowledge,
            complexity_threshold=context.complexity_threshold,
            min_subtask_count=context.min_subtask_count,
            max_subtask_count=context.max_subtask_count
        )
        
        # Decompose current task
        subtasks = await self._decompose_task(task, context)
        task.subtasks = subtasks
        
        # Recursively decompose subtasks
        for subtask in subtasks:
            await self._recursive_decomposition(subtask, new_context)
        
        return task
    
    async def process_prd(self, 
                         prd_content: str,
                         max_depth: int = 3,
                         output_file: str = None) -> Dict[str, Any]:
        """
        Process a PRD document and generate recursive task breakdown
        
        Args:
            prd_content: PRD document content
            max_depth: Maximum recursion depth
            output_file: Optional output file path
            
        Returns:
            Dictionary containing task hierarchy and processing metadata
        """
        start_time = time.time()
        
        # Initialize processing context
        self.processing_context = DecompositionContext(max_depth=max_depth)
        
        # Extract main tasks from PRD
        main_tasks = await self._extract_main_tasks_from_prd(prd_content)
        
        # Process each main task recursively
        processed_tasks = []
        for main_task in main_tasks:
            self.tasks[main_task.id] = main_task
            processed_task = await self._recursive_decomposition(main_task, self.processing_context)
            processed_tasks.append(processed_task)
        
        # Generate processing report
        processing_report = {
            "timestamp": datetime.now().isoformat(),
            "processing_time": time.time() - start_time,
            "total_tasks": len(self.tasks),
            "main_tasks": len(processed_tasks),
            "max_depth_used": max_depth,
            "tasks": [task.to_dict() for task in processed_tasks],
            "performance_metrics": self.performance_metrics
        }
        
        # Save results
        if output_file:
            output_path = self.output_dir / output_file
        else:
            output_path = self.output_dir / f"prd_breakdown_{int(time.time())}.json"
        
        with open(output_path, 'w') as f:
            json.dump(processing_report, f, indent=2)
        
        logger.info(f"PRD processing completed in {time.time() - start_time:.2f}s")
        logger.info(f"Generated {len(self.tasks)} total tasks")
        logger.info(f"Results saved to {output_path}")
        
        return processing_report
    
    async def _extract_main_tasks_from_prd(self, prd_content: str) -> List[Task]:
        """Extract main tasks from PRD content using local LLM"""
        
        extraction_prompt = f"""
        Analyze this PRD document and extract the main high-level tasks/features:
        
        PRD CONTENT:
        {prd_content}
        
        Extract 3-10 main tasks that represent the core deliverables. Each task should be:
        1. A significant feature or capability
        2. Implementable as a standalone unit
        3. Clearly defined with measurable outcomes
        4. Appropriately scoped (not too broad, not too narrow)
        
        Provide response in this JSON format:
        {{
            "main_tasks": [
                {{
                    "title": "Clear, concise task title",
                    "description": "Detailed description of what needs to be accomplished",
                    "priority": "high|medium|low",
                    "domain": "technical|research|planning|implementation|design",
                    "acceptance_criteria": "How to know when this task is complete",
                    "estimated_complexity": "low|medium|high"
                }}
            ],
            "project_context": "Brief summary of the overall project",
            "dependencies": "Any cross-task dependencies identified"
        }}
        """
        
        try:
            response = await self.api.generate(
                extraction_prompt,
                task_type=TaskType.ANALYSIS,
                temperature=0.3
            )
            
            # Parse JSON response
            try:
                extraction_data = json.loads(response.content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                content = response.content
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    extraction_data = json.loads(content[start_idx:end_idx])
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Create main tasks
            main_tasks = []
            for i, task_data in enumerate(extraction_data.get('main_tasks', [])):
                task_id = str(i + 1)
                
                task = Task(
                    id=task_id,
                    title=task_data.get('title', f'Task {i+1}'),
                    description=task_data.get('description', ''),
                    priority=task_data.get('priority', 'medium'),
                    metadata={
                        'domain': task_data.get('domain', 'general'),
                        'acceptance_criteria': task_data.get('acceptance_criteria', ''),
                        'estimated_complexity': task_data.get('estimated_complexity', 'medium'),
                        'extracted_from': 'prd',
                        'model_used': response.model_used,
                        'project_context': extraction_data.get('project_context', ''),
                        'depth': 0
                    }
                )
                
                main_tasks.append(task)
            
            logger.info(f"Extracted {len(main_tasks)} main tasks from PRD")
            return main_tasks
            
        except Exception as e:
            logger.error(f"Failed to extract main tasks from PRD: {e}")
            # Fallback: create a single generic task
            return [Task(
                id="1",
                title="Project Implementation",
                description="Implement the project as described in the PRD",
                priority="high",
                metadata={'domain': 'implementation', 'depth': 0}
            )]
    
    async def expand_task(self, task_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Expand a specific task with additional subtasks"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        current_depth = task.metadata.get('depth', 0)
        
        if current_depth >= max_depth:
            logger.warning(f"Task {task_id} already at maximum depth")
            return {"expanded": False, "reason": "max_depth_reached"}
        
        # Create expansion context
        expansion_context = DecompositionContext(
            max_depth=max_depth,
            current_depth=current_depth,
            complexity_threshold=0.3  # Lower threshold for expansion
        )
        
        # Expand the task
        expanded_task = await self._recursive_decomposition(task, expansion_context)
        
        return {
            "expanded": True,
            "task_id": task_id,
            "new_subtasks": len(expanded_task.subtasks),
            "task_data": expanded_task.to_dict()
        }
    
    def get_task_hierarchy(self) -> Dict[str, Any]:
        """Get complete task hierarchy"""
        top_level_tasks = [task for task in self.tasks.values() if '.' not in task.id]
        
        return {
            "total_tasks": len(self.tasks),
            "top_level_tasks": len(top_level_tasks),
            "hierarchy": [task.to_dict() for task in top_level_tasks],
            "all_tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()}
        }
    
    def export_to_taskmaster_format(self, output_file: str = None) -> str:
        """Export tasks to Task Master AI format"""
        if output_file is None:
            output_file = f"tasks_export_{int(time.time())}.json"
        
        output_path = self.output_dir / output_file
        
        # Convert to Task Master format
        taskmaster_format = {
            "version": "1.0",
            "generated_by": "recursive_prd_processor",
            "timestamp": datetime.now().isoformat(),
            "tasks": []
        }
        
        for task_id, task in self.tasks.items():
            taskmaster_task = {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "status": task.status,
                "priority": task.priority,
                "dependencies": task.dependencies,
                "details": task.metadata.get('acceptance_criteria', ''),
                "testStrategy": f"Validate completion of: {task.title}",
                "subtasks": [subtask.to_dict() for subtask in task.subtasks]
            }
            
            taskmaster_format["tasks"].append(taskmaster_task)
        
        with open(output_path, 'w') as f:
            json.dump(taskmaster_format, f, indent=2)
        
        logger.info(f"Exported {len(self.tasks)} tasks to Task Master format: {output_path}")
        return str(output_path)

# Example usage
if __name__ == "__main__":
    async def test_recursive_prd_processor():
        # Initialize API with local models
        api = UnifiedModelAPI()
        api.add_model("ollama-llama2", ModelConfigFactory.create_ollama_config(
            "llama2", capabilities=[TaskType.ANALYSIS, TaskType.PLANNING]
        ))
        
        # Initialize processor
        processor = RecursivePRDProcessor(api)
        
        # Sample PRD content
        sample_prd = """
        # Task Master AI Local LLM Migration
        
        ## Overview
        Migrate Task Master AI from external AI services to local LLM infrastructure.
        
        ## Key Requirements
        1. Replace Perplexity research with local vector search
        2. Implement local model routing and fallback
        3. Maintain all current functionality
        4. Add performance monitoring
        5. Ensure privacy compliance
        
        ## Success Criteria
        - Zero external API calls for AI processing
        - Performance parity with external services
        - Complete offline operation capability
        - Comprehensive testing and validation
        """
        
        # Process PRD
        result = await processor.process_prd(sample_prd, max_depth=3)
        print(f"Generated {result['total_tasks']} tasks")
        
        # Export to Task Master format
        export_path = processor.export_to_taskmaster_format()
        print(f"Exported to: {export_path}")
        
        # Get hierarchy
        hierarchy = processor.get_task_hierarchy()
        print(f"Task hierarchy: {json.dumps(hierarchy, indent=2)}")
    
    # Run test
    asyncio.run(test_recursive_prd_processor())