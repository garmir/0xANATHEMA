#!/usr/bin/env python3
"""
Recursive Todo Processing System
Implements parallel GitHub Actions runners for todo validation and improvement

This system:
1. Extracts todos from all sources (Task Master, code comments, documentation)
2. Creates parallel validation runners for each todo
3. Atomizes GitHub Actions output into improvement prompts
4. Implements recursive improvement cycles
5. Integrates with Task Master workflow
"""

import asyncio
import json
import logging
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TodoItem:
    """Represents a todo item from any source"""
    id: str
    description: str
    source: str
    type: str  # 'task', 'subtask', 'code-todo', 'action-item', 'documentation'
    priority: str = 'medium'
    status: str = 'pending'
    dependencies: List[str] = None
    parent: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ValidationResult:
    """Result of todo validation"""
    todo: TodoItem
    validation_status: str  # 'valid', 'invalid', 'error', 'missing'
    validation_details: Dict[str, Any]
    improvement_suggestions: List[str]
    atomized_prompts: List[str]
    execution_time: float
    timestamp: datetime

@dataclass
class ImprovementCycle:
    """Represents one cycle of recursive improvement"""
    cycle_number: int
    prompts_processed: List[str]
    improvements_made: int
    validation_results: List[ValidationResult]
    convergence_metrics: Dict[str, float]
    next_prompts: List[str]

class TodoExtractor:
    """Extracts todos from various sources"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger("TodoExtractor")
    
    def extract_all_todos(self) -> List[TodoItem]:
        """Extract todos from all available sources"""
        todos = []
        
        # Source 1: TaskMaster tasks
        todos.extend(self._extract_taskmaster_todos())
        
        # Source 2: Code TODOs
        todos.extend(self._extract_code_todos())
        
        # Source 3: Documentation todos
        todos.extend(self._extract_documentation_todos())
        
        # Source 4: Markdown action items
        todos.extend(self._extract_markdown_actions())
        
        # Source 5: GitHub Issues (if available)
        todos.extend(self._extract_github_issues())
        
        self.logger.info(f"Extracted {len(todos)} todos from {len(set(t.source for t in todos))} sources")
        return todos
    
    def _extract_taskmaster_todos(self) -> List[TodoItem]:
        """Extract todos from TaskMaster tasks.json"""
        todos = []
        tasks_file = self.project_root / '.taskmaster' / 'tasks' / 'tasks.json'
        
        if not tasks_file.exists():
            self.logger.warning("TaskMaster tasks.json not found")
            return todos
        
        try:
            with open(tasks_file, 'r') as f:
                data = json.load(f)
            
            if 'master' not in data or 'tasks' not in data['master']:
                return todos
            
            for task in data['master']['tasks']:
                # Main task
                todo = TodoItem(
                    id=f"task-{task['id']}",
                    description=task['title'],
                    source='taskmaster',
                    type='task',
                    priority=task.get('priority', 'medium'),
                    status=task.get('status', 'pending'),
                    dependencies=[f"task-{dep}" for dep in task.get('dependencies', [])],
                    metadata={
                        'details': task.get('details', ''),
                        'test_strategy': task.get('testStrategy', '')
                    }
                )
                todos.append(todo)
                
                # Subtasks
                for subtask in task.get('subtasks', []):
                    subtodo = TodoItem(
                        id=f"subtask-{subtask['id']}",
                        description=subtask['title'],
                        source='taskmaster',
                        type='subtask',
                        priority=subtask.get('priority', 'medium'),
                        status=subtask.get('status', 'pending'),
                        parent=f"task-{task['id']}",
                        metadata={
                            'details': subtask.get('details', ''),
                            'test_strategy': subtask.get('testStrategy', '')
                        }
                    )
                    todos.append(subtodo)
        
        except Exception as e:
            self.logger.error(f"Error extracting TaskMaster todos: {e}")
        
        return todos
    
    def _extract_code_todos(self) -> List[TodoItem]:
        """Extract TODO comments from code files"""
        todos = []
        
        # Common code file extensions
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php'}
        
        for file_path in self.project_root.rglob('*'):
            if file_path.suffix in code_extensions and file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Find TODO comments
                    todo_patterns = [
                        r'#\s*TODO:?\s*(.+)',  # Python style
                        r'//\s*TODO:?\s*(.+)',  # C/JS style
                        r'/\*\s*TODO:?\s*(.+?)\*/',  # Multi-line comments
                        r'<!--\s*TODO:?\s*(.+?)\s*-->',  # HTML comments
                    ]
                    
                    for pattern in todo_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                        for i, match in enumerate(matches):
                            todo = TodoItem(
                                id=f"code-todo-{file_path.stem}-{i}",
                                description=match.group(1).strip()[:200],  # Limit length
                                source=str(file_path.relative_to(self.project_root)),
                                type='code-todo',
                                priority='low',
                                metadata={'line_context': match.group(0)}
                            )
                            todos.append(todo)
                
                except (UnicodeDecodeError, PermissionError):
                    continue
                except Exception as e:
                    self.logger.debug(f"Error processing {file_path}: {e}")
        
        return todos
    
    def _extract_documentation_todos(self) -> List[TodoItem]:
        """Extract todos from CLAUDE.md and other documentation"""
        todos = []
        
        # Find CLAUDE.md files
        claude_files = list(self.project_root.rglob('CLAUDE.md'))
        
        for file_path in claude_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Extract task-master commands as actionable items
                command_patterns = [
                    r'task-master\s+(\w+)(?:\s+[^\n]*)?',
                    r'```bash\s*\n(task-master[^\n]+)',
                    r'`(task-master[^`]+)`'
                ]
                
                for pattern in command_patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for i, match in enumerate(matches):
                        command = match.group(1) if len(match.groups()) == 1 else match.group(0)
                        todo = TodoItem(
                            id=f"doc-cmd-{file_path.stem}-{i}",
                            description=f"Execute: {command}",
                            source=str(file_path.relative_to(self.project_root)),
                            type='documentation',
                            priority='medium',
                            metadata={'command': command}
                        )
                        todos.append(todo)
            
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
        
        return todos
    
    def _extract_markdown_actions(self) -> List[TodoItem]:
        """Extract action items from markdown files"""
        todos = []
        
        md_files = [f for f in self.project_root.rglob('*.md') if f.name != 'CLAUDE.md']
        
        for file_path in md_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Extract checkbox items and numbered lists
                action_patterns = [
                    r'- \[ \] (.+)',  # Unchecked checkboxes
                    r'\* \[ \] (.+)',  # Alternative checkbox format
                    r'^\d+\.\s+(.+?)(?=\n|$)',  # Numbered lists
                    r'## (.+?)(?=\n#|\n\n|\Z)',  # Headers as action items
                ]
                
                for pattern in action_patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
                    for i, match in enumerate(matches):
                        description = match.group(1).strip()
                        if len(description) > 10:  # Filter out very short items
                            todo = TodoItem(
                                id=f"md-action-{file_path.stem}-{i}",
                                description=description[:150],
                                source=str(file_path.relative_to(self.project_root)),
                                type='action-item',
                                priority='medium'
                            )
                            todos.append(todo)
            
            except Exception as e:
                self.logger.debug(f"Error processing {file_path}: {e}")
        
        return todos
    
    def _extract_github_issues(self) -> List[TodoItem]:
        """Extract todos from GitHub issues (if gh CLI available)"""
        todos = []
        
        try:
            # Check if gh CLI is available
            result = subprocess.run(['gh', 'issue', 'list', '--json', 'number,title,state,labels'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                issues = json.loads(result.stdout)
                for issue in issues:
                    todo = TodoItem(
                        id=f"github-issue-{issue['number']}",
                        description=issue['title'],
                        source='github-issues',
                        type='github-issue',
                        priority='high' if any(label['name'] == 'urgent' for label in issue.get('labels', [])) else 'medium',
                        status='pending' if issue['state'] == 'open' else 'completed',
                        metadata={'issue_number': issue['number'], 'labels': issue.get('labels', [])}
                    )
                    todos.append(todo)
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            self.logger.debug("GitHub CLI not available or not in a git repository")
        except Exception as e:
            self.logger.debug(f"Error extracting GitHub issues: {e}")
        
        return todos


class TodoValidator:
    """Validates todo items and generates improvement suggestions"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger("TodoValidator")
    
    async def validate_todo(self, todo: TodoItem) -> ValidationResult:
        """Validate a single todo item"""
        start_time = time.time()
        
        validation_details = {}
        improvement_suggestions = []
        atomized_prompts = []
        
        try:
            if todo.type == 'task' or todo.type == 'subtask':
                status = await self._validate_taskmaster_todo(todo, validation_details, improvement_suggestions)
            elif todo.type == 'code-todo':
                status = await self._validate_code_todo(todo, validation_details, improvement_suggestions)
            elif todo.type == 'documentation':
                status = await self._validate_documentation_todo(todo, validation_details, improvement_suggestions)
            elif todo.type == 'action-item':
                status = await self._validate_action_item(todo, validation_details, improvement_suggestions)
            elif todo.type == 'github-issue':
                status = await self._validate_github_issue(todo, validation_details, improvement_suggestions)
            else:
                status = 'unknown'
                validation_details['error'] = f"Unknown todo type: {todo.type}"
            
            # Generate atomized prompts based on validation results
            atomized_prompts = self._generate_atomized_prompts(todo, status, validation_details)
            
        except Exception as e:
            status = 'error'
            validation_details['error'] = str(e)
            self.logger.error(f"Error validating todo {todo.id}: {e}")
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            todo=todo,
            validation_status=status,
            validation_details=validation_details,
            improvement_suggestions=improvement_suggestions,
            atomized_prompts=atomized_prompts,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
    
    async def _validate_taskmaster_todo(self, todo: TodoItem, details: Dict, suggestions: List[str]) -> str:
        """Validate TaskMaster task/subtask"""
        # Check if TaskMaster is initialized
        taskmaster_dir = self.project_root / '.taskmaster'
        if not taskmaster_dir.exists():
            details['taskmaster_initialized'] = False
            suggestions.append("Initialize TaskMaster: task-master init")
            return 'missing_taskmaster'
        
        details['taskmaster_initialized'] = True
        
        # Check task status and dependencies
        tasks_file = taskmaster_dir / 'tasks' / 'tasks.json'
        if tasks_file.exists():
            try:
                with open(tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                
                # Find the specific task
                task_found = False
                if 'master' in tasks_data and 'tasks' in tasks_data['master']:
                    for task in tasks_data['master']['tasks']:
                        if f"task-{task['id']}" == todo.id:
                            task_found = True
                            details['current_status'] = task.get('status', 'pending')
                            details['has_dependencies'] = len(task.get('dependencies', [])) > 0
                            break
                        
                        # Check subtasks
                        for subtask in task.get('subtasks', []):
                            if f"subtask-{subtask['id']}" == todo.id:
                                task_found = True
                                details['current_status'] = subtask.get('status', 'pending')
                                break
                
                if task_found:
                    if details.get('current_status') == 'done':
                        return 'valid'
                    else:
                        suggestions.append(f"Complete task: task-master set-status --id={todo.id.split('-')[1]} --status=done")
                        return 'incomplete'
                else:
                    details['task_found'] = False
                    suggestions.append(f"Task not found in TaskMaster: {todo.id}")
                    return 'missing'
            
            except Exception as e:
                details['error'] = f"Error reading tasks.json: {e}"
                return 'error'
        
        return 'unknown'
    
    async def _validate_code_todo(self, todo: TodoItem, details: Dict, suggestions: List[str]) -> str:
        """Validate code TODO item"""
        source_file = self.project_root / todo.source
        
        if not source_file.exists():
            details['file_exists'] = False
            suggestions.append(f"Create missing file: {todo.source}")
            return 'missing_file'
        
        details['file_exists'] = True
        
        # Check syntax if it's a Python file
        if source_file.suffix == '.py':
            try:
                result = subprocess.run(['python', '-m', 'py_compile', str(source_file)], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    details['syntax_valid'] = True
                    suggestions.append(f"Implement TODO: {todo.description}")
                    return 'valid'
                else:
                    details['syntax_valid'] = False
                    details['syntax_error'] = result.stderr
                    suggestions.append(f"Fix syntax error in {todo.source}")
                    return 'syntax_error'
            except subprocess.TimeoutExpired:
                details['syntax_check'] = 'timeout'
                return 'error'
            except Exception as e:
                details['syntax_check_error'] = str(e)
                return 'error'
        
        # For non-Python files, just check if TODO still exists
        try:
            with open(source_file, 'r') as f:
                content = f.read()
            
            todo_description = todo.description.split('.')[0]  # First sentence
            if todo_description.lower() in content.lower():
                details['todo_still_present'] = True
                suggestions.append(f"Resolve TODO: {todo.description}")
                return 'incomplete'
            else:
                details['todo_still_present'] = False
                return 'valid'  # TODO has been resolved
        
        except Exception as e:
            details['file_read_error'] = str(e)
            return 'error'
    
    async def _validate_documentation_todo(self, todo: TodoItem, details: Dict, suggestions: List[str]) -> str:
        """Validate documentation todo"""
        source_file = self.project_root / todo.source
        
        if not source_file.exists():
            details['file_exists'] = False
            suggestions.append(f"Create documentation file: {todo.source}")
            return 'missing_file'
        
        details['file_exists'] = True
        
        # If it's a command, try to validate the command exists
        if 'command' in todo.metadata:
            command = todo.metadata['command']
            if command.startswith('task-master'):
                # Validate TaskMaster command
                try:
                    result = subprocess.run(['task-master', '--help'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        details['command_available'] = True
                        suggestions.append(f"Execute command: {command}")
                        return 'valid'
                    else:
                        details['command_available'] = False
                        suggestions.append("Install TaskMaster: npm install -g task-master-ai")
                        return 'missing_dependency'
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    details['command_available'] = False
                    suggestions.append("Install TaskMaster: npm install -g task-master-ai")
                    return 'missing_dependency'
        
        return 'valid'
    
    async def _validate_action_item(self, todo: TodoItem, details: Dict, suggestions: List[str]) -> str:
        """Validate markdown action item"""
        source_file = self.project_root / todo.source
        
        if not source_file.exists():
            details['file_exists'] = False
            return 'missing_file'
        
        details['file_exists'] = True
        
        # Check if action item is still unchecked
        try:
            with open(source_file, 'r') as f:
                content = f.read()
            
            # Look for unchecked boxes related to this item
            unchecked_pattern = r'- \[ \] ' + re.escape(todo.description[:50])
            if re.search(unchecked_pattern, content):
                details['action_pending'] = True
                suggestions.append(f"Complete action item: {todo.description}")
                return 'incomplete'
            else:
                details['action_pending'] = False
                return 'valid'  # Item has been checked off
        
        except Exception as e:
            details['file_read_error'] = str(e)
            return 'error'
    
    async def _validate_github_issue(self, todo: TodoItem, details: Dict, suggestions: List[str]) -> str:
        """Validate GitHub issue"""
        if todo.status == 'completed':
            details['issue_closed'] = True
            return 'valid'
        
        details['issue_closed'] = False
        suggestions.append(f"Address GitHub issue #{todo.metadata.get('issue_number', 'unknown')}: {todo.description}")
        return 'incomplete'
    
    def _generate_atomized_prompts(self, todo: TodoItem, status: str, details: Dict) -> List[str]:
        """Generate atomized improvement prompts based on validation results"""
        prompts = []
        
        if status != 'valid':
            # Base prompt
            prompts.append(f"Fix todo item: {todo.description}")
            
            # Type-specific prompts
            if todo.type in ['task', 'subtask']:
                if status == 'missing_taskmaster':
                    prompts.extend([
                        "Initialize TaskMaster project structure",
                        "Set up .taskmaster directory and configuration",
                        "Create initial tasks.json file"
                    ])
                elif status == 'incomplete':
                    prompts.extend([
                        f"Complete TaskMaster task: {todo.description}",
                        f"Validate task dependencies for {todo.id}",
                        f"Update task status to done: {todo.id}"
                    ])
            
            elif todo.type == 'code-todo':
                if status == 'missing_file':
                    prompts.extend([
                        f"Create missing source file: {todo.source}",
                        f"Implement basic structure for {todo.source}",
                        f"Add placeholder implementation for: {todo.description}"
                    ])
                elif status == 'syntax_error':
                    prompts.extend([
                        f"Fix syntax error in {todo.source}",
                        f"Validate Python syntax for {todo.source}",
                        f"Debug compilation error: {details.get('syntax_error', '')[:100]}"
                    ])
                elif status == 'incomplete':
                    prompts.extend([
                        f"Implement TODO in {todo.source}: {todo.description}",
                        f"Replace TODO comment with actual implementation",
                        f"Add tests for implemented functionality in {todo.source}"
                    ])
            
            elif todo.type == 'documentation':
                if status == 'missing_file':
                    prompts.extend([
                        f"Create documentation file: {todo.source}",
                        f"Write comprehensive documentation for: {todo.description}"
                    ])
                elif status == 'missing_dependency':
                    prompts.extend([
                        "Install required dependencies",
                        f"Set up environment for: {todo.metadata.get('command', 'unknown command')}"
                    ])
            
            elif todo.type == 'action-item':
                if status == 'incomplete':
                    prompts.extend([
                        f"Complete action item: {todo.description}",
                        f"Update progress in {todo.source}",
                        f"Mark action item as completed in documentation"
                    ])
            
            elif todo.type == 'github-issue':
                if status == 'incomplete':
                    prompts.extend([
                        f"Address GitHub issue: {todo.description}",
                        f"Implement solution for issue #{todo.metadata.get('issue_number', 'unknown')}",
                        f"Test and validate fix for GitHub issue"
                    ])
        
        return prompts


class RecursiveImprovementEngine:
    """Implements recursive improvement cycles"""
    
    def __init__(self, project_root: Path, max_cycles: int = 3):
        self.project_root = Path(project_root)
        self.max_cycles = max_cycles
        self.logger = logging.getLogger("RecursiveImprovement")
        self.extractor = TodoExtractor(project_root)
        self.validator = TodoValidator(project_root)
    
    async def run_recursive_improvement(self) -> Dict[str, Any]:
        """Run the complete recursive improvement process"""
        self.logger.info("Starting recursive todo improvement process")
        
        # Extract initial todos
        initial_todos = self.extractor.extract_all_todos()
        self.logger.info(f"Found {len(initial_todos)} initial todos")
        
        # Run improvement cycles
        cycles = []
        current_todos = initial_todos
        
        for cycle_num in range(1, self.max_cycles + 1):
            self.logger.info(f"Starting improvement cycle {cycle_num}/{self.max_cycles}")
            
            cycle = await self._run_improvement_cycle(cycle_num, current_todos)
            cycles.append(cycle)
            
            # Check for convergence
            if self._check_convergence(cycle):
                self.logger.info(f"Convergence achieved at cycle {cycle_num}")
                break
            
            # Prepare for next cycle
            current_todos = await self._generate_next_cycle_todos(cycle)
        
        # Generate final report
        final_report = self._generate_final_report(initial_todos, cycles)
        
        return final_report
    
    async def _run_improvement_cycle(self, cycle_num: int, todos: List[TodoItem]) -> ImprovementCycle:
        """Run a single improvement cycle"""
        self.logger.info(f"Cycle {cycle_num}: Processing {len(todos)} todos")
        
        # Validate all todos in parallel
        validation_results = await self._parallel_validate_todos(todos)
        
        # Collect all atomized prompts
        all_prompts = []
        for result in validation_results:
            all_prompts.extend(result.atomized_prompts)
        
        # Execute improvements for this cycle's prompts
        improvements_made = await self._execute_improvements(all_prompts[:10])  # Limit to 10 per cycle
        
        # Calculate convergence metrics
        convergence_metrics = self._calculate_convergence_metrics(validation_results)
        
        # Generate prompts for next cycle
        next_prompts = all_prompts[10:] if len(all_prompts) > 10 else []
        
        return ImprovementCycle(
            cycle_number=cycle_num,
            prompts_processed=all_prompts[:10],
            improvements_made=improvements_made,
            validation_results=validation_results,
            convergence_metrics=convergence_metrics,
            next_prompts=next_prompts
        )
    
    async def _parallel_validate_todos(self, todos: List[TodoItem]) -> List[ValidationResult]:
        """Validate todos in parallel"""
        max_workers = min(10, len(todos))  # Limit concurrent validations
        
        async def validate_batch(todo_batch):
            results = []
            for todo in todo_batch:
                result = await self.validator.validate_todo(todo)
                results.append(result)
            return results
        
        # Split todos into batches
        batch_size = max(1, len(todos) // max_workers)
        batches = [todos[i:i + batch_size] for i in range(0, len(todos), batch_size)]
        
        # Process batches concurrently
        all_results = []
        tasks = [validate_batch(batch) for batch in batches]
        
        for task in asyncio.as_completed(tasks):
            batch_results = await task
            all_results.extend(batch_results)
        
        return all_results
    
    async def _execute_improvements(self, prompts: List[str]) -> int:
        """Execute improvement prompts (simulate for now)"""
        improvements_made = 0
        
        for prompt in prompts:
            self.logger.info(f"Processing improvement prompt: {prompt[:80]}...")
            
            # In a real implementation, this would:
            # 1. Parse the prompt to understand the required action
            # 2. Execute the action (create files, run commands, etc.)
            # 3. Validate the improvement
            
            # For now, simulate improvement based on prompt content
            if any(keyword in prompt.lower() for keyword in ['create', 'implement', 'fix', 'install']):
                # Simulate successful improvement
                await asyncio.sleep(0.1)  # Simulate work
                improvements_made += 1
                self.logger.debug(f"Simulated improvement: {prompt[:50]}...")
        
        return improvements_made
    
    def _calculate_convergence_metrics(self, validation_results: List[ValidationResult]) -> Dict[str, float]:
        """Calculate metrics to determine convergence"""
        if not validation_results:
            return {'success_rate': 0.0, 'error_rate': 0.0, 'improvement_potential': 1.0}
        
        total_todos = len(validation_results)
        valid_todos = len([r for r in validation_results if r.validation_status == 'valid'])
        error_todos = len([r for r in validation_results if r.validation_status == 'error'])
        
        success_rate = valid_todos / total_todos
        error_rate = error_todos / total_todos
        improvement_potential = 1.0 - success_rate
        
        return {
            'success_rate': success_rate,
            'error_rate': error_rate,
            'improvement_potential': improvement_potential,
            'total_todos': total_todos,
            'valid_todos': valid_todos
        }
    
    def _check_convergence(self, cycle: ImprovementCycle) -> bool:
        """Check if convergence has been achieved"""
        metrics = cycle.convergence_metrics
        
        # Convergence criteria
        high_success_rate = metrics.get('success_rate', 0) >= 0.9
        low_improvement_potential = metrics.get('improvement_potential', 1) <= 0.1
        no_more_prompts = len(cycle.next_prompts) == 0
        
        return high_success_rate or low_improvement_potential or no_more_prompts
    
    async def _generate_next_cycle_todos(self, cycle: ImprovementCycle) -> List[TodoItem]:
        """Generate todos for the next cycle based on remaining prompts"""
        next_todos = []
        
        for i, prompt in enumerate(cycle.next_prompts):
            todo = TodoItem(
                id=f"generated-cycle-{cycle.cycle_number + 1}-{i}",
                description=prompt,
                source='recursive_improvement',
                type='generated',
                priority='high',
                metadata={'generated_from_cycle': cycle.cycle_number}
            )
            next_todos.append(todo)
        
        return next_todos
    
    def _generate_final_report(self, initial_todos: List[TodoItem], cycles: List[ImprovementCycle]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        total_improvements = sum(cycle.improvements_made for cycle in cycles)
        final_metrics = cycles[-1].convergence_metrics if cycles else {}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'initial_todos': len(initial_todos),
                'cycles_completed': len(cycles),
                'total_improvements_made': total_improvements,
                'final_success_rate': final_metrics.get('success_rate', 0),
                'convergence_achieved': len(cycles) < self.max_cycles or self._check_convergence(cycles[-1]) if cycles else False
            },
            'cycle_details': [asdict(cycle) for cycle in cycles],
            'recommendations': self._generate_recommendations(cycles),
            'next_steps': self._generate_next_steps(cycles)
        }
        
        return report
    
    def _generate_recommendations(self, cycles: List[ImprovementCycle]) -> List[str]:
        """Generate recommendations based on cycle results"""
        recommendations = []
        
        if not cycles:
            return ["Run initial todo extraction and validation"]
        
        final_cycle = cycles[-1]
        
        if final_cycle.convergence_metrics.get('success_rate', 0) < 0.5:
            recommendations.append("Focus on resolving high-priority validation failures")
        
        if final_cycle.convergence_metrics.get('error_rate', 0) > 0.2:
            recommendations.append("Investigate and fix recurring validation errors")
        
        if len(final_cycle.next_prompts) > 20:
            recommendations.append("Consider increasing max_cycles or batch size for better coverage")
        
        recommendations.extend([
            "Integrate with CI/CD pipeline for continuous validation",
            "Set up automated improvement execution",
            "Add more sophisticated prompt generation logic"
        ])
        
        return recommendations
    
    def _generate_next_steps(self, cycles: List[ImprovementCycle]) -> List[str]:
        """Generate next steps based on results"""
        if not cycles:
            return ["Initialize recursive improvement process"]
        
        final_cycle = cycles[-1]
        
        next_steps = []
        
        if len(final_cycle.next_prompts) > 0:
            next_steps.append(f"Process remaining {len(final_cycle.next_prompts)} improvement prompts")
        
        if final_cycle.convergence_metrics.get('success_rate', 0) < 1.0:
            next_steps.append("Address remaining validation failures")
        
        next_steps.extend([
            "Integrate with GitHub Actions for automated execution",
            "Set up monitoring for continuous improvement",
            "Expand todo extraction to additional sources"
        ])
        
        return next_steps


async def main():
    """Main entry point for the recursive todo processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Recursive Todo Processing System")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--max-cycles", type=int, default=3, help="Maximum improvement cycles")
    parser.add_argument("--output", type=str, default="recursive-improvement-results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Initialize the improvement engine
    engine = RecursiveImprovementEngine(
        project_root=Path(args.project_root),
        max_cycles=args.max_cycles
    )
    
    # Run the recursive improvement process
    try:
        results = await engine.run_recursive_improvement()
        
        # Save results
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Recursive improvement completed successfully!")
        print(f"Results saved to: {output_path}")
        print(f"Cycles completed: {results['summary']['cycles_completed']}")
        print(f"Total improvements: {results['summary']['total_improvements_made']}")
        print(f"Final success rate: {results['summary']['final_success_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"Recursive improvement failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())