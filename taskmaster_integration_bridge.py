#!/usr/bin/env python3
"""
TaskMaster Integration Bridge for Recursive Todo System
Integrates the recursive todo validation and improvement system with TaskMaster workflow

This bridge:
1. Syncs todos between recursive processor and TaskMaster tasks.json
2. Executes TaskMaster commands based on improvement prompts
3. Reports validation results back to TaskMaster
4. Manages bidirectional workflow between systems
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TaskMasterTask:
    """Represents a TaskMaster task"""
    id: str
    title: str
    description: str
    status: str
    priority: str
    dependencies: List[str]
    details: str = ""
    test_strategy: str = ""
    subtasks: List['TaskMasterTask'] = None
    
    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []

@dataclass
class SyncResult:
    """Result of synchronization between systems"""
    todos_synced: int
    tasks_created: int
    tasks_updated: int
    commands_executed: int
    validation_results: Dict[str, Any]
    errors: List[str]

class TaskMasterBridge:
    """Bridge between recursive todo system and TaskMaster"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.taskmaster_dir = self.project_root / '.taskmaster'
        self.tasks_file = self.taskmaster_dir / 'tasks' / 'tasks.json'
        self.logger = logging.getLogger("TaskMasterBridge")
    
    def sync_todos_to_taskmaster(self, todos: List[Dict]) -> SyncResult:
        """Sync todos from recursive processor to TaskMaster"""
        self.logger.info(f"Syncing {len(todos)} todos to TaskMaster")
        
        result = SyncResult(
            todos_synced=0,
            tasks_created=0,
            tasks_updated=0,
            commands_executed=0,
            validation_results={},
            errors=[]
        )
        
        # Ensure TaskMaster is initialized
        if not self._ensure_taskmaster_initialized():
            result.errors.append("Failed to initialize TaskMaster")
            return result
        
        # Load existing tasks
        existing_tasks = self._load_taskmaster_tasks()
        
        # Process todos and sync to TaskMaster
        for todo in todos:
            try:
                sync_success = self._sync_single_todo(todo, existing_tasks)
                if sync_success:
                    result.todos_synced += 1
                    if self._is_new_task(todo, existing_tasks):
                        result.tasks_created += 1
                    else:
                        result.tasks_updated += 1
            except Exception as e:
                error_msg = f"Error syncing todo {todo.get('id', 'unknown')}: {e}"
                self.logger.error(error_msg)
                result.errors.append(error_msg)
        
        # Save updated tasks
        if result.todos_synced > 0:
            self._save_taskmaster_tasks(existing_tasks)
        
        return result
    
    def execute_improvement_prompts(self, prompts: List[str]) -> SyncResult:
        """Execute improvement prompts as TaskMaster commands"""
        self.logger.info(f"Executing {len(prompts)} improvement prompts")
        
        result = SyncResult(
            todos_synced=0,
            tasks_created=0,
            tasks_updated=0,
            commands_executed=0,
            validation_results={},
            errors=[]
        )
        
        for prompt in prompts:
            try:
                commands = self._convert_prompt_to_commands(prompt)
                for command in commands:
                    if self._execute_taskmaster_command(command):
                        result.commands_executed += 1
                    else:
                        result.errors.append(f"Failed to execute: {command}")
            except Exception as e:
                error_msg = f"Error processing prompt '{prompt[:50]}...': {e}"
                self.logger.error(error_msg)
                result.errors.append(error_msg)
        
        return result
    
    def validate_taskmaster_integration(self) -> Dict[str, Any]:
        """Validate the TaskMaster integration"""
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'taskmaster_initialized': False,
            'tasks_file_exists': False,
            'tasks_count': 0,
            'config_valid': False,
            'command_availability': {},
            'integration_health': 'unknown'
        }
        
        try:
            # Check TaskMaster initialization
            validation_results['taskmaster_initialized'] = self.taskmaster_dir.exists()
            validation_results['tasks_file_exists'] = self.tasks_file.exists()
            
            if validation_results['tasks_file_exists']:
                tasks = self._load_taskmaster_tasks()
                validation_results['tasks_count'] = len(tasks.get('master', {}).get('tasks', []))
            
            # Check TaskMaster command availability
            commands_to_test = ['list', 'show', 'next', 'set-status', 'add-task']
            for cmd in commands_to_test:
                validation_results['command_availability'][cmd] = self._test_taskmaster_command(cmd)
            
            # Check config validity
            config_file = self.taskmaster_dir / 'config.json'
            validation_results['config_valid'] = config_file.exists()
            
            # Determine overall health
            if all([
                validation_results['taskmaster_initialized'],
                validation_results['tasks_file_exists'],
                any(validation_results['command_availability'].values())
            ]):
                validation_results['integration_health'] = 'healthy'
            elif validation_results['taskmaster_initialized']:
                validation_results['integration_health'] = 'partial'
            else:
                validation_results['integration_health'] = 'failed'
            
        except Exception as e:
            validation_results['integration_health'] = 'error'
            validation_results['error'] = str(e)
            self.logger.error(f"Validation error: {e}")
        
        return validation_results
    
    def create_recursive_workflow(self) -> Dict[str, Any]:
        """Create a recursive workflow integrating both systems"""
        workflow = {
            'workflow_id': f"recursive-integration-{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'steps': [
                {
                    'step': 1,
                    'name': 'Initialize TaskMaster',
                    'action': 'ensure_taskmaster_initialized',
                    'description': 'Ensure TaskMaster is properly initialized'
                },
                {
                    'step': 2,
                    'name': 'Extract Todos',
                    'action': 'extract_all_todos',
                    'description': 'Extract todos from all sources including TaskMaster'
                },
                {
                    'step': 3,
                    'name': 'Validate Todos',
                    'action': 'parallel_validate_todos',
                    'description': 'Run parallel validation on all todos'
                },
                {
                    'step': 4,
                    'name': 'Generate Improvement Prompts',
                    'action': 'atomize_validation_output',
                    'description': 'Convert validation results to improvement prompts'
                },
                {
                    'step': 5,
                    'name': 'Execute Improvements',
                    'action': 'execute_improvement_prompts',
                    'description': 'Execute improvement prompts via TaskMaster'
                },
                {
                    'step': 6,
                    'name': 'Sync Results',
                    'action': 'sync_results_to_taskmaster',
                    'description': 'Sync all results back to TaskMaster tasks'
                },
                {
                    'step': 7,
                    'name': 'Validate Convergence',
                    'action': 'check_convergence',
                    'description': 'Check if recursive improvement has converged'
                },
                {
                    'step': 8,
                    'name': 'Iterate or Complete',
                    'action': 'iterate_or_complete',
                    'description': 'Either start next cycle or complete workflow'
                }
            ],
            'cycle_config': {
                'max_cycles': 5,
                'convergence_threshold': 0.95,
                'improvement_batch_size': 10,
                'validation_timeout': 300
            }
        }
        
        return workflow
    
    def _ensure_taskmaster_initialized(self) -> bool:
        """Ensure TaskMaster is initialized in the project"""
        if not self.taskmaster_dir.exists():
            self.logger.info("Initializing TaskMaster...")
            try:
                result = subprocess.run(['task-master', 'init'], 
                                      cwd=self.project_root, 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=30)
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.logger.error("Failed to initialize TaskMaster")
                return False
        return True
    
    def _load_taskmaster_tasks(self) -> Dict[str, Any]:
        """Load tasks from TaskMaster tasks.json"""
        if not self.tasks_file.exists():
            return {'master': {'tasks': []}}
        
        try:
            with open(self.tasks_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading tasks.json: {e}")
            return {'master': {'tasks': []}}
    
    def _save_taskmaster_tasks(self, tasks: Dict[str, Any]) -> bool:
        """Save tasks to TaskMaster tasks.json"""
        try:
            # Ensure directory exists
            self.tasks_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.tasks_file, 'w') as f:
                json.dump(tasks, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving tasks.json: {e}")
            return False
    
    def _sync_single_todo(self, todo: Dict, existing_tasks: Dict[str, Any]) -> bool:
        """Sync a single todo to TaskMaster format"""
        try:
            # Convert todo to TaskMaster task format
            task_data = self._convert_todo_to_task(todo)
            
            # Find existing task or create new one
            tasks_list = existing_tasks.setdefault('master', {}).setdefault('tasks', [])
            
            existing_task = None
            for task in tasks_list:
                if task.get('title') == task_data['title'] or \
                   str(task.get('id')) == todo.get('id', '').replace('task-', ''):
                    existing_task = task
                    break
            
            if existing_task:
                # Update existing task
                existing_task.update(task_data)
            else:
                # Add new task
                # Generate new ID
                max_id = max([task.get('id', 0) for task in tasks_list] + [0])
                task_data['id'] = max_id + 1
                tasks_list.append(task_data)
            
            return True
        except Exception as e:
            self.logger.error(f"Error syncing todo: {e}")
            return False
    
    def _convert_todo_to_task(self, todo: Dict) -> Dict[str, Any]:
        """Convert a todo to TaskMaster task format"""
        return {
            'title': todo.get('description', 'Untitled Task')[:100],
            'description': todo.get('description', ''),
            'status': self._map_todo_status(todo.get('status', 'pending')),
            'priority': todo.get('priority', 'medium'),
            'dependencies': [],  # Would need to map dependencies
            'details': f"Converted from todo: {todo.get('source', 'unknown')}",
            'testStrategy': f"Validate completion of: {todo.get('type', 'unknown')} item",
            'subtasks': []
        }
    
    def _map_todo_status(self, todo_status: str) -> str:
        """Map todo status to TaskMaster status"""
        status_mapping = {
            'pending': 'pending',
            'in_progress': 'in-progress',
            'completed': 'done',
            'done': 'done',
            'valid': 'done',
            'cancelled': 'cancelled',
            'error': 'pending'  # Reset errored todos to pending
        }
        return status_mapping.get(todo_status, 'pending')
    
    def _is_new_task(self, todo: Dict, existing_tasks: Dict[str, Any]) -> bool:
        """Check if todo represents a new task"""
        tasks_list = existing_tasks.get('master', {}).get('tasks', [])
        todo_title = todo.get('description', '')[:100]
        
        for task in tasks_list:
            if task.get('title') == todo_title:
                return False
        return True
    
    def _convert_prompt_to_commands(self, prompt: str) -> List[str]:
        """Convert improvement prompt to TaskMaster commands"""
        commands = []
        prompt_lower = prompt.lower()
        
        # Map prompts to TaskMaster commands
        if 'initialize taskmaster' in prompt_lower:
            commands.append('task-master init')
        
        elif 'complete task' in prompt_lower or 'mark as done' in prompt_lower:
            # Extract task ID if possible
            task_id = self._extract_task_id_from_prompt(prompt)
            if task_id:
                commands.append(f'task-master set-status --id={task_id} --status=done')
        
        elif 'create' in prompt_lower and 'file' in prompt_lower:
            # File creation prompts - would need more sophisticated parsing
            commands.append('# File creation command - requires manual implementation')
        
        elif 'fix syntax' in prompt_lower:
            commands.append('# Syntax fix - requires code analysis and repair')
        
        elif 'add task' in prompt_lower:
            # Extract task description
            task_desc = prompt.replace('add task', '').strip()[:100]
            if task_desc:
                commands.append(f'task-master add-task --prompt="{task_desc}"')
        
        # Default: add as new task if no specific command mapped
        if not commands:
            escaped_prompt = prompt.replace('"', '\\"')[:100]
            commands.append(f'task-master add-task --prompt="{escaped_prompt}"')
        
        return commands
    
    def _extract_task_id_from_prompt(self, prompt: str) -> Optional[str]:
        """Extract task ID from improvement prompt"""
        import re
        
        # Look for various task ID patterns
        patterns = [
            r'task[- ](\d+(?:\.\d+)?)',
            r'todo[- ](\d+(?:\.\d+)?)',
            r'id[- ](\d+(?:\.\d+)?)',
            r'#(\d+(?:\.\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                return match.group(1)
        
        return None
    
    def _execute_taskmaster_command(self, command: str) -> bool:
        """Execute a TaskMaster command"""
        if command.startswith('#'):  # Skip comments
            return True
        
        try:
            # Parse command into parts
            parts = command.split()
            if not parts or parts[0] != 'task-master':
                return False
            
            self.logger.info(f"Executing TaskMaster command: {command}")
            result = subprocess.run(parts, 
                                  cwd=self.project_root, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=60)
            
            if result.returncode == 0:
                self.logger.debug(f"Command successful: {command}")
                return True
            else:
                self.logger.warning(f"Command failed: {command} - {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f"Error executing command {command}: {e}")
            return False
    
    def _test_taskmaster_command(self, subcommand: str) -> bool:
        """Test if a TaskMaster subcommand is available"""
        try:
            result = subprocess.run(['task-master', subcommand, '--help'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


async def run_integrated_recursive_workflow(project_root: Path, max_cycles: int = 3) -> Dict[str, Any]:
    """Run the complete integrated recursive workflow"""
    bridge = TaskMasterBridge(project_root)
    logger.info("Starting integrated recursive workflow")
    
    # Validate integration
    validation = bridge.validate_taskmaster_integration()
    if validation['integration_health'] not in ['healthy', 'partial']:
        logger.error("TaskMaster integration is not healthy, aborting workflow")
        return {'error': 'TaskMaster integration failed', 'validation': validation}
    
    # Import recursive processors
    sys.path.insert(0, str(project_root))
    try:
        from recursive_todo_processor import TodoExtractor, RecursiveImprovementEngine
    except ImportError as e:
        logger.error(f"Failed to import recursive processors: {e}")
        return {'error': 'Import failed', 'details': str(e)}
    
    # Create workflow
    workflow = bridge.create_recursive_workflow()
    
    # Initialize components
    extractor = TodoExtractor(project_root)
    improvement_engine = RecursiveImprovementEngine(project_root, max_cycles)
    
    results = {
        'workflow_id': workflow['workflow_id'],
        'start_time': datetime.now().isoformat(),
        'validation': validation,
        'cycles': [],
        'final_sync': {},
        'summary': {}
    }
    
    for cycle in range(max_cycles):
        logger.info(f"Starting integrated cycle {cycle + 1}/{max_cycles}")
        
        cycle_results = {
            'cycle_number': cycle + 1,
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'errors': []
        }
        
        try:
            # Step 1: Extract todos
            todos = extractor.extract_all_todos()
            cycle_results['todos_extracted'] = len(todos)
            cycle_results['steps_completed'].append('extract_todos')
            
            # Step 2: Sync to TaskMaster
            sync_result = bridge.sync_todos_to_taskmaster([asdict(todo) for todo in todos])
            cycle_results['sync_result'] = asdict(sync_result)
            cycle_results['steps_completed'].append('sync_to_taskmaster')
            
            # Step 3: Run validation (simulated)
            validation_results = []
            for todo in todos[:10]:  # Limit for testing
                # Simulate validation result
                validation_results.append({
                    'todo_id': todo.id,
                    'status': 'valid' if 'done' in todo.status else 'incomplete',
                    'improvement_prompts': [f"Complete todo: {todo.description}"] if 'done' not in todo.status else []
                })
            
            cycle_results['validation_results'] = len(validation_results)
            cycle_results['steps_completed'].append('validate_todos')
            
            # Step 4: Execute improvements
            all_prompts = []
            for vr in validation_results:
                all_prompts.extend(vr.get('improvement_prompts', []))
            
            if all_prompts:
                improvement_result = bridge.execute_improvement_prompts(all_prompts[:5])  # Limit for testing
                cycle_results['improvement_result'] = asdict(improvement_result)
                cycle_results['steps_completed'].append('execute_improvements')
            
            # Step 5: Check convergence
            valid_count = len([vr for vr in validation_results if vr['status'] == 'valid'])
            convergence_rate = valid_count / len(validation_results) if validation_results else 0
            cycle_results['convergence_rate'] = convergence_rate
            cycle_results['steps_completed'].append('check_convergence')
            
            if convergence_rate >= 0.9:
                logger.info(f"Convergence achieved at cycle {cycle + 1}")
                cycle_results['converged'] = True
                results['cycles'].append(cycle_results)
                break
            
        except Exception as e:
            error_msg = f"Error in cycle {cycle + 1}: {e}"
            logger.error(error_msg)
            cycle_results['errors'].append(error_msg)
        
        cycle_results['end_time'] = datetime.now().isoformat()
        results['cycles'].append(cycle_results)
    
    # Final sync and summary
    final_todos = extractor.extract_all_todos()
    final_sync = bridge.sync_todos_to_taskmaster([asdict(todo) for todo in final_todos])
    results['final_sync'] = asdict(final_sync)
    
    results['summary'] = {
        'cycles_completed': len(results['cycles']),
        'total_todos_processed': len(final_todos),
        'final_convergence_rate': results['cycles'][-1].get('convergence_rate', 0) if results['cycles'] else 0,
        'integration_successful': validation['integration_health'] == 'healthy'
    }
    
    results['end_time'] = datetime.now().isoformat()
    
    return results


async def main():
    """Main entry point for TaskMaster integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TaskMaster Integration Bridge")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--max-cycles", type=int, default=3, help="Maximum cycles")
    parser.add_argument("--action", type=str, default="workflow", 
                       choices=['validate', 'sync', 'workflow'],
                       help="Action to perform")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root)
    bridge = TaskMasterBridge(project_root)
    
    try:
        if args.action == 'validate':
            results = bridge.validate_taskmaster_integration()
            print(f"TaskMaster Integration Health: {results['integration_health']}")
            
        elif args.action == 'sync':
            # Import and extract todos for sync test
            sys.path.insert(0, str(project_root))
            from recursive_todo_processor import TodoExtractor
            
            extractor = TodoExtractor(project_root)
            todos = extractor.extract_all_todos()
            
            results = bridge.sync_todos_to_taskmaster([asdict(todo) for todo in todos[:10]])  # Limit for testing
            print(f"Synced {results.todos_synced} todos to TaskMaster")
            
        elif args.action == 'workflow':
            results = await run_integrated_recursive_workflow(project_root, args.max_cycles)
            print(f"Workflow completed: {results['summary']['cycles_completed']} cycles")
            print(f"Final convergence rate: {results['summary']['final_convergence_rate']:.2%}")
        
        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"TaskMaster integration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())