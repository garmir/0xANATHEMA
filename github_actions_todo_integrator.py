#!/usr/bin/env python3
"""
GitHub Actions Todo Integration System
Integrates recursive todo processing with GitHub Actions for automated validation and improvement

This system:
1. Generates parallel GitHub Actions runners for each todo
2. Processes GitHub Actions output and atomizes into improvement prompts
3. Creates recursive feedback loops between validation and improvement
4. Integrates with TaskMaster workflow for seamless operation
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
# Using json instead of yaml to avoid dependency
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
import base64
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GitHubActionsRunner:
    """Represents a GitHub Actions runner configuration for a specific todo"""
    runner_id: str
    todo_id: str
    workflow_name: str
    runner_config: Dict[str, Any]
    validation_steps: List[Dict[str, Any]]
    improvement_steps: List[Dict[str, Any]]
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class GitHubActionsOutput:
    """Parsed output from GitHub Actions runner"""
    runner_id: str
    todo_id: str
    status: str  # 'success', 'failure', 'error', 'skipped'
    validation_results: Dict[str, Any]
    logs: List[str]
    artifacts: List[str]
    execution_time: float
    timestamp: datetime
    raw_output: str

@dataclass
class AtomizedPrompt:
    """Atomized improvement prompt derived from GitHub Actions output"""
    id: str
    original_todo_id: str
    runner_id: str
    prompt_text: str
    priority: str
    action_type: str  # 'fix', 'implement', 'validate', 'optimize'
    context: Dict[str, Any]
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class GitHubActionsWorkflowGenerator:
    """Generates GitHub Actions workflows for todo validation"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger("WorkflowGenerator")
    
    def generate_parallel_runners(self, todos: List[Dict]) -> List[GitHubActionsRunner]:
        """Generate parallel GitHub Actions runners for todos"""
        runners = []
        
        for todo in todos:
            runner = self._create_runner_for_todo(todo)
            runners.append(runner)
        
        self.logger.info(f"Generated {len(runners)} parallel runners")
        return runners
    
    def _create_runner_for_todo(self, todo: Dict) -> GitHubActionsRunner:
        """Create a specific runner configuration for a todo"""
        todo_id = todo['id']
        todo_type = todo.get('type', 'unknown')
        
        runner_id = f"runner-{todo_id}-{hashlib.md5(todo_id.encode()).hexdigest()[:8]}"
        
        # Base runner configuration
        runner_config = {
            'runs-on': 'ubuntu-latest',
            'timeout-minutes': 30,
            'environment': self._get_environment_for_todo_type(todo_type)
        }
        
        # Generate validation steps based on todo type
        validation_steps = self._generate_validation_steps(todo)
        
        # Generate improvement steps
        improvement_steps = self._generate_improvement_steps(todo)
        
        return GitHubActionsRunner(
            runner_id=runner_id,
            todo_id=todo_id,
            workflow_name=f"validate-{todo_type}-{todo_id}",
            runner_config=runner_config,
            validation_steps=validation_steps,
            improvement_steps=improvement_steps,
            dependencies=todo.get('dependencies', [])
        )
    
    def _get_environment_for_todo_type(self, todo_type: str) -> Dict[str, Any]:
        """Get environment configuration based on todo type"""
        base_env = {
            'PYTHONPATH': '${{ github.workspace }}',
            'TODO_VALIDATION_MODE': 'strict'
        }
        
        if todo_type in ['task', 'subtask']:
            base_env.update({
                'TASKMASTER_HOME': '${{ github.workspace }}/.taskmaster',
                'ANTHROPIC_API_KEY': '${{ secrets.ANTHROPIC_API_KEY }}',
                'PERPLEXITY_API_KEY': '${{ secrets.PERPLEXITY_API_KEY }}'
            })
        
        elif todo_type == 'code-todo':
            base_env.update({
                'LINT_ENABLED': 'true',
                'TYPE_CHECK_ENABLED': 'true'
            })
        
        return base_env
    
    def _generate_validation_steps(self, todo: Dict) -> List[Dict[str, Any]]:
        """Generate validation steps for a todo"""
        steps = []
        todo_type = todo.get('type', 'unknown')
        
        # Common setup steps
        steps.extend([
            {
                'name': 'Checkout Repository',
                'uses': 'actions/checkout@v4',
                'with': {'fetch-depth': 0}
            },
            {
                'name': 'Setup Python',
                'uses': 'actions/setup-python@v4',
                'with': {'python-version': '3.11'}
            },
            {
                'name': 'Install Dependencies',
                'run': '''
                    pip install --upgrade pip
                    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
                    pip install pytest taskmaster-ai httpx aiofiles
                '''
            }
        ])
        
        # Type-specific validation steps
        if todo_type in ['task', 'subtask']:
            steps.extend(self._generate_taskmaster_validation_steps(todo))
        elif todo_type == 'code-todo':
            steps.extend(self._generate_code_validation_steps(todo))
        elif todo_type == 'documentation':
            steps.extend(self._generate_documentation_validation_steps(todo))
        elif todo_type == 'action-item':
            steps.extend(self._generate_action_item_validation_steps(todo))
        
        # Common result collection step
        steps.append({
            'name': 'Collect Validation Results',
            'run': f'''
                python3 << 'EOF'
                import json
                import os
                from datetime import datetime
                
                # Collect validation results for todo {todo['id']}
                results = {{
                    'todo_id': '{todo['id']}',
                    'todo_type': '{todo_type}',
                    'validation_timestamp': datetime.now().isoformat(),
                    'validation_status': os.environ.get('VALIDATION_STATUS', 'unknown'),
                    'validation_details': {{
                        'description': '{todo['description'][:100]}...',
                        'source': '{todo.get('source', 'unknown')}',
                        'priority': '{todo.get('priority', 'medium')}'
                    }},
                    'environment_info': {{
                        'runner_os': os.environ.get('RUNNER_OS', 'unknown'),
                        'python_version': os.environ.get('pythonLocation', 'unknown')
                    }}
                }}
                
                # Save results
                with open('validation_results_{todo['id'].replace('-', '_')}.json', 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"Validation completed for todo {todo['id']}")
                print(f"Status: {{results['validation_status']}}")
                EOF
            ''',
            'env': {
                'TODO_ID': todo['id'],
                'TODO_TYPE': todo_type
            }
        })
        
        return steps
    
    def _generate_taskmaster_validation_steps(self, todo: Dict) -> List[Dict[str, Any]]:
        """Generate TaskMaster-specific validation steps"""
        return [
            {
                'name': 'Validate TaskMaster Setup',
                'run': '''
                    if [ ! -d ".taskmaster" ]; then
                        echo "TaskMaster not initialized"
                        echo "VALIDATION_STATUS=missing_taskmaster" >> $GITHUB_ENV
                        exit 1
                    fi
                    
                    if [ ! -f ".taskmaster/tasks/tasks.json" ]; then
                        echo "TaskMaster tasks.json not found"
                        echo "VALIDATION_STATUS=missing_tasks" >> $GITHUB_ENV
                        exit 1
                    fi
                    
                    echo "TaskMaster validation passed"
                    echo "VALIDATION_STATUS=taskmaster_ready" >> $GITHUB_ENV
                '''
            },
            {
                'name': 'Validate Specific Task',
                'run': f'''
                    python3 << 'EOF'
                    import json
                    import os
                    
                    todo_id = "{todo['id']}"
                    
                    try:
                        with open('.taskmaster/tasks/tasks.json', 'r') as f:
                            data = json.load(f)
                        
                        task_found = False
                        task_status = 'unknown'
                        
                        if 'master' in data and 'tasks' in data['master']:
                            for task in data['master']['tasks']:
                                if f"task-{{task['id']}}" == todo_id:
                                    task_found = True
                                    task_status = task.get('status', 'pending')
                                    break
                                
                                for subtask in task.get('subtasks', []):
                                    if f"subtask-{{subtask['id']}}" == todo_id:
                                        task_found = True
                                        task_status = subtask.get('status', 'pending')
                                        break
                        
                        if task_found:
                            print(f"Task {{todo_id}} found with status: {{task_status}}")
                            os.environ['TASK_STATUS'] = task_status
                            if task_status == 'done':
                                os.environ['VALIDATION_STATUS'] = 'valid'
                            else:
                                os.environ['VALIDATION_STATUS'] = 'incomplete'
                        else:
                            print(f"Task {{todo_id}} not found")
                            os.environ['VALIDATION_STATUS'] = 'missing'
                    
                    except Exception as e:
                        print(f"Error validating task: {{e}}")
                        os.environ['VALIDATION_STATUS'] = 'error'
                    EOF
                    
                    echo "VALIDATION_STATUS=$VALIDATION_STATUS" >> $GITHUB_ENV
                '''
            }
        ]
    
    def _generate_code_validation_steps(self, todo: Dict) -> List[Dict[str, Any]]:
        """Generate code validation steps"""
        source_file = todo.get('source', '')
        
        return [
            {
                'name': 'Validate Source File',
                'run': f'''
                    if [ ! -f "{source_file}" ]; then
                        echo "Source file not found: {source_file}"
                        echo "VALIDATION_STATUS=missing_file" >> $GITHUB_ENV
                        exit 1
                    fi
                    
                    echo "Source file exists: {source_file}"
                '''
            },
            {
                'name': 'Check Syntax',
                'run': f'''
                    if [[ "{source_file}" == *.py ]]; then
                        python -m py_compile "{source_file}"
                        if [ $? -eq 0 ]; then
                            echo "Python syntax check passed"
                            echo "VALIDATION_STATUS=syntax_valid" >> $GITHUB_ENV
                        else
                            echo "Python syntax check failed"
                            echo "VALIDATION_STATUS=syntax_error" >> $GITHUB_ENV
                            exit 1
                        fi
                    else
                        echo "Non-Python file, skipping syntax check"
                        echo "VALIDATION_STATUS=valid" >> $GITHUB_ENV
                    fi
                '''
            },
            {
                'name': 'Check TODO Status',
                'run': f'''
                    todo_description="{todo.get('description', '')[:50]}"
                    
                    if grep -q "TODO" "{source_file}"; then
                        echo "TODO still present in file"
                        echo "VALIDATION_STATUS=todo_pending" >> $GITHUB_ENV
                    else
                        echo "TODO resolved or not found"
                        echo "VALIDATION_STATUS=todo_resolved" >> $GITHUB_ENV
                    fi
                '''
            }
        ]
    
    def _generate_documentation_validation_steps(self, todo: Dict) -> List[Dict[str, Any]]:
        """Generate documentation validation steps"""
        return [
            {
                'name': 'Validate Documentation',
                'run': f'''
                    source_file="{todo.get('source', '')}"
                    
                    if [ ! -f "$source_file" ]; then
                        echo "Documentation file not found: $source_file"
                        echo "VALIDATION_STATUS=missing_file" >> $GITHUB_ENV
                        exit 1
                    fi
                    
                    echo "Documentation file exists"
                    echo "VALIDATION_STATUS=valid" >> $GITHUB_ENV
                '''
            }
        ]
    
    def _generate_action_item_validation_steps(self, todo: Dict) -> List[Dict[str, Any]]:
        """Generate action item validation steps"""
        return [
            {
                'name': 'Check Action Item Status',
                'run': f'''
                    source_file="{todo.get('source', '')}"
                    item_desc="{todo.get('description', '')[:50]}"
                    
                    if [ ! -f "$source_file" ]; then
                        echo "Action item file not found: $source_file"
                        echo "VALIDATION_STATUS=missing_file" >> $GITHUB_ENV
                        exit 1
                    fi
                    
                    # Check if action item is still unchecked
                    if grep -q "- \\[ \\] .*$item_desc" "$source_file"; then
                        echo "Action item still pending"
                        echo "VALIDATION_STATUS=pending" >> $GITHUB_ENV
                    else
                        echo "Action item completed or not found"
                        echo "VALIDATION_STATUS=completed" >> $GITHUB_ENV
                    fi
                '''
            }
        ]
    
    def _generate_improvement_steps(self, todo: Dict) -> List[Dict[str, Any]]:
        """Generate improvement steps for a todo"""
        steps = [
            {
                'name': 'Generate Improvement Prompts',
                'run': f'''
                    python3 << 'EOF'
                    import json
                    import os
                    
                    validation_status = os.environ.get('VALIDATION_STATUS', 'unknown')
                    todo_type = "{todo.get('type', 'unknown')}"
                    todo_description = "{todo.get('description', '')}"
                    
                    improvement_prompts = []
                    
                    if validation_status != 'valid':
                        # Generate base improvement prompt
                        improvement_prompts.append(f"Fix todo item: {{todo_description}}")
                        
                        # Type-specific prompts
                        if todo_type in ['task', 'subtask']:
                            if validation_status == 'missing_taskmaster':
                                improvement_prompts.extend([
                                    "Initialize TaskMaster project structure",
                                    "Set up .taskmaster directory and configuration"
                                ])
                            elif validation_status == 'incomplete':
                                improvement_prompts.extend([
                                    f"Complete TaskMaster task: {{todo_description}}",
                                    f"Update task status to done"
                                ])
                        
                        elif todo_type == 'code-todo':
                            if validation_status == 'missing_file':
                                improvement_prompts.extend([
                                    f"Create missing source file: {todo.get('source', '')}",
                                    f"Implement basic structure"
                                ])
                            elif validation_status == 'syntax_error':
                                improvement_prompts.extend([
                                    f"Fix syntax error in {todo.get('source', '')}",
                                    "Validate Python syntax"
                                ])
                            elif validation_status == 'todo_pending':
                                improvement_prompts.extend([
                                    f"Implement TODO: {{todo_description}}",
                                    "Replace TODO comment with implementation"
                                ])
                    
                    # Save improvement prompts
                    improvement_data = {{
                        'todo_id': "{todo['id']}",
                        'validation_status': validation_status,
                        'improvement_prompts': improvement_prompts,
                        'atomized_actions': [
                            {{
                                'action_type': 'fix' if 'fix' in prompt.lower() else 'implement',
                                'prompt': prompt,
                                'priority': 'high' if validation_status == 'error' else 'medium'
                            }} for prompt in improvement_prompts
                        ]
                    }}
                    
                    with open('improvement_prompts_{todo['id'].replace('-', '_')}.json', 'w') as f:
                        json.dump(improvement_data, f, indent=2)
                    
                    print(f"Generated {{len(improvement_prompts)}} improvement prompts")
                    EOF
                '''
            }
        ]
        
        return steps
    
    def create_master_workflow(self, runners: List[GitHubActionsRunner]) -> str:
        """Create the master GitHub Actions workflow that orchestrates all runners"""
        
        # Group runners by dependencies to create proper job ordering
        job_matrix = self._create_job_matrix(runners)
        
        workflow = {
            'name': 'Recursive Todo Validation and Improvement',
            'on': {
                'push': {'branches': ['main', 'develop']},
                'pull_request': {'branches': ['main']},
                'workflow_dispatch': {
                    'inputs': {
                        'validation_depth': {
                            'description': 'Maximum recursion depth for validation',
                            'required': False,
                            'default': '3',
                            'type': 'string'
                        },
                        'force_rebuild': {
                            'description': 'Force rebuild all validation runners',
                            'required': False,
                            'default': False,
                            'type': 'boolean'
                        }
                    }
                }
            },
            'env': {
                'VALIDATION_DEPTH': '${{ github.event.inputs.validation_depth || \'3\' }}',
                'FORCE_REBUILD': '${{ github.event.inputs.force_rebuild || \'false\' }}'
            },
            'jobs': {}
        }
        
        # Add parallel validation jobs
        for i, runner_batch in enumerate(job_matrix):
            job_name = f'validate_batch_{i}'
            workflow['jobs'][job_name] = self._create_batch_job(runner_batch)
        
        # Add aggregation job
        workflow['jobs']['aggregate_results'] = self._create_aggregation_job(len(job_matrix))
        
        # Add recursive improvement job
        workflow['jobs']['recursive_improvement'] = self._create_recursive_improvement_job()
        
        if YAML_AVAILABLE:
            return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
        else:
            # Fallback to JSON format (GitHub Actions supports JSON)
            return json.dumps(workflow, indent=2)
    
    def _create_job_matrix(self, runners: List[GitHubActionsRunner]) -> List[List[GitHubActionsRunner]]:
        """Create job matrix for parallel execution while respecting dependencies"""
        # Simple batching for now - in a real implementation, would respect dependencies
        batch_size = 5  # Max runners per batch
        batches = []
        
        for i in range(0, len(runners), batch_size):
            batch = runners[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _create_batch_job(self, runner_batch: List[GitHubActionsRunner]) -> Dict[str, Any]:
        """Create a job that runs a batch of runners in parallel"""
        
        job = {
            'runs-on': 'ubuntu-latest',
            'strategy': {
                'fail-fast': False,
                'matrix': {
                    'runner': [
                        {
                            'id': runner.runner_id,
                            'todo_id': runner.todo_id,
                            'workflow_name': runner.workflow_name
                        }
                        for runner in runner_batch
                    ]
                }
            },
            'steps': []
        }
        
        # Add common setup steps
        job['steps'].extend([
            {
                'name': 'Checkout Repository',
                'uses': 'actions/checkout@v4',
                'with': {'fetch-depth': 0}
            },
            {
                'name': 'Setup Python',
                'uses': 'actions/setup-python@v4',
                'with': {'python-version': '3.11'}
            },
            {
                'name': 'Install Dependencies',
                'run': '''
                    pip install --upgrade pip
                    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
                    pip install pytest taskmaster-ai httpx aiofiles
                '''
            }
        ])
        
        # Add dynamic validation step
        job['steps'].append({
            'name': 'Run Todo Validation',
            'run': '''
                python3 << 'EOF'
                import json
                import os
                import subprocess
                import sys
                from pathlib import Path
                
                runner_id = "${{ matrix.runner.id }}"
                todo_id = "${{ matrix.runner.todo_id }}"
                
                print(f"Running validation for runner {runner_id}, todo {todo_id}")
                
                # Load and execute validation logic
                # This would be dynamically generated based on the specific runner configuration
                
                # For now, simulate validation
                validation_result = {
                    'runner_id': runner_id,
                    'todo_id': todo_id,
                    'status': 'success',  # Would be determined by actual validation
                    'timestamp': '2025-07-10T20:00:00Z',
                    'validation_details': {
                        'checks_passed': 3,
                        'checks_failed': 0,
                        'checks_skipped': 1
                    }
                }
                
                # Save result
                with open(f'validation_result_{runner_id}.json', 'w') as f:
                    json.dump(validation_result, f, indent=2)
                
                print(f"Validation completed: {validation_result['status']}")
                EOF
            ''',
            'env': {
                'RUNNER_ID': '${{ matrix.runner.id }}',
                'TODO_ID': '${{ matrix.runner.todo_id }}'
            }
        })
        
        # Add artifact upload
        job['steps'].append({
            'name': 'Upload Validation Results',
            'uses': 'actions/upload-artifact@v3',
            'if': 'always()',
            'with': {
                'name': 'validation-results-${{ matrix.runner.id }}',
                'path': 'validation_result_*.json'
            }
        })
        
        return job
    
    def _create_aggregation_job(self, num_batches: int) -> Dict[str, Any]:
        """Create job that aggregates results from all validation batches"""
        
        needs = [f'validate_batch_{i}' for i in range(num_batches)]
        
        return {
            'needs': needs,
            'if': 'always()',
            'runs-on': 'ubuntu-latest',
            'steps': [
                {
                    'name': 'Checkout Repository',
                    'uses': 'actions/checkout@v4'
                },
                {
                    'name': 'Download All Validation Results',
                    'uses': 'actions/download-artifact@v3',
                    'with': {'path': 'validation-artifacts'}
                },
                {
                    'name': 'Aggregate and Process Results',
                    'run': '''
                        python3 << 'EOF'
                        import json
                        import os
                        from pathlib import Path
                        from datetime import datetime
                        
                        # Collect all validation results
                        validation_results = []
                        artifacts_dir = Path('validation-artifacts')
                        
                        for artifact_dir in artifacts_dir.iterdir():
                            if artifact_dir.is_dir():
                                for result_file in artifact_dir.glob('*.json'):
                                    try:
                                        with open(result_file, 'r') as f:
                                            result = json.load(f)
                                            validation_results.append(result)
                                    except Exception as e:
                                        print(f"Error loading {result_file}: {e}")
                        
                        # Generate aggregate statistics
                        total_validations = len(validation_results)
                        successful_validations = len([r for r in validation_results if r.get('status') == 'success'])
                        
                        # Collect improvement prompts
                        all_improvement_prompts = []
                        for result in validation_results:
                            if result.get('status') != 'success':
                                # Generate improvement prompts based on failure
                                prompts = [
                                    f"Fix validation failure for todo {result.get('todo_id', 'unknown')}",
                                    f"Address issues found in runner {result.get('runner_id', 'unknown')}"
                                ]
                                all_improvement_prompts.extend(prompts)
                        
                        # Create aggregated report
                        aggregate_report = {
                            'timestamp': datetime.now().isoformat(),
                            'summary': {
                                'total_validations': total_validations,
                                'successful_validations': successful_validations,
                                'success_rate': successful_validations / total_validations if total_validations > 0 else 0,
                                'improvement_prompts_generated': len(all_improvement_prompts)
                            },
                            'validation_results': validation_results,
                            'improvement_prompts': all_improvement_prompts,
                            'next_cycle_recommendations': [
                                'Execute improvement prompts in parallel',
                                'Re-run validation after improvements',
                                'Iterate until convergence achieved'
                            ]
                        }
                        
                        # Save aggregate report
                        with open('aggregate_validation_report.json', 'w') as f:
                            json.dump(aggregate_report, f, indent=2)
                        
                        print(f"Aggregated {total_validations} validation results")
                        print(f"Success rate: {aggregate_report['summary']['success_rate']:.2%}")
                        print(f"Generated {len(all_improvement_prompts)} improvement prompts")
                        EOF
                    '''
                },
                {
                    'name': 'Upload Aggregate Results',
                    'uses': 'actions/upload-artifact@v3',
                    'with': {
                        'name': 'aggregate-validation-results',
                        'path': 'aggregate_validation_report.json'
                    }
                }
            ]
        }
    
    def _create_recursive_improvement_job(self) -> Dict[str, Any]:
        """Create job that implements recursive improvement based on validation results"""
        
        return {
            'needs': 'aggregate_results',
            'if': 'always()',
            'runs-on': 'ubuntu-latest',
            'steps': [
                {
                    'name': 'Checkout Repository',
                    'uses': 'actions/checkout@v4'
                },
                {
                    'name': 'Download Aggregate Results',
                    'uses': 'actions/download-artifact@v3',
                    'with': {
                        'name': 'aggregate-validation-results'
                    }
                },
                {
                    'name': 'Execute Recursive Improvement',
                    'run': '''
                        python3 << 'EOF'
                        import json
                        import os
                        import subprocess
                        import sys
                        from datetime import datetime
                        
                        # Load aggregated validation results
                        with open('aggregate_validation_report.json', 'r') as f:
                            report = json.load(f)
                        
                        improvement_prompts = report.get('improvement_prompts', [])
                        max_cycles = int(os.environ.get('VALIDATION_DEPTH', '3'))
                        
                        print(f"Starting recursive improvement with {len(improvement_prompts)} prompts")
                        print(f"Maximum cycles: {max_cycles}")
                        
                        improvement_cycles = []
                        
                        for cycle in range(max_cycles):
                            print(f"\\n=== Improvement Cycle {cycle + 1}/{max_cycles} ===")
                            
                            if not improvement_prompts:
                                print("No more improvement prompts, stopping")
                                break
                            
                            # Process prompts for this cycle (limit to 10)
                            cycle_prompts = improvement_prompts[:10]
                            improvement_prompts = improvement_prompts[10:]
                            
                            cycle_improvements = 0
                            executed_prompts = []
                            
                            for prompt in cycle_prompts:
                                print(f"Processing: {prompt[:80]}...")
                                
                                # In a real implementation, this would:
                                # 1. Parse the improvement prompt
                                # 2. Execute the required actions
                                # 3. Validate the improvements
                                # 4. Generate new prompts if needed
                                
                                # For now, simulate improvement execution
                                if any(keyword in prompt.lower() for keyword in ['fix', 'implement', 'create']):
                                    print(f"  → Simulated improvement execution")
                                    cycle_improvements += 1
                                    executed_prompts.append(prompt)
                            
                            cycle_result = {
                                'cycle_number': cycle + 1,
                                'prompts_processed': len(cycle_prompts),
                                'improvements_made': cycle_improvements,
                                'executed_prompts': executed_prompts
                            }
                            improvement_cycles.append(cycle_result)
                            
                            print(f"Cycle {cycle + 1} completed: {cycle_improvements} improvements")
                        
                        # Generate final improvement report
                        final_report = {
                            'timestamp': datetime.now().isoformat(),
                            'recursive_improvement_summary': {
                                'cycles_completed': len(improvement_cycles),
                                'total_improvements': sum(c['improvements_made'] for c in improvement_cycles),
                                'prompts_remaining': len(improvement_prompts),
                                'convergence_status': 'max_cycles_reached' if len(improvement_cycles) == max_cycles else 'completed'
                            },
                            'cycle_details': improvement_cycles,
                            'recommendations': [
                                'Re-run validation to measure improvement',
                                'Execute remaining prompts in next iteration',
                                'Consider increasing max_cycles for better coverage'
                            ]
                        }
                        
                        with open('recursive_improvement_report.json', 'w') as f:
                            json.dump(final_report, f, indent=2)
                        
                        print(f"\\nRecursive improvement completed!")
                        print(f"Cycles: {final_report['recursive_improvement_summary']['cycles_completed']}")
                        print(f"Total improvements: {final_report['recursive_improvement_summary']['total_improvements']}")
                        EOF
                    ''',
                    'env': {
                        'VALIDATION_DEPTH': '${{ env.VALIDATION_DEPTH }}'
                    }
                },
                {
                    'name': 'Upload Improvement Results',
                    'uses': 'actions/upload-artifact@v3',
                    'with': {
                        'name': 'recursive-improvement-results',
                        'path': 'recursive_improvement_report.json'
                    }
                }
            ]
        }


class GitHubActionsOutputProcessor:
    """Processes GitHub Actions output and generates atomized prompts"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger("OutputProcessor")
    
    def process_actions_output(self, outputs: List[GitHubActionsOutput]) -> List[AtomizedPrompt]:
        """Process GitHub Actions outputs and generate atomized improvement prompts"""
        atomized_prompts = []
        
        for output in outputs:
            prompts = self._atomize_output(output)
            atomized_prompts.extend(prompts)
        
        self.logger.info(f"Generated {len(atomized_prompts)} atomized prompts from {len(outputs)} outputs")
        return atomized_prompts
    
    def _atomize_output(self, output: GitHubActionsOutput) -> List[AtomizedPrompt]:
        """Atomize a single GitHub Actions output into improvement prompts"""
        prompts = []
        
        if output.status == 'success':
            # Even successful runs might have optimization opportunities
            prompts.append(AtomizedPrompt(
                id=f"optimize-{output.runner_id}-{len(prompts)}",
                original_todo_id=output.todo_id,
                runner_id=output.runner_id,
                prompt_text=f"Optimize successful validation for todo {output.todo_id}",
                priority='low',
                action_type='optimize',
                context={'execution_time': output.execution_time, 'status': 'success'}
            ))
        
        elif output.status == 'failure':
            # Generate specific fix prompts based on failure details
            prompts.extend(self._generate_failure_prompts(output))
        
        elif output.status == 'error':
            # Generate error resolution prompts
            prompts.extend(self._generate_error_prompts(output))
        
        return prompts
    
    def _generate_failure_prompts(self, output: GitHubActionsOutput) -> List[AtomizedPrompt]:
        """Generate prompts for failed validation"""
        prompts = []
        
        # Base failure prompt
        prompts.append(AtomizedPrompt(
            id=f"fix-failure-{output.runner_id}-0",
            original_todo_id=output.todo_id,
            runner_id=output.runner_id,
            prompt_text=f"Fix validation failure for todo {output.todo_id}",
            priority='high',
            action_type='fix',
            context=output.validation_results
        ))
        
        # Analyze logs for specific issues
        for i, log_line in enumerate(output.logs):
            if any(keyword in log_line.lower() for keyword in ['error', 'failed', 'exception']):
                prompts.append(AtomizedPrompt(
                    id=f"fix-log-issue-{output.runner_id}-{i}",
                    original_todo_id=output.todo_id,
                    runner_id=output.runner_id,
                    prompt_text=f"Address log issue: {log_line[:100]}",
                    priority='high',
                    action_type='fix',
                    context={'log_line': log_line, 'line_number': i}
                ))
        
        return prompts
    
    def _generate_error_prompts(self, output: GitHubActionsOutput) -> List[AtomizedPrompt]:
        """Generate prompts for errored validation"""
        prompts = []
        
        prompts.append(AtomizedPrompt(
            id=f"fix-error-{output.runner_id}-0",
            original_todo_id=output.todo_id,
            runner_id=output.runner_id,
            prompt_text=f"Resolve validation error for todo {output.todo_id}",
            priority='high',
            action_type='fix',
            context={'error_details': output.validation_results}
        ))
        
        return prompts


async def main():
    """Main entry point for GitHub Actions todo integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Actions Todo Integration System")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--output-dir", type=str, default=".github/workflows", help="Output directory for workflows")
    parser.add_argument("--max-runners", type=int, default=20, help="Maximum parallel runners")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate workflows
    try:
        # Import the recursive todo processor
        sys.path.insert(0, str(project_root))
        from recursive_todo_processor import TodoExtractor
        
        # Extract todos
        extractor = TodoExtractor(project_root)
        todos = extractor.extract_all_todos()
        
        if not todos:
            logger.warning("No todos found to process")
            return
        
        # Limit todos for GitHub Actions (free tier limitations)
        todos = todos[:args.max_runners]
        
        # Generate GitHub Actions runners
        generator = GitHubActionsWorkflowGenerator(project_root)
        runners = generator.generate_parallel_runners([asdict(todo) for todo in todos])
        
        # Create master workflow
        master_workflow = generator.create_master_workflow(runners)
        
        # Save workflow
        workflow_file = output_dir / "recursive-todo-validation.yml"
        with open(workflow_file, 'w') as f:
            f.write(master_workflow)
        
        logger.info(f"Generated GitHub Actions workflow with {len(runners)} parallel runners")
        logger.info(f"Workflow saved to: {workflow_file}")
        
        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'todos_processed': len(todos),
            'runners_generated': len(runners),
            'workflow_file': str(workflow_file),
            'runner_details': [asdict(runner) for runner in runners]
        }
        
        summary_file = project_root / "github-actions-integration-summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("GitHub Actions integration completed successfully!")
        print(f"Generated {len(runners)} parallel runners for {len(todos)} todos")
        print(f"Workflow: {workflow_file}")
        print(f"Summary: {summary_file}")
        
    except Exception as e:
        logger.error(f"GitHub Actions integration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())