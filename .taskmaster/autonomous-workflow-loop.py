#!/usr/bin/env python3

"""
Autonomous Workflow Loop with Research-Driven Problem Solving
Hardcoded workflow: Get stuck → Research with task-master + perplexity → Parse todos to Claude → Execute until success
"""

import json
import logging
import subprocess
import time
import os
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WorkflowState:
    """Current state of the autonomous workflow"""
    current_task_id: Optional[str] = None
    execution_attempts: int = 0
    research_attempts: int = 0
    last_error: Optional[str] = None
    stuck_indicator: bool = False
    success_count: int = 0
    failure_count: int = 0

class AutonomousWorkflowLoop:
    """
    Autonomous workflow loop with hardcoded research-driven problem solving
    
    Core Logic:
    1. Get next task from task-master
    2. Try to execute task
    3. If stuck/failed → Research solution with task-master + perplexity
    4. Parse research results into actionable todos
    5. Execute todos via Claude until success
    6. Repeat until all tasks complete
    """
    
    def __init__(self):
        self.state = WorkflowState()
        self.max_execution_attempts = 3
        self.max_research_attempts = 2
        self.stuck_threshold = 2  # Consider stuck after 2 failed attempts
        self.workflow_log = []
        
    def run_autonomous_loop(self) -> bool:
        """Main autonomous workflow loop"""
        logger.info("🚀 Starting Autonomous Workflow Loop with Research-Driven Problem Solving")
        
        try:
            while True:
                # Step 1: Get next task
                current_task = self._get_next_task()
                if not current_task:
                    logger.info("✅ All tasks completed successfully!")
                    return True
                
                self.state.current_task_id = current_task['id']
                logger.info(f"🎯 Working on task {current_task['id']}: {current_task['title']}")
                
                # Step 2: Execute task with retry logic
                success = self._execute_task_with_retries(current_task)
                
                if success:
                    self._handle_task_success(current_task)
                else:
                    # Step 3: Research-driven problem solving when stuck
                    research_success = self._research_driven_problem_solving(current_task)
                    
                    if not research_success:
                        logger.error(f"❌ Failed to solve task {current_task['id']} even after research")
                        self.state.failure_count += 1
                        # Continue to next task rather than failing completely
                        self._mark_task_failed(current_task)
                
                # Reset state for next task
                self._reset_task_state()
                
        except KeyboardInterrupt:
            logger.info("🛑 Workflow interrupted by user")
            return False
        except Exception as e:
            logger.error(f"💥 Workflow loop crashed: {e}")
            return False
    
    def _get_next_task(self) -> Optional[Dict]:
        """Get next available task from task-master"""
        try:
            result = subprocess.run(
                ['task-master', 'next'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse task from output
                output = result.stdout.strip()
                if "No available tasks" in output or not output:
                    return None
                
                # Extract task ID and get full details
                lines = output.split('\n')
                task_id = None
                for line in lines:
                    if 'Next task:' in line or 'Task ID:' in line:
                        task_id = line.split(':')[-1].strip()
                        break
                
                if task_id:
                    return self._get_task_details(task_id)
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to get next task: {e}")
            return None
    
    def _get_task_details(self, task_id: str) -> Optional[Dict]:
        """Get detailed task information"""
        try:
            result = subprocess.run(
                ['task-master', 'show', task_id],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse task details from output
                output = result.stdout
                # Extract key information
                task_info = {
                    'id': task_id,
                    'title': self._extract_field(output, 'Title'),
                    'description': self._extract_field(output, 'Description'),
                    'details': self._extract_field(output, 'Details'),
                    'status': self._extract_field(output, 'Status'),
                    'priority': self._extract_field(output, 'Priority')
                }
                return task_info
                
        except Exception as e:
            logger.error(f"Failed to get task details for {task_id}: {e}")
            
        return None
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract field value from task-master output"""
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith(f'{field_name}:'):
                return line.split(':', 1)[1].strip()
        return ""
    
    def _execute_task_with_retries(self, task: Dict) -> bool:
        """Execute task with retry logic and stuck detection"""
        
        for attempt in range(1, self.max_execution_attempts + 1):
            logger.info(f"🔄 Execution attempt {attempt}/{self.max_execution_attempts}")
            
            try:
                # Mark task as in-progress
                self._set_task_status(task['id'], 'in-progress')
                
                # Execute the task
                success = self._execute_single_task(task)
                
                if success:
                    self._set_task_status(task['id'], 'done')
                    return True
                else:
                    self.state.execution_attempts = attempt
                    
                    # Check if we're stuck
                    if attempt >= self.stuck_threshold:
                        self.state.stuck_indicator = True
                        logger.warning(f"🚫 Stuck on task {task['id']} after {attempt} attempts")
                        break
                        
            except Exception as e:
                error_msg = f"Execution attempt {attempt} failed: {e}"
                logger.error(error_msg)
                self.state.last_error = error_msg
                
        return False
    
    def _execute_single_task(self, task: Dict) -> bool:
        """Execute a single task using various strategies"""
        task_id = task['id']
        task_description = task.get('description', '')
        task_details = task.get('details', '')
        
        logger.info(f"🔨 Executing task {task_id}")
        
        # Strategy 1: Try direct execution if task has clear implementation details
        if self._has_implementation_details(task):
            success = self._execute_with_details(task)
            if success:
                return True
        
        # Strategy 2: Try automated execution patterns
        success = self._execute_with_patterns(task)
        if success:
            return True
        
        # Strategy 3: Try Claude Code integration for code-related tasks
        if self._is_code_task(task):
            success = self._execute_with_claude_code(task)
            if success:
                return True
        
        # If all strategies fail, we're stuck
        return False
    
    def _research_driven_problem_solving(self, task: Dict) -> bool:
        """Core research-driven problem solving workflow"""
        logger.info(f"🔬 Starting research-driven problem solving for task {task['id']}")
        
        for research_attempt in range(1, self.max_research_attempts + 1):
            logger.info(f"📚 Research attempt {research_attempt}/{self.max_research_attempts}")
            
            # Step 1: Research solution using task-master + perplexity
            research_results = self._research_solution(task)
            
            if not research_results:
                continue
            
            # Step 2: Parse research into actionable todos
            todos = self._parse_research_to_todos(research_results, task)
            
            if not todos:
                continue
            
            # Step 3: Execute todos via Claude until success
            success = self._execute_todos_via_claude(todos, task)
            
            if success:
                logger.info(f"✅ Research-driven solution successful for task {task['id']}")
                return True
                
        logger.error(f"❌ Research-driven problem solving failed for task {task['id']}")
        return False
    
    def _research_solution(self, task: Dict) -> Optional[str]:
        """Research solution using task-master with perplexity integration"""
        logger.info(f"🔍 Researching solution for: {task['title']}")
        
        try:
            # Construct research query
            research_query = self._build_research_query(task)
            
            # Use task-master research command with perplexity
            result = subprocess.run([
                'task-master', 'add-task', 
                '--prompt', f"Research and provide step-by-step solution for: {research_query}",
                '--research'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Get the research task ID from output
                research_task_id = self._extract_new_task_id(result.stdout)
                
                if research_task_id:
                    # Get the research results
                    research_details = self._get_task_details(research_task_id)
                    if research_details:
                        return research_details.get('details', '')
                        
        except Exception as e:
            logger.error(f"Research failed: {e}")
            
        return None
    
    def _build_research_query(self, task: Dict) -> str:
        """Build comprehensive research query"""
        query_parts = [
            f"Task: {task['title']}",
            f"Description: {task.get('description', '')}",
            f"Context: Task-Master autonomous system",
            f"Error: {self.state.last_error}" if self.state.last_error else "",
            "Requirement: Provide step-by-step implementation solution"
        ]
        
        return " | ".join([part for part in query_parts if part])
    
    def _extract_new_task_id(self, output: str) -> Optional[str]:
        """Extract newly created task ID from task-master output"""
        lines = output.split('\n')
        for line in lines:
            if 'Task created' in line or 'Added task' in line:
                # Extract ID from various formats
                words = line.split()
                for word in words:
                    if word.isdigit():
                        return word
        return None
    
    def _parse_research_to_todos(self, research_results: str, original_task: Dict) -> List[Dict]:
        """Parse research results into actionable todo steps"""
        logger.info("📝 Parsing research results into actionable todos")
        
        try:
            # Use task-master to break down research into subtasks
            result = subprocess.run([
                'task-master', 'expand',
                '--id', original_task['id'],
                '--research',
                '--force'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Get expanded subtasks
                updated_task = self._get_task_details(original_task['id'])
                if updated_task:
                    # Extract subtasks (this would need task-master to support subtask listing)
                    return self._extract_subtasks(updated_task)
                    
        except Exception as e:
            logger.error(f"Failed to parse research to todos: {e}")
        
        # Fallback: Create basic todos from research text
        return self._create_basic_todos_from_text(research_results)
    
    def _create_basic_todos_from_text(self, text: str) -> List[Dict]:
        """Create basic todo structure from research text"""
        todos = []
        lines = text.split('\n')
        
        todo_id = 1
        for line in lines:
            line = line.strip()
            # Look for step patterns
            if (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 
                line.startswith(('Step', 'Phase', 'First', 'Then', 'Next', 'Finally'))):
                
                todos.append({
                    'id': f"research-todo-{todo_id}",
                    'description': line,
                    'status': 'pending'
                })
                todo_id += 1
                
        return todos[:10]  # Limit to 10 todos max
    
    def _execute_todos_via_claude(self, todos: List[Dict], original_task: Dict) -> bool:
        """Execute todos via Claude Code integration until success"""
        logger.info(f"🤖 Executing {len(todos)} todos via Claude Code")
        
        # Create Claude prompt with todos
        claude_prompt = self._build_claude_execution_prompt(todos, original_task)
        
        try:
            # Execute via Claude Code
            result = subprocess.run([
                'claude', '--headless',
                '--prompt', claude_prompt
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Check if original task is now complete
                return self._verify_task_completion(original_task['id'])
            else:
                logger.error(f"Claude execution failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to execute todos via Claude: {e}")
            
        return False
    
    def _build_claude_execution_prompt(self, todos: List[Dict], original_task: Dict) -> str:
        """Build comprehensive Claude execution prompt"""
        prompt_parts = [
            f"# Autonomous Task Execution",
            f"",
            f"## Original Task",
            f"**ID:** {original_task['id']}",
            f"**Title:** {original_task['title']}",
            f"**Description:** {original_task.get('description', '')}",
            f"",
            f"## Research-Driven Action Plan",
            f"Execute the following steps to complete the task:",
            f""
        ]
        
        for i, todo in enumerate(todos, 1):
            prompt_parts.append(f"{i}. {todo['description']}")
        
        prompt_parts.extend([
            f"",
            f"## Success Criteria",
            f"- Complete all action steps above",
            f"- Verify task implementation works correctly",
            f"- Mark task as done using: task-master set-status --id={original_task['id']} --status=done",
            f"",
            f"## Context",
            f"- You are part of an autonomous workflow loop",
            f"- Use task-master commands as needed",
            f"- If you encounter issues, document them clearly",
            f"- Focus on practical implementation over explanation"
        ])
        
        return '\n'.join(prompt_parts)
    
    def _verify_task_completion(self, task_id: str) -> bool:
        """Verify that task was completed successfully"""
        try:
            task_details = self._get_task_details(task_id)
            if task_details:
                return task_details.get('status', '').lower() == 'done'
        except Exception as e:
            logger.error(f"Failed to verify task completion: {e}")
        return False
    
    def _has_implementation_details(self, task: Dict) -> bool:
        """Check if task has sufficient implementation details"""
        details = task.get('details', '')
        return len(details) > 100 and any(keyword in details.lower() for keyword in [
            'implement', 'create', 'build', 'execute', 'run', 'install', 'configure'
        ])
    
    def _is_code_task(self, task: Dict) -> bool:
        """Check if task is code-related"""
        text = f"{task.get('title', '')} {task.get('description', '')}".lower()
        return any(keyword in text for keyword in [
            'code', 'python', 'script', 'function', 'class', 'implement', 
            'algorithm', 'optimization', 'programming', 'development'
        ])
    
    def _execute_with_details(self, task: Dict) -> bool:
        """Execute task using its implementation details"""
        # This would implement task execution based on details
        logger.info("🔧 Executing with implementation details")
        return False  # Placeholder
    
    def _execute_with_patterns(self, task: Dict) -> bool:
        """Execute task using automated patterns"""
        # This would implement pattern-based execution
        logger.info("🔄 Executing with automated patterns")
        return False  # Placeholder
    
    def _execute_with_claude_code(self, task: Dict) -> bool:
        """Execute task using Claude Code integration"""
        logger.info("🤖 Executing via Claude Code")
        
        try:
            prompt = f"""
            Execute this task autonomously:
            
            **Task:** {task['title']}
            **Description:** {task.get('description', '')}
            **Details:** {task.get('details', '')}
            
            Requirements:
            - Implement the task completely
            - Use task-master commands as needed
            - Mark complete when done: task-master set-status --id={task['id']} --status=done
            """
            
            result = subprocess.run([
                'claude', '--headless', '--prompt', prompt
            ], capture_output=True, text=True, timeout=180)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Claude Code execution failed: {e}")
            return False
    
    def _extract_subtasks(self, task: Dict) -> List[Dict]:
        """Extract subtasks from expanded task"""
        # This would need task-master API to list subtasks
        return []
    
    def _set_task_status(self, task_id: str, status: str) -> bool:
        """Set task status using task-master"""
        try:
            result = subprocess.run([
                'task-master', 'set-status',
                '--id', task_id,
                '--status', status
            ], capture_output=True, text=True, timeout=30)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to set task status: {e}")
            return False
    
    def _handle_task_success(self, task: Dict) -> None:
        """Handle successful task completion"""
        self.state.success_count += 1
        logger.info(f"✅ Task {task['id']} completed successfully")
        
        # Log success
        self.workflow_log.append({
            'timestamp': datetime.now().isoformat(),
            'task_id': task['id'],
            'status': 'success',
            'attempts': self.state.execution_attempts,
            'research_used': self.state.research_attempts > 0
        })
    
    def _mark_task_failed(self, task: Dict) -> None:
        """Mark task as failed and continue"""
        logger.error(f"❌ Marking task {task['id']} as failed")
        self._set_task_status(task['id'], 'cancelled')
        
        # Log failure
        self.workflow_log.append({
            'timestamp': datetime.now().isoformat(),
            'task_id': task['id'],
            'status': 'failed',
            'attempts': self.state.execution_attempts,
            'research_attempts': self.state.research_attempts,
            'last_error': self.state.last_error
        })
    
    def _reset_task_state(self) -> None:
        """Reset state for next task"""
        self.state.current_task_id = None
        self.state.execution_attempts = 0
        self.state.research_attempts = 0
        self.state.last_error = None
        self.state.stuck_indicator = False
    
    def generate_workflow_report(self) -> Dict:
        """Generate comprehensive workflow execution report"""
        return {
            'workflow_summary': {
                'total_success': self.state.success_count,
                'total_failures': self.state.failure_count,
                'success_rate': self.state.success_count / max(1, self.state.success_count + self.state.failure_count),
                'research_driven_solutions': len([log for log in self.workflow_log if log.get('research_used', False)])
            },
            'execution_log': self.workflow_log,
            'final_state': {
                'current_task_id': self.state.current_task_id,
                'stuck_indicator': self.state.stuck_indicator,
                'last_error': self.state.last_error
            }
        }

def main():
    """Main execution function"""
    print("🚀 Autonomous Workflow Loop with Research-Driven Problem Solving")
    print("=" * 80)
    print("Workflow Pattern: Get Stuck → Research with Task-Master + Perplexity → Parse Todos → Execute via Claude → Success")
    print("=" * 80)
    
    workflow = AutonomousWorkflowLoop()
    
    try:
        success = workflow.run_autonomous_loop()
        
        # Generate final report
        report = workflow.generate_workflow_report()
        
        # Save workflow report
        os.makedirs('.taskmaster/reports', exist_ok=True)
        with open('.taskmaster/reports/autonomous-workflow-report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "=" * 80)
        print("🎯 Workflow Summary")
        print("=" * 80)
        print(f"✅ Successful Tasks: {report['workflow_summary']['total_success']}")
        print(f"❌ Failed Tasks: {report['workflow_summary']['total_failures']}")
        print(f"📊 Success Rate: {report['workflow_summary']['success_rate']:.1%}")
        print(f"🔬 Research-Driven Solutions: {report['workflow_summary']['research_driven_solutions']}")
        print("=" * 80)
        
        return success
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)