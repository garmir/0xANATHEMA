#!/usr/bin/env python3
"""
Atomic Task Breakdown Workflow Rule
==================================

Hard-coded workflow rule for recursively breaking down tasks to atomic levels
to ease prompt completion and reduce prompt complexity when prompts get stuck.

This implements the core workflow rule:
1. When prompts get stuck -> run 'task-master research' for breakdown analysis
2. Recursively expand tasks to atomic levels using task-master expand
3. Use 'task-master next' to get atomic tasks for simple execution
4. Execute atomic tasks as prompts with reduced complexity
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

class AtomicTaskBreakdownWorkflow:
    """Core workflow rule for atomic task breakdown when prompts get stuck"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.taskmaster_dir = self.project_root / ".taskmaster"
        self.workflow_rule_active = True
        
    def execute_stuck_prompt_resolution(self, stuck_task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the hard-coded workflow rule for resolving stuck prompts
        through recursive atomic breakdown
        """
        
        print("🔄 STUCK PROMPT RESOLUTION WORKFLOW ACTIVATED")
        print("=" * 60)
        print("Implementing hard-coded rule: Recursive atomic breakdown")
        print()
        
        workflow_results = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "workflow_rule": "atomic_task_breakdown",
            "trigger": "stuck_prompt_detection",
            "steps_executed": [],
            "atomic_tasks_generated": [],
            "execution_prompts": []
        }
        
        # Step 1: Research analysis for breakdown strategy
        print("📊 Step 1: Research Analysis for Task Breakdown")
        research_result = self._execute_research_analysis(stuck_task_id)
        workflow_results["steps_executed"].append({
            "step": 1,
            "action": "research_analysis",
            "status": "completed",
            "result": research_result
        })
        
        # Step 2: Identify target task for breakdown
        print("\n🎯 Step 2: Identify Target Task for Breakdown")
        target_task = self._identify_target_task(stuck_task_id)
        workflow_results["steps_executed"].append({
            "step": 2,
            "action": "target_identification",
            "status": "completed",
            "target_task": target_task
        })
        
        # Step 3: Recursive expansion to atomic levels
        print(f"\n🔄 Step 3: Recursive Expansion of Task {target_task}")
        atomic_tasks = self._recursive_expansion_to_atomic(target_task)
        workflow_results["atomic_tasks_generated"] = atomic_tasks
        workflow_results["steps_executed"].append({
            "step": 3,
            "action": "recursive_expansion",
            "status": "completed",
            "atomic_count": len(atomic_tasks)
        })
        
        # Step 4: Generate execution prompts for atomic tasks
        print(f"\n⚡ Step 4: Generate Execution Prompts for {len(atomic_tasks)} Atomic Tasks")
        execution_prompts = self._generate_atomic_execution_prompts(atomic_tasks)
        workflow_results["execution_prompts"] = execution_prompts
        workflow_results["steps_executed"].append({
            "step": 4,
            "action": "prompt_generation",
            "status": "completed",
            "prompt_count": len(execution_prompts)
        })
        
        # Step 5: Execute next atomic task
        print("\n🚀 Step 5: Execute Next Atomic Task")
        next_execution = self._execute_next_atomic_task()
        workflow_results["steps_executed"].append({
            "step": 5,
            "action": "atomic_execution",
            "status": "completed",
            "next_task": next_execution
        })
        
        # Save workflow results
        self._save_workflow_results(workflow_results)
        
        print(f"\n✅ ATOMIC BREAKDOWN WORKFLOW COMPLETE")
        print(f"📊 Generated {len(atomic_tasks)} atomic tasks")
        print(f"⚡ Created {len(execution_prompts)} execution prompts")
        print(f"🎯 Next task ready: {next_execution}")
        
        return workflow_results
    
    def _execute_research_analysis(self, task_id: Optional[str] = None) -> str:
        """Execute task-master research for breakdown analysis"""
        
        if task_id:
            research_query = f"recursively break down task {task_id} to atomic levels for easier execution"
        else:
            research_query = "recursively break down current complex tasks to atomic levels to reduce prompt complexity"
        
        try:
            cmd = ["task-master", "research", research_query]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("✅ Research analysis completed")
                return "research_completed"
            else:
                print(f"⚠️ Research command failed: {result.stderr}")
                return "research_fallback_used"
        except Exception as e:
            print(f"⚠️ Research execution error: {e}")
            return "research_error_fallback"
    
    def _identify_target_task(self, stuck_task_id: Optional[str] = None) -> str:
        """Identify the target task for atomic breakdown"""
        
        if stuck_task_id:
            print(f"   Target task specified: {stuck_task_id}")
            return stuck_task_id
        
        # Get current task from task-master next
        try:
            cmd = ["task-master", "next"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                # Parse task ID from output
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "Next Task: #" in line:
                        # Extract task ID from format "Next Task: #45.3"
                        task_id = line.split("#")[1].split(" ")[0]
                        print(f"   Current next task: {task_id}")
                        return task_id
                
                print("   No specific task found, using fallback")
                return "45.3"  # Current working task
            else:
                print(f"   Task-master next failed, using fallback")
                return "45.3"
        except Exception as e:
            print(f"   Error getting next task: {e}")
            return "45.3"
    
    def _recursive_expansion_to_atomic(self, task_id: str) -> List[str]:
        """Recursively expand task to atomic levels"""
        
        atomic_tasks = []
        expansion_queue = [task_id]
        max_depth = 3  # Prevent infinite recursion
        current_depth = 0
        
        while expansion_queue and current_depth < max_depth:
            current_task = expansion_queue.pop(0)
            current_depth += 1
            
            print(f"   Expanding {current_task} (depth {current_depth})")
            
            # Expand current task
            try:
                cmd = ["task-master", "expand", f"--id={current_task}", "--research", "--force"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode == 0:
                    # Get subtasks that were created
                    subtasks = self._get_task_subtasks(current_task)
                    
                    if subtasks:
                        print(f"   ✅ Expanded {current_task} into {len(subtasks)} subtasks")
                        
                        # Check if subtasks are atomic enough
                        for subtask in subtasks:
                            if self._is_task_atomic(subtask):
                                atomic_tasks.append(subtask)
                                print(f"   🔬 Task {subtask} is atomic")
                            else:
                                expansion_queue.append(subtask)
                                print(f"   🔄 Task {subtask} needs further breakdown")
                    else:
                        # No subtasks created, treat as atomic
                        atomic_tasks.append(current_task)
                        print(f"   🔬 Task {current_task} is atomic (no subtasks created)")
                else:
                    # Expansion failed, treat as atomic
                    atomic_tasks.append(current_task)
                    print(f"   🔬 Task {current_task} treated as atomic (expansion failed)")
                    
            except Exception as e:
                print(f"   ⚠️ Error expanding {current_task}: {e}")
                atomic_tasks.append(current_task)
        
        # Add any remaining tasks in queue as atomic
        atomic_tasks.extend(expansion_queue)
        
        return atomic_tasks
    
    def _get_task_subtasks(self, task_id: str) -> List[str]:
        """Get subtasks for a given task ID"""
        
        try:
            tasks_file = self.taskmaster_dir / "tasks" / "tasks.json"
            if not tasks_file.exists():
                return []
            
            with open(tasks_file, 'r') as f:
                data = json.load(f)
            
            # Navigate to the task and get subtasks
            task_parts = task_id.split('.')
            current_level = data.get("tasks", [])
            
            # Find the task
            for part in task_parts:
                task_found = False
                for task in current_level:
                    if str(task["id"]) == part:
                        if "subtasks" in task:
                            current_level = task["subtasks"]
                            task_found = True
                            break
                if not task_found:
                    return []
            
            # Get subtask IDs
            subtask_ids = []
            for subtask in current_level:
                if isinstance(subtask, dict) and "id" in subtask:
                    subtask_id = f"{task_id}.{subtask['id']}"
                    subtask_ids.append(subtask_id)
            
            return subtask_ids
            
        except Exception as e:
            print(f"   Error getting subtasks for {task_id}: {e}")
            return []
    
    def _is_task_atomic(self, task_id: str) -> bool:
        """Determine if a task is atomic (cannot be broken down further)"""
        
        # Simple heuristic: tasks with depth >= 3 are considered atomic
        depth = len(task_id.split('.'))
        
        if depth >= 4:  # e.g., 45.3.1.2 is depth 4
            return True
        
        # Check task description length/complexity
        try:
            tasks_file = self.taskmaster_dir / "tasks" / "tasks.json"
            if not tasks_file.exists():
                return True
            
            with open(tasks_file, 'r') as f:
                data = json.load(f)
            
            # Find task description
            task_parts = task_id.split('.')
            current_level = data.get("tasks", [])
            
            for i, part in enumerate(task_parts):
                for task in current_level:
                    if str(task["id"]) == part:
                        if i == len(task_parts) - 1:
                            # Found the task
                            description = task.get("description", "")
                            # Atomic if description is short and specific
                            return len(description.split()) < 20
                        else:
                            if "subtasks" in task:
                                current_level = task["subtasks"]
                                break
                else:
                    return True  # Task not found, assume atomic
            
            return True
            
        except Exception:
            return True  # Default to atomic if error
    
    def _generate_atomic_execution_prompts(self, atomic_tasks: List[str]) -> List[Dict[str, str]]:
        """Generate simple execution prompts for atomic tasks"""
        
        execution_prompts = []
        
        for task_id in atomic_tasks:
            try:
                # Get task details
                task_info = self._get_task_info(task_id)
                
                if task_info:
                    prompt = {
                        "task_id": task_id,
                        "title": task_info.get("title", "Unknown Task"),
                        "simple_prompt": f"Execute atomic task {task_id}: {task_info.get('title', '')}",
                        "detailed_prompt": f"""Execute atomic task {task_id}:

**Task**: {task_info.get('title', 'Unknown Task')}
**Description**: {task_info.get('description', 'No description available')}

This is an atomic-level task designed for simple execution without further breakdown.
Complete this specific task and mark as done when finished.

Use: task-master set-status --id={task_id} --status=done""",
                        "complexity": "atomic",
                        "estimated_time": "5-15 minutes"
                    }
                else:
                    prompt = {
                        "task_id": task_id,
                        "title": f"Atomic Task {task_id}",
                        "simple_prompt": f"Execute atomic task {task_id}",
                        "detailed_prompt": f"Execute atomic task {task_id} - details to be determined",
                        "complexity": "atomic",
                        "estimated_time": "5-15 minutes"
                    }
                
                execution_prompts.append(prompt)
                
            except Exception as e:
                print(f"   Error generating prompt for {task_id}: {e}")
        
        return execution_prompts
    
    def _get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information from tasks.json"""
        
        try:
            tasks_file = self.taskmaster_dir / "tasks" / "tasks.json"
            if not tasks_file.exists():
                return None
            
            with open(tasks_file, 'r') as f:
                data = json.load(f)
            
            # Navigate to the task
            task_parts = task_id.split('.')
            current_level = data.get("tasks", [])
            
            for i, part in enumerate(task_parts):
                for task in current_level:
                    if str(task["id"]) == part:
                        if i == len(task_parts) - 1:
                            # Found the task
                            return task
                        else:
                            if "subtasks" in task:
                                current_level = task["subtasks"]
                                break
                else:
                    return None
            
            return None
            
        except Exception:
            return None
    
    def _execute_next_atomic_task(self) -> str:
        """Execute task-master next to get the next atomic task"""
        
        try:
            cmd = ["task-master", "next"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                # Parse next task ID
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "Next Task: #" in line:
                        task_id = line.split("#")[1].split(" ")[0]
                        print(f"   Next atomic task ready: {task_id}")
                        return task_id
                
                print("   No next task available")
                return "no_task_available"
            else:
                print(f"   Error getting next task: {result.stderr}")
                return "error_getting_next_task"
        except Exception as e:
            print(f"   Exception getting next task: {e}")
            return "exception_getting_next_task"
    
    def _save_workflow_results(self, results: Dict[str, Any]):
        """Save workflow execution results"""
        
        timestamp = int(time.time())
        results_file = self.taskmaster_dir / "automation" / f"atomic_breakdown_workflow_{timestamp}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n📄 Workflow results saved: {results_file}")
        except Exception as e:
            print(f"\n⚠️ Error saving workflow results: {e}")

class StuckPromptDetector:
    """Detects when prompts are getting stuck and triggers atomic breakdown"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.workflow = AtomicTaskBreakdownWorkflow(project_root)
    
    def detect_and_resolve_stuck_prompt(self, context: str = "unknown") -> Dict[str, Any]:
        """Detect stuck prompt situation and resolve with atomic breakdown"""
        
        print(f"🚨 STUCK PROMPT DETECTED: {context}")
        print("Triggering hard-coded atomic breakdown workflow...")
        print()
        
        return self.workflow.execute_stuck_prompt_resolution()

def create_hard_coded_workflow_rule():
    """Create the hard-coded workflow rule for atomic task breakdown"""
    
    rule_content = '''
# HARD-CODED WORKFLOW RULE: Atomic Task Breakdown for Stuck Prompts

## Rule Definition
When prompts get stuck or become too complex:
1. Run 'task-master research' for breakdown analysis
2. Recursively expand tasks to atomic levels using task-master expand
3. Use 'task-master next' to get atomic tasks for simple execution
4. Execute atomic tasks as prompts with reduced complexity

## Trigger Conditions
- Prompt complexity too high
- Task execution stalled
- Multiple failed attempts
- User requests atomic breakdown

## Execution Protocol
```python
from atomic_task_breakdown_workflow import StuckPromptDetector

detector = StuckPromptDetector("/Users/anam/archive")
results = detector.detect_and_resolve_stuck_prompt("manual_trigger")
```

## Success Criteria
- Complex tasks broken into atomic components
- Each atomic task executable in 5-15 minutes
- Clear, simple execution prompts generated
- Reduced cognitive load for task completion

This rule is now HARD-CODED into the Task-Master workflow system.
'''
    
    rule_file = Path("/Users/anam/archive/.taskmaster/automation/HARD_CODED_WORKFLOW_RULE.md")
    rule_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(rule_file, 'w') as f:
        f.write(rule_content)
    
    print(f"✅ Hard-coded workflow rule created: {rule_file}")

def main():
    """Execute atomic task breakdown workflow"""
    
    project_root = "/Users/anam/archive"
    
    print("🔄 ATOMIC TASK BREAKDOWN WORKFLOW SYSTEM")
    print("=" * 50)
    print("Hard-coding workflow rule for stuck prompt resolution")
    print()
    
    # Create hard-coded workflow rule
    create_hard_coded_workflow_rule()
    
    # Execute workflow for current situation
    workflow = AtomicTaskBreakdownWorkflow(project_root)
    results = workflow.execute_stuck_prompt_resolution("45.3")
    
    print("\n🎯 WORKFLOW RULE IMPLEMENTATION COMPLETE")
    print("This atomic breakdown rule is now hard-coded for all future tasks")
    print("Use when prompts get stuck: StuckPromptDetector.detect_and_resolve_stuck_prompt()")
    
    return results

if __name__ == "__main__":
    main()