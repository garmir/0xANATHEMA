"""
TaskMaster Integration Layer for LABRYS Framework
Enhanced TaskMaster with dual-blade methodology
"""

import os
import sys
import json
import asyncio
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '.labrys'))
from coordination.labrys_coordinator import LabrysCoordinator
from analytical.analytical_blade import AnalyticalBlade
from synthesis.synthesis_blade import SynthesisBlade

class TaskType(Enum):
    ANALYTICAL = "analytical"
    SYNTHESIS = "synthesis"
    COORDINATION = "coordination"
    VALIDATION = "validation"
    INTEGRATION = "integration"

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class LabrysTask:
    """Enhanced task structure with LABRYS methodology"""
    id: str
    title: str
    description: str
    type: TaskType
    priority: str
    dependencies: List[str]
    validation: List[str]
    status: TaskStatus = TaskStatus.PENDING
    blade_assignment: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

class TaskMasterLabrys:
    """
    Enhanced TaskMaster with LABRYS dual-blade methodology
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.coordinator = LabrysCoordinator()
        self.tasks = []
        self.active_tasks = {}
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Task execution tracking
        self.execution_log = []
        self.performance_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_execution_time": 0,
            "blade_utilization": {"analytical": 0, "synthesis": 0}
        }
        
        # Initialize coordination system
        self.coordination_initialized = False
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load TaskMaster configuration"""
        default_config = {
            "auto_blade_assignment": True,
            "parallel_execution": True,
            "max_concurrent_tasks": 5,
            "task_timeout": 300,  # 5 minutes
            "auto_retry": True,
            "retry_attempts": 3,
            "validation_required": True,
            "logging_enabled": True
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
        
        return default_config
    
    async def initialize_labrys_system(self) -> Dict[str, Any]:
        """
        Initialize the LABRYS dual-blade system
        """
        try:
            initialization_result = await self.coordinator.initialize_dual_blades()
            
            if initialization_result.get("synchronization", {}).get("status") == "synchronized":
                self.coordination_initialized = True
                return {
                    "status": "success",
                    "message": "LABRYS system initialized successfully",
                    "details": initialization_result
                }
            else:
                return {
                    "status": "partial",
                    "message": "LABRYS system partially initialized",
                    "details": initialization_result
                }
        
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Failed to initialize LABRYS system: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def load_tasks_from_json(self, tasks_json: Dict[str, Any]) -> List[LabrysTask]:
        """
        Load tasks from JSON configuration with LABRYS enhancements
        """
        loaded_tasks = []
        
        # Process different task phases
        for phase_name, phase_data in tasks_json.items():
            if "tasks" in phase_data:
                for task_data in phase_data["tasks"]:
                    # Determine task type based on content
                    task_type = self._determine_task_type(task_data)
                    
                    # Create enhanced task
                    task = LabrysTask(
                        id=task_data["id"],
                        title=task_data["title"],
                        description=task_data["description"],
                        type=task_type,
                        priority=task_data.get("priority", "medium"),
                        dependencies=task_data.get("dependencies", []),
                        validation=task_data.get("validation", [])
                    )
                    
                    # Auto-assign blade if enabled
                    if self.config["auto_blade_assignment"]:
                        task.blade_assignment = self._assign_blade(task)
                    
                    loaded_tasks.append(task)
                    self.tasks.append(task)
        
        return loaded_tasks
    
    def _determine_task_type(self, task_data: Dict[str, Any]) -> TaskType:
        """
        Determine task type based on task content
        """
        description = task_data.get("description", "").lower()
        title = task_data.get("title", "").lower()
        
        if any(keyword in description or keyword in title for keyword in 
               ["analyze", "research", "review", "evaluate", "assess"]):
            return TaskType.ANALYTICAL
        
        elif any(keyword in description or keyword in title for keyword in 
                ["generate", "create", "build", "implement", "develop"]):
            return TaskType.SYNTHESIS
        
        elif any(keyword in description or keyword in title for keyword in 
                ["coordinate", "sync", "integrate", "combine"]):
            return TaskType.COORDINATION
        
        elif any(keyword in description or keyword in title for keyword in 
                ["validate", "test", "verify", "check"]):
            return TaskType.VALIDATION
        
        else:
            return TaskType.INTEGRATION
    
    def _assign_blade(self, task: LabrysTask) -> str:
        """
        Assign appropriate blade based on task type
        """
        if task.type == TaskType.ANALYTICAL:
            return "analytical"
        elif task.type == TaskType.SYNTHESIS:
            return "synthesis"
        elif task.type == TaskType.COORDINATION:
            return "coordination"
        else:
            return "analytical"  # Default to analytical for validation/integration
    
    async def execute_task(self, task: LabrysTask) -> Dict[str, Any]:
        """
        Execute a single task using appropriate blade
        """
        if not self.coordination_initialized:
            init_result = await self.initialize_labrys_system()
            if init_result["status"] == "failed":
                return {"error": "Failed to initialize LABRYS system", "task_id": task.id}
        
        task.status = TaskStatus.IN_PROGRESS
        self.active_tasks[task.id] = task
        
        execution_start = datetime.now()
        
        try:
            # Execute based on blade assignment
            if task.blade_assignment == "analytical":
                result = await self._execute_analytical_task(task)
            elif task.blade_assignment == "synthesis":
                result = await self._execute_synthesis_task(task)
            elif task.blade_assignment == "coordination":
                result = await self._execute_coordination_task(task)
            else:
                result = {"error": f"Unknown blade assignment: {task.blade_assignment}"}
            
            # Validate task completion if required
            if self.config["validation_required"] and task.validation:
                validation_result = await self._validate_task_completion(task, result)
                if not validation_result["valid"]:
                    task.status = TaskStatus.FAILED
                    task.error = validation_result["error"]
                    return {"error": validation_result["error"], "task_id": task.id}
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.results = result
            task.completed_at = datetime.now().isoformat()
            
            # Update metrics
            execution_time = (datetime.now() - execution_start).total_seconds()
            self._update_performance_metrics(task, execution_time)
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now().isoformat()
            
            self.failed_tasks.append(task)
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
            return {"error": str(e), "task_id": task.id}
    
    async def _execute_analytical_task(self, task: LabrysTask) -> Dict[str, Any]:
        """
        Execute task using analytical blade
        """
        analytical_blade = self.coordinator.analytical_blade
        
        # Determine specific analytical operation
        if "research" in task.description.lower():
            result = await analytical_blade.computational_research(
                task.description, 
                domain="development"
            )
        elif "analyze" in task.description.lower():
            # For analysis tasks, we need code content
            # This is a simplified example
            result = await analytical_blade.static_analysis(
                "# Sample code for analysis", 
                language="python"
            )
            result = asdict(result)
        else:
            # Generic constraint identification
            result = await analytical_blade.constraint_identification(task.description)
            result = asdict(result)
        
        return {
            "task_id": task.id,
            "blade": "analytical",
            "operation": "analysis",
            "result": result
        }
    
    async def _execute_synthesis_task(self, task: LabrysTask) -> Dict[str, Any]:
        """
        Execute task using synthesis blade
        """
        synthesis_blade = self.coordinator.synthesis_blade
        
        # Prepare synthesis specifications
        specifications = {
            "type": "function" if "function" in task.description.lower() else "class",
            "name": task.title.replace(" ", "_").lower(),
            "language": "python",
            "requirements": [task.description],
            "constraints": []
        }
        
        result = await synthesis_blade.claude_sparc_generation(specifications)
        
        return {
            "task_id": task.id,
            "blade": "synthesis",
            "operation": "code_generation",
            "result": asdict(result)
        }
    
    async def _execute_coordination_task(self, task: LabrysTask) -> Dict[str, Any]:
        """
        Execute task using coordination system
        """
        # Create workflow specification
        workflow_spec = {
            "analytical_tasks": [
                {
                    "type": "research",
                    "query": task.description,
                    "domain": "development"
                }
            ],
            "synthesis_tasks": [
                {
                    "type": "code_generation",
                    "specifications": {
                        "type": "function",
                        "name": task.title.replace(" ", "_").lower(),
                        "language": "python",
                        "requirements": [task.description]
                    }
                }
            ],
            "dependencies": task.dependencies,
            "priority": task.priority
        }
        
        result = await self.coordinator.execute_coordinated_workflow(workflow_spec)
        
        return {
            "task_id": task.id,
            "blade": "coordination",
            "operation": "workflow_execution",
            "result": result
        }
    
    async def _validate_task_completion(self, task: LabrysTask, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate task completion based on validation criteria
        """
        if not task.validation:
            return {"valid": True, "message": "No validation criteria specified"}
        
        # Execute validation commands
        validation_results = []
        
        for validation_cmd in task.validation:
            try:
                if validation_cmd.startswith("test"):
                    # File/directory existence test
                    validation_result = os.path.exists(validation_cmd.split()[1])
                elif validation_cmd.startswith("python"):
                    # Python validation
                    process = subprocess.run(
                        validation_cmd, 
                        shell=True, 
                        capture_output=True, 
                        text=True,
                        timeout=30
                    )
                    validation_result = process.returncode == 0
                else:
                    # Generic command validation
                    process = subprocess.run(
                        validation_cmd, 
                        shell=True, 
                        capture_output=True, 
                        text=True,
                        timeout=30
                    )
                    validation_result = process.returncode == 0
                
                validation_results.append({
                    "command": validation_cmd,
                    "passed": validation_result
                })
            
            except Exception as e:
                validation_results.append({
                    "command": validation_cmd,
                    "passed": False,
                    "error": str(e)
                })
        
        # Check if all validations passed
        all_passed = all(v["passed"] for v in validation_results)
        
        return {
            "valid": all_passed,
            "validation_results": validation_results,
            "error": None if all_passed else "Some validations failed"
        }
    
    def _update_performance_metrics(self, task: LabrysTask, execution_time: float):
        """
        Update performance metrics
        """
        self.performance_metrics["total_tasks"] += 1
        
        if task.status == TaskStatus.COMPLETED:
            self.performance_metrics["completed_tasks"] += 1
        elif task.status == TaskStatus.FAILED:
            self.performance_metrics["failed_tasks"] += 1
        
        # Update average execution time
        current_avg = self.performance_metrics["avg_execution_time"]
        total_tasks = self.performance_metrics["total_tasks"]
        
        self.performance_metrics["avg_execution_time"] = (
            (current_avg * (total_tasks - 1) + execution_time) / total_tasks
        )
        
        # Update blade utilization
        if task.blade_assignment in self.performance_metrics["blade_utilization"]:
            self.performance_metrics["blade_utilization"][task.blade_assignment] += 1
    
    async def execute_task_sequence(self, tasks: List[LabrysTask]) -> Dict[str, Any]:
        """
        Execute a sequence of tasks with dependency resolution
        """
        execution_results = {}
        
        # Resolve dependencies and create execution order
        execution_order = self._resolve_task_dependencies(tasks)
        
        for task_batch in execution_order:
            # Execute tasks in current batch
            if self.config["parallel_execution"]:
                # Parallel execution
                batch_results = await asyncio.gather(
                    *[self.execute_task(task) for task in task_batch],
                    return_exceptions=True
                )
                
                for i, result in enumerate(batch_results):
                    task_id = task_batch[i].id
                    if isinstance(result, Exception):
                        execution_results[task_id] = {"error": str(result)}
                    else:
                        execution_results[task_id] = result
            else:
                # Sequential execution
                for task in task_batch:
                    result = await self.execute_task(task)
                    execution_results[task.id] = result
        
        return {
            "execution_results": execution_results,
            "total_tasks": len(tasks),
            "completed_tasks": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in tasks if t.status == TaskStatus.FAILED]),
            "performance_metrics": self.performance_metrics
        }
    
    def _resolve_task_dependencies(self, tasks: List[LabrysTask]) -> List[List[LabrysTask]]:
        """
        Resolve task dependencies and create execution batches
        """
        # Simple dependency resolution - topological sort
        batches = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            ready_tasks = []
            
            for task in remaining_tasks:
                if not task.dependencies or all(
                    dep_id in [completed_task.id for completed_task in self.completed_tasks]
                    for dep_id in task.dependencies
                ):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Break potential circular dependencies
                ready_tasks = [remaining_tasks[0]]
            
            batches.append(ready_tasks)
            for task in ready_tasks:
                remaining_tasks.remove(task)
        
        return batches
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        """
        coordination_status = self.coordinator.get_coordination_status()
        
        return {
            "taskmaster_status": {
                "coordination_initialized": self.coordination_initialized,
                "total_tasks": len(self.tasks),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "performance_metrics": self.performance_metrics
            },
            "coordination_status": coordination_status,
            "config": self.config
        }
    
    async def shutdown(self):
        """
        Shutdown TaskMaster and LABRYS system
        """
        # Cancel active tasks
        for task in self.active_tasks.values():
            task.status = TaskStatus.FAILED
            task.error = "System shutdown"
        
        # Shutdown coordinator
        await self.coordinator.shutdown()
        
        self.coordination_initialized = False

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TaskMaster LABRYS Integration")
    parser.add_argument("--initialize", action="store_true", help="Initialize LABRYS system")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--execute", help="Execute tasks from JSON file")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    taskmaster = TaskMasterLabrys(args.config)
    
    if args.initialize:
        result = asyncio.run(taskmaster.initialize_labrys_system())
        print(json.dumps(result, indent=2))
    
    elif args.status:
        status = taskmaster.get_system_status()
        print(json.dumps(status, indent=2))
    
    elif args.execute:
        if os.path.exists(args.execute):
            with open(args.execute, 'r') as f:
                tasks_json = json.load(f)
            
            tasks = taskmaster.load_tasks_from_json(tasks_json)
            result = asyncio.run(taskmaster.execute_task_sequence(tasks))
            print(json.dumps(result, indent=2))
        else:
            print(f"Task file not found: {args.execute}")
    
    else:
        print("TaskMaster LABRYS Integration ready")
        print("Use --help for available commands")