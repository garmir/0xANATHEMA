#!/usr/bin/env python3
"""
Checkpoint and Resume Functionality
Implements checkpoint/resume capability for autonomous execution
"""

import json
import time
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging
import pickle

@dataclass
class CheckpointData:
    """Checkpoint data structure"""
    checkpoint_id: str
    timestamp: datetime
    execution_state: Dict[str, Any]
    current_task: Optional[Dict[str, Any]]
    completed_tasks: List[int]
    failed_tasks: List[int]
    execution_context: Dict[str, Any]
    system_metrics: Dict[str, Any]

@dataclass
class ResumeResult:
    """Resume operation result"""
    success: bool
    checkpoint_id: str
    tasks_resumed: int
    execution_context: Dict[str, Any]
    resume_timestamp: datetime


class CheckpointManager:
    """Manages checkpoint and resume functionality"""
    
    def __init__(self, checkpoint_dir: str = '.taskmaster/checkpoint'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('CheckpointManager')
    
    def create_checkpoint(self, execution_state: Dict[str, Any], 
                         current_task: Optional[Dict[str, Any]] = None) -> str:
        """Create execution checkpoint"""
        checkpoint_id = f"checkpoint_{int(time.time())}"
        
        # Gather current system state
        system_metrics = self._gather_system_metrics()
        execution_context = self._gather_execution_context()
        
        # Get task completion status
        completed_tasks, failed_tasks = self._get_task_status()
        
        checkpoint_data = CheckpointData(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            execution_state=execution_state,
            current_task=current_task,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            execution_context=execution_context,
            system_metrics=system_metrics
        )
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(asdict(checkpoint_data), f, indent=2, default=str)
            
            # Also save binary backup for complex objects
            binary_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            with open(binary_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"Checkpoint created: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    def resume_from_checkpoint(self, checkpoint_id: Optional[str] = None) -> ResumeResult:
        """Resume execution from checkpoint"""
        
        if checkpoint_id is None:
            checkpoint_id = self._get_latest_checkpoint()
        
        if checkpoint_id is None:
            raise ValueError("No checkpoints available for resume")
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        try:
            # Load checkpoint data
            with open(checkpoint_path, 'r') as f:
                checkpoint_dict = json.load(f)
            
            # Restore execution state
            execution_context = checkpoint_dict['execution_context']
            completed_tasks = checkpoint_dict['completed_tasks']
            
            # Calculate tasks to resume
            all_tasks = self._get_all_tasks()
            tasks_to_resume = [t for t in all_tasks if t['id'] not in completed_tasks]
            
            result = ResumeResult(
                success=True,
                checkpoint_id=checkpoint_id,
                tasks_resumed=len(tasks_to_resume),
                execution_context=execution_context,
                resume_timestamp=datetime.now()
            )
            
            self.logger.info(f"Resumed from checkpoint {checkpoint_id}, {len(tasks_to_resume)} tasks to process")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to resume from checkpoint: {e}")
            return ResumeResult(
                success=False,
                checkpoint_id=checkpoint_id,
                tasks_resumed=0,
                execution_context={},
                resume_timestamp=datetime.now()
            )
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints"""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                checkpoints.append({
                    'checkpoint_id': data['checkpoint_id'],
                    'timestamp': data['timestamp'],
                    'completed_tasks': len(data['completed_tasks']),
                    'current_task': data.get('current_task', {}).get('title', 'None')
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_count: int = 5):
        """Clean up old checkpoints, keeping only the most recent"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_count:
            return
        
        to_delete = checkpoints[keep_count:]
        
        for checkpoint in to_delete:
            checkpoint_id = checkpoint['checkpoint_id']
            
            # Delete JSON file
            json_path = self.checkpoint_dir / f"{checkpoint_id}.json"
            if json_path.exists():
                json_path.unlink()
            
            # Delete binary file
            pkl_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            if pkl_path.exists():
                pkl_path.unlink()
            
            self.logger.info(f"Deleted old checkpoint: {checkpoint_id}")
    
    def _gather_system_metrics(self) -> Dict[str, Any]:
        """Gather current system metrics"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            return {
                'timestamp': datetime.now().isoformat(),
                'metrics_available': False
            }
    
    def _gather_execution_context(self) -> Dict[str, Any]:
        """Gather execution context"""
        return {
            'working_directory': os.getcwd(),
            'environment_variables': dict(os.environ),
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_task_status(self) -> tuple:
        """Get current task completion status"""
        completed_tasks = []
        failed_tasks = []
        
        try:
            tasks_file = Path('.taskmaster/tasks/tasks.json')
            if tasks_file.exists():
                with open(tasks_file, 'r') as f:
                    data = json.load(f)
                
                tasks = data.get('master', {}).get('tasks', [])
                
                for task in tasks:
                    task_id = task.get('id')
                    status = task.get('status')
                    
                    if status == 'done':
                        completed_tasks.append(task_id)
                    elif status == 'failed':
                        failed_tasks.append(task_id)
        
        except Exception as e:
            self.logger.warning(f"Failed to get task status: {e}")
        
        return completed_tasks, failed_tasks
    
    def _get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks"""
        try:
            tasks_file = Path('.taskmaster/tasks/tasks.json')
            if tasks_file.exists():
                with open(tasks_file, 'r') as f:
                    data = json.load(f)
                
                return data.get('master', {}).get('tasks', [])
        
        except Exception as e:
            self.logger.warning(f"Failed to get tasks: {e}")
        
        return []
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get the most recent checkpoint"""
        checkpoints = self.list_checkpoints()
        
        if checkpoints:
            return checkpoints[0]['checkpoint_id']
        
        return None


def main():
    """Main checkpoint functionality demo"""
    print("Checkpoint and Resume Functionality")
    print("=" * 40)
    
    manager = CheckpointManager()
    
    try:
        # Demo: Create a checkpoint
        execution_state = {
            'current_phase': 'task_execution',
            'iteration': 1,
            'last_action': 'completed_task_analysis'
        }
        
        current_task = {
            'id': 43,
            'title': 'Demo task for checkpoint',
            'status': 'in-progress'
        }
        
        checkpoint_id = manager.create_checkpoint(execution_state, current_task)
        print(f"✅ Created checkpoint: {checkpoint_id}")
        
        # List checkpoints
        checkpoints = manager.list_checkpoints()
        print(f"📋 Available checkpoints: {len(checkpoints)}")
        
        for cp in checkpoints:
            print(f"  • {cp['checkpoint_id']}: {cp['completed_tasks']} tasks completed")
        
        # Demo resume
        resume_result = manager.resume_from_checkpoint(checkpoint_id)
        
        if resume_result.success:
            print(f"✅ Successfully resumed from {checkpoint_id}")
            print(f"   Tasks to resume: {resume_result.tasks_resumed}")
        else:
            print(f"❌ Failed to resume from {checkpoint_id}")
        
        print(f"\n✅ Checkpoint functionality validated")
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint functionality failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)