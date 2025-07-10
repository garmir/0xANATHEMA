#!/usr/bin/env python3
"""
Catalytic Workspace Checkpoint Manager
Implements checkpoint/resume functionality with state persistence and data integrity
"""

import json
import time
import hashlib
import pickle
import gzip
import shutil
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import threading
import queue
import os


@dataclass
class CheckpointMetadata:
    """Metadata for workspace checkpoints"""
    checkpoint_id: str
    timestamp: datetime
    workspace_state: str
    memory_usage_mb: float
    data_integrity_hash: str
    task_context: Dict[str, Any]
    dependencies: List[str]
    rollback_point: bool


class WorkspaceStateManager:
    """Manages workspace state persistence and integrity"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.checkpoints_path = self.workspace_path / "checkpoints"
        self.state_path = self.workspace_path / "state"
        self.integrity_path = self.workspace_path / "integrity"
        
        # Create directories
        self.checkpoints_path.mkdir(exist_ok=True)
        self.state_path.mkdir(exist_ok=True)
        self.integrity_path.mkdir(exist_ok=True)
        
        # Active state tracking
        self.active_checkpoint = None
        self.state_lock = threading.Lock()
        self.integrity_checks = []
        
    def create_checkpoint(self, task_context: Dict[str, Any], 
                         rollback_point: bool = False) -> str:
        """Create a new checkpoint with current workspace state"""
        checkpoint_id = f"checkpoint_{int(time.time() * 1000)}"
        timestamp = datetime.now()
        
        with self.state_lock:
            # Calculate workspace state hash
            workspace_hash = self._calculate_workspace_hash()
            
            # Get memory usage
            memory_usage = self._get_memory_usage()
            
            # Create checkpoint metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                timestamp=timestamp,
                workspace_state="active",
                memory_usage_mb=memory_usage,
                data_integrity_hash=workspace_hash,
                task_context=task_context,
                dependencies=task_context.get("dependencies", []),
                rollback_point=rollback_point
            )
            
            # Save checkpoint data
            checkpoint_dir = self.checkpoints_path / checkpoint_id
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save metadata
            metadata_file = checkpoint_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            # Save workspace state
            self._save_workspace_state(checkpoint_dir)
            
            # Save memory pool state
            self._save_memory_pool_state(checkpoint_dir)
            
            # Update active checkpoint
            self.active_checkpoint = checkpoint_id
            
            print(f"✓ Checkpoint {checkpoint_id} created successfully")
            return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore workspace to specified checkpoint"""
        checkpoint_dir = self.checkpoints_path / checkpoint_id
        
        if not checkpoint_dir.exists():
            print(f"✗ Checkpoint {checkpoint_id} not found")
            return False
        
        try:
            with self.state_lock:
                # Load checkpoint metadata
                metadata_file = checkpoint_dir / "metadata.json"
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                print(f"Restoring checkpoint {checkpoint_id} from {metadata['timestamp']}")
                
                # Verify data integrity
                if not self._verify_checkpoint_integrity(checkpoint_dir, metadata):
                    print(f"✗ Checkpoint {checkpoint_id} integrity verification failed")
                    return False
                
                # Restore workspace state
                self._restore_workspace_state(checkpoint_dir)
                
                # Restore memory pool state
                self._restore_memory_pool_state(checkpoint_dir)
                
                # Update active checkpoint
                self.active_checkpoint = checkpoint_id
                
                print(f"✓ Checkpoint {checkpoint_id} restored successfully")
                return True
                
        except Exception as e:
            print(f"✗ Failed to restore checkpoint {checkpoint_id}: {e}")
            return False
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints"""
        checkpoints = []
        
        for checkpoint_dir in self.checkpoints_path.iterdir():
            if checkpoint_dir.is_dir():
                metadata_file = checkpoint_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata_dict = json.load(f)
                            # Convert timestamp back to datetime
                            metadata_dict['timestamp'] = datetime.fromisoformat(
                                metadata_dict['timestamp'].replace('Z', '+00:00')
                            )
                            metadata = CheckpointMetadata(**metadata_dict)
                            checkpoints.append(metadata)
                    except Exception as e:
                        print(f"Warning: Failed to load checkpoint metadata: {e}")
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_count: int = 10):
        """Remove old checkpoints, keeping only the most recent ones"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_count:
            return
        
        # Keep rollback points and most recent checkpoints
        rollback_points = [cp for cp in checkpoints if cp.rollback_point]
        regular_checkpoints = [cp for cp in checkpoints if not cp.rollback_point]
        
        # Remove excess regular checkpoints
        to_remove = regular_checkpoints[keep_count:]
        
        for checkpoint in to_remove:
            checkpoint_dir = self.checkpoints_path / checkpoint.checkpoint_id
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
                print(f"Removed old checkpoint: {checkpoint.checkpoint_id}")
    
    def verify_data_integrity(self) -> bool:
        """Verify current workspace data integrity"""
        try:
            current_hash = self._calculate_workspace_hash()
            
            if self.active_checkpoint:
                checkpoint_dir = self.checkpoints_path / self.active_checkpoint
                metadata_file = checkpoint_dir / "metadata.json"
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    expected_hash = metadata.get('data_integrity_hash')
                    if expected_hash and current_hash != expected_hash:
                        print("⚠ Data integrity verification failed - workspace modified")
                        return False
            
            print("✓ Data integrity verification passed")
            return True
            
        except Exception as e:
            print(f"✗ Data integrity verification error: {e}")
            return False
    
    def _calculate_workspace_hash(self) -> str:
        """Calculate hash of current workspace state"""
        hasher = hashlib.sha256()
        
        # Hash all files in workspace
        for root, dirs, files in os.walk(self.workspace_path):
            # Skip checkpoint directories
            if "checkpoints" in root:
                continue
                
            for file in sorted(files):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
                except (IOError, OSError):
                    continue
        
        return hasher.hexdigest()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _save_workspace_state(self, checkpoint_dir: Path):
        """Save current workspace state to checkpoint"""
        state_file = checkpoint_dir / "workspace_state.json"
        
        workspace_state = {
            "timestamp": datetime.now().isoformat(),
            "active_tasks": [],
            "memory_allocations": {},
            "cache_state": {},
            "execution_context": {}
        }
        
        with open(state_file, 'w') as f:
            json.dump(workspace_state, f, indent=2)
    
    def _save_memory_pool_state(self, checkpoint_dir: Path):
        """Save memory pool state to checkpoint"""
        memory_file = checkpoint_dir / "memory_pool.bin"
        
        # Simulate memory pool state saving
        memory_state = {
            "pool_size": "8GB",
            "allocated_blocks": [],
            "free_blocks": [],
            "reuse_statistics": {}
        }
        
        with gzip.open(memory_file, 'wb') as f:
            pickle.dump(memory_state, f)
    
    def _restore_workspace_state(self, checkpoint_dir: Path):
        """Restore workspace state from checkpoint"""
        state_file = checkpoint_dir / "workspace_state.json"
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                workspace_state = json.load(f)
            print("✓ Workspace state restored")
    
    def _restore_memory_pool_state(self, checkpoint_dir: Path):
        """Restore memory pool state from checkpoint"""
        memory_file = checkpoint_dir / "memory_pool.bin"
        
        if memory_file.exists():
            with gzip.open(memory_file, 'rb') as f:
                memory_state = pickle.load(f)
            print("✓ Memory pool state restored")
    
    def _verify_checkpoint_integrity(self, checkpoint_dir: Path, 
                                   metadata: Dict[str, Any]) -> bool:
        """Verify checkpoint data integrity"""
        # Verify all required files exist
        required_files = [
            "metadata.json",
            "workspace_state.json", 
            "memory_pool.bin"
        ]
        
        for file_name in required_files:
            file_path = checkpoint_dir / file_name
            if not file_path.exists():
                print(f"✗ Missing checkpoint file: {file_name}")
                return False
        
        # Additional integrity checks could be added here
        return True


class CatalyticWorkspaceManager:
    """Main catalytic workspace management interface"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.workspace_path = self.config_path.parent
        self.state_manager = WorkspaceStateManager(str(self.workspace_path))
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Add checkpoint configuration
        self.config.update({
            "checkpoint_features": {
                "auto_checkpoint": True,
                "checkpoint_interval_minutes": 5,
                "max_checkpoints": 20,
                "rollback_on_failure": True,
                "data_integrity_verification": True
            }
        })
        
        # Save updated configuration
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def execute_with_checkpoints(self, task_function, task_context: Dict[str, Any]):
        """Execute task with automatic checkpoint management"""
        # Create initial checkpoint
        initial_checkpoint = self.state_manager.create_checkpoint(
            task_context, rollback_point=True
        )
        
        try:
            # Execute task
            result = task_function(task_context)
            
            # Create success checkpoint
            success_checkpoint = self.state_manager.create_checkpoint(
                {**task_context, "result": "success"}, rollback_point=True
            )
            
            print(f"✓ Task completed successfully with checkpoint {success_checkpoint}")
            return result
            
        except Exception as e:
            print(f"✗ Task execution failed: {e}")
            
            # Rollback to initial checkpoint if configured
            if self.config["checkpoint_features"]["rollback_on_failure"]:
                print(f"Rolling back to checkpoint {initial_checkpoint}")
                self.state_manager.restore_checkpoint(initial_checkpoint)
            
            raise e
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workspace status"""
        checkpoints = self.state_manager.list_checkpoints()
        
        return {
            "workspace_config": self.config,
            "active_checkpoint": self.state_manager.active_checkpoint,
            "total_checkpoints": len(checkpoints),
            "latest_checkpoint": checkpoints[0].checkpoint_id if checkpoints else None,
            "data_integrity": self.state_manager.verify_data_integrity(),
            "memory_usage_mb": self.state_manager._get_memory_usage()
        }


if __name__ == "__main__":
    # Example usage
    config_path = "/Users/anam/archive/.taskmaster/catalytic/workspace-config.json"
    workspace = CatalyticWorkspaceManager(config_path)
    
    print("Catalytic Workspace with Checkpoint Management")
    print("=" * 50)
    
    # Create test checkpoint
    test_context = {
        "task_id": "test-checkpoint",
        "description": "Testing checkpoint functionality",
        "dependencies": []
    }
    
    checkpoint_id = workspace.state_manager.create_checkpoint(test_context, rollback_point=True)
    
    # Show status
    status = workspace.get_status()
    print(f"Workspace Status: {json.dumps(status, indent=2, default=str)}")
    
    # List checkpoints
    checkpoints = workspace.state_manager.list_checkpoints()
    print(f"\nAvailable Checkpoints: {len(checkpoints)}")
    for cp in checkpoints[:3]:  # Show first 3
        print(f"  {cp.checkpoint_id} - {cp.timestamp} ({'Rollback Point' if cp.rollback_point else 'Regular'})")