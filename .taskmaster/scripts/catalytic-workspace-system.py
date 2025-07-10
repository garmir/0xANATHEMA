#!/usr/bin/env python3
"""
Catalytic Workspace System for Task-Master
Implements memory reuse, checkpoint/resume, and workspace isolation
Based on the execution plan from task-master research
"""

import json
import os
import sys
import time
import pickle
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import uuid

class WorkspaceState(Enum):
    INITIALIZED = "initialized"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CHECKPOINT = "checkpoint"
    FAILED = "failed"
    COMPLETED = "completed"

@dataclass
class MemoryBlock:
    """Represents a reusable memory block in catalytic workspace"""
    id: str
    size: int
    content_hash: str
    last_used: float
    reuse_count: int
    data_type: str
    is_locked: bool = False

@dataclass
class CheckpointState:
    """Checkpoint state for recovery"""
    checkpoint_id: str
    timestamp: float
    workspace_state: Dict[str, Any]
    memory_blocks: List[MemoryBlock]
    task_execution_state: Dict[str, Any]
    integrity_hash: str

@dataclass
class WorkspaceConfig:
    """Configuration for catalytic workspace"""
    workspace_size: int = 10 * 1024 * 1024 * 1024  # 10GB default
    reuse_factor: float = 0.8
    checkpoint_interval: int = 300  # 5 minutes
    max_memory_blocks: int = 10000
    isolation_enabled: bool = True
    integrity_check_enabled: bool = True

class CatalyticWorkspaceSystem:
    """Main catalytic workspace implementation"""
    
    def __init__(self, workspace_dir: str = None, config: WorkspaceConfig = None):
        self.workspace_dir = Path(workspace_dir or ".taskmaster/catalytic-workspace")
        self.config = config or WorkspaceConfig()
        self.state = WorkspaceState.INITIALIZED
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        self.checkpoints: Dict[str, CheckpointState] = {}
        self.lock = threading.RLock()
        self.current_task_id: Optional[str] = None
        
        # Initialize workspace
        self._initialize_workspace()
        
    def _initialize_workspace(self):
        """Initialize the catalytic workspace"""
        print(f"🧪 Initializing Catalytic Workspace System...")
        print(f"📁 Workspace Directory: {self.workspace_dir}")
        print(f"💾 Workspace Size: {self.config.workspace_size / (1024**3):.1f}GB")
        print(f"♻️  Target Reuse Factor: {self.config.reuse_factor}")
        
        # Create workspace directories
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "memory-blocks").mkdir(exist_ok=True)
        (self.workspace_dir / "checkpoints").mkdir(exist_ok=True)
        (self.workspace_dir / "isolation").mkdir(exist_ok=True)
        (self.workspace_dir / "metadata").mkdir(exist_ok=True)
        
        # Load existing state if available
        self._load_workspace_state()
        
        self.state = WorkspaceState.ACTIVE
        print("✅ Catalytic workspace initialized successfully")
    
    def _load_workspace_state(self):
        """Load existing workspace state from disk"""
        state_file = self.workspace_dir / "metadata" / "workspace_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Restore memory blocks
                for block_data in state_data.get('memory_blocks', []):
                    block = MemoryBlock(**block_data)
                    self.memory_blocks[block.id] = block
                
                print(f"🔄 Restored {len(self.memory_blocks)} memory blocks from previous session")
                
            except Exception as e:
                print(f"⚠️  Failed to load workspace state: {e}")
    
    def _save_workspace_state(self):
        """Save current workspace state to disk"""
        state_file = self.workspace_dir / "metadata" / "workspace_state.json"
        
        state_data = {
            'memory_blocks': [asdict(block) for block in self.memory_blocks.values()],
            'state': self.state.value,
            'timestamp': time.time()
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def allocate_memory(self, size: int, data_type: str = "generic", 
                       task_id: str = None) -> Optional[str]:
        """Allocate memory with catalytic reuse capability"""
        with self.lock:
            # Try to find reusable memory block
            reusable_block = self._find_reusable_block(size, data_type)
            
            if reusable_block:
                # Reuse existing block
                reusable_block.last_used = time.time()
                reusable_block.reuse_count += 1
                reusable_block.is_locked = True
                
                print(f"♻️  Reusing memory block {reusable_block.id} "
                      f"(size: {size}, reuse count: {reusable_block.reuse_count})")
                
                return reusable_block.id
            
            else:
                # Create new memory block
                block_id = str(uuid.uuid4())
                content_hash = hashlib.sha256(f"{size}{data_type}{time.time()}".encode()).hexdigest()
                
                memory_block = MemoryBlock(
                    id=block_id,
                    size=size,
                    content_hash=content_hash,
                    last_used=time.time(),
                    reuse_count=0,
                    data_type=data_type,
                    is_locked=True
                )
                
                self.memory_blocks[block_id] = memory_block
                
                # Create physical memory block file
                block_file = self.workspace_dir / "memory-blocks" / f"{block_id}.block"
                with open(block_file, 'wb') as f:
                    f.write(b'\x00' * size)  # Initialize with zeros
                
                print(f"🆕 Created new memory block {block_id} (size: {size})")
                
                return block_id
    
    def _find_reusable_block(self, size: int, data_type: str) -> Optional[MemoryBlock]:
        """Find a suitable memory block for reuse"""
        best_block = None
        best_score = 0
        
        for block in self.memory_blocks.values():
            if block.is_locked:
                continue
                
            # Calculate reuse suitability score
            size_match = 1.0 if block.size >= size else 0.0
            type_match = 1.0 if block.data_type == data_type else 0.5
            recency = 1.0 / (1.0 + time.time() - block.last_used)
            reuse_benefit = min(block.reuse_count / 10.0, 0.8)  # Cap benefit
            
            score = size_match * type_match * recency * (1.0 + reuse_benefit)
            
            if score > best_score and score > 0.6:  # Minimum reuse threshold
                best_score = score
                best_block = block
        
        return best_block
    
    def release_memory(self, block_id: str) -> bool:
        """Release a memory block for potential reuse"""
        with self.lock:
            if block_id in self.memory_blocks:
                self.memory_blocks[block_id].is_locked = False
                print(f"🔓 Released memory block {block_id} for reuse")
                return True
            return False
    
    def create_checkpoint(self, task_execution_state: Dict[str, Any] = None) -> str:
        """Create a checkpoint for recovery"""
        checkpoint_id = f"checkpoint_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        print(f"📋 Creating checkpoint: {checkpoint_id}")
        
        with self.lock:
            # Prepare checkpoint state
            workspace_state = {
                'state': self.state.value,
                'current_task_id': self.current_task_id,
                'memory_blocks_count': len(self.memory_blocks),
                'workspace_config': asdict(self.config)
            }
            
            memory_blocks_copy = [
                MemoryBlock(
                    id=block.id,
                    size=block.size,
                    content_hash=block.content_hash,
                    last_used=block.last_used,
                    reuse_count=block.reuse_count,
                    data_type=block.data_type,
                    is_locked=False  # Unlock in checkpoint
                )
                for block in self.memory_blocks.values()
            ]
            
            # Create integrity hash
            checkpoint_data = {
                'workspace_state': workspace_state,
                'memory_blocks': [asdict(block) for block in memory_blocks_copy],
                'task_execution_state': task_execution_state or {}
            }
            
            integrity_hash = hashlib.sha256(
                json.dumps(checkpoint_data, sort_keys=True).encode()
            ).hexdigest()
            
            checkpoint = CheckpointState(
                checkpoint_id=checkpoint_id,
                timestamp=time.time(),
                workspace_state=workspace_state,
                memory_blocks=memory_blocks_copy,
                task_execution_state=task_execution_state or {},
                integrity_hash=integrity_hash
            )
            
            # Save checkpoint to disk
            checkpoint_file = self.workspace_dir / "checkpoints" / f"{checkpoint_id}.checkpoint"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            self.checkpoints[checkpoint_id] = checkpoint
            
            print(f"✅ Checkpoint {checkpoint_id} created successfully")
            print(f"💾 Saved {len(memory_blocks_copy)} memory blocks state")
            
            return checkpoint_id
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore workspace from a checkpoint"""
        checkpoint_file = self.workspace_dir / "checkpoints" / f"{checkpoint_id}.checkpoint"
        
        if not checkpoint_file.exists():
            print(f"❌ Checkpoint file not found: {checkpoint_id}")
            return False
        
        try:
            print(f"🔄 Restoring from checkpoint: {checkpoint_id}")
            
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Verify integrity
            checkpoint_data = {
                'workspace_state': checkpoint.workspace_state,
                'memory_blocks': [asdict(block) for block in checkpoint.memory_blocks],
                'task_execution_state': checkpoint.task_execution_state
            }
            
            calculated_hash = hashlib.sha256(
                json.dumps(checkpoint_data, sort_keys=True).encode()
            ).hexdigest()
            
            if calculated_hash != checkpoint.integrity_hash:
                print(f"❌ Checkpoint integrity check failed for {checkpoint_id}")
                return False
            
            with self.lock:
                # Restore state
                self.state = WorkspaceState(checkpoint.workspace_state['state'])
                self.current_task_id = checkpoint.workspace_state.get('current_task_id')
                
                # Restore memory blocks
                self.memory_blocks.clear()
                for block in checkpoint.memory_blocks:
                    self.memory_blocks[block.id] = block
                
                print(f"✅ Restored {len(self.memory_blocks)} memory blocks")
                print(f"📊 Workspace state: {self.state.value}")
                
                return True
                
        except Exception as e:
            print(f"❌ Failed to restore checkpoint {checkpoint_id}: {e}")
            return False
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int:
        """Clean up old checkpoints"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        checkpoint_dir = self.workspace_dir / "checkpoints"
        
        for checkpoint_file in checkpoint_dir.glob("*.checkpoint"):
            file_age = current_time - checkpoint_file.stat().st_mtime
            
            if file_age > max_age_seconds:
                checkpoint_file.unlink()
                cleaned_count += 1
        
        print(f"🧹 Cleaned up {cleaned_count} old checkpoints")
        return cleaned_count
    
    def get_memory_utilization_stats(self) -> Dict[str, Any]:
        """Get memory utilization statistics"""
        with self.lock:
            total_blocks = len(self.memory_blocks)
            locked_blocks = sum(1 for block in self.memory_blocks.values() if block.is_locked)
            total_size = sum(block.size for block in self.memory_blocks.values())
            reused_blocks = sum(1 for block in self.memory_blocks.values() if block.reuse_count > 0)
            total_reuses = sum(block.reuse_count for block in self.memory_blocks.values())
            
            reuse_efficiency = (total_reuses / total_blocks) if total_blocks > 0 else 0
            utilization_ratio = total_size / self.config.workspace_size
            
            return {
                'total_blocks': total_blocks,
                'locked_blocks': locked_blocks,
                'available_blocks': total_blocks - locked_blocks,
                'total_size_mb': total_size / (1024 * 1024),
                'workspace_size_mb': self.config.workspace_size / (1024 * 1024),
                'utilization_ratio': utilization_ratio,
                'reused_blocks': reused_blocks,
                'total_reuses': total_reuses,
                'reuse_efficiency': reuse_efficiency,
                'checkpoints_count': len(self.checkpoints)
            }
    
    def run_integrity_check(self) -> bool:
        """Run comprehensive integrity check"""
        print("🔍 Running workspace integrity check...")
        
        issues_found = 0
        
        with self.lock:
            # Check memory block files
            for block_id, block in self.memory_blocks.items():
                block_file = self.workspace_dir / "memory-blocks" / f"{block_id}.block"
                
                if not block_file.exists():
                    print(f"❌ Missing memory block file: {block_id}")
                    issues_found += 1
                    continue
                
                # Check file size
                actual_size = block_file.stat().st_size
                if actual_size != block.size:
                    print(f"❌ Size mismatch for block {block_id}: "
                          f"expected {block.size}, actual {actual_size}")
                    issues_found += 1
            
            # Check checkpoint integrity
            for checkpoint_id in self.checkpoints:
                checkpoint_file = self.workspace_dir / "checkpoints" / f"{checkpoint_id}.checkpoint"
                if not checkpoint_file.exists():
                    print(f"❌ Missing checkpoint file: {checkpoint_id}")
                    issues_found += 1
        
        if issues_found == 0:
            print("✅ Workspace integrity check passed")
            return True
        else:
            print(f"❌ Found {issues_found} integrity issues")
            return False
    
    def simulate_catalytic_execution(self, task_count: int = 10) -> Dict[str, Any]:
        """Simulate catalytic execution for testing"""
        print(f"🧪 Simulating catalytic execution with {task_count} tasks...")
        
        execution_stats = {
            'tasks_executed': 0,
            'memory_allocations': 0,
            'memory_reuses': 0,
            'checkpoints_created': 0,
            'total_execution_time': 0
        }
        
        start_time = time.time()
        
        for i in range(task_count):
            task_id = f"task_{i+1}"
            print(f"📋 Executing {task_id}...")
            
            # Simulate task execution with memory allocation
            memory_size = 1024 * 1024 * (1 + i % 5)  # 1-5MB per task
            data_type = ["optimization", "analysis", "validation", "generation"][i % 4]
            
            # Allocate memory
            block_id = self.allocate_memory(memory_size, data_type, task_id)
            if block_id:
                execution_stats['memory_allocations'] += 1
                
                # Check if this was a reuse
                if self.memory_blocks[block_id].reuse_count > 0:
                    execution_stats['memory_reuses'] += 1
            
            # Simulate task work
            time.sleep(0.1)
            
            # Release memory
            if block_id:
                self.release_memory(block_id)
            
            # Create checkpoint every 3 tasks
            if (i + 1) % 3 == 0:
                checkpoint_id = self.create_checkpoint({'task_id': task_id, 'progress': i+1})
                execution_stats['checkpoints_created'] += 1
            
            execution_stats['tasks_executed'] += 1
        
        execution_stats['total_execution_time'] = time.time() - start_time
        execution_stats['reuse_ratio'] = (
            execution_stats['memory_reuses'] / execution_stats['memory_allocations']
            if execution_stats['memory_allocations'] > 0 else 0
        )
        
        print("🎉 Catalytic execution simulation complete!")
        return execution_stats
    
    def shutdown(self):
        """Shutdown workspace and save state"""
        print("🔄 Shutting down catalytic workspace...")
        
        with self.lock:
            # Save current state
            self._save_workspace_state()
            
            # Release all locked memory blocks
            for block in self.memory_blocks.values():
                block.is_locked = False
            
            self.state = WorkspaceState.SUSPENDED
            
        print("✅ Catalytic workspace shutdown complete")

def main():
    """Main execution function for testing"""
    print("🚀 Starting Catalytic Workspace System Test")
    print("=" * 60)
    
    # Initialize workspace
    workspace = CatalyticWorkspaceSystem()
    
    try:
        # Run simulation
        stats = workspace.simulate_catalytic_execution(10)
        
        # Display results
        print("\n📊 EXECUTION STATISTICS:")
        print("-" * 40)
        print(f"Tasks Executed: {stats['tasks_executed']}")
        print(f"Memory Allocations: {stats['memory_allocations']}")
        print(f"Memory Reuses: {stats['memory_reuses']}")
        print(f"Reuse Ratio: {stats['reuse_ratio']:.1%}")
        print(f"Checkpoints Created: {stats['checkpoints_created']}")
        print(f"Execution Time: {stats['total_execution_time']:.2f}s")
        
        # Get utilization stats
        util_stats = workspace.get_memory_utilization_stats()
        print(f"\n💾 MEMORY UTILIZATION:")
        print("-" * 40)
        print(f"Total Memory Blocks: {util_stats['total_blocks']}")
        print(f"Reuse Efficiency: {util_stats['reuse_efficiency']:.2f}")
        print(f"Memory Usage: {util_stats['total_size_mb']:.1f}MB / "
              f"{util_stats['workspace_size_mb']:.1f}MB "
              f"({util_stats['utilization_ratio']:.1%})")
        
        # Run integrity check
        print(f"\n🔍 INTEGRITY CHECK:")
        print("-" * 40)
        integrity_ok = workspace.run_integrity_check()
        
        # Test checkpoint/resume
        print(f"\n📋 CHECKPOINT/RESUME TEST:")
        print("-" * 40)
        checkpoint_id = workspace.create_checkpoint({'test': 'final_checkpoint'})
        print(f"Created checkpoint: {checkpoint_id}")
        
        # Simulate some changes
        workspace.allocate_memory(1024*1024, "test")
        
        # Restore checkpoint
        if workspace.restore_from_checkpoint(checkpoint_id):
            print("✅ Checkpoint/resume test passed")
        else:
            print("❌ Checkpoint/resume test failed")
        
        print(f"\n🎉 CATALYTIC WORKSPACE SYSTEM TEST COMPLETE!")
        print(f"Status: {'✅ SUCCESS' if integrity_ok else '❌ ISSUES FOUND'}")
        
        return 0 if integrity_ok else 1
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return 1
        
    finally:
        workspace.shutdown()

if __name__ == "__main__":
    sys.exit(main())