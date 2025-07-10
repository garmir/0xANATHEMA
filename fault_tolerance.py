#!/usr/bin/env python3
"""
Fault Tolerance and Resilience Features
Atomic Task 49.4: Integrate Fault Tolerance and Resilience Features

This module implements comprehensive fault tolerance mechanisms including agent health
monitoring, retry logic, state checkpointing, and automated recovery procedures.
"""

import asyncio
import json
import logging
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from pathlib import Path
import hashlib
import threading
from collections import defaultdict, deque


class HealthStatus(Enum):
    """Agent health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    RESTART = "restart"
    FAILOVER = "failover"
    CIRCUIT_BREAKER = "circuit_breaker"
    BACKOFF = "backoff"
    MANUAL = "manual"


class CheckpointType(Enum):
    """Types of checkpoints"""
    TASK_STATE = "task_state"
    AGENT_STATE = "agent_state"
    WORKFLOW_STATE = "workflow_state"
    SYSTEM_STATE = "system_state"


@dataclass
class HealthMetrics:
    """Agent health metrics"""
    agent_id: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_heartbeat: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    task_completion_rate: float = 1.0
    uptime: float = 0.0
    consecutive_failures: int = 0
    recovery_attempts: int = 0
    last_recovery: Optional[datetime] = None
    
    def update_metrics(self, **kwargs):
        """Update health metrics"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_heartbeat = datetime.now()
    
    def calculate_health_score(self) -> float:
        """Calculate overall health score (0.0 - 1.0)"""
        if self.status == HealthStatus.FAILED:
            return 0.0
        
        # Weight factors for different metrics
        weights = {
            "response_time": 0.25,  # Lower is better
            "cpu_usage": 0.15,      # Lower is better  
            "memory_usage": 0.15,   # Lower is better
            "error_rate": 0.25,     # Lower is better
            "completion_rate": 0.20  # Higher is better
        }
        
        # Normalize metrics (0-1 where 1 is best)
        response_score = max(0, 1.0 - min(self.response_time / 10.0, 1.0))
        cpu_score = max(0, 1.0 - self.cpu_usage)
        memory_score = max(0, 1.0 - self.memory_usage)
        error_score = max(0, 1.0 - self.error_rate)
        completion_score = self.task_completion_rate
        
        # Calculate weighted score
        health_score = (
            response_score * weights["response_time"] +
            cpu_score * weights["cpu_usage"] +
            memory_score * weights["memory_usage"] +
            error_score * weights["error_rate"] +
            completion_score * weights["completion_rate"]
        )
        
        # Apply penalty for consecutive failures
        failure_penalty = min(self.consecutive_failures * 0.1, 0.5)
        health_score = max(0, health_score - failure_penalty)
        
        return health_score


@dataclass
class Checkpoint:
    """State checkpoint for recovery"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    checkpoint_type: CheckpointType = CheckpointType.TASK_STATE
    entity_id: str = ""  # Agent ID, task ID, etc.
    state_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for integrity verification"""
        data_str = json.dumps(self.state_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify checkpoint integrity"""
        return self.checksum == self._calculate_checksum()


@dataclass
class RecoveryPolicy:
    """Recovery policy configuration"""
    strategy: RecoveryStrategy = RecoveryStrategy.RESTART
    max_attempts: int = 3
    retry_intervals: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])
    health_threshold: float = 0.5
    escalation_timeout: float = 300.0  # 5 minutes
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    enable_checkpointing: bool = True
    checkpoint_interval: float = 30.0
    
    def get_retry_interval(self, attempt: int) -> float:
        """Get retry interval for given attempt"""
        if attempt < len(self.retry_intervals):
            return self.retry_intervals[attempt]
        return self.retry_intervals[-1]


class HealthMonitor:
    """Agent health monitoring system"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.agent_metrics: Dict[str, HealthMetrics] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.health_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.is_running = False
        self.logger = logging.getLogger("HealthMonitor")
        
    def register_agent(self, agent_id: str):
        """Register agent for health monitoring"""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = HealthMetrics(agent_id=agent_id)
            self.logger.info(f"Registered agent for health monitoring: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister agent from health monitoring"""
        if agent_id in self.agent_metrics:
            del self.agent_metrics[agent_id]
            
        if agent_id in self.monitoring_tasks:
            self.monitoring_tasks[agent_id].cancel()
            del self.monitoring_tasks[agent_id]
            
        self.logger.info(f"Unregistered agent from health monitoring: {agent_id}")
    
    def add_health_callback(self, agent_id: str, callback: Callable):
        """Add callback for health status changes"""
        self.health_callbacks[agent_id].append(callback)
    
    async def start_monitoring(self):
        """Start health monitoring for all registered agents"""
        self.is_running = True
        self.logger.info("Starting health monitoring")
        
        for agent_id in self.agent_metrics:
            task = asyncio.create_task(self._monitor_agent(agent_id))
            self.monitoring_tasks[agent_id] = task
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_running = False
        
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        self.monitoring_tasks.clear()
        
        self.logger.info("Health monitoring stopped")
    
    async def check_agent_health(self, agent_id: str) -> HealthStatus:
        """Perform immediate health check for agent"""
        if agent_id not in self.agent_metrics:
            return HealthStatus.UNKNOWN
        
        metrics = self.agent_metrics[agent_id]
        
        try:
            # Simulate health check (in real implementation, would ping agent)
            start_time = time.time()
            
            # Here you would send a health check message to the agent
            # For demo, we'll simulate based on stored metrics
            await asyncio.sleep(0.1)  # Simulate network delay
            
            response_time = time.time() - start_time
            
            # Update metrics
            metrics.update_metrics(
                response_time=response_time,
                last_heartbeat=datetime.now()
            )
            
            # Determine health status based on metrics
            health_score = metrics.calculate_health_score()
            
            if health_score >= 0.8:
                new_status = HealthStatus.HEALTHY
            elif health_score >= 0.6:
                new_status = HealthStatus.DEGRADED
            elif health_score >= 0.3:
                new_status = HealthStatus.UNHEALTHY
            else:
                new_status = HealthStatus.FAILED
            
            # Check for timeout
            time_since_heartbeat = (datetime.now() - metrics.last_heartbeat).total_seconds()
            if time_since_heartbeat > 60.0:  # 1 minute timeout
                new_status = HealthStatus.FAILED
            
            # Update status and notify callbacks if changed
            if metrics.status != new_status:
                old_status = metrics.status
                metrics.status = new_status
                await self._notify_health_change(agent_id, old_status, new_status)
            
            return new_status
            
        except Exception as e:
            self.logger.error(f"Health check failed for {agent_id}: {e}")
            metrics.consecutive_failures += 1
            metrics.status = HealthStatus.FAILED
            return HealthStatus.FAILED
    
    async def _monitor_agent(self, agent_id: str):
        """Continuous monitoring for a specific agent"""
        while self.is_running:
            try:
                await self.check_agent_health(agent_id)
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error monitoring agent {agent_id}: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _notify_health_change(self, agent_id: str, old_status: HealthStatus, new_status: HealthStatus):
        """Notify callbacks of health status change"""
        self.logger.info(f"Agent {agent_id} health changed: {old_status.value} -> {new_status.value}")
        
        for callback in self.health_callbacks[agent_id]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, old_status, new_status)
                else:
                    callback(agent_id, old_status, new_status)
            except Exception as e:
                self.logger.error(f"Error in health callback: {e}")
    
    def get_agent_health(self, agent_id: str) -> Optional[HealthMetrics]:
        """Get current health metrics for agent"""
        return self.agent_metrics.get(agent_id)
    
    def get_unhealthy_agents(self) -> List[str]:
        """Get list of unhealthy agents"""
        return [
            agent_id for agent_id, metrics in self.agent_metrics.items()
            if metrics.status in [HealthStatus.UNHEALTHY, HealthStatus.FAILED]
        ]


class CheckpointManager:
    """State checkpoint management for recovery"""
    
    def __init__(self, checkpoint_dir: str = ".taskmaster/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.auto_checkpoint_tasks: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger("CheckpointManager")
        
    async def create_checkpoint(self, checkpoint_type: CheckpointType, entity_id: str, 
                              state_data: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Create a checkpoint"""
        checkpoint = Checkpoint(
            checkpoint_type=checkpoint_type,
            entity_id=entity_id,
            state_data=state_data,
            metadata=metadata or {}
        )
        
        # Store in memory
        self.checkpoints[checkpoint.id] = checkpoint
        
        # Persist to disk
        await self._persist_checkpoint(checkpoint)
        
        self.logger.info(f"Created checkpoint {checkpoint.id} for {entity_id}")
        return checkpoint.id
    
    async def restore_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Restore from checkpoint"""
        # Try memory first
        if checkpoint_id in self.checkpoints:
            checkpoint = self.checkpoints[checkpoint_id]
            if checkpoint.verify_integrity():
                self.logger.info(f"Restored checkpoint {checkpoint_id} from memory")
                return checkpoint
            else:
                self.logger.error(f"Checkpoint {checkpoint_id} failed integrity check")
                return None
        
        # Try disk
        checkpoint = await self._load_checkpoint(checkpoint_id)
        if checkpoint and checkpoint.verify_integrity():
            self.checkpoints[checkpoint_id] = checkpoint
            self.logger.info(f"Restored checkpoint {checkpoint_id} from disk")
            return checkpoint
        
        return None
    
    async def get_latest_checkpoint(self, entity_id: str, checkpoint_type: CheckpointType = None) -> Optional[Checkpoint]:
        """Get latest checkpoint for entity"""
        matching_checkpoints = []
        
        for checkpoint in self.checkpoints.values():
            if checkpoint.entity_id == entity_id:
                if checkpoint_type is None or checkpoint.checkpoint_type == checkpoint_type:
                    matching_checkpoints.append(checkpoint)
        
        if not matching_checkpoints:
            # Try loading from disk
            await self._load_entity_checkpoints(entity_id)
            matching_checkpoints = [
                cp for cp in self.checkpoints.values()
                if cp.entity_id == entity_id and (checkpoint_type is None or cp.checkpoint_type == checkpoint_type)
            ]
        
        if matching_checkpoints:
            return max(matching_checkpoints, key=lambda cp: cp.timestamp)
        
        return None
    
    async def start_auto_checkpoint(self, entity_id: str, checkpoint_type: CheckpointType,
                                  interval: float, state_provider: Callable):
        """Start automatic checkpointing for entity"""
        if entity_id in self.auto_checkpoint_tasks:
            self.auto_checkpoint_tasks[entity_id].cancel()
        
        task = asyncio.create_task(
            self._auto_checkpoint_loop(entity_id, checkpoint_type, interval, state_provider)
        )
        self.auto_checkpoint_tasks[entity_id] = task
        
        self.logger.info(f"Started auto-checkpointing for {entity_id} every {interval}s")
    
    async def stop_auto_checkpoint(self, entity_id: str):
        """Stop automatic checkpointing for entity"""
        if entity_id in self.auto_checkpoint_tasks:
            self.auto_checkpoint_tasks[entity_id].cancel()
            del self.auto_checkpoint_tasks[entity_id]
            self.logger.info(f"Stopped auto-checkpointing for {entity_id}")
    
    async def _auto_checkpoint_loop(self, entity_id: str, checkpoint_type: CheckpointType,
                                  interval: float, state_provider: Callable):
        """Automatic checkpointing loop"""
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Get current state
                if asyncio.iscoroutinefunction(state_provider):
                    state_data = await state_provider()
                else:
                    state_data = state_provider()
                
                # Create checkpoint
                await self.create_checkpoint(checkpoint_type, entity_id, state_data)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in auto-checkpoint for {entity_id}: {e}")
    
    async def _persist_checkpoint(self, checkpoint: Checkpoint):
        """Persist checkpoint to disk"""
        try:
            file_path = self.checkpoint_dir / f"{checkpoint.id}.pkl"
            
            with open(file_path, 'wb') as f:
                pickle.dump(checkpoint, f)
                
        except Exception as e:
            self.logger.error(f"Failed to persist checkpoint {checkpoint.id}: {e}")
    
    async def _load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load checkpoint from disk"""
        try:
            file_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
        
        return None
    
    async def _load_entity_checkpoints(self, entity_id: str):
        """Load all checkpoints for entity from disk"""
        try:
            for file_path in self.checkpoint_dir.glob("*.pkl"):
                try:
                    with open(file_path, 'rb') as f:
                        checkpoint = pickle.load(f)
                        if checkpoint.entity_id == entity_id:
                            self.checkpoints[checkpoint.id] = checkpoint
                except Exception as e:
                    self.logger.warning(f"Failed to load checkpoint from {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error loading entity checkpoints for {entity_id}: {e}")


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.logger = logging.getLogger("CircuitBreaker")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                self.logger.info("Circuit breaker entering half-open state")
            else:
                raise RuntimeError("Circuit breaker is open")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "half-open":
                self._reset()
                self.logger.info("Circuit breaker reset to closed state")
            
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.timeout
    
    def _record_failure(self):
        """Record a failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _reset(self):
        """Reset circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"


class RecoveryManager:
    """Automated recovery management"""
    
    def __init__(self, health_monitor: HealthMonitor, checkpoint_manager: CheckpointManager):
        self.health_monitor = health_monitor
        self.checkpoint_manager = checkpoint_manager
        self.recovery_policies: Dict[str, RecoveryPolicy] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        self.logger = logging.getLogger("RecoveryManager")
        
    def set_recovery_policy(self, entity_id: str, policy: RecoveryPolicy):
        """Set recovery policy for entity"""
        self.recovery_policies[entity_id] = policy
        
        if policy.enable_circuit_breaker:
            self.circuit_breakers[entity_id] = CircuitBreaker(
                policy.circuit_breaker_threshold,
                policy.circuit_breaker_timeout
            )
        
        self.logger.info(f"Set recovery policy for {entity_id}: {policy.strategy.value}")
    
    async def start_recovery_monitoring(self):
        """Start recovery monitoring"""
        self.is_running = True
        
        # Add health callbacks for all entities with recovery policies
        for entity_id in self.recovery_policies:
            self.health_monitor.add_health_callback(entity_id, self._handle_health_change)
        
        self.logger.info("Recovery monitoring started")
    
    async def stop_recovery_monitoring(self):
        """Stop recovery monitoring"""
        self.is_running = False
        
        for task in self.recovery_tasks.values():
            task.cancel()
        
        await asyncio.gather(*self.recovery_tasks.values(), return_exceptions=True)
        self.recovery_tasks.clear()
        
        self.logger.info("Recovery monitoring stopped")
    
    async def _handle_health_change(self, agent_id: str, old_status: HealthStatus, new_status: HealthStatus):
        """Handle agent health status change"""
        if new_status in [HealthStatus.UNHEALTHY, HealthStatus.FAILED]:
            await self._initiate_recovery(agent_id)
    
    async def _initiate_recovery(self, entity_id: str):
        """Initiate recovery for entity"""
        if entity_id not in self.recovery_policies:
            self.logger.warning(f"No recovery policy for {entity_id}")
            return
        
        if entity_id in self.recovery_tasks:
            self.logger.info(f"Recovery already in progress for {entity_id}")
            return
        
        policy = self.recovery_policies[entity_id]
        task = asyncio.create_task(self._recovery_loop(entity_id, policy))
        self.recovery_tasks[entity_id] = task
        
        self.logger.info(f"Initiated recovery for {entity_id}")
    
    async def _recovery_loop(self, entity_id: str, policy: RecoveryPolicy):
        """Recovery loop for entity"""
        try:
            metrics = self.health_monitor.get_agent_health(entity_id)
            if metrics:
                metrics.status = HealthStatus.RECOVERING
            
            for attempt in range(policy.max_attempts):
                self.logger.info(f"Recovery attempt {attempt + 1}/{policy.max_attempts} for {entity_id}")
                
                try:
                    # Perform recovery based on strategy
                    success = await self._perform_recovery(entity_id, policy, attempt)
                    
                    if success:
                        self.logger.info(f"Recovery successful for {entity_id}")
                        if metrics:
                            metrics.recovery_attempts = attempt + 1
                            metrics.last_recovery = datetime.now()
                        return
                    
                except Exception as e:
                    self.logger.error(f"Recovery attempt {attempt + 1} failed for {entity_id}: {e}")
                
                # Wait before next attempt
                if attempt < policy.max_attempts - 1:
                    wait_time = policy.get_retry_interval(attempt)
                    await asyncio.sleep(wait_time)
            
            # All recovery attempts failed
            self.logger.error(f"Recovery failed for {entity_id} after {policy.max_attempts} attempts")
            if metrics:
                metrics.status = HealthStatus.FAILED
                
        except asyncio.CancelledError:
            self.logger.info(f"Recovery cancelled for {entity_id}")
        except Exception as e:
            self.logger.error(f"Error in recovery loop for {entity_id}: {e}")
        finally:
            if entity_id in self.recovery_tasks:
                del self.recovery_tasks[entity_id]
    
    async def _perform_recovery(self, entity_id: str, policy: RecoveryPolicy, attempt: int) -> bool:
        """Perform recovery action"""
        if policy.strategy == RecoveryStrategy.RESTART:
            return await self._restart_recovery(entity_id, policy)
        elif policy.strategy == RecoveryStrategy.FAILOVER:
            return await self._failover_recovery(entity_id, policy)
        elif policy.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return await self._circuit_breaker_recovery(entity_id, policy)
        elif policy.strategy == RecoveryStrategy.BACKOFF:
            return await self._backoff_recovery(entity_id, policy, attempt)
        else:
            self.logger.warning(f"Unknown recovery strategy: {policy.strategy}")
            return False
    
    async def _restart_recovery(self, entity_id: str, policy: RecoveryPolicy) -> bool:
        """Restart-based recovery"""
        try:
            # Restore from latest checkpoint if available
            if policy.enable_checkpointing:
                checkpoint = await self.checkpoint_manager.get_latest_checkpoint(entity_id)
                if checkpoint:
                    self.logger.info(f"Restoring {entity_id} from checkpoint {checkpoint.id}")
                    # In real implementation, would restore agent state
            
            # Simulate restart (in real implementation, would restart agent process)
            await asyncio.sleep(1.0)
            
            # Verify recovery
            await asyncio.sleep(0.5)
            health_status = await self.health_monitor.check_agent_health(entity_id)
            
            return health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            
        except Exception as e:
            self.logger.error(f"Restart recovery failed for {entity_id}: {e}")
            return False
    
    async def _failover_recovery(self, entity_id: str, policy: RecoveryPolicy) -> bool:
        """Failover-based recovery"""
        try:
            # In real implementation, would redirect traffic to backup agent
            self.logger.info(f"Performing failover for {entity_id}")
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            self.logger.error(f"Failover recovery failed for {entity_id}: {e}")
            return False
    
    async def _circuit_breaker_recovery(self, entity_id: str, policy: RecoveryPolicy) -> bool:
        """Circuit breaker recovery"""
        try:
            if entity_id in self.circuit_breakers:
                breaker = self.circuit_breakers[entity_id]
                # Reset circuit breaker
                breaker._reset()
                self.logger.info(f"Reset circuit breaker for {entity_id}")
                return True
        except Exception as e:
            self.logger.error(f"Circuit breaker recovery failed for {entity_id}: {e}")
        return False
    
    async def _backoff_recovery(self, entity_id: str, policy: RecoveryPolicy, attempt: int) -> bool:
        """Exponential backoff recovery"""
        try:
            backoff_time = policy.get_retry_interval(attempt)
            self.logger.info(f"Backoff recovery for {entity_id}: waiting {backoff_time}s")
            await asyncio.sleep(backoff_time)
            
            health_status = await self.health_monitor.check_agent_health(entity_id)
            return health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            
        except Exception as e:
            self.logger.error(f"Backoff recovery failed for {entity_id}: {e}")
            return False


class FaultToleranceSystem:
    """Main fault tolerance system coordinator"""
    
    def __init__(self, checkpoint_dir: str = ".taskmaster/checkpoints"):
        self.health_monitor = HealthMonitor()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.recovery_manager = RecoveryManager(self.health_monitor, self.checkpoint_manager)
        self.is_running = False
        self.logger = logging.getLogger("FaultToleranceSystem")
    
    async def start(self):
        """Start fault tolerance system"""
        self.is_running = True
        
        await self.health_monitor.start_monitoring()
        await self.recovery_manager.start_recovery_monitoring()
        
        self.logger.info("Fault tolerance system started")
    
    async def stop(self):
        """Stop fault tolerance system"""
        self.is_running = False
        
        await self.recovery_manager.stop_recovery_monitoring()
        await self.health_monitor.stop_monitoring()
        
        self.logger.info("Fault tolerance system stopped")
    
    def register_agent(self, agent_id: str, recovery_policy: RecoveryPolicy = None):
        """Register agent with fault tolerance"""
        self.health_monitor.register_agent(agent_id)
        
        if recovery_policy:
            self.recovery_manager.set_recovery_policy(agent_id, recovery_policy)
        else:
            # Default policy
            default_policy = RecoveryPolicy()
            self.recovery_manager.set_recovery_policy(agent_id, default_policy)
        
        self.logger.info(f"Registered agent with fault tolerance: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister agent from fault tolerance"""
        self.health_monitor.unregister_agent(agent_id)
        self.logger.info(f"Unregistered agent from fault tolerance: {agent_id}")
    
    async def create_checkpoint(self, entity_id: str, state_data: Dict[str, Any], 
                               checkpoint_type: CheckpointType = CheckpointType.AGENT_STATE) -> str:
        """Create checkpoint for entity"""
        return await self.checkpoint_manager.create_checkpoint(checkpoint_type, entity_id, state_data)
    
    async def restore_from_checkpoint(self, entity_id: str, 
                                    checkpoint_type: CheckpointType = CheckpointType.AGENT_STATE) -> Optional[Dict[str, Any]]:
        """Restore entity from latest checkpoint"""
        checkpoint = await self.checkpoint_manager.get_latest_checkpoint(entity_id, checkpoint_type)
        if checkpoint:
            return checkpoint.state_data
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        unhealthy_agents = self.health_monitor.get_unhealthy_agents()
        
        agent_health = {}
        for agent_id, metrics in self.health_monitor.agent_metrics.items():
            agent_health[agent_id] = {
                "status": metrics.status.value,
                "health_score": metrics.calculate_health_score(),
                "consecutive_failures": metrics.consecutive_failures,
                "recovery_attempts": metrics.recovery_attempts
            }
        
        return {
            "is_running": self.is_running,
            "total_agents": len(self.health_monitor.agent_metrics),
            "unhealthy_agents": len(unhealthy_agents),
            "active_recoveries": len(self.recovery_manager.recovery_tasks),
            "agent_health": agent_health,
            "total_checkpoints": len(self.checkpoint_manager.checkpoints)
        }


# Export key classes
__all__ = [
    "HealthStatus", "RecoveryStrategy", "CheckpointType", "HealthMetrics",
    "Checkpoint", "RecoveryPolicy", "HealthMonitor", "CheckpointManager",
    "CircuitBreaker", "RecoveryManager", "FaultToleranceSystem"
]


if __name__ == "__main__":
    # Demo usage
    async def demo():
        logging.basicConfig(level=logging.INFO)
        
        # Create fault tolerance system
        ft_system = FaultToleranceSystem()
        
        # Start system
        await ft_system.start()
        
        # Register test agents
        agents = ["research-agent-1", "execution-agent-1", "validation-agent-1"]
        
        for agent_id in agents:
            policy = RecoveryPolicy(
                strategy=RecoveryStrategy.RESTART,
                max_attempts=3,
                enable_checkpointing=True
            )
            ft_system.register_agent(agent_id, policy)
        
        print(f"🛡️ Fault Tolerance System Demo")
        print(f"Registered {len(agents)} agents with fault tolerance")
        
        # Create some checkpoints
        for agent_id in agents:
            state = {"status": "active", "tasks": [], "last_update": time.time()}
            checkpoint_id = await ft_system.create_checkpoint(agent_id, state)
            print(f"Created checkpoint {checkpoint_id[:8]} for {agent_id}")
        
        # Simulate health monitoring
        print("\n🏥 Health Monitoring Results:")
        for agent_id in agents:
            health = await ft_system.health_monitor.check_agent_health(agent_id)
            print(f"  {agent_id}: {health.value}")
        
        # Get system status
        status = ft_system.get_system_status()
        print(f"\n📊 System Status:")
        print(f"  Total Agents: {status['total_agents']}")
        print(f"  Unhealthy Agents: {status['unhealthy_agents']}")
        print(f"  Active Recoveries: {status['active_recoveries']}")
        print(f"  Total Checkpoints: {status['total_checkpoints']}")
        
        # Stop system
        await ft_system.stop()
    
    # Run demo
    asyncio.run(demo())