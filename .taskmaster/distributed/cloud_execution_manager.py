#!/usr/bin/env python3
"""
Distributed Cloud Execution Manager
Coordinates task execution across multiple cloud providers with intelligent load balancing
"""

import json
import asyncio
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
from concurrent.futures import ThreadPoolExecutor
import threading

class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    GCP = "gcp" 
    AZURE = "azure"
    DIGITAL_OCEAN = "digitalocean"
    LINODE = "linode"
    LOCAL = "local"

class ExecutionStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class CloudResource:
    """Cloud resource configuration"""
    provider: CloudProvider
    region: str
    instance_type: str
    max_concurrent_tasks: int
    cost_per_hour: float
    performance_score: float
    availability_zones: List[str]
    is_active: bool = True

@dataclass
class TaskExecution:
    """Individual task execution"""
    task_id: str
    execution_id: str
    provider: CloudProvider
    region: str
    status: ExecutionStatus
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    resource_usage: Dict[str, float]
    cost_estimate: float

@dataclass
class DistributionStrategy:
    """Task distribution strategy"""
    strategy_name: str
    load_balancing_algorithm: str
    failover_enabled: bool
    cost_optimization: bool
    performance_priority: bool
    geographic_distribution: bool
    max_providers_per_task: int

class CloudExecutionManager:
    """Manages distributed task execution across multiple cloud providers"""
    
    def __init__(self, config_dir: str = '.taskmaster/distributed'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration files
        self.providers_config = self.config_dir / 'cloud_providers.json'
        self.strategies_config = self.config_dir / 'distribution_strategies.json'
        self.executions_log = self.config_dir / 'executions.json'
        
        # Runtime state
        self.cloud_resources: Dict[str, CloudResource] = {}
        self.active_executions: Dict[str, TaskExecution] = {}
        self.execution_queue: List[Dict[str, Any]] = []
        self.distribution_strategies: Dict[str, DistributionStrategy] = {}
        
        # Performance tracking
        self.provider_performance: Dict[CloudProvider, Dict[str, float]] = {}
        self.cost_tracking: Dict[CloudProvider, float] = {}
        
        # Concurrency control
        self.max_global_concurrent = 50
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.execution_lock = threading.Lock()
        
        self.initialize_cloud_infrastructure()
    
    def initialize_cloud_infrastructure(self):
        """Initialize cloud provider configurations"""
        
        # Define default cloud resources
        default_resources = [
            CloudResource(
                provider=CloudProvider.AWS,
                region="us-east-1",
                instance_type="t3.medium",
                max_concurrent_tasks=5,
                cost_per_hour=0.0416,
                performance_score=0.85,
                availability_zones=["us-east-1a", "us-east-1b", "us-east-1c"]
            ),
            CloudResource(
                provider=CloudProvider.GCP,
                region="us-central1",
                instance_type="e2-standard-2",
                max_concurrent_tasks=4,
                cost_per_hour=0.0335,
                performance_score=0.80,
                availability_zones=["us-central1-a", "us-central1-b", "us-central1-c"]
            ),
            CloudResource(
                provider=CloudProvider.AZURE,
                region="eastus",
                instance_type="Standard_B2s",
                max_concurrent_tasks=4,
                cost_per_hour=0.0408,
                performance_score=0.82,
                availability_zones=["eastus-1", "eastus-2", "eastus-3"]
            ),
            CloudResource(
                provider=CloudProvider.DIGITAL_OCEAN,
                region="nyc1",
                instance_type="s-2vcpu-4gb",
                max_concurrent_tasks=3,
                cost_per_hour=0.024,
                performance_score=0.75,
                availability_zones=["nyc1"]
            ),
            CloudResource(
                provider=CloudProvider.LOCAL,
                region="local",
                instance_type="local-machine",
                max_concurrent_tasks=2,
                cost_per_hour=0.0,
                performance_score=0.90,
                availability_zones=["local"]
            )
        ]
        
        # Store resources
        for resource in default_resources:
            resource_id = f"{resource.provider.value}_{resource.region}"
            self.cloud_resources[resource_id] = resource
        
        # Define distribution strategies
        self.distribution_strategies = {
            "cost_optimized": DistributionStrategy(
                strategy_name="cost_optimized",
                load_balancing_algorithm="least_cost",
                failover_enabled=True,
                cost_optimization=True,
                performance_priority=False,
                geographic_distribution=False,
                max_providers_per_task=1
            ),
            "performance_first": DistributionStrategy(
                strategy_name="performance_first",
                load_balancing_algorithm="highest_performance",
                failover_enabled=True,
                cost_optimization=False,
                performance_priority=True,
                geographic_distribution=False,
                max_providers_per_task=1
            ),
            "balanced": DistributionStrategy(
                strategy_name="balanced",
                load_balancing_algorithm="weighted_round_robin",
                failover_enabled=True,
                cost_optimization=True,
                performance_priority=True,
                geographic_distribution=True,
                max_providers_per_task=2
            ),
            "high_availability": DistributionStrategy(
                strategy_name="high_availability",
                load_balancing_algorithm="geographic_distribution",
                failover_enabled=True,
                cost_optimization=False,
                performance_priority=False,
                geographic_distribution=True,
                max_providers_per_task=3
            )
        }
        
        self.save_configuration()
        print(f"✅ Initialized {len(self.cloud_resources)} cloud resources with {len(self.distribution_strategies)} strategies")
    
    async def execute_task_distributed(self, task_data: Dict[str, Any], 
                                     strategy_name: str = "balanced") -> TaskExecution:
        """Execute a task using distributed cloud infrastructure"""
        
        task_id = task_data.get('id', f"task_{int(time.time())}")
        execution_id = f"exec_{task_id}_{int(time.time())}"
        
        print(f"🚀 Starting distributed execution for task {task_id}")
        
        # Select optimal cloud provider
        selected_resource = await self._select_optimal_provider(task_data, strategy_name)
        
        if not selected_resource:
            raise Exception("No available cloud resources for task execution")
        
        # Create execution record
        execution = TaskExecution(
            task_id=task_id,
            execution_id=execution_id,
            provider=selected_resource.provider,
            region=selected_resource.region,
            status=ExecutionStatus.QUEUED,
            start_time=None,
            end_time=None,
            result=None,
            error_message=None,
            resource_usage={},
            cost_estimate=self._estimate_task_cost(task_data, selected_resource)
        )
        
        # Queue execution
        with self.execution_lock:
            self.active_executions[execution_id] = execution
        
        try:
            # Execute task on selected provider
            execution = await self._execute_on_provider(execution, task_data, selected_resource)
            
        except Exception as e:
            # Handle execution failure
            execution.status = ExecutionStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = datetime.now()
            
            # Attempt failover if enabled
            strategy = self.distribution_strategies.get(strategy_name)
            if strategy and strategy.failover_enabled:
                print(f"🔄 Attempting failover for task {task_id}")
                execution = await self._attempt_failover(execution, task_data, strategy_name)
        
        finally:
            # Update tracking
            self._update_performance_tracking(execution)
            self._log_execution(execution)
            
            # Cleanup
            with self.execution_lock:
                if execution_id in self.active_executions:
                    del self.active_executions[execution_id]
        
        return execution
    
    async def execute_multiple_tasks(self, tasks: List[Dict[str, Any]], 
                                   strategy_name: str = "balanced",
                                   max_concurrent: int = 10) -> List[TaskExecution]:
        """Execute multiple tasks with intelligent distribution"""
        
        print(f"🎯 Executing {len(tasks)} tasks with {strategy_name} strategy")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(min(max_concurrent, self.max_global_concurrent))
        
        async def execute_with_semaphore(task_data):
            async with semaphore:
                return await self.execute_task_distributed(task_data, strategy_name)
        
        # Execute tasks concurrently
        execution_tasks = [execute_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Process results
        executions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create failed execution record
                execution = TaskExecution(
                    task_id=tasks[i].get('id', f"task_{i}"),
                    execution_id=f"failed_{int(time.time())}_{i}",
                    provider=CloudProvider.LOCAL,
                    region="local",
                    status=ExecutionStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    result=None,
                    error_message=str(result),
                    resource_usage={},
                    cost_estimate=0.0
                )
                executions.append(execution)
            else:
                executions.append(result)
        
        # Generate summary
        successful = len([e for e in executions if e.status == ExecutionStatus.COMPLETED])
        total_cost = sum(e.cost_estimate for e in executions)
        
        print(f"📊 Distributed execution completed: {successful}/{len(tasks)} successful, ${total_cost:.4f} total cost")
        
        return executions
    
    async def _select_optimal_provider(self, task_data: Dict[str, Any], 
                                     strategy_name: str) -> Optional[CloudResource]:
        """Select optimal cloud provider based on strategy"""
        
        strategy = self.distribution_strategies.get(strategy_name)
        if not strategy:
            strategy = self.distribution_strategies["balanced"]
        
        available_resources = [r for r in self.cloud_resources.values() if r.is_active]
        
        if not available_resources:
            return None
        
        # Apply selection algorithm
        if strategy.load_balancing_algorithm == "least_cost":
            selected = min(available_resources, key=lambda r: r.cost_per_hour)
        elif strategy.load_balancing_algorithm == "highest_performance":
            selected = max(available_resources, key=lambda r: r.performance_score)
        elif strategy.load_balancing_algorithm == "weighted_round_robin":
            selected = self._weighted_round_robin_selection(available_resources)
        elif strategy.load_balancing_algorithm == "geographic_distribution":
            selected = self._geographic_distribution_selection(available_resources)
        else:
            selected = available_resources[0]  # Default fallback
        
        # Check capacity
        current_load = self._get_current_load(selected)
        if current_load >= selected.max_concurrent_tasks:
            # Find alternative
            alternatives = [r for r in available_resources 
                          if r != selected and self._get_current_load(r) < r.max_concurrent_tasks]
            selected = alternatives[0] if alternatives else None
        
        return selected
    
    async def _execute_on_provider(self, execution: TaskExecution, 
                                 task_data: Dict[str, Any], 
                                 resource: CloudResource) -> TaskExecution:
        """Execute task on specific cloud provider"""
        
        execution.status = ExecutionStatus.RUNNING
        execution.start_time = datetime.now()
        
        print(f"▶️ Executing task {execution.task_id} on {resource.provider.value} ({resource.region})")
        
        try:
            if resource.provider == CloudProvider.LOCAL:
                result = await self._execute_locally(task_data)
            else:
                result = await self._execute_on_cloud(task_data, resource)
            
            execution.status = ExecutionStatus.COMPLETED
            execution.result = result
            execution.resource_usage = result.get('resource_usage', {})
            
        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.error_message = str(e)
            raise
        
        finally:
            execution.end_time = datetime.now()
        
        return execution
    
    async def _execute_locally(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task locally"""
        
        # Simulate task execution with task-master
        task_id = task_data.get('id', 'unknown')
        
        # Create local execution environment
        start_time = time.time()
        
        try:
            # Execute using task-master CLI (simulated)
            command = f"echo 'Executing task {task_id} locally'"
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
            
            execution_time = time.time() - start_time
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr,
                'execution_time': execution_time,
                'resource_usage': {
                    'cpu_percent': 25.0,
                    'memory_mb': 128.0,
                    'disk_io_mb': 10.0
                }
            }
            
        except subprocess.TimeoutExpired:
            raise Exception("Task execution timed out")
        except Exception as e:
            raise Exception(f"Local execution failed: {e}")
    
    async def _execute_on_cloud(self, task_data: Dict[str, Any], 
                              resource: CloudResource) -> Dict[str, Any]:
        """Execute task on cloud provider"""
        
        # Simulate cloud execution
        task_id = task_data.get('id', 'unknown')
        
        print(f"☁️ Simulating cloud execution on {resource.provider.value}")
        
        # Simulate cloud API call delay
        await asyncio.sleep(2)
        
        # Simulate execution
        start_time = time.time()
        execution_time = 30 + (hash(task_id) % 60)  # Simulated variable execution time
        
        await asyncio.sleep(min(execution_time / 10, 5))  # Speed up for demo
        
        return {
            'success': True,
            'output': f"Task {task_id} executed successfully on {resource.provider.value}",
            'execution_time': execution_time,
            'resource_usage': {
                'cpu_percent': 40.0 + (hash(task_id) % 30),
                'memory_mb': 256.0 + (hash(task_id) % 128),
                'disk_io_mb': 50.0 + (hash(task_id) % 25)
            },
            'cloud_metadata': {
                'provider': resource.provider.value,
                'region': resource.region,
                'instance_type': resource.instance_type
            }
        }
    
    async def _attempt_failover(self, failed_execution: TaskExecution, 
                              task_data: Dict[str, Any], 
                              strategy_name: str) -> TaskExecution:
        """Attempt failover to alternative provider"""
        
        print(f"🔄 Attempting failover for task {failed_execution.task_id}")
        
        # Find alternative provider
        current_provider = failed_execution.provider
        available_alternatives = [
            r for r in self.cloud_resources.values() 
            if r.is_active and r.provider != current_provider
        ]
        
        if not available_alternatives:
            print("❌ No alternative providers available for failover")
            return failed_execution
        
        # Select best alternative
        alternative = max(available_alternatives, key=lambda r: r.performance_score)
        
        # Create new execution attempt
        failover_execution = TaskExecution(
            task_id=failed_execution.task_id,
            execution_id=f"failover_{failed_execution.execution_id}",
            provider=alternative.provider,
            region=alternative.region,
            status=ExecutionStatus.QUEUED,
            start_time=None,
            end_time=None,
            result=None,
            error_message=None,
            resource_usage={},
            cost_estimate=self._estimate_task_cost(task_data, alternative)
        )
        
        try:
            failover_execution = await self._execute_on_provider(failover_execution, task_data, alternative)
            print(f"✅ Failover successful for task {failed_execution.task_id}")
            return failover_execution
            
        except Exception as e:
            print(f"❌ Failover failed for task {failed_execution.task_id}: {e}")
            failover_execution.status = ExecutionStatus.FAILED
            failover_execution.error_message = f"Failover failed: {e}"
            return failover_execution
    
    def _weighted_round_robin_selection(self, resources: List[CloudResource]) -> CloudResource:
        """Select resource using weighted round robin"""
        # Weight by performance score and inverse cost
        weights = []
        for resource in resources:
            weight = resource.performance_score * (1.0 / (resource.cost_per_hour + 0.001))
            weights.append(weight)
        
        # Select based on weights (simplified)
        total_weight = sum(weights)
        if total_weight == 0:
            return resources[0]
        
        import random
        selection_point = random.uniform(0, total_weight)
        current_weight = 0
        
        for i, weight in enumerate(weights):
            current_weight += weight
            if current_weight >= selection_point:
                return resources[i]
        
        return resources[-1]
    
    def _geographic_distribution_selection(self, resources: List[CloudResource]) -> CloudResource:
        """Select resource for geographic distribution"""
        # Prefer different regions for distribution
        used_regions = set()
        for execution in self.active_executions.values():
            if execution.status in [ExecutionStatus.RUNNING, ExecutionStatus.QUEUED]:
                used_regions.add(execution.region)
        
        # Find resource in unused region
        for resource in resources:
            if resource.region not in used_regions:
                return resource
        
        # If all regions used, fall back to performance-based selection
        return max(resources, key=lambda r: r.performance_score)
    
    def _get_current_load(self, resource: CloudResource) -> int:
        """Get current load for a resource"""
        load = 0
        resource_id = f"{resource.provider.value}_{resource.region}"
        
        for execution in self.active_executions.values():
            if (execution.provider == resource.provider and 
                execution.region == resource.region and
                execution.status in [ExecutionStatus.RUNNING, ExecutionStatus.QUEUED]):
                load += 1
        
        return load
    
    def _estimate_task_cost(self, task_data: Dict[str, Any], resource: CloudResource) -> float:
        """Estimate cost for task execution"""
        # Estimate execution time based on task complexity
        complexity = len(task_data.get('details', '')) / 1000  # Simple heuristic
        estimated_hours = max(0.1, min(2.0, 0.5 + complexity))  # 6 minutes to 2 hours
        
        return resource.cost_per_hour * estimated_hours
    
    def _update_performance_tracking(self, execution: TaskExecution):
        """Update performance tracking metrics"""
        provider = execution.provider
        
        if provider not in self.provider_performance:
            self.provider_performance[provider] = {
                'total_executions': 0,
                'successful_executions': 0,
                'total_execution_time': 0,
                'average_execution_time': 0,
                'success_rate': 0
            }
        
        metrics = self.provider_performance[provider]
        metrics['total_executions'] += 1
        
        if execution.status == ExecutionStatus.COMPLETED:
            metrics['successful_executions'] += 1
            
            if execution.start_time and execution.end_time:
                exec_time = (execution.end_time - execution.start_time).total_seconds()
                metrics['total_execution_time'] += exec_time
                metrics['average_execution_time'] = metrics['total_execution_time'] / metrics['successful_executions']
        
        metrics['success_rate'] = metrics['successful_executions'] / metrics['total_executions']
        
        # Update cost tracking
        if provider not in self.cost_tracking:
            self.cost_tracking[provider] = 0.0
        self.cost_tracking[provider] += execution.cost_estimate
    
    def _log_execution(self, execution: TaskExecution):
        """Log execution details"""
        try:
            log_entry = asdict(execution)
            log_entry['start_time'] = execution.start_time.isoformat() if execution.start_time else None
            log_entry['end_time'] = execution.end_time.isoformat() if execution.end_time else None
            log_entry['provider'] = execution.provider.value
            log_entry['status'] = execution.status.value
            
            # Append to log file
            logs = []
            if self.executions_log.exists():
                with open(self.executions_log, 'r') as f:
                    logs = json.load(f)
            
            logs.append(log_entry)
            
            # Keep only last 1000 executions
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            with open(self.executions_log, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to log execution: {e}")
    
    def save_configuration(self):
        """Save cloud configuration to disk"""
        try:
            # Save providers
            providers_data = {}
            for resource_id, resource in self.cloud_resources.items():
                providers_data[resource_id] = asdict(resource)
                providers_data[resource_id]['provider'] = resource.provider.value
            
            with open(self.providers_config, 'w') as f:
                json.dump(providers_data, f, indent=2)
            
            # Save strategies
            strategies_data = {}
            for strategy_name, strategy in self.distribution_strategies.items():
                strategies_data[strategy_name] = asdict(strategy)
            
            with open(self.strategies_config, 'w') as f:
                json.dump(strategies_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save configuration: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        return {
            'provider_performance': {
                provider.value: metrics 
                for provider, metrics in self.provider_performance.items()
            },
            'cost_tracking': {
                provider.value: cost 
                for provider, cost in self.cost_tracking.items()
            },
            'active_executions': len(self.active_executions),
            'total_resources': len(self.cloud_resources),
            'active_resources': len([r for r in self.cloud_resources.values() if r.is_active])
        }

async def main():
    """Demo of distributed cloud execution"""
    print("Distributed Cloud Execution Manager Demo")
    print("=" * 50)
    
    manager = CloudExecutionManager()
    
    # Demo tasks
    demo_tasks = [
        {
            'id': 'dist-1',
            'title': 'Process user data',
            'description': 'Extract and validate user information',
            'details': 'Implement data processing pipeline with validation rules and error handling',
            'priority': 'high'
        },
        {
            'id': 'dist-2',
            'title': 'Generate analytics report',
            'description': 'Create comprehensive analytics dashboard',
            'details': 'Build interactive dashboard with charts, graphs, and real-time data updates',
            'priority': 'medium'
        },
        {
            'id': 'dist-3',
            'title': 'Optimize database queries',
            'description': 'Improve query performance and indexing',
            'details': 'Analyze slow queries, add appropriate indexes, and optimize JOIN operations',
            'priority': 'low'
        }
    ]
    
    print(f"🎯 Testing distributed execution with {len(demo_tasks)} tasks")
    
    # Test different strategies
    strategies = ["cost_optimized", "performance_first", "balanced"]
    
    for strategy in strategies:
        print(f"\n📊 Testing {strategy} strategy:")
        
        executions = await manager.execute_multiple_tasks(demo_tasks, strategy, max_concurrent=2)
        
        successful = len([e for e in executions if e.status == ExecutionStatus.COMPLETED])
        total_cost = sum(e.cost_estimate for e in executions)
        
        print(f"  Results: {successful}/{len(demo_tasks)} successful, ${total_cost:.4f} cost")
        
        # Show provider distribution
        provider_usage = {}
        for execution in executions:
            provider = execution.provider.value
            provider_usage[provider] = provider_usage.get(provider, 0) + 1
        
        print(f"  Provider distribution: {provider_usage}")
    
    # Generate performance report
    report = manager.get_performance_report()
    print(f"\n📈 Performance Report:")
    print(f"  Active executions: {report['active_executions']}")
    print(f"  Total resources: {report['total_resources']}")
    
    for provider, metrics in report['provider_performance'].items():
        print(f"  {provider}: {metrics['success_rate']:.1%} success rate, "
              f"{metrics['average_execution_time']:.1f}s avg time")
    
    print(f"\n✅ Distributed cloud execution demo completed")

if __name__ == "__main__":
    asyncio.run(main())