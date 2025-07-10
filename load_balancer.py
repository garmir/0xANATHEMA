#!/usr/bin/env python3
"""
Load Balancing and Task Distribution
Atomic Task 49.3: Develop Load Balancing and Task Distribution Mechanisms

This module implements advanced load balancing strategies and task distribution
algorithms for optimal resource utilization in multi-agent systems.
"""

import asyncio
import json
import logging
import math
import statistics
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from collections import defaultdict, deque
import heapq


class LoadBalancingStrategy(Enum):
    """Load balancing strategy types"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_AWARE = "resource_aware"
    PERFORMANCE_BASED = "performance_based"
    SPECIALTY_BASED = "specialty_based"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


class TaskPriority(Enum):
    """Task priority levels for distribution"""
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1


@dataclass
class AgentWorkload:
    """Agent workload tracking"""
    agent_id: str
    active_tasks: int = 0
    pending_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time: float = 0.0
    current_cpu_usage: float = 0.0
    current_memory_usage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    performance_score: float = 1.0
    specialization_score: Dict[str, float] = field(default_factory=dict)
    
    @property
    def total_load(self) -> float:
        """Calculate total load score"""
        task_load = self.active_tasks + (self.pending_tasks * 0.5)
        resource_load = (self.current_cpu_usage + self.current_memory_usage) / 2
        return task_load * 0.7 + resource_load * 0.3
    
    @property
    def utilization_rate(self) -> float:
        """Calculate utilization rate"""
        total_tasks = self.completed_tasks + self.failed_tasks
        if total_tasks == 0:
            return 0.0
        return self.completed_tasks / total_tasks


@dataclass
class TaskDistributionMetrics:
    """Task distribution performance metrics"""
    total_tasks_distributed: int = 0
    average_distribution_time: float = 0.0
    load_balance_variance: float = 0.0
    agent_utilization_rates: Dict[str, float] = field(default_factory=dict)
    strategy_performance: Dict[str, float] = field(default_factory=dict)
    last_rebalance: datetime = field(default_factory=datetime.now)
    
    def update_metrics(self, agents: Dict[str, AgentWorkload]):
        """Update distribution metrics"""
        loads = [agent.total_load for agent in agents.values()]
        self.load_balance_variance = statistics.variance(loads) if len(loads) > 1 else 0.0
        self.agent_utilization_rates = {
            aid: agent.utilization_rate for aid, agent in agents.items()
        }


@dataclass
class TaskRequest:
    """Task distribution request"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    estimated_duration: float = 0.0
    required_capabilities: Set[str] = field(default_factory=set)
    deadline: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3


class LoadBalancingStrategy(ABC):
    """Abstract base class for load balancing strategies"""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.logger = logging.getLogger(f"Strategy.{strategy_name}")
        self.performance_history: List[float] = []
        
    @abstractmethod
    async def select_agent(self, task: TaskRequest, agents: Dict[str, AgentWorkload]) -> Optional[str]:
        """Select best agent for task"""
        pass
    
    def update_performance(self, score: float):
        """Update strategy performance score"""
        self.performance_history.append(score)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    @property
    def average_performance(self) -> float:
        """Get average performance score"""
        return statistics.mean(self.performance_history) if self.performance_history else 0.0


class RoundRobinStrategy(LoadBalancingStrategy):
    """Simple round-robin load balancing"""
    
    def __init__(self):
        super().__init__("round_robin")
        self.current_index = 0
    
    async def select_agent(self, task: TaskRequest, agents: Dict[str, AgentWorkload]) -> Optional[str]:
        """Select agent using round-robin"""
        if not agents:
            return None
        
        agent_ids = list(agents.keys())
        selected_id = agent_ids[self.current_index % len(agent_ids)]
        self.current_index += 1
        
        self.logger.debug(f"Round-robin selected agent: {selected_id}")
        return selected_id


class LeastLoadedStrategy(LoadBalancingStrategy):
    """Least loaded agent selection"""
    
    def __init__(self):
        super().__init__("least_loaded")
    
    async def select_agent(self, task: TaskRequest, agents: Dict[str, AgentWorkload]) -> Optional[str]:
        """Select least loaded agent"""
        if not agents:
            return None
        
        # Find agent with minimum load
        min_load_agent = min(agents.items(), key=lambda x: x[1].total_load)
        selected_id = min_load_agent[0]
        
        self.logger.debug(f"Least loaded selected agent: {selected_id} (load: {min_load_agent[1].total_load:.2f})")
        return selected_id


class WeightedRoundRobinStrategy(LoadBalancingStrategy):
    """Weighted round-robin based on agent capabilities"""
    
    def __init__(self):
        super().__init__("weighted_round_robin")
        self.agent_weights: Dict[str, int] = {}
        self.current_weights: Dict[str, int] = {}
    
    async def select_agent(self, task: TaskRequest, agents: Dict[str, AgentWorkload]) -> Optional[str]:
        """Select agent using weighted round-robin"""
        if not agents:
            return None
        
        # Initialize weights if not set
        for agent_id in agents:
            if agent_id not in self.agent_weights:
                self.agent_weights[agent_id] = max(1, int(agents[agent_id].performance_score * 10))
                self.current_weights[agent_id] = 0
        
        # Find agent with highest current weight
        max_weight = max(self.current_weights.values())
        if max_weight <= 0:
            # Reset weights
            self.current_weights = {aid: weight for aid, weight in self.agent_weights.items()}
            max_weight = max(self.current_weights.values())
        
        # Select agent with max weight
        for agent_id, weight in self.current_weights.items():
            if weight == max_weight:
                self.current_weights[agent_id] -= 1
                self.logger.debug(f"Weighted RR selected agent: {agent_id} (weight: {weight})")
                return agent_id
        
        return None


class ResourceAwareStrategy(LoadBalancingStrategy):
    """Resource-aware load balancing"""
    
    def __init__(self):
        super().__init__("resource_aware")
    
    async def select_agent(self, task: TaskRequest, agents: Dict[str, AgentWorkload]) -> Optional[str]:
        """Select agent based on resource requirements"""
        if not agents:
            return None
        
        cpu_req = task.resource_requirements.get("cpu", 0.1)
        memory_req = task.resource_requirements.get("memory", 0.1)
        
        # Find agents that can handle the resource requirements
        suitable_agents = []
        for agent_id, workload in agents.items():
            available_cpu = max(0, 1.0 - workload.current_cpu_usage)
            available_memory = max(0, 1.0 - workload.current_memory_usage)
            
            if available_cpu >= cpu_req and available_memory >= memory_req:
                # Calculate efficiency score
                efficiency = (available_cpu - cpu_req) + (available_memory - memory_req)
                suitable_agents.append((agent_id, efficiency, workload.total_load))
        
        if not suitable_agents:
            # Fallback to least loaded
            return min(agents.items(), key=lambda x: x[1].total_load)[0]
        
        # Select agent with best efficiency and lowest load
        suitable_agents.sort(key=lambda x: (-x[1], x[2]))
        selected_id = suitable_agents[0][0]
        
        self.logger.debug(f"Resource-aware selected agent: {selected_id}")
        return selected_id


class PerformanceBasedStrategy(LoadBalancingStrategy):
    """Performance-based load balancing"""
    
    def __init__(self):
        super().__init__("performance_based")
    
    async def select_agent(self, task: TaskRequest, agents: Dict[str, AgentWorkload]) -> Optional[str]:
        """Select agent based on performance history"""
        if not agents:
            return None
        
        # Calculate performance score considering load
        scored_agents = []
        for agent_id, workload in agents.items():
            # Higher performance score and lower load is better
            score = workload.performance_score / max(1, workload.total_load)
            scored_agents.append((agent_id, score))
        
        # Select agent with highest score
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        selected_id = scored_agents[0][0]
        
        self.logger.debug(f"Performance-based selected agent: {selected_id} (score: {scored_agents[0][1]:.2f})")
        return selected_id


class SpecialtyBasedStrategy(LoadBalancingStrategy):
    """Specialty-based load balancing"""
    
    def __init__(self):
        super().__init__("specialty_based")
    
    async def select_agent(self, task: TaskRequest, agents: Dict[str, AgentWorkload]) -> Optional[str]:
        """Select agent based on task specialization"""
        if not agents:
            return None
        
        # Find agents with required capabilities
        suitable_agents = []
        for agent_id, workload in agents.items():
            specialization_scores = []
            for capability in task.required_capabilities:
                score = workload.specialization_score.get(capability, 0.0)
                specialization_scores.append(score)
            
            if specialization_scores:
                avg_specialization = statistics.mean(specialization_scores)
                # Combine specialization with inverse load
                combined_score = avg_specialization / max(1, workload.total_load)
                suitable_agents.append((agent_id, combined_score))
        
        if not suitable_agents:
            # Fallback to least loaded
            return min(agents.items(), key=lambda x: x[1].total_load)[0]
        
        # Select agent with highest combined score
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        selected_id = suitable_agents[0][0]
        
        self.logger.debug(f"Specialty-based selected agent: {selected_id}")
        return selected_id


class PredictiveStrategy(LoadBalancingStrategy):
    """Predictive load balancing using historical patterns"""
    
    def __init__(self):
        super().__init__("predictive")
        self.execution_history: Dict[str, List[float]] = defaultdict(list)
        self.pattern_weights: Dict[str, float] = {}
    
    async def select_agent(self, task: TaskRequest, agents: Dict[str, AgentWorkload]) -> Optional[str]:
        """Select agent using predictive analysis"""
        if not agents:
            return None
        
        # Predict execution times for each agent
        predictions = {}
        for agent_id, workload in agents.items():
            predicted_time = self._predict_execution_time(agent_id, task)
            current_queue_time = workload.active_tasks * workload.average_execution_time
            total_predicted_time = predicted_time + current_queue_time
            predictions[agent_id] = total_predicted_time
        
        # Select agent with minimum predicted completion time
        selected_id = min(predictions.items(), key=lambda x: x[1])[0]
        
        self.logger.debug(f"Predictive selected agent: {selected_id} (predicted time: {predictions[selected_id]:.2f}s)")
        return selected_id
    
    def _predict_execution_time(self, agent_id: str, task: TaskRequest) -> float:
        """Predict execution time for agent"""
        history = self.execution_history[agent_id]
        if not history:
            return task.estimated_duration or 10.0
        
        # Simple moving average prediction
        recent_history = history[-10:]  # Last 10 executions
        return statistics.mean(recent_history)
    
    def update_execution_time(self, agent_id: str, execution_time: float):
        """Update execution time history"""
        self.execution_history[agent_id].append(execution_time)
        if len(self.execution_history[agent_id]) > 100:
            self.execution_history[agent_id].pop(0)


class AdaptiveStrategy(LoadBalancingStrategy):
    """Adaptive strategy that switches between other strategies"""
    
    def __init__(self):
        super().__init__("adaptive")
        self.strategies = {
            "least_loaded": LeastLoadedStrategy(),
            "resource_aware": ResourceAwareStrategy(),
            "performance_based": PerformanceBasedStrategy(),
            "specialty_based": SpecialtyBasedStrategy()
        }
        self.strategy_scores = {name: 1.0 for name in self.strategies}
        self.current_strategy = "least_loaded"
        self.adaptation_interval = 50  # tasks
        self.task_count = 0
    
    async def select_agent(self, task: TaskRequest, agents: Dict[str, AgentWorkload]) -> Optional[str]:
        """Select agent using adaptive strategy selection"""
        self.task_count += 1
        
        # Adapt strategy periodically
        if self.task_count % self.adaptation_interval == 0:
            self._adapt_strategy()
        
        # Use current best strategy
        strategy = self.strategies[self.current_strategy]
        selected_id = await strategy.select_agent(task, agents)
        
        self.logger.debug(f"Adaptive using {self.current_strategy}, selected agent: {selected_id}")
        return selected_id
    
    def _adapt_strategy(self):
        """Adapt to best performing strategy"""
        # Find strategy with highest average performance
        best_strategy = max(self.strategy_scores.items(), key=lambda x: x[1])
        self.current_strategy = best_strategy[0]
        
        # Update scores based on recent performance
        for name, strategy in self.strategies.items():
            if strategy.performance_history:
                recent_performance = statistics.mean(strategy.performance_history[-20:])
                self.strategy_scores[name] = recent_performance
        
        self.logger.info(f"Adapted to strategy: {self.current_strategy}")


class TaskQueue:
    """Priority-based task queue"""
    
    def __init__(self):
        self.queues = {priority: deque() for priority in TaskPriority}
        self.task_index = {}  # task_id -> (priority, position)
        self.lock = asyncio.Lock()
    
    async def enqueue(self, task: TaskRequest):
        """Add task to queue"""
        async with self.lock:
            self.queues[task.priority].append(task)
            self.task_index[task.task_id] = (task.priority, len(self.queues[task.priority]) - 1)
    
    async def dequeue(self) -> Optional[TaskRequest]:
        """Remove and return highest priority task"""
        async with self.lock:
            # Process queues in priority order
            for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
                if self.queues[priority]:
                    task = self.queues[priority].popleft()
                    del self.task_index[task.task_id]
                    return task
            return None
    
    async def remove_task(self, task_id: str) -> bool:
        """Remove specific task from queue"""
        async with self.lock:
            if task_id in self.task_index:
                priority, position = self.task_index[task_id]
                queue = self.queues[priority]
                # Linear search and remove (not efficient for large queues)
                for i, task in enumerate(queue):
                    if task.task_id == task_id:
                        del queue[i]
                        del self.task_index[task_id]
                        return True
            return False
    
    @property
    def size(self) -> int:
        """Total queue size"""
        return sum(len(queue) for queue in self.queues.values())
    
    def get_priority_counts(self) -> Dict[TaskPriority, int]:
        """Get count of tasks by priority"""
        return {priority: len(queue) for priority, queue in self.queues.items()}


class LoadBalancer:
    """Main load balancer coordinating task distribution"""
    
    def __init__(self):
        self.agents: Dict[str, AgentWorkload] = {}
        self.strategies: Dict[str, LoadBalancingStrategy] = {
            "round_robin": RoundRobinStrategy(),
            "least_loaded": LeastLoadedStrategy(),
            "weighted_round_robin": WeightedRoundRobinStrategy(),
            "resource_aware": ResourceAwareStrategy(),
            "performance_based": PerformanceBasedStrategy(),
            "specialty_based": SpecialtyBasedStrategy(),
            "predictive": PredictiveStrategy(),
            "adaptive": AdaptiveStrategy()
        }
        self.current_strategy = "adaptive"
        self.task_queue = TaskQueue()
        self.metrics = TaskDistributionMetrics()
        self.is_running = False
        self.logger = logging.getLogger("LoadBalancer")
        self.rebalance_threshold = 0.5  # Variance threshold for rebalancing
        self.rebalance_interval = 60.0  # seconds
        
    def register_agent(self, agent_id: str, capabilities: Set[str] = None, initial_weight: float = 1.0):
        """Register agent with load balancer"""
        workload = AgentWorkload(agent_id=agent_id, performance_score=initial_weight)
        if capabilities:
            workload.specialization_score = {cap: 1.0 for cap in capabilities}
        
        self.agents[agent_id] = workload
        self.logger.info(f"Registered agent: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister agent from load balancer"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")
    
    def update_agent_workload(self, agent_id: str, **kwargs):
        """Update agent workload information"""
        if agent_id in self.agents:
            workload = self.agents[agent_id]
            for key, value in kwargs.items():
                if hasattr(workload, key):
                    setattr(workload, key, value)
            workload.last_updated = datetime.now()
    
    async def submit_task(self, task: TaskRequest) -> str:
        """Submit task for distribution"""
        await self.task_queue.enqueue(task)
        self.logger.debug(f"Task {task.task_id} queued with priority {task.priority.name}")
        return task.task_id
    
    async def distribute_task(self) -> Optional[Tuple[str, str]]:
        """Distribute next task to appropriate agent"""
        task = await self.task_queue.dequeue()
        if not task:
            return None
        
        start_time = time.time()
        
        # Select agent using current strategy
        strategy = self.strategies[self.current_strategy]
        selected_agent = await strategy.select_agent(task, self.agents)
        
        if selected_agent:
            # Update agent workload
            self.update_agent_workload(selected_agent, pending_tasks=self.agents[selected_agent].pending_tasks + 1)
            
            # Update metrics
            distribution_time = time.time() - start_time
            self.metrics.total_tasks_distributed += 1
            self.metrics.average_distribution_time = (
                (self.metrics.average_distribution_time * (self.metrics.total_tasks_distributed - 1) + distribution_time) /
                self.metrics.total_tasks_distributed
            )
            
            self.logger.info(f"Distributed task {task.task_id} to agent {selected_agent}")
            return task.task_id, selected_agent
        else:
            # Re-queue task if no agent available
            await self.task_queue.enqueue(task)
            self.logger.warning(f"No available agent for task {task.task_id}, re-queuing")
            return None
    
    async def start_distribution_loop(self):
        """Start continuous task distribution loop"""
        self.is_running = True
        self.logger.info("Load balancer distribution loop started")
        
        while self.is_running:
            try:
                result = await self.distribute_task()
                if result:
                    await asyncio.sleep(0.1)  # Small delay between distributions
                else:
                    await asyncio.sleep(1.0)  # Longer delay when no tasks
                    
                # Check for rebalancing
                if self._should_rebalance():
                    await self._rebalance_workload()
                    
            except Exception as e:
                self.logger.error(f"Error in distribution loop: {e}")
                await asyncio.sleep(1.0)
    
    async def stop_distribution_loop(self):
        """Stop task distribution loop"""
        self.is_running = False
        self.logger.info("Load balancer distribution loop stopped")
    
    def _should_rebalance(self) -> bool:
        """Check if workload rebalancing is needed"""
        if len(self.agents) < 2:
            return False
        
        # Check time since last rebalance
        time_since_rebalance = (datetime.now() - self.metrics.last_rebalance).total_seconds()
        if time_since_rebalance < self.rebalance_interval:
            return False
        
        # Check load variance
        loads = [agent.total_load for agent in self.agents.values()]
        if len(loads) > 1:
            variance = statistics.variance(loads)
            return variance > self.rebalance_threshold
        
        return False
    
    async def _rebalance_workload(self):
        """Rebalance workload across agents"""
        self.logger.info("Starting workload rebalancing")
        
        # Find overloaded and underloaded agents
        loads = [(aid, agent.total_load) for aid, agent in self.agents.items()]
        loads.sort(key=lambda x: x[1])
        
        mean_load = statistics.mean([load for _, load in loads])
        
        overloaded = [aid for aid, load in loads if load > mean_load * 1.5]
        underloaded = [aid for aid, load in loads if load < mean_load * 0.5]
        
        if overloaded and underloaded:
            self.logger.info(f"Rebalancing: {len(overloaded)} overloaded, {len(underloaded)} underloaded agents")
            # In a real implementation, you would move tasks between agents
            # For now, we just log the rebalancing action
        
        self.metrics.last_rebalance = datetime.now()
        self.metrics.update_metrics(self.agents)
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        if not self.agents:
            return {}
        
        loads = [agent.total_load for agent in self.agents.values()]
        utilizations = [agent.utilization_rate for agent in self.agents.values()]
        
        return {
            "agent_count": len(self.agents),
            "queue_size": self.task_queue.size,
            "queue_breakdown": self.task_queue.get_priority_counts(),
            "load_stats": {
                "mean": statistics.mean(loads),
                "variance": statistics.variance(loads) if len(loads) > 1 else 0.0,
                "min": min(loads),
                "max": max(loads)
            },
            "utilization_stats": {
                "mean": statistics.mean(utilizations),
                "min": min(utilizations),
                "max": max(utilizations)
            },
            "distribution_metrics": asdict(self.metrics),
            "current_strategy": self.current_strategy
        }
    
    def switch_strategy(self, strategy_name: str):
        """Switch to different load balancing strategy"""
        if strategy_name in self.strategies:
            self.current_strategy = strategy_name
            self.logger.info(f"Switched to strategy: {strategy_name}")
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")


# Export key classes
__all__ = [
    "LoadBalancingStrategy", "TaskPriority", "AgentWorkload", "TaskDistributionMetrics",
    "TaskRequest", "RoundRobinStrategy", "LeastLoadedStrategy", "WeightedRoundRobinStrategy",
    "ResourceAwareStrategy", "PerformanceBasedStrategy", "SpecialtyBasedStrategy",
    "PredictiveStrategy", "AdaptiveStrategy", "TaskQueue", "LoadBalancer"
]


if __name__ == "__main__":
    # Demo usage
    async def demo():
        logging.basicConfig(level=logging.INFO)
        
        # Create load balancer
        lb = LoadBalancer()
        
        # Register agents with different capabilities
        lb.register_agent("research-agent-1", {"research", "analysis"}, 1.2)
        lb.register_agent("execution-agent-1", {"execution", "deployment"}, 1.0)
        lb.register_agent("validation-agent-1", {"validation", "testing"}, 0.8)
        
        # Submit test tasks
        tasks = [
            TaskRequest(
                task_type="research",
                priority=TaskPriority.HIGH,
                required_capabilities={"research"},
                estimated_duration=15.0
            ),
            TaskRequest(
                task_type="execution",
                priority=TaskPriority.NORMAL,
                required_capabilities={"execution"},
                estimated_duration=25.0
            ),
            TaskRequest(
                task_type="validation",
                priority=TaskPriority.LOW,
                required_capabilities={"validation"},
                estimated_duration=10.0
            )
        ]
        
        for task in tasks:
            await lb.submit_task(task)
        
        # Distribute tasks
        distributions = []
        for _ in range(3):
            result = await lb.distribute_task()
            if result:
                distributions.append(result)
        
        print("Task Distributions:")
        for task_id, agent_id in distributions:
            print(f"  Task {task_id[:8]} -> Agent {agent_id}")
        
        # Show statistics
        stats = lb.get_load_statistics()
        print(f"\nLoad Statistics:")
        print(f"  Agent Count: {stats['agent_count']}")
        print(f"  Queue Size: {stats['queue_size']}")
        print(f"  Mean Load: {stats['load_stats']['mean']:.2f}")
        print(f"  Load Variance: {stats['load_stats']['variance']:.2f}")
        print(f"  Strategy: {stats['current_strategy']}")
    
    # Run demo
    asyncio.run(demo())