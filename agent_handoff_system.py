#!/usr/bin/env python3
"""
Agent Handoff System for Graph-Based Orchestration
Implements sophisticated agent handoff mechanisms with context preservation,
load balancing, and failure recovery.
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from pathlib import Path
import weakref

# Import base components
from graph_based_orchestration import GraphState, GraphBasedOrchestrator
from multi_agent_orchestration import (
    BaseAgent, AgentRole, Task, TaskStatus, AgentMessage, MessageType,
    AgentCapabilities, CommunicationProtocol
)


class HandoffReason(Enum):
    """Reasons for agent handoff"""
    TASK_COMPLETED = "task_completed"
    AGENT_OVERLOADED = "agent_overloaded"
    AGENT_FAILED = "agent_failed"
    SKILL_MISMATCH = "skill_mismatch"
    TIMEOUT = "timeout"
    BETTER_AGENT_AVAILABLE = "better_agent_available"
    WORKFLOW_TRANSITION = "workflow_transition"
    MANUAL_OVERRIDE = "manual_override"


class HandoffStatus(Enum):
    """Status of handoff execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class HandoffContext:
    """Context information for agent handoffs"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent_id: Optional[str] = None
    to_agent_id: Optional[str] = None
    reason: HandoffReason = HandoffReason.WORKFLOW_TRANSITION
    task_data: Dict[str, Any] = field(default_factory=dict)
    state_snapshot: Optional[GraphState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "from_agent_id": self.from_agent_id,
            "to_agent_id": self.to_agent_id,
            "reason": self.reason.value,
            "task_data": self.task_data,
            "state_snapshot": asdict(self.state_snapshot) if self.state_snapshot else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class HandoffResult:
    """Result of a handoff operation"""
    success: bool
    context: HandoffContext
    status: HandoffStatus
    new_agent_id: Optional[str] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    recovery_actions: List[str] = field(default_factory=list)


class HandoffStrategy(ABC):
    """Abstract base class for handoff strategies"""
    
    @abstractmethod
    async def select_target_agent(self, context: HandoffContext, 
                                available_agents: List[BaseAgent]) -> Optional[BaseAgent]:
        """Select the best target agent for handoff"""
        pass
    
    @abstractmethod
    async def prepare_handoff(self, context: HandoffContext) -> bool:
        """Prepare for the handoff"""
        pass
    
    @abstractmethod
    async def execute_handoff(self, context: HandoffContext, 
                            target_agent: BaseAgent) -> HandoffResult:
        """Execute the handoff"""
        pass


class LoadBalancedHandoffStrategy(HandoffStrategy):
    """Load-balanced handoff strategy"""
    
    async def select_target_agent(self, context: HandoffContext, 
                                available_agents: List[BaseAgent]) -> Optional[BaseAgent]:
        """Select agent with lowest load"""
        if not available_agents:
            return None
        
        # Filter agents that can handle the task
        suitable_agents = []
        required_role = context.metadata.get('required_role')
        
        for agent in available_agents:
            # Check role compatibility
            if required_role and agent.role != required_role:
                continue
            
            # Check capacity
            if len(agent.current_tasks) >= agent.capabilities.max_concurrent_tasks:
                continue
            
            suitable_agents.append(agent)
        
        if not suitable_agents:
            return None
        
        # Select agent with lowest load
        return min(suitable_agents, key=lambda a: len(a.current_tasks))
    
    async def prepare_handoff(self, context: HandoffContext) -> bool:
        """Prepare context and validate handoff"""
        # Validate context completeness
        if not context.task_data:
            context.metadata['preparation_error'] = "No task data available"
            return False
        
        # Add preparation timestamp
        context.metadata['preparation_time'] = datetime.now().isoformat()
        return True
    
    async def execute_handoff(self, context: HandoffContext, 
                            target_agent: BaseAgent) -> HandoffResult:
        """Execute the handoff with monitoring"""
        start_time = datetime.now()
        
        try:
            # Create handoff message
            handoff_message = AgentMessage(
                sender_id=context.from_agent_id or "orchestrator",
                recipient_id=target_agent.agent_id,
                message_type=MessageType.TASK_ASSIGNMENT,
                payload={
                    "handoff_context": context.to_dict(),
                    "task_data": context.task_data,
                    "state_snapshot": asdict(context.state_snapshot) if context.state_snapshot else None
                },
                correlation_id=context.id
            )
            
            # Send to target agent
            if hasattr(target_agent, 'receive_message'):
                await target_agent.receive_message(handoff_message)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return HandoffResult(
                success=True,
                context=context,
                status=HandoffStatus.COMPLETED,
                new_agent_id=target_agent.agent_id,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return HandoffResult(
                success=False,
                context=context,
                status=HandoffStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e),
                recovery_actions=["retry_with_different_agent", "escalate_to_manual"]
            )


class SkillBasedHandoffStrategy(HandoffStrategy):
    """Skill-based handoff strategy"""
    
    def __init__(self):
        self.agent_skills: Dict[str, Set[str]] = {}
        self.skill_requirements: Dict[str, Set[str]] = {}
    
    def register_agent_skills(self, agent_id: str, skills: Set[str]):
        """Register skills for an agent"""
        self.agent_skills[agent_id] = skills
    
    def set_task_requirements(self, task_type: str, required_skills: Set[str]):
        """Set skill requirements for a task type"""
        self.skill_requirements[task_type] = required_skills
    
    async def select_target_agent(self, context: HandoffContext, 
                                available_agents: List[BaseAgent]) -> Optional[BaseAgent]:
        """Select agent based on skill matching"""
        task_type = context.task_data.get('type', 'default')
        required_skills = self.skill_requirements.get(task_type, set())
        
        if not required_skills:
            # Fall back to role-based selection
            required_role = context.metadata.get('required_role')
            if required_role:
                suitable_agents = [a for a in available_agents if a.role == required_role]
                return suitable_agents[0] if suitable_agents else None
            return available_agents[0] if available_agents else None
        
        # Score agents based on skill match
        agent_scores = []
        for agent in available_agents:
            agent_skills = self.agent_skills.get(agent.agent_id, set())
            
            # Calculate skill match score
            matching_skills = required_skills.intersection(agent_skills)
            skill_score = len(matching_skills) / len(required_skills) if required_skills else 0
            
            # Add load penalty
            load_penalty = len(agent.current_tasks) / agent.capabilities.max_concurrent_tasks
            
            total_score = skill_score - (load_penalty * 0.3)
            agent_scores.append((agent, total_score))
        
        if not agent_scores:
            return None
        
        # Select highest scoring agent
        return max(agent_scores, key=lambda x: x[1])[0]
    
    async def prepare_handoff(self, context: HandoffContext) -> bool:
        """Prepare skill-based handoff"""
        task_type = context.task_data.get('type', 'default')
        required_skills = self.skill_requirements.get(task_type, set())
        
        context.metadata['required_skills'] = list(required_skills)
        context.metadata['skill_matching_enabled'] = True
        
        return True
    
    async def execute_handoff(self, context: HandoffContext, 
                            target_agent: BaseAgent) -> HandoffResult:
        """Execute skill-based handoff"""
        # Use similar execution logic as LoadBalancedHandoffStrategy
        strategy = LoadBalancedHandoffStrategy()
        return await strategy.execute_handoff(context, target_agent)


class AgentHandoffManager:
    """Manages agent handoffs in the graph orchestration system"""
    
    def __init__(self, orchestrator_ref):
        self.orchestrator_ref = orchestrator_ref
        self.handoff_strategies: Dict[str, HandoffStrategy] = {
            'load_balanced': LoadBalancedHandoffStrategy(),
            'skill_based': SkillBasedHandoffStrategy()
        }
        self.default_strategy = 'load_balanced'
        self.handoff_history: List[HandoffResult] = []
        self.active_handoffs: Dict[str, HandoffContext] = {}
        self.logger = logging.getLogger("AgentHandoffManager")
        
        # Performance metrics
        self.metrics = {
            "total_handoffs": 0,
            "successful_handoffs": 0,
            "failed_handoffs": 0,
            "avg_handoff_time": 0.0,
            "handoffs_by_reason": {}
        }
    
    def set_default_strategy(self, strategy_name: str):
        """Set the default handoff strategy"""
        if strategy_name in self.handoff_strategies:
            self.default_strategy = strategy_name
            self.logger.info(f"Set default handoff strategy to: {strategy_name}")
        else:
            raise ValueError(f"Unknown handoff strategy: {strategy_name}")
    
    def register_custom_strategy(self, name: str, strategy: HandoffStrategy):
        """Register a custom handoff strategy"""
        self.handoff_strategies[name] = strategy
        self.logger.info(f"Registered custom handoff strategy: {name}")
    
    async def initiate_handoff(self, from_agent_id: Optional[str], 
                             to_agent_role: AgentRole,
                             reason: HandoffReason,
                             task_data: Dict[str, Any],
                             state: Optional[GraphState] = None,
                             strategy_name: Optional[str] = None) -> HandoffResult:
        """Initiate an agent handoff"""
        
        # Create handoff context
        context = HandoffContext(
            from_agent_id=from_agent_id,
            reason=reason,
            task_data=task_data,
            state_snapshot=state,
            metadata={'required_role': to_agent_role}
        )
        
        self.active_handoffs[context.id] = context
        self.logger.info(f"Initiating handoff {context.id}: {reason.value}")
        
        try:
            # Select strategy
            strategy_name = strategy_name or self.default_strategy
            strategy = self.handoff_strategies[strategy_name]
            
            # Get available agents
            orchestrator = self.orchestrator_ref()
            if not orchestrator:
                raise RuntimeError("Orchestrator reference is invalid")
            
            available_agents = [
                agent for agent in orchestrator.agents.values()
                if agent.role == to_agent_role
            ]
            
            if not available_agents:
                raise RuntimeError(f"No agents available for role: {to_agent_role}")
            
            # Prepare handoff
            preparation_success = await strategy.prepare_handoff(context)
            if not preparation_success:
                raise RuntimeError("Handoff preparation failed")
            
            # Select target agent
            target_agent = await strategy.select_target_agent(context, available_agents)
            if not target_agent:
                raise RuntimeError("No suitable target agent found")
            
            context.to_agent_id = target_agent.agent_id
            
            # Execute handoff
            result = await strategy.execute_handoff(context, target_agent)
            
            # Update metrics
            self._update_metrics(result)
            
            # Store result
            self.handoff_history.append(result)
            if len(self.handoff_history) > 1000:  # Keep last 1000 handoffs
                self.handoff_history.pop(0)
            
            # Clean up active handoff
            if context.id in self.active_handoffs:
                del self.active_handoffs[context.id]
            
            self.logger.info(f"Handoff {context.id} completed: {result.status.value}")
            return result
            
        except Exception as e:
            # Create failure result
            result = HandoffResult(
                success=False,
                context=context,
                status=HandoffStatus.FAILED,
                error_message=str(e)
            )
            
            self._update_metrics(result)
            self.handoff_history.append(result)
            
            if context.id in self.active_handoffs:
                del self.active_handoffs[context.id]
            
            self.logger.error(f"Handoff {context.id} failed: {e}")
            return result
    
    async def handle_agent_failure(self, failed_agent_id: str, 
                                 active_tasks: List[Task]) -> List[HandoffResult]:
        """Handle failure of an agent by redistributing tasks"""
        self.logger.warning(f"Handling failure of agent: {failed_agent_id}")
        
        handoff_results = []
        
        for task in active_tasks:
            # Determine appropriate agent role for the task
            task_type = task.requirements.get('type', '')
            
            # Map task types to agent roles (simplified)
            role_mapping = {
                'research_query': AgentRole.RESEARCH,
                'task_planning': AgentRole.PLANNING,
                'task_implementation': AgentRole.EXECUTION,
                'result_validation': AgentRole.VALIDATION
            }
            
            target_role = role_mapping.get(task_type, AgentRole.EXECUTION)
            
            # Initiate handoff
            result = await self.initiate_handoff(
                from_agent_id=failed_agent_id,
                to_agent_role=target_role,
                reason=HandoffReason.AGENT_FAILED,
                task_data=asdict(task)
            )
            
            handoff_results.append(result)
        
        return handoff_results
    
    async def rebalance_load(self) -> List[HandoffResult]:
        """Rebalance load across agents"""
        orchestrator = self.orchestrator_ref()
        if not orchestrator:
            return []
        
        # Calculate load distribution
        agent_loads = []
        for agent in orchestrator.agents.values():
            load_ratio = len(agent.current_tasks) / agent.capabilities.max_concurrent_tasks
            agent_loads.append((agent, load_ratio))
        
        # Sort by load (highest first)
        agent_loads.sort(key=lambda x: x[1], reverse=True)
        
        handoff_results = []
        
        # Move tasks from overloaded to underloaded agents
        for overloaded_agent, load_ratio in agent_loads:
            if load_ratio > 0.8:  # Overloaded threshold
                # Find underloaded agents of same role
                underloaded_agents = [
                    (agent, ratio) for agent, ratio in agent_loads
                    if agent.role == overloaded_agent.role and ratio < 0.5
                ]
                
                if underloaded_agents and overloaded_agent.current_tasks:
                    # Move one task
                    task = list(overloaded_agent.current_tasks.values())[0]
                    
                    result = await self.initiate_handoff(
                        from_agent_id=overloaded_agent.agent_id,
                        to_agent_role=overloaded_agent.role,
                        reason=HandoffReason.AGENT_OVERLOADED,
                        task_data=asdict(task)
                    )
                    
                    handoff_results.append(result)
        
        return handoff_results
    
    def _update_metrics(self, result: HandoffResult):
        """Update handoff metrics"""
        self.metrics["total_handoffs"] += 1
        
        if result.success:
            self.metrics["successful_handoffs"] += 1
        else:
            self.metrics["failed_handoffs"] += 1
        
        # Update average handoff time
        total_handoffs = self.metrics["total_handoffs"]
        current_avg = self.metrics["avg_handoff_time"]
        self.metrics["avg_handoff_time"] = (
            (current_avg * (total_handoffs - 1) + result.execution_time) / total_handoffs
        )
        
        # Update handoffs by reason
        reason = result.context.reason.value
        if reason not in self.metrics["handoffs_by_reason"]:
            self.metrics["handoffs_by_reason"][reason] = 0
        self.metrics["handoffs_by_reason"][reason] += 1
    
    def get_handoff_statistics(self) -> Dict[str, Any]:
        """Get handoff statistics"""
        success_rate = 0.0
        if self.metrics["total_handoffs"] > 0:
            success_rate = (self.metrics["successful_handoffs"] / 
                          self.metrics["total_handoffs"]) * 100
        
        return {
            **self.metrics,
            "success_rate_percentage": success_rate,
            "active_handoffs": len(self.active_handoffs),
            "handoff_history_size": len(self.handoff_history)
        }
    
    def get_recent_handoffs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent handoff results"""
        recent = self.handoff_history[-limit:] if self.handoff_history else []
        return [
            {
                "id": result.context.id,
                "from_agent": result.context.from_agent_id,
                "to_agent": result.new_agent_id,
                "reason": result.context.reason.value,
                "status": result.status.value,
                "success": result.success,
                "execution_time": result.execution_time,
                "timestamp": result.context.timestamp.isoformat()
            }
            for result in recent
        ]


# Integration with GraphBasedOrchestrator
class EnhancedGraphOrchestrator(GraphBasedOrchestrator):
    """Enhanced orchestrator with agent handoff capabilities"""
    
    def __init__(self, communication=None):
        super().__init__(communication)
        self.handoff_manager = AgentHandoffManager(weakref.ref(self))
        self.logger = logging.getLogger("EnhancedGraphOrchestrator")
    
    async def execute_node_with_handoff(self, execution_id: str, node_id: str, 
                                      state: GraphState) -> Tuple[GraphState, List[str]]:
        """Execute node with intelligent agent handoff"""
        node = self.graph.nodes.get(node_id)
        if not node or not node.agent_role:
            # Fall back to original execution
            return await self._execute_node(execution_id, node_id, state)
        
        try:
            # Check for handoff triggers
            handoff_needed = await self._check_handoff_triggers(node, state)
            
            if handoff_needed:
                # Initiate handoff
                result = await self.handoff_manager.initiate_handoff(
                    from_agent_id=None,
                    to_agent_role=node.agent_role,
                    reason=HandoffReason.WORKFLOW_TRANSITION,
                    task_data={
                        'node_id': node_id,
                        'description': f"Execute {node.name}",
                        'type': node.agent_role.value
                    },
                    state=state
                )
                
                if result.success:
                    # Update metrics
                    self.metrics["agent_handoffs"] += 1
                    
                    # Continue with normal execution
                    return await self._execute_node(execution_id, node_id, state)
                else:
                    self.logger.warning(f"Handoff failed for node {node_id}: {result.error_message}")
            
            # Execute normally
            return await self._execute_node(execution_id, node_id, state)
            
        except Exception as e:
            self.logger.error(f"Error in enhanced node execution: {e}")
            # Fall back to original execution
            return await self._execute_node(execution_id, node_id, state)
    
    async def _check_handoff_triggers(self, node, state: GraphState) -> bool:
        """Check if handoff is needed based on triggers"""
        # Check if current agent assignment is optimal
        required_role = node.agent_role
        
        # Find current best agent for this role
        suitable_agents = [
            agent for agent in self.agents.values()
            if agent.role == required_role
        ]
        
        if not suitable_agents:
            return False
        
        # Check load distribution
        avg_load = sum(len(a.current_tasks) for a in suitable_agents) / len(suitable_agents)
        min_load_agent = min(suitable_agents, key=lambda a: len(a.current_tasks))
        
        # Trigger handoff if there's significant load imbalance
        return len(min_load_agent.current_tasks) < avg_load * 0.7
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics including handoff statistics"""
        base_metrics = self.metrics
        handoff_stats = self.handoff_manager.get_handoff_statistics()
        
        return {
            **base_metrics,
            "handoff_metrics": handoff_stats,
            "recent_handoffs": self.handoff_manager.get_recent_handoffs(5)
        }


async def demonstrate_handoff_system():
    """Demonstrate the agent handoff system"""
    print("🔄 Agent Handoff System Demonstration")
    print("=" * 50)
    
    try:
        # Create enhanced orchestrator
        from multi_agent_orchestration import InMemoryMessageBus
        communication = InMemoryMessageBus()
        orchestrator = EnhancedGraphOrchestrator(communication)
        
        # Create mock agents for testing
        class MockAgent:
            def __init__(self, agent_id, role, max_tasks=3):
                self.agent_id = agent_id
                self.role = role
                self.current_tasks = {}
                self.capabilities = type('obj', (object,), {
                    'max_concurrent_tasks': max_tasks,
                    'processing_time_estimate': 1.0
                })()
        
        # Register agents
        agents = [
            MockAgent("research-1", AgentRole.RESEARCH, 2),
            MockAgent("research-2", AgentRole.RESEARCH, 3),
            MockAgent("planning-1", AgentRole.PLANNING, 2),
            MockAgent("execution-1", AgentRole.EXECUTION, 4)
        ]
        
        for agent in agents:
            orchestrator.register_agent(agent)
        
        print(f"✅ Registered {len(agents)} agents")
        
        # Test load-balanced handoff
        print("\n⚖️ Testing Load-Balanced Handoff...")
        
        # Simulate some load on research-1
        agents[0].current_tasks = {"task1": None, "task2": None}  # Overloaded
        
        handoff_result = await orchestrator.handoff_manager.initiate_handoff(
            from_agent_id="research-1",
            to_agent_role=AgentRole.RESEARCH,
            reason=HandoffReason.AGENT_OVERLOADED,
            task_data={"type": "research_query", "description": "Test research task"}
        )
        
        print(f"   Handoff success: {handoff_result.success}")
        print(f"   Target agent: {handoff_result.new_agent_id}")
        print(f"   Execution time: {handoff_result.execution_time:.3f}s")
        
        # Test skill-based handoff
        print("\n🎯 Testing Skill-Based Handoff...")
        
        skill_strategy = orchestrator.handoff_manager.handoff_strategies['skill_based']
        skill_strategy.register_agent_skills("research-1", {"data_analysis", "machine_learning"})
        skill_strategy.register_agent_skills("research-2", {"natural_language", "research"})
        skill_strategy.set_task_requirements("nlp_task", {"natural_language"})
        
        orchestrator.handoff_manager.set_default_strategy('skill_based')
        
        skill_handoff_result = await orchestrator.handoff_manager.initiate_handoff(
            from_agent_id=None,
            to_agent_role=AgentRole.RESEARCH,
            reason=HandoffReason.SKILL_MISMATCH,
            task_data={"type": "nlp_task", "description": "Natural language processing task"}
        )
        
        print(f"   Skill-based handoff success: {skill_handoff_result.success}")
        print(f"   Selected agent: {skill_handoff_result.new_agent_id}")
        
        # Test agent failure recovery
        print("\n🚨 Testing Agent Failure Recovery...")
        
        # Simulate active tasks for failed agent
        failed_tasks = [
            Task(id="task1", description="Active task 1", requirements={"type": "research_query"}),
            Task(id="task2", description="Active task 2", requirements={"type": "task_planning"})
        ]
        
        recovery_results = await orchestrator.handoff_manager.handle_agent_failure(
            "research-1", failed_tasks
        )
        
        print(f"   Recovery handoffs: {len(recovery_results)}")
        successful_recoveries = sum(1 for r in recovery_results if r.success)
        print(f"   Successful recoveries: {successful_recoveries}/{len(recovery_results)}")
        
        # Display metrics
        print(f"\n📊 Handoff System Metrics:")
        stats = orchestrator.handoff_manager.get_handoff_statistics()
        for key, value in stats.items():
            if key not in ["handoffs_by_reason"]:
                print(f"   {key}: {value}")
        
        # Display recent handoffs
        print(f"\n📋 Recent Handoffs:")
        recent = orchestrator.handoff_manager.get_recent_handoffs(3)
        for handoff in recent:
            print(f"   {handoff['id'][:8]}... {handoff['from_agent']} -> {handoff['to_agent']} ({handoff['reason']})")
        
        # Save demonstration results
        demo_results = {
            "demonstration": "Agent Handoff System",
            "timestamp": datetime.now().isoformat(),
            "handoff_statistics": stats,
            "recent_handoffs": recent,
            "features_tested": [
                "Load-balanced handoff strategy",
                "Skill-based handoff strategy", 
                "Agent failure recovery",
                "Handoff metrics collection",
                "Context preservation during handoffs"
            ]
        }
        
        Path(".taskmaster/reports").mkdir(parents=True, exist_ok=True)
        with open(".taskmaster/reports/agent-handoff-demo.json", 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        print(f"\n📄 Demo results saved to: .taskmaster/reports/agent-handoff-demo.json")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Agent Handoff System demonstration completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    asyncio.run(demonstrate_handoff_system())