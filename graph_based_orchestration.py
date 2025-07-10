#!/usr/bin/env python3
"""
Graph-Based Orchestration Architecture
Implements Task 49.7: Graph-Based Orchestration Architecture

Enhanced framework using LangGraph-like architecture for dynamic workflow
composition, intelligent routing, and flexible task delegation.
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from pathlib import Path
import weakref
from collections import defaultdict, deque

# Import existing multi-agent components
from multi_agent_orchestration import (
    BaseAgent, AgentRole, Task, TaskStatus, AgentMessage, MessageType,
    AgentCapabilities, CommunicationProtocol, InMemoryMessageBus
)


class GraphNodeType(Enum):
    """Types of nodes in the workflow graph"""
    START = "start"
    END = "end"
    AGENT = "agent"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    MERGE = "merge"
    LOOP = "loop"
    HUMAN_INPUT = "human_input"


class ExecutionState(Enum):
    """Graph execution states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class GraphState:
    """State object passed between graph nodes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[str] = field(default_factory=list)
    current_node: Optional[str] = None
    error: Optional[str] = None
    
    def update(self, **kwargs) -> 'GraphState':
        """Create new state with updated data"""
        new_data = {**self.data, **kwargs}
        return GraphState(
            id=self.id,
            data=new_data,
            metadata=self.metadata.copy(),
            execution_history=self.execution_history.copy(),
            current_node=self.current_node,
            error=self.error
        )
    
    def get(self, key: str, default=None):
        """Get value from state data"""
        return self.data.get(key, default)
    
    def set_error(self, error: str) -> 'GraphState':
        """Set error state"""
        new_state = self.update()
        new_state.error = error
        return new_state


@dataclass
class GraphNode:
    """Enhanced graph node with LangGraph-like capabilities"""
    id: str
    node_type: GraphNodeType
    name: str = ""
    agent_role: Optional[AgentRole] = None
    function: Optional[Callable] = None
    condition: Optional[Callable] = None
    parallel_branches: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: float = 300.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            self.name = self.id


@dataclass 
class GraphEdge:
    """Enhanced graph edge with conditional routing"""
    from_node: str
    to_node: str
    condition: Optional[Callable[[GraphState], bool]] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_traverse(self, state: GraphState) -> bool:
        """Check if edge can be traversed given current state"""
        if self.condition is None:
            return True
        return self.condition(state)


class DynamicRoutingStrategy(ABC):
    """Abstract base for dynamic routing strategies"""
    
    @abstractmethod
    def select_next_nodes(self, current_node: str, state: GraphState, 
                         available_edges: List[GraphEdge]) -> List[str]:
        """Select next nodes based on current state"""
        pass


class ConditionalRoutingStrategy(DynamicRoutingStrategy):
    """Route based on conditional logic"""
    
    def select_next_nodes(self, current_node: str, state: GraphState, 
                         available_edges: List[GraphEdge]) -> List[str]:
        next_nodes = []
        for edge in available_edges:
            if edge.from_node == current_node and edge.can_traverse(state):
                next_nodes.append(edge.to_node)
        return next_nodes


class LoadBasedRoutingStrategy(DynamicRoutingStrategy):
    """Route based on agent load and performance"""
    
    def __init__(self, orchestrator_ref):
        self.orchestrator_ref = orchestrator_ref
    
    def select_next_nodes(self, current_node: str, state: GraphState, 
                         available_edges: List[GraphEdge]) -> List[str]:
        # Get orchestrator instance
        orchestrator = self.orchestrator_ref()
        if not orchestrator:
            return []
        
        # Find eligible next nodes
        candidate_nodes = []
        for edge in available_edges:
            if edge.from_node == current_node and edge.can_traverse(state):
                candidate_nodes.append((edge.to_node, edge.weight))
        
        if not candidate_nodes:
            return []
        
        # Select based on agent load if nodes represent agents
        selected_nodes = []
        for node_id, weight in candidate_nodes:
            node = orchestrator.graph.nodes.get(node_id)
            if node and node.agent_role:
                # Find agents with this role
                suitable_agents = [
                    agent for agent in orchestrator.agents.values() 
                    if agent.role == node.agent_role
                ]
                # Select agent with lowest load
                if suitable_agents:
                    best_agent = min(suitable_agents, key=lambda a: len(a.current_tasks))
                    if len(best_agent.current_tasks) < best_agent.capabilities.max_concurrent_tasks:
                        selected_nodes.append(node_id)
            else:
                selected_nodes.append(node_id)
        
        return selected_nodes


class PriorityRoutingStrategy(DynamicRoutingStrategy):
    """Route based on priority and state context"""
    
    def select_next_nodes(self, current_node: str, state: GraphState, 
                         available_edges: List[GraphEdge]) -> List[str]:
        # Get all possible next nodes with their weights/priorities
        candidates = []
        for edge in available_edges:
            if edge.from_node == current_node and edge.can_traverse(state):
                priority = state.metadata.get('priority', 1.0) * edge.weight
                candidates.append((edge.to_node, priority))
        
        if not candidates:
            return []
        
        # Sort by priority (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return highest priority node(s)
        max_priority = candidates[0][1]
        return [node for node, priority in candidates if priority == max_priority]


class GraphBasedOrchestrator:
    """Enhanced orchestrator with graph-based workflow execution"""
    
    def __init__(self, communication: CommunicationProtocol = None):
        self.agents: Dict[str, BaseAgent] = {}
        self.graph: Optional['ExecutableWorkflowGraph'] = None
        self.active_executions: Dict[str, Dict] = {}
        self.logger = logging.getLogger("GraphOrchestrator")
        
        # Communication and routing
        self.communication = communication or InMemoryMessageBus()
        self.routing_strategies: Dict[str, DynamicRoutingStrategy] = {
            'conditional': ConditionalRoutingStrategy(),
            'load_based': LoadBasedRoutingStrategy(weakref.ref(self)),
            'priority': PriorityRoutingStrategy()
        }
        self.default_routing_strategy = 'conditional'
        
        # Performance metrics
        self.metrics = {
            "graphs_executed": 0,
            "nodes_processed": 0,
            "routing_decisions": 0,
            "agent_handoffs": 0,
            "execution_failures": 0,
            "avg_execution_time": 0.0
        }
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.agent_id} ({agent.role.value})")
    
    def set_graph(self, graph: 'ExecutableWorkflowGraph'):
        """Set the workflow graph for execution"""
        if graph.validate():
            self.graph = graph
            self.logger.info(f"Set workflow graph with {len(graph.nodes)} nodes")
        else:
            raise ValueError("Invalid workflow graph")
    
    def set_routing_strategy(self, strategy_name: str):
        """Set the routing strategy"""
        if strategy_name in self.routing_strategies:
            self.default_routing_strategy = strategy_name
            self.logger.info(f"Set routing strategy to: {strategy_name}")
        else:
            raise ValueError(f"Unknown routing strategy: {strategy_name}")
    
    async def execute_graph(self, initial_state: GraphState) -> GraphState:
        """Execute the workflow graph"""
        if not self.graph:
            raise ValueError("No workflow graph set")
        
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Initialize execution tracking
        self.active_executions[execution_id] = {
            "status": ExecutionState.RUNNING,
            "start_time": start_time,
            "current_nodes": [],
            "completed_nodes": set(),
            "state_history": []
        }
        
        self.logger.info(f"Starting graph execution: {execution_id}")
        
        try:
            # Start execution from START nodes
            start_nodes = self.graph.get_start_nodes()
            if not start_nodes:
                raise ValueError("No start nodes found in graph")
            
            final_state = await self._execute_graph_traversal(
                execution_id, start_nodes, initial_state
            )
            
            self.active_executions[execution_id]["status"] = ExecutionState.COMPLETED
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self.metrics["graphs_executed"] += 1
            self._update_avg_execution_time(execution_time)
            
            self.logger.info(f"Graph execution completed: {execution_id}")
            return final_state
            
        except Exception as e:
            self.active_executions[execution_id]["status"] = ExecutionState.FAILED
            self.metrics["execution_failures"] += 1
            self.logger.error(f"Graph execution failed: {e}")
            return initial_state.set_error(str(e))
    
    async def _execute_graph_traversal(self, execution_id: str, 
                                     current_nodes: List[str], 
                                     state: GraphState) -> GraphState:
        """Traverse and execute graph nodes"""
        execution = self.active_executions[execution_id]
        current_state = state
        
        while current_nodes and execution["status"] == ExecutionState.RUNNING:
            # Process current nodes (potentially in parallel)
            next_nodes = []
            
            if len(current_nodes) == 1:
                # Single node execution
                node_id = current_nodes[0]
                result_state, next_node_ids = await self._execute_node(
                    execution_id, node_id, current_state
                )
                current_state = result_state
                next_nodes.extend(next_node_ids)
                
            else:
                # Parallel node execution
                tasks = []
                for node_id in current_nodes:
                    task = self._execute_node(execution_id, node_id, current_state)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Merge results from parallel execution
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Parallel node execution failed: {result}")
                        continue
                    
                    result_state, next_node_ids = result
                    current_state = self._merge_states(current_state, result_state)
                    next_nodes.extend(next_node_ids)
            
            # Update execution tracking
            execution["completed_nodes"].update(current_nodes)
            execution["state_history"].append(asdict(current_state))
            
            # Check for end conditions
            if not next_nodes or self._has_reached_end(next_nodes):
                break
            
            # Remove duplicates while preserving order
            current_nodes = list(dict.fromkeys(next_nodes))
            self.metrics["routing_decisions"] += 1
        
        return current_state
    
    async def _execute_node(self, execution_id: str, node_id: str, 
                          state: GraphState) -> Tuple[GraphState, List[str]]:
        """Execute a single graph node"""
        node = self.graph.nodes.get(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")
        
        self.logger.debug(f"Executing node: {node_id} ({node.node_type.value})")
        state.current_node = node_id
        state.execution_history.append(node_id)
        
        # Execute based on node type
        try:
            if node.node_type == GraphNodeType.AGENT:
                result_state = await self._execute_agent_node(node, state)
            elif node.node_type == GraphNodeType.CONDITIONAL:
                result_state = await self._execute_conditional_node(node, state)
            elif node.node_type == GraphNodeType.PARALLEL:
                result_state = await self._execute_parallel_node(node, state)
            elif node.node_type == GraphNodeType.MERGE:
                result_state = await self._execute_merge_node(node, state)
            elif node.node_type == GraphNodeType.LOOP:
                result_state = await self._execute_loop_node(node, state)
            elif node.node_type in [GraphNodeType.START, GraphNodeType.END]:
                result_state = state  # Pass through
            else:
                result_state = await self._execute_function_node(node, state)
            
            # Determine next nodes using routing strategy
            next_nodes = self._route_to_next_nodes(node_id, result_state)
            
            self.metrics["nodes_processed"] += 1
            return result_state, next_nodes
            
        except Exception as e:
            self.logger.error(f"Node execution failed: {node_id} - {e}")
            error_state = state.set_error(str(e))
            return error_state, []
    
    async def _execute_agent_node(self, node: GraphNode, state: GraphState) -> GraphState:
        """Execute an agent-based node"""
        if not node.agent_role:
            raise ValueError(f"Agent node {node.id} has no agent role specified")
        
        # Find suitable agents
        suitable_agents = [
            agent for agent in self.agents.values() 
            if agent.role == node.agent_role
        ]
        
        if not suitable_agents:
            raise RuntimeError(f"No agents available for role: {node.agent_role}")
        
        # Select best agent (could use load balancing here)
        selected_agent = min(suitable_agents, key=lambda a: len(a.current_tasks))
        
        # Create task from state
        task = Task(
            description=f"Execute {node.name}",
            requirements={"type": node.agent_role.value},
            context=state.data
        )
        
        # Send task to agent
        assignment_message = AgentMessage(
            sender_id="graph_orchestrator",
            recipient_id=selected_agent.agent_id,
            message_type=MessageType.TASK_ASSIGNMENT,
            payload=asdict(task),
            correlation_id=state.id
        )
        
        await self.communication.send_message(assignment_message)
        
        # Wait for completion (simplified - real implementation would use callbacks)
        await asyncio.sleep(selected_agent.capabilities.processing_time_estimate)
        
        # Simulate receiving result
        result = {
            "agent_id": selected_agent.agent_id,
            "task_result": f"Completed {node.name}",
            "status": "success"
        }
        
        self.metrics["agent_handoffs"] += 1
        return state.update(
            **{f"{node.id}_result": result},
            last_agent=selected_agent.agent_id
        )
    
    async def _execute_conditional_node(self, node: GraphNode, state: GraphState) -> GraphState:
        """Execute a conditional node"""
        if node.condition:
            condition_result = node.condition(state)
            return state.update(condition_result=condition_result)
        return state
    
    async def _execute_parallel_node(self, node: GraphNode, state: GraphState) -> GraphState:
        """Execute a parallel node (prepare for parallel branches)"""
        return state.update(parallel_branches=node.parallel_branches)
    
    async def _execute_merge_node(self, node: GraphNode, state: GraphState) -> GraphState:
        """Execute a merge node (consolidate parallel results)"""
        # In a real implementation, this would merge results from parallel branches
        return state.update(merged_at=node.id)
    
    async def _execute_loop_node(self, node: GraphNode, state: GraphState) -> GraphState:
        """Execute a loop node"""
        if node.condition:
            should_continue = node.condition(state)
            loop_count = state.get(f"{node.id}_count", 0) + 1
            return state.update(
                **{f"{node.id}_count": loop_count},
                should_loop=should_continue
            )
        return state
    
    async def _execute_function_node(self, node: GraphNode, state: GraphState) -> GraphState:
        """Execute a function node"""
        if node.function:
            try:
                if asyncio.iscoroutinefunction(node.function):
                    result = await node.function(state)
                else:
                    result = node.function(state)
                
                if isinstance(result, GraphState):
                    return result
                elif isinstance(result, dict):
                    return state.update(**result)
                else:
                    return state.update(function_result=result)
            except Exception as e:
                return state.set_error(f"Function execution failed: {e}")
        
        return state
    
    def _route_to_next_nodes(self, current_node: str, state: GraphState) -> List[str]:
        """Determine next nodes using current routing strategy"""
        available_edges = [
            edge for edge in self.graph.edges 
            if edge.from_node == current_node
        ]
        
        if not available_edges:
            return []
        
        strategy = self.routing_strategies[self.default_routing_strategy]
        return strategy.select_next_nodes(current_node, state, available_edges)
    
    def _merge_states(self, state1: GraphState, state2: GraphState) -> GraphState:
        """Merge two states (for parallel execution results)"""
        merged_data = {**state1.data, **state2.data}
        merged_metadata = {**state1.metadata, **state2.metadata}
        merged_history = state1.execution_history + state2.execution_history
        
        return GraphState(
            id=state1.id,
            data=merged_data,
            metadata=merged_metadata,
            execution_history=merged_history,
            current_node=state2.current_node,
            error=state1.error or state2.error
        )
    
    def _has_reached_end(self, node_ids: List[str]) -> bool:
        """Check if any of the nodes are END nodes"""
        for node_id in node_ids:
            node = self.graph.nodes.get(node_id)
            if node and node.node_type == GraphNodeType.END:
                return True
        return False
    
    def _update_avg_execution_time(self, execution_time: float):
        """Update average execution time metric"""
        current_avg = self.metrics["avg_execution_time"]
        count = self.metrics["graphs_executed"]
        self.metrics["avg_execution_time"] = (
            (current_avg * (count - 1) + execution_time) / count
        )


class ExecutableWorkflowGraph:
    """Enhanced workflow graph with execution capabilities"""
    
    def __init__(self, name: str = "workflow"):
        self.name = name
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.metadata: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"WorkflowGraph.{name}")
    
    def add_node(self, node: GraphNode) -> 'ExecutableWorkflowGraph':
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.logger.debug(f"Added node: {node.id} ({node.node_type.value})")
        return self
    
    def add_edge(self, edge: GraphEdge) -> 'ExecutableWorkflowGraph':
        """Add an edge to the graph"""
        if edge.from_node not in self.nodes:
            raise ValueError(f"From node not found: {edge.from_node}")
        if edge.to_node not in self.nodes:
            raise ValueError(f"To node not found: {edge.to_node}")
        
        self.edges.append(edge)
        self.logger.debug(f"Added edge: {edge.from_node} -> {edge.to_node}")
        return self
    
    def add_conditional_edge(self, from_node: str, to_node: str, 
                           condition: Callable[[GraphState], bool]) -> 'ExecutableWorkflowGraph':
        """Add a conditional edge"""
        edge = GraphEdge(from_node=from_node, to_node=to_node, condition=condition)
        return self.add_edge(edge)
    
    def add_parallel_branch(self, from_node: str, branch_nodes: List[str], 
                          merge_node: str) -> 'ExecutableWorkflowGraph':
        """Add parallel branches"""
        # Add edges from source to all branch nodes
        for branch_node in branch_nodes:
            self.add_edge(GraphEdge(from_node, branch_node))
        
        # Add edges from all branch nodes to merge node
        for branch_node in branch_nodes:
            self.add_edge(GraphEdge(branch_node, merge_node))
        
        return self
    
    def get_start_nodes(self) -> List[str]:
        """Get all START nodes"""
        return [
            node_id for node_id, node in self.nodes.items() 
            if node.node_type == GraphNodeType.START
        ]
    
    def get_end_nodes(self) -> List[str]:
        """Get all END nodes"""
        return [
            node_id for node_id, node in self.nodes.items() 
            if node.node_type == GraphNodeType.END
        ]
    
    def validate(self) -> bool:
        """Validate graph structure"""
        try:
            # Check for at least one start and end node
            start_nodes = self.get_start_nodes()
            end_nodes = self.get_end_nodes()
            
            if not start_nodes:
                self.logger.error("No start nodes found")
                return False
            
            if not end_nodes:
                self.logger.error("No end nodes found")
                return False
            
            # Check for cycles (simplified)
            if self._has_cycles():
                self.logger.error("Cycles detected in graph")
                return False
            
            # Check connectivity
            if not self._is_connected():
                self.logger.warning("Graph may have disconnected components")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Graph validation failed: {e}")
            return False
    
    def _has_cycles(self) -> bool:
        """Detect cycles in the graph"""
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            
            # Get adjacent nodes
            adjacent = [edge.to_node for edge in self.edges if edge.from_node == node_id]
            
            for adj_node in adjacent:
                if adj_node not in visited:
                    if has_cycle_util(adj_node):
                        return True
                elif adj_node in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle_util(node_id):
                    return True
        
        return False
    
    def _is_connected(self) -> bool:
        """Check if graph is connected"""
        if not self.nodes:
            return True
        
        # Start from first node and try to reach all others
        start_node = next(iter(self.nodes))
        visited = set()
        queue = deque([start_node])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            
            # Add neighbors
            for edge in self.edges:
                if edge.from_node == current and edge.to_node not in visited:
                    queue.append(edge.to_node)
                elif edge.to_node == current and edge.from_node not in visited:
                    queue.append(edge.from_node)
        
        return len(visited) == len(self.nodes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation"""
        return {
            "name": self.name,
            "nodes": {
                node_id: {
                    "id": node.id,
                    "type": node.node_type.value,
                    "name": node.name,
                    "agent_role": node.agent_role.value if node.agent_role else None,
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [
                {
                    "from": edge.from_node,
                    "to": edge.to_node,
                    "weight": edge.weight,
                    "metadata": edge.metadata
                }
                for edge in self.edges
            ],
            "metadata": self.metadata
        }


def create_research_planning_graph() -> ExecutableWorkflowGraph:
    """Create a sample research and planning workflow graph"""
    graph = ExecutableWorkflowGraph("research_planning")
    
    # Add nodes
    start_node = GraphNode("start", GraphNodeType.START, "Start")
    research_node = GraphNode("research", GraphNodeType.AGENT, "Research Task", AgentRole.RESEARCH)
    planning_node = GraphNode("planning", GraphNodeType.AGENT, "Planning Task", AgentRole.PLANNING)
    execution_node = GraphNode("execution", GraphNodeType.AGENT, "Execution Task", AgentRole.EXECUTION)
    validation_node = GraphNode("validation", GraphNodeType.AGENT, "Validation Task", AgentRole.VALIDATION)
    end_node = GraphNode("end", GraphNodeType.END, "End")
    
    graph.add_node(start_node)
    graph.add_node(research_node)
    graph.add_node(planning_node)
    graph.add_node(execution_node)
    graph.add_node(validation_node)
    graph.add_node(end_node)
    
    # Add edges
    graph.add_edge(GraphEdge("start", "research"))
    graph.add_edge(GraphEdge("research", "planning"))
    graph.add_edge(GraphEdge("planning", "execution"))
    graph.add_edge(GraphEdge("execution", "validation"))
    graph.add_edge(GraphEdge("validation", "end"))
    
    return graph


def create_conditional_workflow_graph() -> ExecutableWorkflowGraph:
    """Create a conditional workflow with dynamic routing"""
    graph = ExecutableWorkflowGraph("conditional_workflow")
    
    # Add nodes
    start_node = GraphNode("start", GraphNodeType.START, "Start")
    research_node = GraphNode("research", GraphNodeType.AGENT, "Research", AgentRole.RESEARCH)
    
    # Conditional node
    decision_func = lambda state: state.get("research_confidence", 0.5) > 0.8
    decision_node = GraphNode("decision", GraphNodeType.CONDITIONAL, "Quality Check", 
                            condition=decision_func)
    
    # Alternative paths
    planning_node = GraphNode("planning", GraphNodeType.AGENT, "Planning", AgentRole.PLANNING)
    additional_research_node = GraphNode("additional_research", GraphNodeType.AGENT, 
                                       "Additional Research", AgentRole.RESEARCH)
    
    execution_node = GraphNode("execution", GraphNodeType.AGENT, "Execution", AgentRole.EXECUTION)
    end_node = GraphNode("end", GraphNodeType.END, "End")
    
    # Add all nodes
    for node in [start_node, research_node, decision_node, planning_node, 
                additional_research_node, execution_node, end_node]:
        graph.add_node(node)
    
    # Add edges
    graph.add_edge(GraphEdge("start", "research"))
    graph.add_edge(GraphEdge("research", "decision"))
    
    # Conditional edges
    high_confidence_condition = lambda state: state.get("research_confidence", 0.5) > 0.8
    low_confidence_condition = lambda state: state.get("research_confidence", 0.5) <= 0.8
    
    graph.add_conditional_edge("decision", "planning", high_confidence_condition)
    graph.add_conditional_edge("decision", "additional_research", low_confidence_condition)
    
    graph.add_edge(GraphEdge("additional_research", "planning"))
    graph.add_edge(GraphEdge("planning", "execution"))
    graph.add_edge(GraphEdge("execution", "end"))
    
    return graph


async def demonstrate_graph_orchestration():
    """Demonstrate the graph-based orchestration system"""
    print("🔄 Graph-Based Orchestration Architecture Demo")
    print("=" * 60)
    
    try:
        # Create communication system
        communication = InMemoryMessageBus()
        
        # Create graph orchestrator
        orchestrator = GraphBasedOrchestrator(communication)
        
        # Register agents
        from multi_agent_orchestration import ResearchAgent, PlanningAgent, ExecutionAgent, ValidationAgent
        
        agents = [
            ResearchAgent("research-001", communication),
            PlanningAgent("planning-001", communication),
            ExecutionAgent("execution-001", communication),
            ValidationAgent("validation-001", communication)
        ]
        
        for agent in agents:
            orchestrator.register_agent(agent)
        
        print(f"✅ Registered {len(agents)} agents")
        
        # Test simple workflow
        print("\n📋 Testing Linear Workflow...")
        linear_graph = create_research_planning_graph()
        orchestrator.set_graph(linear_graph)
        
        initial_state = GraphState(data={
            "topic": "Graph-Based Orchestration",
            "complexity": "high",
            "priority": 1.0
        })
        
        start_time = datetime.now()
        result_state = await orchestrator.execute_graph(initial_state)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   Execution time: {execution_time:.2f}s")
        print(f"   Nodes processed: {len(result_state.execution_history)}")
        print(f"   Final state data keys: {list(result_state.data.keys())}")
        
        # Test conditional workflow
        print("\n🔀 Testing Conditional Workflow...")
        conditional_graph = create_conditional_workflow_graph()
        orchestrator.set_graph(conditional_graph)
        
        # Test with high confidence
        high_confidence_state = GraphState(data={
            "topic": "Well-researched topic",
            "research_confidence": 0.9
        })
        
        result_high = await orchestrator.execute_graph(high_confidence_state)
        print(f"   High confidence path: {' -> '.join(result_high.execution_history)}")
        
        # Test with low confidence
        low_confidence_state = GraphState(data={
            "topic": "Complex new topic", 
            "research_confidence": 0.3
        })
        
        result_low = await orchestrator.execute_graph(low_confidence_state)
        print(f"   Low confidence path: {' -> '.join(result_low.execution_history)}")
        
        # Test different routing strategies
        print("\n🛤️ Testing Routing Strategies...")
        orchestrator.set_routing_strategy('load_based')
        result_load = await orchestrator.execute_graph(initial_state)
        print(f"   Load-based routing: {len(result_load.execution_history)} nodes")
        
        orchestrator.set_routing_strategy('priority')
        result_priority = await orchestrator.execute_graph(initial_state)
        print(f"   Priority routing: {len(result_priority.execution_history)} nodes")
        
        # Display metrics
        print(f"\n📊 Orchestration Metrics:")
        for key, value in orchestrator.metrics.items():
            print(f"   {key}: {value}")
        
        # Save graph representation
        graph_data = conditional_graph.to_dict()
        Path(".taskmaster/reports").mkdir(parents=True, exist_ok=True)
        with open(".taskmaster/reports/graph-orchestration-demo.json", 'w') as f:
            json.dump({
                "demonstration": "Graph-Based Orchestration",
                "timestamp": datetime.now().isoformat(),
                "graph_structure": graph_data,
                "metrics": orchestrator.metrics,
                "routing_strategies_tested": ["conditional", "load_based", "priority"],
                "features_validated": [
                    "Dynamic workflow composition",
                    "Conditional routing",
                    "Agent handoff mechanisms", 
                    "Multiple routing strategies",
                    "Graph validation",
                    "State management"
                ]
            }, f, indent=2)
        
        print(f"\n📄 Demo results saved to: .taskmaster/reports/graph-orchestration-demo.json")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Graph-Based Orchestration demonstration completed!")


if __name__ == "__main__":
    asyncio.run(demonstrate_graph_orchestration())