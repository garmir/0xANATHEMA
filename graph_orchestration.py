#!/usr/bin/env python3
"""
Graph-Based Orchestration Architecture
Atomic Task 49.7: Implement Graph-Based Orchestration Architecture

This module implements a sophisticated graph-based workflow orchestration system
that enables dynamic routing, parallel execution, and complex dependency management.
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
import networkx as nx
import heapq
from collections import defaultdict, deque


class NodeType(Enum):
    """Types of workflow nodes"""
    TASK = "task"
    DECISION = "decision"
    PARALLEL = "parallel"
    SYNCHRONIZATION = "synchronization"
    CONDITION = "condition"
    LOOP = "loop"
    SUBWORKFLOW = "subworkflow"


class NodeStatus(Enum):
    """Node execution status"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class EdgeType(Enum):
    """Types of workflow edges"""
    SEQUENCE = "sequence"
    CONDITIONAL = "conditional"
    PARALLEL_SPLIT = "parallel_split"
    PARALLEL_JOIN = "parallel_join"
    DATA_FLOW = "data_flow"
    ERROR_HANDLING = "error_handling"


class ExecutionMode(Enum):
    """Workflow execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    STREAMING = "streaming"


@dataclass
class WorkflowData:
    """Data container for workflow execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    version: int = 1
    
    def update(self, key: str, value: Any, metadata: Dict[str, Any] = None):
        """Update data with versioning"""
        self.data[key] = value
        self.last_modified = datetime.now()
        self.version += 1
        if metadata:
            self.metadata.update(metadata)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get data value"""
        return self.data.get(key, default)
    
    def merge(self, other_data: 'WorkflowData'):
        """Merge with another workflow data"""
        self.data.update(other_data.data)
        self.metadata.update(other_data.metadata)
        self.last_modified = datetime.now()
        self.version += 1


@dataclass
class WorkflowNode:
    """Workflow graph node"""
    id: str
    node_type: NodeType
    name: str = ""
    description: str = ""
    status: NodeStatus = NodeStatus.PENDING
    processor: Optional[Callable] = None
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    execution_config: Dict[str, Any] = field(default_factory=dict)
    retry_config: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[timedelta] = None
    dependencies: Set[str] = field(default_factory=set)
    conditions: List[str] = field(default_factory=list)
    
    # Runtime data
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    result_data: Optional[WorkflowData] = None
    
    def can_execute(self, workflow_data: WorkflowData, 
                   completed_nodes: Set[str]) -> bool:
        """Check if node can be executed"""
        if self.status not in [NodeStatus.PENDING, NodeStatus.READY]:
            return False
        
        # Check dependencies
        if not self.dependencies.issubset(completed_nodes):
            return False
        
        # Check conditions
        for condition in self.conditions:
            if not self._evaluate_condition(condition, workflow_data):
                return False
        
        return True
    
    def _evaluate_condition(self, condition: str, data: WorkflowData) -> bool:
        """Evaluate execution condition"""
        try:
            # Simple condition evaluation (in production, use a proper expression parser)
            # Example: "data.status == 'approved'"
            context = {"data": data.data, "metadata": data.metadata}
            return eval(condition, {"__builtins__": {}}, context)
        except Exception:
            return False
    
    def should_retry(self) -> bool:
        """Check if node should be retried"""
        max_retries = self.retry_config.get("max_retries", 3)
        return self.error_count < max_retries
    
    def get_retry_delay(self) -> float:
        """Get retry delay in seconds"""
        base_delay = self.retry_config.get("base_delay", 1.0)
        backoff_factor = self.retry_config.get("backoff_factor", 2.0)
        return base_delay * (backoff_factor ** self.error_count)


@dataclass
class WorkflowEdge:
    """Workflow graph edge"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node: str = ""
    target_node: str = ""
    edge_type: EdgeType = EdgeType.SEQUENCE
    condition: Optional[str] = None
    weight: float = 1.0
    data_mapping: Dict[str, str] = field(default_factory=dict)
    
    def can_traverse(self, workflow_data: WorkflowData) -> bool:
        """Check if edge can be traversed"""
        if not self.condition:
            return True
        
        try:
            context = {"data": workflow_data.data, "metadata": workflow_data.metadata}
            return eval(self.condition, {"__builtins__": {}}, context)
        except Exception:
            return False
    
    def transform_data(self, input_data: WorkflowData) -> WorkflowData:
        """Transform data according to edge mapping"""
        if not self.data_mapping:
            return input_data
        
        output_data = WorkflowData()
        for target_key, source_key in self.data_mapping.items():
            if source_key in input_data.data:
                output_data.update(target_key, input_data.get(source_key))
        
        return output_data


class WorkflowGraph:
    """Graph-based workflow representation"""
    
    def __init__(self, workflow_id: str, name: str = ""):
        self.workflow_id = workflow_id
        self.name = name
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[str, WorkflowEdge] = {}
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
        
    def add_node(self, node: WorkflowNode) -> bool:
        """Add node to workflow graph"""
        if node.id in self.nodes:
            return False
        
        self.nodes[node.id] = node
        self.graph.add_node(node.id, node_data=node)
        return True
    
    def add_edge(self, edge: WorkflowEdge) -> bool:
        """Add edge to workflow graph"""
        if (edge.source_node not in self.nodes or 
            edge.target_node not in self.nodes):
            return False
        
        self.edges[edge.id] = edge
        self.graph.add_edge(
            edge.source_node, 
            edge.target_node,
            edge_data=edge
        )
        
        # Update target node dependencies
        self.nodes[edge.target_node].dependencies.add(edge.source_node)
        return True
    
    def get_ready_nodes(self, completed_nodes: Set[str], 
                       workflow_data: WorkflowData) -> List[WorkflowNode]:
        """Get nodes ready for execution"""
        ready_nodes = []
        for node in self.nodes.values():
            if node.can_execute(workflow_data, completed_nodes):
                ready_nodes.append(node)
        return ready_nodes
    
    def get_successors(self, node_id: str) -> List[WorkflowNode]:
        """Get successor nodes"""
        successor_ids = list(self.graph.successors(node_id))
        return [self.nodes[nid] for nid in successor_ids]
    
    def get_predecessors(self, node_id: str) -> List[WorkflowNode]:
        """Get predecessor nodes"""
        predecessor_ids = list(self.graph.predecessors(node_id))
        return [self.nodes[nid] for nid in predecessor_ids]
    
    def find_cycles(self) -> List[List[str]]:
        """Find cycles in the workflow graph"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except Exception:
            return []
    
    def validate_graph(self) -> List[str]:
        """Validate workflow graph"""
        issues = []
        
        # Check for cycles
        cycles = self.find_cycles()
        if cycles:
            issues.append(f"Graph contains cycles: {cycles}")
        
        # Check for unreachable nodes
        if self.nodes:
            start_nodes = [n for n in self.nodes.values() if not n.dependencies]
            if not start_nodes:
                issues.append("No start nodes found (all nodes have dependencies)")
            
            # Check reachability
            reachable = set()
            for start_node in start_nodes:
                reachable.update(nx.descendants(self.graph, start_node.id))
                reachable.add(start_node.id)
            
            unreachable = set(self.nodes.keys()) - reachable
            if unreachable:
                issues.append(f"Unreachable nodes: {unreachable}")
        
        # Check edge validity
        for edge in self.edges.values():
            if edge.source_node not in self.nodes:
                issues.append(f"Edge {edge.id} references missing source node: {edge.source_node}")
            if edge.target_node not in self.nodes:
                issues.append(f"Edge {edge.id} references missing target node: {edge.target_node}")
        
        return issues
    
    def get_execution_order(self) -> List[List[str]]:
        """Get topological execution order (levels for parallel execution)"""
        try:
            # Get topological sort
            topo_order = list(nx.topological_sort(self.graph))
            
            # Group nodes by execution level
            levels = []
            remaining = set(topo_order)
            
            while remaining:
                # Find nodes with no dependencies in remaining set
                current_level = []
                for node_id in topo_order:
                    if node_id in remaining:
                        deps = self.nodes[node_id].dependencies
                        if deps.issubset(set(topo_order) - remaining):
                            current_level.append(node_id)
                
                if not current_level:
                    # Break potential deadlock
                    current_level = [remaining.pop()]
                
                levels.append(current_level)
                remaining -= set(current_level)
            
            return levels
        except nx.NetworkXError:
            return []
    
    def export_graph(self) -> Dict[str, Any]:
        """Export graph to dictionary"""
        # Custom serialization to avoid recursion issues
        nodes_data = {}
        for nid, node in self.nodes.items():
            node_data = {
                "id": node.id,
                "node_type": node.node_type.value,
                "name": node.name,
                "description": node.description,
                "status": node.status.value,
                "execution_config": node.execution_config,
                "retry_config": node.retry_config,
                "timeout": node.timeout.total_seconds() if node.timeout else None,
                "dependencies": list(node.dependencies),
                "conditions": node.conditions,
                "execution_count": node.execution_count,
                "error_count": node.error_count
            }
            nodes_data[nid] = node_data
        
        edges_data = {}
        for eid, edge in self.edges.items():
            edge_data = {
                "id": edge.id,
                "source_node": edge.source_node,
                "target_node": edge.target_node,
                "edge_type": edge.edge_type.value,
                "condition": edge.condition,
                "weight": edge.weight,
                "data_mapping": edge.data_mapping
            }
            edges_data[eid] = edge_data
        
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "nodes": nodes_data,
            "edges": edges_data,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def import_graph(cls, graph_data: Dict[str, Any]) -> 'WorkflowGraph':
        """Import graph from dictionary"""
        workflow = cls(graph_data["workflow_id"], graph_data["name"])
        workflow.metadata = graph_data.get("metadata", {})
        
        # Import nodes
        for node_data in graph_data["nodes"].values():
            node = WorkflowNode(**node_data)
            workflow.add_node(node)
        
        # Import edges
        for edge_data in graph_data["edges"].values():
            edge = WorkflowEdge(**edge_data)
            workflow.add_edge(edge)
        
        return workflow


class NodeProcessor(ABC):
    """Abstract base class for node processors"""
    
    @abstractmethod
    async def process(self, node: WorkflowNode, 
                     input_data: WorkflowData) -> WorkflowData:
        """Process node with input data"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: WorkflowData, 
                      schema: Dict[str, Any]) -> bool:
        """Validate input data against schema"""
        pass


class TaskNodeProcessor(NodeProcessor):
    """Processor for task nodes"""
    
    def __init__(self, task_executor: Callable):
        self.task_executor = task_executor
        self.logger = logging.getLogger("TaskNodeProcessor")
    
    async def process(self, node: WorkflowNode, 
                     input_data: WorkflowData) -> WorkflowData:
        """Process task node"""
        try:
            self.logger.info(f"Processing task node: {node.name}")
            
            # Validate input
            if not self.validate_input(input_data, node.input_schema):
                raise ValueError("Input validation failed")
            
            # Execute task
            if asyncio.iscoroutinefunction(self.task_executor):
                result = await self.task_executor(node, input_data)
            else:
                result = self.task_executor(node, input_data)
            
            # Ensure result is WorkflowData
            if not isinstance(result, WorkflowData):
                output_data = WorkflowData()
                output_data.update("result", result)
                return output_data
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing task node {node.name}: {e}")
            raise
    
    def validate_input(self, input_data: WorkflowData, 
                      schema: Dict[str, Any]) -> bool:
        """Validate input data"""
        if not schema:
            return True
        
        # Simple schema validation (in production, use jsonschema)
        for field, requirements in schema.items():
            if requirements.get("required", False):
                if field not in input_data.data:
                    return False
        
        return True


class DecisionNodeProcessor(NodeProcessor):
    """Processor for decision nodes"""
    
    def __init__(self):
        self.logger = logging.getLogger("DecisionNodeProcessor")
    
    async def process(self, node: WorkflowNode, 
                     input_data: WorkflowData) -> WorkflowData:
        """Process decision node"""
        self.logger.info(f"Processing decision node: {node.name}")
        
        decision_logic = node.execution_config.get("decision_logic", "")
        if not decision_logic:
            raise ValueError("Decision node missing decision logic")
        
        try:
            # Evaluate decision
            context = {"data": input_data.data, "metadata": input_data.metadata}
            decision_result = eval(decision_logic, {"__builtins__": {}}, context)
            
            output_data = WorkflowData()
            output_data.update("decision", decision_result)
            output_data.update("original_data", input_data.data)
            
            return output_data
            
        except Exception as e:
            self.logger.error(f"Error in decision logic: {e}")
            raise
    
    def validate_input(self, input_data: WorkflowData, 
                      schema: Dict[str, Any]) -> bool:
        """Validate input data"""
        return True  # Decision nodes are flexible


class ParallelNodeProcessor(NodeProcessor):
    """Processor for parallel execution nodes"""
    
    def __init__(self):
        self.logger = logging.getLogger("ParallelNodeProcessor")
    
    async def process(self, node: WorkflowNode, 
                     input_data: WorkflowData) -> WorkflowData:
        """Process parallel node (split or join)"""
        self.logger.info(f"Processing parallel node: {node.name}")
        
        parallel_type = node.execution_config.get("type", "split")
        
        if parallel_type == "split":
            # Split data for parallel processing
            split_count = node.execution_config.get("split_count", 2)
            output_data = WorkflowData()
            
            for i in range(split_count):
                branch_data = WorkflowData()
                branch_data.data = input_data.data.copy()
                branch_data.update("branch_id", i)
                output_data.update(f"branch_{i}", branch_data.data)
            
            return output_data
            
        elif parallel_type == "join":
            # Join results from parallel branches
            output_data = WorkflowData()
            joined_results = []
            
            for key, value in input_data.data.items():
                if key.startswith("branch_"):
                    joined_results.append(value)
            
            output_data.update("joined_results", joined_results)
            return output_data
        
        else:
            raise ValueError(f"Unknown parallel type: {parallel_type}")
    
    def validate_input(self, input_data: WorkflowData, 
                      schema: Dict[str, Any]) -> bool:
        """Validate input data"""
        return True


class WorkflowExecutor:
    """Orchestrator for executing workflow graphs"""
    
    def __init__(self):
        self.processors: Dict[NodeType, NodeProcessor] = {}
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("WorkflowExecutor")
        
        # Register default processors
        self._register_default_processors()
    
    def _register_default_processors(self):
        """Register default node processors"""
        self.processors[NodeType.DECISION] = DecisionNodeProcessor()
        self.processors[NodeType.PARALLEL] = ParallelNodeProcessor()
    
    def register_processor(self, node_type: NodeType, processor: NodeProcessor):
        """Register custom node processor"""
        self.processors[node_type] = processor
        self.logger.info(f"Registered processor for {node_type.value}")
    
    async def execute_workflow(self, workflow: WorkflowGraph, 
                             initial_data: WorkflowData = None,
                             execution_mode: ExecutionMode = ExecutionMode.PARALLEL) -> Dict[str, Any]:
        """Execute workflow graph"""
        execution_id = str(uuid.uuid4())
        self.logger.info(f"Starting workflow execution: {execution_id}")
        
        # Validate workflow
        validation_issues = workflow.validate_graph()
        if validation_issues:
            raise ValueError(f"Workflow validation failed: {validation_issues}")
        
        # Initialize execution state
        execution_state = {
            "execution_id": execution_id,
            "workflow_id": workflow.workflow_id,
            "status": "running",
            "start_time": datetime.now(),
            "completed_nodes": set(),
            "failed_nodes": set(),
            "current_data": initial_data or WorkflowData(),
            "node_results": {},
            "execution_mode": execution_mode
        }
        
        self.active_executions[execution_id] = execution_state
        
        try:
            if execution_mode == ExecutionMode.PARALLEL:
                result = await self._execute_parallel(workflow, execution_state)
            elif execution_mode == ExecutionMode.SEQUENTIAL:
                result = await self._execute_sequential(workflow, execution_state)
            else:
                raise ValueError(f"Unsupported execution mode: {execution_mode}")
            
            execution_state["status"] = "completed"
            execution_state["end_time"] = datetime.now()
            
            self.logger.info(f"Workflow execution completed: {execution_id}")
            return result
            
        except Exception as e:
            execution_state["status"] = "failed"
            execution_state["error"] = str(e)
            execution_state["end_time"] = datetime.now()
            self.logger.error(f"Workflow execution failed: {execution_id}, error: {e}")
            raise
        
        finally:
            # Move to history
            self.execution_history.append(execution_state.copy())
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def _execute_parallel(self, workflow: WorkflowGraph, 
                               execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with parallel processing"""
        completed_nodes = execution_state["completed_nodes"]
        current_data = execution_state["current_data"]
        
        # Get execution levels
        execution_levels = workflow.get_execution_order()
        
        for level_index, level_nodes in enumerate(execution_levels):
            self.logger.info(f"Executing level {level_index}: {level_nodes}")
            
            # Execute all nodes in current level in parallel
            level_tasks = []
            for node_id in level_nodes:
                node = workflow.nodes[node_id]
                if node.can_execute(current_data, completed_nodes):
                    task = self._execute_node(node, current_data, execution_state)
                    level_tasks.append((node_id, task))
            
            # Wait for all level tasks to complete
            level_results = await asyncio.gather(
                *[task for _, task in level_tasks],
                return_exceptions=True
            )
            
            # Process results
            for (node_id, _), result in zip(level_tasks, level_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Node {node_id} failed: {result}")
                    execution_state["failed_nodes"].add(node_id)
                    workflow.nodes[node_id].status = NodeStatus.FAILED
                    workflow.nodes[node_id].last_error = str(result)
                else:
                    completed_nodes.add(node_id)
                    execution_state["node_results"][node_id] = result
                    workflow.nodes[node_id].status = NodeStatus.COMPLETED
                    workflow.nodes[node_id].result_data = result
                    
                    # Merge result data into current data
                    current_data.merge(result)
        
        return {
            "execution_id": execution_state["execution_id"],
            "completed_nodes": list(completed_nodes),
            "failed_nodes": list(execution_state["failed_nodes"]),
            "final_data": current_data,
            "node_results": execution_state["node_results"]
        }
    
    async def _execute_sequential(self, workflow: WorkflowGraph,
                                 execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow sequentially"""
        completed_nodes = execution_state["completed_nodes"]
        current_data = execution_state["current_data"]
        
        # Get topological order
        try:
            execution_order = list(nx.topological_sort(workflow.graph))
        except nx.NetworkXError:
            raise ValueError("Cannot execute workflow with cycles sequentially")
        
        for node_id in execution_order:
            node = workflow.nodes[node_id]
            
            if node.can_execute(current_data, completed_nodes):
                try:
                    result = await self._execute_node(node, current_data, execution_state)
                    completed_nodes.add(node_id)
                    execution_state["node_results"][node_id] = result
                    node.status = NodeStatus.COMPLETED
                    node.result_data = result
                    
                    # Merge result into current data
                    current_data.merge(result)
                    
                except Exception as e:
                    self.logger.error(f"Node {node_id} failed: {e}")
                    execution_state["failed_nodes"].add(node_id)
                    node.status = NodeStatus.FAILED
                    node.last_error = str(e)
                    
                    # Continue or fail based on error handling policy
                    if node.execution_config.get("continue_on_error", False):
                        continue
                    else:
                        raise
        
        return {
            "execution_id": execution_state["execution_id"],
            "completed_nodes": list(completed_nodes),
            "failed_nodes": list(execution_state["failed_nodes"]),
            "final_data": current_data,
            "node_results": execution_state["node_results"]
        }
    
    async def _execute_node(self, node: WorkflowNode, input_data: WorkflowData,
                           execution_state: Dict[str, Any]) -> WorkflowData:
        """Execute individual node"""
        self.logger.debug(f"Executing node: {node.name} ({node.id})")
        
        node.status = NodeStatus.RUNNING
        node.start_time = datetime.now()
        node.execution_count += 1
        
        try:
            # Get processor for node type
            processor = self.processors.get(node.node_type)
            if not processor:
                # For task nodes, create processor if one exists
                if node.node_type == NodeType.TASK and node.processor:
                    processor = TaskNodeProcessor(node.processor)
                else:
                    raise ValueError(f"No processor for node type: {node.node_type}")
            
            # Execute with timeout
            if node.timeout:
                result = await asyncio.wait_for(
                    processor.process(node, input_data),
                    timeout=node.timeout.total_seconds()
                )
            else:
                result = await processor.process(node, input_data)
            
            node.end_time = datetime.now()
            return result
            
        except Exception as e:
            node.error_count += 1
            node.last_error = str(e)
            node.end_time = datetime.now()
            
            # Retry if configured
            if node.should_retry():
                self.logger.warning(f"Retrying node {node.name} (attempt {node.error_count})")
                await asyncio.sleep(node.get_retry_delay())
                return await self._execute_node(node, input_data, execution_state)
            
            raise
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status"""
        return self.active_executions.get(execution_id)
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_history[-limit:]


class WorkflowBuilder:
    """Builder for creating workflow graphs"""
    
    def __init__(self, workflow_id: str = None, name: str = ""):
        self.workflow = WorkflowGraph(
            workflow_id or str(uuid.uuid4()), 
            name
        )
        self.logger = logging.getLogger("WorkflowBuilder")
    
    def add_task_node(self, node_id: str, name: str, processor: Callable,
                     **kwargs) -> 'WorkflowBuilder':
        """Add task node to workflow"""
        node = WorkflowNode(
            id=node_id,
            node_type=NodeType.TASK,
            name=name,
            processor=processor,
            **kwargs
        )
        self.workflow.add_node(node)
        return self
    
    def add_decision_node(self, node_id: str, name: str, decision_logic: str,
                         **kwargs) -> 'WorkflowBuilder':
        """Add decision node to workflow"""
        node = WorkflowNode(
            id=node_id,
            node_type=NodeType.DECISION,
            name=name,
            execution_config={"decision_logic": decision_logic},
            **kwargs
        )
        self.workflow.add_node(node)
        return self
    
    def add_parallel_node(self, node_id: str, name: str, parallel_type: str = "split",
                         **kwargs) -> 'WorkflowBuilder':
        """Add parallel node to workflow"""
        node = WorkflowNode(
            id=node_id,
            node_type=NodeType.PARALLEL,
            name=name,
            execution_config={"type": parallel_type},
            **kwargs
        )
        self.workflow.add_node(node)
        return self
    
    def connect(self, from_node: str, to_node: str, 
               edge_type: EdgeType = EdgeType.SEQUENCE,
               condition: str = None, **kwargs) -> 'WorkflowBuilder':
        """Connect two nodes with an edge"""
        edge = WorkflowEdge(
            source_node=from_node,
            target_node=to_node,
            edge_type=edge_type,
            condition=condition,
            **kwargs
        )
        self.workflow.add_edge(edge)
        return self
    
    def build(self) -> WorkflowGraph:
        """Build and validate workflow"""
        validation_issues = self.workflow.validate_graph()
        if validation_issues:
            self.logger.warning(f"Workflow validation issues: {validation_issues}")
        
        return self.workflow


# Export key classes
__all__ = [
    "NodeType", "NodeStatus", "EdgeType", "ExecutionMode", "WorkflowData",
    "WorkflowNode", "WorkflowEdge", "WorkflowGraph", "NodeProcessor",
    "TaskNodeProcessor", "DecisionNodeProcessor", "ParallelNodeProcessor",
    "WorkflowExecutor", "WorkflowBuilder"
]


if __name__ == "__main__":
    # Demo usage
    async def demo():
        logging.basicConfig(level=logging.INFO)
        
        # Create sample task processors
        async def research_task(node: WorkflowNode, data: WorkflowData) -> WorkflowData:
            print(f"🔬 Executing research task: {node.name}")
            await asyncio.sleep(1.0)
            result = WorkflowData()
            result.update("research_findings", ["finding1", "finding2", "finding3"])
            result.update("confidence_score", 0.85)
            return result
        
        async def planning_task(node: WorkflowNode, data: WorkflowData) -> WorkflowData:
            print(f"📋 Executing planning task: {node.name}")
            await asyncio.sleep(0.8)
            result = WorkflowData()
            result.update("execution_plan", {"phases": ["prep", "impl", "test"]})
            result.update("estimated_time", "6 hours")
            return result
        
        async def execution_task(node: WorkflowNode, data: WorkflowData) -> WorkflowData:
            print(f"⚡ Executing implementation task: {node.name}")
            await asyncio.sleep(1.5)
            result = WorkflowData()
            result.update("implementation_status", "completed")
            result.update("test_results", {"passed": 95, "failed": 5})
            return result
        
        async def validation_task(node: WorkflowNode, data: WorkflowData) -> WorkflowData:
            print(f"✅ Executing validation task: {node.name}")
            await asyncio.sleep(1.2)
            result = WorkflowData()
            result.update("validation_status", "approved")
            result.update("quality_score", 0.92)
            return result
        
        # Build workflow using builder pattern
        print("🏗️ Building Graph-Based Workflow")
        builder = WorkflowBuilder(name="Multi-Agent Development Workflow")
        
        workflow = (builder
            .add_task_node("research", "Research Phase", research_task)
            .add_decision_node("quality_check", "Quality Gate", 
                             "data.get('confidence_score', 0) >= 0.8")
            .add_task_node("planning", "Planning Phase", planning_task)
            .add_parallel_node("parallel_split", "Parallel Split", "split")
            .add_task_node("implementation", "Implementation", execution_task)
            .add_task_node("testing", "Testing", execution_task)
            .add_parallel_node("parallel_join", "Parallel Join", "join")
            .add_task_node("validation", "Final Validation", validation_task)
            
            # Connect nodes
            .connect("research", "quality_check")
            .connect("quality_check", "planning", 
                    condition="data.get('decision') == True")
            .connect("planning", "parallel_split")
            .connect("parallel_split", "implementation")
            .connect("parallel_split", "testing")
            .connect("implementation", "parallel_join")
            .connect("testing", "parallel_join")
            .connect("parallel_join", "validation")
            .build()
        )
        
        print(f"📊 Workflow created with {len(workflow.nodes)} nodes and {len(workflow.edges)} edges")
        
        # Validate workflow
        validation_issues = workflow.validate_graph()
        if validation_issues:
            print(f"⚠️ Validation issues: {validation_issues}")
        else:
            print("✅ Workflow validation passed")
        
        # Show execution order
        execution_levels = workflow.get_execution_order()
        print(f"📋 Execution levels: {execution_levels}")
        
        # Execute workflow
        print("\n🚀 Executing Workflow")
        executor = WorkflowExecutor()
        
        initial_data = WorkflowData()
        initial_data.update("project_requirements", "Build AI agent system")
        initial_data.update("complexity", "high")
        
        try:
            result = await executor.execute_workflow(
                workflow, 
                initial_data, 
                ExecutionMode.PARALLEL
            )
            
            print(f"✅ Workflow execution completed!")
            print(f"   Execution ID: {result['execution_id']}")
            print(f"   Completed nodes: {len(result['completed_nodes'])}")
            print(f"   Failed nodes: {len(result['failed_nodes'])}")
            print(f"   Final data keys: {list(result['final_data'].data.keys())}")
            
            # Show execution times
            print(f"\n⏱️ Node Execution Times:")
            for node_id, node in workflow.nodes.items():
                if node.start_time and node.end_time:
                    duration = (node.end_time - node.start_time).total_seconds()
                    print(f"   {node.name}: {duration:.2f}s")
                    
        except Exception as e:
            print(f"❌ Workflow execution failed: {e}")
        
        # Export workflow
        print(f"\n💾 Workflow Export Size: {len(str(workflow.export_graph()))} characters")
    
    # Run demo
    asyncio.run(demo())