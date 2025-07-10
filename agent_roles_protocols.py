#!/usr/bin/env python3
"""
Agent Roles and Interaction Protocols
Atomic Task 49.6: Design Agent Roles and Interaction Protocols

This module defines specialized agent roles and formal interaction protocols
for agent-to-agent communication, including handoff logic and context sharing.
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


class AgentRole(Enum):
    """Specialized agent role definitions"""
    RESEARCH = "research"
    PLANNING = "planning" 
    EXECUTION = "execution"
    VALIDATION = "validation"
    ORCHESTRATOR = "orchestrator"
    MONITOR = "monitor"


class InteractionType(Enum):
    """Types of agent interactions"""
    HANDOFF = "handoff"
    COLLABORATION = "collaboration"
    CONSULTATION = "consultation"
    DELEGATION = "delegation"
    FEEDBACK = "feedback"
    ESCALATION = "escalation"


class ContextScope(Enum):
    """Scope of context sharing"""
    PRIVATE = "private"
    TASK_LOCAL = "task_local"
    WORKFLOW_GLOBAL = "workflow_global"
    SYSTEM_WIDE = "system_wide"


@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    input_types: Set[str] = field(default_factory=set)
    output_types: Set[str] = field(default_factory=set)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    def can_handle(self, input_type: str) -> bool:
        """Check if capability can handle input type"""
        return input_type in self.input_types or len(self.input_types) == 0
    
    def produces(self, output_type: str) -> bool:
        """Check if capability produces output type"""
        return output_type in self.output_types


@dataclass
class ContextData:
    """Context data structure for sharing between agents"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scope: ContextScope = ContextScope.TASK_LOCAL
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_permissions: Set[str] = field(default_factory=set)
    read_count: int = 0
    max_reads: Optional[int] = None
    
    def is_accessible_by(self, agent_id: str) -> bool:
        """Check if agent has access to this context"""
        if self.scope == ContextScope.SYSTEM_WIDE:
            return True
        if not self.access_permissions:
            return True
        return agent_id in self.access_permissions
    
    def is_expired(self) -> bool:
        """Check if context has expired"""
        return (self.expires_at and datetime.now() > self.expires_at) or \
               (self.max_reads and self.read_count >= self.max_reads)
    
    def increment_read(self) -> bool:
        """Increment read count, return False if max reads exceeded"""
        if self.max_reads and self.read_count >= self.max_reads:
            return False
        self.read_count += 1
        return True


@dataclass
class HandoffContext:
    """Context for agent handoffs"""
    from_agent: str
    to_agent: str
    task_id: str
    handoff_reason: str
    context_data: List[ContextData] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    rollback_plan: Optional[str] = None
    timeout: Optional[timedelta] = None
    priority: int = 1


@dataclass
class InteractionProtocol:
    """Protocol definition for agent interactions"""
    interaction_type: InteractionType
    initiator_role: AgentRole
    target_role: AgentRole
    required_capabilities: Set[str] = field(default_factory=set)
    context_requirements: Dict[str, ContextScope] = field(default_factory=dict)
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)
    
    def is_compatible(self, initiator: 'Agent', target: 'Agent') -> bool:
        """Check if agents are compatible for this interaction"""
        if initiator.role != self.initiator_role or target.role != self.target_role:
            return False
        
        # Check if target has required capabilities
        target_capabilities = {cap.name for cap in target.capabilities}
        return self.required_capabilities.issubset(target_capabilities)


class Agent(ABC):
    """Abstract base class for specialized agents"""
    
    def __init__(self, agent_id: str, role: AgentRole, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.status = "idle"
        self.current_tasks: Dict[str, Any] = {}
        self.context_store: Dict[str, ContextData] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(f"Agent.{role.value}.{agent_id}")
        
        # Weak references to avoid circular dependencies
        self.peers: Dict[str, weakref.ReferenceType] = {}
        self.orchestrator: Optional[weakref.ReferenceType] = None
        
    @abstractmethod
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task according to agent's specialization"""
        pass
    
    @abstractmethod
    async def handle_interaction(self, interaction_type: InteractionType, 
                               from_agent: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming interactions from other agents"""
        pass
    
    def add_context(self, context: ContextData) -> bool:
        """Add context data to agent's store"""
        if context.is_accessible_by(self.agent_id):
            self.context_store[context.id] = context
            self.logger.debug(f"Added context {context.id} with scope {context.scope.value}")
            return True
        return False
    
    def get_context(self, context_id: str) -> Optional[ContextData]:
        """Retrieve context data if accessible"""
        if context_id in self.context_store:
            context = self.context_store[context_id]
            if context.is_expired():
                del self.context_store[context_id]
                return None
            if context.increment_read():
                return context
            else:
                del self.context_store[context_id]
        return None
    
    def find_context_by_scope(self, scope: ContextScope) -> List[ContextData]:
        """Find all contexts with given scope"""
        return [ctx for ctx in self.context_store.values() if ctx.scope == scope]
    
    async def initiate_handoff(self, target_agent_id: str, task_id: str, 
                             context: HandoffContext) -> bool:
        """Initiate handoff to another agent"""
        target_agent = self.peers.get(target_agent_id)
        if not target_agent or not target_agent():
            self.logger.error(f"Target agent {target_agent_id} not found")
            return False
        
        try:
            response = await target_agent().handle_interaction(
                InteractionType.HANDOFF, 
                self.agent_id,
                {"handoff_context": context, "task_id": task_id}
            )
            
            success = response.get("accepted", False)
            if success:
                self.logger.info(f"Successfully handed off task {task_id} to {target_agent_id}")
                # Remove task from current tasks
                if task_id in self.current_tasks:
                    del self.current_tasks[task_id]
            else:
                self.logger.warning(f"Handoff rejected by {target_agent_id}: {response.get('reason', 'Unknown')}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error during handoff to {target_agent_id}: {e}")
            return False
    
    async def request_collaboration(self, peer_agent_id: str, 
                                  collaboration_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Request collaboration from peer agent"""
        peer_agent = self.peers.get(peer_agent_id)
        if not peer_agent or not peer_agent():
            return None
        
        try:
            response = await peer_agent().handle_interaction(
                InteractionType.COLLABORATION,
                self.agent_id,
                collaboration_data
            )
            return response
        except Exception as e:
            self.logger.error(f"Error during collaboration with {peer_agent_id}: {e}")
            return None
    
    def register_peer(self, peer_agent: 'Agent'):
        """Register peer agent for interactions"""
        self.peers[peer_agent.agent_id] = weakref.ref(peer_agent)
        self.logger.debug(f"Registered peer agent: {peer_agent.agent_id}")
    
    def get_capability_names(self) -> Set[str]:
        """Get set of capability names"""
        return {cap.name for cap in self.capabilities}


class ResearchAgent(Agent):
    """Specialized agent for research and information gathering"""
    
    def __init__(self, agent_id: str):
        capabilities = [
            AgentCapability(
                name="information_gathering",
                description="Gather information from various sources",
                input_types={"query", "topic", "requirements"},
                output_types={"research_report", "data_analysis", "findings"},
                quality_metrics={"accuracy": 0.9, "completeness": 0.85}
            ),
            AgentCapability(
                name="data_analysis",
                description="Analyze and synthesize data",
                input_types={"raw_data", "dataset", "information"},
                output_types={"analysis_report", "insights", "patterns"},
                quality_metrics={"accuracy": 0.92, "relevance": 0.88}
            ),
            AgentCapability(
                name="literature_review",
                description="Review and summarize literature",
                input_types={"papers", "documents", "sources"},
                output_types={"literature_summary", "citation_analysis"},
                quality_metrics={"thoroughness": 0.9, "relevance": 0.85}
            )
        ]
        super().__init__(agent_id, AgentRole.RESEARCH, capabilities)
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process research task"""
        task_type = task_data.get("type", "general_research")
        query = task_data.get("query", "")
        scope = task_data.get("scope", "broad")
        
        self.logger.info(f"Processing research task: {task_type}")
        
        # Simulate research process
        await asyncio.sleep(2.0)  # Research takes time
        
        findings = {
            "task_id": task_data.get("task_id"),
            "research_type": task_type,
            "query": query,
            "findings": [
                "Key finding 1: Research results indicate...",
                "Key finding 2: Analysis shows...",
                "Key finding 3: Evidence suggests..."
            ],
            "sources": ["source1.pdf", "source2.html", "database_query_results"],
            "confidence_score": 0.87,
            "completion_time": datetime.now().isoformat()
        }
        
        # Create context for findings
        context = ContextData(
            scope=ContextScope.WORKFLOW_GLOBAL,
            data=findings,
            created_by=self.agent_id,
            metadata={"task_type": task_type, "confidence": 0.87}
        )
        self.add_context(context)
        
        return {
            "status": "completed",
            "results": findings,
            "context_id": context.id,
            "next_suggested_action": "planning"
        }
    
    async def handle_interaction(self, interaction_type: InteractionType,
                               from_agent: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interactions from other agents"""
        if interaction_type == InteractionType.CONSULTATION:
            # Provide research consultation
            consultation_query = data.get("query", "")
            self.logger.info(f"Providing research consultation for: {consultation_query}")
            
            return {
                "consultation_response": f"Research recommendation for: {consultation_query}",
                "confidence": 0.85,
                "recommended_sources": ["academic_db", "industry_reports"],
                "estimated_effort": "2-4 hours"
            }
        
        elif interaction_type == InteractionType.HANDOFF:
            # Accept handoff if we can handle the task
            handoff_context = data.get("handoff_context")
            task_id = data.get("task_id")
            
            if self._can_accept_handoff(handoff_context):
                self.current_tasks[task_id] = handoff_context
                return {"accepted": True, "estimated_completion": "4 hours"}
            else:
                return {"accepted": False, "reason": "Task outside research scope"}
        
        return {"status": "interaction_not_supported"}
    
    def _can_accept_handoff(self, handoff_context: HandoffContext) -> bool:
        """Check if we can accept a handoff"""
        # Simple capacity check
        return len(self.current_tasks) < 3


class PlanningAgent(Agent):
    """Specialized agent for planning and strategy development"""
    
    def __init__(self, agent_id: str):
        capabilities = [
            AgentCapability(
                name="strategic_planning",
                description="Develop strategic plans and roadmaps",
                input_types={"requirements", "research_report", "constraints"},
                output_types={"execution_plan", "roadmap", "strategy"},
                quality_metrics={"feasibility": 0.9, "completeness": 0.92}
            ),
            AgentCapability(
                name="task_decomposition",
                description="Break down complex tasks into subtasks",
                input_types={"complex_task", "requirements"},
                output_types={"task_breakdown", "dependency_graph"},
                quality_metrics={"granularity": 0.88, "accuracy": 0.9}
            ),
            AgentCapability(
                name="resource_allocation",
                description="Plan resource allocation and scheduling",
                input_types={"tasks", "resources", "constraints"},
                output_types={"allocation_plan", "schedule"},
                quality_metrics={"efficiency": 0.85, "feasibility": 0.9}
            )
        ]
        super().__init__(agent_id, AgentRole.PLANNING, capabilities)
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process planning task"""
        planning_type = task_data.get("type", "general_planning")
        requirements = task_data.get("requirements", {})
        
        self.logger.info(f"Processing planning task: {planning_type}")
        
        # Simulate planning process
        await asyncio.sleep(1.5)
        
        execution_plan = {
            "task_id": task_data.get("task_id"),
            "planning_type": planning_type,
            "phases": [
                {
                    "phase": "preparation",
                    "duration": "2 hours",
                    "tasks": ["setup_environment", "gather_resources"]
                },
                {
                    "phase": "execution",
                    "duration": "6 hours", 
                    "tasks": ["implement_core", "integrate_components"]
                },
                {
                    "phase": "validation",
                    "duration": "2 hours",
                    "tasks": ["run_tests", "verify_requirements"]
                }
            ],
            "dependencies": ["research_completed", "resources_available"],
            "risks": ["technical_complexity", "time_constraints"],
            "mitigation_strategies": ["parallel_execution", "fallback_plans"]
        }
        
        # Create planning context
        context = ContextData(
            scope=ContextScope.WORKFLOW_GLOBAL,
            data=execution_plan,
            created_by=self.agent_id,
            metadata={"planning_type": planning_type}
        )
        self.add_context(context)
        
        return {
            "status": "completed", 
            "execution_plan": execution_plan,
            "context_id": context.id,
            "next_suggested_action": "execution"
        }
    
    async def handle_interaction(self, interaction_type: InteractionType,
                               from_agent: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interactions from other agents"""
        if interaction_type == InteractionType.CONSULTATION:
            planning_query = data.get("query", "")
            self.logger.info(f"Providing planning consultation for: {planning_query}")
            
            return {
                "planning_recommendations": [
                    "Break task into smaller phases",
                    "Identify critical path dependencies", 
                    "Plan for risk mitigation"
                ],
                "estimated_timeline": "8-12 hours",
                "resource_requirements": ["execution_agent", "validation_agent"]
            }
        
        elif interaction_type == InteractionType.HANDOFF:
            handoff_context = data.get("handoff_context")
            task_id = data.get("task_id")
            
            if self._can_accept_handoff(handoff_context):
                self.current_tasks[task_id] = handoff_context
                return {"accepted": True, "estimated_completion": "3 hours"}
            else:
                return {"accepted": False, "reason": "Planning queue full"}
        
        return {"status": "interaction_not_supported"}
    
    def _can_accept_handoff(self, handoff_context: HandoffContext) -> bool:
        """Check if we can accept a handoff"""
        return len(self.current_tasks) < 2


class ExecutionAgent(Agent):
    """Specialized agent for task execution and implementation"""
    
    def __init__(self, agent_id: str):
        capabilities = [
            AgentCapability(
                name="code_implementation",
                description="Implement code based on specifications",
                input_types={"specification", "execution_plan", "requirements"},
                output_types={"implementation", "code", "system"},
                quality_metrics={"functionality": 0.95, "performance": 0.88}
            ),
            AgentCapability(
                name="system_deployment",
                description="Deploy and configure systems",
                input_types={"implementation", "configuration", "environment"},
                output_types={"deployed_system", "configuration_report"},
                quality_metrics={"reliability": 0.92, "performance": 0.9}
            ),
            AgentCapability(
                name="integration",
                description="Integrate components and systems",
                input_types={"components", "interfaces", "protocols"},
                output_types={"integrated_system", "integration_report"},
                quality_metrics={"compatibility": 0.9, "stability": 0.88}
            )
        ]
        super().__init__(agent_id, AgentRole.EXECUTION, capabilities)
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process execution task"""
        execution_type = task_data.get("type", "general_execution")
        plan = task_data.get("execution_plan", {})
        
        self.logger.info(f"Processing execution task: {execution_type}")
        
        # Simulate execution process
        await asyncio.sleep(3.0)  # Execution takes longer
        
        implementation_result = {
            "task_id": task_data.get("task_id"),
            "execution_type": execution_type,
            "implemented_components": [
                "core_module.py",
                "integration_layer.py", 
                "configuration.json"
            ],
            "deployment_status": "successful",
            "performance_metrics": {
                "response_time": "120ms",
                "throughput": "1000 req/s",
                "error_rate": "0.01%"
            },
            "test_results": {
                "unit_tests": "98% passed",
                "integration_tests": "95% passed",
                "performance_tests": "all passed"
            }
        }
        
        # Create execution context
        context = ContextData(
            scope=ContextScope.WORKFLOW_GLOBAL,
            data=implementation_result,
            created_by=self.agent_id,
            metadata={"execution_type": execution_type}
        )
        self.add_context(context)
        
        return {
            "status": "completed",
            "implementation_result": implementation_result,
            "context_id": context.id,
            "next_suggested_action": "validation"
        }
    
    async def handle_interaction(self, interaction_type: InteractionType,
                               from_agent: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interactions from other agents"""
        if interaction_type == InteractionType.CONSULTATION:
            implementation_query = data.get("query", "")
            self.logger.info(f"Providing implementation consultation for: {implementation_query}")
            
            return {
                "implementation_approach": "Use modular architecture with clean interfaces",
                "technology_recommendations": ["Python", "Docker", "PostgreSQL"],
                "estimated_effort": "6-8 hours",
                "risk_factors": ["complexity", "dependencies"]
            }
        
        elif interaction_type == InteractionType.HANDOFF:
            handoff_context = data.get("handoff_context")
            task_id = data.get("task_id")
            
            if self._can_accept_handoff(handoff_context):
                self.current_tasks[task_id] = handoff_context
                return {"accepted": True, "estimated_completion": "6 hours"}
            else:
                return {"accepted": False, "reason": "Execution capacity exceeded"}
        
        return {"status": "interaction_not_supported"}
    
    def _can_accept_handoff(self, handoff_context: HandoffContext) -> bool:
        """Check if we can accept a handoff"""
        return len(self.current_tasks) < 2


class ValidationAgent(Agent):
    """Specialized agent for validation and quality assurance"""
    
    def __init__(self, agent_id: str):
        capabilities = [
            AgentCapability(
                name="quality_assurance",
                description="Perform quality assurance and testing",
                input_types={"implementation", "test_plan", "requirements"},
                output_types={"qa_report", "test_results", "validation_report"},
                quality_metrics={"thoroughness": 0.95, "accuracy": 0.98}
            ),
            AgentCapability(
                name="compliance_check",
                description="Check compliance with standards and regulations",
                input_types={"system", "standards", "regulations"},
                output_types={"compliance_report", "audit_results"},
                quality_metrics={"completeness": 0.95, "accuracy": 0.99}
            ),
            AgentCapability(
                name="security_validation",
                description="Validate security aspects of implementation",
                input_types={"implementation", "security_requirements"},
                output_types={"security_report", "vulnerability_assessment"},
                quality_metrics={"thoroughness": 0.98, "accuracy": 0.97}
            )
        ]
        super().__init__(agent_id, AgentRole.VALIDATION, capabilities)
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process validation task"""
        validation_type = task_data.get("type", "general_validation")
        implementation = task_data.get("implementation", {})
        
        self.logger.info(f"Processing validation task: {validation_type}")
        
        # Simulate validation process
        await asyncio.sleep(2.5)
        
        validation_result = {
            "task_id": task_data.get("task_id"),
            "validation_type": validation_type,
            "test_coverage": "96%",
            "test_results": {
                "total_tests": 250,
                "passed": 242,
                "failed": 5,
                "skipped": 3
            },
            "quality_metrics": {
                "code_quality": "A",
                "maintainability": "8.5/10",
                "security_score": "9.2/10"
            },
            "issues_found": [
                {"severity": "low", "description": "Minor code style issues"},
                {"severity": "medium", "description": "Performance optimization opportunities"}
            ],
            "recommendations": [
                "Address performance optimizations",
                "Improve error handling in edge cases",
                "Add more comprehensive logging"
            ],
            "overall_status": "approved_with_minor_issues"
        }
        
        # Create validation context
        context = ContextData(
            scope=ContextScope.WORKFLOW_GLOBAL,
            data=validation_result,
            created_by=self.agent_id,
            metadata={"validation_type": validation_type, "approval_status": "approved"}
        )
        self.add_context(context)
        
        return {
            "status": "completed",
            "validation_result": validation_result,
            "context_id": context.id,
            "approval_status": "approved_with_minor_issues"
        }
    
    async def handle_interaction(self, interaction_type: InteractionType,
                               from_agent: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interactions from other agents"""
        if interaction_type == InteractionType.CONSULTATION:
            validation_query = data.get("query", "")
            self.logger.info(f"Providing validation consultation for: {validation_query}")
            
            return {
                "validation_approach": "Comprehensive testing with automated and manual verification",
                "test_categories": ["unit", "integration", "performance", "security"],
                "estimated_effort": "3-4 hours",
                "quality_gates": ["95% test coverage", "zero critical issues"]
            }
        
        elif interaction_type == InteractionType.HANDOFF:
            handoff_context = data.get("handoff_context")
            task_id = data.get("task_id")
            
            if self._can_accept_handoff(handoff_context):
                self.current_tasks[task_id] = handoff_context
                return {"accepted": True, "estimated_completion": "4 hours"}
            else:
                return {"accepted": False, "reason": "Validation queue full"}
        
        return {"status": "interaction_not_supported"}
    
    def _can_accept_handoff(self, handoff_context: HandoffContext) -> bool:
        """Check if we can accept a handoff"""
        return len(self.current_tasks) < 3


class OrchestratorAgent(Agent):
    """Orchestrator agent for coordinating multi-agent workflows"""
    
    def __init__(self, agent_id: str):
        capabilities = [
            AgentCapability(
                name="workflow_orchestration",
                description="Orchestrate multi-agent workflows",
                input_types={"workflow_definition", "task_requirements"},
                output_types={"workflow_plan", "coordination_strategy"},
                quality_metrics={"efficiency": 0.9, "coordination": 0.95}
            ),
            AgentCapability(
                name="agent_coordination",
                description="Coordinate agent interactions and handoffs",
                input_types={"agent_status", "task_progress"},
                output_types={"coordination_decisions", "handoff_instructions"},
                quality_metrics={"optimization": 0.88, "reliability": 0.92}
            )
        ]
        super().__init__(agent_id, AgentRole.ORCHESTRATOR, capabilities)
        self.workflow_protocols: Dict[str, InteractionProtocol] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
    
    def register_protocol(self, protocol_name: str, protocol: InteractionProtocol):
        """Register interaction protocol"""
        self.workflow_protocols[protocol_name] = protocol
        self.logger.info(f"Registered protocol: {protocol_name}")
    
    async def orchestrate_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Orchestrate a multi-agent workflow"""
        workflow_id = str(uuid.uuid4())
        
        workflow_plan = {
            "workflow_id": workflow_id,
            "stages": workflow_definition.get("stages", []),
            "current_stage": 0,
            "agent_assignments": {},
            "context_sharing_plan": {},
            "status": "active"
        }
        
        self.active_workflows[workflow_id] = workflow_plan
        self.logger.info(f"Started workflow orchestration: {workflow_id}")
        
        # Execute workflow stages
        await self._execute_workflow(workflow_id)
        
        return workflow_id
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute workflow stages sequentially"""
        workflow = self.active_workflows[workflow_id]
        
        for stage_index, stage in enumerate(workflow["stages"]):
            workflow["current_stage"] = stage_index
            self.logger.info(f"Executing stage {stage_index}: {stage.get('name', 'Unnamed')}")
            
            # Find appropriate agent for stage
            required_role = AgentRole(stage.get("role", "execution"))
            target_agent = self._find_agent_by_role(required_role)
            
            if target_agent:
                # Prepare context for handoff
                stage_context = HandoffContext(
                    from_agent=self.agent_id,
                    to_agent=target_agent.agent_id,
                    task_id=stage.get("task_id", f"{workflow_id}_stage_{stage_index}"),
                    handoff_reason=f"Workflow stage execution: {stage.get('name')}"
                )
                
                # Execute handoff
                success = await self.initiate_handoff(
                    target_agent.agent_id,
                    stage_context.task_id,
                    stage_context
                )
                
                if success:
                    # Wait for completion (simplified)
                    await asyncio.sleep(1.0)
                else:
                    self.logger.error(f"Failed to execute stage {stage_index}")
                    break
        
        workflow["status"] = "completed"
        self.logger.info(f"Workflow {workflow_id} completed")
    
    def _find_agent_by_role(self, role: AgentRole) -> Optional[Agent]:
        """Find agent by role"""
        for peer_ref in self.peers.values():
            peer = peer_ref()
            if peer and peer.role == role:
                return peer
        return None
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process orchestration task"""
        workflow_id = await self.orchestrate_workflow(task_data)
        
        return {
            "status": "orchestrating",
            "workflow_id": workflow_id,
            "estimated_completion": "12 hours"
        }
    
    async def handle_interaction(self, interaction_type: InteractionType,
                               from_agent: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interactions from other agents"""
        if interaction_type == InteractionType.ESCALATION:
            self.logger.warning(f"Escalation received from {from_agent}: {data}")
            # Handle escalation
            return {"escalation_handled": True, "resolution": "Reassigning task"}
        
        return {"status": "interaction_handled"}


class ProtocolRegistry:
    """Registry for interaction protocols"""
    
    def __init__(self):
        self.protocols: Dict[str, InteractionProtocol] = {}
        self.logger = logging.getLogger("ProtocolRegistry")
        self._register_default_protocols()
    
    def _register_default_protocols(self):
        """Register default interaction protocols"""
        
        # Research to Planning handoff
        self.register_protocol("research_to_planning", InteractionProtocol(
            interaction_type=InteractionType.HANDOFF,
            initiator_role=AgentRole.RESEARCH,
            target_role=AgentRole.PLANNING,
            required_capabilities={"strategic_planning"},
            context_requirements={"research_findings": ContextScope.WORKFLOW_GLOBAL},
            timeout=timedelta(hours=2)
        ))
        
        # Planning to Execution handoff
        self.register_protocol("planning_to_execution", InteractionProtocol(
            interaction_type=InteractionType.HANDOFF,
            initiator_role=AgentRole.PLANNING,
            target_role=AgentRole.EXECUTION,
            required_capabilities={"code_implementation"},
            context_requirements={"execution_plan": ContextScope.WORKFLOW_GLOBAL},
            timeout=timedelta(hours=8)
        ))
        
        # Execution to Validation handoff
        self.register_protocol("execution_to_validation", InteractionProtocol(
            interaction_type=InteractionType.HANDOFF,
            initiator_role=AgentRole.EXECUTION,
            target_role=AgentRole.VALIDATION,
            required_capabilities={"quality_assurance"},
            context_requirements={"implementation": ContextScope.WORKFLOW_GLOBAL},
            timeout=timedelta(hours=4)
        ))
        
        # Cross-role consultation protocols
        self.register_protocol("any_to_research_consultation", InteractionProtocol(
            interaction_type=InteractionType.CONSULTATION,
            initiator_role=AgentRole.EXECUTION,  # Any role can consult research
            target_role=AgentRole.RESEARCH,
            required_capabilities={"information_gathering"},
            timeout=timedelta(minutes=30)
        ))
    
    def register_protocol(self, name: str, protocol: InteractionProtocol):
        """Register an interaction protocol"""
        self.protocols[name] = protocol
        self.logger.info(f"Registered protocol: {name}")
    
    def get_protocol(self, name: str) -> Optional[InteractionProtocol]:
        """Get protocol by name"""
        return self.protocols.get(name)
    
    def find_protocols(self, initiator_role: AgentRole, 
                      target_role: AgentRole, 
                      interaction_type: InteractionType) -> List[InteractionProtocol]:
        """Find protocols matching criteria"""
        matching_protocols = []
        for protocol in self.protocols.values():
            if (protocol.initiator_role == initiator_role and 
                protocol.target_role == target_role and
                protocol.interaction_type == interaction_type):
                matching_protocols.append(protocol)
        return matching_protocols


class AgentNetwork:
    """Network of agents with orchestrated interactions"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.protocol_registry = ProtocolRegistry()
        self.orchestrator: Optional[OrchestratorAgent] = None
        self.logger = logging.getLogger("AgentNetwork")
    
    def add_agent(self, agent: Agent):
        """Add agent to network"""
        self.agents[agent.agent_id] = agent
        
        # Register agent with all other agents as peers
        for other_agent in self.agents.values():
            if other_agent.agent_id != agent.agent_id:
                agent.register_peer(other_agent)
                other_agent.register_peer(agent)
        
        self.logger.info(f"Added agent to network: {agent.agent_id} ({agent.role.value})")
    
    def set_orchestrator(self, orchestrator: OrchestratorAgent):
        """Set network orchestrator"""
        self.orchestrator = orchestrator
        self.add_agent(orchestrator)
        
        # Register orchestrator protocols
        for name, protocol in self.protocol_registry.protocols.items():
            orchestrator.register_protocol(name, protocol)
        
        self.logger.info(f"Set network orchestrator: {orchestrator.agent_id}")
    
    def get_agents_by_role(self, role: AgentRole) -> List[Agent]:
        """Get all agents with specific role"""
        return [agent for agent in self.agents.values() if agent.role == role]
    
    async def execute_workflow(self, workflow_definition: Dict[str, Any]) -> Optional[str]:
        """Execute workflow through orchestrator"""
        if not self.orchestrator:
            self.logger.error("No orchestrator set for workflow execution")
            return None
        
        return await self.orchestrator.orchestrate_workflow(workflow_definition)
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get network status overview"""
        agents_by_role = {}
        for role in AgentRole:
            agents_by_role[role.value] = len(self.get_agents_by_role(role))
        
        total_tasks = sum(len(agent.current_tasks) for agent in self.agents.values())
        
        return {
            "total_agents": len(self.agents),
            "agents_by_role": agents_by_role,
            "total_active_tasks": total_tasks,
            "has_orchestrator": self.orchestrator is not None,
            "registered_protocols": len(self.protocol_registry.protocols)
        }


# Export key classes
__all__ = [
    "AgentRole", "InteractionType", "ContextScope", "AgentCapability", 
    "ContextData", "HandoffContext", "InteractionProtocol", "Agent",
    "ResearchAgent", "PlanningAgent", "ExecutionAgent", "ValidationAgent",
    "OrchestratorAgent", "ProtocolRegistry", "AgentNetwork"
]


if __name__ == "__main__":
    # Demo usage
    async def demo():
        logging.basicConfig(level=logging.INFO)
        
        # Create agent network
        network = AgentNetwork()
        
        # Create specialized agents
        research_agent = ResearchAgent("research-001")
        planning_agent = PlanningAgent("planning-001")  
        execution_agent = ExecutionAgent("execution-001")
        validation_agent = ValidationAgent("validation-001")
        orchestrator = OrchestratorAgent("orchestrator-001")
        
        # Add agents to network
        network.add_agent(research_agent)
        network.add_agent(planning_agent)
        network.add_agent(execution_agent)
        network.add_agent(validation_agent)
        network.set_orchestrator(orchestrator)
        
        print(f"🤖 Agent Network Initialized")
        print(f"Network Status: {network.get_network_status()}")
        
        # Test individual agent capabilities
        print(f"\n🔬 Testing Research Agent")
        research_result = await research_agent.process_task({
            "task_id": "research-001",
            "type": "market_analysis",
            "query": "AI agent frameworks",
            "scope": "comprehensive"
        })
        print(f"Research completed: {research_result['status']}")
        
        # Test agent consultation
        print(f"\n💬 Testing Agent Consultation")
        consultation_result = await planning_agent.request_collaboration(
            research_agent.agent_id,
            {"query": "What are the key findings for planning?"}
        )
        print(f"Consultation result: {consultation_result}")
        
        # Test workflow execution
        print(f"\n🔄 Testing Workflow Execution")
        workflow_definition = {
            "name": "Complete Development Workflow",
            "stages": [
                {"name": "Research Phase", "role": "research", "task_id": "wf-research"},
                {"name": "Planning Phase", "role": "planning", "task_id": "wf-planning"},
                {"name": "Execution Phase", "role": "execution", "task_id": "wf-execution"},
                {"name": "Validation Phase", "role": "validation", "task_id": "wf-validation"}
            ]
        }
        
        workflow_id = await network.execute_workflow(workflow_definition)
        print(f"Workflow executed: {workflow_id}")
        
        # Display final network status
        print(f"\n📊 Final Network Status: {network.get_network_status()}")
    
    # Run demo
    asyncio.run(demo())