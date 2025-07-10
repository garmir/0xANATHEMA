#!/usr/bin/env python3
"""
Local Research Workflow for Task Master AI
Refactored version that replaces external API dependencies with local LLM inference
Preserves recursive research loops and meta-improvement analysis capabilities
"""

import json
import os
import subprocess
import sys
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import the local LLM research engine
from .local_llm_research_engine import LocalLLMResearchEngine, LocalResearchInterface

class LocalResearchWorkflow:
    """
    Refactored research workflow that replaces Perplexity API with local LLM inference
    Maintains compatibility with existing task-master CLI integration
    """
    
    def __init__(self, 
                 model_endpoint: str = "http://localhost:11434",
                 enable_task_master_integration: bool = True):
        """
        Initialize local research workflow
        
        Args:
            model_endpoint: Local LLM inference endpoint
            enable_task_master_integration: Whether to integrate with task-master CLI
        """
        self.research_engine = LocalLLMResearchEngine(model_endpoint=model_endpoint)
        self.enable_task_master = enable_task_master_integration
        
        # Workflow state tracking
        self.workflow_state = {
            "active_research_sessions": {},
            "completed_analyses": [],
            "performance_metrics": {
                "total_research_calls": 0,
                "avg_response_time": 0,
                "success_rate": 0
            }
        }
        
        # Initialize workflow directories
        self.workflow_dir = Path(".taskmaster/research/workflows")
        self.results_dir = Path(".taskmaster/research/results")
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def execute_research_workflow(self, 
                                      research_query: str,
                                      task_context: str = "",
                                      workflow_type: str = "autonomous") -> Dict[str, Any]:
        """
        Execute research workflow using local LLM instead of external APIs
        
        Args:
            research_query: The research question or problem to analyze
            task_context: Additional context about the current task
            workflow_type: Type of workflow (autonomous, guided, analysis)
            
        Returns:
            Dictionary containing research results and execution metadata
        """
        start_time = time.time()
        session_id = f"research_{int(start_time)}"
        
        print(f"🔬 Starting local research workflow: {session_id}")
        print(f"📋 Query: {research_query}")
        
        # Track this research session
        self.workflow_state["active_research_sessions"][session_id] = {
            "query": research_query,
            "context": task_context,
            "type": workflow_type,
            "start_time": start_time,
            "status": "running"
        }
        
        try:
            # Execute research based on workflow type
            if workflow_type == "autonomous":
                result = await self._autonomous_research_workflow(research_query, task_context, session_id)
            elif workflow_type == "guided":
                result = await self._guided_research_workflow(research_query, task_context, session_id)
            elif workflow_type == "analysis":
                result = await self._analysis_research_workflow(research_query, task_context, session_id)
            else:
                result = await self._default_research_workflow(research_query, task_context, session_id)
            
            # Update workflow state
            execution_time = time.time() - start_time
            self.workflow_state["active_research_sessions"][session_id]["status"] = "completed"
            self.workflow_state["active_research_sessions"][session_id]["execution_time"] = execution_time
            
            # Move to completed analyses
            self.workflow_state["completed_analyses"].append({
                **self.workflow_state["active_research_sessions"][session_id],
                "result_summary": result.get("summary", ""),
                "todo_count": len(result.get("todo_steps", []))
            })
            
            # Update performance metrics
            self._update_performance_metrics(execution_time, True)
            
            # Save result to file
            result_file = self.results_dir / f"{session_id}_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"✅ Research workflow completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"❌ Research workflow failed: {e}")
            self.workflow_state["active_research_sessions"][session_id]["status"] = "failed"
            self.workflow_state["active_research_sessions"][session_id]["error"] = str(e)
            self._update_performance_metrics(time.time() - start_time, False)
            
            # Return fallback result
            return self._generate_fallback_result(research_query, task_context, str(e))
        
        finally:
            # Clean up active session
            if session_id in self.workflow_state["active_research_sessions"]:
                del self.workflow_state["active_research_sessions"][session_id]
    
    async def _autonomous_research_workflow(self, query: str, context: str, session_id: str) -> Dict[str, Any]:
        """Autonomous research workflow - fully automated analysis and solution generation"""
        print("🤖 Executing autonomous research workflow...")
        
        # Phase 1: Problem analysis
        analysis_result = await self.research_engine.conduct_research(
            query=f"Analyze this problem: {query}",
            research_type="analysis",
            context=context
        )
        
        # Phase 2: Solution research
        solution_result = await self.research_engine.conduct_research(
            query=f"Provide solutions for: {query}",
            research_type="comprehensive",
            context=f"Problem analysis: {analysis_result['response'][:500]}... Context: {context}"
        )
        
        # Phase 3: Implementation planning
        implementation_result = await self.research_engine.conduct_research(
            query=f"Create implementation plan for: {query}",
            research_type="technical",
            context=f"Solution approach: {solution_result['response'][:500]}..."
        )
        
        # Generate actionable todo steps
        todo_steps = []
        for result in [analysis_result, solution_result, implementation_result]:
            steps = self.research_engine.generate_research_todo_steps(result)
            todo_steps.extend(steps)
        
        # Remove duplicates and limit to reasonable number
        unique_todos = list(dict.fromkeys(todo_steps))[:10]
        
        # If task-master integration is enabled, attempt to create tasks
        task_master_results = []
        if self.enable_task_master:
            task_master_results = await self._integrate_with_task_master(unique_todos, context)
        
        return {
            "session_id": session_id,
            "workflow_type": "autonomous",
            "query": query,
            "context": context,
            "phases": {
                "analysis": analysis_result,
                "solution": solution_result,
                "implementation": implementation_result
            },
            "todo_steps": unique_todos,
            "task_master_integration": task_master_results,
            "summary": f"Autonomous research completed with {len(unique_todos)} actionable steps",
            "timestamp": datetime.now().isoformat(),
            "source": "local_research_workflow"
        }
    
    async def _guided_research_workflow(self, query: str, context: str, session_id: str) -> Dict[str, Any]:
        """Guided research workflow - structured step-by-step analysis"""
        print("🎯 Executing guided research workflow...")
        
        # Structured research phases
        phases = [
            ("background", "Provide background research and context"),
            ("current_state", "Analyze the current state and existing solutions"),
            ("gap_analysis", "Identify gaps and improvement opportunities"),
            ("recommendations", "Generate specific recommendations and next steps")
        ]
        
        phase_results = {}
        all_todo_steps = []
        
        for phase_name, phase_description in phases:
            phase_query = f"{phase_description} for: {query}"
            
            result = await self.research_engine.conduct_research(
                query=phase_query,
                research_type="comprehensive",
                context=context
            )
            
            phase_results[phase_name] = result
            
            # Extract todos from each phase
            todos = self.research_engine.generate_research_todo_steps(result)
            all_todo_steps.extend(todos)
        
        # Consolidate todos
        unique_todos = list(dict.fromkeys(all_todo_steps))[:12]
        
        # Task-master integration
        task_master_results = []
        if self.enable_task_master:
            task_master_results = await self._integrate_with_task_master(unique_todos, context)
        
        return {
            "session_id": session_id,
            "workflow_type": "guided",
            "query": query,
            "context": context,
            "phases": phase_results,
            "todo_steps": unique_todos,
            "task_master_integration": task_master_results,
            "summary": f"Guided research completed across {len(phases)} phases with {len(unique_todos)} action items",
            "timestamp": datetime.now().isoformat(),
            "source": "local_research_workflow"
        }
    
    async def _analysis_research_workflow(self, query: str, context: str, session_id: str) -> Dict[str, Any]:
        """Analysis research workflow - deep analytical assessment"""
        print("🔍 Executing analysis research workflow...")
        
        # Use the autonomous stuck handler for analysis-focused research
        analysis_result = await self.research_engine.autonomous_stuck_handler(query, context)
        
        # Additional deep analysis
        technical_analysis = await self.research_engine.conduct_research(
            query=f"Perform technical analysis of: {query}",
            research_type="technical",
            context=context
        )
        
        # Combine results
        combined_todos = analysis_result["todo_steps"] + \
                        self.research_engine.generate_research_todo_steps(technical_analysis)
        
        unique_todos = list(dict.fromkeys(combined_todos))[:8]
        
        # Task-master integration
        task_master_results = []
        if self.enable_task_master:
            task_master_results = await self._integrate_with_task_master(unique_todos, context)
        
        return {
            "session_id": session_id,
            "workflow_type": "analysis",
            "query": query,
            "context": context,
            "analysis_result": analysis_result,
            "technical_analysis": technical_analysis,
            "todo_steps": unique_todos,
            "task_master_integration": task_master_results,
            "summary": f"Analysis research completed with comprehensive technical assessment",
            "timestamp": datetime.now().isoformat(),
            "source": "local_research_workflow"
        }
    
    async def _default_research_workflow(self, query: str, context: str, session_id: str) -> Dict[str, Any]:
        """Default research workflow - basic research and solution generation"""
        print("📚 Executing default research workflow...")
        
        # Single comprehensive research call
        research_result = await self.research_engine.conduct_research(
            query=query,
            research_type="comprehensive",
            context=context
        )
        
        # Generate todos
        todo_steps = self.research_engine.generate_research_todo_steps(research_result)
        
        # Task-master integration
        task_master_results = []
        if self.enable_task_master:
            task_master_results = await self._integrate_with_task_master(todo_steps, context)
        
        return {
            "session_id": session_id,
            "workflow_type": "default",
            "query": query,
            "context": context,
            "research_result": research_result,
            "todo_steps": todo_steps,
            "task_master_integration": task_master_results,
            "summary": f"Default research completed with {len(todo_steps)} action items",
            "timestamp": datetime.now().isoformat(),
            "source": "local_research_workflow"
        }
    
    async def _integrate_with_task_master(self, todo_steps: List[str], context: str) -> List[Dict[str, Any]]:
        """Integrate research results with task-master CLI (without external API calls)"""
        integration_results = []
        
        for i, todo in enumerate(todo_steps[:5]):  # Limit to 5 tasks to avoid overwhelming
            try:
                # Use task-master add-task without --research flag (no external API)
                cmd = [
                    "task-master", "add-task",
                    "--prompt", f"{todo}\n\nContext: {context}"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                integration_results.append({
                    "todo_step": todo,
                    "task_master_result": "success" if result.returncode == 0 else "failed",
                    "output": result.stdout if result.returncode == 0 else result.stderr,
                    "command": " ".join(cmd)
                })
                
            except subprocess.TimeoutExpired:
                integration_results.append({
                    "todo_step": todo,
                    "task_master_result": "timeout",
                    "output": "Task-master command timed out",
                    "command": " ".join(cmd)
                })
            except Exception as e:
                integration_results.append({
                    "todo_step": todo,
                    "task_master_result": "error",
                    "output": str(e),
                    "command": " ".join(cmd)
                })
        
        return integration_results
    
    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update workflow performance metrics"""
        metrics = self.workflow_state["performance_metrics"]
        
        metrics["total_research_calls"] += 1
        
        # Update average response time
        current_avg = metrics["avg_response_time"]
        total_calls = metrics["total_research_calls"]
        metrics["avg_response_time"] = ((current_avg * (total_calls - 1)) + execution_time) / total_calls
        
        # Update success rate
        current_successes = metrics["success_rate"] * (total_calls - 1)
        if success:
            current_successes += 1
        metrics["success_rate"] = current_successes / total_calls
    
    def _generate_fallback_result(self, query: str, context: str, error: str) -> Dict[str, Any]:
        """Generate fallback result when research fails"""
        fallback_todos = [
            f"Investigate the issue: {query}",
            f"Review documentation related to: {query}",
            f"Seek help or consultation for: {query}",
            f"Break down the problem: {query}",
            f"Test alternative approaches for: {query}"
        ]
        
        return {
            "session_id": f"fallback_{int(time.time())}",
            "workflow_type": "fallback",
            "query": query,
            "context": context,
            "error": error,
            "todo_steps": fallback_todos,
            "task_master_integration": [],
            "summary": f"Fallback research result due to error: {error}",
            "timestamp": datetime.now().isoformat(),
            "source": "local_research_workflow_fallback"
        }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and performance metrics"""
        return {
            "active_sessions": len(self.workflow_state["active_research_sessions"]),
            "completed_analyses": len(self.workflow_state["completed_analyses"]),
            "performance_metrics": self.workflow_state["performance_metrics"],
            "recent_analyses": self.workflow_state["completed_analyses"][-5:],  # Last 5
            "timestamp": datetime.now().isoformat()
        }

# Backwards compatibility functions for existing integrations
async def local_autonomous_stuck_handler(problem_description: str, task_context: str = "") -> Dict[str, Any]:
    """
    Drop-in replacement for the original autonomous stuck handler
    Uses local LLM instead of Perplexity API
    """
    workflow = LocalResearchWorkflow()
    return await workflow.execute_research_workflow(
        research_query=f"PROBLEM: {problem_description}",
        task_context=task_context,
        workflow_type="autonomous"
    )

async def local_research_with_task_master(query: str, context: str = "") -> Dict[str, Any]:
    """
    Drop-in replacement for research + task-master integration
    Uses local LLM and preserves task-master CLI integration
    """
    workflow = LocalResearchWorkflow(enable_task_master_integration=True)
    return await workflow.execute_research_workflow(
        research_query=query,
        task_context=context,
        workflow_type="guided"
    )

# CLI interface for direct execution
def main():
    """CLI interface for testing and direct execution"""
    if len(sys.argv) < 2:
        print("Usage: python local_research_workflow.py <research_query> [context] [workflow_type]")
        print("Workflow types: autonomous, guided, analysis, default")
        sys.exit(1)
    
    query = sys.argv[1]
    context = sys.argv[2] if len(sys.argv) > 2 else ""
    workflow_type = sys.argv[3] if len(sys.argv) > 3 else "autonomous"
    
    async def run_research():
        workflow = LocalResearchWorkflow()
        result = await workflow.execute_research_workflow(query, context, workflow_type)
        print(json.dumps(result, indent=2))
    
    asyncio.run(run_research())

if __name__ == "__main__":
    main()