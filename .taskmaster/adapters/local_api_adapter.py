#!/usr/bin/env python3
"""
Local API Adapter for Task Master AI
Provides drop-in replacements for external API calls (Perplexity, Claude, etc.)
Uses adapter pattern to maintain compatibility while switching to local LLMs
"""

import json
import os
import asyncio
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Import local research components
from ..research.local_llm_research_engine import LocalLLMResearchEngine
from ..research.local_research_workflow import LocalResearchWorkflow

class LocalAPIAdapter:
    """
    Adapter class that provides drop-in replacements for external API calls
    Maintains interface compatibility while using local LLM infrastructure
    """
    
    def __init__(self, 
                 model_endpoint: str = "http://localhost:11434",
                 enable_logging: bool = True):
        """
        Initialize the local API adapter
        
        Args:
            model_endpoint: Local LLM inference endpoint
            enable_logging: Whether to log API calls and responses
        """
        self.model_endpoint = model_endpoint
        self.enable_logging = enable_logging
        
        # Initialize local components
        self.research_engine = LocalLLMResearchEngine(model_endpoint=model_endpoint)
        self.research_workflow = LocalResearchWorkflow(model_endpoint=model_endpoint)
        
        # Setup logging directory
        if enable_logging:
            self.log_dir = Path(".taskmaster/logs/api_adapter")
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # API call tracking
        self.api_call_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_response_time": 0,
            "calls_by_type": {}
        }
    
    def _log_api_call(self, api_type: str, request_data: Dict[str, Any], response_data: Dict[str, Any], success: bool):
        """Log API call details for debugging and monitoring"""
        if not self.enable_logging:
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "api_type": api_type,
            "success": success,
            "request": request_data,
            "response": response_data if success else {"error": response_data},
            "local_source": True
        }
        
        log_file = self.log_dir / f"api_calls_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Update stats
        self.api_call_stats["total_calls"] += 1
        if success:
            self.api_call_stats["successful_calls"] += 1
        else:
            self.api_call_stats["failed_calls"] += 1
        
        if api_type not in self.api_call_stats["calls_by_type"]:
            self.api_call_stats["calls_by_type"][api_type] = 0
        self.api_call_stats["calls_by_type"][api_type] += 1
    
    # Perplexity API Replacement Methods
    
    async def perplexity_chat_completion(self, 
                                       messages: List[Dict[str, str]], 
                                       model: str = "llama-3.1-sonar-large-128k-online",
                                       **kwargs) -> Dict[str, Any]:
        """
        Drop-in replacement for Perplexity API chat completion
        
        Args:
            messages: List of message objects with 'role' and 'content'
            model: Model name (ignored, uses local model selection)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Response in Perplexity API format
        """
        try:
            # Extract the user query from messages
            user_message = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if not user_message:
                raise ValueError("No user message found in request")
            
            # Use local research engine
            research_result = await self.research_engine.conduct_research(
                query=user_message,
                research_type="comprehensive"
            )
            
            # Format response in Perplexity-compatible format
            response = {
                "id": f"local-{int(datetime.now().timestamp())}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": f"local-{self.research_engine._select_optimal_model('research')}",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": research_result["response"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(research_result["response"].split()),
                    "total_tokens": len(user_message.split()) + len(research_result["response"].split())
                }
            }
            
            self._log_api_call("perplexity_chat", {"messages": messages}, response, True)
            return response
            
        except Exception as e:
            error_response = {"error": {"message": str(e), "type": "local_api_error"}}
            self._log_api_call("perplexity_chat", {"messages": messages}, str(e), False)
            raise Exception(f"Local Perplexity replacement failed: {e}")
    
    async def perplexity_research_query(self, query: str, context: str = "") -> str:
        """
        Simplified Perplexity research query replacement
        
        Args:
            query: Research query
            context: Additional context
            
        Returns:
            Research response text
        """
        try:
            research_result = await self.research_engine.conduct_research(
                query=query,
                research_type="quick",
                context=context
            )
            
            self._log_api_call("perplexity_research", {"query": query, "context": context}, 
                             research_result["response"], True)
            return research_result["response"]
            
        except Exception as e:
            self._log_api_call("perplexity_research", {"query": query, "context": context}, str(e), False)
            # Return fallback response
            return f"Local research unavailable. Manual research needed for: {query}"
    
    # Claude API Replacement Methods (if needed in future)
    
    async def claude_completion(self, 
                              prompt: str, 
                              model: str = "claude-3-sonnet",
                              max_tokens: int = 1000,
                              temperature: float = 0.7) -> Dict[str, Any]:
        """
        Drop-in replacement for Claude API completion
        
        Args:
            prompt: Input prompt
            model: Model name (ignored)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Response in Claude API format
        """
        try:
            # Use local LLM for code/technical tasks
            response_text = await self.research_engine._generate_local_inference(
                prompt=prompt,
                model=self.research_engine._select_optimal_model("code"),
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            response = {
                "id": f"local-claude-{int(datetime.now().timestamp())}",
                "type": "completion",
                "content": [{"type": "text", "text": response_text}],
                "model": f"local-{self.research_engine.code_model}",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": len(prompt.split()),
                    "output_tokens": len(response_text.split())
                }
            }
            
            self._log_api_call("claude_completion", {"prompt": prompt[:100] + "..."}, response, True)
            return response
            
        except Exception as e:
            self._log_api_call("claude_completion", {"prompt": prompt[:100] + "..."}, str(e), False)
            raise Exception(f"Local Claude replacement failed: {e}")
    
    # Task-Master Integration Methods
    
    async def task_master_research_integration(self, 
                                             prompt: str, 
                                             use_research: bool = True,
                                             task_context: str = "") -> Dict[str, Any]:
        """
        Replace task-master --research calls with local research + task creation
        
        Args:
            prompt: Task prompt/description
            use_research: Whether to use research (always True for compatibility)
            task_context: Additional context
            
        Returns:
            Task creation result with research integration
        """
        try:
            # Conduct local research first
            research_result = await self.research_workflow.execute_research_workflow(
                research_query=prompt,
                task_context=task_context,
                workflow_type="guided"
            )
            
            # Create task using task-master CLI (without --research flag)
            task_creation_results = []
            for todo in research_result.get("todo_steps", [])[:3]:  # Limit to 3 tasks
                try:
                    cmd = ["task-master", "add-task", "--prompt", f"{todo}\n\nContext: {task_context}"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    task_creation_results.append({
                        "todo": todo,
                        "success": result.returncode == 0,
                        "output": result.stdout if result.returncode == 0 else result.stderr
                    })
                except Exception as e:
                    task_creation_results.append({
                        "todo": todo,
                        "success": False,
                        "output": str(e)
                    })
            
            final_result = {
                "research_conducted": True,
                "research_result": research_result,
                "task_creation_results": task_creation_results,
                "local_api_used": True,
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_api_call("task_master_research", {"prompt": prompt}, final_result, True)
            return final_result
            
        except Exception as e:
            error_result = {"error": str(e), "local_api_used": True}
            self._log_api_call("task_master_research", {"prompt": prompt}, str(e), False)
            return error_result
    
    # Workflow Methods
    
    async def autonomous_stuck_handler_replacement(self, 
                                                 problem_description: str, 
                                                 task_context: str = "") -> Dict[str, Any]:
        """
        Drop-in replacement for autonomous stuck handler workflow
        
        Args:
            problem_description: Description of the problem/stuck situation
            task_context: Current task context
            
        Returns:
            Analysis result with actionable steps
        """
        try:
            # Use local research workflow
            result = await self.research_workflow.execute_research_workflow(
                research_query=f"PROBLEM: {problem_description}",
                task_context=task_context,
                workflow_type="autonomous"
            )
            
            # Format for compatibility with original workflow
            compatible_result = {
                "problem": problem_description,
                "context": task_context,
                "research_findings": result.get("summary", ""),
                "todo_steps": result.get("todo_steps", []),
                "success_rate_target": 0.7,
                "execution_strategy": "incremental_implementation",
                "local_api_used": True,
                "timestamp": datetime.now().isoformat(),
                "full_research_result": result
            }
            
            self._log_api_call("stuck_handler", {"problem": problem_description}, compatible_result, True)
            return compatible_result
            
        except Exception as e:
            error_result = {
                "problem": problem_description,
                "context": task_context,
                "error": str(e),
                "todo_steps": [
                    f"Manually investigate: {problem_description}",
                    f"Review available documentation",
                    f"Seek alternative solutions"
                ],
                "local_api_used": True,
                "timestamp": datetime.now().isoformat()
            }
            self._log_api_call("stuck_handler", {"problem": problem_description}, str(e), False)
            return error_result
    
    # Utility Methods
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            **self.api_call_stats,
            "local_api_adapter": True,
            "model_endpoint": self.model_endpoint,
            "timestamp": datetime.now().isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of local API components"""
        health_status = {
            "overall_health": "unknown",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check model availability
            available_models = asyncio.run(self.research_engine._check_model_availability())
            health_status["components"]["models"] = {
                "status": "healthy" if any(available_models.values()) else "unhealthy",
                "available_models": available_models
            }
            
            # Check research engine
            health_status["components"]["research_engine"] = {
                "status": "healthy",
                "knowledge_base_exists": self.research_engine.knowledge_base_path.exists()
            }
            
            # Check workflow system
            health_status["components"]["workflow_system"] = {
                "status": "healthy",
                "active_sessions": len(self.research_workflow.workflow_state["active_research_sessions"])
            }
            
            # Overall health
            component_health = [comp["status"] for comp in health_status["components"].values()]
            health_status["overall_health"] = "healthy" if all(h == "healthy" for h in component_health) else "unhealthy"
            
        except Exception as e:
            health_status["overall_health"] = "error"
            health_status["error"] = str(e)
        
        return health_status

# Global adapter instance for easy import
_local_adapter = None

def get_local_adapter(model_endpoint: str = "http://localhost:11434") -> LocalAPIAdapter:
    """Get or create global local API adapter instance"""
    global _local_adapter
    if _local_adapter is None:
        _local_adapter = LocalAPIAdapter(model_endpoint=model_endpoint)
    return _local_adapter

# Convenience functions for backwards compatibility

async def replace_perplexity_call(query: str, context: str = "") -> str:
    """Simple function to replace Perplexity API calls"""
    adapter = get_local_adapter()
    return await adapter.perplexity_research_query(query, context)

async def replace_task_master_research(prompt: str, context: str = "") -> Dict[str, Any]:
    """Simple function to replace task-master --research calls"""
    adapter = get_local_adapter()
    return await adapter.task_master_research_integration(prompt, True, context)

async def replace_autonomous_stuck_handler(problem: str, context: str = "") -> Dict[str, Any]:
    """Simple function to replace autonomous stuck handler"""
    adapter = get_local_adapter()
    return await adapter.autonomous_stuck_handler_replacement(problem, context)

# Test function
if __name__ == "__main__":
    async def test_adapter():
        adapter = LocalAPIAdapter()
        
        print("Testing Local API Adapter...")
        
        # Test health check
        health = adapter.health_check()
        print(f"Health Check: {health}")
        
        # Test research query
        result = await adapter.perplexity_research_query(
            "How to optimize recursive task decomposition?",
            "Working on Task Master AI local migration"
        )
        print(f"Research Result: {result[:200]}...")
        
        # Test stuck handler
        stuck_result = await adapter.autonomous_stuck_handler_replacement(
            "Local LLM integration is complex",
            "Migrating from external APIs to local inference"
        )
        print(f"Stuck Handler Result: {stuck_result['todo_steps']}")
        
        # Get stats
        stats = adapter.get_api_stats()
        print(f"API Stats: {stats}")
    
    asyncio.run(test_adapter())