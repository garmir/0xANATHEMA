
#!/usr/bin/env python3
"""
Local Research Module - Backwards Compatibility Interface
Provides drop-in replacements for external research functions
"""
def handle_error_gracefully(func_name: str, error: Exception, fallback_result=None):
    """Graceful error handling with logging and fallback"""
    import traceback
    print(f"⚠️ Error in {func_name}: {error}")
    if fallback_result is not None:
        print(f"🔄 Using fallback result")
        return fallback_result
    return {"error": str(error), "status": "failed", "fallback": True}


import asyncio
from pathlib import Path
import sys

# Add taskmaster modules to path
sys.path.append(str(Path(__file__).parent / ".taskmaster"))

try:
    from adapters.local_api_adapter import (
        LocalAPIAdapter,
        replace_perplexity_call,
        replace_autonomous_stuck_handler
    )
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False

class LocalResearchModule:
    """Local research module for backwards compatibility"""
    
    def __init__(self):
        if LOCAL_LLM_AVAILABLE:
            self.adapter = LocalAPIAdapter()
        else:
            self.adapter = None
    
    async def research_query(self, query: str, context: str = "") -> str:
        """Research query using local LLM"""
        if self.adapter:
            return await self.adapter.perplexity_research_query(query, context)
        else:
            return f"Local research unavailable. Manual research needed: {query}"
    
    async def autonomous_stuck_handler(self, problem: str, context: str = "") -> dict:
        """Autonomous stuck handler using local LLM"""
        if self.adapter:
            return await self.adapter.autonomous_stuck_handler_replacement(problem, context)
        else:
            return {
                "problem": problem,
                "todo_steps": [f"Manual investigation needed: {problem}"]
            }

def create_local_perplexity_replacement():
    """Create local Perplexity replacement"""
    return LocalResearchModule()

# Global instance
local_research = LocalResearchModule()

# Backwards compatibility functions
async def research_with_local_llm(query: str, context: str = "") -> str:
    """Backwards compatible research function"""
    return await local_research.research_query(query, context)

async def autonomous_stuck_handler_local(problem: str, context: str = "") -> dict:
    """Backwards compatible stuck handler"""
    return await local_research.autonomous_stuck_handler(problem, context)

def run_sync_research(query: str, context: str = "") -> str:
    """Synchronous wrapper for async research"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(research_with_local_llm(query, context))

def enhanced_fallback_research(query: str, context: str = "") -> str:
    """Enhanced fallback with structured responses"""
    fallback_responses = {
        "optimization": "Consider analyzing current performance metrics, identifying bottlenecks, implementing incremental improvements, and measuring results.",
        "implementation": "Break down the task into smaller components, create a step-by-step plan, implement iteratively, and test each component.",
        "debugging": "Review error logs, isolate the problem area, test potential solutions, and implement the most promising fix.",
        "integration": "Verify compatibility requirements, test in isolated environment, implement gradually, and monitor for issues.",
        "default": "Research the topic thoroughly, create an implementation plan, execute step by step, and validate results."
    }
    
    query_lower = query.lower()
    for category, response in fallback_responses.items():
        if category in query_lower:
            return f"Manual research needed for {category}: {response}"
    
    return f"Manual research needed: {fallback_responses['default']}"
