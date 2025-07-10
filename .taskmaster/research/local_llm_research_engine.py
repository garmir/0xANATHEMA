#!/usr/bin/env python3
"""
Local LLM Research Engine for Task Master AI
Replaces external API dependencies with local inference capabilities
"""

import json
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import asyncio
import httpx

class LocalLLMResearchEngine:
    """
    Local LLM-powered research engine that replaces Perplexity API integration
    with local inference endpoints for autonomous research and planning.
    """
    
    def __init__(self, 
                 model_endpoint: str = "http://localhost:11434",
                 primary_model: str = "llama2",
                 research_model: str = "mistral",
                 code_model: str = "codellama",
                 fallback_model: str = "llama2"):
        """
        Initialize Local LLM Research Engine
        
        Args:
            model_endpoint: Base URL for local LLM inference server (Ollama/LocalAI)
            primary_model: Primary model for general research and analysis
            research_model: Specialized model for deep research tasks
            code_model: Code-focused model for technical analysis
            fallback_model: Fallback model if others unavailable
        """
        self.model_endpoint = model_endpoint.rstrip('/')
        self.primary_model = primary_model
        self.research_model = research_model
        self.code_model = code_model
        self.fallback_model = fallback_model
        
        # Initialize local knowledge base
        self.knowledge_base_path = Path(".taskmaster/research/knowledge_base.json")
        self.research_cache_path = Path(".taskmaster/research/cache/")
        self.research_cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize knowledge base
        self._initialize_knowledge_base()
        
        # Model availability cache
        self._available_models = {}
        self._last_model_check = 0
        
    def _initialize_knowledge_base(self):
        """Initialize local knowledge base with existing research data"""
        if not self.knowledge_base_path.exists():
            initial_knowledge = {
                "domains": {
                    "autonomous_systems": {
                        "patterns": ["agentic design", "self-healing", "adaptive workflows"],
                        "best_practices": ["modularity", "fault tolerance", "continuous monitoring"],
                        "frameworks": ["AI-Ops", "DevOps automation", "evolutionary optimization"]
                    },
                    "memory_optimization": {
                        "algorithms": ["Williams 2025 sqrt-space", "Cook-Mertz tree evaluation", "pebbling strategies"],
                        "complexity_bounds": ["O(√n)", "O(log n · log log n)", "space-time tradeoffs"],
                        "techniques": ["catalytic computing", "memory reuse", "adaptive allocation"]
                    },
                    "task_management": {
                        "methods": ["recursive decomposition", "atomic task breakdown", "hierarchical planning"],
                        "ai_approaches": ["ML-based prioritization", "intelligent scheduling", "evolutionary optimization"],
                        "workflows": ["research-driven development", "autonomous execution", "meta-improvement"]
                    },
                    "research_integration": {
                        "methodologies": ["API-driven research loops", "automated knowledge ingestion", "continuous learning"],
                        "tools": ["semantic analysis", "knowledge graphs", "real-time adaptation"],
                        "validation": ["evidence assessment", "bias detection", "quality metrics"]
                    }
                },
                "research_patterns": {
                    "literature_review": "systematic survey → synthesis → gap analysis → recommendations",
                    "benchmarking": "baseline establishment → comparative analysis → performance validation",
                    "optimization": "problem identification → solution design → implementation → validation"
                },
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "source": "Task Master AI research synthesis"
                }
            }
            
            self.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(initial_knowledge, f, indent=2)
    
    async def _check_model_availability(self) -> Dict[str, bool]:
        """Check which models are available on the local endpoint"""
        current_time = time.time()
        
        # Cache model availability for 5 minutes
        if current_time - self._last_model_check < 300:
            return self._available_models
        
        try:
            async with httpx.AsyncClient() as client:
                # Try Ollama API format first
                response = await client.get(f"{self.model_endpoint}/api/tags", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    self._available_models = {
                        self.primary_model: self.primary_model in models,
                        self.research_model: self.research_model in models,
                        self.code_model: self.code_model in models,
                        self.fallback_model: self.fallback_model in models
                    }
                else:
                    # Try LocalAI format
                    response = await client.get(f"{self.model_endpoint}/models", timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        models = [model["id"] for model in data.get("data", [])]
                        self._available_models = {
                            self.primary_model: self.primary_model in models,
                            self.research_model: self.research_model in models,
                            self.code_model: self.code_model in models,
                            self.fallback_model: self.fallback_model in models
                        }
        except Exception as e:
            print(f"Warning: Could not check model availability: {e}")
            # Assume all models available if endpoint check fails
            self._available_models = {
                self.primary_model: True,
                self.research_model: True,
                self.code_model: True,
                self.fallback_model: True
            }
        
        self._last_model_check = current_time
        return self._available_models
    
    def _select_optimal_model(self, task_type: str = "general") -> str:
        """Select optimal model based on task type and availability"""
        available_models = asyncio.run(self._check_model_availability())
        
        # Task-specific model selection
        if task_type == "research" and available_models.get(self.research_model):
            return self.research_model
        elif task_type == "code" and available_models.get(self.code_model):
            return self.code_model
        elif available_models.get(self.primary_model):
            return self.primary_model
        elif available_models.get(self.fallback_model):
            return self.fallback_model
        else:
            return self.primary_model  # Default fallback
    
    async def _generate_local_inference(self, 
                                      prompt: str, 
                                      model: str = None,
                                      max_tokens: int = 2000,
                                      temperature: float = 0.7) -> str:
        """Generate response using local LLM inference"""
        if model is None:
            model = self._select_optimal_model()
        
        try:
            async with httpx.AsyncClient() as client:
                # Try Ollama API format
                ollama_payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                
                response = await client.post(
                    f"{self.model_endpoint}/api/generate",
                    json=ollama_payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "").strip()
                
                # Try LocalAI/OpenAI-compatible format
                openai_payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                response = await client.post(
                    f"{self.model_endpoint}/v1/chat/completions",
                    json=openai_payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                
        except Exception as e:
            print(f"Local inference error: {e}")
            
        # Fallback to knowledge base if inference fails
        return self._knowledge_base_fallback(prompt)
    
    def _knowledge_base_fallback(self, prompt: str) -> str:
        """Fallback research using local knowledge base"""
        try:
            with open(self.knowledge_base_path, 'r') as f:
                knowledge = json.load(f)
            
            # Simple keyword matching for relevant information
            prompt_lower = prompt.lower()
            relevant_info = []
            
            for domain, content in knowledge.get("domains", {}).items():
                if any(keyword in prompt_lower for keyword in [domain.replace("_", " "), domain]):
                    relevant_info.append(f"**{domain.replace('_', ' ').title()}:**")
                    for category, items in content.items():
                        if isinstance(items, list):
                            relevant_info.append(f"- {category.replace('_', ' ').title()}: {', '.join(items)}")
            
            if relevant_info:
                return "\n".join(relevant_info)
            else:
                return "Based on available knowledge base, consider exploring best practices in autonomous systems, memory optimization, task management, and research integration methodologies."
                
        except Exception as e:
            return f"Research system temporarily unavailable. Consider manual research on the topic. Error: {e}"
    
    async def conduct_research(self, 
                             query: str, 
                             research_type: str = "comprehensive",
                             context: str = "") -> Dict[str, Any]:
        """
        Conduct research using local LLM capabilities
        
        Args:
            query: Research question or topic
            research_type: Type of research (comprehensive, quick, technical, analysis)
            context: Additional context for the research
            
        Returns:
            Dictionary containing research results and metadata
        """
        start_time = time.time()
        
        # Cache check
        cache_key = f"{hash(query + research_type + context)}.json"
        cache_file = self.research_cache_path / cache_key
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_result = json.load(f)
                if time.time() - cached_result.get("timestamp", 0) < 3600:  # 1 hour cache
                    cached_result["source"] = "cache"
                    return cached_result
        
        # Select appropriate model based on research type
        if research_type == "technical" or "code" in query.lower():
            model = self._select_optimal_model("code")
        elif research_type == "comprehensive":
            model = self._select_optimal_model("research")
        else:
            model = self._select_optimal_model("general")
        
        # Craft research prompt based on type
        research_prompts = {
            "comprehensive": f"""
            Conduct a comprehensive research analysis on: {query}
            
            Context: {context}
            
            Please provide:
            1. Overview and current state analysis
            2. Key findings and insights
            3. Best practices and methodologies
            4. Potential challenges and limitations
            5. Actionable recommendations
            6. Future considerations
            
            Focus on practical, implementable solutions and evidence-based recommendations.
            """,
            
            "quick": f"""
            Provide a quick research summary on: {query}
            
            Context: {context}
            
            Include:
            - Key points (3-5 bullet points)
            - Main recommendation
            - Next steps
            
            Keep response concise and actionable.
            """,
            
            "technical": f"""
            Conduct technical analysis on: {query}
            
            Context: {context}
            
            Provide:
            1. Technical approach and methodology
            2. Implementation considerations
            3. Performance implications
            4. Code examples or pseudocode (if relevant)
            5. Testing and validation strategies
            6. Technical recommendations
            """,
            
            "analysis": f"""
            Analyze the following topic: {query}
            
            Context: {context}
            
            Provide analytical assessment including:
            1. Problem decomposition
            2. Root cause analysis
            3. Impact assessment
            4. Solution alternatives
            5. Risk analysis
            6. Implementation roadmap
            """
        }
        
        prompt = research_prompts.get(research_type, research_prompts["comprehensive"])
        
        # Generate research response
        response = await self._generate_local_inference(
            prompt, 
            model=model,
            max_tokens=3000,
            temperature=0.3  # Lower temperature for more focused research
        )
        
        # Structure the research result
        research_result = {
            "query": query,
            "research_type": research_type,
            "context": context,
            "model_used": model,
            "response": response,
            "timestamp": time.time(),
            "execution_time": time.time() - start_time,
            "source": "local_llm",
            "metadata": {
                "endpoint": self.model_endpoint,
                "cache_key": cache_key,
                "created": datetime.now().isoformat()
            }
        }
        
        # Cache the result
        try:
            with open(cache_file, 'w') as f:
                json.dump(research_result, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not cache research result: {e}")
        
        return research_result
    
    def generate_research_todo_steps(self, research_result: Dict[str, Any]) -> List[str]:
        """
        Convert research result into actionable todo steps
        Similar to the original Perplexity workflow parsing
        """
        response = research_result.get("response", "")
        
        # Extract actionable items from research response
        todo_steps = []
        
        # Look for numbered lists, bullet points, or recommendations
        lines = response.split('\n')
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify section headers
            if line.lower().startswith(('recommendations', 'next steps', 'implementation', 'actions', 'todo')):
                current_section = line.lower()
                continue
            
            # Extract actionable items
            if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']):
                # Clean up the item
                cleaned_item = line
                for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']:
                    cleaned_item = cleaned_item.replace(prefix, '').strip()
                
                if len(cleaned_item) > 10:  # Filter out very short items
                    todo_steps.append(cleaned_item)
            
            # Also look for imperative sentences (start with action verbs)
            elif any(line.lower().startswith(verb) for verb in ['implement', 'create', 'develop', 'build', 'design', 'configure', 'setup', 'install', 'test', 'validate', 'update', 'refactor']):
                if len(line) > 15:
                    todo_steps.append(line)
        
        # If no specific steps found, generate generic implementation steps
        if not todo_steps:
            query = research_result.get("query", "")
            todo_steps = [
                f"Research and analyze {query} requirements",
                f"Design implementation approach for {query}",
                f"Implement core functionality for {query}",
                f"Test and validate {query} implementation",
                f"Document and finalize {query} solution"
            ]
        
        # Limit to reasonable number of steps
        return todo_steps[:8]
    
    async def autonomous_stuck_handler(self, 
                                     problem_description: str, 
                                     task_context: str = "") -> Dict[str, Any]:
        """
        Local LLM version of the autonomous stuck handler
        Replaces the original Perplexity-based research workflow
        """
        print(f"🔍 Local research analysis initiated for: {problem_description}")
        
        # Conduct comprehensive research using local LLM
        research_result = await self.conduct_research(
            query=f"PROBLEM: {problem_description}",
            research_type="analysis",
            context=f"Task Context: {task_context}"
        )
        
        # Generate actionable todo steps
        todo_steps = self.generate_research_todo_steps(research_result)
        
        # Structure the result similar to original workflow
        result = {
            "problem": problem_description,
            "context": task_context,
            "research_findings": research_result["response"],
            "todo_steps": todo_steps,
            "success_rate_target": 0.7,
            "execution_strategy": "incremental_implementation",
            "timestamp": datetime.now().isoformat(),
            "source": "local_llm_research_engine"
        }
        
        print(f"✅ Local research completed. Generated {len(todo_steps)} actionable steps.")
        return result
    
    def update_knowledge_base(self, domain: str, new_knowledge: Dict[str, Any]):
        """Update local knowledge base with new information"""
        try:
            with open(self.knowledge_base_path, 'r') as f:
                knowledge = json.load(f)
            
            if domain not in knowledge.get("domains", {}):
                knowledge["domains"][domain] = {}
            
            # Merge new knowledge
            for key, value in new_knowledge.items():
                if key in knowledge["domains"][domain]:
                    if isinstance(value, list) and isinstance(knowledge["domains"][domain][key], list):
                        # Merge lists and remove duplicates
                        combined = knowledge["domains"][domain][key] + value
                        knowledge["domains"][domain][key] = list(set(combined))
                    else:
                        knowledge["domains"][domain][key] = value
                else:
                    knowledge["domains"][domain][key] = value
            
            # Update metadata
            knowledge["metadata"]["updated"] = datetime.now().isoformat()
            
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(knowledge, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not update knowledge base: {e}")

# Backwards compatibility interface for existing workflows
class LocalResearchInterface:
    """Compatibility interface that replaces external API calls"""
    
    def __init__(self):
        self.engine = LocalLLMResearchEngine()
    
    async def research_with_perplexity_replacement(self, query: str, context: str = "") -> str:
        """Drop-in replacement for Perplexity API calls"""
        result = await self.engine.conduct_research(query, context=context)
        return result["response"]
    
    async def autonomous_research_loop(self, problem: str, context: str = "") -> Dict[str, Any]:
        """Drop-in replacement for autonomous research workflow"""
        return await self.engine.autonomous_stuck_handler(problem, context)

# Example usage and testing
if __name__ == "__main__":
    async def test_local_research():
        engine = LocalLLMResearchEngine()
        
        # Test basic research
        result = await engine.conduct_research(
            "How to optimize recursive task decomposition for autonomous systems?",
            research_type="technical"
        )
        
        print("Research Result:")
        print(json.dumps(result, indent=2))
        
        # Test stuck handler
        stuck_result = await engine.autonomous_stuck_handler(
            "Implementation of recursive PRD decomposition is stalling",
            "Working on Task Master AI migration to local LLMs"
        )
        
        print("\nStuck Handler Result:")
        print(json.dumps(stuck_result, indent=2))
    
    # Run test
    asyncio.run(test_local_research())