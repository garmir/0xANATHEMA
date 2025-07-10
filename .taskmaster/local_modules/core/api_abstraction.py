#!/usr/bin/env python3
"""
API Abstraction Layer for Task Master AI Local LLM Integration
Provides unified interface for local and external model access with intelligent routing
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import httpx
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Supported model providers"""
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    LOCAL_AI = "local_ai"
    TEXT_GEN_WEBUI = "text_gen_webui"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"

class TaskType(Enum):
    """Task types for intelligent model routing"""
    RESEARCH = "research"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    OPTIMIZATION = "optimization"
    GENERAL = "general"

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    provider: ModelProvider
    model_name: str
    endpoint: str
    api_key: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 120
    priority: int = 1  # Higher values = higher priority
    capabilities: List[TaskType] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = [TaskType.GENERAL]

@dataclass
class ModelResponse:
    """Standardized response from model inference"""
    content: str
    model_used: str
    provider: ModelProvider
    timestamp: float
    execution_time: float
    tokens_used: Optional[int] = None
    cached: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseModelAdapter(ABC):
    """Base class for model adapters"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = None
        self.is_healthy = True
        self.last_health_check = 0
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from model"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if model is available and healthy"""
        pass
    
    async def _ensure_healthy(self):
        """Ensure model is healthy before use"""
        current_time = time.time()
        if current_time - self.last_health_check > 300:  # 5 minute cache
            self.is_healthy = await self.health_check()
            self.last_health_check = current_time
        
        if not self.is_healthy:
            raise Exception(f"Model {self.config.model_name} is not healthy")

class OllamaAdapter(BaseModelAdapter):
    """Adapter for Ollama models"""
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        await self._ensure_healthy()
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", self.config.temperature),
                        "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
                    }
                }
                
                response = await client.post(
                    f"{self.config.endpoint}/api/generate",
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return ModelResponse(
                        content=data.get("response", "").strip(),
                        model_used=self.config.model_name,
                        provider=ModelProvider.OLLAMA,
                        timestamp=time.time(),
                        execution_time=time.time() - start_time,
                        metadata={"endpoint": self.config.endpoint}
                    )
                else:
                    raise Exception(f"Ollama API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.endpoint}/api/version", timeout=10)
                return response.status_code == 200
        except:
            return False

class LocalAIAdapter(BaseModelAdapter):
    """Adapter for LocalAI models"""
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        await self._ensure_healthy()
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "model": self.config.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "temperature": kwargs.get("temperature", self.config.temperature)
                }
                
                response = await client.post(
                    f"{self.config.endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return ModelResponse(
                        content=data["choices"][0]["message"]["content"].strip(),
                        model_used=self.config.model_name,
                        provider=ModelProvider.LOCAL_AI,
                        timestamp=time.time(),
                        execution_time=time.time() - start_time,
                        tokens_used=data.get("usage", {}).get("total_tokens"),
                        metadata={"endpoint": self.config.endpoint}
                    )
                else:
                    raise Exception(f"LocalAI API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"LocalAI generation failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.endpoint}/v1/models", timeout=10)
                return response.status_code == 200
        except:
            return False

class LMStudioAdapter(BaseModelAdapter):
    """Adapter for LM Studio models"""
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        await self._ensure_healthy()
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "model": self.config.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "stream": False
                }
                
                response = await client.post(
                    f"{self.config.endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return ModelResponse(
                        content=data["choices"][0]["message"]["content"].strip(),
                        model_used=self.config.model_name,
                        provider=ModelProvider.LM_STUDIO,
                        timestamp=time.time(),
                        execution_time=time.time() - start_time,
                        tokens_used=data.get("usage", {}).get("total_tokens"),
                        metadata={"endpoint": self.config.endpoint}
                    )
                else:
                    raise Exception(f"LM Studio API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"LM Studio generation failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.endpoint}/v1/models", timeout=10)
                return response.status_code == 200
        except:
            return False

class ModelRouter:
    """Intelligent model routing and load balancing"""
    
    def __init__(self, cache_dir: str = ".taskmaster/local_modules/cache"):
        self.adapters: Dict[str, BaseModelAdapter] = {}
        self.task_model_map: Dict[TaskType, List[str]] = {}
        self.performance_cache = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load performance history
        self._load_performance_history()
    
    def register_model(self, model_id: str, config: ModelConfig):
        """Register a new model with the router"""
        # Create appropriate adapter
        if config.provider == ModelProvider.OLLAMA:
            adapter = OllamaAdapter(config)
        elif config.provider == ModelProvider.LOCAL_AI:
            adapter = LocalAIAdapter(config)
        elif config.provider == ModelProvider.LM_STUDIO:
            adapter = LMStudioAdapter(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        self.adapters[model_id] = adapter
        
        # Update task mapping
        for task_type in config.capabilities:
            if task_type not in self.task_model_map:
                self.task_model_map[task_type] = []
            self.task_model_map[task_type].append(model_id)
        
        logger.info(f"Registered model {model_id} with provider {config.provider}")
    
    def _load_performance_history(self):
        """Load performance history from cache"""
        try:
            history_file = self.cache_dir / "performance_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.performance_cache = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load performance history: {e}")
            self.performance_cache = {}
    
    def _save_performance_history(self):
        """Save performance history to cache"""
        try:
            history_file = self.cache_dir / "performance_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.performance_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save performance history: {e}")
    
    def _update_performance_metrics(self, model_id: str, response: ModelResponse):
        """Update performance metrics for a model"""
        if model_id not in self.performance_cache:
            self.performance_cache[model_id] = {
                "total_requests": 0,
                "total_time": 0,
                "success_rate": 1.0,
                "avg_response_time": 0
            }
        
        metrics = self.performance_cache[model_id]
        metrics["total_requests"] += 1
        metrics["total_time"] += response.execution_time
        metrics["avg_response_time"] = metrics["total_time"] / metrics["total_requests"]
        
        # Save periodically
        if metrics["total_requests"] % 10 == 0:
            self._save_performance_history()
    
    def _select_best_model(self, task_type: TaskType, exclude_models: List[str] = None) -> Optional[str]:
        """Select the best model for a given task type"""
        if exclude_models is None:
            exclude_models = []
        
        # Get available models for task type
        available_models = self.task_model_map.get(task_type, [])
        available_models = [m for m in available_models if m not in exclude_models]
        
        if not available_models:
            # Fallback to general models
            available_models = self.task_model_map.get(TaskType.GENERAL, [])
            available_models = [m for m in available_models if m not in exclude_models]
        
        if not available_models:
            return None
        
        # Sort by performance metrics and priority
        def model_score(model_id: str) -> float:
            adapter = self.adapters.get(model_id)
            if not adapter or not adapter.is_healthy:
                return -1
            
            # Base score from priority
            score = adapter.config.priority
            
            # Adjust based on performance history
            if model_id in self.performance_cache:
                metrics = self.performance_cache[model_id]
                # Prefer faster models with high success rate
                time_factor = 1.0 / (1.0 + metrics["avg_response_time"])
                success_factor = metrics["success_rate"]
                score *= time_factor * success_factor
            
            return score
        
        # Sort by score (highest first)
        available_models.sort(key=model_score, reverse=True)
        return available_models[0] if available_models else None
    
    async def generate(self, 
                      prompt: str, 
                      task_type: TaskType = TaskType.GENERAL,
                      fallback_attempts: int = 3,
                      **kwargs) -> ModelResponse:
        """Generate response with intelligent model selection and fallback"""
        last_error = None
        excluded_models = []
        
        for attempt in range(fallback_attempts):
            try:
                # Select best available model
                model_id = self._select_best_model(task_type, excluded_models)
                if not model_id:
                    raise Exception("No available models for task type")
                
                # Generate response
                adapter = self.adapters[model_id]
                response = await adapter.generate(prompt, **kwargs)
                
                # Update performance metrics
                self._update_performance_metrics(model_id, response)
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_id} failed (attempt {attempt + 1}): {e}")
                
                # Mark model as unhealthy and exclude from next attempts
                if model_id in self.adapters:
                    self.adapters[model_id].is_healthy = False
                    excluded_models.append(model_id)
        
        # All attempts failed
        raise Exception(f"All models failed after {fallback_attempts} attempts. Last error: {last_error}")
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered models"""
        health_status = {}
        
        for model_id, adapter in self.adapters.items():
            try:
                health_status[model_id] = await adapter.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {model_id}: {e}")
                health_status[model_id] = False
        
        return health_status

class ResponseCache:
    """Cache for model responses to improve performance"""
    
    def __init__(self, cache_dir: str = ".taskmaster/local_modules/cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
    
    def _get_cache_key(self, prompt: str, task_type: TaskType, **kwargs) -> str:
        """Generate cache key for a request"""
        key_data = {
            "prompt": prompt,
            "task_type": task_type.value,
            "kwargs": kwargs
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def get(self, prompt: str, task_type: TaskType, **kwargs) -> Optional[ModelResponse]:
        """Get cached response if available and not expired"""
        cache_key = self._get_cache_key(prompt, task_type, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid
            if time.time() - cached_data["timestamp"] > self.ttl:
                cache_file.unlink()  # Remove expired cache
                return None
            
            # Reconstruct ModelResponse
            response = ModelResponse(
                content=cached_data["content"],
                model_used=cached_data["model_used"],
                provider=ModelProvider(cached_data["provider"]),
                timestamp=cached_data["timestamp"],
                execution_time=cached_data["execution_time"],
                tokens_used=cached_data.get("tokens_used"),
                cached=True,
                metadata=cached_data.get("metadata", {})
            )
            
            return response
            
        except Exception as e:
            logger.warning(f"Failed to load cached response: {e}")
            return None
    
    def set(self, prompt: str, task_type: TaskType, response: ModelResponse, **kwargs):
        """Cache a response"""
        cache_key = self._get_cache_key(prompt, task_type, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cache_data = {
                "content": response.content,
                "model_used": response.model_used,
                "provider": response.provider.value,
                "timestamp": response.timestamp,
                "execution_time": response.execution_time,
                "tokens_used": response.tokens_used,
                "metadata": response.metadata
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

class UnifiedModelAPI:
    """Unified API that combines routing, caching, and fallback capabilities"""
    
    def __init__(self, cache_ttl: int = 3600):
        self.router = ModelRouter()
        self.cache = ResponseCache(ttl=cache_ttl)
    
    def add_model(self, model_id: str, config: ModelConfig):
        """Add a model to the unified API"""
        self.router.register_model(model_id, config)
    
    async def generate(self, 
                      prompt: str, 
                      task_type: TaskType = TaskType.GENERAL,
                      use_cache: bool = True,
                      **kwargs) -> ModelResponse:
        """Generate response with caching and intelligent routing"""
        # Check cache first
        if use_cache:
            cached_response = self.cache.get(prompt, task_type, **kwargs)
            if cached_response:
                logger.info(f"Cache hit for task type {task_type.value}")
                return cached_response
        
        # Generate new response
        response = await self.router.generate(prompt, task_type, **kwargs)
        
        # Cache the response
        if use_cache:
            self.cache.set(prompt, task_type, response, **kwargs)
        
        return response
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all models"""
        return await self.router.health_check_all()

# Factory for creating common model configurations
class ModelConfigFactory:
    """Factory for creating standard model configurations"""
    
    @staticmethod
    def create_ollama_config(model_name: str, 
                           endpoint: str = "http://localhost:11434",
                           capabilities: List[TaskType] = None) -> ModelConfig:
        """Create configuration for Ollama model"""
        if capabilities is None:
            capabilities = [TaskType.GENERAL]
        
        return ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name=model_name,
            endpoint=endpoint,
            capabilities=capabilities,
            priority=2
        )
    
    @staticmethod
    def create_lm_studio_config(model_name: str,
                              endpoint: str = "http://localhost:1234",
                              capabilities: List[TaskType] = None) -> ModelConfig:
        """Create configuration for LM Studio model"""
        if capabilities is None:
            capabilities = [TaskType.GENERAL, TaskType.CODE_GENERATION]
        
        return ModelConfig(
            provider=ModelProvider.LM_STUDIO,
            model_name=model_name,
            endpoint=endpoint,
            capabilities=capabilities,
            priority=2
        )
    
    @staticmethod
    def create_local_ai_config(model_name: str,
                             endpoint: str = "http://localhost:8080",
                             capabilities: List[TaskType] = None) -> ModelConfig:
        """Create configuration for LocalAI model"""
        if capabilities is None:
            capabilities = [TaskType.GENERAL]
        
        return ModelConfig(
            provider=ModelProvider.LOCAL_AI,
            model_name=model_name,
            endpoint=endpoint,
            capabilities=capabilities,
            priority=2
        )

# Example usage
if __name__ == "__main__":
    async def test_api_abstraction():
        # Initialize unified API
        api = UnifiedModelAPI()
        
        # Add models
        api.add_model("ollama-llama2", ModelConfigFactory.create_ollama_config(
            "llama2", capabilities=[TaskType.GENERAL, TaskType.ANALYSIS]
        ))
        api.add_model("ollama-mistral", ModelConfigFactory.create_ollama_config(
            "mistral", capabilities=[TaskType.RESEARCH, TaskType.ANALYSIS]
        ))
        api.add_model("lm-studio-codellama", ModelConfigFactory.create_lm_studio_config(
            "codellama", capabilities=[TaskType.CODE_GENERATION, TaskType.ANALYSIS]
        ))
        
        # Test generation
        try:
            response = await api.generate(
                "Explain the benefits of local LLM deployment",
                task_type=TaskType.RESEARCH
            )
            print(f"Response from {response.model_used}: {response.content[:200]}...")
            print(f"Execution time: {response.execution_time:.2f}s")
            print(f"Cached: {response.cached}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        # Health check
        health = await api.health_check()
        print(f"Health status: {health}")
    
    # Run test
    asyncio.run(test_api_abstraction())