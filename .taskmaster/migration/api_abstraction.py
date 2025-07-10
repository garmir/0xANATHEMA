#!/usr/bin/env python3
"""
API Abstraction Layer for Task Master AI Local LLM Migration
Provides unified interface for all LLM providers with intelligent routing and fallback
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import concurrent.futures
from collections import defaultdict

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class TaskComplexity(Enum):
    """Task complexity levels for intelligent routing"""
    SIMPLE = "simple"           # Quick responses, basic queries
    MODERATE = "moderate"       # Standard processing, medium context
    COMPLEX = "complex"         # Deep analysis, large context
    CRITICAL = "critical"       # Maximum quality, highest priority

class ModelCapability(Enum):
    """Model capabilities for provider selection"""
    RESEARCH = "research"
    PLANNING = "planning"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    CHAT = "chat"

class ProviderStatus(Enum):
    """Provider availability status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

@dataclass
class ModelConfig:
    """Configuration for individual models"""
    provider_id: str
    model_name: str
    endpoint: str
    api_key: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 60
    capabilities: List[ModelCapability] = None
    complexity_levels: List[TaskComplexity] = None
    priority: int = 1  # Higher number = higher priority
    cost_per_token: float = 0.0  # For cost optimization
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = list(ModelCapability)
        if self.complexity_levels is None:
            self.complexity_levels = list(TaskComplexity)

@dataclass
class LLMRequest:
    """Standardized request structure"""
    messages: List[Dict[str, str]]
    task_type: ModelCapability
    complexity: TaskComplexity
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    timeout: Optional[int] = None
    metadata: Dict[str, Any] = None
    priority: int = 1
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.metadata['request_id'] = self._generate_request_id()
        self.metadata['timestamp'] = datetime.now().isoformat()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        content = json.dumps(self.messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass
class LLMResponse:
    """Standardized response structure"""
    content: str
    provider_id: str
    model_name: str
    request_id: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.usage is None:
            self.usage = {}
        self.metadata['response_timestamp'] = datetime.now().isoformat()

@dataclass
class ProviderHealth:
    """Provider health status"""
    provider_id: str
    status: ProviderStatus
    response_time: float = 0.0
    success_rate: float = 0.0
    last_check: datetime = None
    error_count: int = 0
    total_requests: int = 0
    
    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now()

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.health = ProviderHealth(provider_id=config.provider_id, status=ProviderStatus.OFFLINE)
        self.client = None
        
    @abstractmethod
    async def initialize(self):
        """Initialize provider connection"""
        pass
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response for request"""
        pass
    
    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Check provider health"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup provider resources"""
        pass

class OllamaProvider(LLMProvider):
    """Ollama local LLM provider"""
    
    async def initialize(self):
        """Initialize Ollama connection"""
        if HTTPX_AVAILABLE:
            self.client = httpx.AsyncClient(timeout=self.config.timeout)
            await self.health_check()
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Ollama"""
        if not self.client:
            return LLMResponse(
                content="",
                provider_id=self.config.provider_id,
                model_name=self.config.model_name,
                request_id=request.metadata['request_id'],
                success=False,
                error="Client not initialized"
            )
        
        try:
            # Convert messages to Ollama prompt format
            prompt = self._messages_to_prompt(request.messages)
            
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature or self.config.temperature,
                    "num_predict": request.max_tokens or self.config.max_tokens
                }
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.config.endpoint}/api/generate",
                json=payload,
                timeout=request.timeout or self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            execution_time = time.time() - start_time
            
            return LLMResponse(
                content=result.get("response", ""),
                provider_id=self.config.provider_id,
                model_name=self.config.model_name,
                request_id=request.metadata['request_id'],
                success=True,
                metadata={
                    "execution_time": execution_time,
                    "provider_type": "ollama"
                },
                usage={
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                provider_id=self.config.provider_id,
                model_name=self.config.model_name,
                request_id=request.metadata['request_id'],
                success=False,
                error=str(e)
            )
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Ollama prompt"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    async def health_check(self) -> ProviderHealth:
        """Check Ollama health"""
        try:
            if not self.client:
                self.health.status = ProviderStatus.OFFLINE
                return self.health
            
            start_time = time.time()
            response = await self.client.get(f"{self.config.endpoint}/api/tags")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.health.status = ProviderStatus.HEALTHY
                self.health.response_time = response_time
            else:
                self.health.status = ProviderStatus.DEGRADED
                
        except Exception as e:
            self.health.status = ProviderStatus.UNHEALTHY
            self.health.error_count += 1
        
        self.health.last_check = datetime.now()
        return self.health
    
    async def cleanup(self):
        """Cleanup Ollama resources"""
        if self.client:
            await self.client.aclose()

class LocalAIProvider(LLMProvider):
    """LocalAI provider (OpenAI-compatible)"""
    
    async def initialize(self):
        """Initialize LocalAI connection"""
        if HTTPX_AVAILABLE:
            self.client = httpx.AsyncClient(timeout=self.config.timeout)
            await self.health_check()
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using LocalAI"""
        if not self.client:
            return LLMResponse(
                content="",
                provider_id=self.config.provider_id,
                model_name=self.config.model_name,
                request_id=request.metadata['request_id'],
                success=False,
                error="Client not initialized"
            )
        
        try:
            payload = {
                "model": self.config.model_name,
                "messages": request.messages,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "stream": False
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.config.endpoint}/v1/chat/completions",
                json=payload,
                timeout=request.timeout or self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            execution_time = time.time() - start_time
            
            return LLMResponse(
                content=result["choices"][0]["message"]["content"],
                provider_id=self.config.provider_id,
                model_name=self.config.model_name,
                request_id=request.metadata['request_id'],
                success=True,
                metadata={
                    "execution_time": execution_time,
                    "provider_type": "localai"
                },
                usage=result.get("usage", {})
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                provider_id=self.config.provider_id,
                model_name=self.config.model_name,
                request_id=request.metadata['request_id'],
                success=False,
                error=str(e)
            )
    
    async def health_check(self) -> ProviderHealth:
        """Check LocalAI health"""
        try:
            if not self.client:
                self.health.status = ProviderStatus.OFFLINE
                return self.health
            
            start_time = time.time()
            response = await self.client.get(f"{self.config.endpoint}/v1/models")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.health.status = ProviderStatus.HEALTHY
                self.health.response_time = response_time
            else:
                self.health.status = ProviderStatus.DEGRADED
                
        except Exception as e:
            self.health.status = ProviderStatus.UNHEALTHY
            self.health.error_count += 1
        
        self.health.last_check = datetime.now()
        return self.health
    
    async def cleanup(self):
        """Cleanup LocalAI resources"""
        if self.client:
            await self.client.aclose()

class CacheManager:
    """Intelligent caching for LLM responses"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.ttl = ttl
        self.local_cache = {}
        self.redis_client = None
        
        if REDIS_AVAILABLE:
            try:
                import redis.asyncio as aioredis
                self.redis_client = aioredis.from_url(redis_url)
            except ImportError:
                pass
    
    def _get_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request"""
        cache_data = {
            "messages": request.messages,
            "task_type": request.task_type.value,
            "complexity": request.complexity.value
        }
        content = json.dumps(cache_data, sort_keys=True)
        return f"llm_cache:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Get cached response"""
        cache_key = self._get_cache_key(request)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    return LLMResponse(**data)
            except Exception:
                pass
        
        # Try local cache
        if cache_key in self.local_cache:
            cache_entry = self.local_cache[cache_key]
            if cache_entry['expires'] > datetime.now():
                return LLMResponse(**cache_entry['data'])
            else:
                del self.local_cache[cache_key]
        
        return None
    
    async def set(self, request: LLMRequest, response: LLMResponse):
        """Cache response"""
        if not response.success:
            return  # Don't cache failed responses
        
        cache_key = self._get_cache_key(request)
        cache_data = asdict(response)
        
        # Cache in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    self.ttl,
                    json.dumps(cache_data, default=str)
                )
            except Exception:
                pass
        
        # Cache locally
        self.local_cache[cache_key] = {
            'data': cache_data,
            'expires': datetime.now() + timedelta(seconds=self.ttl)
        }
    
    async def cleanup(self):
        """Cleanup cache resources"""
        if self.redis_client:
            await self.redis_client.close()

class LoadBalancer:
    """Intelligent load balancer for LLM providers"""
    
    def __init__(self):
        self.provider_metrics = defaultdict(lambda: {
            'requests': 0,
            'successes': 0,
            'total_time': 0.0,
            'errors': 0,
            'last_request': None
        })
        
    def select_provider(self, providers: List[LLMProvider], request: LLMRequest) -> Optional[LLMProvider]:
        """Select best provider for request"""
        # Filter providers by capability and complexity
        suitable_providers = [
            p for p in providers
            if (request.task_type in p.config.capabilities and
                request.complexity in p.config.complexity_levels and
                p.health.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED])
        ]
        
        if not suitable_providers:
            return None
        
        # Score providers
        scored_providers = []
        for provider in suitable_providers:
            score = self._calculate_provider_score(provider, request)
            scored_providers.append((score, provider))
        
        # Sort by score (higher is better)
        scored_providers.sort(reverse=True, key=lambda x: x[0])
        
        return scored_providers[0][1]
    
    def _calculate_provider_score(self, provider: LLMProvider, request: LLMRequest) -> float:
        """Calculate provider score for request"""
        metrics = self.provider_metrics[provider.config.provider_id]
        
        # Base score from provider priority
        score = provider.config.priority * 10
        
        # Health factor
        health_multiplier = {
            ProviderStatus.HEALTHY: 1.0,
            ProviderStatus.DEGRADED: 0.7,
            ProviderStatus.UNHEALTHY: 0.3,
            ProviderStatus.OFFLINE: 0.0
        }
        score *= health_multiplier[provider.health.status]
        
        # Performance factor
        if metrics['requests'] > 0:
            success_rate = metrics['successes'] / metrics['requests']
            avg_response_time = metrics['total_time'] / metrics['successes'] if metrics['successes'] > 0 else 10.0
            
            # Higher success rate is better
            score += success_rate * 20
            
            # Lower response time is better (inverse relationship)
            score += max(0, 10 - avg_response_time)
        
        # Load balancing - prefer less loaded providers
        recent_requests = self._count_recent_requests(provider.config.provider_id)
        score -= recent_requests * 2
        
        return score
    
    def _count_recent_requests(self, provider_id: str, window_minutes: int = 5) -> int:
        """Count recent requests to provider"""
        metrics = self.provider_metrics[provider_id]
        if not metrics['last_request']:
            return 0
        
        time_since_last = datetime.now() - metrics['last_request']
        if time_since_last.total_seconds() > window_minutes * 60:
            return 0
        
        return min(metrics['requests'], 10)  # Cap at 10 for scoring
    
    def record_request(self, provider_id: str, success: bool, response_time: float):
        """Record request metrics"""
        metrics = self.provider_metrics[provider_id]
        metrics['requests'] += 1
        metrics['last_request'] = datetime.now()
        
        if success:
            metrics['successes'] += 1
            metrics['total_time'] += response_time
        else:
            metrics['errors'] += 1

class UnifiedLLMInterface:
    """Unified interface for all LLM providers with intelligent routing"""
    
    def __init__(self, configs: List[ModelConfig], cache_ttl: int = 3600):
        self.providers = {}
        self.cache = CacheManager(ttl=cache_ttl)
        self.load_balancer = LoadBalancer()
        self.request_queue = asyncio.Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        for config in configs:
            if config.provider_id.startswith("ollama"):
                self.providers[config.provider_id] = OllamaProvider(config)
            elif config.provider_id.startswith("localai"):
                self.providers[config.provider_id] = LocalAIProvider(config)
    
    async def initialize(self):
        """Initialize all providers"""
        tasks = []
        for provider in self.providers.values():
            tasks.append(provider.initialize())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Start health monitoring
        asyncio.create_task(self._health_monitor())
        
        self.logger.info(f"Initialized {len(self.providers)} LLM providers")
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response with intelligent routing and caching"""
        # Check cache first
        cached_response = await self.cache.get(request)
        if cached_response:
            self.logger.debug(f"Cache hit for request {request.metadata['request_id']}")
            return cached_response
        
        # Select best provider
        provider = self.load_balancer.select_provider(list(self.providers.values()), request)
        if not provider:
            return LLMResponse(
                content="",
                provider_id="none",
                model_name="none",
                request_id=request.metadata['request_id'],
                success=False,
                error="No suitable provider available"
            )
        
        # Generate response
        start_time = time.time()
        response = await provider.generate(request)
        response_time = time.time() - start_time
        
        # Record metrics
        self.load_balancer.record_request(
            provider.config.provider_id,
            response.success,
            response_time
        )
        
        # Cache successful responses
        if response.success:
            await self.cache.set(request, response)
        
        return response
    
    async def generate_with_fallback(self, request: LLMRequest, max_retries: int = 3) -> LLMResponse:
        """Generate with automatic fallback to other providers"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = await self.generate(request)
                if response.success:
                    return response
                last_error = response.error
            except Exception as e:
                last_error = str(e)
            
            # Wait before retry
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return LLMResponse(
            content="",
            provider_id="fallback",
            model_name="fallback",
            request_id=request.metadata['request_id'],
            success=False,
            error=f"All providers failed after {max_retries} attempts. Last error: {last_error}"
        )
    
    async def _health_monitor(self):
        """Monitor provider health"""
        while True:
            try:
                tasks = []
                for provider in self.providers.values():
                    tasks.append(provider.health_check())
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log health status
                healthy_count = sum(1 for p in self.providers.values() 
                                  if p.health.status == ProviderStatus.HEALTHY)
                self.logger.info(f"Provider health check: {healthy_count}/{len(self.providers)} healthy")
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all providers"""
        status = {}
        for provider_id, provider in self.providers.items():
            health = provider.health
            metrics = self.load_balancer.provider_metrics[provider_id]
            
            status[provider_id] = {
                "status": health.status.value,
                "response_time": health.response_time,
                "success_rate": metrics['successes'] / max(metrics['requests'], 1),
                "total_requests": metrics['requests'],
                "error_count": metrics['errors'],
                "last_check": health.last_check.isoformat() if health.last_check else None
            }
        
        return status
    
    async def cleanup(self):
        """Cleanup all resources"""
        tasks = []
        for provider in self.providers.values():
            tasks.append(provider.cleanup())
        
        tasks.append(self.cache.cleanup())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.executor.shutdown(wait=True)

# Factory functions for common configurations
def create_default_configs() -> List[ModelConfig]:
    """Create default configurations for common providers"""
    return [
        # Primary reasoning models
        ModelConfig(
            provider_id="ollama_llama3_70b",
            model_name="llama3.1:70b-instruct-q4_0",
            endpoint="http://localhost:11434",
            capabilities=[ModelCapability.REASONING, ModelCapability.PLANNING, ModelCapability.ANALYSIS],
            complexity_levels=[TaskComplexity.COMPLEX, TaskComplexity.CRITICAL],
            priority=10,
            max_tokens=4096,
            temperature=0.2
        ),
        
        # Efficient general-purpose models
        ModelConfig(
            provider_id="ollama_mistral_7b",
            model_name="mistral:7b-instruct",
            endpoint="http://localhost:11434",
            capabilities=[ModelCapability.CHAT, ModelCapability.RESEARCH, ModelCapability.ANALYSIS],
            complexity_levels=[TaskComplexity.SIMPLE, TaskComplexity.MODERATE],
            priority=8,
            max_tokens=2048,
            temperature=0.3
        ),
        
        # Code generation models
        ModelConfig(
            provider_id="ollama_codellama_13b",
            model_name="codellama:13b-instruct",
            endpoint="http://localhost:11434",
            capabilities=[ModelCapability.CODE_GENERATION, ModelCapability.ANALYSIS],
            complexity_levels=[TaskComplexity.MODERATE, TaskComplexity.COMPLEX],
            priority=9,
            max_tokens=2048,
            temperature=0.1
        ),
        
        # LocalAI fallback
        ModelConfig(
            provider_id="localai_general",
            model_name="gpt-3.5-turbo",
            endpoint="http://localhost:8080",
            capabilities=list(ModelCapability),
            complexity_levels=list(TaskComplexity),
            priority=5,
            max_tokens=2048,
            temperature=0.3
        )
    ]

# Task Master AI compatibility layer
class TaskMasterLLMInterface:
    """Task Master AI compatible interface"""
    
    def __init__(self, configs: List[ModelConfig] = None):
        if configs is None:
            configs = create_default_configs()
        
        self.llm = UnifiedLLMInterface(configs)
    
    async def initialize(self):
        """Initialize the interface"""
        await self.llm.initialize()
    
    async def research(self, query: str, context: str = "") -> str:
        """Research method compatible with Task Master"""
        messages = [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": f"Query: {query}\nContext: {context}"}
        ]
        
        request = LLMRequest(
            messages=messages,
            task_type=ModelCapability.RESEARCH,
            complexity=TaskComplexity.MODERATE
        )
        
        response = await self.llm.generate_with_fallback(request)
        return response.content
    
    async def plan(self, task: str, context: str = "") -> str:
        """Planning method compatible with Task Master"""
        messages = [
            {"role": "system", "content": "You are an expert task planner."},
            {"role": "user", "content": f"Plan this task: {task}\nContext: {context}"}
        ]
        
        request = LLMRequest(
            messages=messages,
            task_type=ModelCapability.PLANNING,
            complexity=TaskComplexity.COMPLEX
        )
        
        response = await self.llm.generate_with_fallback(request)
        return response.content
    
    async def analyze(self, data: str, focus: str = "") -> str:
        """Analysis method compatible with Task Master"""
        messages = [
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": f"Analyze: {data}\nFocus: {focus}"}
        ]
        
        request = LLMRequest(
            messages=messages,
            task_type=ModelCapability.ANALYSIS,
            complexity=TaskComplexity.COMPLEX
        )
        
        response = await self.llm.generate_with_fallback(request)
        return response.content
    
    async def generate_code(self, specification: str, language: str = "python") -> str:
        """Code generation method compatible with Task Master"""
        messages = [
            {"role": "system", "content": f"You are an expert {language} programmer."},
            {"role": "user", "content": f"Generate {language} code for: {specification}"}
        ]
        
        request = LLMRequest(
            messages=messages,
            task_type=ModelCapability.CODE_GENERATION,
            complexity=TaskComplexity.MODERATE
        )
        
        response = await self.llm.generate_with_fallback(request)
        return response.content
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        return await self.llm.get_health_status()
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.llm.cleanup()

async def main():
    """Test the API abstraction layer"""
    print("🔧 Testing API Abstraction Layer")
    print("=" * 50)
    
    # Create interface
    interface = TaskMasterLLMInterface()
    
    try:
        await interface.initialize()
        
        # Test health status
        health = await interface.get_health_status()
        print("Provider Health Status:")
        for provider, status in health.items():
            print(f"  {provider}: {status['status']} (Success rate: {status['success_rate']:.2%})")
        
        # Test research capability
        print("\nTesting research capability...")
        research_result = await interface.research(
            "What are the key benefits of local LLM deployment?",
            "Task Master AI migration context"
        )
        print(f"Research result: {research_result[:200]}...")
        
        # Test planning capability
        print("\nTesting planning capability...")
        plan_result = await interface.plan(
            "Implement caching layer for LLM responses",
            "Performance optimization context"
        )
        print(f"Plan result: {plan_result[:200]}...")
        
        print("\n✅ API Abstraction Layer test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await interface.cleanup()

if __name__ == "__main__":
    asyncio.run(main())