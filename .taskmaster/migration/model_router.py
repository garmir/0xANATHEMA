#!/usr/bin/env python3
"""
Intelligent Model Router for Task Master AI Local LLM Migration
Routes requests to optimal models based on task complexity, resource availability, and performance
"""

import asyncio
import json
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from api_abstraction import (
    ModelConfig, LLMRequest, LLMResponse, TaskComplexity, 
    ModelCapability, ProviderStatus, UnifiedLLMInterface
)

class ResourceType(Enum):
    """System resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU_MEMORY = "gpu_memory"
    DISK_IO = "disk_io"
    NETWORK = "network"

class ModelTier(Enum):
    """Model performance tiers"""
    LIGHTWEIGHT = "lightweight"    # 1-7B parameters
    STANDARD = "standard"          # 8-20B parameters
    HEAVY = "heavy"               # 21-70B parameters
    ENTERPRISE = "enterprise"      # 70B+ parameters

@dataclass
class ResourceMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: float
    disk_io_percent: float
    network_io_percent: float
    timestamp: datetime
    
    def is_high_load(self, threshold: float = 80.0) -> bool:
        """Check if system is under high load"""
        return (self.cpu_percent > threshold or 
                self.memory_percent > threshold or
                self.gpu_memory_percent > threshold)

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for individual models"""
    model_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    total_tokens_generated: int = 0
    average_tokens_per_second: float = 0.0
    quality_score: float = 0.0
    last_used: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests

@dataclass
class RoutingDecision:
    """Model routing decision with reasoning"""
    selected_model: str
    confidence: float
    reasoning: List[str]
    alternative_models: List[str]
    estimated_response_time: float
    resource_requirements: Dict[str, float]

class ResourceMonitor:
    """Real-time system resource monitoring"""
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.current_metrics = None
        self.metrics_history = deque(maxlen=100)
        self.monitoring = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Resource monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU metrics (simplified - would use nvidia-ml-py in production)
        gpu_memory_percent = 0.0
        try:
            # Placeholder for GPU monitoring
            gpu_memory_percent = 0.0
        except:
            pass
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        disk_io_percent = 0.0  # Would calculate based on baseline
        
        # Network I/O metrics
        network_io = psutil.net_io_counters()
        network_io_percent = 0.0  # Would calculate based on baseline
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_memory_percent=gpu_memory_percent,
            disk_io_percent=disk_io_percent,
            network_io_percent=network_io_percent,
            timestamp=datetime.now()
        )
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get current resource metrics"""
        return self.current_metrics
    
    def get_average_metrics(self, minutes: int = 5) -> Optional[ResourceMetrics]:
        """Get average metrics over specified time window"""
        if not self.metrics_history:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_gpu = sum(m.gpu_memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_io_percent for m in recent_metrics) / len(recent_metrics)
        avg_network = sum(m.network_io_percent for m in recent_metrics) / len(recent_metrics)
        
        return ResourceMetrics(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            gpu_memory_percent=avg_gpu,
            disk_io_percent=avg_disk,
            network_io_percent=avg_network,
            timestamp=datetime.now()
        )

class ModelCapabilityAnalyzer:
    """Analyzes and scores model capabilities for different tasks"""
    
    def __init__(self):
        self.capability_scores = self._initialize_capability_scores()
        self.complexity_multipliers = self._initialize_complexity_multipliers()
    
    def _initialize_capability_scores(self) -> Dict[str, Dict[ModelCapability, float]]:
        """Initialize capability scores for different model types"""
        return {
            # Large models (70B+)
            "llama3.1:70b": {
                ModelCapability.REASONING: 0.95,
                ModelCapability.PLANNING: 0.95,
                ModelCapability.ANALYSIS: 0.92,
                ModelCapability.RESEARCH: 0.88,
                ModelCapability.CODE_GENERATION: 0.85,
                ModelCapability.CHAT: 0.90
            },
            
            # Medium models (13B-20B)
            "codellama:13b": {
                ModelCapability.CODE_GENERATION: 0.95,
                ModelCapability.ANALYSIS: 0.85,
                ModelCapability.REASONING: 0.80,
                ModelCapability.PLANNING: 0.75,
                ModelCapability.RESEARCH: 0.70,
                ModelCapability.CHAT: 0.75
            },
            
            # Small models (7B-8B)
            "mistral:7b": {
                ModelCapability.CHAT: 0.90,
                ModelCapability.RESEARCH: 0.85,
                ModelCapability.ANALYSIS: 0.80,
                ModelCapability.REASONING: 0.75,
                ModelCapability.PLANNING: 0.70,
                ModelCapability.CODE_GENERATION: 0.65
            },
            
            # Specialized models
            "qwen:32b": {
                ModelCapability.RESEARCH: 0.95,
                ModelCapability.ANALYSIS: 0.92,
                ModelCapability.REASONING: 0.88,
                ModelCapability.PLANNING: 0.85,
                ModelCapability.CHAT: 0.82,
                ModelCapability.CODE_GENERATION: 0.75
            }
        }
    
    def _initialize_complexity_multipliers(self) -> Dict[TaskComplexity, float]:
        """Initialize complexity multipliers"""
        return {
            TaskComplexity.SIMPLE: 1.0,
            TaskComplexity.MODERATE: 1.2,
            TaskComplexity.COMPLEX: 1.5,
            TaskComplexity.CRITICAL: 2.0
        }
    
    def score_model_for_task(self, model_name: str, capability: ModelCapability, 
                           complexity: TaskComplexity) -> float:
        """Score model for specific task"""
        # Get base capability score
        base_score = self.capability_scores.get(model_name, {}).get(capability, 0.5)
        
        # Apply complexity multiplier
        complexity_multiplier = self.complexity_multipliers.get(complexity, 1.0)
        
        # Higher complexity tasks may benefit more from capable models
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]:
            if base_score > 0.8:
                base_score *= 1.1  # Boost for high-capability models
        
        return min(base_score * complexity_multiplier, 1.0)

class QueueManager:
    """Manages request queues and load balancing"""
    
    def __init__(self, max_queue_size: int = 100):
        self.max_queue_size = max_queue_size
        self.queues = defaultdict(lambda: asyncio.Queue(maxsize=max_queue_size))
        self.processing_counts = defaultdict(int)
        self.queue_metrics = defaultdict(lambda: {
            'total_processed': 0,
            'total_wait_time': 0.0,
            'current_size': 0
        })
    
    async def enqueue_request(self, model_id: str, request: LLMRequest, 
                            priority: int = 1) -> bool:
        """Enqueue request for specific model"""
        queue = self.queues[model_id]
        
        if queue.full():
            return False
        
        request.metadata['queue_entry_time'] = datetime.now()
        request.metadata['priority'] = priority
        
        await queue.put(request)
        self.queue_metrics[model_id]['current_size'] = queue.qsize()
        
        return True
    
    async def dequeue_request(self, model_id: str) -> Optional[LLMRequest]:
        """Dequeue request for specific model"""
        queue = self.queues[model_id]
        
        try:
            request = await asyncio.wait_for(queue.get(), timeout=0.1)
            
            # Calculate wait time
            if 'queue_entry_time' in request.metadata:
                wait_time = (datetime.now() - request.metadata['queue_entry_time']).total_seconds()
                self.queue_metrics[model_id]['total_wait_time'] += wait_time
            
            self.queue_metrics[model_id]['current_size'] = queue.qsize()
            self.queue_metrics[model_id]['total_processed'] += 1
            
            return request
            
        except asyncio.TimeoutError:
            return None
    
    def get_queue_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all queues"""
        status = {}
        for model_id, metrics in self.queue_metrics.items():
            queue = self.queues[model_id]
            status[model_id] = {
                'current_size': queue.qsize(),
                'processing_count': self.processing_counts[model_id],
                'total_processed': metrics['total_processed'],
                'average_wait_time': (metrics['total_wait_time'] / max(metrics['total_processed'], 1)),
                'is_full': queue.full()
            }
        return status

class IntelligentModelRouter:
    """Intelligent router for selecting optimal models based on multiple factors"""
    
    def __init__(self, llm_interface: UnifiedLLMInterface, enable_monitoring: bool = True):
        self.llm_interface = llm_interface
        self.resource_monitor = ResourceMonitor()
        self.capability_analyzer = ModelCapabilityAnalyzer()
        self.queue_manager = QueueManager()
        
        # Performance tracking
        self.model_metrics = defaultdict(ModelPerformanceMetrics)
        self.routing_history = deque(maxlen=1000)
        
        # Configuration
        self.resource_thresholds = {
            ResourceType.CPU: 80.0,
            ResourceType.MEMORY: 85.0,
            ResourceType.GPU_MEMORY: 90.0
        }
        
        self.model_tiers = self._classify_models()
        self.logger = logging.getLogger(__name__)
        
        if enable_monitoring:
            self.resource_monitor.start_monitoring()
    
    def _classify_models(self) -> Dict[str, ModelTier]:
        """Classify models into performance tiers"""
        return {
            "llama3.1:70b": ModelTier.ENTERPRISE,
            "qwen:32b": ModelTier.HEAVY,
            "codellama:13b": ModelTier.STANDARD,
            "mistral:7b": ModelTier.LIGHTWEIGHT,
            "llama3.1:8b": ModelTier.LIGHTWEIGHT
        }
    
    def select_optimal_model(self, request: LLMRequest) -> RoutingDecision:
        """Select optimal model for request based on multiple factors"""
        available_models = self._get_available_models()
        
        if not available_models:
            return RoutingDecision(
                selected_model="none",
                confidence=0.0,
                reasoning=["No models available"],
                alternative_models=[],
                estimated_response_time=0.0,
                resource_requirements={}
            )
        
        # Score all available models
        model_scores = []
        for model_config in available_models:
            score = self._score_model_for_request(model_config, request)
            model_scores.append((score, model_config))
        
        # Sort by score (highest first)
        model_scores.sort(reverse=True, key=lambda x: x[0])
        
        best_score, best_model = model_scores[0]
        alternatives = [model.provider_id for score, model in model_scores[1:4]]
        
        # Generate reasoning
        reasoning = self._generate_routing_reasoning(best_model, request)
        
        # Estimate response time and resource requirements
        estimated_time = self._estimate_response_time(best_model, request)
        resource_reqs = self._estimate_resource_requirements(best_model, request)
        
        decision = RoutingDecision(
            selected_model=best_model.provider_id,
            confidence=min(best_score, 1.0),
            reasoning=reasoning,
            alternative_models=alternatives,
            estimated_response_time=estimated_time,
            resource_requirements=resource_reqs
        )
        
        # Record decision
        self.routing_history.append({
            'timestamp': datetime.now(),
            'request_id': request.metadata.get('request_id'),
            'decision': asdict(decision)
        })
        
        return decision
    
    def _get_available_models(self) -> List[ModelConfig]:
        """Get list of available models"""
        available = []
        for provider in self.llm_interface.providers.values():
            if provider.health.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]:
                available.append(provider.config)
        return available
    
    def _score_model_for_request(self, model_config: ModelConfig, request: LLMRequest) -> float:
        """Score model for specific request"""
        # Base capability score
        capability_score = self.capability_analyzer.score_model_for_task(
            model_config.model_name,
            request.task_type,
            request.complexity
        )
        
        # Performance score based on historical metrics
        metrics = self.model_metrics[model_config.provider_id]
        performance_score = self._calculate_performance_score(metrics)
        
        # Resource availability score
        resource_score = self._calculate_resource_score(model_config)
        
        # Queue status score
        queue_score = self._calculate_queue_score(model_config.provider_id)
        
        # Priority boost
        priority_score = request.priority / 10.0
        
        # Weighted combination
        total_score = (
            capability_score * 0.4 +
            performance_score * 0.25 +
            resource_score * 0.2 +
            queue_score * 0.1 +
            priority_score * 0.05
        )
        
        return total_score
    
    def _calculate_performance_score(self, metrics: ModelPerformanceMetrics) -> float:
        """Calculate performance score from historical metrics"""
        if metrics.total_requests == 0:
            return 0.7  # Default score for untested models
        
        # Success rate component
        success_component = metrics.success_rate
        
        # Response time component (inverse relationship)
        avg_time = metrics.average_response_time
        time_component = max(0, 1.0 - (avg_time / 30.0))  # 30s as baseline
        
        # Tokens per second component
        tps_component = min(1.0, metrics.average_tokens_per_second / 20.0)  # 20 tps as baseline
        
        # Quality component
        quality_component = metrics.quality_score
        
        return (success_component * 0.4 + 
                time_component * 0.3 + 
                tps_component * 0.2 + 
                quality_component * 0.1)
    
    def _calculate_resource_score(self, model_config: ModelConfig) -> float:
        """Calculate resource availability score"""
        current_metrics = self.resource_monitor.get_current_metrics()
        if not current_metrics:
            return 0.5  # Default if no metrics available
        
        # Get model tier resource requirements
        tier = self.model_tiers.get(model_config.model_name, ModelTier.STANDARD)
        
        resource_multipliers = {
            ModelTier.LIGHTWEIGHT: 0.2,
            ModelTier.STANDARD: 0.5,
            ModelTier.HEAVY: 0.8,
            ModelTier.ENTERPRISE: 1.0
        }
        
        base_requirement = resource_multipliers[tier]
        
        # Calculate availability based on current usage
        cpu_availability = max(0, 1.0 - (current_metrics.cpu_percent / 100.0))
        memory_availability = max(0, 1.0 - (current_metrics.memory_percent / 100.0))
        gpu_availability = max(0, 1.0 - (current_metrics.gpu_memory_percent / 100.0))
        
        # Weight by model requirements
        weighted_availability = (
            cpu_availability * 0.3 +
            memory_availability * 0.3 +
            gpu_availability * 0.4
        )
        
        # Penalize if model requires more resources than available
        if current_metrics.is_high_load() and tier in [ModelTier.HEAVY, ModelTier.ENTERPRISE]:
            weighted_availability *= 0.5
        
        return weighted_availability
    
    def _calculate_queue_score(self, model_id: str) -> float:
        """Calculate queue status score"""
        queue_status = self.queue_manager.get_queue_status().get(model_id, {})
        
        current_size = queue_status.get('current_size', 0)
        processing_count = queue_status.get('processing_count', 0)
        
        # Prefer models with shorter queues
        queue_penalty = (current_size + processing_count) / 20.0  # 20 as baseline
        
        return max(0, 1.0 - queue_penalty)
    
    def _generate_routing_reasoning(self, model_config: ModelConfig, request: LLMRequest) -> List[str]:
        """Generate human-readable reasoning for routing decision"""
        reasoning = []
        
        # Capability match
        capability_score = self.capability_analyzer.score_model_for_task(
            model_config.model_name, request.task_type, request.complexity
        )
        reasoning.append(f"Capability match: {capability_score:.2f} for {request.task_type.value}")
        
        # Model characteristics
        tier = self.model_tiers.get(model_config.model_name, ModelTier.STANDARD)
        reasoning.append(f"Model tier: {tier.value}")
        
        # Resource status
        current_metrics = self.resource_monitor.get_current_metrics()
        if current_metrics:
            if current_metrics.is_high_load():
                reasoning.append("System under high load - considering efficiency")
            else:
                reasoning.append("System resources available")
        
        # Queue status
        queue_status = self.queue_manager.get_queue_status().get(model_config.provider_id, {})
        if queue_status.get('current_size', 0) > 5:
            reasoning.append("Model queue has backlog")
        else:
            reasoning.append("Model queue available")
        
        return reasoning
    
    def _estimate_response_time(self, model_config: ModelConfig, request: LLMRequest) -> float:
        """Estimate response time for model and request"""
        # Base time from historical metrics
        metrics = self.model_metrics[model_config.provider_id]
        base_time = metrics.average_response_time if metrics.total_requests > 0 else 10.0
        
        # Adjust for complexity
        complexity_multipliers = {
            TaskComplexity.SIMPLE: 0.5,
            TaskComplexity.MODERATE: 1.0,
            TaskComplexity.COMPLEX: 1.8,
            TaskComplexity.CRITICAL: 2.5
        }
        
        complexity_multiplier = complexity_multipliers.get(request.complexity, 1.0)
        
        # Adjust for token count
        estimated_tokens = request.max_tokens or model_config.max_tokens
        token_multiplier = estimated_tokens / 1000.0  # Per 1k tokens
        
        # Adjust for current load
        current_metrics = self.resource_monitor.get_current_metrics()
        load_multiplier = 1.0
        if current_metrics and current_metrics.is_high_load():
            load_multiplier = 1.5
        
        estimated_time = base_time * complexity_multiplier * token_multiplier * load_multiplier
        return max(1.0, estimated_time)  # Minimum 1 second
    
    def _estimate_resource_requirements(self, model_config: ModelConfig, 
                                      request: LLMRequest) -> Dict[str, float]:
        """Estimate resource requirements for request"""
        tier = self.model_tiers.get(model_config.model_name, ModelTier.STANDARD)
        
        base_requirements = {
            ModelTier.LIGHTWEIGHT: {"cpu": 20, "memory": 10, "gpu_memory": 15},
            ModelTier.STANDARD: {"cpu": 40, "memory": 25, "gpu_memory": 30},
            ModelTier.HEAVY: {"cpu": 60, "memory": 40, "gpu_memory": 50},
            ModelTier.ENTERPRISE: {"cpu": 80, "memory": 60, "gpu_memory": 70}
        }
        
        requirements = base_requirements[tier].copy()
        
        # Adjust for complexity
        if request.complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]:
            for resource in requirements:
                requirements[resource] *= 1.3
        
        return requirements
    
    def record_model_performance(self, model_id: str, request: LLMRequest, 
                               response: LLMResponse, response_time: float):
        """Record performance metrics for model"""
        metrics = self.model_metrics[model_id]
        
        metrics.total_requests += 1
        metrics.last_used = datetime.now()
        
        if response.success:
            metrics.successful_requests += 1
            metrics.total_response_time += response_time
            
            # Estimate tokens and calculate TPS
            estimated_tokens = len(response.content.split()) * 1.3  # Rough token estimate
            metrics.total_tokens_generated += estimated_tokens
            
            if response_time > 0:
                current_tps = estimated_tokens / response_time
                # Moving average for TPS
                if metrics.average_tokens_per_second == 0:
                    metrics.average_tokens_per_second = current_tps
                else:
                    metrics.average_tokens_per_second = (
                        metrics.average_tokens_per_second * 0.8 + current_tps * 0.2
                    )
            
            # Simple quality score based on response length and structure
            quality_indicators = [
                len(response.content) > 50,  # Minimum length
                any(marker in response.content for marker in ['.', '!', '?']),  # Proper sentences
                '\n' in response.content or len(response.content.split()) > 20,  # Structure
            ]
            current_quality = sum(quality_indicators) / len(quality_indicators)
            
            # Moving average for quality
            if metrics.quality_score == 0:
                metrics.quality_score = current_quality
            else:
                metrics.quality_score = metrics.quality_score * 0.9 + current_quality * 0.1
        else:
            metrics.failed_requests += 1
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics"""
        # Model performance summary
        model_performance = {}
        for model_id, metrics in self.model_metrics.items():
            model_performance[model_id] = {
                'success_rate': metrics.success_rate,
                'average_response_time': metrics.average_response_time,
                'tokens_per_second': metrics.average_tokens_per_second,
                'quality_score': metrics.quality_score,
                'total_requests': metrics.total_requests,
                'last_used': metrics.last_used.isoformat() if metrics.last_used else None
            }
        
        # Resource utilization
        current_resources = self.resource_monitor.get_current_metrics()
        avg_resources = self.resource_monitor.get_average_metrics(30)  # 30 minute average
        
        # Queue analytics
        queue_status = self.queue_manager.get_queue_status()
        
        # Recent routing decisions
        recent_decisions = []
        for entry in list(self.routing_history)[-10:]:  # Last 10 decisions
            recent_decisions.append({
                'timestamp': entry['timestamp'].isoformat(),
                'selected_model': entry['decision']['selected_model'],
                'confidence': entry['decision']['confidence'],
                'reasoning': entry['decision']['reasoning'][:2]  # First 2 reasons
            })
        
        return {
            'model_performance': model_performance,
            'current_resources': asdict(current_resources) if current_resources else None,
            'average_resources': asdict(avg_resources) if avg_resources else None,
            'queue_status': queue_status,
            'recent_decisions': recent_decisions,
            'total_routing_decisions': len(self.routing_history)
        }
    
    def cleanup(self):
        """Cleanup router resources"""
        self.resource_monitor.stop_monitoring()

async def main():
    """Test the model router"""
    print("🧠 Testing Intelligent Model Router")
    print("=" * 50)
    
    # Mock LLM interface for testing
    from api_abstraction import create_default_configs, UnifiedLLMInterface
    
    configs = create_default_configs()
    llm_interface = UnifiedLLMInterface(configs)
    
    # Create router
    router = IntelligentModelRouter(llm_interface)
    
    try:
        # Test routing decisions
        test_requests = [
            LLMRequest(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                task_type=ModelCapability.CHAT,
                complexity=TaskComplexity.SIMPLE
            ),
            LLMRequest(
                messages=[{"role": "user", "content": "Analyze complex data patterns"}],
                task_type=ModelCapability.ANALYSIS,
                complexity=TaskComplexity.COMPLEX
            ),
            LLMRequest(
                messages=[{"role": "user", "content": "Generate Python code for ML pipeline"}],
                task_type=ModelCapability.CODE_GENERATION,
                complexity=TaskComplexity.MODERATE
            )
        ]
        
        print("Testing routing decisions:")
        for i, request in enumerate(test_requests, 1):
            decision = router.select_optimal_model(request)
            print(f"\nRequest {i}: {request.task_type.value} ({request.complexity.value})")
            print(f"  Selected: {decision.selected_model}")
            print(f"  Confidence: {decision.confidence:.2f}")
            print(f"  Reasoning: {', '.join(decision.reasoning[:2])}")
            print(f"  Est. time: {decision.estimated_response_time:.1f}s")
        
        # Test analytics
        print("\nRouting Analytics:")
        analytics = router.get_routing_analytics()
        print(f"  Total decisions: {analytics['total_routing_decisions']}")
        print(f"  Models tracked: {len(analytics['model_performance'])}")
        
        if analytics['current_resources']:
            print(f"  CPU usage: {analytics['current_resources']['cpu_percent']:.1f}%")
            print(f"  Memory usage: {analytics['current_resources']['memory_percent']:.1f}%")
        
        print("\n✅ Model Router test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        router.cleanup()

if __name__ == "__main__":
    asyncio.run(main())