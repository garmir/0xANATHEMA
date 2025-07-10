#!/usr/bin/env python3
"""
Local LLM Adapter
Adapter layer for integrating local LLMs with Task Master AI modules
Replaces external API dependencies while preserving all functionality
"""

import json
import time
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from collections import defaultdict
import queue
import threading

class ModelType(Enum):
    """Types of local models"""
    GENERAL_REASONING = "general_reasoning"
    CODE_GENERATION = "code_generation"
    RESEARCH = "research"
    PLANNING = "planning"
    ANALYSIS = "analysis"

class ProviderType(Enum):
    """Local inference providers"""
    OLLAMA = "ollama"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    TRANSFORMERS = "transformers"

@dataclass
class ModelConfig:
    """Configuration for a local model"""
    model_id: str
    model_name: str
    provider: ProviderType
    model_type: ModelType
    context_length: int
    temperature: float
    max_tokens: int
    timeout_seconds: int
    available: bool = False

@dataclass
class InferenceRequest:
    """Request for local model inference"""
    request_id: str
    model_type: ModelType
    prompt: str
    context: Dict[str, Any]
    max_tokens: int
    temperature: float
    priority: int = 5
    timeout: int = 30

@dataclass
class InferenceResponse:
    """Response from local model inference"""
    request_id: str
    model_id: str
    response_text: str
    success: bool
    execution_time_ms: float
    token_count: int
    error_message: Optional[str]
    metadata: Dict[str, Any]

class LocalLLMAdapter:
    """Adapter for local LLM integration with Task Master AI"""
    
    def __init__(self, config_dir: str = '.taskmaster/ai/local'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration files
        self.models_config_file = self.config_dir / 'models_config.json'
        self.adapter_config_file = self.config_dir / 'adapter_config.json'
        
        # Model configurations (based on our benchmark analysis)
        self.model_configs = {
            "mistral:7b-instruct": ModelConfig(
                model_id="mistral:7b-instruct",
                model_name="Mistral 7B Instruct",
                provider=ProviderType.OLLAMA,
                model_type=ModelType.GENERAL_REASONING,
                context_length=8192,
                temperature=0.2,
                max_tokens=1000,
                timeout_seconds=30
            ),
            "codellama:13b": ModelConfig(
                model_id="codellama:13b",
                model_name="Code Llama 13B",
                provider=ProviderType.OLLAMA,
                model_type=ModelType.CODE_GENERATION,
                context_length=16384,
                temperature=0.1,
                max_tokens=2000,
                timeout_seconds=45
            ),
            "mixtral:8x7b-instruct": ModelConfig(
                model_id="mixtral:8x7b-instruct",
                model_name="Mixtral 8x7B Instruct",
                provider=ProviderType.OLLAMA,
                model_type=ModelType.PLANNING,
                context_length=32768,
                temperature=0.15,
                max_tokens=1500,
                timeout_seconds=60
            ),
            "deepseek-coder:6.7b": ModelConfig(
                model_id="deepseek-coder:6.7b",
                model_name="DeepSeek Coder 6.7B",
                provider=ProviderType.OLLAMA,
                model_type=ModelType.ANALYSIS,
                context_length=16384,
                temperature=0.1,
                max_tokens=1500,
                timeout_seconds=35
            )
        }
        
        # Model routing rules
        self.routing_rules = {
            "research": ModelType.GENERAL_REASONING,
            "planning": ModelType.PLANNING,
            "code_generation": ModelType.CODE_GENERATION,
            "code_analysis": ModelType.ANALYSIS,
            "task_breakdown": ModelType.GENERAL_REASONING,
            "optimization": ModelType.ANALYSIS,
            "reasoning": ModelType.GENERAL_REASONING,
            "synthesis": ModelType.PLANNING
        }
        
        # Request queue and processing
        self.request_queue = queue.PriorityQueue()
        self.active_requests = {}
        self.request_history = []
        
        # Performance tracking
        self.model_performance = defaultdict(list)
        self.total_requests = 0
        self.successful_requests = 0
        
        # Threading for async processing
        self.processing_thread = None
        self.shutdown_event = threading.Event()
        
        self.initialize_adapter()
    
    def initialize_adapter(self):
        """Initialize the local LLM adapter"""
        
        # Check model availability
        self.check_model_availability()
        
        # Load configurations
        self.load_configurations()
        
        # Start processing thread
        self.start_processing_thread()
        
        available_models = [m for m in self.model_configs.values() if m.available]
        print(f"✅ Local LLM Adapter initialized with {len(available_models)} available models")
    
    def check_model_availability(self):
        """Check which models are available locally"""
        
        try:
            # Check Ollama availability
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                available_models = set()
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        available_models.add(model_name)
                
                # Update availability
                for model_id, config in self.model_configs.items():
                    config.available = model_id in available_models
            else:
                print("⚠️ Ollama not available - all models marked as unavailable")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️ Ollama not found - install Ollama for local LLM support")
    
    def get_model_for_task(self, task_type: str, fallback: bool = True) -> Optional[ModelConfig]:
        """Get the best model for a specific task type"""
        
        # Determine model type
        model_type = self.routing_rules.get(task_type, ModelType.GENERAL_REASONING)
        
        # Find available models of the required type
        candidates = [
            config for config in self.model_configs.values()
            if config.model_type == model_type and config.available
        ]
        
        if candidates:
            # Return the first available model of the correct type
            return candidates[0]
        
        if fallback:
            # Fallback to any available model
            fallback_candidates = [
                config for config in self.model_configs.values()
                if config.available
            ]
            if fallback_candidates:
                return fallback_candidates[0]
        
        return None
    
    def submit_request(self, task_type: str, prompt: str, context: Dict[str, Any] = None,
                      max_tokens: int = None, temperature: float = None,
                      priority: int = 5, timeout: int = None) -> str:
        """Submit an inference request to the queue"""
        
        request_id = f"req_{int(time.time() * 1000)}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
        
        # Get model type for task
        model_type = self.routing_rules.get(task_type, ModelType.GENERAL_REASONING)
        
        # Get model config for defaults
        model_config = self.get_model_for_task(task_type)
        if not model_config:
            raise Exception(f"No available models for task type: {task_type}")
        
        # Use provided values or defaults from model config
        max_tokens = max_tokens or model_config.max_tokens
        temperature = temperature if temperature is not None else model_config.temperature
        timeout = timeout or model_config.timeout_seconds
        
        # Create request
        request = InferenceRequest(
            request_id=request_id,
            model_type=model_type,
            prompt=prompt,
            context=context or {},
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            timeout=timeout
        )
        
        # Add to queue (priority queue uses tuple comparison)
        self.request_queue.put((priority, time.time(), request))
        self.active_requests[request_id] = request
        
        return request_id
    
    def get_response(self, request_id: str, timeout: float = 30.0) -> Optional[InferenceResponse]:
        """Get response for a request (blocking)"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if response is ready
            for response in self.request_history:
                if response.request_id == request_id:
                    return response
            
            time.sleep(0.1)
        
        return None  # Timeout
    
    def process_request_sync(self, task_type: str, prompt: str, context: Dict[str, Any] = None,
                           max_tokens: int = None, temperature: float = None,
                           timeout: float = 30.0) -> Optional[InferenceResponse]:
        """Process a request synchronously"""
        
        request_id = self.submit_request(task_type, prompt, context, max_tokens, temperature)
        return self.get_response(request_id, timeout)
    
    def start_processing_thread(self):
        """Start the background processing thread"""
        
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.processing_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.processing_thread.start()
    
    def _process_requests(self):
        """Background thread to process inference requests"""
        
        while not self.shutdown_event.is_set():
            try:
                # Get request from queue (timeout to check shutdown)
                try:
                    priority, timestamp, request = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the request
                response = self._execute_inference(request)
                
                # Store response
                self.request_history.append(response)
                
                # Clean up
                if request.request_id in self.active_requests:
                    del self.active_requests[request.request_id]
                
                # Update statistics
                self.total_requests += 1
                if response.success:
                    self.successful_requests += 1
                
                # Track performance
                self.model_performance[response.model_id].append({
                    'execution_time': response.execution_time_ms,
                    'success': response.success,
                    'timestamp': datetime.now()
                })
                
                # Mark queue task done
                self.request_queue.task_done()
                
            except Exception as e:
                print(f"Error processing request: {e}")
    
    def _execute_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Execute inference for a request"""
        
        start_time = time.time()
        
        # Get model for request
        model_config = self.get_model_for_task("", fallback=True)  # Get any available model
        if not model_config:
            return InferenceResponse(
                request_id=request.request_id,
                model_id="none",
                response_text="",
                success=False,
                execution_time_ms=0,
                token_count=0,
                error_message="No available models",
                metadata={}
            )
        
        # Filter models by type
        type_models = [
            config for config in self.model_configs.values()
            if config.model_type == request.model_type and config.available
        ]
        
        if type_models:
            model_config = type_models[0]
        
        try:
            # Execute using Ollama
            if model_config.provider == ProviderType.OLLAMA:
                response_text, success, error_msg = self._execute_ollama(
                    model_config.model_id, request.prompt, request.max_tokens, request.timeout
                )
            else:
                response_text = ""
                success = False
                error_msg = f"Provider {model_config.provider.value} not implemented"
            
            execution_time = (time.time() - start_time) * 1000
            
            return InferenceResponse(
                request_id=request.request_id,
                model_id=model_config.model_id,
                response_text=response_text,
                success=success,
                execution_time_ms=execution_time,
                token_count=len(response_text.split()) if response_text else 0,
                error_message=error_msg,
                metadata={
                    'model_type': model_config.model_type.value,
                    'provider': model_config.provider.value,
                    'context_length': model_config.context_length
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                model_id=model_config.model_id,
                response_text="",
                success=False,
                execution_time_ms=execution_time,
                token_count=0,
                error_message=str(e),
                metadata={}
            )
    
    def _execute_ollama(self, model_id: str, prompt: str, max_tokens: int, timeout: int) -> tuple[str, bool, Optional[str]]:
        """Execute inference using Ollama"""
        
        try:
            # Prepare Ollama command
            cmd = ['ollama', 'generate', model_id, prompt]
            
            # Execute
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                response_text = result.stdout.strip()
                # Truncate if necessary
                if max_tokens and len(response_text.split()) > max_tokens:
                    words = response_text.split()[:max_tokens]
                    response_text = ' '.join(words)
                
                return response_text, True, None
            else:
                return "", False, result.stderr
                
        except subprocess.TimeoutExpired:
            return "", False, f"Timeout after {timeout} seconds"
        except Exception as e:
            return "", False, str(e)
    
    # External API Compatibility Layer
    def research_query(self, prompt: str, context: Dict[str, Any] = None, 
                      detail_level: str = "medium") -> str:
        """Compatible interface for research queries (replaces Perplexity)"""
        
        # Enhance prompt for research context
        research_prompt = f"""You are a research assistant helping with project analysis. 
        Provide a {detail_level} level response to this research query:
        
        {prompt}
        
        Include relevant details, analysis, and actionable insights."""
        
        response = self.process_request_sync("research", research_prompt, context, timeout=45.0)
        
        if response and response.success:
            return response.response_text
        else:
            error_msg = response.error_message if response else "Request timeout"
            return f"Research query failed: {error_msg}"
    
    def planning_request(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Compatible interface for planning requests (replaces Claude planning)"""
        
        planning_prompt = f"""You are an expert project planner and architect. 
        Provide a comprehensive, structured response to this planning request:
        
        {prompt}
        
        Include step-by-step breakdown, dependencies, and implementation considerations."""
        
        response = self.process_request_sync("planning", planning_prompt, context, timeout=60.0)
        
        if response and response.success:
            return response.response_text
        else:
            error_msg = response.error_message if response else "Request timeout"
            return f"Planning request failed: {error_msg}"
    
    def code_generation_request(self, prompt: str, language: str = "python", 
                              context: Dict[str, Any] = None) -> str:
        """Compatible interface for code generation (replaces OpenAI Codex)"""
        
        code_prompt = f"""You are an expert {language} developer. 
        Generate high-quality, well-documented code for this request:
        
        {prompt}
        
        Include proper error handling, type hints, and comprehensive docstrings."""
        
        response = self.process_request_sync("code_generation", code_prompt, context, 
                                           max_tokens=2000, timeout=45.0)
        
        if response and response.success:
            return response.response_text
        else:
            error_msg = response.error_message if response else "Request timeout"
            return f"Code generation failed: {error_msg}"
    
    def reasoning_request(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Compatible interface for reasoning tasks (replaces Claude reasoning)"""
        
        reasoning_prompt = f"""You are an expert analyst with strong reasoning capabilities.
        Provide a logical, step-by-step analysis for this request:
        
        {prompt}
        
        Show your reasoning process and justify your conclusions."""
        
        response = self.process_request_sync("reasoning", reasoning_prompt, context, timeout=30.0)
        
        if response and response.success:
            return response.response_text
        else:
            error_msg = response.error_message if response else "Request timeout"
            return f"Reasoning request failed: {error_msg}"
    
    def load_configurations(self):
        """Load adapter configurations from disk"""
        try:
            if self.adapter_config_file.exists():
                with open(self.adapter_config_file, 'r') as f:
                    config = json.load(f)
                
                # Load routing rules
                if 'routing_rules' in config:
                    self.routing_rules.update(config['routing_rules'])
                    
        except Exception as e:
            print(f"⚠️ Failed to load configurations: {e}")
    
    def save_configurations(self):
        """Save adapter configurations to disk"""
        try:
            config = {
                'routing_rules': self.routing_rules,
                'model_configs': {
                    model_id: asdict(config) for model_id, config in self.model_configs.items()
                }
            }
            
            # Convert enums to strings
            for model_data in config['model_configs'].values():
                model_data['provider'] = model_data['provider'].value if hasattr(model_data['provider'], 'value') else model_data['provider']
                model_data['model_type'] = model_data['model_type'].value if hasattr(model_data['model_type'], 'value') else model_data['model_type']
            
            with open(self.adapter_config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save configurations: {e}")
    
    def get_adapter_status(self) -> Dict[str, Any]:
        """Get current adapter status and performance metrics"""
        
        available_models = [m for m in self.model_configs.values() if m.available]
        success_rate = (self.successful_requests / self.total_requests) if self.total_requests > 0 else 0
        
        return {
            'available_models': len(available_models),
            'total_models': len(self.model_configs),
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'success_rate': success_rate,
            'active_requests': len(self.active_requests),
            'request_queue_size': self.request_queue.qsize(),
            'model_types': list(set(m.model_type.value for m in available_models))
        }
    
    def shutdown(self):
        """Shutdown the adapter gracefully"""
        
        self.shutdown_event.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Save configurations
        self.save_configurations()
        
        print("🔄 Local LLM Adapter shutdown complete")

def main():
    """Demo of Local LLM Adapter"""
    print("Local LLM Adapter Demo")
    print("=" * 30)
    
    adapter = LocalLLMAdapter()
    
    # Show adapter status
    status = adapter.get_adapter_status()
    print(f"\n📊 Adapter Status:")
    print(f"  Available models: {status['available_models']}/{status['total_models']}")
    print(f"  Model types: {', '.join(status['model_types'])}")
    
    if status['available_models'] > 0:
        # Demo requests
        print(f"\n🧪 Testing adapter interfaces...")
        
        # Test research query
        print(f"\n🔍 Research Query Test:")
        research_result = adapter.research_query(
            "What are the best practices for implementing recursive task breakdown in AI systems?"
        )
        print(f"Result: {research_result[:200]}...")
        
        # Test planning request
        print(f"\n📋 Planning Request Test:")
        planning_result = adapter.planning_request(
            "Create a migration plan for transitioning from external APIs to local LLMs"
        )
        print(f"Result: {planning_result[:200]}...")
        
        # Test code generation
        print(f"\n💻 Code Generation Test:")
        code_result = adapter.code_generation_request(
            "Create a Python function that implements exponential backoff for retry logic"
        )
        print(f"Result: {code_result[:200]}...")
        
        # Show final status
        final_status = adapter.get_adapter_status()
        print(f"\n📈 Final Status:")
        print(f"  Total requests: {final_status['total_requests']}")
        print(f"  Success rate: {final_status['success_rate']:.1%}")
        
    else:
        print(f"\n⚠️ No models available - install Ollama and pull models for testing")
        print(f"   Example: ollama pull mistral:7b-instruct")
    
    # Shutdown
    adapter.shutdown()
    
    print(f"\n✅ Local LLM Adapter demo completed")

if __name__ == "__main__":
    main()