#!/usr/bin/env python3
"""
Local LLM Adapter for Task Master AI
Unified interface for local LLM inference replacing external API dependencies
"""

import os
import json
import subprocess
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
import time

# Conditional import for requests - only used if available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests module not available - local LLM providers will use fallback mode")

@dataclass
class ModelConfig:
    """Configuration for local LLM models"""
    name: str
    provider: str  # ollama, localai, text-generation-webui, lm-studio
    endpoint: str
    model_id: str
    max_tokens: int = 2000
    temperature: float = 0.2
    top_p: float = 0.9
    context_window: int = 4096

class LocalLLMInterface(ABC):
    """Abstract base class for local LLM providers"""
    
    @abstractmethod
    def inference(self, messages: List[Dict], **kwargs) -> Dict:
        """Perform inference with the local model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass
    
    @abstractmethod
    def get_models(self) -> List[str]:
        """Get list of available models"""
        pass

class OllamaProvider(LocalLLMInterface):
    """Ollama local LLM provider"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        
    def inference(self, messages: List[Dict], model_id: str = "llama2", **kwargs) -> Dict:
        """Perform inference using Ollama API"""
        if not REQUESTS_AVAILABLE:
            return {"error": "requests module not available - cannot connect to Ollama"}
        
        try:
            # Convert messages to Ollama format
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "model": model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.2),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_predict": kwargs.get("max_tokens", 2000)
                }
            }
            
            response = requests.post(f"{self.api_url}/generate", json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            return {
                "choices": [{
                    "message": {
                        "content": result.get("response", ""),
                        "role": "assistant"
                    }
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(result.get("response", "").split()),
                    "total_tokens": len(prompt.split()) + len(result.get("response", "").split())
                }
            }
            
        except Exception as e:
            return {"error": f"Ollama inference failed: {str(e)}"}
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert OpenAI-style messages to single prompt"""
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
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        if not REQUESTS_AVAILABLE:
            return False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_models(self) -> List[str]:
        """Get available Ollama models"""
        if not REQUESTS_AVAILABLE:
            return []
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except:
            pass
        return []

class LocalAIProvider(LocalLLMInterface):
    """LocalAI provider - OpenAI-compatible API"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
    
    def inference(self, messages: List[Dict], model_id: str = "gpt-3.5-turbo", **kwargs) -> Dict:
        """Perform inference using LocalAI OpenAI-compatible API"""
        if not REQUESTS_AVAILABLE:
            return {"error": "requests module not available - cannot connect to LocalAI"}
        
        try:
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 2000),
                "temperature": kwargs.get("temperature", 0.2),
                "top_p": kwargs.get("top_p", 0.9),
                "stream": False
            }
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(f"{self.base_url}/v1/chat/completions", 
                                   json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return {"error": f"LocalAI inference failed: {str(e)}"}
    
    def is_available(self) -> bool:
        """Check if LocalAI is running"""
        if not REQUESTS_AVAILABLE:
            return False
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_models(self) -> List[str]:
        """Get available LocalAI models"""
        if not REQUESTS_AVAILABLE:
            return []
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]
        except:
            pass
        return []

class TextGenWebuiProvider(LocalLLMInterface):
    """Text-generation-webui provider"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
    
    def inference(self, messages: List[Dict], model_id: str = None, **kwargs) -> Dict:
        """Perform inference using text-generation-webui API"""
        if not REQUESTS_AVAILABLE:
            return {"error": "requests module not available - cannot connect to text-generation-webui"}
        
        try:
            # Convert messages to single prompt
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "prompt": prompt,
                "max_new_tokens": kwargs.get("max_tokens", 2000),
                "temperature": kwargs.get("temperature", 0.2),
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": True,
                "stopping_strings": ["Human:", "User:"]
            }
            
            response = requests.post(f"{self.base_url}/api/v1/generate", 
                                   json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("results", [{}])[0].get("text", "")
            
            return {
                "choices": [{
                    "message": {
                        "content": generated_text,
                        "role": "assistant"
                    }
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split())
                }
            }
            
        except Exception as e:
            return {"error": f"Text-generation-webui inference failed: {str(e)}"}
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"{content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"
    
    def is_available(self) -> bool:
        """Check if text-generation-webui is running"""
        if not REQUESTS_AVAILABLE:
            return False
        try:
            response = requests.get(f"{self.base_url}/api/v1/model", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_models(self) -> List[str]:
        """Get current loaded model"""
        if not REQUESTS_AVAILABLE:
            return []
        try:
            response = requests.get(f"{self.base_url}/api/v1/model", timeout=5)
            if response.status_code == 200:
                data = response.json()
                model_name = data.get("result", "")
                return [model_name] if model_name else []
        except:
            pass
        return []

class LocalLLMAdapter:
    """Unified adapter for local LLM providers"""
    
    def __init__(self, config_path: str = ".taskmaster/config.json"):
        self.config_path = config_path
        self.providers = {}
        self.current_config = None
        self.logger = self._setup_logging()
        
        # Initialize providers
        self._initialize_providers()
        self._load_config()
    
    def _setup_logging(self):
        """Setup logging"""
        logger = logging.getLogger("local_llm_adapter")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_providers(self):
        """Initialize all local LLM providers"""
        self.providers = {
            "ollama": OllamaProvider(),
            "localai": LocalAIProvider(),
            "text-generation-webui": TextGenWebuiProvider()
        }
    
    def _load_config(self):
        """Load configuration from Task Master config file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.current_config = json.load(f)
            else:
                self.current_config = self._create_default_config()
                self._save_config()
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.current_config = self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """Create default configuration with local LLMs"""
        return {
            "models": {
                "main": {
                    "provider": "ollama",
                    "modelId": "llama2",
                    "maxTokens": 2000,
                    "temperature": 0.2
                },
                "research": {
                    "provider": "ollama", 
                    "modelId": "llama2:13b",
                    "maxTokens": 4000,
                    "temperature": 0.1
                },
                "fallback": {
                    "provider": "localai",
                    "modelId": "gpt-3.5-turbo",
                    "maxTokens": 2000,
                    "temperature": 0.2
                }
            },
            "local_llm": {
                "providers": {
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "enabled": True
                    },
                    "localai": {
                        "base_url": "http://localhost:8080", 
                        "enabled": True
                    },
                    "text-generation-webui": {
                        "base_url": "http://localhost:5000",
                        "enabled": True
                    }
                }
            }
        }
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.current_config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Check which providers are available"""
        availability = {}
        for name, provider in self.providers.items():
            availability[name] = provider.is_available()
        return availability
    
    def get_provider_models(self, provider_name: str) -> List[str]:
        """Get available models for a provider"""
        if provider_name in self.providers:
            return self.providers[provider_name].get_models()
        return []
    
    def inference(self, messages: List[Dict], role: str = "main", **kwargs) -> Dict:
        """Perform inference using configured model for role"""
        
        # Get model config for role
        model_config = self.current_config.get("models", {}).get(role, {})
        if not model_config:
            return {"error": f"No model configured for role: {role}"}
        
        provider_name = model_config.get("provider", "ollama")
        model_id = model_config.get("modelId", "llama2")
        
        # Get provider
        provider = self.providers.get(provider_name)
        if not provider:
            return {"error": f"Provider not available: {provider_name}"}
        
        # Check if provider is running
        if not provider.is_available():
            # Try fallback
            fallback_config = self.current_config.get("models", {}).get("fallback", {})
            fallback_provider_name = fallback_config.get("provider", "localai")
            fallback_provider = self.providers.get(fallback_provider_name)
            
            if fallback_provider and fallback_provider.is_available():
                self.logger.warning(f"Primary provider {provider_name} unavailable, using fallback {fallback_provider_name}")
                provider = fallback_provider
                model_id = fallback_config.get("modelId", "gpt-3.5-turbo")
            else:
                return {"error": f"Provider {provider_name} not available and no fallback"}
        
        # Merge model config with kwargs
        inference_params = {
            "model_id": model_id,
            "max_tokens": model_config.get("maxTokens", 2000),
            "temperature": model_config.get("temperature", 0.2),
            **kwargs
        }
        
        # Perform inference
        start_time = time.time()
        result = provider.inference(messages, **inference_params)
        end_time = time.time()
        
        # Add timing metadata
        if "error" not in result:
            result["metadata"] = {
                "provider": provider_name,
                "model": model_id,
                "inference_time": end_time - start_time,
                "timestamp": datetime.now().isoformat()
            }
        
        return result
    
    def research(self, query: str, **kwargs) -> Dict:
        """Research method compatible with Perplexity client interface"""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful research assistant providing accurate, detailed information and analysis."
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        return self.inference(messages, role="research", **kwargs)
    
    def analyze_code_patterns(self, code_snippet: str) -> Dict:
        """Analyze code patterns using local LLM"""
        query = f"""
        Analyze this code snippet for patterns, potential issues, and improvements:
        
        ```
        {code_snippet}
        ```
        
        Provide analysis on:
        1. Code quality and patterns
        2. Potential security issues
        3. Performance considerations
        4. Best practices recommendations
        """
        
        return self.research(query)
    
    def feasibility_assessment(self, project_description: str) -> Dict:
        """Assess implementation feasibility using local LLM"""
        query = f"""
        Assess the technical feasibility of this project:
        
        {project_description}
        
        Provide analysis on:
        1. Technical complexity
        2. Resource requirements
        3. Potential challenges
        4. Implementation timeline
        5. Required technologies
        """
        
        return self.research(query)
    
    def update_model_config(self, role: str, provider: str, model_id: str, **params):
        """Update model configuration for a role"""
        if "models" not in self.current_config:
            self.current_config["models"] = {}
        
        self.current_config["models"][role] = {
            "provider": provider,
            "modelId": model_id,
            **params
        }
        
        self._save_config()
        self.logger.info(f"Updated {role} model to {provider}/{model_id}")

def main():
    """Test the local LLM adapter"""
    print("🤖 Testing Local LLM Adapter")
    print("=" * 50)
    
    adapter = LocalLLMAdapter()
    
    # Check provider availability
    availability = adapter.get_available_providers()
    print("Provider Availability:")
    for provider, available in availability.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"  {provider}: {status}")
    
    # List models for available providers
    print("\nAvailable Models:")
    for provider, available in availability.items():
        if available:
            models = adapter.get_provider_models(provider)
            print(f"  {provider}: {models}")
    
    # Test inference if any provider available
    if any(availability.values()):
        print("\nTesting inference...")
        test_query = "What are the key principles of good software architecture?"
        result = adapter.research(test_query)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ Response ({len(content)} chars): {content[:200]}...")
    else:
        print("\n❌ No providers available for testing")

if __name__ == "__main__":
    main()