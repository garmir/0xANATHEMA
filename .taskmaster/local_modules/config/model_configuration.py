#!/usr/bin/env python3
"""
Model Configuration System for Task Master AI
Manages configuration for local vs external models with intelligent switching
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
import logging
from enum import Enum

from ..core.api_abstraction import ModelProvider, TaskType, ModelConfig, UnifiedModelAPI, ModelConfigFactory

logger = logging.getLogger(__name__)

class DeploymentMode(Enum):
    """Deployment modes for the system"""
    LOCAL_ONLY = "local_only"
    EXTERNAL_ONLY = "external_only"
    HYBRID = "hybrid"
    LOCAL_PREFERRED = "local_preferred"
    EXTERNAL_PREFERRED = "external_preferred"

class ModelTier(Enum):
    """Model performance tiers"""
    FAST = "fast"          # Quick responses, lower accuracy
    BALANCED = "balanced"   # Good balance of speed and accuracy
    ACCURATE = "accurate"   # High accuracy, slower responses
    SPECIALIZED = "specialized"  # Domain-specific models

@dataclass
class ModelPerformanceProfile:
    """Performance profile for a model"""
    avg_response_time: float = 0.0
    accuracy_score: float = 0.0
    reliability_score: float = 1.0
    cost_per_request: float = 0.0
    tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    success_rate: float = 1.0
    last_updated: float = field(default_factory=time.time)
    request_count: int = 0
    
    def update_performance(self, response_time: float, success: bool, tokens: int = 0):
        """Update performance metrics"""
        self.request_count += 1
        
        # Update response time (weighted average)
        if self.request_count == 1:
            self.avg_response_time = response_time
        else:
            weight = min(0.1, 1.0 / self.request_count)  # Diminishing weight for new samples
            self.avg_response_time = (1 - weight) * self.avg_response_time + weight * response_time
        
        # Update success rate
        self.success_rate = (self.success_rate * (self.request_count - 1) + (1.0 if success else 0.0)) / self.request_count
        
        # Update tokens per second
        if tokens > 0 and response_time > 0:
            current_tps = tokens / response_time
            if self.tokens_per_second == 0:
                self.tokens_per_second = current_tps
            else:
                weight = min(0.1, 1.0 / self.request_count)
                self.tokens_per_second = (1 - weight) * self.tokens_per_second + weight * current_tps
        
        self.last_updated = time.time()

@dataclass
class ModelConfigurationProfile:
    """Complete configuration profile for a model"""
    model_id: str
    model_config: ModelConfig
    performance_profile: ModelPerformanceProfile
    tier: ModelTier
    enabled: bool = True
    fallback_priority: int = 1
    task_preferences: Dict[TaskType, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    last_health_check: float = 0.0
    health_status: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        config_dict = asdict(self.model_config)
        config_dict["provider"] = self.model_config.provider.value
        config_dict["capabilities"] = [cap.value for cap in self.model_config.capabilities]
        
        task_prefs = {task.value: pref for task, pref in self.task_preferences.items()}
        
        return {
            "model_id": self.model_id,
            "model_config": config_dict,
            "performance_profile": asdict(self.performance_profile),
            "tier": self.tier.value,
            "enabled": self.enabled,
            "fallback_priority": self.fallback_priority,
            "task_preferences": task_prefs,
            "constraints": self.constraints,
            "last_health_check": self.last_health_check,
            "health_status": self.health_status
        }

class ModelConfigurationManager:
    """
    Manages model configurations and intelligent switching between local and external models
    """
    
    def __init__(self, config_dir: str = ".taskmaster/local_modules/config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration files
        self.config_file = self.config_dir / "model_configurations.json"
        self.deployment_config_file = self.config_dir / "deployment_config.json"
        self.performance_file = self.config_dir / "model_performance.json"
        
        # Model configurations
        self.model_profiles: Dict[str, ModelConfigurationProfile] = {}
        self.deployment_mode = DeploymentMode.HYBRID
        self.deployment_config = {}
        
        # Load configurations
        self._load_configurations()
        
        # Initialize default configurations if none exist
        if not self.model_profiles:
            self._initialize_default_configurations()
    
    def _load_configurations(self):
        """Load existing configurations"""
        # Load model profiles
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    profiles_data = json.load(f)
                    self._deserialize_profiles(profiles_data)
                logger.info(f"Loaded {len(self.model_profiles)} model configurations")
            except Exception as e:
                logger.error(f"Failed to load model configurations: {e}")
        
        # Load deployment configuration
        if self.deployment_config_file.exists():
            try:
                with open(self.deployment_config_file, 'r') as f:
                    deploy_data = json.load(f)
                    self.deployment_mode = DeploymentMode(deploy_data.get("deployment_mode", "hybrid"))
                    self.deployment_config = deploy_data.get("config", {})
                logger.info(f"Loaded deployment configuration: {self.deployment_mode.value}")
            except Exception as e:
                logger.error(f"Failed to load deployment configuration: {e}")
    
    def _deserialize_profiles(self, profiles_data: Dict[str, Any]):
        """Deserialize model profiles from JSON data"""
        for model_id, profile_data in profiles_data.items():
            try:
                # Reconstruct ModelConfig
                config_data = profile_data["model_config"]
                model_config = ModelConfig(
                    provider=ModelProvider(config_data["provider"]),
                    model_name=config_data["model_name"],
                    endpoint=config_data["endpoint"],
                    api_key=config_data.get("api_key"),
                    max_tokens=config_data.get("max_tokens", 4000),
                    temperature=config_data.get("temperature", 0.7),
                    timeout=config_data.get("timeout", 120),
                    priority=config_data.get("priority", 1),
                    capabilities=[TaskType(cap) for cap in config_data.get("capabilities", [])]
                )
                
                # Reconstruct PerformanceProfile
                perf_data = profile_data["performance_profile"]
                performance_profile = ModelPerformanceProfile(**perf_data)
                
                # Reconstruct task preferences
                task_prefs = {}
                for task_str, pref in profile_data.get("task_preferences", {}).items():
                    task_prefs[TaskType(task_str)] = pref
                
                # Create profile
                profile = ModelConfigurationProfile(
                    model_id=model_id,
                    model_config=model_config,
                    performance_profile=performance_profile,
                    tier=ModelTier(profile_data.get("tier", "balanced")),
                    enabled=profile_data.get("enabled", True),
                    fallback_priority=profile_data.get("fallback_priority", 1),
                    task_preferences=task_prefs,
                    constraints=profile_data.get("constraints", {}),
                    last_health_check=profile_data.get("last_health_check", 0.0),
                    health_status=profile_data.get("health_status", True)
                )
                
                self.model_profiles[model_id] = profile
                
            except Exception as e:
                logger.error(f"Failed to deserialize profile {model_id}: {e}")
    
    def _initialize_default_configurations(self):
        """Initialize default model configurations"""
        logger.info("Initializing default model configurations")
        
        # Local models
        local_models = [
            {
                "model_id": "ollama_llama2",
                "config": ModelConfigFactory.create_ollama_config(
                    "llama2", capabilities=[TaskType.GENERAL, TaskType.ANALYSIS]
                ),
                "tier": ModelTier.BALANCED
            },
            {
                "model_id": "ollama_mistral",
                "config": ModelConfigFactory.create_ollama_config(
                    "mistral", capabilities=[TaskType.RESEARCH, TaskType.ANALYSIS]
                ),
                "tier": ModelTier.ACCURATE
            },
            {
                "model_id": "ollama_codellama",
                "config": ModelConfigFactory.create_ollama_config(
                    "codellama", capabilities=[TaskType.CODE_GENERATION, TaskType.ANALYSIS]
                ),
                "tier": ModelTier.SPECIALIZED
            },
            {
                "model_id": "lm_studio_general",
                "config": ModelConfigFactory.create_lm_studio_config(
                    "mistral-7b", capabilities=[TaskType.GENERAL, TaskType.CODE_GENERATION]
                ),
                "tier": ModelTier.FAST
            }
        ]
        
        # External models (if API keys available)
        external_models = []
        
        if os.getenv("ANTHROPIC_API_KEY"):
            external_models.append({
                "model_id": "claude_sonnet",
                "config": ModelConfig(
                    provider=ModelProvider.ANTHROPIC,
                    model_name="claude-3-5-sonnet-20241022",
                    endpoint="https://api.anthropic.com",
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    capabilities=[TaskType.GENERAL, TaskType.ANALYSIS, TaskType.RESEARCH, TaskType.CODE_GENERATION]
                ),
                "tier": ModelTier.ACCURATE
            })
        
        if os.getenv("OPENAI_API_KEY"):
            external_models.append({
                "model_id": "gpt4_turbo",
                "config": ModelConfig(
                    provider=ModelProvider.OPENAI,
                    model_name="gpt-4-turbo-preview",
                    endpoint="https://api.openai.com",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    capabilities=[TaskType.GENERAL, TaskType.ANALYSIS, TaskType.CODE_GENERATION]
                ),
                "tier": ModelTier.ACCURATE
            })
        
        if os.getenv("PERPLEXITY_API_KEY"):
            external_models.append({
                "model_id": "perplexity_research",
                "config": ModelConfig(
                    provider=ModelProvider.PERPLEXITY,
                    model_name="llama-3.1-sonar-large-128k-online",
                    endpoint="https://api.perplexity.ai",
                    api_key=os.getenv("PERPLEXITY_API_KEY"),
                    capabilities=[TaskType.RESEARCH]
                ),
                "tier": ModelTier.SPECIALIZED
            })
        
        # Create profiles for all models
        all_models = local_models + external_models
        
        for model_data in all_models:
            self.add_model_configuration(
                model_id=model_data["model_id"],
                model_config=model_data["config"],
                tier=model_data["tier"]
            )
        
        # Set deployment mode based on available models
        if local_models and external_models:
            self.deployment_mode = DeploymentMode.HYBRID
        elif local_models:
            self.deployment_mode = DeploymentMode.LOCAL_ONLY
        elif external_models:
            self.deployment_mode = DeploymentMode.EXTERNAL_ONLY
        
        # Save initial configuration
        self.save_configurations()
    
    def add_model_configuration(self,
                               model_id: str,
                               model_config: ModelConfig,
                               tier: ModelTier = ModelTier.BALANCED,
                               task_preferences: Dict[TaskType, float] = None) -> bool:
        """Add a new model configuration"""
        try:
            # Create performance profile
            performance_profile = ModelPerformanceProfile()
            
            # Set default task preferences based on capabilities
            if task_preferences is None:
                task_preferences = {}
                for capability in model_config.capabilities:
                    task_preferences[capability] = 1.0
            
            # Create configuration profile
            profile = ModelConfigurationProfile(
                model_id=model_id,
                model_config=model_config,
                performance_profile=performance_profile,
                tier=tier,
                task_preferences=task_preferences,
                fallback_priority=model_config.priority
            )
            
            self.model_profiles[model_id] = profile
            logger.info(f"Added model configuration: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model configuration {model_id}: {e}")
            return False
    
    def remove_model_configuration(self, model_id: str) -> bool:
        """Remove a model configuration"""
        if model_id in self.model_profiles:
            del self.model_profiles[model_id]
            logger.info(f"Removed model configuration: {model_id}")
            return True
        return False
    
    def update_model_performance(self, model_id: str, response_time: float, success: bool, tokens: int = 0):
        """Update performance metrics for a model"""
        if model_id in self.model_profiles:
            self.model_profiles[model_id].performance_profile.update_performance(response_time, success, tokens)
    
    def set_deployment_mode(self, mode: DeploymentMode, config: Dict[str, Any] = None):
        """Set deployment mode"""
        self.deployment_mode = mode
        if config:
            self.deployment_config.update(config)
        
        logger.info(f"Deployment mode set to: {mode.value}")
        self.save_configurations()
    
    def get_optimal_model(self, 
                         task_type: TaskType,
                         tier_preference: ModelTier = None,
                         constraints: Dict[str, Any] = None) -> Optional[str]:
        """Get optimal model for a task"""
        available_models = self._filter_available_models(task_type, tier_preference, constraints)
        
        if not available_models:
            return None
        
        # Score models based on multiple factors
        scored_models = []
        for model_id, profile in available_models.items():
            score = self._calculate_model_score(profile, task_type)
            scored_models.append((score, model_id, profile))
        
        # Sort by score (highest first)
        scored_models.sort(reverse=True)
        
        # Apply deployment mode preferences
        best_model = self._apply_deployment_preferences(scored_models)
        
        return best_model
    
    def _filter_available_models(self, 
                                task_type: TaskType,
                                tier_preference: ModelTier = None,
                                constraints: Dict[str, Any] = None) -> Dict[str, ModelConfigurationProfile]:
        """Filter models based on criteria"""
        available = {}
        
        for model_id, profile in self.model_profiles.items():
            # Check if model is enabled and healthy
            if not profile.enabled or not profile.health_status:
                continue
            
            # Check if model supports the task type
            if task_type not in profile.model_config.capabilities:
                continue
            
            # Check tier preference
            if tier_preference and profile.tier != tier_preference:
                continue
            
            # Check constraints
            if constraints:
                if not self._check_constraints(profile, constraints):
                    continue
            
            available[model_id] = profile
        
        return available
    
    def _check_constraints(self, profile: ModelConfigurationProfile, constraints: Dict[str, Any]) -> bool:
        """Check if model meets constraints"""
        perf = profile.performance_profile
        
        # Response time constraint
        if "max_response_time" in constraints:
            if perf.avg_response_time > constraints["max_response_time"]:
                return False
        
        # Cost constraint
        if "max_cost_per_request" in constraints:
            if perf.cost_per_request > constraints["max_cost_per_request"]:
                return False
        
        # Provider constraint
        if "allowed_providers" in constraints:
            if profile.model_config.provider not in constraints["allowed_providers"]:
                return False
        
        # Memory usage constraint
        if "max_memory_mb" in constraints:
            if perf.memory_usage_mb > constraints["max_memory_mb"]:
                return False
        
        return True
    
    def _calculate_model_score(self, profile: ModelConfigurationProfile, task_type: TaskType) -> float:
        """Calculate overall score for a model"""
        perf = profile.performance_profile
        
        # Base score from task preference
        base_score = profile.task_preferences.get(task_type, 0.5)
        
        # Performance factors
        reliability_factor = perf.success_rate
        speed_factor = 1.0 / (1.0 + perf.avg_response_time / 10.0)  # Normalize around 10s
        
        # Tier bonuses
        tier_bonus = {
            ModelTier.FAST: 0.8,
            ModelTier.BALANCED: 1.0,
            ModelTier.ACCURATE: 1.2,
            ModelTier.SPECIALIZED: 1.1
        }.get(profile.tier, 1.0)
        
        # Calculate final score
        score = base_score * reliability_factor * speed_factor * tier_bonus
        
        # Apply fallback priority
        score *= (profile.fallback_priority / 10.0)
        
        return score
    
    def _apply_deployment_preferences(self, scored_models: List[Tuple[float, str, ModelConfigurationProfile]]) -> Optional[str]:
        """Apply deployment mode preferences to model selection"""
        if not scored_models:
            return None
        
        local_models = [(score, model_id, profile) for score, model_id, profile in scored_models 
                       if profile.model_config.provider in [ModelProvider.OLLAMA, ModelProvider.LM_STUDIO, ModelProvider.LOCAL_AI]]
        external_models = [(score, model_id, profile) for score, model_id, profile in scored_models 
                          if profile.model_config.provider not in [ModelProvider.OLLAMA, ModelProvider.LM_STUDIO, ModelProvider.LOCAL_AI]]
        
        if self.deployment_mode == DeploymentMode.LOCAL_ONLY:
            return local_models[0][1] if local_models else None
        elif self.deployment_mode == DeploymentMode.EXTERNAL_ONLY:
            return external_models[0][1] if external_models else None
        elif self.deployment_mode == DeploymentMode.LOCAL_PREFERRED:
            return local_models[0][1] if local_models else (external_models[0][1] if external_models else None)
        elif self.deployment_mode == DeploymentMode.EXTERNAL_PREFERRED:
            return external_models[0][1] if external_models else (local_models[0][1] if local_models else None)
        else:  # HYBRID
            return scored_models[0][1]  # Best overall score
    
    def get_fallback_models(self, primary_model: str, task_type: TaskType) -> List[str]:
        """Get fallback models for a primary model"""
        available_models = self._filter_available_models(task_type)
        
        # Remove primary model
        if primary_model in available_models:
            del available_models[primary_model]
        
        # Sort by fallback priority and performance
        fallbacks = []
        for model_id, profile in available_models.items():
            score = profile.fallback_priority * profile.performance_profile.success_rate
            fallbacks.append((score, model_id))
        
        fallbacks.sort(reverse=True)
        return [model_id for _, model_id in fallbacks[:3]]  # Top 3 fallbacks
    
    def configure_for_api(self, api: UnifiedModelAPI):
        """Configure UnifiedModelAPI with current model configurations"""
        for model_id, profile in self.model_profiles.items():
            if profile.enabled and profile.health_status:
                api.add_model(model_id, profile.model_config)
                logger.info(f"Added model to API: {model_id}")
    
    def update_health_status(self, model_id: str, is_healthy: bool):
        """Update health status for a model"""
        if model_id in self.model_profiles:
            self.model_profiles[model_id].health_status = is_healthy
            self.model_profiles[model_id].last_health_check = time.time()
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        local_count = sum(1 for p in self.model_profiles.values() 
                         if p.model_config.provider in [ModelProvider.OLLAMA, ModelProvider.LM_STUDIO, ModelProvider.LOCAL_AI])
        external_count = len(self.model_profiles) - local_count
        
        enabled_count = sum(1 for p in self.model_profiles.values() if p.enabled)
        healthy_count = sum(1 for p in self.model_profiles.values() if p.health_status)
        
        return {
            "deployment_mode": self.deployment_mode.value,
            "total_models": len(self.model_profiles),
            "local_models": local_count,
            "external_models": external_count,
            "enabled_models": enabled_count,
            "healthy_models": healthy_count,
            "model_tiers": {tier.value: sum(1 for p in self.model_profiles.values() if p.tier == tier) for tier in ModelTier},
            "supported_tasks": list(set(task.value for p in self.model_profiles.values() for task in p.model_config.capabilities))
        }
    
    def save_configurations(self):
        """Save all configurations to files"""
        try:
            # Save model profiles
            profiles_data = {model_id: profile.to_dict() for model_id, profile in self.model_profiles.items()}
            with open(self.config_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            # Save deployment configuration
            deployment_data = {
                "deployment_mode": self.deployment_mode.value,
                "config": self.deployment_config,
                "last_updated": time.time()
            }
            with open(self.deployment_config_file, 'w') as f:
                json.dump(deployment_data, f, indent=2)
            
            logger.info("Model configurations saved")
            
        except Exception as e:
            logger.error(f"Failed to save configurations: {e}")
    
    def export_configuration(self, export_path: str) -> bool:
        """Export configuration to a file"""
        try:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "deployment_mode": self.deployment_mode.value,
                "deployment_config": self.deployment_config,
                "model_profiles": {model_id: profile.to_dict() for model_id, profile in self.model_profiles.items()},
                "summary": self.get_configuration_summary()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_configuration(self, import_path: str) -> bool:
        """Import configuration from a file"""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            # Import deployment mode
            if "deployment_mode" in import_data:
                self.deployment_mode = DeploymentMode(import_data["deployment_mode"])
            
            # Import deployment config
            if "deployment_config" in import_data:
                self.deployment_config = import_data["deployment_config"]
            
            # Import model profiles
            if "model_profiles" in import_data:
                self._deserialize_profiles(import_data["model_profiles"])
            
            self.save_configurations()
            logger.info(f"Configuration imported from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False

# Example usage
if __name__ == "__main__":
    def test_model_configuration():
        # Initialize configuration manager
        config_manager = ModelConfigurationManager()
        
        # Get configuration summary
        summary = config_manager.get_configuration_summary()
        print(f"Configuration summary: {json.dumps(summary, indent=2)}")
        
        # Get optimal model for research task
        optimal_model = config_manager.get_optimal_model(TaskType.RESEARCH)
        print(f"Optimal model for research: {optimal_model}")
        
        # Get fallback models
        if optimal_model:
            fallbacks = config_manager.get_fallback_models(optimal_model, TaskType.RESEARCH)
            print(f"Fallback models: {fallbacks}")
        
        # Test different deployment modes
        for mode in DeploymentMode:
            config_manager.set_deployment_mode(mode)
            optimal = config_manager.get_optimal_model(TaskType.GENERAL)
            print(f"Mode {mode.value}: Optimal model = {optimal}")
        
        # Export configuration
        config_manager.export_configuration("test_config_export.json")
        print("Configuration exported")
    
    # Run test
    test_model_configuration()