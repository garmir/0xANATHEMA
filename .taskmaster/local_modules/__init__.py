#!/usr/bin/env python3
"""
Task Master AI Local Modules Package
Provides local LLM integration replacing external API dependencies
"""

from .core.api_abstraction import (
    UnifiedModelAPI, 
    TaskType, 
    ModelProvider, 
    ModelConfig,
    ModelConfigFactory,
    ModelResponse
)

from .core.recursive_prd_processor import (
    RecursivePRDProcessor,
    Task,
    DecompositionContext
)

from .research.local_rag_system import (
    LocalRAGSystem,
    Document,
    SearchResult,
    KnowledgeBase
)

from .optimization.evolutionary_optimization import (
    EvolutionaryOptimizer,
    LocalLLMFitnessEvaluator,
    PerformanceFitnessEvaluator,
    EvolutionConfig,
    Individual
)

from .meta_learning.meta_learning_framework import (
    MetaLearningEngine,
    LearningExperience,
    MetaPattern,
    MetaLearningTask
)

from .failure_recovery.failure_detection_recovery import (
    FailureRecoverySystem,
    FailureDetector,
    FailureEvent,
    FailureType,
    SeverityLevel
)

from .config.model_configuration import (
    ModelConfigurationManager,
    DeploymentMode,
    ModelTier,
    ModelConfigurationProfile
)

from .utils.performance_monitor import (
    PerformanceMonitor,
    CachedPerformanceMonitor,
    PerformanceCache,
    PerformanceMetric
)

from .utils.validation_tests import (
    ValidationTestSuite,
    run_validation
)

__version__ = "1.0.0"
__author__ = "Task Master AI"
__description__ = "Local LLM integration modules for Task Master AI"

# Package-level exports
__all__ = [
    # Core modules
    "UnifiedModelAPI",
    "TaskType", 
    "ModelProvider",
    "ModelConfig",
    "ModelConfigFactory",
    "ModelResponse",
    "RecursivePRDProcessor",
    "Task",
    "DecompositionContext",
    
    # Research modules
    "LocalRAGSystem",
    "Document",
    "SearchResult", 
    "KnowledgeBase",
    
    # Optimization modules
    "EvolutionaryOptimizer",
    "LocalLLMFitnessEvaluator",
    "PerformanceFitnessEvaluator",
    "EvolutionConfig",
    "Individual",
    
    # Meta-learning modules
    "MetaLearningEngine",
    "LearningExperience",
    "MetaPattern",
    "MetaLearningTask",
    
    # Failure recovery modules
    "FailureRecoverySystem",
    "FailureDetector",
    "FailureEvent",
    "FailureType",
    "SeverityLevel",
    
    # Configuration modules
    "ModelConfigurationManager",
    "DeploymentMode",
    "ModelTier",
    "ModelConfigurationProfile",
    
    # Utility modules
    "PerformanceMonitor",
    "CachedPerformanceMonitor",
    "PerformanceCache",
    "PerformanceMetric",
    "ValidationTestSuite",
    "run_validation"
]