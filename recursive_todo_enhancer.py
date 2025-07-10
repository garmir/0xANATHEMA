#!/usr/bin/env python3
"""
Recursive Todo Enhancement Engine - Core Implementation
Atomic Task 51.3: Implement Core Recursive Enhancement Logic

This module provides the production implementation of the recursive todo
enhancement engine with complete recursion logic, safety mechanisms,
and performance optimization.
"""

import asyncio
import json
import logging
import time
import uuid
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from pathlib import Path
import statistics
import hashlib
from collections import defaultdict, deque
import copy


class RecursionLimitExceeded(Exception):
    """Raised when recursion depth limit is exceeded"""
    pass


class EnhancementTimeout(Exception):
    """Raised when enhancement processing times out"""
    pass


class EnhancementQualityMetrics:
    """Metrics for measuring enhancement quality"""
    
    @staticmethod
    def calculate_clarity_score(text: str) -> float:
        """Calculate clarity score based on text characteristics"""
        if not text.strip():
            return 0.0
        
        words = text.lower().split()
        word_count = len(words)
        
        # Clarity factors
        factors = {
            "has_action_verb": any(verb in words for verb in [
                "implement", "create", "design", "build", "test", "fix", 
                "develop", "write", "add", "remove", "update", "refactor"
            ]),
            "is_specific": word_count >= 4,
            "has_concrete_outcome": any(outcome in words for outcome in [
                "complete", "finish", "deliver", "achieve", "resolve", "deploy"
            ]),
            "avoids_ambiguity": not any(ambiguous in words for ambiguous in [
                "maybe", "probably", "might", "could", "perhaps", "somehow"
            ]),
            "proper_length": 4 <= word_count <= 20,
            "has_clear_subject": not text.lower().startswith(("fix", "debug", "handle")),
            "technical_specificity": any(tech in words for tech in [
                "api", "database", "frontend", "backend", "service", "component"
            ])
        }
        
        return sum(factors.values()) / len(factors)
    
    @staticmethod
    def calculate_complexity_score(text: str) -> float:
        """Calculate complexity score based on content analysis"""
        if not text.strip():
            return 1.0
        
        words = text.lower().split()
        word_count = len(words)
        
        complexity_indicators = {
            "length_complexity": min(10.0, word_count / 3),
            "technical_complexity": sum(1 for word in words if word in [
                "architecture", "system", "framework", "integration", "optimization",
                "algorithm", "infrastructure", "scalability", "performance"
            ]),
            "scope_complexity": sum(1 for word in words if word in [
                "comprehensive", "complete", "full", "entire", "multiple", "various"
            ]),
            "action_complexity": sum(1 for word in words if word in [
                "design", "architect", "implement", "integrate", "optimize"
            ])
        }
        
        return min(10.0, sum(complexity_indicators.values()))
    
    @staticmethod
    def calculate_improvement_score(original_text: str, enhanced_text: str) -> float:
        """Calculate improvement score between original and enhanced text"""
        original_clarity = EnhancementQualityMetrics.calculate_clarity_score(original_text)
        enhanced_clarity = EnhancementQualityMetrics.calculate_clarity_score(enhanced_text)
        
        # Weight clarity improvement more heavily
        clarity_improvement = enhanced_clarity - original_clarity
        
        # Additional improvement factors
        length_improvement = 0.0
        if len(enhanced_text.split()) > len(original_text.split()):
            length_improvement = 0.1
        
        specificity_improvement = 0.0
        if "with" in enhanced_text.lower() and "with" not in original_text.lower():
            specificity_improvement = 0.1
        
        return max(0.0, clarity_improvement + length_improvement + specificity_improvement)


@dataclass
class RecursiveEnhancementSession:
    """Session tracking for recursive enhancement operations"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_depth: int = 3
    current_depth: int = 0
    quality_threshold: float = 0.8
    improvement_threshold: float = 0.05
    timeout_seconds: float = 30.0
    started_at: datetime = field(default_factory=datetime.now)
    enhancement_history: List[Dict[str, Any]] = field(default_factory=list)
    total_processing_time: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if session has exceeded timeout"""
        elapsed = (datetime.now() - self.started_at).total_seconds()
        return elapsed >= self.timeout_seconds
    
    def should_recurse(self, current_quality: float, improvement: float) -> bool:
        """Determine if recursion should continue"""
        return (
            self.current_depth < self.max_depth and
            current_quality < self.quality_threshold and
            improvement > self.improvement_threshold and
            not self.is_expired()
        )
    
    def enter_recursion(self) -> 'RecursiveEnhancementSession':
        """Create new session for deeper recursion"""
        new_session = copy.deepcopy(self)
        new_session.current_depth += 1
        new_session.session_id = str(uuid.uuid4())
        return new_session


class BaseEnhancementStrategy(ABC):
    """Abstract base class for enhancement strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.usage_count = 0
        self.success_count = 0
        self.total_improvement = 0.0
        self.average_processing_time = 0.0
    
    @abstractmethod
    async def can_enhance(self, text: str, context: Dict[str, Any] = None) -> bool:
        """Check if this strategy can enhance the given text"""
        pass
    
    @abstractmethod
    async def enhance(self, text: str, context: Dict[str, Any] = None) -> str:
        """Apply enhancement to the text"""
        pass
    
    def record_usage(self, improvement: float, processing_time: float, success: bool):
        """Record strategy usage statistics"""
        self.usage_count += 1
        if success:
            self.success_count += 1
            self.total_improvement += improvement
        
        # Update average processing time
        if self.average_processing_time == 0:
            self.average_processing_time = processing_time
        else:
            self.average_processing_time = (
                (self.average_processing_time * (self.usage_count - 1) + processing_time) 
                / self.usage_count
            )
    
    def get_effectiveness_score(self) -> float:
        """Calculate strategy effectiveness score"""
        if self.usage_count == 0:
            return 0.5
        
        success_rate = self.success_count / self.usage_count
        avg_improvement = self.total_improvement / max(1, self.success_count)
        speed_factor = max(0.1, 1.0 - (self.average_processing_time / 10.0))
        
        return (success_rate * 0.4 + avg_improvement * 0.4 + speed_factor * 0.2)


class ClarityEnhancementStrategy(BaseEnhancementStrategy):
    """Strategy for improving text clarity and specificity"""
    
    def __init__(self):
        super().__init__("clarity_enhancement")
        self.action_verbs = [
            "implement", "create", "design", "build", "develop", "write",
            "add", "remove", "update", "refactor", "fix", "test", "deploy"
        ]
        self.specificity_phrases = [
            "with clear acceptance criteria",
            "with comprehensive documentation",
            "with unit tests",
            "with error handling",
            "with performance monitoring"
        ]
    
    async def can_enhance(self, text: str, context: Dict[str, Any] = None) -> bool:
        clarity_score = EnhancementQualityMetrics.calculate_clarity_score(text)
        return clarity_score < 0.7
    
    async def enhance(self, text: str, context: Dict[str, Any] = None) -> str:
        enhanced = text.strip()
        
        # Add action verb if missing
        has_action_verb = any(verb in enhanced.lower() for verb in self.action_verbs)
        if not has_action_verb:
            if enhanced.lower().startswith(("todo", "task", "work on")):
                enhanced = re.sub(r'^(todo|task|work on)\s*:?\s*', 'Implement ', enhanced, flags=re.IGNORECASE)
            else:
                enhanced = f"Implement {enhanced}"
        
        # Improve specificity for short descriptions
        words = enhanced.split()
        if len(words) < 5:
            if "system" in enhanced.lower() or "component" in enhanced.lower():
                enhanced += " with modular architecture and comprehensive testing"
            elif "bug" in enhanced.lower() or "fix" in enhanced.lower():
                enhanced += " with root cause analysis and regression tests"
            elif "feature" in enhanced.lower() or "functionality" in enhanced.lower():
                enhanced += " with user acceptance criteria and documentation"
            else:
                enhanced += " with clear deliverables and acceptance criteria"
        
        # Remove ambiguous language
        ambiguity_replacements = {
            r'\bmybe\b': 'should',
            r'\bprobably\b': 'will',
            r'\bmight\b': 'should',
            r'\bcould\b': 'will',
            r'\bperhaps\b': 'should',
            r'\bsomehow\b': 'by',
            r'\bsort of\b': '',
            r'\bkind of\b': ''
        }
        
        for pattern, replacement in ambiguity_replacements.items():
            enhanced = re.sub(pattern, replacement, enhanced, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        enhanced = ' '.join(enhanced.split())
        
        return enhanced


class AtomicDecompositionStrategy(BaseEnhancementStrategy):
    """Strategy for breaking complex tasks into atomic components"""
    
    def __init__(self):
        super().__init__("atomic_decomposition")
        self.complexity_threshold = 7.0
    
    async def can_enhance(self, text: str, context: Dict[str, Any] = None) -> bool:
        complexity = EnhancementQualityMetrics.calculate_complexity_score(text)
        word_count = len(text.split())
        return complexity > self.complexity_threshold or word_count > 15
    
    async def enhance(self, text: str, context: Dict[str, Any] = None) -> str:
        # For atomic decomposition, we enhance by adding structure
        enhanced = text.strip()
        
        # Identify task type and add structure
        if "implement" in enhanced.lower():
            enhanced += " (Phases: 1) Requirements analysis 2) Design 3) Implementation 4) Testing)"
        elif "design" in enhanced.lower():
            enhanced += " (Steps: 1) Research 2) Architecture 3) Prototype 4) Review)"
        elif "test" in enhanced.lower():
            enhanced += " (Approach: 1) Unit tests 2) Integration tests 3) Performance tests 4) Documentation)"
        elif "fix" in enhanced.lower() or "bug" in enhanced.lower():
            enhanced += " (Process: 1) Reproduce issue 2) Root cause analysis 3) Fix implementation 4) Verification)"
        else:
            enhanced += " (Breakdown: 1) Planning 2) Execution 3) Validation 4) Documentation)"
        
        return enhanced


class ContextEnrichmentStrategy(BaseEnhancementStrategy):
    """Strategy for adding relevant context and dependencies"""
    
    def __init__(self):
        super().__init__("context_enrichment")
    
    async def can_enhance(self, text: str, context: Dict[str, Any] = None) -> bool:
        # Check if text lacks context indicators
        context_indicators = ["with", "for", "using", "via", "through", "including"]
        has_context = any(indicator in text.lower() for indicator in context_indicators)
        return not has_context and len(text.split()) >= 3
    
    async def enhance(self, text: str, context: Dict[str, Any] = None) -> str:
        enhanced = text.strip()
        
        # Add context based on keywords
        if "api" in enhanced.lower():
            enhanced += " using RESTful design principles and proper authentication"
        elif "database" in enhanced.lower():
            enhanced += " with proper indexing and transaction management"
        elif "frontend" in enhanced.lower():
            enhanced += " with responsive design and accessibility compliance"
        elif "backend" in enhanced.lower():
            enhanced += " with scalable architecture and error handling"
        elif "test" in enhanced.lower():
            enhanced += " including unit, integration, and end-to-end scenarios"
        elif "deploy" in enhanced.lower():
            enhanced += " with CI/CD pipeline and monitoring"
        else:
            enhanced += " with appropriate documentation and error handling"
        
        return enhanced


class PriorityOptimizationStrategy(BaseEnhancementStrategy):
    """Strategy for optimizing task priority and urgency"""
    
    def __init__(self):
        super().__init__("priority_optimization")
    
    async def can_enhance(self, text: str, context: Dict[str, Any] = None) -> bool:
        # Check if priority indicators are missing
        priority_indicators = ["urgent", "critical", "important", "asap", "priority"]
        blocking_indicators = ["blocking", "blocks", "dependency", "prerequisite"]
        
        has_priority = any(indicator in text.lower() for indicator in priority_indicators)
        has_blocking = any(indicator in text.lower() for indicator in blocking_indicators)
        
        return not (has_priority or has_blocking)
    
    async def enhance(self, text: str, context: Dict[str, Any] = None) -> str:
        enhanced = text.strip()
        
        # Add priority context based on content
        if any(word in enhanced.lower() for word in ["bug", "fix", "broken", "error"]):
            enhanced += " [HIGH PRIORITY - Bug fix]"
        elif any(word in enhanced.lower() for word in ["security", "vulnerability", "auth"]):
            enhanced += " [CRITICAL - Security related]"
        elif any(word in enhanced.lower() for word in ["performance", "optimize", "slow"]):
            enhanced += " [MEDIUM PRIORITY - Performance improvement]"
        elif any(word in enhanced.lower() for word in ["documentation", "docs", "readme"]):
            enhanced += " [LOW PRIORITY - Documentation]"
        else:
            enhanced += " [MEDIUM PRIORITY - Standard development task]"
        
        return enhanced


class ImplementationGuidanceStrategy(BaseEnhancementStrategy):
    """Strategy for adding implementation hints and guidance"""
    
    def __init__(self):
        super().__init__("implementation_guidance")
    
    async def can_enhance(self, text: str, context: Dict[str, Any] = None) -> bool:
        # Check if implementation guidance is missing
        guidance_indicators = ["steps:", "approach:", "method:", "how to", "process:"]
        has_guidance = any(indicator in text.lower() for indicator in guidance_indicators)
        return not has_guidance and len(text.split()) >= 4
    
    async def enhance(self, text: str, context: Dict[str, Any] = None) -> str:
        enhanced = text.strip()
        
        # Add implementation guidance based on task type
        if "implement" in enhanced.lower() and "api" in enhanced.lower():
            enhanced += " (Approach: Define endpoints → Implement handlers → Add validation → Write tests)"
        elif "design" in enhanced.lower():
            enhanced += " (Method: Research → Wireframes → Prototyping → Review & iterate)"
        elif "test" in enhanced.lower():
            enhanced += " (Strategy: Test planning → Unit tests → Integration tests → Coverage analysis)"
        elif "deploy" in enhanced.lower():
            enhanced += " (Process: Build → Stage → Test → Production deployment)"
        elif "optimize" in enhanced.lower():
            enhanced += " (Steps: Profiling → Bottleneck identification → Implementation → Validation)"
        else:
            enhanced += " (Suggested approach: Plan → Implement → Test → Review)"
        
        return enhanced


class RecursiveTodoEnhancer:
    """Core recursive todo enhancement engine"""
    
    def __init__(self, max_depth: int = 3, timeout_seconds: float = 30.0):
        self.max_depth = max_depth
        self.timeout_seconds = timeout_seconds
        self.strategies: List[BaseEnhancementStrategy] = []
        self.enhancement_cache: Dict[str, Dict[str, Any]] = {}
        self.session_history: List[RecursiveEnhancementSession] = []
        self.logger = logging.getLogger("RecursiveTodoEnhancer")
        
        # Performance metrics
        self.total_enhancements = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        
        # Initialize strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize enhancement strategies"""
        self.strategies = [
            ClarityEnhancementStrategy(),
            AtomicDecompositionStrategy(),
            ContextEnrichmentStrategy(),
            PriorityOptimizationStrategy(),
            ImplementationGuidanceStrategy()
        ]
        
        self.logger.info(f"Initialized {len(self.strategies)} enhancement strategies")
    
    def _get_cache_key(self, text: str, context: Dict[str, Any] = None) -> str:
        """Generate cache key for enhancement results"""
        context_str = json.dumps(context or {}, sort_keys=True)
        combined = f"{text}|{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _select_best_strategy(self, text: str, context: Dict[str, Any] = None) -> Optional[BaseEnhancementStrategy]:
        """Select the most effective strategy for the given text"""
        applicable_strategies = []
        
        for strategy in self.strategies:
            if await strategy.can_enhance(text, context):
                effectiveness = strategy.get_effectiveness_score()
                applicable_strategies.append((strategy, effectiveness))
        
        if not applicable_strategies:
            return None
        
        # Sort by effectiveness and return best strategy
        applicable_strategies.sort(key=lambda x: x[1], reverse=True)
        return applicable_strategies[0][0]
    
    async def enhance_recursive(self, text: str, context: Dict[str, Any] = None,
                              session: RecursiveEnhancementSession = None) -> Dict[str, Any]:
        """Recursively enhance todo text with safety mechanisms"""
        
        # Initialize session if not provided
        if session is None:
            session = RecursiveEnhancementSession(
                max_depth=self.max_depth,
                timeout_seconds=self.timeout_seconds
            )
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(text, context)
        if cache_key in self.enhancement_cache:
            self.cache_hits += 1
            cached_result = self.enhancement_cache[cache_key]
            self.logger.debug(f"Cache hit for text: {text[:50]}...")
            return cached_result
        
        # Safety checks
        if session.is_expired():
            raise EnhancementTimeout(f"Enhancement session expired after {session.timeout_seconds}s")
        
        if session.current_depth >= session.max_depth:
            raise RecursionLimitExceeded(f"Maximum recursion depth {session.max_depth} exceeded")
        
        self.logger.info(f"Enhancing at depth {session.current_depth}: {text[:50]}...")
        
        # Calculate initial quality metrics
        original_clarity = EnhancementQualityMetrics.calculate_clarity_score(text)
        original_complexity = EnhancementQualityMetrics.calculate_complexity_score(text)
        
        # Select and apply best strategy
        strategy = await self._select_best_strategy(text, context)
        if not strategy:
            # No applicable strategy found
            result = {
                "original_text": text,
                "enhanced_text": text,
                "improvement_score": 0.0,
                "strategy_used": None,
                "recursion_depth": session.current_depth,
                "processing_time": time.time() - start_time,
                "quality_metrics": {
                    "original_clarity": original_clarity,
                    "enhanced_clarity": original_clarity,
                    "complexity": original_complexity
                }
            }
            self.enhancement_cache[cache_key] = result
            return result
        
        # Apply enhancement
        strategy_start_time = time.time()
        enhanced_text = await strategy.enhance(text, context)
        strategy_processing_time = time.time() - strategy_start_time
        
        # Calculate improvement
        enhanced_clarity = EnhancementQualityMetrics.calculate_clarity_score(enhanced_text)
        improvement_score = EnhancementQualityMetrics.calculate_improvement_score(text, enhanced_text)
        
        # Record strategy usage
        success = improvement_score > 0.01
        strategy.record_usage(improvement_score, strategy_processing_time, success)
        
        # Update session history
        enhancement_record = {
            "depth": session.current_depth,
            "strategy": strategy.name,
            "original_text": text,
            "enhanced_text": enhanced_text,
            "improvement": improvement_score,
            "processing_time": strategy_processing_time
        }
        session.enhancement_history.append(enhancement_record)
        
        # Determine if recursive enhancement should continue
        if session.should_recurse(enhanced_clarity, improvement_score):
            self.logger.info(f"Applying recursive enhancement at depth {session.current_depth + 1}")
            
            # Create new session for recursion
            recursive_session = session.enter_recursion()
            
            try:
                # Recursive call
                recursive_result = await self.enhance_recursive(
                    enhanced_text, context, recursive_session
                )
                
                # Use recursive result if better
                if recursive_result["improvement_score"] > improvement_score:
                    enhanced_text = recursive_result["enhanced_text"]
                    improvement_score = recursive_result["improvement_score"]
                    enhancement_record["recursive_improvement"] = recursive_result
                
            except (RecursionLimitExceeded, EnhancementTimeout) as e:
                self.logger.warning(f"Recursion terminated: {e}")
        
        # Calculate final metrics
        total_processing_time = time.time() - start_time
        session.total_processing_time += total_processing_time
        
        # Create result
        result = {
            "original_text": text,
            "enhanced_text": enhanced_text,
            "improvement_score": improvement_score,
            "strategy_used": strategy.name,
            "recursion_depth": session.current_depth,
            "processing_time": total_processing_time,
            "quality_metrics": {
                "original_clarity": original_clarity,
                "enhanced_clarity": enhanced_clarity,
                "complexity": original_complexity
            },
            "session_id": session.session_id,
            "enhancement_history": session.enhancement_history
        }
        
        # Cache result
        self.enhancement_cache[cache_key] = result
        
        # Update global metrics
        self.total_enhancements += 1
        self.total_processing_time += total_processing_time
        
        self.logger.info(f"Enhancement completed: {improvement_score:.3f} improvement in {total_processing_time:.3f}s")
        
        return result
    
    async def enhance_batch(self, texts: List[str], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Enhance multiple todo texts in batch"""
        self.logger.info(f"Starting batch enhancement of {len(texts)} items")
        
        results = []
        for i, text in enumerate(texts):
            try:
                self.logger.debug(f"Processing item {i+1}/{len(texts)}")
                result = await self.enhance_recursive(text, context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error enhancing text '{text[:50]}...': {e}")
                # Add error result
                results.append({
                    "original_text": text,
                    "enhanced_text": text,
                    "improvement_score": 0.0,
                    "error": str(e),
                    "processing_time": 0.0
                })
        
        self.logger.info(f"Batch enhancement completed: {len(results)} items processed")
        return results
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the enhancement engine"""
        if self.total_enhancements == 0:
            return {"status": "no_enhancements_performed"}
        
        # Strategy statistics
        strategy_stats = {}
        for strategy in self.strategies:
            strategy_stats[strategy.name] = {
                "usage_count": strategy.usage_count,
                "success_count": strategy.success_count,
                "success_rate": strategy.success_count / max(1, strategy.usage_count),
                "average_improvement": strategy.total_improvement / max(1, strategy.success_count),
                "average_processing_time": strategy.average_processing_time,
                "effectiveness_score": strategy.get_effectiveness_score()
            }
        
        return {
            "total_enhancements": self.total_enhancements,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / self.total_enhancements,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / self.total_enhancements,
            "strategy_statistics": strategy_stats,
            "cache_size": len(self.enhancement_cache),
            "sessions_completed": len(self.session_history)
        }
    
    def clear_cache(self):
        """Clear enhancement cache"""
        self.enhancement_cache.clear()
        self.cache_hits = 0
        self.logger.info("Enhancement cache cleared")


# Export key classes
__all__ = [
    "RecursiveTodoEnhancer", "RecursiveEnhancementSession", "BaseEnhancementStrategy",
    "ClarityEnhancementStrategy", "AtomicDecompositionStrategy", "ContextEnrichmentStrategy",
    "PriorityOptimizationStrategy", "ImplementationGuidanceStrategy", "EnhancementQualityMetrics",
    "RecursionLimitExceeded", "EnhancementTimeout"
]


if __name__ == "__main__":
    # Demo implementation
    async def demo():
        logging.basicConfig(level=logging.INFO)
        
        print("🔄 Recursive Todo Enhancement Engine - Core Logic Demo")
        print("=" * 70)
        
        # Create enhancer
        enhancer = RecursiveTodoEnhancer(max_depth=3, timeout_seconds=60.0)
        
        # Test cases with varying complexity
        test_todos = [
            "fix bug",
            "maybe add some documentation",
            "implement comprehensive user authentication system with oauth and jwt",
            "optimize database performance",
            "create api endpoint for user management",
            "refactor legacy code",
            "setup ci/cd pipeline with automated testing and deployment",
            "research machine learning algorithms for recommendation engine"
        ]
        
        print(f"🧪 Testing recursive enhancement on {len(test_todos)} todos...\n")
        
        # Process each todo
        for i, todo in enumerate(test_todos, 1):
            print(f"📝 Todo {i}: '{todo}'")
            print(f"   Original clarity: {EnhancementQualityMetrics.calculate_clarity_score(todo):.3f}")
            print(f"   Original complexity: {EnhancementQualityMetrics.calculate_complexity_score(todo):.3f}")
            
            try:
                result = await enhancer.enhance_recursive(todo)
                
                print(f"   ✨ Enhanced: '{result['enhanced_text']}'")
                print(f"   📈 Improvement: {result['improvement_score']:.3f}")
                print(f"   🔧 Strategy: {result['strategy_used']}")
                print(f"   🔄 Recursion depth: {result['recursion_depth']}")
                print(f"   ⏱️  Processing time: {result['processing_time']:.3f}s")
                print(f"   📊 Enhanced clarity: {result['quality_metrics']['enhanced_clarity']:.3f}")
                
                if result['enhancement_history']:
                    print(f"   📚 Enhancement steps: {len(result['enhancement_history'])}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            print()
        
        # Show performance statistics
        stats = enhancer.get_performance_statistics()
        print("📊 Performance Statistics:")
        print(f"   Total enhancements: {stats['total_enhancements']}")
        print(f"   Average processing time: {stats['average_processing_time']:.3f}s")
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
        
        print("\n🔧 Strategy Performance:")
        for strategy_name, strategy_stats in stats['strategy_statistics'].items():
            if strategy_stats['usage_count'] > 0:
                print(f"   • {strategy_name}:")
                print(f"     Usage: {strategy_stats['usage_count']}")
                print(f"     Success rate: {strategy_stats['success_rate']:.1%}")
                print(f"     Avg improvement: {strategy_stats['average_improvement']:.3f}")
                print(f"     Effectiveness: {strategy_stats['effectiveness_score']:.3f}")
        
        # Test batch processing
        print(f"\n🚀 Testing batch enhancement...")
        batch_results = await enhancer.enhance_batch(test_todos[:4])
        
        print(f"   Processed {len(batch_results)} items in batch")
        successful_enhancements = [r for r in batch_results if r.get('improvement_score', 0) > 0]
        print(f"   Successful enhancements: {len(successful_enhancements)}")
        
        if successful_enhancements:
            avg_improvement = statistics.mean(r['improvement_score'] for r in successful_enhancements)
            print(f"   Average improvement: {avg_improvement:.3f}")
        
        print(f"\n✅ Recursive Todo Enhancement Engine core logic operational!")
    
    # Run demo
    asyncio.run(demo())