#!/usr/bin/env python3
"""
Meta-Learning Framework for Task Master AI
Implements recursive meta-improvement analysis with local model routing
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
import logging
import hashlib
from collections import defaultdict
from enum import Enum

from ..core.api_abstraction import UnifiedModelAPI, TaskType, ModelResponse

logger = logging.getLogger(__name__)

class MetaLearningTask(Enum):
    """Types of meta-learning tasks"""
    PATTERN_RECOGNITION = "pattern_recognition"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    MODEL_SELECTION = "model_selection"
    WORKFLOW_IMPROVEMENT = "workflow_improvement"
    ERROR_PATTERN_ANALYSIS = "error_pattern_analysis"

@dataclass
class LearningExperience:
    """Represents a learning experience or data point"""
    id: str
    task_type: str
    context: Dict[str, Any]
    action_taken: Dict[str, Any]
    outcome: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "context": self.context,
            "action_taken": self.action_taken,
            "outcome": self.outcome,
            "performance_metrics": self.performance_metrics,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningExperience':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            task_type=data["task_type"],
            context=data["context"],
            action_taken=data["action_taken"],
            outcome=data["outcome"],
            performance_metrics=data["performance_metrics"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )

@dataclass
class MetaPattern:
    """Represents a learned meta-pattern"""
    id: str
    pattern_type: str
    conditions: Dict[str, Any]
    recommendations: Dict[str, Any]
    confidence_score: float
    evidence_count: int
    success_rate: float
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "conditions": self.conditions,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score,
            "evidence_count": self.evidence_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }

class MetaLearningEngine:
    """
    Core meta-learning engine that learns from system performance
    and makes strategic improvements using local LLMs
    """
    
    def __init__(self,
                 api: UnifiedModelAPI,
                 data_dir: str = ".taskmaster/local_modules/meta_learning"):
        self.api = api
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Learning data storage
        self.experiences: List[LearningExperience] = []
        self.patterns: Dict[str, MetaPattern] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Meta-learning configuration
        self.min_evidence_count = 3
        self.min_confidence_threshold = 0.7
        self.pattern_update_interval = 10  # Update patterns every N experiences
        
        # Load existing data
        self._load_learning_data()
        
        # Performance tracking
        self.meta_learning_stats = {
            "total_experiences": 0,
            "patterns_discovered": 0,
            "recommendations_made": 0,
            "improvements_implemented": 0,
            "avg_improvement_rate": 0.0
        }
    
    def _load_learning_data(self):
        """Load existing learning data"""
        experiences_file = self.data_dir / "experiences.json"
        patterns_file = self.data_dir / "patterns.json"
        
        if experiences_file.exists():
            try:
                with open(experiences_file, 'r') as f:
                    experiences_data = json.load(f)
                    self.experiences = [LearningExperience.from_dict(exp) for exp in experiences_data]
                logger.info(f"Loaded {len(self.experiences)} learning experiences")
            except Exception as e:
                logger.warning(f"Failed to load experiences: {e}")
        
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    self.patterns = {pid: MetaPattern(**pattern) for pid, pattern in patterns_data.items()}
                logger.info(f"Loaded {len(self.patterns)} meta-patterns")
            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")
    
    def _save_learning_data(self):
        """Save learning data"""
        try:
            experiences_file = self.data_dir / "experiences.json"
            with open(experiences_file, 'w') as f:
                json.dump([exp.to_dict() for exp in self.experiences], f, indent=2)
            
            patterns_file = self.data_dir / "patterns.json"
            with open(patterns_file, 'w') as f:
                json.dump({pid: pattern.to_dict() for pid, pattern in self.patterns.items()}, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def record_experience(self, experience: LearningExperience):
        """Record a new learning experience"""
        self.experiences.append(experience)
        self.meta_learning_stats["total_experiences"] += 1
        
        # Update performance history
        task_type = experience.task_type
        self.performance_history[task_type].append({
            "timestamp": experience.timestamp,
            "metrics": experience.performance_metrics,
            "context": experience.context
        })
        
        # Trigger pattern discovery if enough new experiences
        if len(self.experiences) % self.pattern_update_interval == 0:
            asyncio.create_task(self._discover_patterns())
        
        logger.info(f"Recorded experience: {experience.id} ({experience.task_type})")
    
    async def _discover_patterns(self):
        """Discover new patterns from recent experiences"""
        logger.info("Discovering meta-learning patterns...")
        
        # Group experiences by task type
        task_groups = defaultdict(list)
        for exp in self.experiences[-50:]:  # Last 50 experiences
            task_groups[exp.task_type].append(exp)
        
        # Analyze each task type
        for task_type, experiences in task_groups.items():
            if len(experiences) >= self.min_evidence_count:
                await self._analyze_task_type_patterns(task_type, experiences)
    
    async def _analyze_task_type_patterns(self, task_type: str, experiences: List[LearningExperience]):
        """Analyze patterns for a specific task type"""
        pattern_prompt = f"""
        Analyze these learning experiences to identify meta-patterns and insights:
        
        TASK TYPE: {task_type}
        
        EXPERIENCES:
        {json.dumps([exp.to_dict() for exp in experiences[-10:]], indent=2)}
        
        Identify patterns in:
        1. Context conditions that lead to successful outcomes
        2. Actions that consistently produce good results
        3. Performance characteristics and their relationships
        4. Common failure modes and their causes
        5. Optimization opportunities
        
        Focus on actionable insights that can improve future performance.
        
        Provide response in this JSON format:
        {{
            "patterns_identified": [
                {{
                    "pattern_type": "performance|strategy|error|optimization",
                    "conditions": {{"description": "when this pattern applies"}},
                    "recommendations": {{"action": "what to do", "rationale": "why"}},
                    "confidence": 0.0-1.0,
                    "evidence_strength": "weak|moderate|strong"
                }}
            ],
            "key_insights": ["insight1", "insight2", ...],
            "improvement_opportunities": ["opportunity1", "opportunity2", ...]
        }}
        """
        
        try:
            response = await self.api.generate(
                pattern_prompt,
                task_type=TaskType.ANALYSIS,
                temperature=0.3
            )
            
            # Parse response
            analysis_data = json.loads(response.content)
            
            # Create or update patterns
            for pattern_data in analysis_data.get("patterns_identified", []):
                await self._create_or_update_pattern(task_type, pattern_data, experiences)
            
            # Store insights
            self._store_insights(task_type, analysis_data)
            
        except Exception as e:
            logger.error(f"Pattern analysis failed for {task_type}: {e}")
    
    async def _create_or_update_pattern(self, task_type: str, pattern_data: Dict[str, Any], evidence: List[LearningExperience]):
        """Create or update a meta-pattern"""
        pattern_id = hashlib.md5(f"{task_type}_{pattern_data['pattern_type']}_{json.dumps(pattern_data['conditions'])}".encode()).hexdigest()
        
        confidence = pattern_data.get("confidence", 0.5)
        if confidence < self.min_confidence_threshold:
            return  # Skip low-confidence patterns
        
        if pattern_id in self.patterns:
            # Update existing pattern
            pattern = self.patterns[pattern_id]
            pattern.evidence_count += len(evidence)
            pattern.confidence_score = (pattern.confidence_score + confidence) / 2
            pattern.last_updated = time.time()
            
            # Recalculate success rate
            pattern.success_rate = self._calculate_success_rate(evidence)
            
        else:
            # Create new pattern
            pattern = MetaPattern(
                id=pattern_id,
                pattern_type=pattern_data["pattern_type"],
                conditions=pattern_data["conditions"],
                recommendations=pattern_data["recommendations"],
                confidence_score=confidence,
                evidence_count=len(evidence),
                success_rate=self._calculate_success_rate(evidence)
            )
            self.patterns[pattern_id] = pattern
            self.meta_learning_stats["patterns_discovered"] += 1
        
        logger.info(f"Pattern {'updated' if pattern_id in self.patterns else 'created'}: {pattern_id}")
    
    def _calculate_success_rate(self, experiences: List[LearningExperience]) -> float:
        """Calculate success rate from experiences"""
        if not experiences:
            return 0.0
        
        success_count = 0
        for exp in experiences:
            # Define success based on performance metrics
            metrics = exp.performance_metrics
            if metrics.get("success", False) or metrics.get("performance_score", 0) > 0.7:
                success_count += 1
        
        return success_count / len(experiences)
    
    def _store_insights(self, task_type: str, analysis_data: Dict[str, Any]):
        """Store insights from pattern analysis"""
        insights_file = self.data_dir / f"insights_{task_type}.json"
        
        insights = {
            "task_type": task_type,
            "timestamp": time.time(),
            "key_insights": analysis_data.get("key_insights", []),
            "improvement_opportunities": analysis_data.get("improvement_opportunities", [])
        }
        
        try:
            with open(insights_file, 'w') as f:
                json.dump(insights, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to store insights: {e}")
    
    async def get_recommendations(self, context: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Get meta-learning recommendations for a given context"""
        logger.info(f"Getting recommendations for {task_type}")
        
        # Find applicable patterns
        applicable_patterns = self._find_applicable_patterns(context, task_type)
        
        if not applicable_patterns:
            return await self._generate_generic_recommendations(context, task_type)
        
        # Generate recommendations based on patterns
        recommendations = await self._generate_pattern_based_recommendations(applicable_patterns, context, task_type)
        
        self.meta_learning_stats["recommendations_made"] += 1
        return recommendations
    
    def _find_applicable_patterns(self, context: Dict[str, Any], task_type: str) -> List[MetaPattern]:
        """Find patterns applicable to the current context"""
        applicable = []
        
        for pattern in self.patterns.values():
            if self._pattern_matches_context(pattern, context, task_type):
                applicable.append(pattern)
        
        # Sort by confidence and success rate
        applicable.sort(key=lambda p: (p.confidence_score * p.success_rate), reverse=True)
        
        return applicable[:5]  # Top 5 patterns
    
    def _pattern_matches_context(self, pattern: MetaPattern, context: Dict[str, Any], task_type: str) -> bool:
        """Check if a pattern matches the current context"""
        conditions = pattern.conditions
        
        # Simple matching based on context similarity
        # In a real implementation, this would be more sophisticated
        for key, value in conditions.items():
            if key in context:
                if isinstance(value, str) and isinstance(context[key], str):
                    if value.lower() in context[key].lower():
                        return True
                elif value == context[key]:
                    return True
        
        return False
    
    async def _generate_pattern_based_recommendations(self, patterns: List[MetaPattern], 
                                                    context: Dict[str, Any], 
                                                    task_type: str) -> Dict[str, Any]:
        """Generate recommendations based on applicable patterns"""
        recommendations_prompt = f"""
        Generate strategic recommendations based on learned meta-patterns:
        
        CURRENT CONTEXT:
        Task Type: {task_type}
        Context: {json.dumps(context, indent=2)}
        
        APPLICABLE PATTERNS:
        {json.dumps([pattern.to_dict() for pattern in patterns], indent=2)}
        
        Based on these patterns and the current context, provide:
        1. Strategic recommendations for optimal performance
        2. Specific actions to take
        3. Potential risks to avoid
        4. Performance optimization opportunities
        5. Model/strategy selection guidance
        
        Prioritize recommendations by expected impact and implementation feasibility.
        
        Provide response in this JSON format:
        {{
            "strategic_recommendations": [
                {{
                    "priority": "high|medium|low",
                    "action": "specific action to take",
                    "rationale": "why this action is recommended",
                    "expected_impact": "description of expected outcome",
                    "implementation_difficulty": "easy|medium|hard"
                }}
            ],
            "model_selection": {{
                "recommended_model": "model_identifier",
                "reasoning": "why this model is optimal"
            }},
            "risk_mitigation": ["risk1_mitigation", "risk2_mitigation"],
            "performance_optimizations": ["optimization1", "optimization2"],
            "confidence_score": 0.0-1.0
        }}
        """
        
        try:
            response = await self.api.generate(
                recommendations_prompt,
                task_type=TaskType.PLANNING,
                temperature=0.2
            )
            
            recommendations = json.loads(response.content)
            recommendations["patterns_used"] = [p.id for p in patterns]
            recommendations["recommendation_timestamp"] = time.time()
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Pattern-based recommendation generation failed: {e}")
            return await self._generate_generic_recommendations(context, task_type)
    
    async def _generate_generic_recommendations(self, context: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Generate generic recommendations when no patterns are available"""
        generic_prompt = f"""
        Provide strategic recommendations for this task context:
        
        TASK TYPE: {task_type}
        CONTEXT: {json.dumps(context, indent=2)}
        
        Since no specific learned patterns are available, provide general best practices and recommendations based on:
        1. Common strategies for this task type
        2. Standard optimization approaches
        3. Risk mitigation strategies
        4. Performance monitoring suggestions
        
        Focus on proven, low-risk approaches with good success rates.
        
        Provide response in this JSON format:
        {{
            "strategic_recommendations": [
                {{
                    "priority": "high|medium|low",
                    "action": "specific action to take",
                    "rationale": "why this action is recommended",
                    "expected_impact": "description of expected outcome"
                }}
            ],
            "general_guidance": "Overall strategic guidance",
            "confidence_score": 0.0-1.0
        }}
        """
        
        try:
            response = await self.api.generate(
                generic_prompt,
                task_type=TaskType.PLANNING,
                temperature=0.3
            )
            
            recommendations = json.loads(response.content)
            recommendations["type"] = "generic"
            recommendations["recommendation_timestamp"] = time.time()
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Generic recommendation generation failed: {e}")
            return {
                "strategic_recommendations": [
                    {
                        "priority": "medium",
                        "action": "Apply standard best practices for this task type",
                        "rationale": "Fallback to proven approaches",
                        "expected_impact": "Moderate performance improvement"
                    }
                ],
                "confidence_score": 0.3,
                "type": "fallback"
            }
    
    async def meta_improvement_analysis(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform meta-improvement analysis on system performance
        This is the core meta-learning function that analyzes overall system behavior
        """
        logger.info("Performing meta-improvement analysis...")
        
        # Collect recent performance data
        recent_experiences = self.experiences[-100:]  # Last 100 experiences
        performance_trends = self._analyze_performance_trends()
        
        meta_analysis_prompt = f"""
        Perform comprehensive meta-improvement analysis of the Task Master AI system:
        
        SYSTEM DATA:
        {json.dumps(system_data, indent=2)}
        
        RECENT PERFORMANCE TRENDS:
        {json.dumps(performance_trends, indent=2)}
        
        LEARNING STATISTICS:
        - Total experiences: {len(self.experiences)}
        - Patterns discovered: {len(self.patterns)}
        - Recommendations made: {self.meta_learning_stats['recommendations_made']}
        
        Analyze:
        1. Overall system performance trends
        2. Learning effectiveness and pattern quality
        3. Recommendation accuracy and impact
        4. Areas for meta-level improvements
        5. Strategic system optimizations
        6. Learning algorithm adjustments needed
        
        Focus on recursive self-improvement opportunities and meta-optimization strategies.
        
        Provide response in this JSON format:
        {{
            "performance_assessment": {{
                "overall_trend": "improving|stable|declining",
                "key_strengths": ["strength1", "strength2"],
                "areas_for_improvement": ["area1", "area2"],
                "performance_score": 0.0-1.0
            }},
            "learning_effectiveness": {{
                "pattern_quality": 0.0-1.0,
                "recommendation_accuracy": 0.0-1.0,
                "learning_rate": 0.0-1.0,
                "knowledge_coverage": 0.0-1.0
            }},
            "meta_improvements": [
                {{
                    "type": "algorithm|strategy|parameter|workflow",
                    "description": "what to improve",
                    "implementation": "how to implement",
                    "expected_impact": "expected outcome",
                    "priority": "high|medium|low"
                }}
            ],
            "strategic_optimizations": [
                {{
                    "optimization": "specific optimization",
                    "rationale": "why this optimization",
                    "implementation_complexity": "low|medium|high"
                }}
            ],
            "next_research_directions": ["direction1", "direction2"],
            "confidence_score": 0.0-1.0
        }}
        """
        
        try:
            response = await self.api.generate(
                meta_analysis_prompt,
                task_type=TaskType.ANALYSIS,
                temperature=0.2
            )
            
            meta_analysis = json.loads(response.content)
            meta_analysis["analysis_timestamp"] = time.time()
            meta_analysis["experiences_analyzed"] = len(recent_experiences)
            
            # Save meta-analysis
            analysis_file = self.data_dir / f"meta_analysis_{int(time.time())}.json"
            with open(analysis_file, 'w') as f:
                json.dump(meta_analysis, f, indent=2)
            
            return meta_analysis
            
        except Exception as e:
            logger.error(f"Meta-improvement analysis failed: {e}")
            return {
                "performance_assessment": {"overall_trend": "unknown", "performance_score": 0.5},
                "meta_improvements": [],
                "confidence_score": 0.0,
                "error": str(e)
            }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across task types"""
        trends = {}
        
        for task_type, history in self.performance_history.items():
            if len(history) < 3:
                continue
            
            # Calculate trend metrics
            recent_metrics = [entry["metrics"] for entry in history[-10:]]
            
            # Average performance over time
            performance_scores = []
            for metrics in recent_metrics:
                score = metrics.get("performance_score", metrics.get("success_rate", 0.5))
                performance_scores.append(score)
            
            if performance_scores:
                trends[task_type] = {
                    "avg_performance": sum(performance_scores) / len(performance_scores),
                    "trend_direction": "improving" if performance_scores[-1] > performance_scores[0] else "declining",
                    "data_points": len(history),
                    "recent_performance": performance_scores[-3:] if len(performance_scores) >= 3 else performance_scores
                }
        
        return trends
    
    async def optimize_meta_learning_parameters(self) -> Dict[str, Any]:
        """Optimize meta-learning parameters based on performance"""
        optimization_prompt = f"""
        Optimize meta-learning parameters based on current performance:
        
        CURRENT PARAMETERS:
        - Min evidence count: {self.min_evidence_count}
        - Min confidence threshold: {self.min_confidence_threshold}
        - Pattern update interval: {self.pattern_update_interval}
        
        PERFORMANCE DATA:
        - Total experiences: {len(self.experiences)}
        - Patterns discovered: {len(self.patterns)}
        - Pattern quality (avg confidence): {np.mean([p.confidence_score for p in self.patterns.values()]) if self.patterns else 0}
        
        Recommend optimal parameter values that will:
        1. Improve pattern quality
        2. Increase learning efficiency
        3. Balance exploration vs exploitation
        4. Optimize recommendation accuracy
        
        Provide response in this JSON format:
        {{
            "recommended_parameters": {{
                "min_evidence_count": int,
                "min_confidence_threshold": float,
                "pattern_update_interval": int
            }},
            "rationale": "explanation of recommendations",
            "expected_improvements": ["improvement1", "improvement2"]
        }}
        """
        
        try:
            response = await self.api.generate(
                optimization_prompt,
                task_type=TaskType.OPTIMIZATION,
                temperature=0.3
            )
            
            optimization_result = json.loads(response.content)
            
            # Apply recommended parameters
            recommended = optimization_result["recommended_parameters"]
            self.min_evidence_count = recommended.get("min_evidence_count", self.min_evidence_count)
            self.min_confidence_threshold = recommended.get("min_confidence_threshold", self.min_confidence_threshold)
            self.pattern_update_interval = recommended.get("pattern_update_interval", self.pattern_update_interval)
            
            logger.info("Meta-learning parameters optimized")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return {"error": str(e)}
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of meta-learning progress"""
        pattern_confidence_scores = [p.confidence_score for p in self.patterns.values()]
        pattern_success_rates = [p.success_rate for p in self.patterns.values()]
        
        return {
            "total_experiences": len(self.experiences),
            "patterns_discovered": len(self.patterns),
            "avg_pattern_confidence": np.mean(pattern_confidence_scores) if pattern_confidence_scores else 0,
            "avg_pattern_success_rate": np.mean(pattern_success_rates) if pattern_success_rates else 0,
            "recommendations_made": self.meta_learning_stats["recommendations_made"],
            "learning_domains": list(self.performance_history.keys()),
            "most_confident_pattern": max(self.patterns.values(), key=lambda p: p.confidence_score).id if self.patterns else None,
            "learning_active": len(self.experiences) > 0
        }
    
    def cleanup_old_data(self, max_experiences: int = 1000, max_age_days: int = 30):
        """Clean up old learning data"""
        max_age_seconds = max_age_days * 24 * 3600
        current_time = time.time()
        
        # Remove old experiences
        self.experiences = [
            exp for exp in self.experiences 
            if (current_time - exp.timestamp) < max_age_seconds
        ][-max_experiences:]
        
        # Remove low-confidence patterns
        patterns_to_remove = [
            pid for pid, pattern in self.patterns.items()
            if pattern.confidence_score < 0.3 or pattern.evidence_count < 2
        ]
        
        for pid in patterns_to_remove:
            del self.patterns[pid]
        
        self._save_learning_data()
        logger.info(f"Cleaned up learning data: {len(self.experiences)} experiences, {len(self.patterns)} patterns retained")

# Example usage
if __name__ == "__main__":
    async def test_meta_learning():
        from ..core.api_abstraction import UnifiedModelAPI, ModelConfigFactory
        
        # Initialize API
        api = UnifiedModelAPI()
        api.add_model("ollama-llama2", ModelConfigFactory.create_ollama_config(
            "llama2", capabilities=[TaskType.ANALYSIS, TaskType.PLANNING, TaskType.OPTIMIZATION]
        ))
        
        # Initialize meta-learning engine
        meta_engine = MetaLearningEngine(api)
        
        # Record some sample experiences
        experience1 = LearningExperience(
            id="exp_1",
            task_type="task_decomposition",
            context={"complexity": "high", "domain": "technical"},
            action_taken={"strategy": "recursive_breakdown", "depth": 3},
            outcome={"success": True, "subtasks_generated": 12},
            performance_metrics={"performance_score": 0.85, "success": True}
        )
        
        meta_engine.record_experience(experience1)
        
        # Get recommendations
        recommendations = await meta_engine.get_recommendations(
            context={"complexity": "medium", "domain": "technical"},
            task_type="task_decomposition"
        )
        
        print(f"Recommendations: {json.dumps(recommendations, indent=2)}")
        
        # Perform meta-analysis
        system_data = {"version": "1.0", "uptime": 3600, "total_tasks": 50}
        meta_analysis = await meta_engine.meta_improvement_analysis(system_data)
        
        print(f"Meta-analysis: {json.dumps(meta_analysis, indent=2)}")
        
        # Get learning summary
        summary = meta_engine.get_learning_summary()
        print(f"Learning summary: {json.dumps(summary, indent=2)}")
    
    # Run test
    asyncio.run(test_meta_learning())