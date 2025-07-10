#!/usr/bin/env python3
"""
Meta-Learning Engine
Monitors improvement effectiveness and adaptively adjusts strategies
Now integrated with Local LLM Adapter for enhanced strategy evaluation
"""

import json
import time
import statistics
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import math

# Import Local LLM Adapter
try:
    from local_llm_adapter import LocalLLMAdapter
except ImportError:
    print("⚠️ Local LLM Adapter not found - running in simulation mode")
    LocalLLMAdapter = None

class StrategyType(Enum):
    """Types of improvement strategies"""
    CODE_OPTIMIZATION = "code_optimization"
    PROCESS_IMPROVEMENT = "process_improvement"
    RESOURCE_MANAGEMENT = "resource_management"
    WORKFLOW_AUTOMATION = "workflow_automation"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    PERFORMANCE_TUNING = "performance_tuning"

class LearningPhase(Enum):
    """Phases of meta-learning"""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    REFINEMENT = "refinement"
    CONVERGENCE = "convergence"

@dataclass
class StrategyPerformance:
    """Performance tracking for individual strategies"""
    strategy_id: str
    strategy_type: StrategyType
    success_count: int
    failure_count: int
    total_attempts: int
    average_improvement: float
    success_rate: float
    effectiveness_score: float
    last_used: datetime
    adaptation_count: int

@dataclass
class LearningPattern:
    """Detected learning pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    conditions: Dict[str, Any]
    recommended_strategies: List[str]
    confidence: float
    frequency: int
    discovered_at: datetime

@dataclass
class AdaptationEvent:
    """Strategy adaptation event"""
    event_id: str
    timestamp: datetime
    trigger_condition: str
    old_strategy: str
    new_strategy: str
    reasoning: str
    expected_improvement: float

@dataclass
class MetaLearningState:
    """Current state of meta-learning system"""
    current_phase: LearningPhase
    exploration_rate: float
    exploitation_rate: float
    convergence_threshold: float
    adaptation_sensitivity: float
    learning_rate: float

class MetaLearningEngine:
    """Meta-learning engine that monitors and adapts improvement strategies"""
    
    def __init__(self, meta_dir: str = '.taskmaster/ai/meta', use_local_llm: bool = True):
        self.meta_dir = Path(meta_dir)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Local LLM Adapter
        self.use_local_llm = use_local_llm and LocalLLMAdapter is not None
        if self.use_local_llm:
            try:
                self.llm_adapter = LocalLLMAdapter()
                print("✅ Meta-Learning Engine initialized with Local LLM support")
            except Exception as e:
                print(f"⚠️ Failed to initialize Local LLM Adapter: {e}")
                self.llm_adapter = None
                self.use_local_llm = False
        else:
            self.llm_adapter = None
            if LocalLLMAdapter is None:
                print("⚠️ Meta-Learning Engine running in simulation mode - Local LLM Adapter not available")
        
        # Storage files
        self.performance_file = self.meta_dir / 'strategy_performance.json'
        self.patterns_file = self.meta_dir / 'learning_patterns.json'
        self.adaptations_file = self.meta_dir / 'adaptations.json'
        self.state_file = self.meta_dir / 'meta_learning_state.json'
        
        # Runtime data
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.learning_patterns: Dict[str, LearningPattern] = {}
        self.adaptation_history: deque = deque(maxlen=1000)
        self.effectiveness_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Meta-learning parameters
        self.meta_state = MetaLearningState(
            current_phase=LearningPhase.EXPLORATION,
            exploration_rate=0.3,
            exploitation_rate=0.7,
            convergence_threshold=0.95,
            adaptation_sensitivity=0.1,
            learning_rate=0.1
        )
        
        # Strategy registry
        self.available_strategies = {
            StrategyType.CODE_OPTIMIZATION: [
                "refactor_complex_functions",
                "split_long_functions", 
                "optimize_algorithms",
                "reduce_code_duplication",
                "improve_naming_conventions"
            ],
            StrategyType.PROCESS_IMPROVEMENT: [
                "parallel_execution",
                "smart_scheduling",
                "batch_processing",
                "pipeline_optimization",
                "error_handling_improvement"
            ],
            StrategyType.RESOURCE_MANAGEMENT: [
                "memory_optimization",
                "cpu_optimization",
                "disk_optimization",
                "network_optimization",
                "caching_strategies"
            ],
            StrategyType.WORKFLOW_AUTOMATION: [
                "automate_repetitive_tasks",
                "intelligent_task_routing",
                "dynamic_priority_adjustment",
                "automated_testing",
                "continuous_integration"
            ],
            StrategyType.DEPENDENCY_RESOLUTION: [
                "cycle_elimination",
                "dependency_minimization",
                "modular_architecture",
                "interface_optimization",
                "loose_coupling"
            ],
            StrategyType.PERFORMANCE_TUNING: [
                "load_balancing",
                "connection_pooling",
                "async_processing",
                "indexing_optimization",
                "query_optimization"
            ]
        }
        
        self.initialize_meta_learning()
    
    def initialize_meta_learning(self):
        """Initialize meta-learning engine"""
        
        # Load existing data
        self.load_meta_learning_data()
        
        # Initialize strategy performance tracking
        self.initialize_strategy_tracking()
        
        print(f"✅ Initialized meta-learning engine with {len(self.strategy_performance)} tracked strategies")
    
    def record_strategy_outcome(self, strategy_id: str, strategy_type: StrategyType,
                               success: bool, improvement_score: float) -> str:
        """Record the outcome of a strategy application"""
        
        # Get or create strategy performance record
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = StrategyPerformance(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                success_count=0,
                failure_count=0,
                total_attempts=0,
                average_improvement=0.0,
                success_rate=0.0,
                effectiveness_score=0.0,
                last_used=datetime.now(),
                adaptation_count=0
            )
        
        performance = self.strategy_performance[strategy_id]
        
        # Update performance metrics
        performance.total_attempts += 1
        performance.last_used = datetime.now()
        
        if success:
            performance.success_count += 1
            # Update rolling average improvement
            current_avg = performance.average_improvement
            n = performance.success_count
            performance.average_improvement = ((current_avg * (n-1)) + improvement_score) / n
        else:
            performance.failure_count += 1
        
        # Recalculate derived metrics
        performance.success_rate = performance.success_count / performance.total_attempts
        performance.effectiveness_score = self.calculate_effectiveness_score(performance)
        
        # Track effectiveness trend
        self.effectiveness_trends[strategy_id].append(performance.effectiveness_score)
        
        # Trigger meta-learning analysis
        adaptation_triggered = self.analyze_and_adapt(strategy_id, performance)
        
        # Save updated data
        self.save_meta_learning_data()
        
        outcome_id = f"outcome_{int(time.time())}_{strategy_id}"
        print(f"📊 Recorded strategy outcome: {strategy_id} ({'success' if success else 'failure'})")
        
        return outcome_id
    
    def calculate_effectiveness_score(self, performance: StrategyPerformance) -> float:
        """Calculate comprehensive effectiveness score for a strategy"""
        
        if performance.total_attempts == 0:
            return 0.0
        
        # Base score from success rate
        base_score = performance.success_rate
        
        # Bonus for positive improvement
        improvement_bonus = min(0.5, performance.average_improvement) if performance.average_improvement > 0 else 0
        
        # Penalty for low usage (strategies need sufficient data)
        usage_factor = min(1.0, performance.total_attempts / 10)
        
        # Recency factor (more recent usage is weighted higher)
        days_since_use = (datetime.now() - performance.last_used).days
        recency_factor = max(0.5, 1.0 - (days_since_use / 30))
        
        effectiveness = (base_score + improvement_bonus) * usage_factor * recency_factor
        
        return min(1.0, effectiveness)
    
    def analyze_and_adapt(self, strategy_id: str, performance: StrategyPerformance) -> bool:
        """Analyze strategy performance and trigger adaptations if needed"""
        
        adaptation_triggered = False
        
        # Check for declining performance
        if len(self.effectiveness_trends[strategy_id]) >= 5:
            recent_trend = list(self.effectiveness_trends[strategy_id])[-5:]
            trend_slope = self.calculate_trend_slope(recent_trend)
            
            if trend_slope < -self.meta_state.adaptation_sensitivity:
                adaptation_triggered = self.trigger_strategy_adaptation(
                    strategy_id, "declining_performance", trend_slope
                )
        
        # Check for consistently poor performance
        if performance.total_attempts >= 10 and performance.effectiveness_score < 0.3:
            adaptation_triggered = self.trigger_strategy_adaptation(
                strategy_id, "poor_performance", performance.effectiveness_score
            )
        
        # Check for strategies that haven't been used recently
        days_unused = (datetime.now() - performance.last_used).days
        if days_unused > 14 and performance.effectiveness_score < 0.6:
            adaptation_triggered = self.trigger_strategy_adaptation(
                strategy_id, "underutilized", days_unused
            )
        
        # Update meta-learning phase if needed
        self.update_learning_phase()
        
        return adaptation_triggered
    
    def trigger_strategy_adaptation(self, strategy_id: str, trigger_condition: str, 
                                  trigger_value: float) -> bool:
        """Trigger adaptation for a strategy"""
        
        print(f"🔄 Triggering adaptation for strategy {strategy_id}: {trigger_condition}")
        
        # Find alternative strategy
        current_performance = self.strategy_performance[strategy_id]
        alternative_strategy = self.find_alternative_strategy(current_performance.strategy_type, strategy_id)
        
        if not alternative_strategy:
            print(f"⚠️ No alternative strategy found for {strategy_id}")
            return False
        
        # Create adaptation event
        adaptation = AdaptationEvent(
            event_id=f"adapt_{int(time.time())}",
            timestamp=datetime.now(),
            trigger_condition=trigger_condition,
            old_strategy=strategy_id,
            new_strategy=alternative_strategy,
            reasoning=f"Adapting due to {trigger_condition} (value: {trigger_value})",
            expected_improvement=self.estimate_improvement_potential(alternative_strategy)
        )
        
        self.adaptation_history.append(adaptation)
        
        # Update adaptation count
        current_performance.adaptation_count += 1
        
        print(f"✅ Adaptation triggered: {strategy_id} → {alternative_strategy}")
        
        return True
    
    def find_alternative_strategy(self, strategy_type: StrategyType, 
                                current_strategy: str) -> Optional[str]:
        """Find the best alternative strategy of the same type"""
        
        available = self.available_strategies.get(strategy_type, [])
        alternatives = [s for s in available if s != current_strategy]
        
        if not alternatives:
            return None
        
        # Rank alternatives by effectiveness
        ranked_alternatives = []
        for alt in alternatives:
            if alt in self.strategy_performance:
                score = self.strategy_performance[alt].effectiveness_score
            else:
                score = 0.5  # Unknown strategies get neutral score
            
            ranked_alternatives.append((alt, score))
        
        # Sort by effectiveness (descending)
        ranked_alternatives.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_alternatives[0][0] if ranked_alternatives else alternatives[0]
    
    def estimate_improvement_potential(self, strategy_id: str) -> float:
        """Estimate potential improvement from switching to a strategy"""
        
        if strategy_id in self.strategy_performance:
            performance = self.strategy_performance[strategy_id]
            return performance.effectiveness_score * performance.average_improvement
        else:
            # Unknown strategy - use exploration bonus
            return 0.3 * self.meta_state.exploration_rate
    
    def calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate the slope of a trend (simple linear regression)"""
        
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope using least squares
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def update_learning_phase(self):
        """Update the current learning phase based on system state"""
        
        # Calculate overall system performance
        if self.strategy_performance:
            avg_effectiveness = statistics.mean(
                perf.effectiveness_score for perf in self.strategy_performance.values()
            )
        else:
            avg_effectiveness = 0.0
        
        # Determine appropriate phase
        current_phase = self.meta_state.current_phase
        
        if avg_effectiveness < 0.4:
            new_phase = LearningPhase.EXPLORATION
            self.meta_state.exploration_rate = 0.5
            self.meta_state.exploitation_rate = 0.5
        elif avg_effectiveness < 0.7:
            new_phase = LearningPhase.EXPLOITATION
            self.meta_state.exploration_rate = 0.2
            self.meta_state.exploitation_rate = 0.8
        elif avg_effectiveness < self.meta_state.convergence_threshold:
            new_phase = LearningPhase.REFINEMENT
            self.meta_state.exploration_rate = 0.1
            self.meta_state.exploitation_rate = 0.9
        else:
            new_phase = LearningPhase.CONVERGENCE
            self.meta_state.exploration_rate = 0.05
            self.meta_state.exploitation_rate = 0.95
        
        if new_phase != current_phase:
            print(f"🔄 Learning phase transition: {current_phase.value} → {new_phase.value}")
            self.meta_state.current_phase = new_phase
    
    def discover_learning_patterns(self) -> List[LearningPattern]:
        """Discover patterns in strategy effectiveness"""
        
        patterns = []
        
        # Pattern: High-performing strategy types
        type_performance = defaultdict(list)
        for perf in self.strategy_performance.values():
            type_performance[perf.strategy_type].append(perf.effectiveness_score)
        
        for strategy_type, scores in type_performance.items():
            if len(scores) >= 3:
                avg_score = statistics.mean(scores)
                if avg_score > 0.7:
                    pattern = LearningPattern(
                        pattern_id=f"high_perf_{strategy_type.value}",
                        pattern_type="high_performance_type",
                        description=f"{strategy_type.value} strategies show consistently high performance",
                        conditions={"strategy_type": strategy_type.value, "min_effectiveness": 0.7},
                        recommended_strategies=self.available_strategies[strategy_type],
                        confidence=min(1.0, avg_score),
                        frequency=len(scores),
                        discovered_at=datetime.now()
                    )
                    patterns.append(pattern)
        
        # Pattern: Successful adaptation sequences
        if len(self.adaptation_history) >= 5:
            recent_adaptations = list(self.adaptation_history)[-10:]
            successful_sequences = []
            
            for adaptation in recent_adaptations:
                old_strategy = adaptation.old_strategy
                new_strategy = adaptation.new_strategy
                
                # Check if the new strategy performed better
                if (new_strategy in self.strategy_performance and 
                    old_strategy in self.strategy_performance):
                    
                    old_score = self.strategy_performance[old_strategy].effectiveness_score
                    new_score = self.strategy_performance[new_strategy].effectiveness_score
                    
                    if new_score > old_score:
                        successful_sequences.append((old_strategy, new_strategy))
            
            if successful_sequences:
                pattern = LearningPattern(
                    pattern_id="successful_adaptations",
                    pattern_type="adaptation_sequence",
                    description="Successful strategy adaptation patterns identified",
                    conditions={"adaptation_success_rate": len(successful_sequences) / len(recent_adaptations)},
                    recommended_strategies=[seq[1] for seq in successful_sequences],
                    confidence=len(successful_sequences) / len(recent_adaptations),
                    frequency=len(successful_sequences),
                    discovered_at=datetime.now()
                )
                patterns.append(pattern)
        
        # Store discovered patterns
        for pattern in patterns:
            self.learning_patterns[pattern.pattern_id] = pattern
        
        print(f"🔍 Discovered {len(patterns)} new learning patterns")
        
        return patterns
    
    def recommend_strategies(self, context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Recommend strategies based on current context and learning enhanced with Local LLM reasoning"""
        
        recommendations = []
        
        # Use Local LLM for enhanced strategy recommendation if available
        if self.use_local_llm and self.llm_adapter:
            llm_recommendations = self.generate_llm_strategy_recommendations(context)
            recommendations.extend(llm_recommendations)
        
        # Get strategy type from context
        target_type = context.get('strategy_type')
        if isinstance(target_type, str):
            target_type = StrategyType(target_type)
        
        # If no specific type, consider all types
        strategy_types = [target_type] if target_type else list(StrategyType)
        
        for strategy_type in strategy_types:
            available = self.available_strategies.get(strategy_type, [])
            
            for strategy in available:
                score = self.calculate_recommendation_score(strategy, context)
                recommendations.append((strategy, score))
        
        # Sort by score (descending) and remove duplicates
        unique_recommendations = {}
        for strategy, score in recommendations:
            if strategy not in unique_recommendations or score > unique_recommendations[strategy]:
                unique_recommendations[strategy] = score
        
        sorted_recommendations = sorted(unique_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_recommendations[:5]  # Top 5 recommendations
    
    def generate_llm_strategy_recommendations(self, context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Generate strategy recommendations using Local LLM reasoning"""
        
        # Prepare context for LLM analysis
        strategy_context = {
            'current_phase': self.meta_state.current_phase.value,
            'exploration_rate': self.meta_state.exploration_rate,
            'exploitation_rate': self.meta_state.exploitation_rate,
            'recent_adaptations': len(self.adaptation_history),
            'total_strategies': len(self.strategy_performance)
        }
        
        # Add performance data for context
        performance_summary = {}
        for strategy_id, performance in self.strategy_performance.items():
            performance_summary[strategy_id] = {
                'success_rate': performance.success_rate,
                'effectiveness': performance.effectiveness_score,
                'last_used': performance.last_used.isoformat() if performance.last_used else None
            }
        
        strategy_context['performance_summary'] = performance_summary
        strategy_context.update(context)
        
        recommendation_prompt = f"""
        As a meta-learning optimization expert, analyze the current system state and recommend the most effective improvement strategies.

        Current Meta-Learning State:
        - Learning Phase: {strategy_context['current_phase']}
        - Exploration Rate: {strategy_context['exploration_rate']:.2f}
        - Exploitation Rate: {strategy_context['exploitation_rate']:.2f}
        - Total Strategies Tracked: {strategy_context['total_strategies']}

        Available Strategy Categories:
        1. Code Optimization: refactor_complex_functions, split_long_functions, optimize_algorithms
        2. Process Improvement: parallel_execution, smart_scheduling, batch_processing
        3. Resource Management: memory_optimization, cpu_optimization, caching_strategies
        4. Workflow Automation: automate_repetitive_tasks, intelligent_task_routing
        5. Dependency Resolution: cycle_elimination, dependency_minimization
        6. Performance Tuning: load_balancing, connection_pooling

        Context: {context}

        Based on the current learning phase and system context, recommend 3-5 specific strategies with:
        - Strategy name (from the categories above)
        - Confidence score (0.0-1.0)
        - Reasoning for recommendation

        Focus on strategies that align with the current learning phase and have the highest potential for success.
        """
        
        try:
            llm_response = self.llm_adapter.reasoning_request(recommendation_prompt, context=strategy_context)
            recommendations = self._parse_llm_strategy_recommendations(llm_response)
        except Exception as e:
            print(f"⚠️ LLM strategy recommendation failed: {e}")
            recommendations = []
        
        return recommendations
    
    def _parse_llm_strategy_recommendations(self, llm_response: str) -> List[Tuple[str, float]]:
        """Parse LLM response to extract strategy recommendations"""
        
        recommendations = []
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for strategy recommendations with confidence scores
            strategy_found = None
            confidence = 0.5  # Default confidence
            
            # Check against known strategies
            all_strategies = []
            for strategies in self.available_strategies.values():
                all_strategies.extend(strategies)
            
            for strategy in all_strategies:
                if strategy.replace('_', ' ') in line.lower() or strategy in line.lower():
                    strategy_found = strategy
                    break
            
            if strategy_found:
                # Extract confidence score if present
                import re
                numbers = re.findall(r'0\.\d+|\d+\.\d+', line)
                if numbers:
                    confidence = min(1.0, max(0.0, float(numbers[0])))
                    if confidence > 1.0:  # Assume it's a percentage
                        confidence = confidence / 100.0
                
                recommendations.append((strategy_found, confidence))
        
        # If no specific strategies found, extract generic recommendations
        if not recommendations:
            if 'refactor' in llm_response.lower() or 'code' in llm_response.lower():
                recommendations.append(('refactor_complex_functions', 0.6))
            if 'parallel' in llm_response.lower() or 'concurrency' in llm_response.lower():
                recommendations.append(('parallel_execution', 0.6))
            if 'memory' in llm_response.lower() or 'cache' in llm_response.lower():
                recommendations.append(('memory_optimization', 0.6))
        
        return recommendations[:3]  # Limit to top 3 LLM recommendations
    
    def calculate_recommendation_score(self, strategy: str, context: Dict[str, Any]) -> float:
        """Calculate recommendation score for a strategy"""
        
        base_score = 0.5  # Default for unknown strategies
        
        if strategy in self.strategy_performance:
            performance = self.strategy_performance[strategy]
            base_score = performance.effectiveness_score
            
            # Boost score based on current learning phase
            if self.meta_state.current_phase == LearningPhase.EXPLORATION:
                # Favor less-used strategies
                usage_factor = 1.0 - min(1.0, performance.total_attempts / 20)
                base_score += usage_factor * 0.2
            elif self.meta_state.current_phase == LearningPhase.EXPLOITATION:
                # Favor high-performing strategies
                base_score *= 1.2 if performance.success_rate > 0.7 else 1.0
        else:
            # Unknown strategy - exploration bonus
            if self.meta_state.current_phase == LearningPhase.EXPLORATION:
                base_score += 0.3
        
        # Apply pattern-based bonuses
        for pattern in self.learning_patterns.values():
            if strategy in pattern.recommended_strategies:
                base_score += pattern.confidence * 0.1
        
        return min(1.0, base_score)
    
    def initialize_strategy_tracking(self):
        """Initialize tracking for all available strategies"""
        
        for strategy_type, strategies in self.available_strategies.items():
            for strategy in strategies:
                if strategy not in self.strategy_performance:
                    self.strategy_performance[strategy] = StrategyPerformance(
                        strategy_id=strategy,
                        strategy_type=strategy_type,
                        success_count=0,
                        failure_count=0,
                        total_attempts=0,
                        average_improvement=0.0,
                        success_rate=0.0,
                        effectiveness_score=0.0,
                        last_used=datetime.now(),
                        adaptation_count=0
                    )
    
    def load_meta_learning_data(self):
        """Load meta-learning data from disk"""
        try:
            # Load strategy performance
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    perf_data = json.load(f)
                
                for strategy_id, data in perf_data.items():
                    self.strategy_performance[strategy_id] = StrategyPerformance(
                        strategy_id=data['strategy_id'],
                        strategy_type=StrategyType(data['strategy_type']),
                        success_count=data['success_count'],
                        failure_count=data['failure_count'],
                        total_attempts=data['total_attempts'],
                        average_improvement=data['average_improvement'],
                        success_rate=data['success_rate'],
                        effectiveness_score=data['effectiveness_score'],
                        last_used=datetime.fromisoformat(data['last_used']),
                        adaptation_count=data['adaptation_count']
                    )
            
            # Load learning state
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                self.meta_state = MetaLearningState(
                    current_phase=LearningPhase(state_data['current_phase']),
                    exploration_rate=state_data['exploration_rate'],
                    exploitation_rate=state_data['exploitation_rate'],
                    convergence_threshold=state_data['convergence_threshold'],
                    adaptation_sensitivity=state_data['adaptation_sensitivity'],
                    learning_rate=state_data['learning_rate']
                )
            
        except Exception as e:
            print(f"⚠️ Failed to load meta-learning data: {e}")
    
    def save_meta_learning_data(self):
        """Save meta-learning data to disk"""
        try:
            # Save strategy performance
            perf_data = {}
            for strategy_id, performance in self.strategy_performance.items():
                perf_data[strategy_id] = asdict(performance)
                perf_data[strategy_id]['strategy_type'] = performance.strategy_type.value
                perf_data[strategy_id]['last_used'] = performance.last_used.isoformat()
            
            with open(self.performance_file, 'w') as f:
                json.dump(perf_data, f, indent=2)
            
            # Save learning state
            state_data = asdict(self.meta_state)
            state_data['current_phase'] = self.meta_state.current_phase.value
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
        except Exception as e:
            print(f"⚠️ Failed to save meta-learning data: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        
        if self.strategy_performance:
            avg_effectiveness = statistics.mean(
                perf.effectiveness_score for perf in self.strategy_performance.values()
            )
            best_strategy = max(
                self.strategy_performance.values(), 
                key=lambda p: p.effectiveness_score
            )
        else:
            avg_effectiveness = 0.0
            best_strategy = None
        
        summary = {
            'current_phase': self.meta_state.current_phase.value,
            'tracked_strategies': len(self.strategy_performance),
            'average_effectiveness': avg_effectiveness,
            'best_strategy': best_strategy.strategy_id if best_strategy else None,
            'adaptations_triggered': len(self.adaptation_history),
            'learning_patterns_discovered': len(self.learning_patterns),
            'exploration_rate': self.meta_state.exploration_rate,
            'exploitation_rate': self.meta_state.exploitation_rate,
            'local_llm_enabled': self.use_local_llm
        }
        
        # Add LLM adapter status if available
        if self.llm_adapter:
            adapter_status = self.llm_adapter.get_adapter_status()
            summary['llm_requests'] = adapter_status['total_requests']
            summary['llm_success_rate'] = adapter_status['success_rate']
        
        return summary
    
    def shutdown(self):
        """Shutdown the meta-learning engine gracefully"""
        
        print("🔄 Shutting down Meta-Learning Engine...")
        
        # Save all learning data
        self.save_meta_learning_data()
        
        # Shutdown Local LLM Adapter if available
        if self.llm_adapter:
            try:
                self.llm_adapter.shutdown()
            except Exception as e:
                print(f"⚠️ Error shutting down LLM adapter: {e}")
        
        print("✅ Meta-Learning Engine shutdown complete")

def main():
    """Demo of Meta-Learning Engine with Local LLM Integration"""
    print("Meta-Learning Engine Demo (Local LLM Integration)")
    print("=" * 60)
    
    meta_engine = MetaLearningEngine(use_local_llm=True)
    
    # Show initial adapter status
    if meta_engine.llm_adapter:
        adapter_status = meta_engine.llm_adapter.get_adapter_status()
        print(f"📊 LLM Adapter Status:")
        print(f"  Available models: {adapter_status['available_models']}/{adapter_status['total_models']}")
        print(f"  Model types: {', '.join(adapter_status['model_types'])}")
    
    # Demo: Simulate strategy outcomes
    print("\n📊 Simulating strategy outcomes...")
    
    # Simulate some successful strategies
    meta_engine.record_strategy_outcome(
        "refactor_complex_functions", StrategyType.CODE_OPTIMIZATION, True, 0.3
    )
    meta_engine.record_strategy_outcome(
        "parallel_execution", StrategyType.PROCESS_IMPROVEMENT, True, 0.4
    )
    meta_engine.record_strategy_outcome(
        "memory_optimization", StrategyType.RESOURCE_MANAGEMENT, True, 0.25
    )
    
    # Simulate some failures
    meta_engine.record_strategy_outcome(
        "split_long_functions", StrategyType.CODE_OPTIMIZATION, False, 0.0
    )
    meta_engine.record_strategy_outcome(
        "smart_scheduling", StrategyType.PROCESS_IMPROVEMENT, False, -0.1
    )
    
    # More outcomes to trigger adaptations
    for i in range(5):
        meta_engine.record_strategy_outcome(
            "split_long_functions", StrategyType.CODE_OPTIMIZATION, False, -0.05
        )
    
    # Discover patterns
    patterns = meta_engine.discover_learning_patterns()
    
    # Get recommendations
    recommendations = meta_engine.recommend_strategies({
        'strategy_type': 'code_optimization'
    })
    
    print(f"\n💡 Top Strategy Recommendations:")
    for strategy, score in recommendations[:3]:
        print(f"  • {strategy}: {score:.2f}")
    
    # Show learning summary
    summary = meta_engine.get_learning_summary()
    print(f"\n📈 Learning Summary:")
    print(f"  Current phase: {summary['current_phase']}")
    print(f"  Tracked strategies: {summary['tracked_strategies']}")
    print(f"  Average effectiveness: {summary['average_effectiveness']:.1%}")
    print(f"  Best strategy: {summary['best_strategy']}")
    print(f"  Adaptations triggered: {summary['adaptations_triggered']}")
    print(f"  Patterns discovered: {summary['learning_patterns_discovered']}")
    
    if summary.get('local_llm_enabled'):
        print(f"  Local LLM enabled: {summary['local_llm_enabled']}")
        if 'llm_requests' in summary:
            print(f"  LLM requests: {summary['llm_requests']}")
            print(f"  LLM success rate: {summary['llm_success_rate']:.1%}")
    
    # Graceful shutdown
    meta_engine.shutdown()
    
    print(f"\n✅ Meta-learning engine demo completed")

if __name__ == "__main__":
    main()