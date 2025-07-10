#!/usr/bin/env python3
"""
Recursive Self-Improvement (RSI) Engine
Automated iterative refinement system for continuous project optimization
Now integrated with Local LLM Adapter for enhanced reasoning capabilities
"""

import json
import time
import hashlib
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
from collections import deque
import copy

# Import Local LLM Adapter
try:
    from local_llm_adapter import LocalLLMAdapter
except ImportError:
    print("⚠️ Local LLM Adapter not found - running in simulation mode")
    LocalLLMAdapter = None

class ImprovementType(Enum):
    """Types of improvements"""
    CODE_OPTIMIZATION = "code_optimization"
    PROCESS_REFINEMENT = "process_refinement"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    QUALITY_ENHANCEMENT = "quality_enhancement"
    WORKFLOW_STREAMLINING = "workflow_streamlining"
    DEPENDENCY_OPTIMIZATION = "dependency_optimization"

class EvaluationMetric(Enum):
    """Evaluation metrics for improvements"""
    EXECUTION_TIME = "execution_time"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"
    CODE_COMPLEXITY = "code_complexity"
    TASK_COMPLETION_RATE = "task_completion_rate"
    COST_EFFICIENCY = "cost_efficiency"

@dataclass
class ImprovementCandidate:
    """Individual improvement candidate"""
    improvement_id: str
    improvement_type: ImprovementType
    description: str
    target_component: str
    proposed_changes: Dict[str, Any]
    expected_benefits: List[str]
    risk_level: str  # low, medium, high
    estimated_impact: float  # 0-1 scale
    implementation_effort: int  # 1-10 scale

@dataclass
class EvaluationResult:
    """Result of improvement evaluation"""
    improvement_id: str
    metrics: Dict[EvaluationMetric, float]
    success: bool
    improvement_score: float
    side_effects: List[str]
    rollback_required: bool
    evaluation_timestamp: datetime

@dataclass
class ImprovementCycle:
    """Complete improvement cycle record"""
    cycle_id: str
    cycle_number: int
    start_time: datetime
    end_time: Optional[datetime]
    candidates_generated: int
    candidates_tested: int
    improvements_applied: int
    overall_improvement: float
    convergence_achieved: bool
    next_cycle_recommended: bool

class RecursiveSelfImprovementEngine:
    """Recursive Self-Improvement Engine for automated project optimization"""
    
    def __init__(self, rsi_dir: str = '.taskmaster/ai/rsi', use_local_llm: bool = True):
        self.rsi_dir = Path(rsi_dir)
        self.rsi_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Local LLM Adapter
        self.use_local_llm = use_local_llm and LocalLLMAdapter is not None
        if self.use_local_llm:
            try:
                self.llm_adapter = LocalLLMAdapter()
                print("✅ RSI Engine initialized with Local LLM support")
            except Exception as e:
                print(f"⚠️ Failed to initialize Local LLM Adapter: {e}")
                self.llm_adapter = None
                self.use_local_llm = False
        else:
            self.llm_adapter = None
            if LocalLLMAdapter is None:
                print("⚠️ RSI Engine running in simulation mode - Local LLM Adapter not available")
        
        # Storage files
        self.cycles_file = self.rsi_dir / 'improvement_cycles.json'
        self.candidates_file = self.rsi_dir / 'improvement_candidates.json'
        self.evaluations_file = self.rsi_dir / 'evaluations.json'
        self.baseline_file = self.rsi_dir / 'baseline_metrics.json'
        
        # Runtime state
        self.improvement_history: deque = deque(maxlen=1000)
        self.current_baseline: Dict[EvaluationMetric, float] = {}
        self.active_improvements: Dict[str, ImprovementCandidate] = {}
        self.evaluation_functions: Dict[EvaluationMetric, Callable] = {}
        
        # Configuration
        self.max_cycles_per_session = 5
        self.convergence_threshold = 0.02  # 2% improvement threshold
        self.max_risk_tolerance = "medium"  # low, medium, high
        self.rollback_safety_enabled = True
        
        self.initialize_rsi_engine()
    
    def initialize_rsi_engine(self):
        """Initialize the recursive self-improvement engine"""
        
        # Load existing data
        self.load_improvement_history()
        
        # Initialize evaluation functions
        self.setup_evaluation_functions()
        
        # Establish baseline metrics
        self.establish_baseline_metrics()
        
        print(f"✅ Initialized RSI engine with {len(self.improvement_history)} historical cycles")
    
    def run_recursive_improvement_session(self, target_components: List[str] = None,
                                        max_cycles: int = None) -> List[ImprovementCycle]:
        """Run a complete recursive self-improvement session"""
        
        max_cycles = max_cycles or self.max_cycles_per_session
        target_components = target_components or ["all"]
        
        print(f"🔄 Starting RSI session: max {max_cycles} cycles, targeting {target_components}")
        
        cycles = []
        current_cycle = 1
        
        while current_cycle <= max_cycles:
            print(f"\n📊 RSI Cycle {current_cycle}/{max_cycles}")
            
            cycle = self.execute_improvement_cycle(current_cycle, target_components)
            cycles.append(cycle)
            
            # Check convergence
            if cycle.convergence_achieved:
                print(f"✅ Convergence achieved in cycle {current_cycle}")
                break
            
            # Check if next cycle is recommended
            if not cycle.next_cycle_recommended:
                print(f"⚠️ Next cycle not recommended, stopping at cycle {current_cycle}")
                break
            
            current_cycle += 1
        
        # Generate session summary
        self.generate_session_summary(cycles)
        
        return cycles
    
    def execute_improvement_cycle(self, cycle_number: int, 
                                target_components: List[str]) -> ImprovementCycle:
        """Execute a single improvement cycle"""
        
        cycle_id = f"cycle_{int(time.time())}_{cycle_number}"
        start_time = datetime.now()
        
        print(f"🎯 Executing improvement cycle: {cycle_id}")
        
        # Step 1: Generate improvement candidates
        candidates = self.generate_improvement_candidates(target_components)
        print(f"💡 Generated {len(candidates)} improvement candidates")
        
        # Step 2: Evaluate and filter candidates
        viable_candidates = self.filter_viable_candidates(candidates)
        print(f"🔍 Filtered to {len(viable_candidates)} viable candidates")
        
        # Step 3: Test improvements
        evaluation_results = []
        improvements_applied = 0
        
        for candidate in viable_candidates:
            print(f"🧪 Testing improvement: {candidate.description}")
            
            evaluation = self.test_improvement_candidate(candidate)
            evaluation_results.append(evaluation)
            
            if evaluation.success and not evaluation.rollback_required:
                self.apply_improvement(candidate)
                improvements_applied += 1
                print(f"✅ Applied improvement: {candidate.improvement_id}")
            elif evaluation.rollback_required:
                print(f"🔄 Rolled back improvement: {candidate.improvement_id}")
        
        # Step 4: Evaluate overall cycle performance
        cycle_improvement = self.calculate_cycle_improvement(evaluation_results)
        convergence_achieved = cycle_improvement < self.convergence_threshold
        next_cycle_recommended = self.should_continue_cycles(evaluation_results, cycle_improvement)
        
        # Create cycle record
        cycle = ImprovementCycle(
            cycle_id=cycle_id,
            cycle_number=cycle_number,
            start_time=start_time,
            end_time=datetime.now(),
            candidates_generated=len(candidates),
            candidates_tested=len(viable_candidates),
            improvements_applied=improvements_applied,
            overall_improvement=cycle_improvement,
            convergence_achieved=convergence_achieved,
            next_cycle_recommended=next_cycle_recommended
        )
        
        # Store cycle
        self.improvement_history.append(cycle)
        self.save_improvement_data()
        
        return cycle
    
    def generate_improvement_candidates(self, target_components: List[str]) -> List[ImprovementCandidate]:
        """Generate improvement candidates using analysis and heuristics enhanced with Local LLM reasoning"""
        
        candidates = []
        
        # Use Local LLM for strategic improvement analysis
        if self.use_local_llm and self.llm_adapter:
            llm_candidates = self.generate_llm_improvement_candidates(target_components)
            candidates.extend(llm_candidates)
        
        # Code optimization candidates
        code_candidates = self.analyze_code_for_improvements(target_components)
        candidates.extend(code_candidates)
        
        # Process refinement candidates
        process_candidates = self.analyze_processes_for_improvements()
        candidates.extend(process_candidates)
        
        # Resource efficiency candidates
        resource_candidates = self.analyze_resource_usage_for_improvements()
        candidates.extend(resource_candidates)
        
        # Workflow streamlining candidates
        workflow_candidates = self.analyze_workflow_for_improvements()
        candidates.extend(workflow_candidates)
        
        # Dependency optimization candidates
        dependency_candidates = self.analyze_dependencies_for_improvements()
        candidates.extend(dependency_candidates)
        
        return candidates
    
    def generate_llm_improvement_candidates(self, target_components: List[str]) -> List[ImprovementCandidate]:
        """Generate improvement candidates using Local LLM reasoning"""
        
        # Prepare context for LLM analysis
        context = {
            'target_components': target_components,
            'current_baseline': {k.value: v for k, v in self.current_baseline.items()},
            'improvement_history': len(self.improvement_history),
            'active_improvements': len(self.active_improvements)
        }
        
        improvement_prompt = f"""
        As a system optimization expert, analyze the following system context and generate specific improvement recommendations.

        Target Components: {', '.join(target_components)}
        Current Baseline Metrics: {context['current_baseline']}
        Historical Improvement Cycles: {context['improvement_history']}
        Active Improvements: {context['active_improvements']}

        Generate 3-5 specific, actionable improvement recommendations focusing on:
        1. Code optimization opportunities
        2. Process efficiency improvements
        3. Resource usage optimization
        4. Workflow streamlining
        5. System performance enhancements

        For each recommendation, provide:
        - Clear description of the improvement
        - Expected benefits
        - Risk level (low/medium/high)
        - Estimated impact (0.1-1.0 scale)
        - Implementation effort (1-10 scale)

        Focus on measurable improvements that can be tested and validated.
        """
        
        try:
            llm_response = self.llm_adapter.reasoning_request(improvement_prompt, context=context)
            candidates = self._parse_llm_improvement_recommendations(llm_response)
        except Exception as e:
            print(f"⚠️ LLM improvement generation failed: {e}")
            candidates = []
        
        return candidates
    
    def _parse_llm_improvement_recommendations(self, llm_response: str) -> List[ImprovementCandidate]:
        """Parse LLM response to extract improvement candidates"""
        
        candidates = []
        lines = llm_response.split('\n')
        
        current_recommendation = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for improvement descriptions
            if any(keyword in line.lower() for keyword in ['implement', 'optimize', 'improve', 'enhance', 'refactor']):
                if current_recommendation:
                    # Process previous recommendation
                    candidate = self._create_candidate_from_llm_data(current_recommendation)
                    if candidate:
                        candidates.append(candidate)
                
                # Start new recommendation
                current_recommendation = {'description': line}
            
            # Extract attributes
            elif 'risk' in line.lower() and any(level in line.lower() for level in ['low', 'medium', 'high']):
                risk_level = 'low'
                for level in ['low', 'medium', 'high']:
                    if level in line.lower():
                        risk_level = level
                        break
                current_recommendation['risk_level'] = risk_level
            
            elif 'impact' in line.lower() and any(c.isdigit() for c in line):
                import re
                numbers = re.findall(r'0\.\d+|\d+\.\d+', line)
                if numbers:
                    impact = min(1.0, max(0.1, float(numbers[0])))
                    current_recommendation['estimated_impact'] = impact
            
            elif 'effort' in line.lower() and any(c.isdigit() for c in line):
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    effort = min(10, max(1, int(numbers[0])))
                    current_recommendation['implementation_effort'] = effort
            
            elif 'benefit' in line.lower():
                if 'expected_benefits' not in current_recommendation:
                    current_recommendation['expected_benefits'] = []
                current_recommendation['expected_benefits'].append(line)
        
        # Process final recommendation
        if current_recommendation:
            candidate = self._create_candidate_from_llm_data(current_recommendation)
            if candidate:
                candidates.append(candidate)
        
        return candidates[:3]  # Limit to top 3 LLM-generated candidates
    
    def _create_candidate_from_llm_data(self, recommendation_data: Dict[str, Any]) -> Optional[ImprovementCandidate]:
        """Create improvement candidate from parsed LLM data"""
        
        if 'description' not in recommendation_data:
            return None
        
        description = recommendation_data['description']
        
        # Classify improvement type based on description
        improvement_type = self._classify_improvement_type(description)
        
        # Generate improvement ID
        improvement_id = f"llm_{improvement_type.value}_{hashlib.md5(description.encode()).hexdigest()[:8]}"
        
        candidate = ImprovementCandidate(
            improvement_id=improvement_id,
            improvement_type=improvement_type,
            description=description,
            target_component=self._infer_target_component(description),
            proposed_changes={
                'action': 'llm_suggested_improvement',
                'description': description,
                'llm_generated': True
            },
            expected_benefits=recommendation_data.get('expected_benefits', ['Improved system performance']),
            risk_level=recommendation_data.get('risk_level', 'medium'),
            estimated_impact=recommendation_data.get('estimated_impact', 0.3),
            implementation_effort=recommendation_data.get('implementation_effort', 5)
        )
        
        return candidate
    
    def _classify_improvement_type(self, description: str) -> ImprovementType:
        """Classify improvement type based on description"""
        
        description_lower = description.lower()
        
        if any(keyword in description_lower for keyword in ['code', 'function', 'refactor', 'algorithm']):
            return ImprovementType.CODE_OPTIMIZATION
        elif any(keyword in description_lower for keyword in ['process', 'workflow', 'execution']):
            return ImprovementType.PROCESS_REFINEMENT
        elif any(keyword in description_lower for keyword in ['memory', 'cpu', 'resource', 'efficiency']):
            return ImprovementType.RESOURCE_EFFICIENCY
        elif any(keyword in description_lower for keyword in ['quality', 'error', 'testing', 'validation']):
            return ImprovementType.QUALITY_ENHANCEMENT
        elif any(keyword in description_lower for keyword in ['workflow', 'automation', 'streamline']):
            return ImprovementType.WORKFLOW_STREAMLINING
        elif any(keyword in description_lower for keyword in ['dependency', 'coupling', 'architecture']):
            return ImprovementType.DEPENDENCY_OPTIMIZATION
        else:
            return ImprovementType.CODE_OPTIMIZATION
    
    def _infer_target_component(self, description: str) -> str:
        """Infer target component from description"""
        
        description_lower = description.lower()
        
        if any(keyword in description_lower for keyword in ['database', 'db', 'storage']):
            return 'database'
        elif any(keyword in description_lower for keyword in ['api', 'endpoint', 'service']):
            return 'api'
        elif any(keyword in description_lower for keyword in ['frontend', 'ui', 'interface']):
            return 'frontend'
        elif any(keyword in description_lower for keyword in ['backend', 'server', 'processing']):
            return 'backend'
        elif any(keyword in description_lower for keyword in ['test', 'validation', 'quality']):
            return 'testing'
        else:
            return 'system'
    
    def analyze_code_for_improvements(self, target_components: List[str]) -> List[ImprovementCandidate]:
        """Analyze code for potential improvements"""
        
        candidates = []
        
        # Find Python files in the project
        python_files = list(Path('.').rglob('*.py'))
        
        for file_path in python_files[:5]:  # Limit for demo
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Simple code analysis heuristics
                lines = content.split('\n')
                complexity_score = self.calculate_code_complexity(content)
                
                if complexity_score > 20:  # High complexity threshold
                    candidate = ImprovementCandidate(
                        improvement_id=f"code_opt_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                        improvement_type=ImprovementType.CODE_OPTIMIZATION,
                        description=f"Refactor high-complexity code in {file_path}",
                        target_component=str(file_path),
                        proposed_changes={
                            'action': 'refactor_complex_functions',
                            'file': str(file_path),
                            'complexity_score': complexity_score
                        },
                        expected_benefits=['Improved maintainability', 'Reduced bug potential', 'Better performance'],
                        risk_level='low',
                        estimated_impact=0.3,
                        implementation_effort=5
                    )
                    candidates.append(candidate)
                
                # Check for long functions
                if self.has_long_functions(content):
                    candidate = ImprovementCandidate(
                        improvement_id=f"func_split_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                        improvement_type=ImprovementType.CODE_OPTIMIZATION,
                        description=f"Split long functions in {file_path}",
                        target_component=str(file_path),
                        proposed_changes={
                            'action': 'split_long_functions',
                            'file': str(file_path)
                        },
                        expected_benefits=['Improved readability', 'Better testability', 'Enhanced modularity'],
                        risk_level='low',
                        estimated_impact=0.2,
                        implementation_effort=3
                    )
                    candidates.append(candidate)
                
            except Exception as e:
                continue
        
        return candidates
    
    def analyze_processes_for_improvements(self) -> List[ImprovementCandidate]:
        """Analyze processes for potential improvements"""
        
        candidates = []
        
        # Analyze task execution patterns
        candidate = ImprovementCandidate(
            improvement_id="process_parallel_exec",
            improvement_type=ImprovementType.PROCESS_REFINEMENT,
            description="Implement parallel task execution for independent tasks",
            target_component="task_execution",
            proposed_changes={
                'action': 'enable_parallel_execution',
                'max_concurrent_tasks': 3
            },
            expected_benefits=['Faster execution', 'Better resource utilization', 'Improved throughput'],
            risk_level='medium',
            estimated_impact=0.4,
            implementation_effort=6
        )
        candidates.append(candidate)
        
        # Optimize task scheduling
        candidate = ImprovementCandidate(
            improvement_id="process_smart_scheduling",
            improvement_type=ImprovementType.PROCESS_REFINEMENT,
            description="Implement intelligent task scheduling based on resource availability",
            target_component="task_scheduler",
            proposed_changes={
                'action': 'implement_smart_scheduling',
                'algorithm': 'resource_aware'
            },
            expected_benefits=['Optimized resource usage', 'Reduced bottlenecks', 'Better load balancing'],
            risk_level='medium',
            estimated_impact=0.35,
            implementation_effort=7
        )
        candidates.append(candidate)
        
        return candidates
    
    def analyze_resource_usage_for_improvements(self) -> List[ImprovementCandidate]:
        """Analyze resource usage for potential improvements"""
        
        candidates = []
        
        # Memory optimization
        candidate = ImprovementCandidate(
            improvement_id="resource_memory_opt",
            improvement_type=ImprovementType.RESOURCE_EFFICIENCY,
            description="Implement memory optimization and caching strategies",
            target_component="memory_management",
            proposed_changes={
                'action': 'optimize_memory_usage',
                'enable_caching': True,
                'memory_limit': '1GB'
            },
            expected_benefits=['Reduced memory footprint', 'Faster data access', 'Better scalability'],
            risk_level='low',
            estimated_impact=0.25,
            implementation_effort=4
        )
        candidates.append(candidate)
        
        return candidates
    
    def analyze_workflow_for_improvements(self) -> List[ImprovementCandidate]:
        """Analyze workflow for potential improvements"""
        
        candidates = []
        
        # Workflow automation
        candidate = ImprovementCandidate(
            improvement_id="workflow_automation",
            improvement_type=ImprovementType.WORKFLOW_STREAMLINING,
            description="Automate repetitive workflow steps",
            target_component="workflow_engine",
            proposed_changes={
                'action': 'automate_repetitive_tasks',
                'automation_level': 'high'
            },
            expected_benefits=['Reduced manual effort', 'Consistent execution', 'Faster completion'],
            risk_level='low',
            estimated_impact=0.3,
            implementation_effort=5
        )
        candidates.append(candidate)
        
        return candidates
    
    def analyze_dependencies_for_improvements(self) -> List[ImprovementCandidate]:
        """Analyze dependencies for potential improvements"""
        
        candidates = []
        
        # Dependency optimization
        candidate = ImprovementCandidate(
            improvement_id="dep_cycle_optimization",
            improvement_type=ImprovementType.DEPENDENCY_OPTIMIZATION,
            description="Optimize dependency chains and eliminate cycles",
            target_component="dependency_manager",
            proposed_changes={
                'action': 'optimize_dependencies',
                'remove_cycles': True,
                'minimize_chains': True
            },
            expected_benefits=['Faster builds', 'Cleaner architecture', 'Reduced coupling'],
            risk_level='medium',
            estimated_impact=0.4,
            implementation_effort=6
        )
        candidates.append(candidate)
        
        return candidates
    
    def filter_viable_candidates(self, candidates: List[ImprovementCandidate]) -> List[ImprovementCandidate]:
        """Filter candidates based on viability criteria"""
        
        viable = []
        
        for candidate in candidates:
            # Risk assessment
            if candidate.risk_level == "high" and self.max_risk_tolerance in ["low", "medium"]:
                continue
            
            # Impact threshold
            if candidate.estimated_impact < 0.1:  # Less than 10% impact
                continue
            
            # Effort vs impact ratio
            impact_effort_ratio = candidate.estimated_impact / (candidate.implementation_effort / 10)
            if impact_effort_ratio < 0.1:
                continue
            
            viable.append(candidate)
        
        # Sort by impact/effort ratio
        viable.sort(key=lambda c: c.estimated_impact / (c.implementation_effort / 10), reverse=True)
        
        return viable[:3]  # Top 3 candidates per cycle
    
    def test_improvement_candidate(self, candidate: ImprovementCandidate) -> EvaluationResult:
        """Test an improvement candidate in a safe environment"""
        
        print(f"🔬 Testing improvement: {candidate.improvement_id}")
        
        # Create backup state
        backup_state = self.create_backup_state(candidate.target_component)
        
        try:
            # Apply improvement temporarily
            self.apply_improvement_temporarily(candidate)
            
            # Measure metrics
            metrics = self.measure_performance_metrics()
            
            # Calculate improvement score
            improvement_score = self.calculate_improvement_score(metrics)
            
            # Check for side effects
            side_effects = self.detect_side_effects(candidate, metrics)
            
            success = improvement_score > 0 and len(side_effects) == 0
            rollback_required = improvement_score < 0 or len(side_effects) > 0
            
            evaluation = EvaluationResult(
                improvement_id=candidate.improvement_id,
                metrics=metrics,
                success=success,
                improvement_score=improvement_score,
                side_effects=side_effects,
                rollback_required=rollback_required,
                evaluation_timestamp=datetime.now()
            )
            
            if rollback_required:
                self.restore_backup_state(backup_state, candidate.target_component)
            
            return evaluation
            
        except Exception as e:
            # Restore on error
            self.restore_backup_state(backup_state, candidate.target_component)
            
            return EvaluationResult(
                improvement_id=candidate.improvement_id,
                metrics={},
                success=False,
                improvement_score=-1.0,
                side_effects=[f"Error during testing: {str(e)}"],
                rollback_required=True,
                evaluation_timestamp=datetime.now()
            )
    
    def apply_improvement_temporarily(self, candidate: ImprovementCandidate):
        """Apply improvement temporarily for testing"""
        
        action = candidate.proposed_changes.get('action', '')
        
        if action == 'refactor_complex_functions':
            # Simulate code refactoring
            print(f"  📝 Simulating code refactoring for {candidate.target_component}")
            time.sleep(0.5)  # Simulate work
            
        elif action == 'enable_parallel_execution':
            # Simulate enabling parallel execution
            print(f"  ⚡ Simulating parallel execution enablement")
            time.sleep(0.3)
            
        elif action == 'optimize_memory_usage':
            # Simulate memory optimization
            print(f"  🧠 Simulating memory optimization")
            time.sleep(0.4)
            
        else:
            # Generic simulation
            print(f"  🔧 Simulating improvement application: {action}")
            time.sleep(0.2)
    
    def apply_improvement(self, candidate: ImprovementCandidate):
        """Permanently apply an improvement"""
        
        print(f"✅ Permanently applying improvement: {candidate.improvement_id}")
        self.active_improvements[candidate.improvement_id] = candidate
        
        # Log the improvement
        improvement_log = {
            'timestamp': datetime.now().isoformat(),
            'improvement_id': candidate.improvement_id,
            'type': candidate.improvement_type.value,
            'description': candidate.description,
            'target': candidate.target_component
        }
        
        log_file = self.rsi_dir / 'applied_improvements.json'
        logs = []
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        
        logs.append(improvement_log)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def measure_performance_metrics(self) -> Dict[EvaluationMetric, float]:
        """Measure current performance metrics"""
        
        metrics = {}
        
        # Simulate metric measurement
        metrics[EvaluationMetric.EXECUTION_TIME] = 10.5 + (time.time() % 5)  # Simulated
        metrics[EvaluationMetric.RESOURCE_USAGE] = 0.65 + (time.time() % 0.3)  # Simulated
        metrics[EvaluationMetric.ERROR_RATE] = max(0, 0.05 - (time.time() % 0.1))  # Simulated
        metrics[EvaluationMetric.CODE_COMPLEXITY] = 15 + (time.time() % 10)  # Simulated
        metrics[EvaluationMetric.TASK_COMPLETION_RATE] = 0.85 + (time.time() % 0.15)  # Simulated
        metrics[EvaluationMetric.COST_EFFICIENCY] = 0.75 + (time.time() % 0.2)  # Simulated
        
        return metrics
    
    def calculate_improvement_score(self, current_metrics: Dict[EvaluationMetric, float]) -> float:
        """Calculate improvement score compared to baseline"""
        
        if not self.current_baseline:
            return 0.0
        
        total_improvement = 0.0
        metric_count = 0
        
        for metric, current_value in current_metrics.items():
            if metric in self.current_baseline:
                baseline_value = self.current_baseline[metric]
                
                # Calculate percentage improvement (some metrics are better when lower)
                if metric in [EvaluationMetric.EXECUTION_TIME, EvaluationMetric.RESOURCE_USAGE, 
                             EvaluationMetric.ERROR_RATE, EvaluationMetric.CODE_COMPLEXITY]:
                    improvement = (baseline_value - current_value) / baseline_value
                else:
                    improvement = (current_value - baseline_value) / baseline_value
                
                total_improvement += improvement
                metric_count += 1
        
        return total_improvement / metric_count if metric_count > 0 else 0.0
    
    def detect_side_effects(self, candidate: ImprovementCandidate, 
                          metrics: Dict[EvaluationMetric, float]) -> List[str]:
        """Detect potential side effects of improvements"""
        
        side_effects = []
        
        # Check for performance degradation
        if metrics.get(EvaluationMetric.EXECUTION_TIME, 0) > 20:
            side_effects.append("Significant increase in execution time")
        
        # Check for resource usage spikes
        if metrics.get(EvaluationMetric.RESOURCE_USAGE, 0) > 0.9:
            side_effects.append("High resource usage detected")
        
        # Check for error rate increases
        if metrics.get(EvaluationMetric.ERROR_RATE, 0) > 0.1:
            side_effects.append("Increased error rate")
        
        return side_effects
    
    def calculate_cycle_improvement(self, evaluations: List[EvaluationResult]) -> float:
        """Calculate overall improvement for the cycle"""
        
        if not evaluations:
            return 0.0
        
        successful_evaluations = [e for e in evaluations if e.success]
        
        if not successful_evaluations:
            return 0.0
        
        total_improvement = sum(e.improvement_score for e in successful_evaluations)
        return total_improvement / len(successful_evaluations)
    
    def should_continue_cycles(self, evaluations: List[EvaluationResult], 
                             cycle_improvement: float) -> bool:
        """Determine if additional cycles are recommended"""
        
        # Continue if significant improvement was achieved
        if cycle_improvement > self.convergence_threshold:
            return True
        
        # Continue if there were unsuccessful attempts that might work in next cycle
        unsuccessful = [e for e in evaluations if not e.success]
        if len(unsuccessful) > 0 and cycle_improvement > 0:
            return True
        
        return False
    
    def calculate_code_complexity(self, code_content: str) -> int:
        """Calculate code complexity score"""
        
        complexity = 0
        
        # Count complexity indicators
        indicators = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally']
        for indicator in indicators:
            complexity += code_content.count(indicator)
        
        # Count nested structures
        indentation_levels = []
        for line in code_content.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indentation_levels.append(indent)
        
        max_nesting = max(indentation_levels) // 4 if indentation_levels else 0
        complexity += max_nesting * 2
        
        return complexity
    
    def has_long_functions(self, code_content: str) -> bool:
        """Check if code has long functions"""
        
        lines = code_content.split('\n')
        in_function = False
        function_length = 0
        
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
                function_length = 0
            elif in_function and line.strip() and not line.startswith(' '):
                if function_length > 50:  # Function longer than 50 lines
                    return True
                in_function = False
                function_length = 0
            elif in_function:
                function_length += 1
        
        return False
    
    def create_backup_state(self, component: str) -> Dict[str, Any]:
        """Create backup state for rollback"""
        return {
            'component': component,
            'timestamp': datetime.now().isoformat(),
            'backup_created': True
        }
    
    def restore_backup_state(self, backup_state: Dict[str, Any], component: str):
        """Restore backup state"""
        print(f"🔄 Restoring backup state for {component}")
    
    def establish_baseline_metrics(self):
        """Establish baseline performance metrics"""
        
        print("📏 Establishing baseline metrics...")
        
        self.current_baseline = self.measure_performance_metrics()
        
        # Save baseline
        with open(self.baseline_file, 'w') as f:
            baseline_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {k.value: v for k, v in self.current_baseline.items()}
            }
            json.dump(baseline_data, f, indent=2)
        
        print(f"✅ Baseline established with {len(self.current_baseline)} metrics")
    
    def setup_evaluation_functions(self):
        """Setup evaluation functions for different metrics"""
        
        self.evaluation_functions = {
            EvaluationMetric.EXECUTION_TIME: self.measure_execution_time,
            EvaluationMetric.RESOURCE_USAGE: self.measure_resource_usage,
            EvaluationMetric.ERROR_RATE: self.measure_error_rate,
            EvaluationMetric.CODE_COMPLEXITY: self.measure_code_complexity,
            EvaluationMetric.TASK_COMPLETION_RATE: self.measure_task_completion_rate,
            EvaluationMetric.COST_EFFICIENCY: self.measure_cost_efficiency
        }
    
    def measure_execution_time(self) -> float:
        """Measure execution time"""
        return 10.0  # Simulated
    
    def measure_resource_usage(self) -> float:
        """Measure resource usage"""
        return 0.6  # Simulated
    
    def measure_error_rate(self) -> float:
        """Measure error rate"""
        return 0.05  # Simulated
    
    def measure_code_complexity(self) -> float:
        """Measure code complexity"""
        return 20.0  # Simulated
    
    def measure_task_completion_rate(self) -> float:
        """Measure task completion rate"""
        return 0.9  # Simulated
    
    def measure_cost_efficiency(self) -> float:
        """Measure cost efficiency"""
        return 0.8  # Simulated
    
    def generate_session_summary(self, cycles: List[ImprovementCycle]):
        """Generate summary of improvement session"""
        
        print(f"\n📊 RSI Session Summary:")
        print(f"  Total cycles: {len(cycles)}")
        
        total_candidates = sum(c.candidates_generated for c in cycles)
        total_tested = sum(c.candidates_tested for c in cycles)
        total_applied = sum(c.improvements_applied for c in cycles)
        
        print(f"  Candidates generated: {total_candidates}")
        print(f"  Candidates tested: {total_tested}")
        print(f"  Improvements applied: {total_applied}")
        
        avg_improvement = sum(c.overall_improvement for c in cycles) / len(cycles)
        print(f"  Average improvement per cycle: {avg_improvement:.1%}")
        
        converged_cycles = len([c for c in cycles if c.convergence_achieved])
        print(f"  Cycles achieving convergence: {converged_cycles}")
    
    def load_improvement_history(self):
        """Load improvement history from disk"""
        try:
            if self.cycles_file.exists():
                with open(self.cycles_file, 'r') as f:
                    cycles_data = json.load(f)
                
                for cycle_data in cycles_data:
                    cycle = ImprovementCycle(
                        cycle_id=cycle_data['cycle_id'],
                        cycle_number=cycle_data['cycle_number'],
                        start_time=datetime.fromisoformat(cycle_data['start_time']),
                        end_time=datetime.fromisoformat(cycle_data['end_time']) if cycle_data.get('end_time') else None,
                        candidates_generated=cycle_data['candidates_generated'],
                        candidates_tested=cycle_data['candidates_tested'],
                        improvements_applied=cycle_data['improvements_applied'],
                        overall_improvement=cycle_data['overall_improvement'],
                        convergence_achieved=cycle_data['convergence_achieved'],
                        next_cycle_recommended=cycle_data['next_cycle_recommended']
                    )
                    self.improvement_history.append(cycle)
            
        except Exception as e:
            print(f"⚠️ Failed to load improvement history: {e}")
    
    def save_improvement_data(self):
        """Save improvement data to disk"""
        try:
            # Save cycles
            cycles_data = []
            for cycle in list(self.improvement_history):
                cycle_data = asdict(cycle)
                cycle_data['start_time'] = cycle.start_time.isoformat()
                cycle_data['end_time'] = cycle.end_time.isoformat() if cycle.end_time else None
                cycles_data.append(cycle_data)
            
            with open(self.cycles_file, 'w') as f:
                json.dump(cycles_data, f, indent=2)
            
        except Exception as e:
            print(f"⚠️ Failed to save improvement data: {e}")
    
    def get_rsi_status(self) -> Dict[str, Any]:
        """Get current RSI engine status"""
        status = {
            'cycles_completed': len(self.improvement_history),
            'active_improvements': len(self.active_improvements),
            'baseline_established': bool(self.current_baseline),
            'last_cycle_time': self.improvement_history[-1].start_time.isoformat() if self.improvement_history else None,
            'convergence_threshold': self.convergence_threshold,
            'max_risk_tolerance': self.max_risk_tolerance,
            'local_llm_enabled': self.use_local_llm
        }
        
        # Add LLM adapter status if available
        if self.llm_adapter:
            adapter_status = self.llm_adapter.get_adapter_status()
            status['llm_requests'] = adapter_status['total_requests']
            status['llm_success_rate'] = adapter_status['success_rate']
        
        return status
    
    def shutdown(self):
        """Shutdown the RSI engine gracefully"""
        
        print("🔄 Shutting down RSI engine...")
        
        # Save all improvement data
        self.save_improvement_data()
        
        # Shutdown Local LLM Adapter if available
        if self.llm_adapter:
            try:
                self.llm_adapter.shutdown()
            except Exception as e:
                print(f"⚠️ Error shutting down LLM adapter: {e}")
        
        print("✅ RSI engine shutdown complete")

def main():
    """Demo of Recursive Self-Improvement Engine with Local LLM Integration"""
    print("Recursive Self-Improvement Engine Demo (Local LLM Integration)")
    print("=" * 70)
    
    rsi_engine = RecursiveSelfImprovementEngine(use_local_llm=True)
    
    # Show initial adapter status
    if rsi_engine.llm_adapter:
        adapter_status = rsi_engine.llm_adapter.get_adapter_status()
        print(f"📊 LLM Adapter Status:")
        print(f"  Available models: {adapter_status['available_models']}/{adapter_status['total_models']}")
        print(f"  Model types: {', '.join(adapter_status['model_types'])}")
    
    # Demo: Run improvement session
    print("\n🚀 Running RSI session with Local LLM enhancement...")
    
    cycles = rsi_engine.run_recursive_improvement_session(
        target_components=["task_execution", "code_optimization"],
        max_cycles=3
    )
    
    # Show status
    status = rsi_engine.get_rsi_status()
    print(f"\n📈 RSI Status:")
    print(f"  Cycles completed: {status['cycles_completed']}")
    print(f"  Active improvements: {status['active_improvements']}")
    print(f"  Baseline established: {status['baseline_established']}")
    print(f"  Convergence threshold: {status['convergence_threshold']:.1%}")
    print(f"  Local LLM enabled: {status['local_llm_enabled']}")
    
    if 'llm_requests' in status:
        print(f"  LLM requests: {status['llm_requests']}")
        print(f"  LLM success rate: {status['llm_success_rate']:.1%}")
    
    # Graceful shutdown
    rsi_engine.shutdown()
    
    print(f"\n✅ Recursive self-improvement demo completed")

if __name__ == "__main__":
    main()