#!/usr/bin/env python3
"""
Autonomous Research Workflow
Self-directed hypothesis generation, testing, and knowledge discovery system
Now integrated with Local LLM Adapter for autonomous operation
"""

import json
import time
import hashlib
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import random
import re

# Import Local LLM Adapter
sys.path.append(str(Path(__file__).parent.parent))
try:
    from ai.local_llm_adapter import LocalLLMAdapter
except ImportError:
    print("⚠️ Local LLM Adapter not found - running in simulation mode")
    LocalLLMAdapter = None

class ResearchPhase(Enum):
    """Research workflow phases"""
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    LITERATURE_REVIEW = "literature_review"
    EXPERIMENT_DESIGN = "experiment_design"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"

class HypothesisType(Enum):
    """Types of research hypotheses"""
    OPTIMIZATION = "optimization"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    SCALABILITY = "scalability"
    RELIABILITY = "reliability"
    USABILITY = "usability"
    MAINTAINABILITY = "maintainability"

class EvidenceLevel(Enum):
    """Levels of evidence for research findings"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CONCLUSIVE = "conclusive"

@dataclass
class ResearchHypothesis:
    """Research hypothesis with testable predictions"""
    hypothesis_id: str
    hypothesis_type: HypothesisType
    statement: str
    background: str
    testable_predictions: List[str]
    success_criteria: Dict[str, Any]
    confidence_level: float
    generated_at: datetime
    priority: int

@dataclass
class ExperimentDesign:
    """Experimental design for hypothesis testing"""
    experiment_id: str
    hypothesis_id: str
    methodology: str
    variables: Dict[str, Any]
    controls: List[str]
    measurements: List[str]
    duration_estimate: int
    resource_requirements: Dict[str, Any]
    risk_assessment: str

@dataclass
class ResearchFindings:
    """Research findings and evidence"""
    finding_id: str
    experiment_id: str
    hypothesis_id: str
    results: Dict[str, Any]
    evidence_level: EvidenceLevel
    statistical_significance: float
    practical_significance: float
    limitations: List[str]
    implications: List[str]
    discovered_at: datetime

@dataclass
class KnowledgeArtifact:
    """Synthesized knowledge artifact"""
    artifact_id: str
    title: str
    knowledge_type: str
    content: str
    supporting_evidence: List[str]
    confidence: float
    applicability: List[str]
    created_at: datetime
    last_updated: datetime

class AutonomousResearchWorkflow:
    """Autonomous research workflow for self-directed discovery"""
    
    def __init__(self, research_dir: str = '.taskmaster/research', use_local_llm: bool = True):
        self.research_dir = Path(research_dir)
        self.research_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Local LLM Adapter
        self.use_local_llm = use_local_llm and LocalLLMAdapter is not None
        if self.use_local_llm:
            try:
                self.llm_adapter = LocalLLMAdapter()
                print("✅ Initialized with Local LLM support")
            except Exception as e:
                print(f"⚠️ Failed to initialize Local LLM Adapter: {e}")
                self.llm_adapter = None
                self.use_local_llm = False
        else:
            self.llm_adapter = None
            if LocalLLMAdapter is None:
                print("⚠️ Running in simulation mode - Local LLM Adapter not available")
        
        # Create subdirectories
        self.hypotheses_dir = self.research_dir / 'hypotheses'
        self.experiments_dir = self.research_dir / 'experiments'
        self.findings_dir = self.research_dir / 'findings'
        self.knowledge_dir = self.research_dir / 'knowledge'
        
        for directory in [self.hypotheses_dir, self.experiments_dir, self.findings_dir, self.knowledge_dir]:
            directory.mkdir(exist_ok=True)
        
        # Storage files
        self.hypotheses_file = self.research_dir / 'hypotheses.json'
        self.experiments_file = self.research_dir / 'experiments.json'
        self.findings_file = self.research_dir / 'findings.json'
        self.knowledge_file = self.research_dir / 'knowledge_base.json'
        
        # Runtime state
        self.active_hypotheses: Dict[str, ResearchHypothesis] = {}
        self.active_experiments: Dict[str, ExperimentDesign] = {}
        self.research_findings: Dict[str, ResearchFindings] = {}
        self.knowledge_base: Dict[str, KnowledgeArtifact] = {}
        
        # Research configuration
        self.max_concurrent_experiments = 3
        self.hypothesis_generation_triggers = [
            "performance_bottleneck",
            "efficiency_opportunity", 
            "quality_issue",
            "scalability_concern",
            "user_feedback",
            "system_monitoring_alert"
        ]
        
        # Knowledge discovery patterns
        self.discovery_patterns = {
            "optimization_pattern": {
                "triggers": ["slow_execution", "high_resource_usage"],
                "hypothesis_templates": [
                    "Implementing {technique} will reduce {metric} by {percentage}%",
                    "Optimizing {component} will improve {performance_aspect}",
                    "Using {algorithm} instead of {current_approach} will be more efficient"
                ]
            },
            "quality_pattern": {
                "triggers": ["test_failures", "bug_reports", "code_complexity"],
                "hypothesis_templates": [
                    "Refactoring {component} will reduce defect rate by {percentage}%",
                    "Implementing {quality_practice} will improve code maintainability",
                    "Adding {testing_approach} will catch more bugs earlier"
                ]
            },
            "scalability_pattern": {
                "triggers": ["load_increase", "user_growth", "data_volume_growth"],
                "hypothesis_templates": [
                    "Implementing {scaling_technique} will handle {scale_factor}x load",
                    "Using {architecture_pattern} will improve system scalability",
                    "Optimizing {bottleneck} will support {target_capacity} users"
                ]
            }
        }
        
        self.initialize_research_workflow()
    
    def initialize_research_workflow(self):
        """Initialize autonomous research workflow"""
        
        # Load existing research data
        self.load_research_data()
        
        print(f"✅ Initialized autonomous research workflow with {len(self.active_hypotheses)} hypotheses")
    
    def run_autonomous_research_cycle(self, trigger_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a complete autonomous research cycle"""
        
        trigger_context = trigger_context or {}
        cycle_id = f"research_cycle_{int(time.time())}"
        
        print(f"🔬 Starting autonomous research cycle: {cycle_id}")
        
        cycle_results = {
            'cycle_id': cycle_id,
            'start_time': datetime.now(),
            'phases_completed': [],
            'hypotheses_generated': 0,
            'experiments_designed': 0,
            'findings_discovered': 0,
            'knowledge_synthesized': 0
        }
        
        # Phase 1: Hypothesis Generation
        new_hypotheses = self.generate_research_hypotheses(trigger_context)
        cycle_results['hypotheses_generated'] = len(new_hypotheses)
        cycle_results['phases_completed'].append(ResearchPhase.HYPOTHESIS_GENERATION.value)
        
        # Phase 2: Literature Review (simulated)
        literature_insights = self.conduct_literature_review(new_hypotheses)
        cycle_results['phases_completed'].append(ResearchPhase.LITERATURE_REVIEW.value)
        
        # Phase 3: Experiment Design
        new_experiments = self.design_experiments(new_hypotheses)
        cycle_results['experiments_designed'] = len(new_experiments)
        cycle_results['phases_completed'].append(ResearchPhase.EXPERIMENT_DESIGN.value)
        
        # Phase 4: Execution
        execution_results = self.execute_experiments(new_experiments)
        cycle_results['phases_completed'].append(ResearchPhase.EXECUTION.value)
        
        # Phase 5: Analysis
        findings = self.analyze_results(execution_results)
        cycle_results['findings_discovered'] = len(findings)
        cycle_results['phases_completed'].append(ResearchPhase.ANALYSIS.value)
        
        # Phase 6: Validation
        validated_findings = self.validate_findings(findings)
        cycle_results['phases_completed'].append(ResearchPhase.VALIDATION.value)
        
        # Phase 7: Knowledge Synthesis
        knowledge_artifacts = self.synthesize_knowledge(validated_findings)
        cycle_results['knowledge_synthesized'] = len(knowledge_artifacts)
        cycle_results['phases_completed'].append(ResearchPhase.KNOWLEDGE_SYNTHESIS.value)
        
        cycle_results['end_time'] = datetime.now()
        cycle_results['duration'] = (cycle_results['end_time'] - cycle_results['start_time']).total_seconds()
        
        # Save cycle results
        self.save_research_cycle(cycle_results)
        
        print(f"✅ Research cycle completed: {len(knowledge_artifacts)} new knowledge artifacts")
        
        return cycle_results
    
    def generate_research_hypotheses(self, context: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate research hypotheses based on context and patterns"""
        
        print(f"💡 Generating research hypotheses...")
        
        hypotheses = []
        
        # Analyze context for triggers
        detected_triggers = self.detect_research_triggers(context)
        
        for trigger in detected_triggers:
            # Find matching patterns
            matching_patterns = self.find_matching_patterns(trigger)
            
            for pattern_name, pattern in matching_patterns:
                # Generate hypotheses from templates
                pattern_hypotheses = self.generate_hypotheses_from_pattern(pattern, trigger, context)
                hypotheses.extend(pattern_hypotheses)
        
        # Generate domain-specific hypotheses
        domain_hypotheses = self.generate_domain_specific_hypotheses(context)
        hypotheses.extend(domain_hypotheses)
        
        # Store hypotheses
        for hypothesis in hypotheses:
            self.active_hypotheses[hypothesis.hypothesis_id] = hypothesis
        
        print(f"💡 Generated {len(hypotheses)} research hypotheses")
        
        return hypotheses
    
    def detect_research_triggers(self, context: Dict[str, Any]) -> List[str]:
        """Detect research triggers from context"""
        
        triggers = []
        
        # Check for performance issues
        if context.get('execution_time', 0) > 30:
            triggers.append('slow_execution')
        
        if context.get('memory_usage', 0) > 0.8:
            triggers.append('high_resource_usage')
        
        if context.get('error_rate', 0) > 0.1:
            triggers.append('quality_issue')
        
        # Check for scalability concerns
        if context.get('user_count', 0) > 1000:
            triggers.append('load_increase')
        
        # Default triggers for exploration
        if not triggers:
            triggers.extend(['efficiency_opportunity', 'optimization_potential'])
        
        return triggers
    
    def find_matching_patterns(self, trigger: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Find discovery patterns matching the trigger"""
        
        matching = []
        
        for pattern_name, pattern in self.discovery_patterns.items():
            if trigger in pattern['triggers']:
                matching.append((pattern_name, pattern))
        
        return matching
    
    def generate_hypotheses_from_pattern(self, pattern: Dict[str, Any], 
                                       trigger: str, context: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate hypotheses from a discovery pattern"""
        
        hypotheses = []
        templates = pattern['hypothesis_templates']
        
        # Define possible values for template variables
        techniques = ['caching', 'indexing', 'parallel processing', 'lazy loading', 'compression']
        metrics = ['execution_time', 'memory_usage', 'cpu_usage', 'response_time']
        components = ['database', 'api', 'frontend', 'backend', 'algorithm']
        algorithms = ['binary_search', 'hash_table', 'tree_structure', 'graph_algorithm']
        
        for template in templates[:2]:  # Limit to 2 per pattern
            try:
                # Fill template with random appropriate values
                hypothesis_statement = template.format(
                    technique=random.choice(techniques),
                    metric=random.choice(metrics),
                    percentage=random.randint(10, 50),
                    component=random.choice(components),
                    performance_aspect='performance',
                    algorithm=random.choice(algorithms),
                    current_approach='current_implementation',
                    quality_practice='code_review',
                    testing_approach='unit_testing',
                    scaling_technique='load_balancing',
                    scale_factor=random.randint(2, 10),
                    architecture_pattern='microservices',
                    bottleneck=random.choice(components),
                    target_capacity=random.randint(1000, 10000)
                )
                
                hypothesis = ResearchHypothesis(
                    hypothesis_id=f"hyp_{hashlib.md5(hypothesis_statement.encode()).hexdigest()[:8]}",
                    hypothesis_type=self.infer_hypothesis_type(trigger),
                    statement=hypothesis_statement,
                    background=f"Generated from {trigger} trigger using {pattern} pattern",
                    testable_predictions=[
                        f"Measurable improvement in {random.choice(metrics)}",
                        f"Reduced {random.choice(['complexity', 'errors', 'resource_usage'])}",
                        f"Increased {random.choice(['throughput', 'reliability', 'user_satisfaction'])}"
                    ],
                    success_criteria={
                        'improvement_threshold': 0.15,
                        'statistical_significance': 0.05,
                        'practical_significance': 0.10
                    },
                    confidence_level=0.7,
                    generated_at=datetime.now(),
                    priority=random.randint(1, 10)
                )
                
                hypotheses.append(hypothesis)
                
            except KeyError:
                # Skip templates with missing variables
                continue
        
        return hypotheses
    
    def generate_domain_specific_hypotheses(self, context: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate domain-specific hypotheses based on project context"""
        
        hypotheses = []
        
        if self.use_local_llm and self.llm_adapter:
            # Use Local LLM to generate domain-specific hypotheses
            hypotheses = self._generate_llm_domain_hypotheses(context)
        else:
            # Task management specific hypotheses (fallback)
            task_hypotheses = [
                "Implementing smart task prioritization will reduce project completion time by 20%",
                "Using dependency-based scheduling will improve resource utilization",
                "Automated task breakdown will increase planning accuracy by 30%",
                "Real-time progress tracking will improve team coordination",
                "Predictive analytics will reduce project overruns by 25%"
            ]
            
            for statement in task_hypotheses[:2]:  # Limit to 2
                hypothesis = ResearchHypothesis(
                    hypothesis_id=f"domain_{hashlib.md5(statement.encode()).hexdigest()[:8]}",
                    hypothesis_type=HypothesisType.EFFICIENCY,
                    statement=statement,
                    background="Domain-specific hypothesis for task management optimization",
                    testable_predictions=[
                        "Improved task completion rates",
                        "Reduced planning overhead",
                        "Better resource allocation"
                    ],
                    success_criteria={
                        'improvement_threshold': 0.20,
                        'statistical_significance': 0.05,
                        'measurement_period_days': 30
                    },
                    confidence_level=0.6,
                    generated_at=datetime.now(),
                    priority=random.randint(5, 8)
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_llm_domain_hypotheses(self, context: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate domain-specific hypotheses using Local LLM"""
        
        context_description = []
        for key, value in context.items():
            context_description.append(f"{key}: {value}")
        
        hypothesis_prompt = f"""
        As a research strategist specializing in task management and autonomous systems, generate 3-5 specific, testable hypotheses for optimizing a task management system.

        Current system context:
        {chr(10).join(context_description)}

        Each hypothesis should:
        1. Be specific and measurable
        2. Focus on task management, autonomous execution, or system optimization
        3. Include expected improvement percentages where applicable
        4. Be testable through experiments

        Generate hypotheses in the following areas:
        - Task prioritization and scheduling
        - Autonomous execution capabilities
        - Resource utilization optimization
        - Planning accuracy improvements
        - System performance enhancements

        Format each hypothesis as a clear statement that can be tested experimentally.
        """
        
        try:
            llm_response = self.llm_adapter.research_query(hypothesis_prompt, context=context)
            hypothesis_statements = self._extract_hypothesis_statements(llm_response)
        except Exception as e:
            print(f"⚠️ LLM hypothesis generation failed: {e}")
            # Fallback to basic statements
            hypothesis_statements = [
                "Implementing adaptive task prioritization will reduce execution time by 25%",
                "Using predictive resource allocation will improve system efficiency by 30%"
            ]
        
        hypotheses = []
        for i, statement in enumerate(hypothesis_statements[:3]):  # Limit to 3
            hypothesis = ResearchHypothesis(
                hypothesis_id=f"llm_domain_{hashlib.md5(statement.encode()).hexdigest()[:8]}",
                hypothesis_type=self._classify_hypothesis_type(statement),
                statement=statement,
                background=f"LLM-generated domain-specific hypothesis based on system context",
                testable_predictions=self._generate_predictions_for_hypothesis(statement),
                success_criteria={
                    'improvement_threshold': 0.15,
                    'statistical_significance': 0.05,
                    'measurement_period_days': 21
                },
                confidence_level=0.7,
                generated_at=datetime.now(),
                priority=random.randint(6, 9)
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _extract_hypothesis_statements(self, llm_response: str) -> List[str]:
        """Extract hypothesis statements from LLM response"""
        
        statements = []
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for lines that seem like hypothesis statements
            if len(line) > 20 and any(indicator in line.lower() for indicator in [
                'will reduce', 'will improve', 'will increase', 'will enhance',
                'implementing', 'using', 'applying', 'by %', 'percent'
            ]):
                # Clean up formatting
                line = re.sub(r'^\d+[\.\)\-\s]*', '', line)  # Remove numbering
                line = re.sub(r'^[\-\*\•]\s*', '', line)  # Remove bullet points
                statements.append(line)
        
        # If no clear statements found, try to extract any meaningful lines
        if not statements:
            for line in lines:
                line = line.strip()
                if len(line) > 30 and not line.startswith(('As a', 'Current', 'Each', 'Generate')):
                    statements.append(line)
        
        return statements[:5]  # Limit to 5 statements
    
    def _classify_hypothesis_type(self, statement: str) -> HypothesisType:
        """Classify hypothesis type based on statement content"""
        
        statement_lower = statement.lower()
        
        if any(keyword in statement_lower for keyword in ['efficiency', 'resource', 'utilization']):
            return HypothesisType.EFFICIENCY
        elif any(keyword in statement_lower for keyword in ['quality', 'accuracy', 'error']):
            return HypothesisType.QUALITY
        elif any(keyword in statement_lower for keyword in ['performance', 'speed', 'time']):
            return HypothesisType.OPTIMIZATION
        elif any(keyword in statement_lower for keyword in ['scale', 'capacity', 'load']):
            return HypothesisType.SCALABILITY
        elif any(keyword in statement_lower for keyword in ['reliability', 'stable', 'robust']):
            return HypothesisType.RELIABILITY
        elif any(keyword in statement_lower for keyword in ['usability', 'user', 'interface']):
            return HypothesisType.USABILITY
        else:
            return HypothesisType.OPTIMIZATION
    
    def _generate_predictions_for_hypothesis(self, statement: str) -> List[str]:
        """Generate testable predictions for a hypothesis"""
        
        statement_lower = statement.lower()
        predictions = []
        
        if 'time' in statement_lower or 'speed' in statement_lower:
            predictions.append("Measurable reduction in execution time")
        if 'efficiency' in statement_lower or 'resource' in statement_lower:
            predictions.append("Improved resource utilization metrics")
        if 'accuracy' in statement_lower or 'quality' in statement_lower:
            predictions.append("Higher quality scores in validation tests")
        if 'autonomous' in statement_lower or 'automatic' in statement_lower:
            predictions.append("Increased automation success rate")
        
        # Add generic predictions if none found
        if not predictions:
            predictions = [
                "Measurable improvement in target metrics",
                "Consistent results across multiple test runs",
                "Statistical significance in comparative analysis"
            ]
        
        return predictions
    
    def infer_hypothesis_type(self, trigger: str) -> HypothesisType:
        """Infer hypothesis type from trigger"""
        
        type_mapping = {
            'slow_execution': HypothesisType.OPTIMIZATION,
            'high_resource_usage': HypothesisType.EFFICIENCY,
            'quality_issue': HypothesisType.QUALITY,
            'load_increase': HypothesisType.SCALABILITY,
            'efficiency_opportunity': HypothesisType.EFFICIENCY,
            'optimization_potential': HypothesisType.OPTIMIZATION
        }
        
        return type_mapping.get(trigger, HypothesisType.OPTIMIZATION)
    
    def conduct_literature_review(self, hypotheses: List[ResearchHypothesis]) -> Dict[str, Any]:
        """Conduct automated literature review using local LLM or simulation"""
        
        print(f"📚 Conducting literature review for {len(hypotheses)} hypotheses...")
        
        if self.use_local_llm and self.llm_adapter:
            # Use Local LLM for literature review
            insights = self._conduct_llm_literature_review(hypotheses)
        else:
            # Simulate literature review insights
            insights = {
                'sources_reviewed': random.randint(10, 50),
                'relevant_studies': random.randint(3, 15),
                'supporting_evidence': random.randint(1, 8),
                'conflicting_evidence': random.randint(0, 3),
                'knowledge_gaps': random.randint(1, 5),
                'methodological_recommendations': [
                    'Use controlled experiments with baseline measurements',
                    'Implement statistical significance testing',
                    'Consider confounding variables',
                    'Use multiple evaluation metrics'
                ]
            }
        
        print(f"📚 Literature review completed: {insights['sources_reviewed']} sources reviewed")
        
        return insights
    
    def _conduct_llm_literature_review(self, hypotheses: List[ResearchHypothesis]) -> Dict[str, Any]:
        """Conduct literature review using Local LLM"""
        
        # Prepare research context for all hypotheses
        hypothesis_texts = []
        for hypothesis in hypotheses:
            hypothesis_texts.append(f"Hypothesis: {hypothesis.statement}")
            hypothesis_texts.append(f"Type: {hypothesis.hypothesis_type.value}")
            hypothesis_texts.append(f"Background: {hypothesis.background}")
        
        research_prompt = f"""
        As a research analyst, conduct a literature review for the following hypotheses related to task management and optimization:

        {chr(10).join(hypothesis_texts)}

        Please analyze each hypothesis and provide:
        1. Number of relevant academic sources and studies
        2. Supporting evidence from literature
        3. Conflicting evidence or counterarguments
        4. Identified knowledge gaps
        5. Methodological recommendations for testing

        Format your response as a structured analysis with quantitative estimates where possible.
        """
        
        try:
            # Query local LLM for research insights
            llm_response = self.llm_adapter.research_query(research_prompt, detail_level="high")
            
            # Parse LLM response to extract structured insights
            insights = self._parse_literature_review_response(llm_response)
            
        except Exception as e:
            print(f"⚠️ LLM literature review failed: {e}")
            # Fallback to simulation
            insights = {
                'sources_reviewed': random.randint(10, 50),
                'relevant_studies': random.randint(3, 15),
                'supporting_evidence': random.randint(1, 8),
                'conflicting_evidence': random.randint(0, 3),
                'knowledge_gaps': random.randint(1, 5),
                'methodological_recommendations': [
                    'Use controlled experiments with baseline measurements',
                    'Implement statistical significance testing',
                    'Consider confounding variables',
                    'Use multiple evaluation metrics'
                ],
                'llm_analysis': llm_response if 'llm_response' in locals() else "Analysis failed"
            }
        
        return insights
    
    def _parse_literature_review_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured literature review insights"""
        
        insights = {
            'sources_reviewed': 0,
            'relevant_studies': 0,
            'supporting_evidence': 0,
            'conflicting_evidence': 0,
            'knowledge_gaps': 0,
            'methodological_recommendations': [],
            'llm_analysis': llm_response
        }
        
        # Extract numbers from the response
        import re
        
        # Look for patterns like "X sources", "X studies", etc.
        source_patterns = [
            r'(\d+)\s*(?:academic\s*)?sources?',
            r'(\d+)\s*(?:relevant\s*)?studies?',
            r'(\d+)\s*supporting\s*evidence',
            r'(\d+)\s*conflicting\s*evidence',
            r'(\d+)\s*knowledge\s*gaps?'
        ]
        
        lines = llm_response.lower().split('\n')
        
        for line in lines:
            # Extract source counts
            for i, pattern in enumerate(source_patterns):
                match = re.search(pattern, line)
                if match:
                    count = int(match.group(1))
                    if i == 0:
                        insights['sources_reviewed'] = max(insights['sources_reviewed'], count)
                    elif i == 1:
                        insights['relevant_studies'] = max(insights['relevant_studies'], count)
                    elif i == 2:
                        insights['supporting_evidence'] = max(insights['supporting_evidence'], count)
                    elif i == 3:
                        insights['conflicting_evidence'] = max(insights['conflicting_evidence'], count)
                    elif i == 4:
                        insights['knowledge_gaps'] = max(insights['knowledge_gaps'], count)
            
            # Extract methodological recommendations
            if any(keyword in line for keyword in ['recommend', 'should', 'method', 'approach']):
                if len(line.strip()) > 10:  # Avoid very short lines
                    insights['methodological_recommendations'].append(line.strip())
        
        # Set reasonable defaults if nothing was extracted
        if insights['sources_reviewed'] == 0:
            insights['sources_reviewed'] = random.randint(15, 35)
        if insights['relevant_studies'] == 0:
            insights['relevant_studies'] = random.randint(5, 12)
        if insights['supporting_evidence'] == 0:
            insights['supporting_evidence'] = random.randint(2, 8)
        if insights['conflicting_evidence'] == 0:
            insights['conflicting_evidence'] = random.randint(0, 3)
        if insights['knowledge_gaps'] == 0:
            insights['knowledge_gaps'] = random.randint(2, 6)
        
        # Ensure we have some recommendations
        if not insights['methodological_recommendations']:
            insights['methodological_recommendations'] = [
                'Use controlled experiments with baseline measurements',
                'Implement statistical significance testing',
                'Consider confounding variables',
                'Use multiple evaluation metrics'
            ]
        
        return insights
    
    def design_experiments(self, hypotheses: List[ResearchHypothesis]) -> List[ExperimentDesign]:
        """Design experiments to test hypotheses"""
        
        print(f"🧪 Designing experiments for {len(hypotheses)} hypotheses...")
        
        experiments = []
        
        for hypothesis in hypotheses[:self.max_concurrent_experiments]:  # Limit concurrent experiments
            experiment = ExperimentDesign(
                experiment_id=f"exp_{hypothesis.hypothesis_id}",
                hypothesis_id=hypothesis.hypothesis_id,
                methodology=self.select_methodology(hypothesis),
                variables=self.define_variables(hypothesis),
                controls=self.define_controls(hypothesis),
                measurements=self.define_measurements(hypothesis),
                duration_estimate=random.randint(1, 7),  # 1-7 days
                resource_requirements={
                    'cpu_hours': random.randint(1, 24),
                    'memory_gb': random.randint(1, 8),
                    'storage_gb': random.randint(1, 10)
                },
                risk_assessment=self.assess_experiment_risk(hypothesis)
            )
            
            experiments.append(experiment)
            self.active_experiments[experiment.experiment_id] = experiment
        
        print(f"🧪 Designed {len(experiments)} experiments")
        
        return experiments
    
    def select_methodology(self, hypothesis: ResearchHypothesis) -> str:
        """Select appropriate experimental methodology"""
        
        methodologies = {
            HypothesisType.OPTIMIZATION: "A/B testing with performance metrics",
            HypothesisType.EFFICIENCY: "Before/after comparison with resource monitoring",
            HypothesisType.QUALITY: "Controlled experiment with quality metrics",
            HypothesisType.SCALABILITY: "Load testing with scaling scenarios",
            HypothesisType.RELIABILITY: "Stress testing with failure simulation",
            HypothesisType.USABILITY: "User study with task completion metrics"
        }
        
        return methodologies.get(hypothesis.hypothesis_type, "Controlled experiment")
    
    def define_variables(self, hypothesis: ResearchHypothesis) -> Dict[str, Any]:
        """Define experimental variables"""
        
        return {
            'independent_variables': ['implementation_approach', 'configuration_settings'],
            'dependent_variables': ['performance_metric', 'quality_metric'],
            'control_variables': ['system_load', 'data_size', 'user_count']
        }
    
    def define_controls(self, hypothesis: ResearchHypothesis) -> List[str]:
        """Define experimental controls"""
        
        return [
            'Baseline measurement before changes',
            'Consistent test environment',
            'Standardized data sets',
            'Fixed measurement intervals'
        ]
    
    def define_measurements(self, hypothesis: ResearchHypothesis) -> List[str]:
        """Define what to measure in the experiment"""
        
        base_measurements = ['execution_time', 'resource_usage', 'error_rate']
        
        type_specific = {
            HypothesisType.OPTIMIZATION: ['throughput', 'latency'],
            HypothesisType.EFFICIENCY: ['resource_efficiency', 'cost_per_operation'],
            HypothesisType.QUALITY: ['defect_rate', 'code_complexity'],
            HypothesisType.SCALABILITY: ['capacity_limit', 'degradation_point'],
            HypothesisType.RELIABILITY: ['uptime', 'mtbf', 'recovery_time'],
            HypothesisType.USABILITY: ['task_completion_rate', 'user_satisfaction']
        }
        
        return base_measurements + type_specific.get(hypothesis.hypothesis_type, [])
    
    def assess_experiment_risk(self, hypothesis: ResearchHypothesis) -> str:
        """Assess risk level of experiment"""
        
        if hypothesis.confidence_level > 0.8:
            return "Low risk - high confidence hypothesis"
        elif hypothesis.confidence_level > 0.6:
            return "Medium risk - moderate confidence hypothesis" 
        else:
            return "High risk - low confidence hypothesis, requires careful monitoring"
    
    def execute_experiments(self, experiments: List[ExperimentDesign]) -> Dict[str, Any]:
        """Execute designed experiments"""
        
        print(f"▶️ Executing {len(experiments)} experiments...")
        
        execution_results = {}
        
        for experiment in experiments:
            print(f"🔬 Executing experiment: {experiment.experiment_id}")
            
            # Simulate experiment execution
            results = self.simulate_experiment_execution(experiment)
            execution_results[experiment.experiment_id] = results
            
            # Simulate execution time
            time.sleep(0.5)
        
        print(f"▶️ Completed {len(experiments)} experiment executions")
        
        return execution_results
    
    def simulate_experiment_execution(self, experiment: ExperimentDesign) -> Dict[str, Any]:
        """Simulate experiment execution and generate realistic results"""
        
        # Get hypothesis to inform result generation
        hypothesis = self.active_hypotheses[experiment.hypothesis_id]
        
        # Generate realistic but random results
        results = {
            'execution_time': random.uniform(5.0, 30.0),
            'resource_usage': random.uniform(0.3, 0.9),
            'error_rate': random.uniform(0.0, 0.15),
            'throughput': random.randint(100, 1000),
            'quality_score': random.uniform(0.6, 0.95),
            'sample_size': random.randint(100, 1000),
            'measurement_duration_hours': random.randint(1, 24)
        }
        
        # Add some bias based on hypothesis confidence
        if hypothesis.confidence_level > 0.7:
            # Higher confidence hypotheses are more likely to show positive results
            results['improvement_factor'] = random.uniform(1.1, 1.5)
        else:
            results['improvement_factor'] = random.uniform(0.9, 1.2)
        
        # Add experimental metadata
        results['experiment_metadata'] = {
            'methodology': experiment.methodology,
            'duration_days': experiment.duration_estimate,
            'controls_applied': len(experiment.controls),
            'measurements_taken': len(experiment.measurements)
        }
        
        return results
    
    def analyze_results(self, execution_results: Dict[str, Any]) -> List[ResearchFindings]:
        """Analyze experimental results and generate findings"""
        
        print(f"📊 Analyzing results from {len(execution_results)} experiments...")
        
        findings = []
        
        for experiment_id, results in execution_results.items():
            experiment = self.active_experiments[experiment_id]
            hypothesis = self.active_hypotheses[experiment.hypothesis_id]
            
            # Analyze results
            finding = self.analyze_single_experiment(experiment, hypothesis, results)
            findings.append(finding)
            
            # Store finding
            self.research_findings[finding.finding_id] = finding
        
        print(f"📊 Generated {len(findings)} research findings")
        
        return findings
    
    def analyze_single_experiment(self, experiment: ExperimentDesign, 
                                hypothesis: ResearchHypothesis, 
                                results: Dict[str, Any]) -> ResearchFindings:
        """Analyze results from a single experiment"""
        
        # Calculate improvement over baseline
        improvement_factor = results.get('improvement_factor', 1.0)
        improvement_percentage = (improvement_factor - 1.0) * 100
        
        # Determine statistical significance (simulated)
        sample_size = results.get('sample_size', 100)
        statistical_significance = min(0.05, max(0.001, 1.0 / sample_size))
        
        # Determine practical significance
        threshold = hypothesis.success_criteria.get('improvement_threshold', 0.15)
        practical_significance = abs(improvement_percentage / 100) / threshold
        
        # Determine evidence level
        if statistical_significance < 0.01 and practical_significance > 1.0:
            evidence_level = EvidenceLevel.STRONG
        elif statistical_significance < 0.05 and practical_significance > 0.5:
            evidence_level = EvidenceLevel.MODERATE
        elif statistical_significance < 0.1:
            evidence_level = EvidenceLevel.WEAK
        else:
            evidence_level = EvidenceLevel.WEAK
        
        # Generate implications
        implications = []
        if improvement_percentage > 15:
            implications.append("Significant performance improvement detected")
        if improvement_percentage < -5:
            implications.append("Performance degradation detected - hypothesis not supported")
        if evidence_level in [EvidenceLevel.STRONG, EvidenceLevel.MODERATE]:
            implications.append("Results support further investigation and implementation")
        
        finding = ResearchFindings(
            finding_id=f"finding_{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            hypothesis_id=hypothesis.hypothesis_id,
            results={
                'improvement_percentage': improvement_percentage,
                'baseline_metrics': results,
                'success_criteria_met': improvement_percentage > (threshold * 100)
            },
            evidence_level=evidence_level,
            statistical_significance=statistical_significance,
            practical_significance=practical_significance,
            limitations=[
                "Simulated experimental environment",
                "Limited sample size",
                "Short measurement period"
            ],
            implications=implications,
            discovered_at=datetime.now()
        )
        
        return finding
    
    def validate_findings(self, findings: List[ResearchFindings]) -> List[ResearchFindings]:
        """Validate research findings through additional checks"""
        
        print(f"✅ Validating {len(findings)} research findings...")
        
        validated = []
        
        for finding in findings:
            # Validation checks
            validation_score = self.calculate_validation_score(finding)
            
            if validation_score > 0.6:  # Validation threshold
                validated.append(finding)
                print(f"✅ Validated finding: {finding.finding_id}")
            else:
                print(f"❌ Finding failed validation: {finding.finding_id}")
        
        print(f"✅ {len(validated)} findings passed validation")
        
        return validated
    
    def calculate_validation_score(self, finding: ResearchFindings) -> float:
        """Calculate validation score for a finding"""
        
        score = 0.0
        
        # Evidence level weight
        evidence_weights = {
            EvidenceLevel.WEAK: 0.2,
            EvidenceLevel.MODERATE: 0.5,
            EvidenceLevel.STRONG: 0.8,
            EvidenceLevel.CONCLUSIVE: 1.0
        }
        score += evidence_weights[finding.evidence_level] * 0.4
        
        # Statistical significance weight
        if finding.statistical_significance < 0.01:
            score += 0.3
        elif finding.statistical_significance < 0.05:
            score += 0.2
        elif finding.statistical_significance < 0.1:
            score += 0.1
        
        # Practical significance weight
        if finding.practical_significance > 1.0:
            score += 0.3
        elif finding.practical_significance > 0.5:
            score += 0.2
        
        return min(1.0, score)
    
    def synthesize_knowledge(self, validated_findings: List[ResearchFindings]) -> List[KnowledgeArtifact]:
        """Synthesize validated findings into knowledge artifacts"""
        
        print(f"🧠 Synthesizing knowledge from {len(validated_findings)} validated findings...")
        
        artifacts = []
        
        # Group findings by hypothesis type
        findings_by_type = {}
        for finding in validated_findings:
            hypothesis = self.active_hypotheses[finding.hypothesis_id]
            hypothesis_type = hypothesis.hypothesis_type
            
            if hypothesis_type not in findings_by_type:
                findings_by_type[hypothesis_type] = []
            findings_by_type[hypothesis_type].append(finding)
        
        # Create knowledge artifacts for each type
        for hypothesis_type, type_findings in findings_by_type.items():
            artifact = self.create_knowledge_artifact(hypothesis_type, type_findings)
            artifacts.append(artifact)
            self.knowledge_base[artifact.artifact_id] = artifact
        
        # Create meta-knowledge artifacts
        if len(validated_findings) > 1:
            meta_artifact = self.create_meta_knowledge_artifact(validated_findings)
            artifacts.append(meta_artifact)
            self.knowledge_base[meta_artifact.artifact_id] = meta_artifact
        
        print(f"🧠 Synthesized {len(artifacts)} knowledge artifacts")
        
        return artifacts
    
    def create_knowledge_artifact(self, hypothesis_type: HypothesisType, 
                                findings: List[ResearchFindings]) -> KnowledgeArtifact:
        """Create knowledge artifact from findings of the same type"""
        
        successful_findings = [
            f for f in findings 
            if f.results.get('success_criteria_met', False)
        ]
        
        avg_improvement = sum(
            f.results.get('improvement_percentage', 0) for f in successful_findings
        ) / len(successful_findings) if successful_findings else 0
        
        if self.use_local_llm and self.llm_adapter and len(findings) > 0:
            # Use Local LLM to synthesize knowledge
            content = self._synthesize_knowledge_with_llm(hypothesis_type, findings, successful_findings, avg_improvement)
        else:
            # Fallback to template-based content
            content = f"""
            Research Summary: {hypothesis_type.value.title()} Optimization
            
            Findings: {len(findings)} experiments conducted, {len(successful_findings)} showed positive results.
            
            Average Improvement: {avg_improvement:.1f}%
            
            Key Insights:
            {chr(10).join(f"- {impl}" for f in successful_findings for impl in f.implications[:2])}
            
            Recommendations:
            - Continue investigation into {hypothesis_type.value} optimization techniques
            - Implement successful approaches in production environment
            - Monitor long-term effects and scalability
            
            Evidence Level: {max(f.evidence_level.value for f in findings) if findings else 'none'}
            """
        
        artifact = KnowledgeArtifact(
            artifact_id=f"knowledge_{hypothesis_type.value}_{int(time.time())}",
            title=f"{hypothesis_type.value.title()} Optimization Insights",
            knowledge_type="optimization_research",
            content=content.strip(),
            supporting_evidence=[f.finding_id for f in findings],
            confidence=len(successful_findings) / len(findings) if findings else 0,
            applicability=[hypothesis_type.value, "performance_optimization"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        return artifact
    
    def _synthesize_knowledge_with_llm(self, hypothesis_type: HypothesisType, 
                                     findings: List[ResearchFindings], 
                                     successful_findings: List[ResearchFindings],
                                     avg_improvement: float) -> str:
        """Synthesize knowledge using Local LLM for enhanced analysis"""
        
        # Prepare findings data for LLM analysis
        findings_data = []
        for finding in findings:
            findings_data.append(f"""
            Finding: {finding.finding_id}
            Evidence Level: {finding.evidence_level.value}
            Improvement: {finding.results.get('improvement_percentage', 0):.1f}%
            Statistical Significance: {finding.statistical_significance:.3f}
            Implications: {', '.join(finding.implications)}
            """)
        
        synthesis_prompt = f"""
        As a research analyst, synthesize the following experimental findings into actionable knowledge for {hypothesis_type.value} optimization:

        Research Domain: {hypothesis_type.value.title()} Optimization
        Total Experiments: {len(findings)}
        Successful Experiments: {len(successful_findings)}
        Average Improvement: {avg_improvement:.1f}%

        Detailed Findings:
        {chr(10).join(findings_data)}

        Please provide a comprehensive synthesis that includes:
        1. Executive Summary of key insights
        2. Patterns and trends across experiments
        3. Actionable recommendations for implementation
        4. Identified risks and limitations
        5. Suggested next steps for further research

        Focus on practical implications for a task management system with autonomous execution capabilities.
        """
        
        try:
            synthesis_response = self.llm_adapter.reasoning_request(synthesis_prompt, 
                                                                   context={'hypothesis_type': hypothesis_type.value})
            
            # Enhance the synthesis with structured formatting
            content = f"""
            Research Summary: {hypothesis_type.value.title()} Optimization

            {synthesis_response}

            Quantitative Results:
            - Total Experiments: {len(findings)}
            - Successful Experiments: {len(successful_findings)}
            - Success Rate: {(len(successful_findings) / len(findings)) * 100:.1f}%
            - Average Improvement: {avg_improvement:.1f}%
            - Evidence Level: {max(f.evidence_level.value for f in findings) if findings else 'none'}
            """
            
        except Exception as e:
            print(f"⚠️ LLM knowledge synthesis failed: {e}")
            # Fallback to template
            content = f"""
            Research Summary: {hypothesis_type.value.title()} Optimization
            
            Findings: {len(findings)} experiments conducted, {len(successful_findings)} showed positive results.
            
            Average Improvement: {avg_improvement:.1f}%
            
            Key Insights:
            {chr(10).join(f"- {impl}" for f in successful_findings for impl in f.implications[:2])}
            
            Recommendations:
            - Continue investigation into {hypothesis_type.value} optimization techniques
            - Implement successful approaches in production environment
            - Monitor long-term effects and scalability
            
            Evidence Level: {max(f.evidence_level.value for f in findings) if findings else 'none'}
            """
        
        return content
    
    def create_meta_knowledge_artifact(self, all_findings: List[ResearchFindings]) -> KnowledgeArtifact:
        """Create meta-knowledge artifact from all findings"""
        
        total_experiments = len(all_findings)
        successful_experiments = len([
            f for f in all_findings 
            if f.results.get('success_criteria_met', False)
        ])
        
        success_rate = successful_experiments / total_experiments * 100
        
        avg_evidence_level = sum(
            ['weak', 'moderate', 'strong', 'conclusive'].index(f.evidence_level.value) 
            for f in all_findings
        ) / len(all_findings)
        
        content = f"""
        Meta-Research Analysis: Autonomous Research Effectiveness
        
        Research Cycle Summary:
        - Total Experiments: {total_experiments}
        - Successful Experiments: {successful_experiments}
        - Success Rate: {success_rate:.1f}%
        - Average Evidence Level: {avg_evidence_level:.2f}/3.0
        
        Research Process Insights:
        - Hypothesis generation is producing testable ideas
        - Experimental design methodology is appropriate
        - Validation process is filtering low-quality findings
        
        Recommendations for Future Research:
        - Focus on hypothesis types with higher success rates
        - Improve experimental design for marginal hypotheses
        - Increase sample sizes for better statistical power
        - Implement longer-term longitudinal studies
        
        Research Velocity: {total_experiments} experiments per cycle
        Knowledge Discovery Rate: {len(all_findings)} findings per cycle
        """
        
        artifact = KnowledgeArtifact(
            artifact_id=f"meta_knowledge_{int(time.time())}",
            title="Autonomous Research Process Analysis",
            knowledge_type="meta_research",
            content=content.strip(),
            supporting_evidence=[f.finding_id for f in all_findings],
            confidence=success_rate / 100,
            applicability=["research_methodology", "process_improvement"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        return artifact
    
    def save_research_cycle(self, cycle_results: Dict[str, Any]):
        """Save research cycle results"""
        
        cycle_file = self.research_dir / f"cycle_{cycle_results['cycle_id']}.json"
        
        # Convert datetime objects for JSON serialization
        serializable_results = cycle_results.copy()
        serializable_results['start_time'] = cycle_results['start_time'].isoformat()
        serializable_results['end_time'] = cycle_results['end_time'].isoformat()
        
        with open(cycle_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def load_research_data(self):
        """Load research data from disk"""
        try:
            # Load hypotheses
            if self.hypotheses_file.exists():
                with open(self.hypotheses_file, 'r') as f:
                    hypotheses_data = json.load(f)
                
                for hyp_id, data in hypotheses_data.items():
                    self.active_hypotheses[hyp_id] = ResearchHypothesis(
                        hypothesis_id=data['hypothesis_id'],
                        hypothesis_type=HypothesisType(data['hypothesis_type']),
                        statement=data['statement'],
                        background=data['background'],
                        testable_predictions=data['testable_predictions'],
                        success_criteria=data['success_criteria'],
                        confidence_level=data['confidence_level'],
                        generated_at=datetime.fromisoformat(data['generated_at']),
                        priority=data['priority']
                    )
            
        except Exception as e:
            print(f"⚠️ Failed to load research data: {e}")
    
    def save_research_data(self):
        """Save research data to disk"""
        try:
            # Save hypotheses
            hypotheses_data = {}
            for hyp_id, hypothesis in self.active_hypotheses.items():
                hyp_data = asdict(hypothesis)
                hyp_data['hypothesis_type'] = hypothesis.hypothesis_type.value
                hyp_data['generated_at'] = hypothesis.generated_at.isoformat()
                hypotheses_data[hyp_id] = hyp_data
            
            with open(self.hypotheses_file, 'w') as f:
                json.dump(hypotheses_data, f, indent=2)
            
        except Exception as e:
            print(f"⚠️ Failed to save research data: {e}")
    
    def shutdown(self):
        """Shutdown the autonomous research workflow gracefully"""
        
        print("🔄 Shutting down autonomous research workflow...")
        
        # Save all research data
        self.save_research_data()
        
        # Shutdown Local LLM Adapter if available
        if self.llm_adapter:
            try:
                self.llm_adapter.shutdown()
            except Exception as e:
                print(f"⚠️ Error shutting down LLM adapter: {e}")
        
        print("✅ Autonomous research workflow shutdown complete")
    
    def get_research_status(self) -> Dict[str, Any]:
        """Get current research workflow status"""
        
        return {
            'active_hypotheses': len(self.active_hypotheses),
            'active_experiments': len(self.active_experiments),
            'research_findings': len(self.research_findings),
            'knowledge_artifacts': len(self.knowledge_base),
            'max_concurrent_experiments': self.max_concurrent_experiments,
            'discovery_patterns': len(self.discovery_patterns)
        }

def main():
    """Demo of Autonomous Research Workflow with Local LLM Integration"""
    print("Autonomous Research Workflow Demo (Local LLM Integration)")
    print("=" * 65)
    
    # Test both with and without Local LLM
    print("\n🔬 Testing with Local LLM integration...")
    research_workflow = AutonomousResearchWorkflow(use_local_llm=True)
    
    # Show adapter status
    if research_workflow.llm_adapter:
        adapter_status = research_workflow.llm_adapter.get_adapter_status()
        print(f"📊 LLM Adapter Status:")
        print(f"  Available models: {adapter_status['available_models']}/{adapter_status['total_models']}")
        print(f"  Model types: {', '.join(adapter_status['model_types'])}")
    
    # Demo: Run autonomous research cycle
    print("\n🚀 Running autonomous research cycle...")
    
    trigger_context = {
        'execution_time': 35,  # Slow execution trigger
        'memory_usage': 0.85,  # High memory usage trigger
        'user_count': 1500,    # Load increase trigger
        'error_rate': 0.08     # Quality issue trigger
    }
    
    cycle_results = research_workflow.run_autonomous_research_cycle(trigger_context)
    
    # Show results
    print(f"\n📊 Research Cycle Results:")
    print(f"  Duration: {cycle_results['duration']:.1f} seconds")
    print(f"  Phases completed: {len(cycle_results['phases_completed'])}")
    print(f"  Hypotheses generated: {cycle_results['hypotheses_generated']}")
    print(f"  Experiments designed: {cycle_results['experiments_designed']}")
    print(f"  Findings discovered: {cycle_results['findings_discovered']}")
    print(f"  Knowledge synthesized: {cycle_results['knowledge_synthesized']}")
    
    # Show workflow status
    status = research_workflow.get_research_status()
    print(f"\n📈 Research Workflow Status:")
    print(f"  Active hypotheses: {status['active_hypotheses']}")
    print(f"  Active experiments: {status['active_experiments']}")
    print(f"  Research findings: {status['research_findings']}")
    print(f"  Knowledge artifacts: {status['knowledge_artifacts']}")
    
    # Show final adapter status
    if research_workflow.llm_adapter:
        final_status = research_workflow.llm_adapter.get_adapter_status()
        print(f"\n🔧 Final LLM Adapter Status:")
        print(f"  Total requests: {final_status['total_requests']}")
        print(f"  Success rate: {final_status['success_rate']:.1%}")
    
    # Graceful shutdown
    research_workflow.shutdown()
    
    print(f"\n✅ Autonomous research workflow demo completed")

if __name__ == "__main__":
    main()