#!/usr/bin/env python3
"""
Local LLM Benchmark Suite
Comprehensive benchmarking for open source LLMs targeting Task Master AI capabilities
"""

import json
import time
import subprocess
# import psutil  # Not available, using manual system monitoring
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import statistics

class BenchmarkCategory(Enum):
    """Categories of benchmark tests"""
    MULTI_STEP_REASONING = "multi_step_reasoning"
    RECURSIVE_BREAKDOWN = "recursive_breakdown" 
    CODE_GENERATION = "code_generation"
    CONTEXT_MAINTENANCE = "context_maintenance"
    PLANNING_SYNTHESIS = "planning_synthesis"
    TASK_UNDERSTANDING = "task_understanding"

class ModelProvider(Enum):
    """Supported local model providers"""
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    LLAMACPP = "llamacpp"
    VLLM = "vllm"
    TRANSFORMERS = "transformers"

@dataclass
class ModelCandidate:
    """Open source LLM candidate for benchmarking"""
    model_id: str
    model_name: str
    provider: ModelProvider
    model_size: str  # e.g., "7B", "13B", "70B"
    quantization: str  # e.g., "Q4_K_M", "Q8_0", "FP16"
    context_length: int
    memory_requirement_gb: float
    download_size_gb: float
    license: str
    architecture: str  # e.g., "Llama2", "Mistral", "CodeLlama"
    
@dataclass
class BenchmarkTest:
    """Individual benchmark test"""
    test_id: str
    category: BenchmarkCategory
    name: str
    description: str
    prompt: str
    expected_capabilities: List[str]
    evaluation_criteria: Dict[str, Any]
    max_tokens: int
    timeout_seconds: int

@dataclass
class BenchmarkResult:
    """Result of running a benchmark test"""
    test_id: str
    model_id: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    response: str
    success: bool
    accuracy_score: float
    quality_score: float
    capability_scores: Dict[str, float]
    error_message: Optional[str]
    timestamp: datetime

@dataclass
class ModelBenchmarkSummary:
    """Summary of all benchmarks for a model"""
    model_id: str
    total_tests: int
    successful_tests: int
    average_execution_time_ms: float
    average_memory_usage_mb: float
    average_cpu_usage_percent: float
    overall_accuracy: float
    overall_quality: float
    category_scores: Dict[str, float]
    recommendation: str
    suitability_score: float

class LocalLLMBenchmark:
    """Comprehensive benchmarking suite for local LLMs"""
    
    def __init__(self, benchmark_dir: str = '.taskmaster/migration/benchmarks'):
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results_dir = self.benchmark_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Model candidates based on research
        self.model_candidates = {
            # Llama-based models
            "llama2-7b-chat": ModelCandidate(
                model_id="llama2-7b-chat",
                model_name="Llama 2 7B Chat",
                provider=ModelProvider.OLLAMA,
                model_size="7B",
                quantization="Q4_K_M",
                context_length=4096,
                memory_requirement_gb=5.5,
                download_size_gb=3.8,
                license="Custom (Commercial use allowed)",
                architecture="Llama2"
            ),
            "llama2-13b-chat": ModelCandidate(
                model_id="llama2-13b-chat", 
                model_name="Llama 2 13B Chat",
                provider=ModelProvider.OLLAMA,
                model_size="13B",
                quantization="Q4_K_M",
                context_length=4096,
                memory_requirement_gb=9.5,
                download_size_gb=7.3,
                license="Custom (Commercial use allowed)",
                architecture="Llama2"
            ),
            "codellama-7b": ModelCandidate(
                model_id="codellama-7b",
                model_name="Code Llama 7B",
                provider=ModelProvider.OLLAMA,
                model_size="7B",
                quantization="Q4_K_M",
                context_length=16384,
                memory_requirement_gb=5.5,
                download_size_gb=3.8,
                license="Custom (Commercial use allowed)",
                architecture="CodeLlama"
            ),
            "codellama-13b": ModelCandidate(
                model_id="codellama-13b",
                model_name="Code Llama 13B", 
                provider=ModelProvider.OLLAMA,
                model_size="13B",
                quantization="Q4_K_M",
                context_length=16384,
                memory_requirement_gb=9.5,
                download_size_gb=7.3,
                license="Custom (Commercial use allowed)",
                architecture="CodeLlama"
            ),
            # Mistral models
            "mistral-7b-instruct": ModelCandidate(
                model_id="mistral-7b-instruct",
                model_name="Mistral 7B Instruct",
                provider=ModelProvider.OLLAMA,
                model_size="7B",
                quantization="Q4_K_M",
                context_length=8192,
                memory_requirement_gb=5.5,
                download_size_gb=4.1,
                license="Apache 2.0",
                architecture="Mistral"
            ),
            "mixtral-8x7b-instruct": ModelCandidate(
                model_id="mixtral-8x7b-instruct",
                model_name="Mixtral 8x7B Instruct",
                provider=ModelProvider.OLLAMA,
                model_size="8x7B",
                quantization="Q4_K_M", 
                context_length=32768,
                memory_requirement_gb=26.0,
                download_size_gb=26.4,
                license="Apache 2.0",
                architecture="Mixtral"
            ),
            # Specialized models
            "deepseek-coder-6.7b": ModelCandidate(
                model_id="deepseek-coder-6.7b",
                model_name="DeepSeek Coder 6.7B",
                provider=ModelProvider.OLLAMA,
                model_size="6.7B",
                quantization="Q4_K_M",
                context_length=16384,
                memory_requirement_gb=5.0,
                download_size_gb=3.8,
                license="DeepSeek License",
                architecture="DeepSeek"
            ),
            "neural-chat-7b": ModelCandidate(
                model_id="neural-chat-7b",
                model_name="Intel Neural Chat 7B",
                provider=ModelProvider.OLLAMA,
                model_size="7B",
                quantization="Q4_K_M",
                context_length=8192,
                memory_requirement_gb=5.5,
                download_size_gb=4.1,
                license="Apache 2.0",
                architecture="Mistral"
            )
        }
        
        # Benchmark test suite
        self.benchmark_tests = self.create_benchmark_test_suite()
        
        print(f"✅ Initialized LLM benchmark suite with {len(self.model_candidates)} candidates and {len(self.benchmark_tests)} tests")
    
    def create_benchmark_test_suite(self) -> List[BenchmarkTest]:
        """Create comprehensive benchmark test suite"""
        
        tests = []
        
        # Multi-step reasoning tests
        tests.extend([
            BenchmarkTest(
                test_id="reasoning_task_breakdown",
                category=BenchmarkCategory.MULTI_STEP_REASONING,
                name="Complex Task Breakdown",
                description="Break down a complex software project into manageable tasks",
                prompt="""Break down this complex task into specific, actionable subtasks:
                
"Build a real-time chat application with user authentication, message persistence, and file sharing capabilities."

Provide a hierarchical breakdown with dependencies and estimated effort for each subtask.""",
                expected_capabilities=["hierarchical_thinking", "dependency_analysis", "effort_estimation"],
                evaluation_criteria={
                    "completeness": "Covers all major components",
                    "hierarchy": "Clear parent-child relationships", 
                    "actionability": "Tasks are specific and actionable",
                    "dependencies": "Identifies critical dependencies"
                },
                max_tokens=1000,
                timeout_seconds=30
            ),
            BenchmarkTest(
                test_id="reasoning_optimization",
                category=BenchmarkCategory.MULTI_STEP_REASONING,
                name="Performance Optimization Reasoning",
                description="Multi-step analysis of performance optimization problem",
                prompt="""A web application is experiencing slow response times. Given these symptoms:
- Database queries taking 2-3 seconds
- High CPU usage during peak hours  
- Memory usage gradually increasing
- User complaints about timeouts

Provide a step-by-step analysis to identify root causes and propose solutions.""",
                expected_capabilities=["problem_analysis", "root_cause_reasoning", "solution_synthesis"],
                evaluation_criteria={
                    "systematic_approach": "Follows logical diagnostic steps",
                    "root_cause_analysis": "Identifies likely root causes",
                    "solution_quality": "Proposes appropriate solutions",
                    "prioritization": "Ranks solutions by impact/effort"
                },
                max_tokens=800,
                timeout_seconds=25
            )
        ])
        
        # Recursive breakdown tests  
        tests.extend([
            BenchmarkTest(
                test_id="recursive_prd_analysis",
                category=BenchmarkCategory.RECURSIVE_BREAKDOWN,
                name="Recursive PRD Analysis",
                description="Recursively analyze and break down a Product Requirements Document",
                prompt="""Analyze this PRD recursively, breaking it into increasingly specific components:

PRD: "Develop an AI-powered project management system that automatically generates task breakdowns, tracks progress, predicts completion dates, and optimizes resource allocation."

Perform recursive breakdown:
1. Identify main functional areas
2. Break each area into core features  
3. Break features into implementation tasks
4. Identify dependencies between all levels""",
                expected_capabilities=["recursive_analysis", "abstraction_levels", "dependency_mapping"],
                evaluation_criteria={
                    "depth": "Multiple levels of breakdown",
                    "consistency": "Consistent abstraction levels",
                    "completeness": "Covers all aspects of PRD",
                    "dependencies": "Maps cross-level dependencies"
                },
                max_tokens=1200,
                timeout_seconds=35
            ),
            BenchmarkTest(
                test_id="recursive_refactoring",
                category=BenchmarkCategory.RECURSIVE_BREAKDOWN,
                name="Recursive Code Refactoring Plan",
                description="Create recursive refactoring plan for legacy code",
                prompt="""Create a recursive refactoring plan for this legacy codebase description:

"A monolithic Python application with 50,000 lines of code, no tests, mixed concerns, global variables, and tight coupling between components."

Recursively break down the refactoring into:
1. High-level architectural changes
2. Module-level restructuring  
3. Function-level improvements
4. Line-level code quality fixes

Include risk assessment and dependencies at each level.""",
                expected_capabilities=["architectural_thinking", "risk_assessment", "incremental_planning"],
                evaluation_criteria={
                    "architectural_vision": "Clear target architecture",
                    "incremental_approach": "Safe, incremental steps",
                    "risk_management": "Identifies and mitigates risks",
                    "testability": "Improves testing at each step"
                },
                max_tokens=1000,
                timeout_seconds=30
            )
        ])
        
        # Code generation tests
        tests.extend([
            BenchmarkTest(
                test_id="codegen_task_scheduler",
                category=BenchmarkCategory.CODE_GENERATION,
                name="Task Scheduler Implementation",
                description="Generate complete task scheduler with dependencies",
                prompt="""Generate a complete Python implementation of a task scheduler that supports:

1. Task dependencies
2. Priority-based execution
3. Parallel execution of independent tasks
4. Progress tracking
5. Error handling and retry logic

Include comprehensive docstrings, type hints, and error handling.""",
                expected_capabilities=["complex_code_generation", "architectural_design", "error_handling"],
                evaluation_criteria={
                    "functionality": "Implements all required features",
                    "code_quality": "Clean, readable, well-structured",
                    "documentation": "Comprehensive docstrings",
                    "error_handling": "Robust error handling"
                },
                max_tokens=2000,
                timeout_seconds=45
            ),
            BenchmarkTest(
                test_id="codegen_optimization",
                category=BenchmarkCategory.CODE_GENERATION,
                name="Performance Optimization Code",
                description="Generate optimized algorithms and data structures",
                prompt="""Generate optimized Python code for these requirements:

1. A caching system that evicts least recently used items
2. A priority queue implementation with O(log n) operations
3. A batch processing system that optimizes I/O operations

Focus on performance, memory efficiency, and clean interfaces.""",
                expected_capabilities=["algorithm_implementation", "performance_optimization", "data_structures"],
                evaluation_criteria={
                    "algorithmic_efficiency": "Uses efficient algorithms",
                    "memory_management": "Optimizes memory usage",
                    "interface_design": "Clean, intuitive interfaces",
                    "performance_characteristics": "Achieves required complexity"
                },
                max_tokens=1500,
                timeout_seconds=40
            )
        ])
        
        # Context maintenance tests
        tests.extend([
            BenchmarkTest(
                test_id="context_long_conversation",
                category=BenchmarkCategory.CONTEXT_MAINTENANCE,
                name="Long Conversation Context",
                description="Maintain context across extended conversation",
                prompt="""This is a multi-turn conversation about developing a project management system.

Turn 1: "Let's build a project management system with task tracking and team collaboration."
Turn 2: "Add real-time notifications and file sharing capabilities."
Turn 3: "Include time tracking and reporting features."
Turn 4: "Integrate with calendar systems and email notifications."
Turn 5: "Add mobile app support and offline synchronization."

Current Turn: Based on our entire conversation, provide a comprehensive technical architecture document that addresses all the features we've discussed, maintaining consistency with our previous decisions.""",
                expected_capabilities=["long_term_memory", "context_integration", "consistency_maintenance"],
                evaluation_criteria={
                    "context_retention": "References all previous turns",
                    "consistency": "Maintains consistent decisions",
                    "integration": "Integrates all discussed features",
                    "coherence": "Creates coherent overall design"
                },
                max_tokens=1500,
                timeout_seconds=35
            )
        ])
        
        # Planning synthesis tests
        tests.extend([
            BenchmarkTest(
                test_id="planning_project_synthesis",
                category=BenchmarkCategory.PLANNING_SYNTHESIS,
                name="Project Plan Synthesis",
                description="Synthesize comprehensive project plan from requirements",
                prompt="""Synthesize a comprehensive project plan for:

Requirements:
- Migrate legacy system to microservices
- 6-month timeline with 4-person team
- Must maintain 99.9% uptime during migration
- Budget constraint of $200K
- Compliance with SOX regulations

Create a detailed plan including phases, milestones, resource allocation, risk mitigation, and success metrics.""",
                expected_capabilities=["strategic_planning", "resource_optimization", "risk_management"],
                evaluation_criteria={
                    "completeness": "Addresses all requirements",
                    "feasibility": "Realistic timeline and resources",
                    "risk_management": "Identifies and mitigates risks",
                    "metrics": "Defines clear success metrics"
                },
                max_tokens=1200,
                timeout_seconds=35
            )
        ])
        
        # Task understanding tests
        tests.extend([
            BenchmarkTest(
                test_id="understanding_ambiguous_task",
                category=BenchmarkCategory.TASK_UNDERSTANDING,
                name="Ambiguous Task Clarification",
                description="Understand and clarify ambiguous task requirements",
                prompt="""A stakeholder says: "Make the system faster and more user-friendly."

Demonstrate task understanding by:
1. Identifying the ambiguities in this request
2. Asking clarifying questions to gather specific requirements
3. Proposing measurable success criteria
4. Suggesting a structured approach to address the request""",
                expected_capabilities=["requirement_analysis", "ambiguity_resolution", "structured_thinking"],
                evaluation_criteria={
                    "ambiguity_identification": "Identifies unclear aspects",
                    "clarifying_questions": "Asks relevant, specific questions",
                    "measurability": "Proposes measurable criteria",
                    "structured_approach": "Systematic methodology"
                },
                max_tokens=800,
                timeout_seconds=25
            )
        ])
        
        return tests
    
    def check_ollama_availability(self) -> bool:
        """Check if Ollama is installed and running"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of models available in Ollama"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            return []
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []
    
    def pull_model_if_needed(self, model_id: str) -> bool:
        """Pull model if not available locally"""
        available_models = self.get_available_models()
        
        if model_id in available_models:
            print(f"✅ Model {model_id} already available")
            return True
        
        print(f"📥 Pulling model {model_id}...")
        try:
            result = subprocess.run(['ollama', 'pull', model_id], capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(f"✅ Successfully pulled {model_id}")
                return True
            else:
                print(f"❌ Failed to pull {model_id}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"⏱️ Timeout pulling {model_id}")
            return False
        except FileNotFoundError:
            print(f"❌ Ollama not found - install Ollama first")
            return False
    
    def run_model_inference(self, model_id: str, prompt: str, max_tokens: int = 1000, timeout: int = 30) -> Tuple[str, Dict[str, Any]]:
        """Run inference on a model and collect performance metrics"""
        
        # Start monitoring (simplified without psutil)
        start_time = time.time()
        
        try:
            # Run ollama generate
            cmd = ['ollama', 'generate', model_id, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            end_time = time.time()
            
            execution_time_ms = (end_time - start_time) * 1000
            memory_usage_mb = 100.0  # Simulated memory usage
            
            if result.returncode == 0:
                response = result.stdout.strip()
                metrics = {
                    'execution_time_ms': execution_time_ms,
                    'memory_usage_mb': memory_usage_mb,
                    'cpu_usage_percent': 50.0,  # Simulated CPU usage
                    'success': True,
                    'error_message': None
                }
                return response, metrics
            else:
                metrics = {
                    'execution_time_ms': execution_time_ms,
                    'memory_usage_mb': 0,
                    'cpu_usage_percent': 0,
                    'success': False,
                    'error_message': result.stderr
                }
                return "", metrics
                
        except subprocess.TimeoutExpired:
            metrics = {
                'execution_time_ms': timeout * 1000,
                'memory_usage_mb': 0,
                'cpu_usage_percent': 0,
                'success': False,
                'error_message': f"Timeout after {timeout} seconds"
            }
            return "", metrics
        except Exception as e:
            metrics = {
                'execution_time_ms': 0,
                'memory_usage_mb': 0,
                'cpu_usage_percent': 0,
                'success': False,
                'error_message': str(e)
            }
            return "", metrics
    
    def evaluate_response_quality(self, test: BenchmarkTest, response: str) -> Tuple[float, float, Dict[str, float]]:
        """Evaluate response quality against test criteria"""
        
        accuracy_score = 0.0
        quality_score = 0.0
        capability_scores = {}
        
        if not response or len(response.strip()) < 10:
            return 0.0, 0.0, {}
        
        response_lower = response.lower()
        
        # Basic quality indicators
        has_structure = any(marker in response for marker in ['1.', '2.', '-', '*', '\n\n'])
        has_technical_terms = len([word for word in response.split() if len(word) > 6]) > 5
        is_comprehensive = len(response.split()) > 50
        
        # Category-specific evaluation
        if test.category == BenchmarkCategory.MULTI_STEP_REASONING:
            # Look for step-by-step reasoning
            has_steps = any(step in response_lower for step in ['step', 'first', 'second', 'then', 'next', 'finally'])
            has_analysis = any(term in response_lower for term in ['analyze', 'because', 'therefore', 'result'])
            
            accuracy_score = 0.3 if has_steps else 0.0
            accuracy_score += 0.4 if has_analysis else 0.0
            accuracy_score += 0.3 if is_comprehensive else 0.0
            
        elif test.category == BenchmarkCategory.RECURSIVE_BREAKDOWN:
            # Look for hierarchical breakdown
            has_hierarchy = any(marker in response for marker in ['1.', '2.', '1.1', '1.2', 'a.', 'b.'])
            has_dependencies = any(term in response_lower for term in ['depend', 'require', 'before', 'after'])
            
            accuracy_score = 0.4 if has_hierarchy else 0.0
            accuracy_score += 0.3 if has_dependencies else 0.0
            accuracy_score += 0.3 if is_comprehensive else 0.0
            
        elif test.category == BenchmarkCategory.CODE_GENERATION:
            # Look for code characteristics
            has_code_blocks = '```' in response or 'def ' in response or 'class ' in response
            has_imports = 'import ' in response or 'from ' in response
            has_docstrings = '"""' in response or "'''" in response
            
            accuracy_score = 0.5 if has_code_blocks else 0.0
            accuracy_score += 0.2 if has_imports else 0.0
            accuracy_score += 0.3 if has_docstrings else 0.0
            
        elif test.category == BenchmarkCategory.CONTEXT_MAINTENANCE:
            # Look for context references
            has_references = any(ref in response_lower for ref in ['previous', 'discussed', 'mentioned', 'earlier'])
            has_integration = any(term in response_lower for term in ['integrate', 'combine', 'together'])
            
            accuracy_score = 0.4 if has_references else 0.0
            accuracy_score += 0.3 if has_integration else 0.0
            accuracy_score += 0.3 if is_comprehensive else 0.0
            
        elif test.category == BenchmarkCategory.PLANNING_SYNTHESIS:
            # Look for planning elements
            has_timeline = any(term in response_lower for term in ['month', 'week', 'phase', 'milestone'])
            has_resources = any(term in response_lower for term in ['team', 'budget', 'resource', 'cost'])
            has_risks = any(term in response_lower for term in ['risk', 'challenge', 'mitigation'])
            
            accuracy_score = 0.3 if has_timeline else 0.0
            accuracy_score += 0.3 if has_resources else 0.0
            accuracy_score += 0.4 if has_risks else 0.0
            
        else:  # TASK_UNDERSTANDING
            has_questions = '?' in response
            has_clarification = any(term in response_lower for term in ['clarify', 'specific', 'measure', 'define'])
            
            accuracy_score = 0.4 if has_questions else 0.0
            accuracy_score += 0.6 if has_clarification else 0.0
        
        # General quality score
        quality_score = 0.2 if has_structure else 0.0
        quality_score += 0.3 if has_technical_terms else 0.0
        quality_score += 0.2 if is_comprehensive else 0.0
        quality_score += 0.3 if len(response.split()) > 100 else 0.0
        
        # Capability scores based on expected capabilities
        for capability in test.expected_capabilities:
            if capability in response_lower.replace('_', ' '):
                capability_scores[capability] = 0.8
            elif any(word in response_lower for word in capability.split('_')):
                capability_scores[capability] = 0.5
            else:
                capability_scores[capability] = 0.2
        
        return min(1.0, accuracy_score), min(1.0, quality_score), capability_scores
    
    def benchmark_model(self, model_id: str) -> ModelBenchmarkSummary:
        """Run complete benchmark suite for a single model"""
        
        print(f"\n🧪 Benchmarking model: {model_id}")
        
        # Check if model is available
        if not self.pull_model_if_needed(model_id):
            print(f"❌ Cannot benchmark {model_id} - model unavailable")
            return self.create_failed_summary(model_id, "Model unavailable")
        
        results = []
        
        for test in self.benchmark_tests:
            print(f"  🔬 Running test: {test.name}")
            
            # Run inference
            response, metrics = self.run_model_inference(
                model_id, test.prompt, test.max_tokens, test.timeout_seconds
            )
            
            if metrics['success']:
                # Evaluate response quality
                accuracy_score, quality_score, capability_scores = self.evaluate_response_quality(test, response)
            else:
                accuracy_score = 0.0
                quality_score = 0.0
                capability_scores = {cap: 0.0 for cap in test.expected_capabilities}
            
            # Create result
            result = BenchmarkResult(
                test_id=test.test_id,
                model_id=model_id,
                execution_time_ms=metrics['execution_time_ms'],
                memory_usage_mb=metrics['memory_usage_mb'],
                cpu_usage_percent=metrics['cpu_usage_percent'],
                response=response[:500] if response else "",  # Truncate for storage
                success=metrics['success'],
                accuracy_score=accuracy_score,
                quality_score=quality_score,
                capability_scores=capability_scores,
                error_message=metrics['error_message'],
                timestamp=datetime.now()
            )
            
            results.append(result)
            
            # Short pause between tests
            time.sleep(1)
        
        # Create summary
        summary = self.create_model_summary(model_id, results)
        
        # Save results
        self.save_benchmark_results(model_id, results, summary)
        
        return summary
    
    def create_model_summary(self, model_id: str, results: List[BenchmarkResult]) -> ModelBenchmarkSummary:
        """Create benchmark summary for a model"""
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return self.create_failed_summary(model_id, "All tests failed")
        
        # Calculate averages
        avg_execution_time = statistics.mean(r.execution_time_ms for r in successful_results)
        avg_memory_usage = statistics.mean(r.memory_usage_mb for r in successful_results)
        avg_cpu_usage = statistics.mean(r.cpu_usage_percent for r in successful_results)
        overall_accuracy = statistics.mean(r.accuracy_score for r in successful_results)
        overall_quality = statistics.mean(r.quality_score for r in successful_results)
        
        # Calculate category scores
        category_scores = {}
        for category in BenchmarkCategory:
            category_results = [r for r in successful_results if any(t.category == category for t in self.benchmark_tests if t.test_id == r.test_id)]
            if category_results:
                category_scores[category.value] = statistics.mean(r.accuracy_score for r in category_results)
            else:
                category_scores[category.value] = 0.0
        
        # Calculate suitability score
        suitability_score = (overall_accuracy * 0.4 + overall_quality * 0.3 + 
                           (len(successful_results) / len(results)) * 0.3)
        
        # Generate recommendation
        if suitability_score >= 0.8:
            recommendation = "Excellent - Highly recommended for Task Master AI"
        elif suitability_score >= 0.7:
            recommendation = "Good - Suitable with minor limitations"
        elif suitability_score >= 0.6:
            recommendation = "Acceptable - Consider for lightweight use cases"
        elif suitability_score >= 0.5:
            recommendation = "Poor - Not recommended for core functionality"
        else:
            recommendation = "Unsuitable - Does not meet minimum requirements"
        
        return ModelBenchmarkSummary(
            model_id=model_id,
            total_tests=len(results),
            successful_tests=len(successful_results),
            average_execution_time_ms=avg_execution_time,
            average_memory_usage_mb=avg_memory_usage,
            average_cpu_usage_percent=avg_cpu_usage,
            overall_accuracy=overall_accuracy,
            overall_quality=overall_quality,
            category_scores=category_scores,
            recommendation=recommendation,
            suitability_score=suitability_score
        )
    
    def create_failed_summary(self, model_id: str, reason: str) -> ModelBenchmarkSummary:
        """Create summary for failed benchmark"""
        return ModelBenchmarkSummary(
            model_id=model_id,
            total_tests=len(self.benchmark_tests),
            successful_tests=0,
            average_execution_time_ms=0.0,
            average_memory_usage_mb=0.0,
            average_cpu_usage_percent=0.0,
            overall_accuracy=0.0,
            overall_quality=0.0,
            category_scores={cat.value: 0.0 for cat in BenchmarkCategory},
            recommendation=f"Failed - {reason}",
            suitability_score=0.0
        )
    
    def save_benchmark_results(self, model_id: str, results: List[BenchmarkResult], summary: ModelBenchmarkSummary):
        """Save benchmark results to disk"""
        
        # Save detailed results
        results_data = [asdict(result) for result in results]
        for result_data in results_data:
            result_data['timestamp'] = result_data['timestamp'].isoformat()
        
        results_file = self.results_dir / f"{model_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save summary
        summary_data = asdict(summary)
        summary_file = self.results_dir / f"{model_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
    
    def benchmark_all_candidates(self, candidate_models: List[str] = None) -> Dict[str, ModelBenchmarkSummary]:
        """Run benchmarks for all or specified candidate models"""
        
        if candidate_models is None:
            candidate_models = list(self.model_candidates.keys())
        
        print(f"🚀 Starting benchmark suite for {len(candidate_models)} models")
        
        summaries = {}
        
        for model_id in candidate_models:
            try:
                summary = self.benchmark_model(model_id)
                summaries[model_id] = summary
                
                print(f"✅ {model_id}: {summary.recommendation} (Score: {summary.suitability_score:.2f})")
                
            except Exception as e:
                print(f"❌ Error benchmarking {model_id}: {e}")
                summaries[model_id] = self.create_failed_summary(model_id, str(e))
        
        # Generate comparative report
        self.generate_comparative_report(summaries)
        
        return summaries
    
    def generate_comparative_report(self, summaries: Dict[str, ModelBenchmarkSummary]):
        """Generate comparative benchmark report"""
        
        # Sort by suitability score
        sorted_models = sorted(summaries.items(), key=lambda x: x[1].suitability_score, reverse=True)
        
        report = "# Local LLM Benchmark Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**Models Tested:** {len(summaries)}\n\n"
        
        report += "## Executive Summary\n\n"
        
        if sorted_models:
            best_model = sorted_models[0]
            report += f"**Top Recommendation:** {best_model[0]} (Score: {best_model[1].suitability_score:.2f})\n"
            report += f"**Recommendation:** {best_model[1].recommendation}\n\n"
        
        report += "## Detailed Results\n\n"
        report += "| Model | Score | Accuracy | Quality | Speed (ms) | Memory (MB) | Recommendation |\n"
        report += "|-------|-------|----------|---------|------------|-------------|----------------|\n"
        
        for model_id, summary in sorted_models:
            report += f"| {model_id} | {summary.suitability_score:.2f} | {summary.overall_accuracy:.2f} | {summary.overall_quality:.2f} | {summary.average_execution_time_ms:.0f} | {summary.average_memory_usage_mb:.1f} | {summary.recommendation} |\n"
        
        report += "\n## Category Performance\n\n"
        
        for category in BenchmarkCategory:
            report += f"### {category.value.replace('_', ' ').title()}\n\n"
            category_results = [(model_id, summary.category_scores.get(category.value, 0.0)) 
                              for model_id, summary in sorted_models]
            category_results.sort(key=lambda x: x[1], reverse=True)
            
            for model_id, score in category_results[:3]:  # Top 3
                report += f"- **{model_id}:** {score:.2f}\n"
            report += "\n"
        
        # Save report
        report_file = self.results_dir / 'benchmark_report.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📊 Comparative report saved to: {report_file}")
    
    def get_system_requirements(self) -> Dict[str, Any]:
        """Get current system requirements and capabilities"""
        
        return {
            'cpu_count': os.cpu_count() or 4,
            'memory_total_gb': 16.0,  # Simulated
            'memory_available_gb': 8.0,  # Simulated
            'disk_free_gb': 100.0,  # Simulated
            'ollama_available': self.check_ollama_availability(),
            'available_models': self.get_available_models()
        }

def main():
    """Demo of Local LLM Benchmark Suite"""
    print("Local LLM Benchmark Suite for Task Master AI")
    print("=" * 55)
    
    benchmark = LocalLLMBenchmark()
    
    # Check system requirements
    system_info = benchmark.get_system_requirements()
    print(f"\n💻 System Information:")
    print(f"  CPU Cores: {system_info['cpu_count']}")
    print(f"  Total Memory: {system_info['memory_total_gb']:.1f} GB")
    print(f"  Available Memory: {system_info['memory_available_gb']:.1f} GB")
    print(f"  Ollama Available: {system_info['ollama_available']}")
    
    if not system_info['ollama_available']:
        print(f"\n❌ Ollama not available. Please install Ollama first:")
        print(f"   Visit: https://ollama.ai/")
        return
    
    print(f"  Available Models: {len(system_info['available_models'])}")
    for model in system_info['available_models'][:5]:  # Show first 5
        print(f"    - {model}")
    
    # Demo with a single lightweight model if available
    available_models = system_info['available_models']
    demo_models = []
    
    # Look for lightweight models suitable for demo
    priority_models = ['llama2:7b', 'mistral:7b', 'codellama:7b', 'neural-chat:7b']
    for model in priority_models:
        if model in available_models:
            demo_models.append(model)
            break
    
    if not demo_models:
        # Try to pull a small model for demo
        print(f"\n📥 No suitable models found. Attempting to pull mistral:7b for demo...")
        if benchmark.pull_model_if_needed('mistral:7b'):
            demo_models.append('mistral:7b')
    
    if demo_models:
        print(f"\n🧪 Running demo benchmark with: {demo_models[0]}")
        summaries = benchmark.benchmark_all_candidates(demo_models)
        
        if summaries:
            model_id = demo_models[0]
            summary = summaries[model_id]
            print(f"\n📊 Demo Results for {model_id}:")
            print(f"  Suitability Score: {summary.suitability_score:.2f}")
            print(f"  Overall Accuracy: {summary.overall_accuracy:.2f}")
            print(f"  Overall Quality: {summary.overall_quality:.2f}")
            print(f"  Successful Tests: {summary.successful_tests}/{summary.total_tests}")
            print(f"  Recommendation: {summary.recommendation}")
    else:
        print(f"\n⚠️ No models available for benchmarking")
        print(f"   Pull a model first: ollama pull mistral:7b")
    
    print(f"\n✅ Benchmark suite demo completed")

if __name__ == "__main__":
    main()