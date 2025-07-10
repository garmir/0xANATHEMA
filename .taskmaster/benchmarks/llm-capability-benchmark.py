#!/usr/bin/env python3
"""
Comprehensive LLM Capability Benchmarking Framework for Task Master AI
=====================================================================

This framework provides systematic evaluation of local LLMs across Task Master AI's
core capabilities through standardized benchmarks and custom test suites.

Core Capabilities Evaluated:
1. Recursive Task Breakdown
2. Multi-Step Reasoning
3. Context Maintenance
4. Code Generation & Analysis
5. Research Synthesis
6. Autonomous Execution Planning
7. Meta-Learning

Usage:
    python llm-capability-benchmark.py --model llama-70b --config benchmarks.json
    python llm-capability-benchmark.py --run-all --output results/
"""

import json
import time
import asyncio
import logging
import argparse
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
import subprocess
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import bleu_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmarks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    model_name: str
    capability: str
    test_name: str
    score: float
    latency_ms: float
    memory_usage_mb: float
    tokens_per_second: float
    accuracy: float
    completeness: float
    consistency: float
    human_evaluation: Optional[float] = None
    error_rate: float = 0.0
    context_retention: float = 0.0
    reasoning_depth: int = 0
    code_quality: float = 0.0
    autonomy_score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ModelConfig:
    """Configuration for a model to benchmark"""
    name: str
    model_path: str
    model_type: str  # 'huggingface', 'ollama', 'openai', 'custom'
    quantization: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    custom_loader: Optional[callable] = None

class LLMInterface(ABC):
    """Abstract interface for LLM implementations"""
    
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources"""
        pass

class HuggingFaceModel(LLMInterface):
    """Hugging Face model implementation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_4bit=config.quantization == "4bit",
            load_in_8bit=config.quantization == "8bit"
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True
        )
    
    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from prompt"""
        try:
            result = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                return_full_text=False
            )
            return result[0]['generated_text']
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.config.name,
            "type": "huggingface",
            "path": self.config.model_path,
            "quantization": self.config.quantization,
            "parameters": getattr(self.model, 'num_parameters', lambda: 0)()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class OllamaModel(LLMInterface):
    """Ollama model implementation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.base_url = config.api_endpoint or "http://localhost:11434"
    
    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from prompt"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "num_predict": max_tokens
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.config.name}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"name": self.config.name, "type": "ollama"}
    
    def cleanup(self):
        """Cleanup resources"""
        pass

class BenchmarkTest(ABC):
    """Abstract base class for benchmark tests"""
    
    def __init__(self, name: str, capability: str):
        self.name = name
        self.capability = capability
    
    @abstractmethod
    async def run(self, model: LLMInterface) -> BenchmarkResult:
        """Run the benchmark test"""
        pass
    
    def measure_performance(self, func):
        """Decorator to measure performance metrics"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            latency_ms = (end_time - start_time) * 1000
            memory_usage_mb = end_memory - start_memory
            
            return result, latency_ms, memory_usage_mb
        
        return wrapper

class RecursiveTaskBreakdownTest(BenchmarkTest):
    """Test recursive task breakdown capability"""
    
    def __init__(self):
        super().__init__("recursive_breakdown", "Recursive Task Breakdown")
        self.test_cases = [
            {
                "task": "Build a web application for project management",
                "expected_subtasks": [
                    "Design system architecture",
                    "Implement user authentication",
                    "Create project management features",
                    "Build user interface",
                    "Setup database",
                    "Add testing framework",
                    "Deploy application"
                ],
                "max_depth": 3
            },
            {
                "task": "Optimize database performance for high-traffic application",
                "expected_subtasks": [
                    "Analyze current performance bottlenecks",
                    "Implement indexing strategy",
                    "Optimize query patterns",
                    "Setup caching layer",
                    "Configure connection pooling",
                    "Monitor and validate improvements"
                ],
                "max_depth": 2
            }
        ]
    
    async def run(self, model: LLMInterface) -> BenchmarkResult:
        """Run recursive task breakdown test"""
        results = []
        
        for test_case in self.test_cases:
            prompt = f"""
            Break down the following task into a hierarchical structure of subtasks:
            
            Task: {test_case['task']}
            
            Requirements:
            - Provide a detailed breakdown with up to {test_case['max_depth']} levels of depth
            - Each subtask should be actionable and specific
            - Include dependencies between subtasks
            - Format as a structured JSON with task hierarchy
            
            Output format:
            {{
                "main_task": "...",
                "subtasks": [
                    {{
                        "id": "1",
                        "title": "...",
                        "description": "...",
                        "dependencies": [],
                        "subtasks": [...]
                    }}
                ]
            }}
            """
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            response = await model.generate(prompt, max_tokens=1024)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Analyze response quality
            accuracy = self._evaluate_breakdown_accuracy(response, test_case)
            completeness = self._evaluate_breakdown_completeness(response, test_case)
            consistency = self._evaluate_breakdown_consistency(response)
            reasoning_depth = self._calculate_reasoning_depth(response)
            
            results.append({
                "accuracy": accuracy,
                "completeness": completeness,
                "consistency": consistency,
                "reasoning_depth": reasoning_depth,
                "latency_ms": (end_time - start_time) * 1000,
                "memory_usage_mb": end_memory - start_memory
            })
        
        # Aggregate results
        avg_accuracy = statistics.mean([r["accuracy"] for r in results])
        avg_completeness = statistics.mean([r["completeness"] for r in results])
        avg_consistency = statistics.mean([r["consistency"] for r in results])
        avg_reasoning_depth = statistics.mean([r["reasoning_depth"] for r in results])
        avg_latency = statistics.mean([r["latency_ms"] for r in results])
        avg_memory = statistics.mean([r["memory_usage_mb"] for r in results])
        
        return BenchmarkResult(
            model_name=model.get_model_info()["name"],
            capability=self.capability,
            test_name=self.name,
            score=(avg_accuracy + avg_completeness + avg_consistency) / 3,
            latency_ms=avg_latency,
            memory_usage_mb=avg_memory,
            tokens_per_second=0,  # Calculate if needed
            accuracy=avg_accuracy,
            completeness=avg_completeness,
            consistency=avg_consistency,
            reasoning_depth=avg_reasoning_depth
        )
    
    def _evaluate_breakdown_accuracy(self, response: str, test_case: Dict) -> float:
        """Evaluate accuracy of task breakdown"""
        try:
            # Parse JSON response
            breakdown = json.loads(response)
            expected_subtasks = test_case["expected_subtasks"]
            
            # Extract generated subtasks
            generated_subtasks = []
            self._extract_subtasks(breakdown.get("subtasks", []), generated_subtasks)
            
            # Calculate overlap with expected subtasks
            matches = 0
            for expected in expected_subtasks:
                for generated in generated_subtasks:
                    if self._similarity_score(expected, generated) > 0.7:
                        matches += 1
                        break
            
            return matches / len(expected_subtasks)
        except Exception as e:
            logger.error(f"Error evaluating breakdown accuracy: {e}")
            return 0.0
    
    def _evaluate_breakdown_completeness(self, response: str, test_case: Dict) -> float:
        """Evaluate completeness of task breakdown"""
        try:
            breakdown = json.loads(response)
            subtasks = breakdown.get("subtasks", [])
            
            # Check for essential elements
            has_structure = len(subtasks) > 0
            has_dependencies = any("dependencies" in task for task in subtasks)
            has_descriptions = any("description" in task for task in subtasks)
            has_hierarchy = any("subtasks" in task for task in subtasks)
            
            completeness_score = sum([has_structure, has_dependencies, has_descriptions, has_hierarchy]) / 4
            return completeness_score
        except Exception as e:
            logger.error(f"Error evaluating breakdown completeness: {e}")
            return 0.0
    
    def _evaluate_breakdown_consistency(self, response: str) -> float:
        """Evaluate consistency of task breakdown"""
        try:
            breakdown = json.loads(response)
            subtasks = breakdown.get("subtasks", [])
            
            # Check for consistent formatting
            has_consistent_ids = all("id" in task for task in subtasks)
            has_consistent_titles = all("title" in task for task in subtasks)
            has_consistent_structure = True  # Add more checks as needed
            
            consistency_score = sum([has_consistent_ids, has_consistent_titles, has_consistent_structure]) / 3
            return consistency_score
        except Exception as e:
            logger.error(f"Error evaluating breakdown consistency: {e}")
            return 0.0
    
    def _calculate_reasoning_depth(self, response: str) -> int:
        """Calculate reasoning depth of breakdown"""
        try:
            breakdown = json.loads(response)
            return self._max_depth(breakdown.get("subtasks", []))
        except Exception as e:
            logger.error(f"Error calculating reasoning depth: {e}")
            return 0
    
    def _max_depth(self, subtasks: List[Dict], current_depth: int = 1) -> int:
        """Calculate maximum depth of task hierarchy"""
        if not subtasks:
            return current_depth
        
        max_depth = current_depth
        for task in subtasks:
            if "subtasks" in task and task["subtasks"]:
                depth = self._max_depth(task["subtasks"], current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _extract_subtasks(self, subtasks: List[Dict], result: List[str]):
        """Extract all subtask titles from hierarchy"""
        for task in subtasks:
            if "title" in task:
                result.append(task["title"])
            if "subtasks" in task:
                self._extract_subtasks(task["subtasks"], result)
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

class MultiStepReasoningTest(BenchmarkTest):
    """Test multi-step reasoning capability"""
    
    def __init__(self):
        super().__init__("multi_step_reasoning", "Multi-Step Reasoning")
        self.test_cases = [
            {
                "problem": "A company has 3 servers. Each server can handle 1000 requests per second. If traffic increases by 50% each month, how many additional servers are needed after 6 months to maintain performance?",
                "steps": [
                    "Calculate current capacity: 3 servers × 1000 req/s = 3000 req/s",
                    "Calculate traffic after 6 months: 3000 × (1.5)^6 = 3000 × 11.39 = 34,171 req/s",
                    "Calculate servers needed: 34,171 ÷ 1000 = 35 servers (rounded up)",
                    "Calculate additional servers: 35 - 3 = 32 additional servers"
                ],
                "answer": 32
            },
            {
                "problem": "Design a caching strategy for a web application that serves 10,000 users with 100 requests per user per day. Cache hit ratio should be 80%. What cache size is needed if average response is 2KB?",
                "steps": [
                    "Calculate total daily requests: 10,000 users × 100 requests = 1,000,000 requests",
                    "Calculate cache hits: 1,000,000 × 0.8 = 800,000 cache hits",
                    "Calculate cache misses: 1,000,000 × 0.2 = 200,000 cache misses",
                    "Calculate cache storage: 800,000 × 2KB = 1,600,000 KB = 1.6 GB",
                    "Add buffer for cache efficiency: 1.6 GB × 1.5 = 2.4 GB recommended"
                ],
                "answer": 2.4
            }
        ]
    
    async def run(self, model: LLMInterface) -> BenchmarkResult:
        """Run multi-step reasoning test"""
        results = []
        
        for test_case in self.test_cases:
            prompt = f"""
            Solve the following problem step by step, showing your reasoning:
            
            Problem: {test_case['problem']}
            
            Please provide:
            1. A clear breakdown of the problem
            2. Step-by-step solution with calculations
            3. Final answer with units
            4. Brief explanation of your reasoning approach
            
            Format your response as:
            BREAKDOWN: [problem analysis]
            STEPS:
            1. [step 1]
            2. [step 2]
            ...
            ANSWER: [final answer]
            REASONING: [explanation]
            """
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            response = await model.generate(prompt, max_tokens=1024)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Analyze response quality
            accuracy = self._evaluate_reasoning_accuracy(response, test_case)
            completeness = self._evaluate_reasoning_completeness(response, test_case)
            consistency = self._evaluate_reasoning_consistency(response)
            reasoning_depth = self._calculate_reasoning_steps(response)
            
            results.append({
                "accuracy": accuracy,
                "completeness": completeness,
                "consistency": consistency,
                "reasoning_depth": reasoning_depth,
                "latency_ms": (end_time - start_time) * 1000,
                "memory_usage_mb": end_memory - start_memory
            })
        
        # Aggregate results
        avg_accuracy = statistics.mean([r["accuracy"] for r in results])
        avg_completeness = statistics.mean([r["completeness"] for r in results])
        avg_consistency = statistics.mean([r["consistency"] for r in results])
        avg_reasoning_depth = statistics.mean([r["reasoning_depth"] for r in results])
        avg_latency = statistics.mean([r["latency_ms"] for r in results])
        avg_memory = statistics.mean([r["memory_usage_mb"] for r in results])
        
        return BenchmarkResult(
            model_name=model.get_model_info()["name"],
            capability=self.capability,
            test_name=self.name,
            score=(avg_accuracy + avg_completeness + avg_consistency) / 3,
            latency_ms=avg_latency,
            memory_usage_mb=avg_memory,
            tokens_per_second=0,
            accuracy=avg_accuracy,
            completeness=avg_completeness,
            consistency=avg_consistency,
            reasoning_depth=avg_reasoning_depth
        )
    
    def _evaluate_reasoning_accuracy(self, response: str, test_case: Dict) -> float:
        """Evaluate accuracy of reasoning"""
        try:
            # Extract answer from response
            answer_line = ""
            for line in response.split('\n'):
                if line.strip().startswith('ANSWER:'):
                    answer_line = line.strip()
                    break
            
            if not answer_line:
                return 0.0
            
            # Extract numeric answer
            import re
            numbers = re.findall(r'\d+\.?\d*', answer_line)
            if not numbers:
                return 0.0
            
            predicted_answer = float(numbers[0])
            expected_answer = test_case["answer"]
            
            # Calculate accuracy based on relative error
            relative_error = abs(predicted_answer - expected_answer) / expected_answer
            accuracy = max(0, 1 - relative_error)
            
            return accuracy
        except Exception as e:
            logger.error(f"Error evaluating reasoning accuracy: {e}")
            return 0.0
    
    def _evaluate_reasoning_completeness(self, response: str, test_case: Dict) -> float:
        """Evaluate completeness of reasoning"""
        required_sections = ["BREAKDOWN:", "STEPS:", "ANSWER:", "REASONING:"]
        present_sections = sum(1 for section in required_sections if section in response)
        
        # Check if steps match expected complexity
        expected_steps = len(test_case["steps"])
        actual_steps = len([line for line in response.split('\n') if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10))])
        
        step_completeness = min(1.0, actual_steps / expected_steps)
        section_completeness = present_sections / len(required_sections)
        
        return (step_completeness + section_completeness) / 2
    
    def _evaluate_reasoning_consistency(self, response: str) -> float:
        """Evaluate consistency of reasoning"""
        # Check for logical flow and consistency
        has_breakdown = "BREAKDOWN:" in response
        has_steps = "STEPS:" in response
        has_answer = "ANSWER:" in response
        has_reasoning = "REASONING:" in response
        
        # Check for step numbering consistency
        step_lines = [line for line in response.split('\n') if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10))]
        consistent_numbering = True
        for i, line in enumerate(step_lines):
            if not line.strip().startswith(f"{i+1}."):
                consistent_numbering = False
                break
        
        consistency_score = sum([has_breakdown, has_steps, has_answer, has_reasoning, consistent_numbering]) / 5
        return consistency_score
    
    def _calculate_reasoning_steps(self, response: str) -> int:
        """Calculate number of reasoning steps"""
        step_lines = [line for line in response.split('\n') if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10))]
        return len(step_lines)

class ContextMaintenanceTest(BenchmarkTest):
    """Test context maintenance capability"""
    
    def __init__(self):
        super().__init__("context_maintenance", "Context Maintenance")
        self.test_cases = [
            {
                "context": "You are managing a software project with the following components: Frontend (React), Backend (Node.js), Database (PostgreSQL), and CI/CD (GitHub Actions). The project has 3 developers: Alice (Frontend), Bob (Backend), and Carol (DevOps).",
                "questions": [
                    "What technology is used for the frontend?",
                    "Who is responsible for the backend development?",
                    "What database system is being used?",
                    "How many developers are working on the project?",
                    "What CI/CD platform is being used?",
                    "If Alice needs help with a database query, who should she contact?",
                    "What would be the impact if the PostgreSQL database goes down?",
                    "How would you coordinate a deployment involving all three developers?"
                ],
                "expected_answers": [
                    "React",
                    "Bob",
                    "PostgreSQL",
                    "3",
                    "GitHub Actions",
                    "Carol",
                    "Backend services would be affected",
                    "Coordinate between Alice, Bob, and Carol"
                ]
            }
        ]
    
    async def run(self, model: LLMInterface) -> BenchmarkResult:
        """Run context maintenance test"""
        results = []
        
        for test_case in self.test_cases:
            context = test_case["context"]
            questions = test_case["questions"]
            expected_answers = test_case["expected_answers"]
            
            # Build conversation with context
            conversation = f"Context: {context}\n\n"
            responses = []
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            for i, question in enumerate(questions):
                prompt = f"{conversation}Question {i+1}: {question}\nAnswer:"
                
                response = await model.generate(prompt, max_tokens=256)
                responses.append(response)
                
                # Add to conversation history
                conversation += f"Question {i+1}: {question}\nAnswer: {response}\n\n"
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Evaluate context maintenance
            accuracy = self._evaluate_context_accuracy(responses, expected_answers)
            consistency = self._evaluate_context_consistency(responses, context)
            retention = self._evaluate_context_retention(responses, questions, context)
            
            results.append({
                "accuracy": accuracy,
                "consistency": consistency,
                "retention": retention,
                "latency_ms": (end_time - start_time) * 1000,
                "memory_usage_mb": end_memory - start_memory
            })
        
        # Aggregate results
        avg_accuracy = statistics.mean([r["accuracy"] for r in results])
        avg_consistency = statistics.mean([r["consistency"] for r in results])
        avg_retention = statistics.mean([r["retention"] for r in results])
        avg_latency = statistics.mean([r["latency_ms"] for r in results])
        avg_memory = statistics.mean([r["memory_usage_mb"] for r in results])
        
        return BenchmarkResult(
            model_name=model.get_model_info()["name"],
            capability=self.capability,
            test_name=self.name,
            score=(avg_accuracy + avg_consistency + avg_retention) / 3,
            latency_ms=avg_latency,
            memory_usage_mb=avg_memory,
            tokens_per_second=0,
            accuracy=avg_accuracy,
            consistency=avg_consistency,
            context_retention=avg_retention
        )
    
    def _evaluate_context_accuracy(self, responses: List[str], expected_answers: List[str]) -> float:
        """Evaluate accuracy of context-based answers"""
        correct = 0
        for response, expected in zip(responses, expected_answers):
            if self._similarity_score(response.lower(), expected.lower()) > 0.5:
                correct += 1
        
        return correct / len(expected_answers)
    
    def _evaluate_context_consistency(self, responses: List[str], context: str) -> float:
        """Evaluate consistency with original context"""
        # Check if responses are consistent with context
        consistent_responses = 0
        for response in responses:
            # Simple check for contradiction detection
            if self._check_consistency(response, context):
                consistent_responses += 1
        
        return consistent_responses / len(responses)
    
    def _evaluate_context_retention(self, responses: List[str], questions: List[str], context: str) -> float:
        """Evaluate context retention across conversation"""
        # Check if later responses still reference earlier context
        retention_score = 0
        for i, response in enumerate(responses):
            if i > 0:  # Skip first response
                # Check if response maintains context from earlier questions
                if self._maintains_context(response, context, questions[:i]):
                    retention_score += 1
        
        return retention_score / max(1, len(responses) - 1)
    
    def _check_consistency(self, response: str, context: str) -> bool:
        """Check if response is consistent with context"""
        # Simple implementation - could be enhanced with NLP
        return True  # Placeholder
    
    def _maintains_context(self, response: str, context: str, previous_questions: List[str]) -> bool:
        """Check if response maintains context from previous interactions"""
        # Simple implementation - could be enhanced with NLP
        return True  # Placeholder

class CodeGenerationTest(BenchmarkTest):
    """Test code generation and analysis capability"""
    
    def __init__(self):
        super().__init__("code_generation", "Code Generation & Analysis")
        self.test_cases = [
            {
                "task": "Create a Python function that implements a LRU cache with a maximum size",
                "requirements": [
                    "Function should be called 'lru_cache'",
                    "Should accept max_size parameter",
                    "Should support get() and put() operations",
                    "Should evict least recently used items when full",
                    "Should include proper error handling"
                ],
                "test_code": """
def test_lru_cache():
    cache = lru_cache(2)
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    assert cache.get("key1") == "value1"
    cache.put("key3", "value3")  # Should evict key2
    assert cache.get("key2") is None
    assert cache.get("key3") == "value3"
                """
            },
            {
                "task": "Write a JavaScript function that debounces API calls",
                "requirements": [
                    "Function should be called 'debounce'",
                    "Should accept function and delay parameters",
                    "Should prevent rapid successive calls",
                    "Should support immediate execution option",
                    "Should handle async functions properly"
                ],
                "test_code": """
function test_debounce() {
    let counter = 0;
    const increment = () => counter++;
    const debouncedIncrement = debounce(increment, 100);
    
    debouncedIncrement();
    debouncedIncrement();
    debouncedIncrement();
    
    setTimeout(() => {
        console.assert(counter === 1, "Debounce should execute once");
    }, 150);
}
                """
            }
        ]
    
    async def run(self, model: LLMInterface) -> BenchmarkResult:
        """Run code generation test"""
        results = []
        
        for test_case in self.test_cases:
            prompt = f"""
            Generate code for the following task:
            
            Task: {test_case['task']}
            
            Requirements:
            {chr(10).join(f"- {req}" for req in test_case['requirements'])}
            
            Please provide:
            1. Complete, working code
            2. Brief explanation of the implementation
            3. Example usage
            4. Any assumptions or limitations
            
            Format your response as:
            CODE:
            ```
            [your code here]
            ```
            
            EXPLANATION:
            [explanation of implementation]
            
            USAGE:
            [example usage]
            
            LIMITATIONS:
            [any limitations or assumptions]
            """
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            response = await model.generate(prompt, max_tokens=1024)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Evaluate code quality
            correctness = self._evaluate_code_correctness(response, test_case)
            completeness = self._evaluate_code_completeness(response, test_case)
            quality = self._evaluate_code_quality(response)
            
            results.append({
                "correctness": correctness,
                "completeness": completeness,
                "quality": quality,
                "latency_ms": (end_time - start_time) * 1000,
                "memory_usage_mb": end_memory - start_memory
            })
        
        # Aggregate results
        avg_correctness = statistics.mean([r["correctness"] for r in results])
        avg_completeness = statistics.mean([r["completeness"] for r in results])
        avg_quality = statistics.mean([r["quality"] for r in results])
        avg_latency = statistics.mean([r["latency_ms"] for r in results])
        avg_memory = statistics.mean([r["memory_usage_mb"] for r in results])
        
        return BenchmarkResult(
            model_name=model.get_model_info()["name"],
            capability=self.capability,
            test_name=self.name,
            score=(avg_correctness + avg_completeness + avg_quality) / 3,
            latency_ms=avg_latency,
            memory_usage_mb=avg_memory,
            tokens_per_second=0,
            accuracy=avg_correctness,
            completeness=avg_completeness,
            code_quality=avg_quality
        )
    
    def _evaluate_code_correctness(self, response: str, test_case: Dict) -> float:
        """Evaluate correctness of generated code"""
        # Extract code from response
        code = self._extract_code_block(response)
        if not code:
            return 0.0
        
        # Check if code meets requirements
        requirements_met = 0
        for requirement in test_case["requirements"]:
            if self._check_requirement(code, requirement):
                requirements_met += 1
        
        return requirements_met / len(test_case["requirements"])
    
    def _evaluate_code_completeness(self, response: str, test_case: Dict) -> float:
        """Evaluate completeness of code generation"""
        required_sections = ["CODE:", "EXPLANATION:", "USAGE:", "LIMITATIONS:"]
        present_sections = sum(1 for section in required_sections if section in response)
        
        # Check if code block exists and is non-empty
        code = self._extract_code_block(response)
        has_code = bool(code and len(code.strip()) > 0)
        
        section_completeness = present_sections / len(required_sections)
        code_completeness = 1.0 if has_code else 0.0
        
        return (section_completeness + code_completeness) / 2
    
    def _evaluate_code_quality(self, response: str) -> float:
        """Evaluate quality of generated code"""
        code = self._extract_code_block(response)
        if not code:
            return 0.0
        
        # Simple quality checks
        quality_indicators = [
            "def " in code or "function " in code,  # Has function definition
            "return" in code,  # Has return statement
            "#" in code or "//" in code,  # Has comments
            "if" in code or "for" in code or "while" in code,  # Has control flow
            len(code.split('\n')) > 5  # Has reasonable length
        ]
        
        return sum(quality_indicators) / len(quality_indicators)
    
    def _extract_code_block(self, response: str) -> str:
        """Extract code block from response"""
        lines = response.split('\n')
        in_code_block = False
        code_lines = []
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    def _check_requirement(self, code: str, requirement: str) -> bool:
        """Check if code meets a specific requirement"""
        # Simple keyword-based checking
        # This could be enhanced with AST parsing
        requirement_lower = requirement.lower()
        code_lower = code.lower()
        
        if "function" in requirement_lower and "called" in requirement_lower:
            # Extract function name from requirement
            import re
            match = re.search(r"called ['\"](.*?)['\"]", requirement_lower)
            if match:
                function_name = match.group(1)
                return function_name in code_lower
        
        # Generic keyword matching
        keywords = ["get", "put", "error", "handling", "async", "debounce", "delay"]
        for keyword in keywords:
            if keyword in requirement_lower and keyword in code_lower:
                return True
        
        return False

class ResearchSynthesisTest(BenchmarkTest):
    """Test research synthesis capability"""
    
    def __init__(self):
        super().__init__("research_synthesis", "Research Synthesis")
        self.test_cases = [
            {
                "sources": [
                    "Source 1: Microservices architecture improves scalability by allowing independent scaling of components. However, it introduces complexity in service coordination and data consistency.",
                    "Source 2: Monolithic architecture provides better performance for small to medium applications due to reduced network overhead. It's easier to deploy and debug but harder to scale specific components.",
                    "Source 3: Event-driven architecture enables loose coupling between services and better fault tolerance. It requires careful design of event schemas and handling of eventual consistency."
                ],
                "question": "What are the trade-offs between monolithic, microservices, and event-driven architectures for a medium-sized e-commerce platform?",
                "expected_points": [
                    "Scalability considerations",
                    "Performance implications",
                    "Deployment complexity",
                    "Development and maintenance overhead",
                    "Data consistency challenges",
                    "Fault tolerance and reliability"
                ]
            }
        ]
    
    async def run(self, model: LLMInterface) -> BenchmarkResult:
        """Run research synthesis test"""
        results = []
        
        for test_case in self.test_cases:
            # Build prompt with sources
            sources_text = "\n\n".join(test_case["sources"])
            
            prompt = f"""
            Analyze the following sources and provide a comprehensive synthesis:
            
            Sources:
            {sources_text}
            
            Question: {test_case['question']}
            
            Please provide:
            1. A comprehensive analysis synthesizing information from all sources
            2. Key trade-offs and considerations
            3. Recommendations based on the synthesis
            4. Any gaps or limitations in the provided information
            
            Format your response as:
            ANALYSIS:
            [comprehensive analysis]
            
            TRADE-OFFS:
            [key trade-offs]
            
            RECOMMENDATIONS:
            [recommendations]
            
            GAPS:
            [information gaps]
            """
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            response = await model.generate(prompt, max_tokens=1024)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Evaluate synthesis quality
            comprehensiveness = self._evaluate_comprehensiveness(response, test_case)
            accuracy = self._evaluate_synthesis_accuracy(response, test_case)
            coherence = self._evaluate_synthesis_coherence(response)
            
            results.append({
                "comprehensiveness": comprehensiveness,
                "accuracy": accuracy,
                "coherence": coherence,
                "latency_ms": (end_time - start_time) * 1000,
                "memory_usage_mb": end_memory - start_memory
            })
        
        # Aggregate results
        avg_comprehensiveness = statistics.mean([r["comprehensiveness"] for r in results])
        avg_accuracy = statistics.mean([r["accuracy"] for r in results])
        avg_coherence = statistics.mean([r["coherence"] for r in results])
        avg_latency = statistics.mean([r["latency_ms"] for r in results])
        avg_memory = statistics.mean([r["memory_usage_mb"] for r in results])
        
        return BenchmarkResult(
            model_name=model.get_model_info()["name"],
            capability=self.capability,
            test_name=self.name,
            score=(avg_comprehensiveness + avg_accuracy + avg_coherence) / 3,
            latency_ms=avg_latency,
            memory_usage_mb=avg_memory,
            tokens_per_second=0,
            accuracy=avg_accuracy,
            completeness=avg_comprehensiveness,
            consistency=avg_coherence
        )
    
    def _evaluate_comprehensiveness(self, response: str, test_case: Dict) -> float:
        """Evaluate comprehensiveness of synthesis"""
        expected_points = test_case["expected_points"]
        covered_points = 0
        
        response_lower = response.lower()
        for point in expected_points:
            point_keywords = point.lower().split()
            if any(keyword in response_lower for keyword in point_keywords):
                covered_points += 1
        
        return covered_points / len(expected_points)
    
    def _evaluate_synthesis_accuracy(self, response: str, test_case: Dict) -> float:
        """Evaluate accuracy of synthesis"""
        # Check if response includes information from all sources
        sources = test_case["sources"]
        sources_referenced = 0
        
        response_lower = response.lower()
        for source in sources:
            # Extract key terms from source
            source_terms = set(source.lower().split())
            common_terms = source_terms.intersection(set(response_lower.split()))
            
            if len(common_terms) > 5:  # Threshold for source reference
                sources_referenced += 1
        
        return sources_referenced / len(sources)
    
    def _evaluate_synthesis_coherence(self, response: str) -> float:
        """Evaluate coherence of synthesis"""
        required_sections = ["ANALYSIS:", "TRADE-OFFS:", "RECOMMENDATIONS:", "GAPS:"]
        present_sections = sum(1 for section in required_sections if section in response)
        
        # Check for logical flow
        has_logical_flow = self._check_logical_flow(response)
        
        section_completeness = present_sections / len(required_sections)
        coherence_score = (section_completeness + has_logical_flow) / 2
        
        return coherence_score
    
    def _check_logical_flow(self, response: str) -> float:
        """Check logical flow of synthesis"""
        # Simple implementation - could be enhanced with NLP
        sentences = response.split('.')
        avg_sentence_length = statistics.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Good synthesis should have reasonable sentence length
        return 1.0 if 10 <= avg_sentence_length <= 30 else 0.5

class AutonomousExecutionTest(BenchmarkTest):
    """Test autonomous execution planning capability"""
    
    def __init__(self):
        super().__init__("autonomous_execution", "Autonomous Execution Planning")
        self.test_cases = [
            {
                "goal": "Deploy a web application to production with zero downtime",
                "constraints": [
                    "Current application is running on single server",
                    "No load balancer currently in place",
                    "Database cannot be taken offline",
                    "Must maintain service during deployment",
                    "Budget allows for additional temporary infrastructure"
                ],
                "expected_phases": [
                    "Setup parallel infrastructure",
                    "Configure load balancer",
                    "Deploy to new infrastructure",
                    "Migrate traffic gradually",
                    "Monitor and validate",
                    "Cleanup old infrastructure"
                ]
            }
        ]
    
    async def run(self, model: LLMInterface) -> BenchmarkResult:
        """Run autonomous execution test"""
        results = []
        
        for test_case in self.test_cases:
            constraints_text = "\n".join(f"- {constraint}" for constraint in test_case["constraints"])
            
            prompt = f"""
            Create a detailed autonomous execution plan for the following goal:
            
            Goal: {test_case['goal']}
            
            Constraints:
            {constraints_text}
            
            Please provide:
            1. Step-by-step execution plan with phases
            2. Risk assessment and mitigation strategies
            3. Success criteria and validation steps
            4. Rollback procedures
            5. Resource requirements and timeline
            
            Format your response as:
            EXECUTION PLAN:
            Phase 1: [title]
            - [detailed steps]
            - [validation criteria]
            
            Phase 2: [title]
            - [detailed steps]
            - [validation criteria]
            
            [continue for all phases]
            
            RISK ASSESSMENT:
            [risks and mitigation strategies]
            
            SUCCESS CRITERIA:
            [measurable success criteria]
            
            ROLLBACK PROCEDURES:
            [rollback steps]
            
            RESOURCES:
            [resource requirements and timeline]
            """
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            response = await model.generate(prompt, max_tokens=1536)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Evaluate execution plan quality
            completeness = self._evaluate_plan_completeness(response, test_case)
            feasibility = self._evaluate_plan_feasibility(response, test_case)
            autonomy = self._evaluate_autonomy_level(response)
            
            results.append({
                "completeness": completeness,
                "feasibility": feasibility,
                "autonomy": autonomy,
                "latency_ms": (end_time - start_time) * 1000,
                "memory_usage_mb": end_memory - start_memory
            })
        
        # Aggregate results
        avg_completeness = statistics.mean([r["completeness"] for r in results])
        avg_feasibility = statistics.mean([r["feasibility"] for r in results])
        avg_autonomy = statistics.mean([r["autonomy"] for r in results])
        avg_latency = statistics.mean([r["latency_ms"] for r in results])
        avg_memory = statistics.mean([r["memory_usage_mb"] for r in results])
        
        return BenchmarkResult(
            model_name=model.get_model_info()["name"],
            capability=self.capability,
            test_name=self.name,
            score=(avg_completeness + avg_feasibility + avg_autonomy) / 3,
            latency_ms=avg_latency,
            memory_usage_mb=avg_memory,
            tokens_per_second=0,
            completeness=avg_completeness,
            accuracy=avg_feasibility,
            autonomy_score=avg_autonomy
        )
    
    def _evaluate_plan_completeness(self, response: str, test_case: Dict) -> float:
        """Evaluate completeness of execution plan"""
        required_sections = ["EXECUTION PLAN:", "RISK ASSESSMENT:", "SUCCESS CRITERIA:", "ROLLBACK PROCEDURES:", "RESOURCES:"]
        present_sections = sum(1 for section in required_sections if section in response)
        
        # Check for expected phases
        expected_phases = test_case["expected_phases"]
        covered_phases = 0
        response_lower = response.lower()
        
        for phase in expected_phases:
            phase_keywords = phase.lower().split()
            if any(keyword in response_lower for keyword in phase_keywords):
                covered_phases += 1
        
        section_completeness = present_sections / len(required_sections)
        phase_completeness = covered_phases / len(expected_phases)
        
        return (section_completeness + phase_completeness) / 2
    
    def _evaluate_plan_feasibility(self, response: str, test_case: Dict) -> float:
        """Evaluate feasibility of execution plan"""
        # Check if plan addresses constraints
        constraints = test_case["constraints"]
        constraints_addressed = 0
        
        response_lower = response.lower()
        for constraint in constraints:
            constraint_keywords = constraint.lower().split()
            if any(keyword in response_lower for keyword in constraint_keywords):
                constraints_addressed += 1
        
        return constraints_addressed / len(constraints)
    
    def _evaluate_autonomy_level(self, response: str) -> float:
        """Evaluate autonomy level of execution plan"""
        # Look for autonomous execution indicators
        autonomy_indicators = [
            "automated",
            "script",
            "monitoring",
            "validation",
            "rollback",
            "self-healing",
            "continuous",
            "pipeline"
        ]
        
        response_lower = response.lower()
        present_indicators = sum(1 for indicator in autonomy_indicators if indicator in response_lower)
        
        return present_indicators / len(autonomy_indicators)

class MetaLearningTest(BenchmarkTest):
    """Test meta-learning capability"""
    
    def __init__(self):
        super().__init__("meta_learning", "Meta-Learning")
        self.test_cases = [
            {
                "scenario": "You are tasked with optimizing a slow database query. Your first attempt using an index on column A improved performance by 20%. Your second attempt using an index on column B improved performance by 40%. Your third attempt using a composite index on columns A and B improved performance by 60%.",
                "new_situation": "You now have a similar slow query on a different table with columns X, Y, and Z. What approach would you take and why?",
                "expected_learning": [
                    "Composite indexes are more effective than single-column indexes",
                    "Should try composite index approach first",
                    "Performance improvements are cumulative",
                    "Should measure and compare different approaches"
                ]
            }
        ]
    
    async def run(self, model: LLMInterface) -> BenchmarkResult:
        """Run meta-learning test"""
        results = []
        
        for test_case in self.test_cases:
            prompt = f"""
            Consider this learning scenario:
            
            Scenario: {test_case['scenario']}
            
            New Situation: {test_case['new_situation']}
            
            Please provide:
            1. Analysis of what you learned from the scenario
            2. How you would apply this learning to the new situation
            3. What principles or patterns you extracted
            4. How you would adapt your approach if the first attempt fails
            
            Format your response as:
            LEARNING ANALYSIS:
            [what you learned from the scenario]
            
            APPLICATION:
            [how you would apply learning to new situation]
            
            PRINCIPLES:
            [extracted principles and patterns]
            
            ADAPTATION:
            [how you would adapt if first attempt fails]
            """
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            response = await model.generate(prompt, max_tokens=1024)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Evaluate meta-learning quality
            learning_extraction = self._evaluate_learning_extraction(response, test_case)
            application_quality = self._evaluate_application_quality(response, test_case)
            adaptation_capability = self._evaluate_adaptation_capability(response)
            
            results.append({
                "learning_extraction": learning_extraction,
                "application_quality": application_quality,
                "adaptation_capability": adaptation_capability,
                "latency_ms": (end_time - start_time) * 1000,
                "memory_usage_mb": end_memory - start_memory
            })
        
        # Aggregate results
        avg_learning = statistics.mean([r["learning_extraction"] for r in results])
        avg_application = statistics.mean([r["application_quality"] for r in results])
        avg_adaptation = statistics.mean([r["adaptation_capability"] for r in results])
        avg_latency = statistics.mean([r["latency_ms"] for r in results])
        avg_memory = statistics.mean([r["memory_usage_mb"] for r in results])
        
        return BenchmarkResult(
            model_name=model.get_model_info()["name"],
            capability=self.capability,
            test_name=self.name,
            score=(avg_learning + avg_application + avg_adaptation) / 3,
            latency_ms=avg_latency,
            memory_usage_mb=avg_memory,
            tokens_per_second=0,
            accuracy=avg_learning,
            completeness=avg_application,
            autonomy_score=avg_adaptation
        )
    
    def _evaluate_learning_extraction(self, response: str, test_case: Dict) -> float:
        """Evaluate quality of learning extraction"""
        expected_learning = test_case["expected_learning"]
        extracted_learning = 0
        
        response_lower = response.lower()
        for learning in expected_learning:
            learning_keywords = learning.lower().split()
            if any(keyword in response_lower for keyword in learning_keywords):
                extracted_learning += 1
        
        return extracted_learning / len(expected_learning)
    
    def _evaluate_application_quality(self, response: str, test_case: Dict) -> float:
        """Evaluate quality of learning application"""
        # Check if response includes specific application steps
        application_indicators = [
            "composite index",
            "columns x, y, z",
            "measure performance",
            "compare approaches",
            "start with"
        ]
        
        response_lower = response.lower()
        present_indicators = sum(1 for indicator in application_indicators if indicator in response_lower)
        
        return present_indicators / len(application_indicators)
    
    def _evaluate_adaptation_capability(self, response: str) -> float:
        """Evaluate adaptation capability"""
        adaptation_indicators = [
            "if",
            "fails",
            "alternative",
            "backup",
            "fallback",
            "try",
            "different",
            "approach"
        ]
        
        response_lower = response.lower()
        present_indicators = sum(1 for indicator in adaptation_indicators if indicator in response_lower)
        
        return min(1.0, present_indicators / len(adaptation_indicators))

class BenchmarkRunner:
    """Main benchmark runner class"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize test suites
        self.tests = [
            RecursiveTaskBreakdownTest(),
            MultiStepReasoningTest(),
            ContextMaintenanceTest(),
            CodeGenerationTest(),
            ResearchSynthesisTest(),
            AutonomousExecutionTest(),
            MetaLearningTest()
        ]
        
        self.results = []
        self.models = []
    
    def add_model(self, config: ModelConfig):
        """Add a model to benchmark"""
        self.models.append(config)
    
    async def run_benchmarks(self, model_names: List[str] = None):
        """Run benchmarks for specified models"""
        if model_names:
            models_to_test = [m for m in self.models if m.name in model_names]
        else:
            models_to_test = self.models
        
        total_tests = len(models_to_test) * len(self.tests)
        progress = tqdm(total=total_tests, desc="Running benchmarks")
        
        for model_config in models_to_test:
            logger.info(f"Benchmarking model: {model_config.name}")
            
            # Initialize model
            model = self._create_model(model_config)
            
            try:
                for test in self.tests:
                    logger.info(f"Running test: {test.name}")
                    
                    try:
                        result = await test.run(model)
                        self.results.append(result)
                        logger.info(f"Test {test.name} completed with score: {result.score:.3f}")
                    except Exception as e:
                        logger.error(f"Test {test.name} failed: {e}")
                        # Add error result
                        error_result = BenchmarkResult(
                            model_name=model_config.name,
                            capability=test.capability,
                            test_name=test.name,
                            score=0.0,
                            latency_ms=0,
                            memory_usage_mb=0,
                            tokens_per_second=0,
                            accuracy=0.0,
                            completeness=0.0,
                            consistency=0.0,
                            error_rate=1.0,
                            metadata={"error": str(e)}
                        )
                        self.results.append(error_result)
                    
                    progress.update(1)
                
            finally:
                model.cleanup()
        
        progress.close()
        
        # Generate reports
        self._generate_reports()
    
    def _create_model(self, config: ModelConfig) -> LLMInterface:
        """Create model instance based on configuration"""
        if config.model_type == "huggingface":
            return HuggingFaceModel(config)
        elif config.model_type == "ollama":
            return OllamaModel(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    def _generate_reports(self):
        """Generate benchmark reports"""
        # Save raw results
        results_data = [asdict(result) for result in self.results]
        
        with open(self.output_dir / "benchmark_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Create DataFrame for analysis
        df = pd.DataFrame(results_data)
        
        # Generate summary report
        self._generate_summary_report(df)
        
        # Generate capability analysis
        self._generate_capability_analysis(df)
        
        # Generate model comparison
        self._generate_model_comparison(df)
        
        # Generate visualizations
        self._generate_visualizations(df)
        
        # Generate recommendations
        self._generate_recommendations(df)
    
    def _generate_summary_report(self, df: pd.DataFrame):
        """Generate summary report"""
        summary = {
            "total_tests": len(df),
            "total_models": df['model_name'].nunique(),
            "total_capabilities": df['capability'].nunique(),
            "average_score": df['score'].mean(),
            "average_latency": df['latency_ms'].mean(),
            "average_memory": df['memory_usage_mb'].mean(),
            "top_performing_model": df.groupby('model_name')['score'].mean().idxmax(),
            "most_challenging_capability": df.groupby('capability')['score'].mean().idxmin(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.output_dir / "summary_report.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report generated: {self.output_dir / 'summary_report.json'}")
    
    def _generate_capability_analysis(self, df: pd.DataFrame):
        """Generate capability analysis report"""
        capability_stats = df.groupby('capability').agg({
            'score': ['mean', 'std', 'min', 'max'],
            'latency_ms': ['mean', 'std'],
            'memory_usage_mb': ['mean', 'std'],
            'accuracy': 'mean',
            'completeness': 'mean',
            'consistency': 'mean'
        }).round(3)
        
        capability_stats.to_csv(self.output_dir / "capability_analysis.csv")
        
        logger.info(f"Capability analysis generated: {self.output_dir / 'capability_analysis.csv'}")
    
    def _generate_model_comparison(self, df: pd.DataFrame):
        """Generate model comparison report"""
        model_stats = df.groupby('model_name').agg({
            'score': ['mean', 'std', 'min', 'max'],
            'latency_ms': ['mean', 'std'],
            'memory_usage_mb': ['mean', 'std'],
            'accuracy': 'mean',
            'completeness': 'mean',
            'consistency': 'mean'
        }).round(3)
        
        model_stats.to_csv(self.output_dir / "model_comparison.csv")
        
        logger.info(f"Model comparison generated: {self.output_dir / 'model_comparison.csv'}")
    
    def _generate_visualizations(self, df: pd.DataFrame):
        """Generate visualization charts"""
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # Model performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall score comparison
        model_scores = df.groupby('model_name')['score'].mean().sort_values(ascending=True)
        axes[0, 0].barh(model_scores.index, model_scores.values)
        axes[0, 0].set_title('Overall Performance Score by Model')
        axes[0, 0].set_xlabel('Average Score')
        
        # Capability heatmap
        capability_matrix = df.pivot_table(
            index='model_name', 
            columns='capability', 
            values='score', 
            aggfunc='mean'
        )
        sns.heatmap(capability_matrix, annot=True, fmt='.3f', ax=axes[0, 1])
        axes[0, 1].set_title('Capability Heatmap')
        
        # Latency comparison
        model_latency = df.groupby('model_name')['latency_ms'].mean().sort_values(ascending=True)
        axes[1, 0].barh(model_latency.index, model_latency.values)
        axes[1, 0].set_title('Average Latency by Model')
        axes[1, 0].set_xlabel('Latency (ms)')
        
        # Memory usage comparison
        model_memory = df.groupby('model_name')['memory_usage_mb'].mean().sort_values(ascending=True)
        axes[1, 1].barh(model_memory.index, model_memory.values)
        axes[1, 1].set_title('Average Memory Usage by Model')
        axes[1, 1].set_xlabel('Memory Usage (MB)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "benchmark_visualizations.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations generated: {self.output_dir / 'benchmark_visualizations.png'}")
    
    def _generate_recommendations(self, df: pd.DataFrame):
        """Generate model recommendations"""
        recommendations = {
            "overall_best": {
                "model": df.groupby('model_name')['score'].mean().idxmax(),
                "score": df.groupby('model_name')['score'].mean().max(),
                "reasoning": "Highest average score across all capabilities"
            },
            "best_for_capabilities": {},
            "performance_efficiency": {},
            "memory_efficiency": {},
            "balanced_choice": {},
            "specialized_recommendations": {}
        }
        
        # Best model for each capability
        for capability in df['capability'].unique():
            capability_df = df[df['capability'] == capability]
            best_model = capability_df.groupby('model_name')['score'].mean().idxmax()
            best_score = capability_df.groupby('model_name')['score'].mean().max()
            
            recommendations["best_for_capabilities"][capability] = {
                "model": best_model,
                "score": best_score
            }
        
        # Performance efficiency (score/latency ratio)
        df['efficiency'] = df['score'] / (df['latency_ms'] + 1)  # +1 to avoid division by zero
        efficiency_ranking = df.groupby('model_name')['efficiency'].mean().sort_values(ascending=False)
        recommendations["performance_efficiency"] = {
            "model": efficiency_ranking.index[0],
            "efficiency": efficiency_ranking.iloc[0],
            "reasoning": "Best score-to-latency ratio"
        }
        
        # Memory efficiency (score/memory ratio)
        df['memory_efficiency'] = df['score'] / (df['memory_usage_mb'] + 1)
        memory_efficiency_ranking = df.groupby('model_name')['memory_efficiency'].mean().sort_values(ascending=False)
        recommendations["memory_efficiency"] = {
            "model": memory_efficiency_ranking.index[0],
            "efficiency": memory_efficiency_ranking.iloc[0],
            "reasoning": "Best score-to-memory ratio"
        }
        
        # Balanced choice (normalized score + efficiency)
        model_stats = df.groupby('model_name').agg({
            'score': 'mean',
            'latency_ms': 'mean',
            'memory_usage_mb': 'mean'
        })
        
        # Normalize metrics
        model_stats['normalized_score'] = model_stats['score'] / model_stats['score'].max()
        model_stats['normalized_latency'] = 1 - (model_stats['latency_ms'] / model_stats['latency_ms'].max())
        model_stats['normalized_memory'] = 1 - (model_stats['memory_usage_mb'] / model_stats['memory_usage_mb'].max())
        
        model_stats['balanced_score'] = (
            model_stats['normalized_score'] * 0.5 +
            model_stats['normalized_latency'] * 0.3 +
            model_stats['normalized_memory'] * 0.2
        )
        
        best_balanced = model_stats['balanced_score'].idxmax()
        recommendations["balanced_choice"] = {
            "model": best_balanced,
            "score": model_stats.loc[best_balanced, 'balanced_score'],
            "reasoning": "Best balance of performance, speed, and memory usage"
        }
        
        # Specialized recommendations
        recommendations["specialized_recommendations"] = {
            "code_generation": self._get_specialized_recommendation(df, "Code Generation & Analysis"),
            "reasoning": self._get_specialized_recommendation(df, "Multi-Step Reasoning"),
            "context_maintenance": self._get_specialized_recommendation(df, "Context Maintenance"),
            "autonomous_execution": self._get_specialized_recommendation(df, "Autonomous Execution Planning")
        }
        
        with open(self.output_dir / "recommendations.json", 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        logger.info(f"Recommendations generated: {self.output_dir / 'recommendations.json'}")
        
        # Generate human-readable recommendation report
        self._generate_recommendation_report(recommendations)
    
    def _get_specialized_recommendation(self, df: pd.DataFrame, capability: str) -> Dict:
        """Get specialized recommendation for a capability"""
        capability_df = df[df['capability'] == capability]
        if capability_df.empty:
            return {"model": "N/A", "score": 0, "reasoning": "No data available"}
        
        best_model = capability_df.groupby('model_name')['score'].mean().idxmax()
        best_score = capability_df.groupby('model_name')['score'].mean().max()
        
        return {
            "model": best_model,
            "score": best_score,
            "reasoning": f"Highest score for {capability}"
        }
    
    def _generate_recommendation_report(self, recommendations: Dict):
        """Generate human-readable recommendation report"""
        report = f"""
# LLM Benchmark Recommendations Report

## Executive Summary

Based on comprehensive benchmarking across 7 core capabilities, here are the key findings and recommendations:

### Overall Best Model
**{recommendations['overall_best']['model']}** (Score: {recommendations['overall_best']['score']:.3f})
- {recommendations['overall_best']['reasoning']}

### Performance Efficiency Champion
**{recommendations['performance_efficiency']['model']}** (Efficiency: {recommendations['performance_efficiency']['efficiency']:.3f})
- {recommendations['performance_efficiency']['reasoning']}

### Memory Efficiency Champion
**{recommendations['memory_efficiency']['model']}** (Efficiency: {recommendations['memory_efficiency']['efficiency']:.3f})
- {recommendations['memory_efficiency']['reasoning']}

### Balanced Choice
**{recommendations['balanced_choice']['model']}** (Balanced Score: {recommendations['balanced_choice']['score']:.3f})
- {recommendations['balanced_choice']['reasoning']}

## Capability-Specific Recommendations

### Code Generation & Analysis
- **Best Model**: {recommendations['specialized_recommendations']['code_generation']['model']}
- **Score**: {recommendations['specialized_recommendations']['code_generation']['score']:.3f}
- **Use Case**: Optimal for automated code generation, code review, and programming assistance

### Multi-Step Reasoning
- **Best Model**: {recommendations['specialized_recommendations']['reasoning']['model']}
- **Score**: {recommendations['specialized_recommendations']['reasoning']['score']:.3f}
- **Use Case**: Ideal for complex problem-solving and logical reasoning tasks

### Context Maintenance
- **Best Model**: {recommendations['specialized_recommendations']['context_maintenance']['model']}
- **Score**: {recommendations['specialized_recommendations']['context_maintenance']['score']:.3f}
- **Use Case**: Essential for long conversations and maintaining context across interactions

### Autonomous Execution Planning
- **Best Model**: {recommendations['specialized_recommendations']['autonomous_execution']['model']}
- **Score**: {recommendations['specialized_recommendations']['autonomous_execution']['score']:.3f}
- **Use Case**: Critical for autonomous workflow planning and execution

## Deployment Recommendations

### For Production Task Master AI:
1. **Primary Model**: Use {recommendations['overall_best']['model']} for general capabilities
2. **Specialized Models**: Consider model routing based on task type
3. **Fallback Strategy**: Implement {recommendations['balanced_choice']['model']} as fallback
4. **Resource Optimization**: Use {recommendations['memory_efficiency']['model']} for resource-constrained environments

### For Development and Testing:
1. **Fast Iteration**: Use {recommendations['performance_efficiency']['model']} for rapid development
2. **Comprehensive Testing**: Validate with multiple top-performing models
3. **Capability Testing**: Use specialized models for specific capability validation

## Implementation Strategy

1. **Phase 1**: Deploy {recommendations['overall_best']['model']} as primary model
2. **Phase 2**: Implement model routing for specialized tasks
3. **Phase 3**: Add performance monitoring and adaptive model selection
4. **Phase 4**: Optimize based on production usage patterns

## Next Steps

1. Conduct integration testing with selected models
2. Implement model serving infrastructure
3. Set up monitoring and performance tracking
4. Plan for model updates and versioning
5. Establish model performance baselines

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        with open(self.output_dir / "recommendation_report.md", 'w') as f:
            f.write(report.strip())
        
        logger.info(f"Recommendation report generated: {self.output_dir / 'recommendation_report.md'}")

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LLM Capability Benchmark")
    parser.add_argument("--config", default="benchmark_config.json", help="Configuration file path")
    parser.add_argument("--models", nargs="+", help="Specific models to benchmark")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--run-all", action="store_true", help="Run all configured models")
    
    args = parser.parse_args()
    
    # Create default config if it doesn't exist
    if not Path(args.config).exists():
        default_config = {
            "models": [
                {
                    "name": "llama-3.1-70b-awq",
                    "model_path": "huggingface/llama-3.1-70b-instruct-awq",
                    "model_type": "huggingface",
                    "quantization": "4bit",
                    "max_tokens": 8192
                },
                {
                    "name": "mistral-7b-instruct",
                    "model_path": "mistralai/Mistral-7B-Instruct-v0.1",
                    "model_type": "huggingface",
                    "max_tokens": 8192
                },
                {
                    "name": "codellama-13b-instruct",
                    "model_path": "codellama/CodeLlama-13b-Instruct-hf",
                    "model_type": "huggingface",
                    "max_tokens": 8192
                },
                {
                    "name": "qwen2.5-14b-instruct",
                    "model_path": "Qwen/Qwen2.5-14B-Instruct",
                    "model_type": "huggingface",
                    "max_tokens": 8192
                }
            ]
        }
        
        with open(args.config, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default configuration: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(args.output)
    
    # Add models from configuration
    for model_config in config["models"]:
        runner.add_model(ModelConfig(**model_config))
    
    # Run benchmarks
    asyncio.run(runner.run_benchmarks(args.models if not args.run_all else None))
    
    logger.info("Benchmark completed successfully!")

if __name__ == "__main__":
    main()