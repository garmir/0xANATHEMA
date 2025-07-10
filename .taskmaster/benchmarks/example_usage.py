#!/usr/bin/env python3
"""
Example Usage of LLM Capability Benchmarking Framework
======================================================

This script demonstrates how to use the benchmarking framework
for evaluating local LLMs for Task Master AI.
"""

import asyncio
import json
import logging
from pathlib import Path

from llm_capability_benchmark import (
    BenchmarkRunner,
    ModelConfig,
    RecursiveTaskBreakdownTest,
    MultiStepReasoningTest,
    CodeGenerationTest
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def simple_benchmark_example():
    """Simple benchmark example with one model and one test"""
    print("=" * 50)
    print("Simple Benchmark Example")
    print("=" * 50)
    
    # Create benchmark runner
    runner = BenchmarkRunner("example_results/simple")
    
    # Add a model (using Ollama for simplicity)
    model_config = ModelConfig(
        name="example-llama",
        model_path="llama3.1:8b",
        model_type="ollama",
        api_endpoint="http://localhost:11434"
    )
    runner.add_model(model_config)
    
    # Use only one test for demonstration
    runner.tests = [RecursiveTaskBreakdownTest()]
    
    # Run benchmark
    print("Running benchmark...")
    await runner.run_benchmarks()
    
    print("Simple benchmark completed! Check example_results/simple/ for results.")

async def comprehensive_benchmark_example():
    """Comprehensive benchmark example with multiple models and tests"""
    print("=" * 50)
    print("Comprehensive Benchmark Example")
    print("=" * 50)
    
    # Create benchmark runner
    runner = BenchmarkRunner("example_results/comprehensive")
    
    # Add multiple models
    models = [
        ModelConfig(
            name="llama-8b",
            model_path="llama3.1:8b",
            model_type="ollama",
            api_endpoint="http://localhost:11434"
        ),
        ModelConfig(
            name="mistral-7b",
            model_path="mistral:7b",
            model_type="ollama",
            api_endpoint="http://localhost:11434"
        )
    ]
    
    for model in models:
        runner.add_model(model)
    
    # Use multiple tests
    runner.tests = [
        RecursiveTaskBreakdownTest(),
        MultiStepReasoningTest(),
        CodeGenerationTest()
    ]
    
    # Run benchmark
    print("Running comprehensive benchmark...")
    await runner.run_benchmarks()
    
    print("Comprehensive benchmark completed! Check example_results/comprehensive/ for results.")

async def custom_test_example():
    """Example of creating and running a custom test"""
    print("=" * 50)
    print("Custom Test Example")
    print("=" * 50)
    
    from llm_capability_benchmark import BenchmarkTest, BenchmarkResult
    
    class SimpleTaskTest(BenchmarkTest):
        """Simple custom test for demonstration"""
        
        def __init__(self):
            super().__init__("simple_task", "Simple Task Capability")
        
        async def run(self, model):
            """Run a simple test"""
            import time
            
            prompt = "List 5 benefits of using local LLMs for enterprise applications:"
            
            start_time = time.time()
            response = await model.generate(prompt, max_tokens=256)
            end_time = time.time()
            
            # Simple scoring based on response length and keywords
            keywords = ["privacy", "security", "cost", "control", "performance", "compliance"]
            keyword_count = sum(1 for keyword in keywords if keyword.lower() in response.lower())
            
            score = min(1.0, keyword_count / len(keywords))
            latency_ms = (end_time - start_time) * 1000
            
            return BenchmarkResult(
                model_name=model.get_model_info()["name"],
                capability=self.capability,
                test_name=self.name,
                score=score,
                latency_ms=latency_ms,
                memory_usage_mb=0,  # Simplified for example
                tokens_per_second=0,
                accuracy=score,
                completeness=score,
                consistency=score
            )
    
    # Create runner with custom test
    runner = BenchmarkRunner("example_results/custom")
    
    # Add model
    model_config = ModelConfig(
        name="test-model",
        model_path="llama3.1:8b",
        model_type="ollama",
        api_endpoint="http://localhost:11434"
    )
    runner.add_model(model_config)
    
    # Use custom test
    runner.tests = [SimpleTaskTest()]
    
    print("Running custom test...")
    await runner.run_benchmarks()
    
    print("Custom test completed! Check example_results/custom/ for results.")

def analyze_results_example():
    """Example of analyzing benchmark results"""
    print("=" * 50)
    print("Results Analysis Example")
    print("=" * 50)
    
    # Example analysis of hypothetical results
    example_results = [
        {
            "model_name": "llama-8b",
            "capability": "Recursive Task Breakdown",
            "score": 0.85,
            "latency_ms": 1200,
            "memory_usage_mb": 2048
        },
        {
            "model_name": "mistral-7b",
            "capability": "Recursive Task Breakdown",
            "score": 0.78,
            "latency_ms": 900,
            "memory_usage_mb": 1536
        }
    ]
    
    print("Example Results Analysis:")
    print(json.dumps(example_results, indent=2))
    
    # Simple analysis
    best_score = max(example_results, key=lambda x: x["score"])
    fastest_model = min(example_results, key=lambda x: x["latency_ms"])
    most_efficient = min(example_results, key=lambda x: x["memory_usage_mb"])
    
    print(f"\nBest performing model: {best_score['model_name']} (score: {best_score['score']})")
    print(f"Fastest model: {fastest_model['model_name']} (latency: {fastest_model['latency_ms']}ms)")
    print(f"Most memory efficient: {most_efficient['model_name']} (memory: {most_efficient['memory_usage_mb']}MB)")

async def task_master_specific_example():
    """Example focused on Task Master AI specific capabilities"""
    print("=" * 50)
    print("Task Master AI Specific Example")
    print("=" * 50)
    
    # This example would use Task Master specific tests
    # focusing on recursive breakdown and autonomous execution
    
    runner = BenchmarkRunner("example_results/task_master")
    
    # Add model optimized for Task Master workloads
    model_config = ModelConfig(
        name="task-master-optimized",
        model_path="llama3.1:70b",  # Larger model for better reasoning
        model_type="ollama",
        api_endpoint="http://localhost:11434",
        temperature=0.1,  # Lower temperature for more consistent results
        max_tokens=2048   # Higher token limit for complex tasks
    )
    runner.add_model(model_config)
    
    # Focus on Task Master core capabilities
    from llm_capability_benchmark import AutonomousExecutionTest, MetaLearningTest
    
    runner.tests = [
        RecursiveTaskBreakdownTest(),  # Core capability for task decomposition
        AutonomousExecutionTest(),     # Essential for workflow planning
        MetaLearningTest()             # Important for system improvement
    ]
    
    print("Running Task Master AI specific benchmark...")
    await runner.run_benchmarks()
    
    print("Task Master benchmark completed! Check example_results/task_master/ for results.")
    
    # Load and analyze results
    results_file = Path("example_results/task_master/recommendations.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            recommendations = json.load(f)
        
        print("\nKey Recommendations:")
        print(f"Overall best model: {recommendations.get('overall_best', {}).get('model', 'N/A')}")
        print(f"Best for autonomous execution: {recommendations.get('specialized_recommendations', {}).get('autonomous_execution', {}).get('model', 'N/A')}")

def main():
    """Main function demonstrating various usage patterns"""
    print("LLM Capability Benchmarking Framework - Usage Examples")
    print("=====================================================")
    
    # Create example results directory
    Path("example_results").mkdir(exist_ok=True)
    
    print("\n1. Simple Benchmark Example")
    print("   - Single model, single test")
    print("   - Good for quick evaluation")
    
    print("\n2. Comprehensive Benchmark Example")
    print("   - Multiple models, multiple tests")
    print("   - Full capability comparison")
    
    print("\n3. Custom Test Example")
    print("   - Creating custom evaluation criteria")
    print("   - Extending the framework")
    
    print("\n4. Results Analysis Example")
    print("   - Interpreting benchmark results")
    print("   - Making data-driven decisions")
    
    print("\n5. Task Master AI Specific Example")
    print("   - Focus on Task Master capabilities")
    print("   - Production readiness evaluation")
    
    choice = input("\nWhich example would you like to run? (1-5, or 'all'): ").strip()
    
    if choice == "1":
        asyncio.run(simple_benchmark_example())
    elif choice == "2":
        asyncio.run(comprehensive_benchmark_example())
    elif choice == "3":
        asyncio.run(custom_test_example())
    elif choice == "4":
        analyze_results_example()
    elif choice == "5":
        asyncio.run(task_master_specific_example())
    elif choice.lower() == "all":
        print("\nRunning all examples...")
        asyncio.run(simple_benchmark_example())
        asyncio.run(comprehensive_benchmark_example())
        asyncio.run(custom_test_example())
        analyze_results_example()
        asyncio.run(task_master_specific_example())
    else:
        print("Invalid choice. Please run the script again and select 1-5 or 'all'.")

if __name__ == "__main__":
    main()