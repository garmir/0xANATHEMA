#!/usr/bin/env python3
"""
Benchmark Suite Runner for Task Master AI LLM Evaluation
========================================================

This script provides convenient interfaces for running specific benchmark
subsets and generating focused reports.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from llm_capability_benchmark import (
    BenchmarkRunner,
    ModelConfig,
    RecursiveTaskBreakdownTest,
    MultiStepReasoningTest,
    ContextMaintenanceTest,
    CodeGenerationTest,
    ResearchSynthesisTest,
    AutonomousExecutionTest,
    MetaLearningTest
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkSuiteRunner:
    """Enhanced benchmark runner with suite-specific functionality"""
    
    def __init__(self, config_path: str = "benchmark_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.test_suites = {
            "core_capabilities": [
                RecursiveTaskBreakdownTest(),
                MultiStepReasoningTest(),
                ContextMaintenanceTest()
            ],
            "code_and_autonomy": [
                CodeGenerationTest(),
                AutonomousExecutionTest()
            ],
            "research_and_learning": [
                ResearchSynthesisTest(),
                MetaLearningTest()
            ],
            "task_master_specific": [
                RecursiveTaskBreakdownTest(),
                AutonomousExecutionTest(),
                MetaLearningTest()
            ],
            "performance_focused": [
                MultiStepReasoningTest(),
                CodeGenerationTest(),
                ContextMaintenanceTest()
            ]
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    async def run_suite(self, suite_name: str, models: List[str] = None, output_dir: str = "results"):
        """Run a specific test suite"""
        if suite_name not in self.test_suites:
            logger.error(f"Unknown test suite: {suite_name}")
            logger.info(f"Available suites: {list(self.test_suites.keys())}")
            return
        
        logger.info(f"Running test suite: {suite_name}")
        
        # Create specialized runner
        runner = BenchmarkRunner(f"{output_dir}/{suite_name}")
        
        # Override tests with suite-specific tests
        runner.tests = self.test_suites[suite_name]
        
        # Add models from configuration
        for model_config in self.config["models"]:
            if models is None or model_config["name"] in models:
                runner.add_model(ModelConfig(**model_config))
        
        # Run benchmarks
        await runner.run_benchmarks(models)
        
        logger.info(f"Suite {suite_name} completed. Results in {output_dir}/{suite_name}")
    
    async def run_capability_comparison(self, capability: str, models: List[str] = None, output_dir: str = "results"):
        """Run comparison for a specific capability across all models"""
        capability_tests = {
            "recursive_breakdown": [RecursiveTaskBreakdownTest()],
            "reasoning": [MultiStepReasoningTest()],
            "context": [ContextMaintenanceTest()],
            "code": [CodeGenerationTest()],
            "research": [ResearchSynthesisTest()],
            "autonomy": [AutonomousExecutionTest()],
            "meta_learning": [MetaLearningTest()]
        }
        
        if capability not in capability_tests:
            logger.error(f"Unknown capability: {capability}")
            logger.info(f"Available capabilities: {list(capability_tests.keys())}")
            return
        
        logger.info(f"Running capability comparison: {capability}")
        
        # Create specialized runner
        runner = BenchmarkRunner(f"{output_dir}/capability_{capability}")
        runner.tests = capability_tests[capability]
        
        # Add all models for comparison
        for model_config in self.config["models"]:
            if models is None or model_config["name"] in models:
                runner.add_model(ModelConfig(**model_config))
        
        # Run benchmarks
        await runner.run_benchmarks(models)
        
        logger.info(f"Capability {capability} comparison completed")
    
    async def run_quick_evaluation(self, models: List[str] = None, output_dir: str = "results"):
        """Run a quick evaluation with reduced test cases"""
        logger.info("Running quick evaluation")
        
        # Create runner with minimal tests
        runner = BenchmarkRunner(f"{output_dir}/quick_eval")
        
        # Use only one test from each capability for speed
        runner.tests = [
            RecursiveTaskBreakdownTest(),
            MultiStepReasoningTest(),
            CodeGenerationTest()
        ]
        
        # Add models
        for model_config in self.config["models"]:
            if models is None or model_config["name"] in models:
                runner.add_model(ModelConfig(**model_config))
        
        # Run benchmarks
        await runner.run_benchmarks(models)
        
        logger.info("Quick evaluation completed")
    
    async def run_model_selection_benchmark(self, output_dir: str = "results"):
        """Run comprehensive benchmark for model selection"""
        logger.info("Running comprehensive model selection benchmark")
        
        # Create comprehensive runner
        runner = BenchmarkRunner(f"{output_dir}/model_selection")
        
        # Use all tests
        runner.tests = [
            RecursiveTaskBreakdownTest(),
            MultiStepReasoningTest(),
            ContextMaintenanceTest(),
            CodeGenerationTest(),
            ResearchSynthesisTest(),
            AutonomousExecutionTest(),
            MetaLearningTest()
        ]
        
        # Add all configured models
        for model_config in self.config["models"]:
            runner.add_model(ModelConfig(**model_config))
        
        # Run benchmarks
        await runner.run_benchmarks()
        
        # Generate model selection report
        await self._generate_model_selection_report(f"{output_dir}/model_selection")
        
        logger.info("Model selection benchmark completed")
    
    async def _generate_model_selection_report(self, results_dir: str):
        """Generate specialized model selection report"""
        import pandas as pd
        
        results_file = Path(results_dir) / "benchmark_results.json"
        if not results_file.exists():
            logger.error("Results file not found")
            return
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        df = pd.DataFrame(results)
        
        # Generate model selection insights
        model_selection_report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "models_evaluated": df['model_name'].nunique(),
            "capabilities_tested": df['capability'].nunique(),
            "total_tests": len(df),
            "top_performers": {
                "overall": df.groupby('model_name')['score'].mean().nlargest(3).to_dict(),
                "by_capability": {}
            },
            "performance_characteristics": {},
            "resource_efficiency": {},
            "recommendations": {}
        }
        
        # Capability-specific top performers
        for capability in df['capability'].unique():
            cap_df = df[df['capability'] == capability]
            model_selection_report["top_performers"]["by_capability"][capability] = \
                cap_df.groupby('model_name')['score'].mean().nlargest(3).to_dict()
        
        # Performance characteristics
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            model_selection_report["performance_characteristics"][model] = {
                "avg_score": model_df['score'].mean(),
                "score_std": model_df['score'].std(),
                "avg_latency": model_df['latency_ms'].mean(),
                "avg_memory": model_df['memory_usage_mb'].mean(),
                "consistency": model_df['consistency'].mean() if 'consistency' in model_df.columns else None
            }
        
        # Resource efficiency ranking
        df['efficiency_score'] = df['score'] / (df['latency_ms'] / 1000 + df['memory_usage_mb'] / 1000)
        efficiency_ranking = df.groupby('model_name')['efficiency_score'].mean().sort_values(ascending=False)
        model_selection_report["resource_efficiency"] = efficiency_ranking.to_dict()
        
        # Generate recommendations
        overall_best = df.groupby('model_name')['score'].mean().idxmax()
        fastest = df.groupby('model_name')['latency_ms'].mean().idxmin()
        most_efficient = efficiency_ranking.index[0]
        
        model_selection_report["recommendations"] = {
            "primary_choice": {
                "model": overall_best,
                "reasoning": "Highest overall performance across all capabilities"
            },
            "speed_optimized": {
                "model": fastest,
                "reasoning": "Lowest average latency for time-critical applications"
            },
            "resource_optimized": {
                "model": most_efficient,
                "reasoning": "Best balance of performance and resource usage"
            },
            "use_case_specific": {
                "code_generation": df[df['capability'] == 'Code Generation & Analysis'].groupby('model_name')['score'].mean().idxmax(),
                "reasoning_tasks": df[df['capability'] == 'Multi-Step Reasoning'].groupby('model_name')['score'].mean().idxmax(),
                "long_context": df[df['capability'] == 'Context Maintenance'].groupby('model_name')['score'].mean().idxmax()
            }
        }
        
        # Save model selection report
        output_file = Path(results_dir) / "model_selection_report.json"
        with open(output_file, 'w') as f:
            json.dump(model_selection_report, f, indent=2)
        
        logger.info(f"Model selection report generated: {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Benchmark Suite Runner")
    parser.add_argument("--config", default="benchmark_config.json", help="Configuration file")
    parser.add_argument("--output", default="results", help="Output directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Suite command
    suite_parser = subparsers.add_parser("suite", help="Run a specific test suite")
    suite_parser.add_argument("suite_name", choices=[
        "core_capabilities", "code_and_autonomy", "research_and_learning",
        "task_master_specific", "performance_focused"
    ], help="Test suite to run")
    suite_parser.add_argument("--models", nargs="+", help="Specific models to test")
    
    # Capability command
    capability_parser = subparsers.add_parser("capability", help="Compare models on specific capability")
    capability_parser.add_argument("capability", choices=[
        "recursive_breakdown", "reasoning", "context", "code", "research", "autonomy", "meta_learning"
    ], help="Capability to test")
    capability_parser.add_argument("--models", nargs="+", help="Specific models to test")
    
    # Quick command
    quick_parser = subparsers.add_parser("quick", help="Run quick evaluation")
    quick_parser.add_argument("--models", nargs="+", help="Specific models to test")
    
    # Selection command
    selection_parser = subparsers.add_parser("selection", help="Run comprehensive model selection benchmark")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available suites, capabilities, and models")
    list_parser.add_argument("--type", choices=["suites", "capabilities", "models"], 
                            default="all", help="What to list")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    runner = BenchmarkSuiteRunner(args.config)
    
    if args.command == "list":
        if args.type in ["suites", "all"]:
            print("Available test suites:")
            for suite in runner.test_suites.keys():
                print(f"  - {suite}")
        
        if args.type in ["capabilities", "all"]:
            print("\nAvailable capabilities:")
            capabilities = ["recursive_breakdown", "reasoning", "context", "code", "research", "autonomy", "meta_learning"]
            for cap in capabilities:
                print(f"  - {cap}")
        
        if args.type in ["models", "all"]:
            print("\nConfigured models:")
            for model in runner.config["models"]:
                print(f"  - {model['name']} ({model['model_type']})")
        
        return
    
    # Run async commands
    if args.command == "suite":
        asyncio.run(runner.run_suite(args.suite_name, args.models, args.output))
    elif args.command == "capability":
        asyncio.run(runner.run_capability_comparison(args.capability, args.models, args.output))
    elif args.command == "quick":
        asyncio.run(runner.run_quick_evaluation(args.models, args.output))
    elif args.command == "selection":
        asyncio.run(runner.run_model_selection_benchmark(args.output))

if __name__ == "__main__":
    main()