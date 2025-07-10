#!/usr/bin/env python3
"""
Comprehensive Evaluation and Benchmarking System for Self-Improving Architecture
Task 50.5 / ml.7 Implementation
"""

import json
import os
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple
import glob

class ArchitectureEvaluator:
    """Comprehensive evaluation system for recursive meta-learning architectures"""
    
    def __init__(self, results_dir: str = ".taskmaster/training_results", 
                 progress_dir: str = ".taskmaster/training_progress"):
        self.results_dir = results_dir
        self.progress_dir = progress_dir
        self.evaluation_metrics = {}
        self.benchmark_results = {}
        
    def load_training_data(self) -> Dict[str, Any]:
        """Load all available training results and progress data"""
        training_data = {
            "pipeline_results": [],
            "training_progress": [],
            "architectures": {},
            "performance_timeline": []
        }
        
        # Load pipeline results
        result_files = glob.glob(f"{self.results_dir}/pipeline_results_*.json")
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    training_data["pipeline_results"].append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Load training progress
        progress_files = glob.glob(f"{self.progress_dir}/training_progress_*.json")
        for file_path in progress_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    training_data["training_progress"].extend(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return training_data
    
    def analyze_architecture_performance(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance of different architectures discovered"""
        architecture_analysis = {}
        
        # Group results by architecture
        for result in training_data["pipeline_results"]:
            if "best_architecture" in result:
                arch = result["best_architecture"]
                arch_id = arch.get("id", "unknown")
                
                if arch_id not in architecture_analysis:
                    architecture_analysis[arch_id] = {
                        "architecture": arch,
                        "scores": [],
                        "iterations": 0,
                        "avg_score": 0,
                        "best_score": 0,
                        "stability": 0
                    }
                
                score = result.get("best_overall_score", 0)
                architecture_analysis[arch_id]["scores"].append(score)
                architecture_analysis[arch_id]["iterations"] += result.get("search_iterations_completed", 0)
                architecture_analysis[arch_id]["best_score"] = max(
                    architecture_analysis[arch_id]["best_score"], score
                )
        
        # Calculate statistics for each architecture
        for arch_id, data in architecture_analysis.items():
            if data["scores"]:
                data["avg_score"] = statistics.mean(data["scores"])
                data["stability"] = 1.0 - (statistics.stdev(data["scores"]) if len(data["scores"]) > 1 else 0)
        
        return architecture_analysis
    
    def evaluate_meta_learning_capabilities(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the meta-learning and adaptation capabilities"""
        meta_learning_metrics = {
            "adaptation_speed": [],
            "generalization_scores": [],
            "task_transfer_rates": [],
            "convergence_stability": [],
            "meta_learning_scores": []
        }
        
        # Analyze training progress for meta-learning indicators
        for progress_entry in training_data["training_progress"]:
            if "meta_learning_score" in progress_entry:
                meta_learning_metrics["meta_learning_scores"].append(
                    progress_entry["meta_learning_score"]
                )
            
            if "convergence_rate" in progress_entry:
                meta_learning_metrics["convergence_stability"].append(
                    progress_entry["convergence_rate"]
                )
        
        # Analyze validation results for adaptation metrics
        for result in training_data["pipeline_results"]:
            if "validation_results" in result:
                for val_result in result["validation_results"]:
                    if "adaptation_speed" in val_result:
                        meta_learning_metrics["adaptation_speed"].append(
                            val_result["adaptation_speed"]
                        )
                    if "generalization_score" in val_result:
                        meta_learning_metrics["generalization_scores"].append(
                            val_result["generalization_score"]
                        )
        
        # Calculate aggregate statistics
        aggregate_metrics = {}
        for metric_name, values in meta_learning_metrics.items():
            if values:
                aggregate_metrics[f"{metric_name}_mean"] = statistics.mean(values)
                aggregate_metrics[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0
                aggregate_metrics[f"{metric_name}_max"] = max(values)
                aggregate_metrics[f"{metric_name}_min"] = min(values)
        
        return {
            "raw_metrics": meta_learning_metrics,
            "aggregate_metrics": aggregate_metrics
        }
    
    def benchmark_recursive_improvement(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark the recursive self-improvement capabilities"""
        improvement_metrics = {
            "iteration_improvements": [],
            "architecture_evolution": [],
            "performance_trajectory": [],
            "recursive_depth_achieved": 0
        }
        
        # Track performance improvements over iterations
        for result in training_data["pipeline_results"]:
            iterations = result.get("search_iterations_completed", 0)
            score = result.get("best_overall_score", 0)
            
            improvement_metrics["performance_trajectory"].append({
                "iterations": iterations,
                "score": score,
                "timestamp": result.get("timestamp", datetime.now().isoformat())
            })
            
            # Track recursive depth based on pipeline stages
            stages = result.get("pipeline_stages", [])
            recursive_stages = [s for s in stages if "recursive" in s.lower()]
            improvement_metrics["recursive_depth_achieved"] = max(
                improvement_metrics["recursive_depth_achieved"],
                len(recursive_stages)
            )
        
        # Calculate improvement rates
        if len(improvement_metrics["performance_trajectory"]) > 1:
            trajectory = improvement_metrics["performance_trajectory"]
            for i in range(1, len(trajectory)):
                prev_score = trajectory[i-1]["score"]
                curr_score = trajectory[i]["score"]
                improvement = (curr_score - prev_score) / max(prev_score, 0.001)
                improvement_metrics["iteration_improvements"].append(improvement)
        
        # Calculate benchmark scores
        benchmark_scores = {
            "average_improvement_rate": statistics.mean(improvement_metrics["iteration_improvements"]) if improvement_metrics["iteration_improvements"] else 0,
            "max_improvement_rate": max(improvement_metrics["iteration_improvements"]) if improvement_metrics["iteration_improvements"] else 0,
            "recursive_depth_score": improvement_metrics["recursive_depth_achieved"] / 10.0,  # Normalize to 0-1
            "stability_score": self._calculate_stability_score(improvement_metrics["performance_trajectory"])
        }
        
        return {
            "improvement_metrics": improvement_metrics,
            "benchmark_scores": benchmark_scores
        }
    
    def _calculate_stability_score(self, trajectory: List[Dict]) -> float:
        """Calculate stability score based on performance trajectory"""
        if len(trajectory) < 2:
            return 1.0
        
        scores = [entry["score"] for entry in trajectory]
        variance = statistics.variance(scores)
        mean_score = statistics.mean(scores)
        
        # Stability is inverse of coefficient of variation
        cv = variance / max(mean_score, 0.001)
        stability = 1.0 / (1.0 + cv)
        
        return min(stability, 1.0)
    
    def generate_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        print("Loading training data...")
        training_data = self.load_training_data()
        
        print("Analyzing architecture performance...")
        architecture_analysis = self.analyze_architecture_performance(training_data)
        
        print("Evaluating meta-learning capabilities...")
        meta_learning_eval = self.evaluate_meta_learning_capabilities(training_data)
        
        print("Benchmarking recursive improvement...")
        recursive_benchmark = self.benchmark_recursive_improvement(training_data)
        
        # Calculate overall system score
        overall_scores = self._calculate_overall_scores(
            architecture_analysis, meta_learning_eval, recursive_benchmark
        )
        
        benchmark_report = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_summary": {
                "architectures_evaluated": len(architecture_analysis),
                "total_training_samples": len(training_data["training_progress"]),
                "total_pipeline_runs": len(training_data["pipeline_results"])
            },
            "architecture_analysis": architecture_analysis,
            "meta_learning_evaluation": meta_learning_eval,
            "recursive_improvement_benchmark": recursive_benchmark,
            "overall_scores": overall_scores,
            "recommendations": self._generate_recommendations(
                architecture_analysis, meta_learning_eval, recursive_benchmark
            )
        }
        
        return benchmark_report
    
    def _calculate_overall_scores(self, arch_analysis: Dict, meta_eval: Dict, 
                                recursive_bench: Dict) -> Dict[str, float]:
        """Calculate overall system performance scores"""
        scores = {}
        
        # Architecture quality score
        if arch_analysis:
            arch_scores = [data["best_score"] for data in arch_analysis.values()]
            scores["architecture_quality"] = statistics.mean(arch_scores) if arch_scores else 0
        else:
            scores["architecture_quality"] = 0
        
        # Meta-learning effectiveness score
        meta_metrics = meta_eval.get("aggregate_metrics", {})
        scores["meta_learning_effectiveness"] = meta_metrics.get("meta_learning_scores_mean", 0)
        
        # Recursive improvement score
        recursive_scores = recursive_bench.get("benchmark_scores", {})
        scores["recursive_improvement"] = statistics.mean([
            recursive_scores.get("average_improvement_rate", 0),
            recursive_scores.get("recursive_depth_score", 0),
            recursive_scores.get("stability_score", 0)
        ])
        
        # Overall system score (weighted average)
        scores["overall_system_score"] = (
            0.4 * scores["architecture_quality"] +
            0.35 * scores["meta_learning_effectiveness"] +
            0.25 * scores["recursive_improvement"]
        )
        
        return scores
    
    def _generate_recommendations(self, arch_analysis: Dict, meta_eval: Dict, 
                                recursive_bench: Dict) -> List[str]:
        """Generate recommendations for system improvement"""
        recommendations = []
        
        # Architecture recommendations
        if arch_analysis:
            best_arch = max(arch_analysis.values(), key=lambda x: x["best_score"])
            recommendations.append(
                f"Best performing architecture: {best_arch['architecture']['id']} "
                f"with score {best_arch['best_score']:.3f}"
            )
            
            low_stability_archs = [
                arch_id for arch_id, data in arch_analysis.items() 
                if data["stability"] < 0.7
            ]
            if low_stability_archs:
                recommendations.append(
                    f"Improve stability for architectures: {', '.join(low_stability_archs)}"
                )
        
        # Meta-learning recommendations
        meta_metrics = meta_eval.get("aggregate_metrics", {})
        if meta_metrics.get("meta_learning_scores_mean", 0) < 0.8:
            recommendations.append(
                "Consider increasing meta-learning training iterations or adjusting hyperparameters"
            )
        
        # Recursive improvement recommendations
        recursive_scores = recursive_bench.get("benchmark_scores", {})
        if recursive_scores.get("recursive_depth_score", 0) < 0.5:
            recommendations.append(
                "Increase recursive depth for better self-improvement capabilities"
            )
        
        if recursive_scores.get("stability_score", 0) < 0.7:
            recommendations.append(
                "Implement better convergence criteria to improve training stability"
            )
        
        return recommendations
    
    def save_benchmark_report(self, report: Dict[str, Any], 
                            filename: str = None) -> str:
        """Save benchmark report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f".taskmaster/evaluation_reports/benchmark_report_{timestamp}.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filename

def main():
    """Main execution function"""
    print("Starting comprehensive evaluation and benchmarking...")
    
    evaluator = ArchitectureEvaluator()
    
    try:
        # Generate comprehensive benchmark
        benchmark_report = evaluator.generate_comprehensive_benchmark()
        
        # Save report
        report_file = evaluator.save_benchmark_report(benchmark_report)
        print(f"Benchmark report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        eval_summary = benchmark_report["evaluation_summary"]
        print(f"Architectures Evaluated: {eval_summary['architectures_evaluated']}")
        print(f"Training Samples: {eval_summary['total_training_samples']}")
        print(f"Pipeline Runs: {eval_summary['total_pipeline_runs']}")
        
        overall_scores = benchmark_report["overall_scores"]
        print(f"\nOverall System Score: {overall_scores['overall_system_score']:.3f}")
        print(f"Architecture Quality: {overall_scores['architecture_quality']:.3f}")
        print(f"Meta-Learning Effectiveness: {overall_scores['meta_learning_effectiveness']:.3f}")
        print(f"Recursive Improvement: {overall_scores['recursive_improvement']:.3f}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(benchmark_report["recommendations"], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*60)
        
        return benchmark_report
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

if __name__ == "__main__":
    main()