#!/usr/bin/env python3
"""
Performance Comparison Framework for Self-Improving Architecture
Compares against baseline and state-of-the-art models
"""

import json
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple

class PerformanceComparator:
    """Framework for comparing recursive meta-learning architecture against baselines"""
    
    def __init__(self):
        self.baseline_benchmarks = {
            "standard_neural_networks": {
                "avg_accuracy": 0.75,
                "convergence_time": 100,
                "adaptation_speed": 0.3,
                "generalization": 0.65
            },
            "meta_learning_baselines": {
                "maml": {
                    "avg_accuracy": 0.82,
                    "convergence_time": 50,
                    "adaptation_speed": 0.7,
                    "generalization": 0.78
                },
                "reptile": {
                    "avg_accuracy": 0.79,
                    "convergence_time": 60,
                    "adaptation_speed": 0.65,
                    "generalization": 0.74
                }
            },
            "nas_baselines": {
                "darts": {
                    "avg_accuracy": 0.85,
                    "convergence_time": 200,
                    "adaptation_speed": 0.4,
                    "generalization": 0.81
                },
                "enas": {
                    "avg_accuracy": 0.83,
                    "convergence_time": 180,
                    "adaptation_speed": 0.45,
                    "generalization": 0.79
                }
            }
        }
    
    def load_our_results(self, report_file: str = None) -> Dict[str, Any]:
        """Load our system's benchmark results"""
        if report_file is None:
            # Find the most recent benchmark report
            import glob
            reports = glob.glob(".taskmaster/evaluation_reports/benchmark_report_*.json")
            if not reports:
                raise FileNotFoundError("No benchmark reports found")
            report_file = max(reports)
        
        with open(report_file, 'r') as f:
            return json.load(f)
    
    def extract_our_metrics(self, benchmark_report: Dict[str, Any]) -> Dict[str, float]:
        """Extract comparable metrics from our benchmark report"""
        our_metrics = {}
        
        # Overall accuracy/performance
        overall_scores = benchmark_report.get("overall_scores", {})
        our_metrics["avg_accuracy"] = overall_scores.get("overall_system_score", 0)
        
        # Meta-learning metrics
        meta_eval = benchmark_report.get("meta_learning_evaluation", {})
        aggregate_metrics = meta_eval.get("aggregate_metrics", {})
        
        our_metrics["adaptation_speed"] = aggregate_metrics.get("adaptation_speed_mean", 0)
        our_metrics["generalization"] = aggregate_metrics.get("generalization_scores_mean", 0)
        our_metrics["meta_learning_effectiveness"] = aggregate_metrics.get("meta_learning_scores_mean", 0)
        
        # Recursive improvement metrics
        recursive_bench = benchmark_report.get("recursive_improvement_benchmark", {})
        benchmark_scores = recursive_bench.get("benchmark_scores", {})
        
        our_metrics["improvement_rate"] = benchmark_scores.get("average_improvement_rate", 0)
        our_metrics["recursive_depth"] = benchmark_scores.get("recursive_depth_score", 0)
        our_metrics["stability"] = benchmark_scores.get("stability_score", 0)
        
        # Architecture quality
        our_metrics["architecture_quality"] = overall_scores.get("architecture_quality", 0)
        
        return our_metrics
    
    def compare_against_baselines(self, our_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare our system against baseline methods"""
        comparisons = {}
        
        # Compare against each baseline category
        for category, baselines in self.baseline_benchmarks.items():
            comparisons[category] = {}
            
            if isinstance(baselines, dict) and "avg_accuracy" in baselines:
                # Single baseline
                comparisons[category]["single"] = self._compare_single_baseline(
                    our_metrics, baselines, category
                )
            else:
                # Multiple baselines in category
                for baseline_name, baseline_metrics in baselines.items():
                    comparisons[category][baseline_name] = self._compare_single_baseline(
                        our_metrics, baseline_metrics, f"{category}_{baseline_name}"
                    )
        
        return comparisons
    
    def _compare_single_baseline(self, our_metrics: Dict[str, float], 
                               baseline_metrics: Dict[str, float], 
                               baseline_name: str) -> Dict[str, Any]:
        """Compare against a single baseline"""
        comparison = {
            "baseline_name": baseline_name,
            "metric_comparisons": {},
            "overall_performance_ratio": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0
        }
        
        performance_ratios = []
        
        for metric, baseline_value in baseline_metrics.items():
            if metric in our_metrics:
                our_value = our_metrics[metric]
                ratio = our_value / max(baseline_value, 0.001)  # Avoid division by zero
                
                comparison["metric_comparisons"][metric] = {
                    "our_value": our_value,
                    "baseline_value": baseline_value,
                    "ratio": ratio,
                    "improvement": (our_value - baseline_value) / max(baseline_value, 0.001),
                    "winner": "ours" if our_value > baseline_value else "baseline" if our_value < baseline_value else "tie"
                }
                
                performance_ratios.append(ratio)
                
                # Count wins/losses/ties
                if our_value > baseline_value * 1.01:  # 1% threshold for tie
                    comparison["wins"] += 1
                elif our_value < baseline_value * 0.99:
                    comparison["losses"] += 1
                else:
                    comparison["ties"] += 1
        
        # Calculate overall performance ratio
        if performance_ratios:
            comparison["overall_performance_ratio"] = statistics.mean(performance_ratios)
        
        return comparison
    
    def generate_competitive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive competitive analysis"""
        print("Loading our benchmark results...")
        our_report = self.load_our_results()
        our_metrics = self.extract_our_metrics(our_report)
        
        print("Comparing against baselines...")
        baseline_comparisons = self.compare_against_baselines(our_metrics)
        
        # Calculate competitive scores
        competitive_scores = self._calculate_competitive_scores(baseline_comparisons)
        
        # Generate insights and recommendations
        insights = self._generate_competitive_insights(baseline_comparisons, competitive_scores)
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "our_metrics": our_metrics,
            "baseline_comparisons": baseline_comparisons,
            "competitive_scores": competitive_scores,
            "insights": insights,
            "summary": self._generate_summary(competitive_scores, insights)
        }
        
        return analysis
    
    def _calculate_competitive_scores(self, comparisons: Dict[str, Any]) -> Dict[str, float]:
        """Calculate competitive performance scores"""
        scores = {
            "vs_standard_nn": 0,
            "vs_meta_learning": 0,
            "vs_nas": 0,
            "overall_competitive_score": 0
        }
        
        # Score against standard neural networks
        if "standard_neural_networks" in comparisons:
            std_comparison = comparisons["standard_neural_networks"]["single"]
            scores["vs_standard_nn"] = std_comparison["overall_performance_ratio"]
        
        # Score against meta-learning baselines
        meta_ratios = []
        if "meta_learning_baselines" in comparisons:
            for baseline_name, comparison in comparisons["meta_learning_baselines"].items():
                meta_ratios.append(comparison["overall_performance_ratio"])
        if meta_ratios:
            scores["vs_meta_learning"] = statistics.mean(meta_ratios)
        
        # Score against NAS baselines
        nas_ratios = []
        if "nas_baselines" in comparisons:
            for baseline_name, comparison in comparisons["nas_baselines"].items():
                nas_ratios.append(comparison["overall_performance_ratio"])
        if nas_ratios:
            scores["vs_nas"] = statistics.mean(nas_ratios)
        
        # Overall competitive score (weighted average)
        scores["overall_competitive_score"] = (
            0.2 * scores["vs_standard_nn"] +
            0.4 * scores["vs_meta_learning"] +
            0.4 * scores["vs_nas"]
        )
        
        return scores
    
    def _generate_competitive_insights(self, comparisons: Dict[str, Any], 
                                     scores: Dict[str, float]) -> List[str]:
        """Generate insights from competitive analysis"""
        insights = []
        
        # Overall performance insight
        overall_score = scores["overall_competitive_score"]
        if overall_score > 1.1:
            insights.append(f"Excellent: System outperforms baselines by {(overall_score-1)*100:.1f}% on average")
        elif overall_score > 1.0:
            insights.append(f"Good: System slightly outperforms baselines by {(overall_score-1)*100:.1f}%")
        elif overall_score > 0.9:
            insights.append(f"Competitive: System performs comparably to baselines ({overall_score:.2f}x)")
        else:
            insights.append(f"Needs improvement: System underperforms baselines ({overall_score:.2f}x)")
        
        # Specific strength insights
        strengths = []
        weaknesses = []
        
        for category, category_comparisons in comparisons.items():
            if isinstance(category_comparisons, dict) and "single" in category_comparisons:
                comparison = category_comparisons["single"]
            else:
                # Average across multiple baselines
                ratios = [comp["overall_performance_ratio"] for comp in category_comparisons.values()]
                avg_ratio = statistics.mean(ratios) if ratios else 0
                if avg_ratio > 1.05:
                    strengths.append(f"Strong performance vs {category}")
                elif avg_ratio < 0.95:
                    weaknesses.append(f"Weak performance vs {category}")
        
        if strengths:
            insights.append(f"Key strengths: {', '.join(strengths)}")
        if weaknesses:
            insights.append(f"Areas for improvement: {', '.join(weaknesses)}")
        
        return insights
    
    def _generate_summary(self, scores: Dict[str, float], insights: List[str]) -> Dict[str, Any]:
        """Generate executive summary of competitive analysis"""
        return {
            "competitive_ranking": self._determine_ranking(scores["overall_competitive_score"]),
            "key_findings": insights[:3],  # Top 3 insights
            "recommendation": self._generate_recommendation(scores),
            "next_steps": self._suggest_next_steps(scores)
        }
    
    def _determine_ranking(self, score: float) -> str:
        """Determine competitive ranking based on score"""
        if score > 1.2:
            return "State-of-the-art"
        elif score > 1.1:
            return "Excellent"
        elif score > 1.0:
            return "Above average"
        elif score > 0.9:
            return "Competitive"
        elif score > 0.8:
            return "Below average"
        else:
            return "Needs significant improvement"
    
    def _generate_recommendation(self, scores: Dict[str, float]) -> str:
        """Generate recommendation based on scores"""
        if scores["overall_competitive_score"] > 1.1:
            return "System demonstrates strong competitive advantage. Focus on optimization and deployment."
        elif scores["vs_meta_learning"] < 0.9:
            return "Improve meta-learning components to better compete with MAML/Reptile baselines."
        elif scores["vs_nas"] < 0.9:
            return "Enhance architecture search capabilities to compete with DARTS/ENAS."
        else:
            return "System is competitive. Focus on specific weak areas identified in analysis."
    
    def _suggest_next_steps(self, scores: Dict[str, float]) -> List[str]:
        """Suggest concrete next steps"""
        steps = []
        
        if scores["vs_meta_learning"] < 1.0:
            steps.append("Increase meta-learning training iterations and hyperparameter optimization")
        
        if scores["vs_nas"] < 1.0:
            steps.append("Implement more sophisticated architecture search strategies")
        
        if scores["overall_competitive_score"] > 1.0:
            steps.append("Prepare for publication and real-world deployment testing")
        
        steps.append("Conduct ablation studies to identify key performance drivers")
        steps.append("Test on additional benchmark datasets for broader validation")
        
        return steps
    
    def save_analysis(self, analysis: Dict[str, Any], filename: str = None) -> str:
        """Save competitive analysis to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f".taskmaster/evaluation_reports/competitive_analysis_{timestamp}.json"
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return filename

def main():
    """Main execution for competitive analysis"""
    print("Starting competitive performance analysis...")
    
    comparator = PerformanceComparator()
    
    try:
        analysis = comparator.generate_competitive_analysis()
        
        # Save analysis
        analysis_file = comparator.save_analysis(analysis)
        print(f"Competitive analysis saved to: {analysis_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("COMPETITIVE ANALYSIS SUMMARY")
        print("="*60)
        
        summary = analysis["summary"]
        scores = analysis["competitive_scores"]
        
        print(f"Competitive Ranking: {summary['competitive_ranking']}")
        print(f"Overall Score: {scores['overall_competitive_score']:.2f}x baseline performance")
        print(f"vs Standard NN: {scores['vs_standard_nn']:.2f}x")
        print(f"vs Meta-Learning: {scores['vs_meta_learning']:.2f}x")
        print(f"vs NAS Methods: {scores['vs_nas']:.2f}x")
        
        print("\nKey Findings:")
        for finding in summary["key_findings"]:
            print(f"• {finding}")
        
        print(f"\nRecommendation: {summary['recommendation']}")
        
        print("\nNext Steps:")
        for step in summary["next_steps"]:
            print(f"• {step}")
        
        print("\n" + "="*60)
        
        return analysis
        
    except Exception as e:
        print(f"Error during competitive analysis: {e}")
        return None

if __name__ == "__main__":
    main()