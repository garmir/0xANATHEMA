#!/usr/bin/env python3
"""
LABRYS Recursive Self-Improvement System
Main orchestrator for recursive self-improvement using dual-blade methodology
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict

# Add LABRYS paths
sys.path.append(os.path.join(os.path.dirname(__file__), '.labrys'))
from analytical.self_analysis_engine import SelfAnalysisEngine
from synthesis.self_synthesis_engine import SelfSynthesisEngine

class RecursiveLabrysImprovement:
    """
    Main orchestrator for LABRYS recursive self-improvement
    Implements the ancient methodology: Analyze → Synthesize → Validate → Iterate
    """
    
    def __init__(self, labrys_root: str = None):
        self.labrys_root = labrys_root or os.path.join(os.path.dirname(__file__), '.labrys')
        self.analysis_engine = SelfAnalysisEngine(self.labrys_root)
        self.synthesis_engine = SelfSynthesisEngine(self.labrys_root)
        
        # Improvement tracking
        self.improvement_history = []
        self.convergence_threshold = 0.1  # 10% improvement threshold
        self.max_iterations = 10
        
        # Safety and validation
        self.safety_enabled = True
        self.validation_required = True
        
        # Performance tracking
        self.performance_metrics = {
            "total_iterations": 0,
            "successful_improvements": 0,
            "failed_improvements": 0,
            "convergence_achieved": False,
            "improvement_velocity": 0.0
        }
    
    async def execute_recursive_improvement(self, max_iterations: int = None) -> Dict[str, Any]:
        """
        Execute the main recursive improvement loop
        Following the LABRYS methodology: Analyze → Synthesize → Validate → Iterate
        """
        max_iterations = max_iterations or self.max_iterations
        
        print("🗲 LABRYS Recursive Self-Improvement System")
        print("   Ancient wisdom meets modern AI development")
        print("   " + "="*50)
        
        improvement_results = []
        previous_score = 0.0
        
        for iteration in range(max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
            print("   ⚡ Left Blade: Analysis Phase")
            
            # ANALYSIS PHASE (Left Blade)
            analysis_result = await self.analysis_engine.analyze_self_architecture()
            
            print(f"   📊 Analysis Complete - Complexity: {analysis_result.complexity_score:.2f}")
            print(f"   📈 Maintainability: {analysis_result.maintainability_score:.2f}")
            print(f"   🔍 Findings: {len(analysis_result.findings)}")
            
            # SYNTHESIS PHASE (Right Blade)
            print("   ⚡ Right Blade: Synthesis Phase")
            
            # Generate improvement suggestions
            suggestions = await self.analysis_engine.generate_improvement_suggestions(analysis_result)
            
            # Apply improvements using synthesis engine
            modifications = await self.synthesis_engine.synthesize_improvements(
                [asdict(s) for s in suggestions]
            )
            
            successful_mods = [m for m in modifications if m.safety_checks_passed]
            failed_mods = [m for m in modifications if not m.safety_checks_passed]
            
            print(f"   🔧 Modifications Applied: {len(successful_mods)}")
            print(f"   ❌ Failed Modifications: {len(failed_mods)}")
            
            # VALIDATION PHASE
            print("   🔍 Validation Phase")
            
            validation_results = await self._validate_improvements(modifications)
            
            # Calculate improvement score
            current_score = self._calculate_improvement_score(analysis_result, modifications)
            improvement_delta = current_score - previous_score
            
            print(f"   📈 Improvement Score: {current_score:.2f} (Δ {improvement_delta:+.2f})")
            
            # Record iteration results
            iteration_result = {
                "iteration": iteration + 1,
                "timestamp": datetime.now().isoformat(),
                "analysis": asdict(analysis_result),
                "suggestions": [asdict(s) for s in suggestions],
                "modifications": [asdict(m) for m in modifications],
                "validation_results": validation_results,
                "improvement_score": current_score,
                "improvement_delta": improvement_delta,
                "convergence_indicators": self._check_convergence_indicators(
                    analysis_result, modifications, improvement_delta
                )
            }
            
            improvement_results.append(iteration_result)
            self.improvement_history.append(iteration_result)
            
            # Update metrics
            self.performance_metrics["total_iterations"] += 1
            self.performance_metrics["successful_improvements"] += len(successful_mods)
            self.performance_metrics["failed_improvements"] += len(failed_mods)
            
            # Check for convergence
            if self._check_convergence(improvement_delta, modifications):
                print("   ✅ Convergence Achieved!")
                self.performance_metrics["convergence_achieved"] = True
                break
            
            previous_score = current_score
            
            # Brief pause between iterations
            await asyncio.sleep(0.1)
        
        # Calculate final metrics
        self.performance_metrics["improvement_velocity"] = (
            self.performance_metrics["successful_improvements"] / 
            self.performance_metrics["total_iterations"]
        )
        
        return {
            "status": "completed",
            "total_iterations": len(improvement_results),
            "improvement_results": improvement_results,
            "performance_metrics": self.performance_metrics,
            "convergence_achieved": self.performance_metrics["convergence_achieved"],
            "final_recommendations": await self._generate_final_recommendations(improvement_results)
        }
    
    def _calculate_improvement_score(self, analysis_result, modifications) -> float:
        """Calculate an overall improvement score"""
        base_score = 100 - analysis_result.complexity_score + analysis_result.maintainability_score
        
        # Add points for successful modifications
        successful_mods = len([m for m in modifications if m.safety_checks_passed])
        modification_bonus = successful_mods * 5
        
        # Subtract points for failed modifications
        failed_mods = len([m for m in modifications if not m.safety_checks_passed])
        failure_penalty = failed_mods * 2
        
        return max(0, base_score + modification_bonus - failure_penalty)
    
    def _check_convergence_indicators(self, analysis_result, modifications, improvement_delta) -> Dict[str, Any]:
        """Check various convergence indicators"""
        return {
            "improvement_delta_small": abs(improvement_delta) < self.convergence_threshold,
            "no_new_modifications": len(modifications) == 0,
            "low_complexity": analysis_result.complexity_score < 5,
            "high_maintainability": analysis_result.maintainability_score > 80,
            "few_findings": len(analysis_result.findings) < 3
        }
    
    def _check_convergence(self, improvement_delta: float, modifications: List) -> bool:
        """Check if convergence has been achieved"""
        # Convergence conditions
        conditions = [
            abs(improvement_delta) < self.convergence_threshold,  # Small improvement
            len(modifications) == 0,  # No new modifications
            len([m for m in modifications if m.safety_checks_passed]) == 0  # No successful modifications
        ]
        
        # Need at least 2 conditions to be met
        return sum(conditions) >= 2
    
    async def _validate_improvements(self, modifications: List) -> Dict[str, Any]:
        """Validate the applied improvements"""
        validation_results = {
            "total_validations": len(modifications),
            "passed_validations": 0,
            "failed_validations": 0,
            "validation_details": []
        }
        
        for mod in modifications:
            if mod.safety_checks_passed:
                validation_results["passed_validations"] += 1
            else:
                validation_results["failed_validations"] += 1
            
            validation_results["validation_details"].append({
                "component": mod.component_name,
                "passed": mod.safety_checks_passed,
                "results": mod.validation_results
            })
        
        return validation_results
    
    async def _generate_final_recommendations(self, improvement_results: List[Dict[str, Any]]) -> List[str]:
        """Generate final recommendations based on improvement results"""
        recommendations = []
        
        # Analyze overall patterns
        total_successful = sum(
            len([m for m in result["modifications"] if m["safety_checks_passed"]]) 
            for result in improvement_results
        )
        
        total_failed = sum(
            len([m for m in result["modifications"] if not m["safety_checks_passed"]]) 
            for result in improvement_results
        )
        
        if total_successful > total_failed:
            recommendations.append("✅ System shows good improvement capacity")
        else:
            recommendations.append("⚠️ Consider adjusting improvement strategies")
        
        # Check for recurring issues
        common_issues = {}
        for result in improvement_results:
            for finding in result["analysis"]["findings"]:
                common_issues[finding] = common_issues.get(finding, 0) + 1
        
        if common_issues:
            most_common = max(common_issues, key=common_issues.get)
            recommendations.append(f"🔧 Address recurring issue: {most_common}")
        
        # Performance recommendations
        if self.performance_metrics["improvement_velocity"] < 0.5:
            recommendations.append("📈 Consider increasing improvement aggressiveness")
        
        recommendations.append("🗲 LABRYS recursive improvement cycle completed")
        
        return recommendations
    
    async def run_targeted_improvement(self, component_name: str, improvement_type: str) -> Dict[str, Any]:
        """Run improvement on a specific component"""
        print(f"🎯 Targeted Improvement: {component_name} ({improvement_type})")
        
        # Analyze specific component
        analysis_result = await self.analysis_engine.analyze_self_architecture()
        
        # Filter for target component
        target_suggestions = []
        suggestions = await self.analysis_engine.generate_improvement_suggestions(analysis_result)
        
        for suggestion in suggestions:
            if component_name.lower() in suggestion.component_target.lower():
                target_suggestions.append(suggestion)
        
        if not target_suggestions:
            return {
                "status": "no_improvements_found",
                "message": f"No improvement opportunities found for {component_name}"
            }
        
        # Apply targeted improvements
        modifications = await self.synthesis_engine.synthesize_improvements(
            [asdict(s) for s in target_suggestions]
        )
        
        return {
            "status": "completed",
            "component": component_name,
            "improvement_type": improvement_type,
            "modifications": [asdict(m) for m in modifications],
            "successful_modifications": len([m for m in modifications if m.safety_checks_passed])
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "system_name": "LABRYS Recursive Self-Improvement",
            "version": "1.0.0",
            "status": "active",
            "labrys_root": self.labrys_root,
            "engines": {
                "analysis_engine": "SelfAnalysisEngine",
                "synthesis_engine": "SelfSynthesisEngine"
            },
            "configuration": {
                "safety_enabled": self.safety_enabled,
                "validation_required": self.validation_required,
                "max_iterations": self.max_iterations,
                "convergence_threshold": self.convergence_threshold
            },
            "performance_metrics": self.performance_metrics,
            "improvement_history_count": len(self.improvement_history)
        }
    
    def enable_safety_mode(self, enabled: bool = True):
        """Enable or disable safety mode"""
        self.safety_enabled = enabled
        self.synthesis_engine.safety_enabled = enabled
    
    def set_convergence_threshold(self, threshold: float):
        """Set the convergence threshold"""
        self.convergence_threshold = max(0.01, min(1.0, threshold))

async def main():
    """Main entry point for recursive improvement"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LABRYS Recursive Self-Improvement System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--execute", action="store_true", 
                       help="Execute recursive improvement")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Maximum iterations (default: 5)")
    parser.add_argument("--target", help="Target specific component")
    parser.add_argument("--type", help="Improvement type for targeted improvement")
    parser.add_argument("--status", action="store_true",
                       help="Show system status")
    parser.add_argument("--unsafe", action="store_true",
                       help="Disable safety mode")
    
    args = parser.parse_args()
    
    # Initialize system
    improvement_system = RecursiveLabrysImprovement()
    
    if args.unsafe:
        improvement_system.enable_safety_mode(False)
        print("⚠️  Safety mode disabled")
    
    if args.status:
        status = await improvement_system.get_system_status()
        print(json.dumps(status, indent=2))
    
    elif args.target:
        result = await improvement_system.run_targeted_improvement(
            args.target, args.type or "general"
        )
        print(json.dumps(result, indent=2))
    
    elif args.execute:
        result = await improvement_system.execute_recursive_improvement(args.iterations)
        print(f"\n🏁 Recursive Improvement Complete!")
        print(f"   Iterations: {result['total_iterations']}")
        print(f"   Convergence: {'✅ Yes' if result['convergence_achieved'] else '❌ No'}")
        print(f"   Successful Improvements: {result['performance_metrics']['successful_improvements']}")
        
        # Save results
        results_file = "recursive_improvement_results.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"   Results saved to: {results_file}")
    
    else:
        parser.print_help()
        print("\n🗲 LABRYS Recursive Self-Improvement System")
        print("   Ancient wisdom meets modern AI development")
        print("   Use --execute to begin recursive improvement")

if __name__ == "__main__":
    asyncio.run(main())