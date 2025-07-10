#!/usr/bin/env python3
"""
Task-Master Meta-Learning Integration Module
Task 50.2: Integration layer for recursive meta-learning framework

This module provides integration between the recursive meta-learning framework
and the existing Task-Master AI system for enhanced task optimization.
"""

import os
import json
import subprocess
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from recursive_meta_learning_framework import RecursiveMetaLearningFramework, create_task_master_meta_learning_system

logger = logging.getLogger(__name__)

@dataclass
class TaskMasterTask:
    """Represents a Task-Master task for meta-learning integration"""
    id: str
    title: str
    description: str
    status: str
    priority: str
    complexity: Optional[str] = None
    dependencies: List[str] = None
    performance_history: List[float] = None

class TaskMasterMetaLearningAdapter:
    """Adapter to integrate meta-learning with Task-Master system"""
    
    def __init__(self, taskmaster_home: str = ".taskmaster"):
        self.taskmaster_home = taskmaster_home
        self.tasks_file = os.path.join(taskmaster_home, "tasks", "tasks.json")
        self.meta_learning_framework = create_task_master_meta_learning_system()
        self.task_performance_cache = {}
        self.adaptation_results = {}
        
    def load_taskmaster_tasks(self) -> List[TaskMasterTask]:
        """Load tasks from Task-Master tasks.json file"""
        try:
            with open(self.tasks_file, 'r') as f:
                data = json.load(f)
            
            tasks = []
            for task_data in data.get("master", {}).get("tasks", []):
                task = TaskMasterTask(
                    id=str(task_data.get("id", "")),
                    title=task_data.get("title", ""),
                    description=task_data.get("description", ""),
                    status=task_data.get("status", "pending"),
                    priority=task_data.get("priority", "medium"),
                    complexity=task_data.get("complexity"),
                    dependencies=task_data.get("dependencies", []),
                    performance_history=[]
                )
                tasks.append(task)
            
            logger.info(f"Loaded {len(tasks)} tasks from Task-Master")
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to load Task-Master tasks: {e}")
            return []
    
    def analyze_task_context(self, task: TaskMasterTask) -> Dict[str, Any]:
        """Analyze task to create context for meta-learning"""
        context = {
            "task_id": task.id,
            "title": task.title,
            "status": task.status,
            "priority": task.priority,
            "complexity": self._assess_complexity(task),
            "dependency_count": len(task.dependencies) if task.dependencies else 0,
            "title_length": len(task.title),
            "description_length": len(task.description),
            "has_subtasks": "subtasks" in task.description.lower(),
            "task_type": self._classify_task_type(task),
            "estimated_effort": self._estimate_effort(task),
            "risk_factors": self._identify_risk_factors(task)
        }
        
        return context
    
    def _assess_complexity(self, task: TaskMasterTask) -> str:
        """Assess task complexity based on various factors"""
        complexity_score = 0
        
        # Factor in description length
        if len(task.description) > 200:
            complexity_score += 2
        elif len(task.description) > 100:
            complexity_score += 1
        
        # Factor in dependencies
        if task.dependencies:
            complexity_score += len(task.dependencies)
        
        # Factor in keywords
        complexity_keywords = ["implement", "design", "optimize", "refactor", "integrate", "test"]
        for keyword in complexity_keywords:
            if keyword in task.description.lower():
                complexity_score += 1
        
        if complexity_score >= 6:
            return "high"
        elif complexity_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _classify_task_type(self, task: TaskMasterTask) -> str:
        """Classify task type based on content analysis"""
        title_lower = task.title.lower()
        desc_lower = task.description.lower()
        
        if any(word in title_lower or word in desc_lower for word in ["implement", "develop", "build", "create"]):
            return "implementation"
        elif any(word in title_lower or word in desc_lower for word in ["test", "validate", "verify"]):
            return "validation"
        elif any(word in title_lower or word in desc_lower for word in ["optimize", "improve", "enhance"]):
            return "optimization"
        elif any(word in title_lower or word in desc_lower for word in ["design", "plan", "architecture"]):
            return "design"
        elif any(word in title_lower or word in desc_lower for word in ["research", "analyze", "investigate"]):
            return "research"
        else:
            return "general"
    
    def _estimate_effort(self, task: TaskMasterTask) -> str:
        """Estimate effort required for task completion"""
        effort_score = 0
        
        # Base effort from complexity
        complexity_map = {"low": 1, "medium": 2, "high": 3}
        effort_score += complexity_map.get(self._assess_complexity(task), 1)
        
        # Additional effort for high-priority tasks
        if task.priority == "high":
            effort_score += 1
        
        # Effort from dependencies
        if task.dependencies:
            effort_score += min(2, len(task.dependencies))
        
        if effort_score >= 5:
            return "high"
        elif effort_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _identify_risk_factors(self, task: TaskMasterTask) -> List[str]:
        """Identify potential risk factors for task completion"""
        risks = []
        
        # High complexity risks
        if self._assess_complexity(task) == "high":
            risks.append("high_complexity")
        
        # Dependency risks
        if task.dependencies and len(task.dependencies) > 3:
            risks.append("many_dependencies")
        
        # Implementation risks
        task_type = self._classify_task_type(task)
        if task_type == "implementation" and task.priority == "high":
            risks.append("critical_implementation")
        
        # Keyword-based risks
        risk_keywords = ["integration", "migration", "refactor", "legacy", "performance"]
        for keyword in risk_keywords:
            if keyword in task.description.lower():
                risks.append(f"keyword_risk_{keyword}")
        
        return risks
    
    def apply_meta_learning_to_task(self, task: TaskMasterTask) -> Dict[str, Any]:
        """Apply meta-learning framework to optimize task execution"""
        logger.info(f"Applying meta-learning to task {task.id}: {task.title}")
        
        # Create context for meta-learning
        context = self.analyze_task_context(task)
        
        # Execute recursive meta-learning
        meta_results = self.meta_learning_framework.execute_recursive_learning(
            task_id=task.id,
            initial_context=context
        )
        
        # Extract optimization recommendations
        recommendations = self._extract_recommendations(meta_results, task)
        
        # Cache results
        self.adaptation_results[task.id] = {
            "meta_results": meta_results,
            "recommendations": recommendations,
            "context": context,
            "timestamp": meta_results.get("timestamp")
        }
        
        logger.info(f"Meta-learning completed for task {task.id}")
        return recommendations
    
    def _extract_recommendations(self, meta_results: Dict[str, Any], task: TaskMasterTask) -> Dict[str, Any]:
        """Extract actionable recommendations from meta-learning results"""
        performance = meta_results.get("performance", 0.5)
        adaptations = meta_results.get("adaptations", {})
        converged = meta_results.get("converged", False)
        
        recommendations = {
            "priority_adjustment": self._recommend_priority_adjustment(performance, task),
            "decomposition_strategy": self._recommend_decomposition(meta_results, task),
            "resource_allocation": self._recommend_resources(adaptations, task),
            "execution_order": self._recommend_execution_order(meta_results, task),
            "risk_mitigation": self._recommend_risk_mitigation(meta_results, task),
            "performance_prediction": performance,
            "confidence_level": "high" if converged else "medium"
        }
        
        return recommendations
    
    def _recommend_priority_adjustment(self, performance: float, task: TaskMasterTask) -> Dict[str, Any]:
        """Recommend priority adjustments based on meta-learning"""
        current_priority = task.priority
        
        if performance > 0.8:
            suggested_priority = "high" if current_priority != "high" else current_priority
            reason = "High predicted performance suggests increased priority"
        elif performance < 0.4:
            suggested_priority = "low" if current_priority == "high" else current_priority
            reason = "Low predicted performance suggests reduced priority"
        else:
            suggested_priority = current_priority
            reason = "Current priority appears optimal"
        
        return {
            "current": current_priority,
            "suggested": suggested_priority,
            "reason": reason,
            "confidence": min(1.0, abs(performance - 0.5) * 2)
        }
    
    def _recommend_decomposition(self, meta_results: Dict[str, Any], task: TaskMasterTask) -> Dict[str, Any]:
        """Recommend task decomposition strategy"""
        depth = meta_results.get("depth", 0)
        child_results = meta_results.get("child_results", [])
        
        if len(child_results) > 0:
            avg_child_performance = sum(cr.get("performance", 0) for cr in child_results) / len(child_results)
            if avg_child_performance > 0.7:
                strategy = "fine_grained_decomposition"
                reason = "Child tasks show high performance - decompose into smaller units"
            else:
                strategy = "coarse_grained_decomposition"
                reason = "Child tasks show mixed performance - keep larger units"
        else:
            complexity = self._assess_complexity(task)
            if complexity == "high":
                strategy = "sequential_decomposition"
                reason = "High complexity task should be broken down sequentially"
            else:
                strategy = "parallel_decomposition"
                reason = "Medium/low complexity allows parallel decomposition"
        
        return {
            "strategy": strategy,
            "reason": reason,
            "suggested_subtasks": min(5, max(2, depth + 2)),
            "decomposition_depth": min(3, depth + 1)
        }
    
    def _recommend_resources(self, adaptations: Dict[str, Any], task: TaskMasterTask) -> Dict[str, Any]:
        """Recommend resource allocation based on adaptations"""
        resource_recommendations = {
            "cpu_priority": "normal",
            "memory_allocation": "standard",
            "time_allocation": "standard",
            "parallel_execution": False
        }
        
        # Analyze gradient-based adaptations
        if "gradient_based" in adaptations:
            gradient_params = adaptations["gradient_based"]
            learning_multiplier = gradient_params.get("learning_rate_multiplier", 1.0)
            
            if learning_multiplier > 1.2:
                resource_recommendations["cpu_priority"] = "high"
                resource_recommendations["time_allocation"] = "extended"
            elif learning_multiplier < 0.8:
                resource_recommendations["cpu_priority"] = "low"
                resource_recommendations["memory_allocation"] = "minimal"
        
        # Factor in task complexity
        complexity = self._assess_complexity(task)
        if complexity == "high":
            resource_recommendations["memory_allocation"] = "high"
            resource_recommendations["parallel_execution"] = True
        
        return resource_recommendations
    
    def _recommend_execution_order(self, meta_results: Dict[str, Any], task: TaskMasterTask) -> Dict[str, Any]:
        """Recommend optimal execution order"""
        performance = meta_results.get("performance", 0.5)
        converged = meta_results.get("converged", False)
        
        if performance > 0.8 and converged:
            order = "immediate"
            reason = "High performance and convergence suggest ready for execution"
        elif performance > 0.6:
            order = "scheduled"
            reason = "Good performance suggests scheduled execution after preparation"
        elif task.dependencies:
            order = "dependent"
            reason = "Dependencies must be resolved first"
        else:
            order = "deferred"
            reason = "Performance concerns suggest deferring until optimization"
        
        return {
            "order": order,
            "reason": reason,
            "estimated_delay": 0 if order == "immediate" else (1 if order == "scheduled" else 3)
        }
    
    def _recommend_risk_mitigation(self, meta_results: Dict[str, Any], task: TaskMasterTask) -> Dict[str, Any]:
        """Recommend risk mitigation strategies"""
        risk_factors = self._identify_risk_factors(task)
        performance = meta_results.get("performance", 0.5)
        
        mitigation_strategies = []
        
        if performance < 0.5:
            mitigation_strategies.append("increase_validation_checkpoints")
            mitigation_strategies.append("implement_rollback_mechanism")
        
        if "high_complexity" in risk_factors:
            mitigation_strategies.append("expert_review_required")
            mitigation_strategies.append("incremental_implementation")
        
        if "many_dependencies" in risk_factors:
            mitigation_strategies.append("dependency_validation")
            mitigation_strategies.append("parallel_dependency_resolution")
        
        return {
            "identified_risks": risk_factors,
            "mitigation_strategies": mitigation_strategies,
            "risk_level": "high" if len(risk_factors) > 2 else ("medium" if risk_factors else "low")
        }
    
    def optimize_all_pending_tasks(self) -> Dict[str, Any]:
        """Apply meta-learning optimization to all pending tasks"""
        tasks = self.load_taskmaster_tasks()
        pending_tasks = [task for task in tasks if task.status == "pending"]
        
        optimization_results = {}
        
        for task in pending_tasks:
            try:
                recommendations = self.apply_meta_learning_to_task(task)
                optimization_results[task.id] = {
                    "task_title": task.title,
                    "recommendations": recommendations,
                    "status": "optimized"
                }
                logger.info(f"Optimized task {task.id}")
            except Exception as e:
                logger.error(f"Failed to optimize task {task.id}: {e}")
                optimization_results[task.id] = {
                    "task_title": task.title,
                    "error": str(e),
                    "status": "failed"
                }
        
        # Generate summary report
        summary = {
            "total_tasks": len(pending_tasks),
            "optimized_tasks": len([r for r in optimization_results.values() if r["status"] == "optimized"]),
            "failed_tasks": len([r for r in optimization_results.values() if r["status"] == "failed"]),
            "optimization_results": optimization_results,
            "framework_status": self.meta_learning_framework.get_framework_status()
        }
        
        return summary
    
    def generate_optimization_report(self, output_file: str = None) -> str:
        """Generate comprehensive optimization report"""
        if not output_file:
            output_file = os.path.join(self.taskmaster_home, "reports", "meta_learning_optimization_report.json")
        
        # Run optimization
        results = self.optimize_all_pending_tasks()
        
        # Create detailed report
        report = {
            "report_timestamp": self.meta_learning_framework.global_state.timestamp,
            "optimization_summary": results,
            "framework_configuration": {
                "max_recursion_depth": self.meta_learning_framework.max_recursion_depth,
                "convergence_threshold": self.meta_learning_framework.convergence_threshold,
                "registered_meta_learners": list(self.meta_learning_framework.meta_learners.keys())
            },
            "performance_metrics": {
                "average_task_performance": results.get("framework_status", {}).get("average_performance", 0),
                "total_adaptations": results.get("framework_status", {}).get("adaptation_cycles", 0),
                "optimization_coverage": results["optimized_tasks"] / max(1, results["total_tasks"])
            },
            "recommendations": self._generate_system_recommendations(results)
        }
        
        # Save report
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Meta-learning optimization report saved to {output_file}")
        return output_file
    
    def _generate_system_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []
        
        optimization_rate = results["optimized_tasks"] / max(1, results["total_tasks"])
        
        if optimization_rate < 0.7:
            recommendations.append("Consider increasing meta-learning framework parameters")
            recommendations.append("Review task complexity assessment algorithms")
        
        framework_status = results.get("framework_status", {})
        avg_performance = framework_status.get("average_performance", 0)
        
        if avg_performance < 0.6:
            recommendations.append("Implement additional meta-learning algorithms")
            recommendations.append("Increase recursion depth for complex tasks")
        
        if framework_status.get("adaptation_cycles", 0) > 100:
            recommendations.append("Implement convergence optimization")
            recommendations.append("Consider meta-learning parameter tuning")
        
        return recommendations

# CLI interface for meta-learning integration
def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Task-Master Meta-Learning Integration")
    parser.add_argument("--optimize-all", action="store_true", help="Optimize all pending tasks")
    parser.add_argument("--report", action="store_true", help="Generate optimization report")
    parser.add_argument("--task-id", help="Optimize specific task by ID")
    parser.add_argument("--output", help="Output file for reports")
    
    args = parser.parse_args()
    
    adapter = TaskMasterMetaLearningAdapter()
    
    if args.optimize_all:
        results = adapter.optimize_all_pending_tasks()
        print(f"Optimized {results['optimized_tasks']}/{results['total_tasks']} tasks")
    
    if args.report:
        report_file = adapter.generate_optimization_report(args.output)
        print(f"Report generated: {report_file}")
    
    if args.task_id:
        tasks = adapter.load_taskmaster_tasks()
        target_task = next((t for t in tasks if t.id == args.task_id), None)
        if target_task:
            recommendations = adapter.apply_meta_learning_to_task(target_task)
            print(f"Optimization recommendations for task {args.task_id}:")
            print(json.dumps(recommendations, indent=2))
        else:
            print(f"Task {args.task_id} not found")

if __name__ == "__main__":
    main()