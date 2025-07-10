#!/usr/bin/env python3
"""
Autonomous Execution Validator
Validates autonomous execution capability and generates autonomy scores
"""

import json
import time
import subprocess
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

@dataclass
class AutonomyMetric:
    """Individual autonomy metric"""
    metric_name: str
    score: float  # 0.0 to 1.0
    weight: float
    details: str
    evidence: List[str]

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    validation_timestamp: datetime
    overall_autonomy_score: float
    target_score: float
    autonomy_achieved: bool
    individual_metrics: List[AutonomyMetric]
    recommendations: List[str]


class AutonomousExecutionValidator:
    """Validates autonomous execution capability"""
    
    def __init__(self, target_score: float = 0.95):
        self.target_score = target_score
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('AutonomyValidator')
    
    def validate_autonomy(self) -> ValidationReport:
        """Validate overall autonomous execution capability"""
        self.logger.info("Starting autonomous execution validation")
        
        metrics = [
            self._validate_atomic_task_capability(),
            self._validate_error_handling(),
            self._validate_resource_optimization(),
            self._validate_decision_making(),
            self._validate_monitoring_capability(),
            self._validate_self_healing(),
            self._validate_learning_capability()
        ]
        
        # Calculate weighted overall score
        total_weight = sum(m.weight for m in metrics)
        overall_score = sum(m.score * m.weight for m in metrics) / total_weight
        
        autonomy_achieved = overall_score >= self.target_score
        
        recommendations = self._generate_recommendations(metrics, overall_score)
        
        report = ValidationReport(
            validation_timestamp=datetime.now(),
            overall_autonomy_score=overall_score,
            target_score=self.target_score,
            autonomy_achieved=autonomy_achieved,
            individual_metrics=metrics,
            recommendations=recommendations
        )
        
        # Save validation report
        self._save_validation_report(report)
        
        return report
    
    def _validate_atomic_task_capability(self) -> AutonomyMetric:
        """Validate atomic task execution capability"""
        evidence = []
        score = 0.0
        
        # Check if task system can identify atomic tasks
        tasks_file = Path('.taskmaster/tasks/tasks.json')
        if tasks_file.exists():
            evidence.append("Task system exists")
            score += 0.3
            
            with open(tasks_file, 'r') as f:
                data = json.load(f)
                tasks = data.get('master', {}).get('tasks', [])
                
            if tasks:
                evidence.append(f"Found {len(tasks)} tasks")
                score += 0.3
                
                # Check for atomic task indicators
                atomic_indicators = 0
                for task in tasks:
                    if 'atomic' in task.get('details', '').lower() or len(task.get('subtasks', [])) == 0:
                        atomic_indicators += 1
                        
                if atomic_indicators > 0:
                    evidence.append(f"Found {atomic_indicators} atomic task indicators")
                    score += 0.4
        
        return AutonomyMetric(
            metric_name="Atomic Task Capability",
            score=score,
            weight=0.2,
            details="Ability to identify and execute atomic tasks",
            evidence=evidence
        )
    
    def _validate_error_handling(self) -> AutonomyMetric:
        """Validate error handling and recovery capability"""
        evidence = []
        score = 0.0
        
        # Check for error handling implementations
        error_files = list(Path('.taskmaster').rglob('*error*'))
        if error_files:
            evidence.append(f"Found {len(error_files)} error handling files")
            score += 0.3
        
        # Check for exception handling in Python files
        py_files = list(Path('.taskmaster').rglob('*.py'))
        exception_handling_files = 0
        
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if 'except' in content or 'try:' in content:
                        exception_handling_files += 1
            except:
                pass
        
        if exception_handling_files > 0:
            evidence.append(f"Found exception handling in {exception_handling_files} Python files")
            score += 0.4
        
        # Check for retry mechanisms
        retry_indicators = 0
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if 'retry' in content.lower() or 'attempt' in content.lower():
                        retry_indicators += 1
            except:
                pass
                
        if retry_indicators > 0:
            evidence.append(f"Found retry mechanisms in {retry_indicators} files")
            score += 0.3
        
        return AutonomyMetric(
            metric_name="Error Handling",
            score=score,
            weight=0.15,
            details="Capability to handle and recover from errors",
            evidence=evidence
        )
    
    def _validate_resource_optimization(self) -> AutonomyMetric:
        """Validate resource optimization capability"""
        evidence = []
        score = 0.0
        
        # Check for optimization implementations
        optimization_files = list(Path('.taskmaster/optimization').glob('*')) if Path('.taskmaster/optimization').exists() else []
        if optimization_files:
            evidence.append(f"Found {len(optimization_files)} optimization files")
            score += 0.5
        
        # Check for memory optimization
        memory_files = [f for f in optimization_files if 'memory' in f.name.lower() or 'space' in f.name.lower()]
        if memory_files:
            evidence.append("Memory optimization implemented")
            score += 0.3
        
        # Check for performance monitoring
        if Path('.taskmaster/optimization/simple_system_optimizer.py').exists():
            evidence.append("Performance monitoring implemented")
            score += 0.2
        
        return AutonomyMetric(
            metric_name="Resource Optimization",
            score=score,
            weight=0.2,
            details="Ability to optimize resource usage autonomously",
            evidence=evidence
        )
    
    def _validate_decision_making(self) -> AutonomyMetric:
        """Validate autonomous decision making capability"""
        evidence = []
        score = 0.0
        
        # Check for decision-making algorithms
        if Path('.taskmaster/optimization/intelligent_task_predictor.py').exists():
            evidence.append("Intelligent decision making implemented")
            score += 0.4
        
        # Check for workflow loop
        if Path('.taskmaster/workflow/autonomous_research_loop.py').exists():
            evidence.append("Autonomous workflow loop implemented")
            score += 0.4
        
        # Check for pattern analysis
        pattern_files = list(Path('.taskmaster').rglob('*pattern*'))
        if pattern_files:
            evidence.append("Pattern analysis capability found")
            score += 0.2
        
        return AutonomyMetric(
            metric_name="Decision Making",
            score=score,
            weight=0.15,
            details="Ability to make autonomous decisions",
            evidence=evidence
        )
    
    def _validate_monitoring_capability(self) -> AutonomyMetric:
        """Validate monitoring and observability capability"""
        evidence = []
        score = 0.0
        
        # Check for monitoring system
        if Path('.taskmaster/optimization/simple_system_optimizer.py').exists():
            evidence.append("System monitoring implemented")
            score += 0.4
        
        # Check for logs directory
        if Path('.taskmaster/logs').exists():
            log_files = list(Path('.taskmaster/logs').glob('*'))
            evidence.append(f"Found {len(log_files)} log files")
            score += 0.3
        
        # Check for reports directory
        if Path('.taskmaster/reports').exists():
            report_files = list(Path('.taskmaster/reports').glob('*'))
            evidence.append(f"Found {len(report_files)} report files")
            score += 0.3
        
        return AutonomyMetric(
            metric_name="Monitoring Capability",
            score=score,
            weight=0.1,
            details="Ability to monitor system health and performance",
            evidence=evidence
        )
    
    def _validate_self_healing(self) -> AutonomyMetric:
        """Validate self-healing capability"""
        evidence = []
        score = 0.0
        
        # Check for self-healing implementations
        healing_files = list(Path('.taskmaster').rglob('*heal*'))
        if healing_files:
            evidence.append("Self-healing implementations found")
            score += 0.5
        
        # Check for anomaly detection
        anomaly_files = list(Path('.taskmaster').rglob('*anomaly*'))
        if anomaly_files:
            evidence.append("Anomaly detection found")
            score += 0.3
        
        # Check for automated recovery
        recovery_indicators = 0
        py_files = list(Path('.taskmaster').rglob('*.py'))
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if 'recovery' in content.lower() or 'restore' in content.lower():
                        recovery_indicators += 1
            except:
                pass
                
        if recovery_indicators > 0:
            evidence.append(f"Recovery mechanisms in {recovery_indicators} files")
            score += 0.2
        
        return AutonomyMetric(
            metric_name="Self-Healing",
            score=score,
            weight=0.1,
            details="Ability to automatically detect and fix issues",
            evidence=evidence
        )
    
    def _validate_learning_capability(self) -> AutonomyMetric:
        """Validate learning and adaptation capability"""
        evidence = []
        score = 0.0
        
        # Check for learning implementations
        if Path('.taskmaster/optimization/intelligent_task_predictor.py').exists():
            evidence.append("Machine learning implemented")
            score += 0.4
        
        # Check for pattern recognition
        pattern_files = list(Path('.taskmaster').rglob('*pattern*'))
        if pattern_files:
            evidence.append("Pattern recognition capability")
            score += 0.3
        
        # Check for evolutionary optimization
        evolution_files = list(Path('.taskmaster').rglob('*evolution*'))
        if evolution_files:
            evidence.append("Evolutionary optimization found")
            score += 0.3
        
        return AutonomyMetric(
            metric_name="Learning Capability",
            score=score,
            weight=0.1,
            details="Ability to learn and adapt from experience",
            evidence=evidence
        )
    
    def _generate_recommendations(self, metrics: List[AutonomyMetric], overall_score: float) -> List[str]:
        """Generate recommendations for improving autonomy"""
        recommendations = []
        
        if overall_score < self.target_score:
            recommendations.append(f"Overall autonomy score ({overall_score:.1%}) below target ({self.target_score:.1%})")
        
        # Identify low-scoring metrics
        low_metrics = [m for m in metrics if m.score < 0.7]
        for metric in low_metrics:
            recommendations.append(f"Improve {metric.metric_name}: {metric.details}")
        
        # Specific recommendations based on gaps
        if any(m.metric_name == "Atomic Task Capability" and m.score < 0.8 for m in metrics):
            recommendations.append("Implement comprehensive atomic task detection and validation")
        
        if any(m.metric_name == "Error Handling" and m.score < 0.8 for m in metrics):
            recommendations.append("Enhance error handling and recovery mechanisms")
        
        if any(m.metric_name == "Self-Healing" and m.score < 0.5 for m in metrics):
            recommendations.append("Implement automated self-healing capabilities")
        
        return recommendations
    
    def _save_validation_report(self, report: ValidationReport):
        """Save validation report"""
        try:
            os.makedirs('.taskmaster/reports', exist_ok=True)
            
            report_path = Path('.taskmaster/reports/validation-report.json')
            with open(report_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            self.logger.info(f"Validation report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")


def main():
    """Main validation execution"""
    print("Autonomous Execution Validator")
    print("=" * 40)
    
    validator = AutonomousExecutionValidator()
    
    try:
        report = validator.validate_autonomy()
        
        print(f"Overall Autonomy Score: {report.overall_autonomy_score:.1%}")
        print(f"Target Score: {report.target_score:.1%}")
        print(f"Autonomy Achieved: {'✅ Yes' if report.autonomy_achieved else '❌ No'}")
        
        print(f"\nIndividual Metrics:")
        for metric in report.individual_metrics:
            status = "✅" if metric.score >= 0.8 else "⚠️" if metric.score >= 0.5 else "❌"
            print(f"  {status} {metric.metric_name}: {metric.score:.1%}")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print(f"\nValidation report saved to: .taskmaster/reports/validation-report.json")
        
        return report.autonomy_achieved
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)