#!/usr/bin/env python3
"""
Autonomy Validation System for Task-Master
Validates autonomous execution capability and calculates autonomy score
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

class AutonomyValidator:
    def __init__(self, taskmaster_home: str = None):
        self.taskmaster_home = Path(taskmaster_home or os.environ.get('TASKMASTER_HOME', '.taskmaster'))
        self.tasks_file = self.taskmaster_home / 'tasks' / 'tasks.json'
        self.autonomy_threshold = 0.95
        
    def load_tasks(self) -> Dict[str, Any]:
        """Load tasks from tasks.json"""
        try:
            with open(self.tasks_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load tasks: {e}")
    
    def validate_task_atomicity(self, tasks: List[Dict]) -> Tuple[bool, float]:
        """Validate that all tasks are atomic (no further decomposition needed)"""
        atomic_score = 0
        total_tasks = len(tasks)
        
        for task in tasks:
            # Check if task has clear, specific description
            desc = task.get('description', '')
            details = task.get('details', '')
            
            # Atomic indicators
            atomic_indicators = [
                'implement', 'create', 'build', 'configure', 'validate',
                'specific', 'concrete', 'single', 'atomic'
            ]
            
            # Non-atomic indicators  
            non_atomic_indicators = [
                'system', 'framework', 'architecture', 'multiple',
                'overall', 'comprehensive', 'general'
            ]
            
            atomic_count = sum(1 for indicator in atomic_indicators 
                             if indicator.lower() in desc.lower() or indicator.lower() in details.lower())
            non_atomic_count = sum(1 for indicator in non_atomic_indicators 
                                 if indicator.lower() in desc.lower() or indicator.lower() in details.lower())
            
            # Score based on specificity and atomicity
            if atomic_count > non_atomic_count and len(desc) > 50:
                atomic_score += 1
            elif len(desc) > 100 and 'details' in task:
                atomic_score += 0.7
            else:
                atomic_score += 0.3
                
        atomicity_ratio = atomic_score / total_tasks if total_tasks > 0 else 0
        return atomicity_ratio >= 0.85, atomicity_ratio
    
    def validate_dependencies(self, tasks: List[Dict]) -> Tuple[bool, float]:
        """Validate that all task dependencies are properly mapped"""
        task_ids = {str(task['id']) for task in tasks}
        valid_deps = 0
        total_deps = 0
        
        for task in tasks:
            deps = task.get('dependencies', [])
            total_deps += len(deps)
            
            for dep in deps:
                if str(dep) in task_ids:
                    valid_deps += 1
                    
        dependency_ratio = valid_deps / total_deps if total_deps > 0 else 1.0
        return dependency_ratio >= 0.95, dependency_ratio
    
    def validate_resource_allocation(self, tasks: List[Dict]) -> Tuple[bool, float]:
        """Validate that resource requirements are defined"""
        tasks_with_resources = 0
        
        for task in tasks:
            # Check for resource indicators in details
            details = task.get('details', '').lower()
            test_strategy = task.get('testStrategy', '').lower()
            
            resource_indicators = [
                'memory', 'cpu', 'time', 'space', 'optimization',
                'performance', 'complexity', 'benchmark'
            ]
            
            has_resources = any(indicator in details or indicator in test_strategy 
                              for indicator in resource_indicators)
            
            if has_resources or task.get('status') == 'done':
                tasks_with_resources += 1
                
        resource_ratio = tasks_with_resources / len(tasks) if tasks else 0
        return resource_ratio >= 0.80, resource_ratio
    
    def validate_execution_readiness(self, tasks: List[Dict]) -> Tuple[bool, float]:
        """Validate that tasks have clear execution steps"""
        executable_tasks = 0
        
        for task in tasks:
            details = task.get('details', '')
            test_strategy = task.get('testStrategy', '')
            status = task.get('status', '')
            
            # Check for execution indicators
            execution_indicators = [
                'implement', 'create', 'build', 'run', 'execute',
                'configure', 'install', 'setup', 'deploy'
            ]
            
            has_execution_plan = (
                any(indicator in details.lower() for indicator in execution_indicators) or
                len(test_strategy) > 50 or
                status == 'done'
            )
            
            if has_execution_plan:
                executable_tasks += 1
                
        execution_ratio = executable_tasks / len(tasks) if tasks else 0
        return execution_ratio >= 0.90, execution_ratio
    
    def validate_monitoring_capability(self) -> Tuple[bool, float]:
        """Validate that monitoring and logging systems are in place"""
        logs_dir = self.taskmaster_home / 'logs'
        scripts_dir = self.taskmaster_home / 'scripts'
        
        monitoring_score = 0
        
        # Check for logs directory and recent logs
        if logs_dir.exists():
            log_files = list(logs_dir.glob('*.log'))
            if log_files:
                monitoring_score += 0.3
                
        # Check for validation scripts
        if scripts_dir.exists():
            script_files = list(scripts_dir.glob('*.py')) + list(scripts_dir.glob('*.sh'))
            if script_files:
                monitoring_score += 0.4
                
        # Check for dashboard or monitoring files
        dashboard_files = list(self.taskmaster_home.glob('*dashboard*')) + list(self.taskmaster_home.glob('*monitor*'))
        if dashboard_files:
            monitoring_score += 0.3
            
        return monitoring_score >= 0.70, monitoring_score
    
    def calculate_autonomy_score(self, validation_results: Dict) -> float:
        """Calculate overall autonomy score"""
        weights = {
            'atomicity': 0.25,
            'dependencies': 0.20,
            'resources': 0.20,
            'execution': 0.25,
            'monitoring': 0.10
        }
        
        total_score = sum(
            validation_results[key][1] * weight 
            for key, weight in weights.items()
        )
        
        return min(total_score, 1.0)
    
    def generate_autonomy_report(self) -> Dict[str, Any]:
        """Generate comprehensive autonomy validation report"""
        try:
            task_data = self.load_tasks()
            tasks = task_data.get('tasks', [])
            
            validation_results = {
                'atomicity': self.validate_task_atomicity(tasks),
                'dependencies': self.validate_dependencies(tasks),
                'resources': self.validate_resource_allocation(tasks),
                'execution': self.validate_execution_readiness(tasks),
                'monitoring': self.validate_monitoring_capability()
            }
            
            autonomy_score = self.calculate_autonomy_score(validation_results)
            is_autonomous = autonomy_score >= self.autonomy_threshold
            
            # Detailed breakdown
            report = {
                'autonomous_capable': is_autonomous,
                'autonomy_score': round(autonomy_score, 3),
                'threshold': self.autonomy_threshold,
                'validation_details': {
                    key: {
                        'passed': result[0],
                        'score': round(result[1], 3)
                    }
                    for key, result in validation_results.items()
                },
                'task_summary': {
                    'total_tasks': len(tasks),
                    'completed_tasks': len([t for t in tasks if t.get('status') == 'done']),
                    'in_progress_tasks': len([t for t in tasks if t.get('status') == 'in-progress']),
                    'pending_tasks': len([t for t in tasks if t.get('status') == 'pending'])
                },
                'recommendations': self.generate_recommendations(validation_results, autonomy_score)
            }
            
            return report
            
        except Exception as e:
            return {
                'autonomous_capable': False,
                'autonomy_score': 0.0,
                'error': str(e),
                'recommendations': ['Fix task loading errors before proceeding']
            }
    
    def generate_recommendations(self, validation_results: Dict, score: float) -> List[str]:
        """Generate specific recommendations for improving autonomy"""
        recommendations = []
        
        if not validation_results['atomicity'][0]:
            recommendations.append(
                f"Improve task atomicity (current: {validation_results['atomicity'][1]:.3f}, need: 0.85). "
                "Break down complex tasks into smaller, more specific components."
            )
            
        if not validation_results['dependencies'][0]:
            recommendations.append(
                f"Fix dependency mapping (current: {validation_results['dependencies'][1]:.3f}, need: 0.95). "
                "Ensure all task dependencies reference valid task IDs."
            )
            
        if not validation_results['resources'][0]:
            recommendations.append(
                f"Define resource requirements (current: {validation_results['resources'][1]:.3f}, need: 0.80). "
                "Add memory, time, and performance specifications to task details."
            )
            
        if not validation_results['execution'][0]:
            recommendations.append(
                f"Improve execution readiness (current: {validation_results['execution'][1]:.3f}, need: 0.90). "
                "Ensure all tasks have clear implementation steps and test strategies."
            )
            
        if not validation_results['monitoring'][0]:
            recommendations.append(
                f"Enhance monitoring capability (current: {validation_results['monitoring'][1]:.3f}, need: 0.70). "
                "Add logging, dashboard, and progress tracking systems."
            )
            
        if score >= 0.95:
            recommendations.append("✅ System is ready for autonomous execution!")
        elif score >= 0.85:
            recommendations.append("System is close to autonomous capability. Address remaining issues.")
        else:
            recommendations.append("Significant improvements needed for autonomous execution.")
            
        return recommendations

def main():
    """Main execution function"""
    validator = AutonomyValidator()
    report = validator.generate_autonomy_report()
    
    # Print formatted report
    print("🤖 AUTONOMY VALIDATION REPORT")
    print("=" * 50)
    print(f"Autonomous Capable: {'✅ YES' if report['autonomous_capable'] else '❌ NO'}")
    print(f"Autonomy Score: {report['autonomy_score']} (threshold: {report.get('threshold', 0.95)})")
    print()
    
    if 'validation_details' in report:
        print("📊 VALIDATION BREAKDOWN:")
        for category, details in report['validation_details'].items():
            status = "✅" if details['passed'] else "❌"
            print(f"  {status} {category.title()}: {details['score']}")
        print()
    
    if 'task_summary' in report:
        summary = report['task_summary']
        print("📋 TASK SUMMARY:")
        print(f"  Total: {summary['total_tasks']}")
        print(f"  Completed: {summary['completed_tasks']}")
        print(f"  In Progress: {summary['in_progress_tasks']}")
        print(f"  Pending: {summary['pending_tasks']}")
        print()
    
    if 'recommendations' in report:
        print("💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        print()
    
    # Save detailed report
    report_file = Path('.taskmaster/autonomy-validation-report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    sys.exit(0 if report['autonomous_capable'] else 1)

if __name__ == "__main__":
    main()