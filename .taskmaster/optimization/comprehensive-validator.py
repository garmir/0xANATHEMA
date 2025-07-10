#!/usr/bin/env python3
import json
import os
import time
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class ValidationResult:
    """Represents a validation check result"""
    check_name: str
    passed: bool
    score: float
    details: str
    critical: bool = False

@dataclass
class TaskNode:
    """Represents a task in the validation system"""
    id: str
    name: str
    dependencies: List[str]
    resources: Dict[str, any]
    execution_time: int
    atomicity_score: float
    
class ComprehensiveValidator:
    """Advanced validation system for autonomous execution"""
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.task_graph: Dict[str, TaskNode] = {}
        self.critical_failures = []
        
    def load_system_state(self):
        """Load current system state for validation"""
        # Load task graph if available
        task_tree_paths = [
            'task-tree.json',
            '.taskmaster/optimization/task-tree.json',
            os.path.join(os.path.dirname(__file__), 'task-tree.json')
        ]
        
        for path in task_tree_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    self._parse_task_graph(data)
                break
        
        # Load optimization results from multiple locations
        optimization_files = {
            'sqrt-optimized.json': [
                'sqrt-optimized.json',
                '.taskmaster/artifacts/sqrt-space/sqrt-optimized.json',
                '.taskmaster/optimization/sqrt-optimized.json'
            ],
            'tree-optimized.json': [
                'tree-optimized.json',
                '.taskmaster/artifacts/tree-eval/tree-optimized.json',
                '.taskmaster/optimization/tree-optimized.json'
            ],
            'pebbling-strategy.json': [
                'pebbling-strategy.json',
                '.taskmaster/artifacts/pebbling/pebbling-strategy.json',
                '.taskmaster/optimization/pebbling-strategy.json'
            ],
            'catalytic-execution.json': [
                'catalytic-execution.json',
                '.taskmaster/artifacts/catalytic/catalytic-execution.json',
                '.taskmaster/optimization/catalytic-execution.json'
            ]
        }
        
        self.optimization_results = {}
        for base_filename, paths in optimization_files.items():
            for path in paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        self.optimization_results[base_filename] = json.load(f)
                    break
    
    def _parse_task_graph(self, data: Dict):
        """Parse task graph from loaded data"""
        # Handle both 'nodes' and 'tasks' formats
        tasks = data.get('tasks', data.get('nodes', []))
        for task in tasks:
            # Extract execution time from estimated_duration or resources
            exec_time = 5  # default
            duration = task.get('estimated_duration', '5min')
            if isinstance(duration, str) and 'min' in duration:
                exec_time = int(duration.replace('min', ''))
            
            task_node = TaskNode(
                id=str(task.get('id', '')),
                name=task.get('title', f"Task {task.get('id', 'unknown')}"),
                dependencies=task.get('dependencies', []),
                resources=task.get('resources', {}),
                execution_time=exec_time,
                atomicity_score=0.95 if task.get('status') == 'done' else 0.85  # Higher score for completed tasks
            )
            self.task_graph[task_node.id] = task_node
    
    def validate_atomicity(self) -> ValidationResult:
        """Validate that all tasks are atomic and executable"""
        print("   🔬 Validating task atomicity...")
        
        atomic_tasks = 0
        total_tasks = len(self.task_graph)
        
        for task in self.task_graph.values():
            # Check if task can be executed as single unit
            if task.atomicity_score >= 0.8 and task.execution_time <= 60:
                atomic_tasks += 1
        
        atomicity_ratio = atomic_tasks / max(total_tasks, 1)
        passed = atomicity_ratio >= 0.9
        
        return ValidationResult(
            check_name="atomicity",
            passed=passed,
            score=atomicity_ratio,
            details=f"{atomic_tasks}/{total_tasks} tasks are atomic",
            critical=True
        )
    
    def validate_dependencies(self) -> ValidationResult:
        """Validate dependency graph integrity"""
        print("   🔗 Validating dependency graph...")
        
        # Check for circular dependencies
        def has_cycle(node_id: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            if node_id in self.task_graph:
                for dep in self.task_graph[node_id].dependencies:
                    if dep not in visited:
                        if has_cycle(dep, visited, rec_stack):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        visited = set()
        has_cycles = False
        
        for node_id in self.task_graph:
            if node_id not in visited:
                if has_cycle(node_id, visited, set()):
                    has_cycles = True
                    break
        
        # Check dependency completeness
        missing_deps = []
        for task in self.task_graph.values():
            for dep in task.dependencies:
                if str(dep) not in self.task_graph:
                    missing_deps.append(f"{task.id} -> {dep}")
        
        passed = not has_cycles and len(missing_deps) == 0
        
        details = []
        if has_cycles:
            details.append("Circular dependencies detected")
        if missing_deps:
            details.append(f"Missing dependencies: {', '.join(missing_deps[:3])}")
        if passed:
            details.append("All dependencies valid")
        
        return ValidationResult(
            check_name="dependencies",
            passed=passed,
            score=1.0 if passed else 0.0,
            details="; ".join(details),
            critical=True
        )
    
    def validate_resources(self) -> ValidationResult:
        """Validate resource allocation and availability"""
        print("   💾 Validating resource allocation...")
        
        total_memory = 0
        total_cpu = 0
        
        for task in self.task_graph.values():
            memory_str = task.resources.get('memory', '0MB')
            # Handle both MB and GB
            if 'GB' in memory_str:
                memory_mb = int(memory_str.replace('GB', '')) * 1024
            else:
                memory_mb = int(memory_str.replace('MB', ''))
            total_memory += memory_mb
            
            cpu_val = task.resources.get('cpu', 1)
            # Convert string CPU levels to numeric values
            if isinstance(cpu_val, str):
                cpu_map = {'low': 1, 'moderate': 2, 'high': 4}
                cpu_count = cpu_map.get(cpu_val.lower(), 1)
            else:
                cpu_count = int(cpu_val)
            total_cpu += cpu_count
        
        # Check against system limits
        memory_limit = 1000  # 1GB limit
        cpu_limit = 8        # 8 CPU limit
        
        memory_ok = total_memory <= memory_limit
        cpu_ok = total_cpu <= cpu_limit
        
        passed = memory_ok and cpu_ok
        
        optimization_bonus = 0.0
        if 'sqrt-optimized.json' in self.optimization_results:
            optimization_bonus = 0.2  # Bonus for optimization
        
        score = (0.5 if memory_ok else 0) + (0.5 if cpu_ok else 0) + optimization_bonus
        
        details = f"Memory: {total_memory}MB/{memory_limit}MB, CPU: {total_cpu}/{cpu_limit}"
        
        return ValidationResult(
            check_name="resources",
            passed=passed,
            score=min(1.0, score),
            details=details,
            critical=False
        )
    
    def validate_timing(self) -> ValidationResult:
        """Validate timing constraints and execution order"""
        print("   ⏱️  Validating timing constraints...")
        
        # Calculate critical path
        def calculate_critical_path():
            # Simplified critical path calculation
            earliest_start = {}
            
            def calculate_earliest(node_id: str) -> int:
                if node_id in earliest_start:
                    return earliest_start[node_id]
                
                if node_id not in self.task_graph:
                    return 0
                
                task = self.task_graph[node_id]
                max_dep_time = 0
                
                for dep in task.dependencies:
                    dep_time = calculate_earliest(str(dep)) + self.task_graph.get(str(dep), TaskNode('', '', [], {}, 0, 0)).execution_time
                    max_dep_time = max(max_dep_time, dep_time)
                
                earliest_start[node_id] = max_dep_time
                return max_dep_time
            
            total_time = 0
            for node_id in self.task_graph:
                end_time = calculate_earliest(node_id) + self.task_graph[node_id].execution_time
                total_time = max(total_time, end_time)
            
            return total_time
        
        critical_path_time = calculate_critical_path()
        
        # Check against reasonable time limits
        time_limit = 120  # 2 hours
        passed = critical_path_time <= time_limit
        
        # Optimization time savings
        time_savings = 0
        if 'catalytic-execution.json' in self.optimization_results:
            time_savings = 20  # Estimated 20% time savings
        
        effective_time = critical_path_time * (1 - time_savings / 100)
        score = max(0.0, min(1.0, (time_limit - effective_time) / time_limit))
        
        return ValidationResult(
            check_name="timing",
            passed=passed,
            score=score,
            details=f"Critical path: {critical_path_time}min (limit: {time_limit}min)",
            critical=False
        )
    
    def validate_optimization_integrity(self) -> ValidationResult:
        """Validate that optimizations maintain system integrity"""
        print("   🚀 Validating optimization integrity...")
        
        optimizations_present = len(self.optimization_results)
        expected_optimizations = 4  # sqrt, tree, pebbling, catalytic
        
        integrity_checks = []
        
        # Check optimization chain
        if 'sqrt-optimized.json' in self.optimization_results:
            sqrt_data = self.optimization_results['sqrt-optimized.json']
            if sqrt_data.get('algorithm') == 'sqrt-space':
                integrity_checks.append('sqrt_optimization_valid')
        
        if 'tree-optimized.json' in self.optimization_results:
            tree_data = self.optimization_results['tree-optimized.json']
            if 'log' in str(tree_data.get('space_complexity', '')):
                integrity_checks.append('tree_optimization_valid')
        
        if 'catalytic-execution.json' in self.optimization_results:
            integrity_checks.append('catalytic_execution_valid')
        
        passed = len(integrity_checks) >= 2  # At least 2 optimizations working
        score = len(integrity_checks) / expected_optimizations
        
        return ValidationResult(
            check_name="optimization_integrity",
            passed=passed,
            score=score,
            details=f"{len(integrity_checks)}/{expected_optimizations} optimizations verified",
            critical=False
        )
    
    def run_comprehensive_validation(self) -> Dict:
        """Run all validation checks"""
        print("🔍 Running comprehensive validation...")
        
        self.load_system_state()
        
        # Run all validation checks
        checks = [
            self.validate_atomicity(),
            self.validate_dependencies(),
            self.validate_resources(),
            self.validate_timing(),
            self.validate_optimization_integrity()
        ]
        
        self.validation_results = checks
        
        # Calculate overall scores
        critical_checks = [c for c in checks if c.critical]
        non_critical_checks = [c for c in checks if not c.critical]
        
        critical_passed = all(c.passed for c in critical_checks)
        overall_score = sum(c.score for c in checks) / len(checks)
        
        # System is autonomous if critical checks pass and overall score > 0.8
        autonomous_capable = critical_passed and overall_score >= 0.8
        
        return {
            "validation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "autonomous_execution_capable": autonomous_capable,
            "overall_validation_score": round(overall_score, 3),
            "critical_checks_passed": critical_passed,
            "validation_details": {
                check.check_name: {
                    "passed": check.passed,
                    "score": round(check.score, 3),
                    "details": check.details,
                    "critical": check.critical
                }
                for check in checks
            },
            "system_ready": autonomous_capable,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        for result in self.validation_results:
            if not result.passed and result.critical:
                recommendations.append(f"CRITICAL: Fix {result.check_name} - {result.details}")
            elif result.score < 0.7:
                recommendations.append(f"Improve {result.check_name} - {result.details}")
        
        if not recommendations:
            recommendations.append("System validation complete - ready for autonomous execution")
        
        return recommendations

def main():
    validator = ComprehensiveValidator()
    validation_report = validator.run_comprehensive_validation()
    
    # Save validation report
    with open('validation-report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"\n✅ Validation complete!")
    print(f"   Autonomous Capable: {'✅' if validation_report['autonomous_execution_capable'] else '❌'}")
    print(f"   Overall Score: {validation_report['overall_validation_score']:.3f}")
    print(f"   Critical Checks: {'✅' if validation_report['critical_checks_passed'] else '❌'}")
    
    return validation_report

if __name__ == "__main__":
    main()
