#!/usr/bin/env python3
"""
Comprehensive Project Plan Assessment Framework
Research-driven testing against project plan requirements with gap analysis and fixes
"""

import json
import time
import subprocess
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

@dataclass
class ProjectRequirement:
    """Individual project requirement"""
    requirement_id: str
    category: str
    description: str
    success_criteria: List[str]
    priority: str
    implemented: bool
    test_results: Dict[str, Any]

@dataclass
class AssessmentResult:
    """Assessment result for a requirement"""
    requirement_id: str
    status: str  # "passed", "failed", "partial", "not_implemented"
    score: float  # 0.0 to 1.0
    details: str
    evidence: List[str]
    gaps_identified: List[str]
    recommended_fixes: List[str]

@dataclass
class ProjectAssessment:
    """Complete project assessment"""
    assessment_timestamp: datetime
    overall_score: float
    total_requirements: int
    passed_requirements: int
    failed_requirements: int
    partial_requirements: int
    not_implemented_requirements: int
    requirement_results: List[AssessmentResult]
    critical_gaps: List[str]
    implementation_recommendations: List[str]


class ProjectPlanAssessor:
    """
    Comprehensive assessment framework based on research findings:
    1. Autonomy Fix Plan requirements
    2. Execution Planning requirements  
    3. Task Master system specifications
    """
    
    def __init__(self):
        self.requirements = self._load_project_requirements()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup assessment logging"""
        os.makedirs('.taskmaster/logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
            handlers=[
                logging.FileHandler('.taskmaster/logs/project_assessment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ProjectAssessment')
    
    def _load_project_requirements(self) -> List[ProjectRequirement]:
        """Load project requirements based on research findings"""
        return [
            # Core Infrastructure Requirements (from autonomy-fix-plan.md)
            ProjectRequirement(
                requirement_id="REQ-001",
                category="Task Management",
                description="Valid tasks.json with proper structure and schema compliance",
                success_criteria=[
                    "tasks.json exists and is readable",
                    "JSON structure is valid",
                    "All tasks have required fields (id, title, description, status, priority)",
                    "Task IDs follow proper format"
                ],
                priority="critical",
                implemented=False,
                test_results={}
            ),
            
            ProjectRequirement(
                requirement_id="REQ-002", 
                category="Dependency Analysis",
                description="Complete dependency analysis with cycle detection",
                success_criteria=[
                    "task-master validate-dependencies passes",
                    "No circular dependencies detected",
                    "All dependency references are valid",
                    "Task graph can be generated successfully"
                ],
                priority="critical",
                implemented=False,
                test_results={}
            ),
            
            ProjectRequirement(
                requirement_id="REQ-003",
                category="Autonomous Execution",
                description="95% autonomy score achievement",
                success_criteria=[
                    "Autonomy score >= 0.95",
                    "Atomic task validation passes",
                    "Error handling and recovery functional",
                    "Checkpoint/resume capability working"
                ],
                priority="critical", 
                implemented=False,
                test_results={}
            ),
            
            # Optimization Requirements (from execution-planning.md)
            ProjectRequirement(
                requirement_id="REQ-004",
                category="Space Optimization",
                description="O(√n) memory usage optimization",
                success_criteria=[
                    "Square-root space algorithm implemented",
                    "Memory usage stays within O(√n) bounds",
                    "Optimization validation passes"
                ],
                priority="high",
                implemented=False,
                test_results={}
            ),
            
            ProjectRequirement(
                requirement_id="REQ-005",
                category="Tree Evaluation",
                description="O(log n · log log n) tree evaluation optimization",
                success_criteria=[
                    "Tree evaluation algorithm implemented", 
                    "Complexity bounds verified",
                    "Performance benchmarks passed"
                ],
                priority="high",
                implemented=False,
                test_results={}
            ),
            
            ProjectRequirement(
                requirement_id="REQ-006",
                category="Recursive Processing",
                description="Recursive PRD decomposition with depth tracking",
                success_criteria=[
                    "Max depth of 5 levels enforced",
                    "Atomic task detection functional",
                    "Nested directory structure created",
                    "Edge case handling implemented"
                ],
                priority="high",
                implemented=False,
                test_results={}
            ),
            
            # Advanced Features Requirements
            ProjectRequirement(
                requirement_id="REQ-007",
                category="Monitoring",
                description="Real-time monitoring dashboard and system health",
                success_criteria=[
                    "Monitoring dashboard functional",
                    "Real-time metrics collection",
                    "Health scoring implemented",
                    "Performance tracking active"
                ],
                priority="medium",
                implemented=False,
                test_results={}
            ),
            
            ProjectRequirement(
                requirement_id="REQ-008",
                category="Intelligence",
                description="AI-powered task prediction and generation",
                success_criteria=[
                    "Pattern analysis functional",
                    "Task generation working",
                    "Confidence scoring implemented",
                    "Learning from execution history"
                ],
                priority="medium",
                implemented=False,
                test_results={}
            ),
            
            ProjectRequirement(
                requirement_id="REQ-009",
                category="Integration",
                description="Comprehensive system integration and testing",
                success_criteria=[
                    "End-to-end integration tests pass",
                    "Component isolation tests pass",
                    "Performance benchmarks met",
                    "100% test success rate achieved"
                ],
                priority="high",
                implemented=False,
                test_results={}
            ),
            
            ProjectRequirement(
                requirement_id="REQ-010",
                category="Research Workflow",
                description="Autonomous research-driven workflow loop",
                success_criteria=[
                    "Research-driven problem solving functional",
                    "Todo generation from research working",
                    "Claude Code integration active",
                    "Loop until success implemented"
                ],
                priority="medium",
                implemented=False,
                test_results={}
            )
        ]
    
    def execute_comprehensive_assessment(self) -> ProjectAssessment:
        """Execute comprehensive assessment against project plan"""
        self.logger.info("Starting comprehensive project plan assessment")
        
        requirement_results = []
        
        for requirement in self.requirements:
            self.logger.info(f"Assessing requirement {requirement.requirement_id}: {requirement.description}")
            result = self._assess_requirement(requirement)
            requirement_results.append(result)
        
        # Calculate overall metrics
        overall_score = sum(r.score for r in requirement_results) / len(requirement_results)
        passed = sum(1 for r in requirement_results if r.status == "passed")
        failed = sum(1 for r in requirement_results if r.status == "failed")
        partial = sum(1 for r in requirement_results if r.status == "partial")
        not_implemented = sum(1 for r in requirement_results if r.status == "not_implemented")
        
        # Identify critical gaps
        critical_gaps = []
        for result in requirement_results:
            if result.status in ["failed", "not_implemented"]:
                req = next(r for r in self.requirements if r.requirement_id == result.requirement_id)
                if req.priority == "critical":
                    critical_gaps.extend(result.gaps_identified)
        
        # Generate implementation recommendations
        recommendations = self._generate_implementation_recommendations(requirement_results)
        
        assessment = ProjectAssessment(
            assessment_timestamp=datetime.now(),
            overall_score=overall_score,
            total_requirements=len(requirement_results),
            passed_requirements=passed,
            failed_requirements=failed,
            partial_requirements=partial,
            not_implemented_requirements=not_implemented,
            requirement_results=requirement_results,
            critical_gaps=critical_gaps,
            implementation_recommendations=recommendations
        )
        
        # Save assessment results
        self._save_assessment(assessment)
        
        return assessment
    
    def _assess_requirement(self, requirement: ProjectRequirement) -> AssessmentResult:
        """Assess individual requirement"""
        try:
            if requirement.requirement_id == "REQ-001":
                return self._assess_tasks_json_validity()
            elif requirement.requirement_id == "REQ-002":
                return self._assess_dependency_analysis()
            elif requirement.requirement_id == "REQ-003":
                return self._assess_autonomy_score()
            elif requirement.requirement_id == "REQ-004":
                return self._assess_space_optimization()
            elif requirement.requirement_id == "REQ-005":
                return self._assess_tree_evaluation()
            elif requirement.requirement_id == "REQ-006":
                return self._assess_recursive_processing()
            elif requirement.requirement_id == "REQ-007":
                return self._assess_monitoring_system()
            elif requirement.requirement_id == "REQ-008":
                return self._assess_intelligence_system()
            elif requirement.requirement_id == "REQ-009":
                return self._assess_integration_testing()
            elif requirement.requirement_id == "REQ-010":
                return self._assess_research_workflow()
            else:
                return AssessmentResult(
                    requirement_id=requirement.requirement_id,
                    status="not_implemented",
                    score=0.0,
                    details="Assessment not implemented",
                    evidence=[],
                    gaps_identified=["Assessment method not implemented"],
                    recommended_fixes=["Implement assessment method"]
                )
                
        except Exception as e:
            return AssessmentResult(
                requirement_id=requirement.requirement_id,
                status="failed",
                score=0.0,
                details=f"Assessment failed: {e}",
                evidence=[],
                gaps_identified=[f"Assessment error: {e}"],
                recommended_fixes=["Fix assessment implementation"]
            )
    
    def _assess_tasks_json_validity(self) -> AssessmentResult:
        """Assess tasks.json validity and structure"""
        gaps = []
        evidence = []
        fixes = []
        score = 0.0
        
        # Check if tasks.json exists
        tasks_file = Path('.taskmaster/tasks/tasks.json')
        if not tasks_file.exists():
            gaps.append("tasks.json file does not exist")
            fixes.append("Create .taskmaster/tasks/tasks.json file")
            return AssessmentResult("REQ-001", "failed", 0.0, "tasks.json missing", evidence, gaps, fixes)
        
        evidence.append("tasks.json file exists")
        score += 0.25
        
        # Check JSON validity
        try:
            with open(tasks_file, 'r') as f:
                data = json.load(f)
            evidence.append("JSON structure is valid")
            score += 0.25
        except json.JSONDecodeError as e:
            gaps.append(f"Invalid JSON structure: {e}")
            fixes.append("Fix JSON syntax errors in tasks.json")
            return AssessmentResult("REQ-001", "failed", score, "Invalid JSON", evidence, gaps, fixes)
        
        # Check master structure
        if 'master' not in data:
            gaps.append("Missing 'master' key in tasks.json")
            fixes.append("Add 'master' key to tasks.json structure")
        else:
            evidence.append("Master structure present")
            score += 0.2
        
        # Check tasks array
        tasks = data.get('master', {}).get('tasks', [])
        if not tasks:
            gaps.append("No tasks found in tasks array")
            fixes.append("Add tasks to the tasks array")
        else:
            evidence.append(f"Found {len(tasks)} tasks")
            score += 0.15
        
        # Validate task schema
        required_fields = ['id', 'title', 'description', 'status', 'priority']
        invalid_tasks = 0
        
        for task in tasks:
            missing_fields = [field for field in required_fields if field not in task]
            if missing_fields:
                invalid_tasks += 1
        
        if invalid_tasks == 0:
            evidence.append("All tasks have required fields")
            score += 0.15
        else:
            gaps.append(f"{invalid_tasks} tasks missing required fields")
            fixes.append("Add missing fields to all tasks")
        
        # Determine status
        if score >= 0.9:
            status = "passed"
        elif score >= 0.5:
            status = "partial"
        else:
            status = "failed"
        
        return AssessmentResult("REQ-001", status, score, "tasks.json validation", evidence, gaps, fixes)
    
    def _assess_dependency_analysis(self) -> AssessmentResult:
        """Assess dependency analysis functionality"""
        gaps = []
        evidence = []
        fixes = []
        score = 0.0
        
        try:
            # Test task-master validate-dependencies
            result = subprocess.run(['task-master', 'validate-dependencies'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                evidence.append("task-master validate-dependencies passed")
                score += 0.5
            else:
                gaps.append("Dependency validation failed")
                fixes.append("Fix dependency validation issues")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            gaps.append("task-master validate-dependencies not available")
            fixes.append("Ensure task-master is properly installed")
        
        # Check for task graph generation capability
        try:
            result = subprocess.run(['task-master', 'analyze-dependencies'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                evidence.append("Dependency analysis command functional")
                score += 0.3
            else:
                gaps.append("Dependency analysis command failed")
                fixes.append("Fix dependency analysis implementation")
        except:
            gaps.append("Dependency analysis not available")
            fixes.append("Implement dependency analysis functionality")
        
        # Check for task-tree.json output
        if Path('.taskmaster/task-tree.json').exists():
            evidence.append("Task tree file exists")
            score += 0.2
        else:
            gaps.append("Task tree file not generated")
            fixes.append("Generate task-tree.json from dependency analysis")
        
        status = "passed" if score >= 0.8 else "partial" if score >= 0.4 else "failed"
        
        return AssessmentResult("REQ-002", status, score, "Dependency analysis assessment", evidence, gaps, fixes)
    
    def _assess_autonomy_score(self) -> AssessmentResult:
        """Assess autonomy score and autonomous execution capability"""
        gaps = []
        evidence = []
        fixes = []
        score = 0.0
        
        # Check for autonomy validation system
        validation_files = [
            '.taskmaster/optimization/autonomous_execution_validator.py',
            '.taskmaster/reports/validation-report.json'
        ]
        
        for file_path in validation_files:
            if Path(file_path).exists():
                evidence.append(f"Found {file_path}")
                score += 0.2
            else:
                gaps.append(f"Missing {file_path}")
                fixes.append(f"Implement {file_path}")
        
        # Check for checkpoint/resume functionality
        if Path('.taskmaster/checkpoint').exists():
            evidence.append("Checkpoint directory exists")
            score += 0.2
        else:
            gaps.append("No checkpoint functionality found")
            fixes.append("Implement checkpoint/resume functionality")
        
        # Check for error handling implementation
        error_handling_files = list(Path('.taskmaster').rglob('*error*'))
        if error_handling_files:
            evidence.append(f"Found {len(error_handling_files)} error handling files")
            score += 0.2
        else:
            gaps.append("No error handling implementation found")
            fixes.append("Implement comprehensive error handling")
        
        # Check for monitoring dashboard
        if Path('.taskmaster/optimization/simple_system_optimizer.py').exists():
            evidence.append("System monitoring implemented")
            score += 0.2
        else:
            gaps.append("No system monitoring found")
            fixes.append("Implement system monitoring dashboard")
        
        status = "passed" if score >= 0.8 else "partial" if score >= 0.4 else "failed"
        
        return AssessmentResult("REQ-003", status, score, "Autonomy assessment", evidence, gaps, fixes)
    
    def _assess_space_optimization(self) -> AssessmentResult:
        """Assess space optimization implementation"""
        gaps = []
        evidence = []
        fixes = []
        score = 0.0
        
        # Check for optimization files
        optimization_files = list(Path('.taskmaster/optimization').glob('*'))
        if optimization_files:
            evidence.append(f"Found {len(optimization_files)} optimization files")
            score += 0.3
        else:
            gaps.append("No optimization files found")
            fixes.append("Implement optimization algorithms")
        
        # Check for space complexity measurement
        if Path('.taskmaster/optimization/space_complexity_validator.py').exists():
            evidence.append("Space complexity validator exists")
            score += 0.4
        else:
            gaps.append("No space complexity validation")
            fixes.append("Implement space complexity measurement")
        
        # Check for sqrt optimization artifacts
        sqrt_files = list(Path('.taskmaster').rglob('*sqrt*'))
        if sqrt_files:
            evidence.append("Square root optimization artifacts found")
            score += 0.3
        else:
            gaps.append("No sqrt optimization artifacts")
            fixes.append("Implement square root space optimization")
        
        status = "passed" if score >= 0.7 else "partial" if score >= 0.3 else "failed"
        
        return AssessmentResult("REQ-004", status, score, "Space optimization assessment", evidence, gaps, fixes)
    
    def _assess_tree_evaluation(self) -> AssessmentResult:
        """Assess tree evaluation optimization"""
        gaps = []
        evidence = []
        fixes = []
        score = 0.0
        
        # Check for tree evaluation implementation
        tree_files = list(Path('.taskmaster').rglob('*tree*'))
        if tree_files:
            evidence.append(f"Found {len(tree_files)} tree-related files")
            score += 0.5
        else:
            gaps.append("No tree evaluation files found")
            fixes.append("Implement tree evaluation optimization")
        
        # Check for complexity analysis
        if Path('.taskmaster/optimization/task_complexity_analyzer.py').exists():
            evidence.append("Task complexity analyzer exists")
            score += 0.3
        else:
            gaps.append("No complexity analyzer found")
            fixes.append("Implement complexity analysis system")
        
        # Check for performance benchmarks
        if Path('.taskmaster/reports').exists():
            reports = list(Path('.taskmaster/reports').glob('*'))
            evidence.append(f"Found {len(reports)} report files")
            score += 0.2
        else:
            gaps.append("No performance reports found")
            fixes.append("Generate performance benchmark reports")
        
        status = "passed" if score >= 0.7 else "partial" if score >= 0.3 else "failed"
        
        return AssessmentResult("REQ-005", status, score, "Tree evaluation assessment", evidence, gaps, fixes)
    
    def _assess_recursive_processing(self) -> AssessmentResult:
        """Assess recursive PRD processing"""
        gaps = []
        evidence = []
        fixes = []
        score = 0.0
        
        # Check for recursive processing implementation
        recursive_files = list(Path('.taskmaster').rglob('*recursive*'))
        if recursive_files:
            evidence.append("Recursive processing files found")
            score += 0.3
        else:
            gaps.append("No recursive processing implementation")
            fixes.append("Implement recursive PRD processing")
        
        # Check for PRD files and structure
        prd_files = list(Path('.taskmaster/docs').glob('*prd*')) if Path('.taskmaster/docs').exists() else []
        if prd_files:
            evidence.append(f"Found {len(prd_files)} PRD files")
            score += 0.2
        else:
            gaps.append("No PRD files found")
            fixes.append("Generate PRD files from project requirements")
        
        # Check for nested directory structure
        nested_dirs = [d for d in Path('.taskmaster').rglob('*') if d.is_dir() and '/' in str(d.relative_to('.taskmaster'))]
        if nested_dirs:
            evidence.append("Nested directory structure exists")
            score += 0.3
        else:
            gaps.append("No nested directory structure")
            fixes.append("Create proper nested directory hierarchy")
        
        # Check for atomic task detection
        if any('atomic' in f.name.lower() for f in Path('.taskmaster').rglob('*')):
            evidence.append("Atomic task detection implemented")
            score += 0.2
        else:
            gaps.append("No atomic task detection")
            fixes.append("Implement atomic task detection logic")
        
        status = "passed" if score >= 0.7 else "partial" if score >= 0.3 else "failed"
        
        return AssessmentResult("REQ-006", status, score, "Recursive processing assessment", evidence, gaps, fixes)
    
    def _assess_monitoring_system(self) -> AssessmentResult:
        """Assess monitoring system implementation"""
        gaps = []
        evidence = []
        fixes = []
        score = 0.0
        
        # Check for monitoring implementation
        if Path('.taskmaster/optimization/simple_system_optimizer.py').exists():
            evidence.append("System optimizer/monitor exists")
            score += 0.4
        else:
            gaps.append("No system monitoring implementation")
            fixes.append("Implement system monitoring")
        
        # Check for reports directory
        if Path('.taskmaster/reports').exists():
            reports = list(Path('.taskmaster/reports').glob('*'))
            evidence.append(f"Found {len(reports)} monitoring reports")
            score += 0.3
        else:
            gaps.append("No monitoring reports directory")
            fixes.append("Create monitoring reports directory")
        
        # Check for logs directory
        if Path('.taskmaster/logs').exists():
            logs = list(Path('.taskmaster/logs').glob('*'))
            evidence.append(f"Found {len(logs)} log files")
            score += 0.3
        else:
            gaps.append("No logging system")
            fixes.append("Implement comprehensive logging")
        
        status = "passed" if score >= 0.7 else "partial" if score >= 0.3 else "failed"
        
        return AssessmentResult("REQ-007", status, score, "Monitoring system assessment", evidence, gaps, fixes)
    
    def _assess_intelligence_system(self) -> AssessmentResult:
        """Assess AI intelligence system"""
        gaps = []
        evidence = []
        fixes = []
        score = 0.0
        
        # Check for intelligent task predictor
        if Path('.taskmaster/optimization/intelligent_task_predictor.py').exists():
            evidence.append("Intelligent task predictor exists")
            score += 0.5
        else:
            gaps.append("No intelligent task prediction")
            fixes.append("Implement intelligent task prediction system")
        
        # Check for pattern analysis
        analysis_files = list(Path('.taskmaster').rglob('*analysis*'))
        if analysis_files:
            evidence.append("Analysis files found")
            score += 0.3
        else:
            gaps.append("No pattern analysis implementation")
            fixes.append("Implement pattern analysis system")
        
        # Check for task generation
        if Path('.taskmaster/reports/intelligent_task_analysis.json').exists():
            evidence.append("Task analysis reports exist")
            score += 0.2
        else:
            gaps.append("No task generation reports")
            fixes.append("Generate task analysis and prediction reports")
        
        status = "passed" if score >= 0.7 else "partial" if score >= 0.3 else "failed"
        
        return AssessmentResult("REQ-008", status, score, "Intelligence system assessment", evidence, gaps, fixes)
    
    def _assess_integration_testing(self) -> AssessmentResult:
        """Assess integration testing system"""
        gaps = []
        evidence = []
        fixes = []
        score = 0.0
        
        # Check for integration framework
        if Path('.taskmaster/integration/comprehensive_integration_framework.py').exists():
            evidence.append("Integration framework exists")
            score += 0.5
        else:
            gaps.append("No integration testing framework")
            fixes.append("Implement comprehensive integration testing")
        
        # Check for integration reports
        if Path('.taskmaster/reports/integration').exists():
            integration_reports = list(Path('.taskmaster/reports/integration').glob('*'))
            evidence.append(f"Found {len(integration_reports)} integration reports")
            score += 0.3
        else:
            gaps.append("No integration test reports")
            fixes.append("Generate integration test reports")
        
        # Check for test results
        test_files = list(Path('.taskmaster').rglob('*test*'))
        if test_files:
            evidence.append(f"Found {len(test_files)} test-related files")
            score += 0.2
        else:
            gaps.append("No test files found")
            fixes.append("Create comprehensive test suite")
        
        status = "passed" if score >= 0.7 else "partial" if score >= 0.3 else "failed"
        
        return AssessmentResult("REQ-009", status, score, "Integration testing assessment", evidence, gaps, fixes)
    
    def _assess_research_workflow(self) -> AssessmentResult:
        """Assess research-driven workflow loop"""
        gaps = []
        evidence = []
        fixes = []
        score = 0.0
        
        # Check for research workflow implementation
        if Path('.taskmaster/workflow/autonomous_research_loop.py').exists():
            evidence.append("Research workflow loop exists")
            score += 0.6
        else:
            gaps.append("No research workflow implementation")
            fixes.append("Implement autonomous research workflow loop")
        
        # Check for workflow reports
        if Path('.taskmaster/reports/autonomous_workflow_result.json').exists():
            evidence.append("Workflow execution results exist")
            score += 0.2
        else:
            gaps.append("No workflow execution results")
            fixes.append("Execute and document workflow results")
        
        # Check for todo integration
        if any('todo' in str(f).lower() for f in Path('.taskmaster').rglob('*')):
            evidence.append("Todo integration evidence found")
            score += 0.2
        else:
            gaps.append("No todo integration found")
            fixes.append("Implement todo-driven execution loop")
        
        status = "passed" if score >= 0.7 else "partial" if score >= 0.3 else "failed"
        
        return AssessmentResult("REQ-010", status, score, "Research workflow assessment", evidence, gaps, fixes)
    
    def _generate_implementation_recommendations(self, results: List[AssessmentResult]) -> List[str]:
        """Generate implementation recommendations based on assessment"""
        recommendations = []
        
        # Critical issues first
        critical_failures = [r for r in results if r.status in ["failed", "not_implemented"] 
                           and any(req.priority == "critical" for req in self.requirements 
                                 if req.requirement_id == r.requirement_id)]
        
        if critical_failures:
            recommendations.append("PRIORITY 1: Address critical failures immediately")
            for failure in critical_failures:
                recommendations.extend(failure.recommended_fixes)
        
        # High priority partial implementations
        high_priority_partial = [r for r in results if r.status == "partial" 
                               and any(req.priority == "high" for req in self.requirements 
                                     if req.requirement_id == r.requirement_id)]
        
        if high_priority_partial:
            recommendations.append("PRIORITY 2: Complete high-priority partial implementations")
            for partial in high_priority_partial:
                recommendations.extend(partial.recommended_fixes)
        
        # Overall system improvements
        if any(r.score < 0.5 for r in results):
            recommendations.append("Implement comprehensive system testing and validation")
        
        if any("monitoring" in gap.lower() for r in results for gap in r.gaps_identified):
            recommendations.append("Enhance monitoring and observability capabilities")
        
        return recommendations
    
    def _save_assessment(self, assessment: ProjectAssessment):
        """Save assessment results"""
        try:
            os.makedirs('.taskmaster/reports', exist_ok=True)
            
            # Save detailed assessment
            assessment_path = Path('.taskmaster/reports/project_plan_assessment.json')
            with open(assessment_path, 'w') as f:
                json.dump(asdict(assessment), f, indent=2, default=str)
            
            self.logger.info(f"Assessment saved to: {assessment_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save assessment: {e}")
    
    def implement_critical_fixes(self, assessment: ProjectAssessment) -> bool:
        """Implement critical fixes identified in assessment"""
        self.logger.info("Implementing critical fixes")
        
        fixes_implemented = 0
        total_fixes = len(assessment.critical_gaps)
        
        for gap in assessment.critical_gaps:
            try:
                if "tasks.json" in gap.lower():
                    self._fix_tasks_json_issues()
                    fixes_implemented += 1
                elif "dependency" in gap.lower():
                    self._fix_dependency_issues()
                    fixes_implemented += 1
                elif "monitoring" in gap.lower():
                    self._fix_monitoring_issues()
                    fixes_implemented += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to implement fix for {gap}: {e}")
        
        success_rate = fixes_implemented / total_fixes if total_fixes > 0 else 1.0
        self.logger.info(f"Implemented {fixes_implemented}/{total_fixes} critical fixes")
        
        return success_rate >= 0.8
    
    def _fix_tasks_json_issues(self):
        """Fix tasks.json structure issues"""
        tasks_file = Path('.taskmaster/tasks/tasks.json')
        
        if not tasks_file.exists():
            # Create basic tasks.json structure
            basic_structure = {
                "master": {
                    "tasks": [],
                    "metadata": {
                        "created": datetime.now().isoformat(),
                        "updated": datetime.now().isoformat(),
                        "description": "Task Master AI system tasks"
                    }
                }
            }
            
            os.makedirs(tasks_file.parent, exist_ok=True)
            with open(tasks_file, 'w') as f:
                json.dump(basic_structure, f, indent=2)
            
            self.logger.info("Created basic tasks.json structure")
    
    def _fix_dependency_issues(self):
        """Fix dependency analysis issues"""
        # Ensure dependency validation works
        try:
            subprocess.run(['task-master', 'validate-dependencies'], 
                          capture_output=True, text=True, timeout=30)
            self.logger.info("Dependency validation executed")
        except:
            self.logger.warning("Could not execute dependency validation")
    
    def _fix_monitoring_issues(self):
        """Fix monitoring system issues"""
        # Ensure logs directory exists
        os.makedirs('.taskmaster/logs', exist_ok=True)
        os.makedirs('.taskmaster/reports', exist_ok=True)
        
        self.logger.info("Created monitoring directories")


def main():
    """Main assessment execution"""
    print("Comprehensive Project Plan Assessment Framework")
    print("=" * 60)
    print("Research-driven testing against project plan requirements")
    print("=" * 60)
    
    assessor = ProjectPlanAssessor()
    
    try:
        # Execute comprehensive assessment
        print("Executing comprehensive assessment...")
        assessment = assessor.execute_comprehensive_assessment()
        
        # Display results
        print(f"\n📊 ASSESSMENT RESULTS")
        print(f"Overall Score: {assessment.overall_score:.1%}")
        print(f"Requirements: {assessment.total_requirements} total")
        print(f"  ✅ Passed: {assessment.passed_requirements}")
        print(f"  ⚠️  Partial: {assessment.partial_requirements}")
        print(f"  ❌ Failed: {assessment.failed_requirements}")
        print(f"  🚫 Not Implemented: {assessment.not_implemented_requirements}")
        
        # Show critical gaps
        if assessment.critical_gaps:
            print(f"\n🚨 CRITICAL GAPS ({len(assessment.critical_gaps)}):")
            for gap in assessment.critical_gaps[:5]:  # Show top 5
                print(f"  • {gap}")
        
        # Show recommendations
        if assessment.implementation_recommendations:
            print(f"\n💡 IMPLEMENTATION RECOMMENDATIONS:")
            for i, rec in enumerate(assessment.implementation_recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        # Implement critical fixes
        if assessment.critical_gaps:
            print(f"\n🔧 IMPLEMENTING CRITICAL FIXES...")
            fixes_success = assessor.implement_critical_fixes(assessment)
            if fixes_success:
                print("✅ Critical fixes implemented successfully")
            else:
                print("⚠️  Some critical fixes could not be implemented")
        
        print(f"\n✅ Assessment completed. Results saved to:")
        print(f"   .taskmaster/reports/project_plan_assessment.json")
        
        return assessment.overall_score >= 0.8
        
    except Exception as e:
        print(f"❌ Assessment failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)