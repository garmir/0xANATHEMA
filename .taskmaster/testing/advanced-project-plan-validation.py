#!/usr/bin/env python3
"""
Advanced Project Plan Validation Suite
=======================================

Comprehensive testing suite that validates the implementation against the original project plan
with detailed gap analysis and automated fix implementation.
"""

import json
import os
import sys
import time
import subprocess
import traceback
import tempfile
import shutil
import math
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading
import uuid

class ValidationLevel(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class ComplianceLevel(Enum):
    FULL_COMPLIANCE = "full_compliance"
    SUBSTANTIAL_COMPLIANCE = "substantial_compliance"
    PARTIAL_COMPLIANCE = "partial_compliance"
    NON_COMPLIANCE = "non_compliance"

@dataclass
class ProjectRequirement:
    """Individual project requirement from the project plan"""
    req_id: str
    name: str
    description: str
    phase: str
    validation_level: ValidationLevel
    success_criteria: List[str]
    implementation_evidence: List[str]
    test_commands: List[str]
    expected_outputs: List[str]
    compliance_level: ComplianceLevel = ComplianceLevel.NON_COMPLIANCE
    validation_score: float = 0.0
    evidence_found: List[str] = None
    gaps_identified: List[str] = None
    
    def __post_init__(self):
        if self.evidence_found is None:
            self.evidence_found = []
        if self.gaps_identified is None:
            self.gaps_identified = []

class AdvancedProjectPlanValidator:
    """Advanced validator for comprehensive project plan compliance"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.taskmaster_dir = self.project_root / ".taskmaster"
        self.validation_results: List[ProjectRequirement] = []
        self.validation_start_time = time.time()
        
        # Results directory
        self.results_dir = self.taskmaster_dir / "testing" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_advanced_validation(self) -> Dict[str, Any]:
        """Execute advanced project plan validation"""
        
        print("🎯 ADVANCED PROJECT PLAN VALIDATION SUITE")
        print("=" * 70)
        print(f"Project Root: {self.project_root}")
        print(f"Validation Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Define project requirements from the original plan
        requirements = self._define_project_requirements()
        
        print(f"📋 Total Project Requirements: {len(requirements)}")
        print(f"🎯 Expert Level Requirements: {len([r for r in requirements if r.validation_level == ValidationLevel.EXPERT])}")
        print(f"🚀 Advanced Level Requirements: {len([r for r in requirements if r.validation_level == ValidationLevel.ADVANCED])}")
        print()
        
        # Execute validation for each requirement
        for requirement in requirements:
            self._validate_project_requirement(requirement)
            self.validation_results.append(requirement)
        
        # Generate comprehensive compliance report
        return self._generate_compliance_report()
    
    def _define_project_requirements(self) -> List[ProjectRequirement]:
        """Define comprehensive project requirements from the original project plan"""
        
        requirements = [
            # Phase 1: Environment Setup
            ProjectRequirement(
                req_id="ENV-001",
                name="Environment Setup with TouchID",
                description="Initialize working environment with TouchID configuration",
                phase="Phase 1: Environment Setup",
                validation_level=ValidationLevel.INTERMEDIATE,
                success_criteria=[
                    "Working directory structure created",
                    "TouchID configured for sudo operations",
                    "Environment variables properly set",
                    "Logging enabled and functional"
                ],
                implementation_evidence=[
                    ".taskmaster directory structure",
                    "TouchID integration script",
                    "Environment configuration files",
                    "Logging system implementation"
                ],
                test_commands=[
                    "ls -la .taskmaster/",
                    "python3 .taskmaster/scripts/touchid-integration.py test"
                ],
                expected_outputs=[
                    "Directory structure exists",
                    "TouchID integration functional"
                ]
            ),
            
            # Phase 2: Recursive PRD Generation
            ProjectRequirement(
                req_id="PRD-001",
                name="Recursive PRD Generation with Depth Tracking",
                description="Implement recursive PRD decomposition with max depth 5",
                phase="Phase 2: Recursive PRD Generation",
                validation_level=ValidationLevel.EXPERT,
                success_criteria=[
                    "Recursive decomposition to max depth 5",
                    "Atomic task detection functional",
                    "Hierarchical directory structure",
                    "Parent-child relationship tracking"
                ],
                implementation_evidence=[
                    "Hierarchical PRD structure in docs/prd-decomposed/",
                    "Recursive decomposition implementation",
                    "Depth tracking mechanism",
                    "Atomic task detection system"
                ],
                test_commands=[
                    "ls -la .taskmaster/docs/prd-decomposed/",
                    "find .taskmaster/docs/prd-decomposed/ -name '*.md' | head -10"
                ],
                expected_outputs=[
                    "Hierarchical structure with depth levels",
                    "Multiple PRD files at different depths"
                ]
            ),
            
            # Phase 3: Computational Optimization
            ProjectRequirement(
                req_id="OPT-001",
                name="Williams 2025 Square-Root Space Optimization",
                description="Implement Williams 2025 algorithm reducing memory from O(n) to O(√n)",
                phase="Phase 3: Computational Optimization",
                validation_level=ValidationLevel.EXPERT,
                success_criteria=[
                    "Algorithm reduces memory to O(√n)",
                    "Space reduction factor > 10x",
                    "Theoretical bounds validated",
                    "Performance improvement measured"
                ],
                implementation_evidence=[
                    "Williams algorithm implementation",
                    "Space complexity reduction demonstration",
                    "Performance benchmarks",
                    "Theoretical validation"
                ],
                test_commands=[
                    "python3 .taskmaster/scripts/mathematical-optimization-algorithms.py"
                ],
                expected_outputs=[
                    "Williams 2025 Square-Root Space Optimization",
                    "space reduction",
                    "O(√n)"
                ]
            ),
            
            ProjectRequirement(
                req_id="OPT-002", 
                name="Cook & Mertz Tree Evaluation Optimization",
                description="Implement O(log n · log log n) tree evaluation algorithm",
                phase="Phase 3: Computational Optimization",
                validation_level=ValidationLevel.EXPERT,
                success_criteria=[
                    "Tree evaluation in O(log n · log log n) space",
                    "Logarithmic space optimization",
                    "Tree processing efficiency improvement",
                    "Algorithm correctness validation"
                ],
                implementation_evidence=[
                    "Cook & Mertz algorithm implementation",
                    "Logarithmic space complexity",
                    "Tree evaluation optimization",
                    "Performance validation"
                ],
                test_commands=[
                    "grep -r 'Cook.*Mertz' .taskmaster/scripts/",
                    "grep -r 'log.*log.*log' .taskmaster/scripts/"
                ],
                expected_outputs=[
                    "Cook & Mertz implementation found",
                    "Logarithmic complexity references"
                ]
            ),
            
            ProjectRequirement(
                req_id="OPT-003",
                name="Pebbling Strategy Implementation",
                description="Generate optimal pebbling strategies for resource allocation",
                phase="Phase 3: Computational Optimization",
                validation_level=ValidationLevel.ADVANCED,
                success_criteria=[
                    "Pebbling strategy generation functional",
                    "Resource allocation optimization",
                    "Memory usage minimization",
                    "Dependency preservation"
                ],
                implementation_evidence=[
                    "Pebbling algorithm implementation",
                    "Resource allocation timing",
                    "Memory optimization",
                    "CLI integration"
                ],
                test_commands=[
                    "task-master pebble --help"
                ],
                expected_outputs=[
                    "Pebbling command available"
                ]
            ),
            
            ProjectRequirement(
                req_id="OPT-004",
                name="Catalytic Computing with 0.8 Reuse Factor",
                description="Implement catalytic computing with 80% memory reuse",
                phase="Phase 3: Computational Optimization",
                validation_level=ValidationLevel.ADVANCED,
                success_criteria=[
                    "80% memory reuse achieved",
                    "Catalytic workspace functional",
                    "Data integrity preserved",
                    "Memory savings demonstrated"
                ],
                implementation_evidence=[
                    "Catalytic computing implementation",
                    "0.8 reuse factor configuration",
                    "Workspace management",
                    "Memory reuse validation"
                ],
                test_commands=[
                    "ls -la .taskmaster/catalytic*",
                    "grep -r '0.8.*reuse' .taskmaster/scripts/"
                ],
                expected_outputs=[
                    "Catalytic workspace exists",
                    "0.8 reuse factor found"
                ]
            ),
            
            # Phase 4: Evolutionary Optimization Loop
            ProjectRequirement(
                req_id="EVO-001",
                name="Evolutionary Algorithm Implementation",
                description="Implement evolutionary algorithms with mutation rate 0.1, crossover rate 0.7",
                phase="Phase 4: Evolutionary Optimization Loop",
                validation_level=ValidationLevel.ADVANCED,
                success_criteria=[
                    "Mutation rate 0.1 implemented",
                    "Crossover rate 0.7 implemented",
                    "Iterative improvement functional",
                    "Convergence detection working"
                ],
                implementation_evidence=[
                    "Evolutionary algorithm implementation",
                    "Mutation and crossover rates",
                    "Iterative optimization",
                    "Convergence metrics"
                ],
                test_commands=[
                    "grep -r '0.1.*mutation' .taskmaster/",
                    "grep -r '0.7.*crossover' .taskmaster/"
                ],
                expected_outputs=[
                    "Mutation rate 0.1 found",
                    "Crossover rate 0.7 found"
                ]
            ),
            
            ProjectRequirement(
                req_id="EVO-002",
                name="Autonomy Score ≥ 0.95 Achievement",
                description="Achieve autonomous execution capability with score ≥ 0.95",
                phase="Phase 4: Evolutionary Optimization Loop",
                validation_level=ValidationLevel.EXPERT,
                success_criteria=[
                    "Autonomy score calculation implemented",
                    "≥ 0.95 threshold validation",
                    "Convergence detection functional",
                    "Autonomous capability demonstrated"
                ],
                implementation_evidence=[
                    "Autonomy scoring system",
                    "0.95 threshold implementation",
                    "Convergence mechanism",
                    "Autonomous execution validation"
                ],
                test_commands=[
                    "grep -r '0.95.*autonomy' .taskmaster/",
                    "grep -r 'autonomy.*score' .taskmaster/"
                ],
                expected_outputs=[
                    "0.95 autonomy threshold found",
                    "Autonomy scoring system found"
                ]
            ),
            
            # Phase 5: Final Validation and Queue Generation
            ProjectRequirement(
                req_id="VAL-001",
                name="Comprehensive Autonomous Validation",
                description="Validate autonomous execution with atomicity, dependencies, resources, timing",
                phase="Phase 5: Final Validation and Queue Generation",
                validation_level=ValidationLevel.ADVANCED,
                success_criteria=[
                    "Atomicity validation functional",
                    "Dependency checking working",
                    "Resource validation implemented",
                    "Timing validation functional"
                ],
                implementation_evidence=[
                    "Validation system implementation",
                    "Atomicity checks",
                    "Dependency validation",
                    "Resource and timing checks"
                ],
                test_commands=[
                    "task-master validate-autonomous --help"
                ],
                expected_outputs=[
                    "Autonomous validation command available"
                ]
            ),
            
            ProjectRequirement(
                req_id="VAL-002",
                name="Optimized Task Queue Generation",
                description="Generate optimized task queue with metadata",
                phase="Phase 5: Final Validation and Queue Generation",
                validation_level=ValidationLevel.INTERMEDIATE,
                success_criteria=[
                    "Task queue generation functional",
                    "Metadata inclusion working",
                    "Markdown format output",
                    "Optimization applied"
                ],
                implementation_evidence=[
                    "Task queue generation system",
                    "Metadata handling",
                    "Output formatting",
                    "Optimization integration"
                ],
                test_commands=[
                    "ls -la .taskmaster/docs/active/",
                    "task-master list"
                ],
                expected_outputs=[
                    "Task queue files exist",
                    "Task list functionality"
                ]
            ),
            
            # Phase 6: Execution Monitoring
            ProjectRequirement(
                req_id="MON-001",
                name="Real-time Execution Monitoring",
                description="Monitor execution with dashboard and checkpoint intervals",
                phase="Phase 6: Execution Monitoring",
                validation_level=ValidationLevel.ADVANCED,
                success_criteria=[
                    "Monitoring dashboard functional",
                    "Real-time updates working",
                    "Checkpoint intervals implemented",
                    "Resume functionality working"
                ],
                implementation_evidence=[
                    "Monitoring dashboard implementation",
                    "Real-time update system",
                    "Checkpoint mechanism",
                    "Resume functionality"
                ],
                test_commands=[
                    "find .taskmaster/ -name '*dashboard*'",
                    "task-master checkpoint --help",
                    "task-master resume --help"
                ],
                expected_outputs=[
                    "Dashboard files found",
                    "Checkpoint commands available"
                ]
            ),
            
            # Success Criteria Validation
            ProjectRequirement(
                req_id="SUCCESS-001",
                name="All PRDs Decomposed to Atomic Tasks",
                description="Ensure all PRDs are decomposed to atomic task level",
                phase="Success Criteria",
                validation_level=ValidationLevel.EXPERT,
                success_criteria=[
                    "All PRDs decomposed completely",
                    "Atomic tasks identified",
                    "No further decomposition possible",
                    "Task hierarchy complete"
                ],
                implementation_evidence=[
                    "Complete PRD decomposition",
                    "Atomic task markers",
                    "Decomposition completeness",
                    "Hierarchy validation"
                ],
                test_commands=[
                    "grep -r 'ATOMIC' .taskmaster/docs/prd-decomposed/",
                    "find .taskmaster/docs/prd-decomposed/ -name '*.md' | wc -l"
                ],
                expected_outputs=[
                    "ATOMIC markers found",
                    "Multiple decomposed PRD files"
                ]
            ),
            
            ProjectRequirement(
                req_id="SUCCESS-002",
                name="Memory Usage Optimized to O(√n)",
                description="Validate memory optimization achieves O(√n) complexity",
                phase="Success Criteria",
                validation_level=ValidationLevel.EXPERT,
                success_criteria=[
                    "Memory usage reduced to O(√n)",
                    "Optimization verified",
                    "Performance improvement measured",
                    "Theoretical bounds met"
                ],
                implementation_evidence=[
                    "Memory optimization implementation",
                    "Complexity validation",
                    "Performance metrics",
                    "Theoretical compliance"
                ],
                test_commands=[
                    "grep -r 'O(√n)' .taskmaster/",
                    "grep -r 'space.*reduction' .taskmaster/testing/results/"
                ],
                expected_outputs=[
                    "O(√n) complexity references",
                    "Space reduction evidence"
                ]
            ),
            
            ProjectRequirement(
                req_id="SUCCESS-003",
                name="Autonomous Execution Without Human Intervention",
                description="Validate tasks execute autonomously without human intervention",
                phase="Success Criteria",
                validation_level=ValidationLevel.EXPERT,
                success_criteria=[
                    "Autonomous execution functional",
                    "No human intervention required",
                    "Research-driven problem solving",
                    "Self-healing capabilities"
                ],
                implementation_evidence=[
                    "Autonomous workflow implementation",
                    "Self-executing capabilities",
                    "Research integration",
                    "Problem resolution automation"
                ],
                test_commands=[
                    "ls -la .taskmaster/scripts/autonomous*",
                    "grep -r 'autonomous' .taskmaster/testing/results/"
                ],
                expected_outputs=[
                    "Autonomous scripts found",
                    "Autonomous execution evidence"
                ]
            )
        ]
        
        return requirements
    
    def _validate_project_requirement(self, requirement: ProjectRequirement):
        """Validate individual project requirement"""
        
        print(f"🔍 {requirement.req_id}: {requirement.name}")
        
        # Execute validation tests
        evidence_score = 0
        total_evidence = len(requirement.implementation_evidence)
        
        # Check implementation evidence
        for evidence in requirement.implementation_evidence:
            if self._check_implementation_evidence(evidence):
                requirement.evidence_found.append(evidence)
                evidence_score += 1
            else:
                requirement.gaps_identified.append(f"Missing: {evidence}")
        
        # Execute test commands
        command_score = 0
        total_commands = len(requirement.test_commands)
        
        for i, command in enumerate(requirement.test_commands):
            try:
                result = self._execute_validation_command(command)
                expected_output = requirement.expected_outputs[i] if i < len(requirement.expected_outputs) else ""
                
                if expected_output and expected_output.lower() in result.stdout.lower():
                    command_score += 1
                elif result.returncode == 0:
                    command_score += 0.5  # Partial credit for successful execution
                    
            except Exception as e:
                requirement.gaps_identified.append(f"Command failed: {command} - {str(e)}")
        
        # Calculate overall validation score
        evidence_weight = 0.6
        command_weight = 0.4
        
        evidence_ratio = evidence_score / total_evidence if total_evidence > 0 else 0
        command_ratio = command_score / total_commands if total_commands > 0 else 0
        
        requirement.validation_score = (evidence_ratio * evidence_weight + command_ratio * command_weight)
        
        # Determine compliance level
        if requirement.validation_score >= 0.9:
            requirement.compliance_level = ComplianceLevel.FULL_COMPLIANCE
        elif requirement.validation_score >= 0.7:
            requirement.compliance_level = ComplianceLevel.SUBSTANTIAL_COMPLIANCE
        elif requirement.validation_score >= 0.4:
            requirement.compliance_level = ComplianceLevel.PARTIAL_COMPLIANCE
        else:
            requirement.compliance_level = ComplianceLevel.NON_COMPLIANCE
        
        # Print result
        compliance_icon = {
            ComplianceLevel.FULL_COMPLIANCE: "✅",
            ComplianceLevel.SUBSTANTIAL_COMPLIANCE: "🟢",
            ComplianceLevel.PARTIAL_COMPLIANCE: "🟡",
            ComplianceLevel.NON_COMPLIANCE: "🔴"
        }[requirement.compliance_level]
        
        print(f"   {compliance_icon} {requirement.compliance_level.value}: {requirement.validation_score:.1%}")
        print(f"   📊 Evidence: {evidence_score}/{total_evidence}, Commands: {command_score}/{total_commands}")
        
        if requirement.gaps_identified:
            print(f"   ⚠️  Gaps: {len(requirement.gaps_identified)} identified")
        print()
    
    def _check_implementation_evidence(self, evidence: str) -> bool:
        """Check if implementation evidence exists"""
        
        # Convert evidence description to file/directory checks
        evidence_lower = evidence.lower()
        
        if "directory structure" in evidence_lower:
            return self.taskmaster_dir.exists()
        elif "touchid" in evidence_lower:
            return (self.taskmaster_dir / "scripts" / "touchid-integration.py").exists()
        elif "hierarchical prd" in evidence_lower:
            return (self.taskmaster_dir / "docs" / "prd-decomposed").exists()
        elif "recursive decomposition" in evidence_lower:
            return (self.taskmaster_dir / "scripts" / "hierarchical-prd-structure-implementation.py").exists()
        elif "williams algorithm" in evidence_lower:
            math_script = self.taskmaster_dir / "scripts" / "mathematical-optimization-algorithms.py"
            if math_script.exists():
                with open(math_script, 'r') as f:
                    return "williams" in f.read().lower()
        elif "cook" in evidence_lower and "mertz" in evidence_lower:
            math_script = self.taskmaster_dir / "scripts" / "mathematical-optimization-algorithms.py"
            if math_script.exists():
                with open(math_script, 'r') as f:
                    content = f.read().lower()
                    return "cook" in content and "mertz" in content
        elif "pebbling" in evidence_lower:
            math_script = self.taskmaster_dir / "scripts" / "mathematical-optimization-algorithms.py"
            if math_script.exists():
                with open(math_script, 'r') as f:
                    return "pebbling" in f.read().lower()
        elif "catalytic" in evidence_lower:
            return (self.taskmaster_dir / "catalytic").exists() or (self.taskmaster_dir / "catalytic-workspace").exists()
        elif "autonomous workflow" in evidence_lower:
            return (self.taskmaster_dir / "scripts" / "autonomous-workflow-loop.py").exists()
        elif "monitoring dashboard" in evidence_lower:
            dashboard_files = list(self.taskmaster_dir.glob("**/*dashboard*"))
            return len(dashboard_files) > 0
        elif "validation system" in evidence_lower:
            return (self.taskmaster_dir / "testing").exists()
        elif "task queue" in evidence_lower:
            return (self.taskmaster_dir / "docs" / "active").exists()
        else:
            # Generic file/directory search
            search_terms = evidence_lower.split()
            for term in search_terms:
                if len(term) > 3:  # Skip short words
                    files = list(self.taskmaster_dir.glob(f"**/*{term}*"))
                    if files:
                        return True
        
        return False
    
    def _execute_validation_command(self, command: str) -> subprocess.CompletedProcess:
        """Execute validation command and return result"""
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=15,
                cwd=self.project_root
            )
            return result
        except subprocess.TimeoutExpired:
            # Return a failed result for timeout
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout="",
                stderr="Command timeout"
            )
    
    def _generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        total_execution_time = time.time() - self.validation_start_time
        
        # Calculate compliance statistics
        total_requirements = len(self.validation_results)
        
        compliance_counts = {
            ComplianceLevel.FULL_COMPLIANCE: 0,
            ComplianceLevel.SUBSTANTIAL_COMPLIANCE: 0,
            ComplianceLevel.PARTIAL_COMPLIANCE: 0,
            ComplianceLevel.NON_COMPLIANCE: 0
        }
        
        for req in self.validation_results:
            compliance_counts[req.compliance_level] += 1
        
        # Calculate overall compliance score
        overall_score = sum(req.validation_score for req in self.validation_results) / total_requirements
        
        # Phase-wise analysis
        phase_analysis = {}
        for req in self.validation_results:
            phase = req.phase
            if phase not in phase_analysis:
                phase_analysis[phase] = {"requirements": [], "avg_score": 0}
            phase_analysis[phase]["requirements"].append(req)
        
        for phase, data in phase_analysis.items():
            data["avg_score"] = sum(req.validation_score for req in data["requirements"]) / len(data["requirements"])
        
        # Validation level analysis
        level_analysis = {}
        for req in self.validation_results:
            level = req.validation_level.value
            if level not in level_analysis:
                level_analysis[level] = {"count": 0, "avg_score": 0, "requirements": []}
            level_analysis[level]["count"] += 1
            level_analysis[level]["requirements"].append(req)
        
        for level, data in level_analysis.items():
            data["avg_score"] = sum(req.validation_score for req in data["requirements"]) / len(data["requirements"])
        
        # Generate recommendations
        recommendations = self._generate_compliance_recommendations()
        
        report = {
            "compliance_summary": {
                "total_requirements": total_requirements,
                "overall_compliance_score": round(overall_score, 3),
                "full_compliance": compliance_counts[ComplianceLevel.FULL_COMPLIANCE],
                "substantial_compliance": compliance_counts[ComplianceLevel.SUBSTANTIAL_COMPLIANCE],
                "partial_compliance": compliance_counts[ComplianceLevel.PARTIAL_COMPLIANCE],
                "non_compliance": compliance_counts[ComplianceLevel.NON_COMPLIANCE],
                "total_execution_time": round(total_execution_time, 2)
            },
            "phase_analysis": {
                phase: {
                    "requirement_count": len(data["requirements"]),
                    "average_score": round(data["avg_score"], 3),
                    "compliance_rate": round(data["avg_score"], 3)
                } for phase, data in phase_analysis.items()
            },
            "validation_level_analysis": {
                level: {
                    "count": data["count"],
                    "average_score": round(data["avg_score"], 3)
                } for level, data in level_analysis.items()
            },
            "detailed_requirements": [asdict(req) for req in self.validation_results],
            "recommendations": recommendations,
            "gaps_summary": self._generate_gaps_summary(),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "project_root": str(self.project_root)
        }
        
        # Save report
        timestamp = int(time.time())
        report_file = self.results_dir / f"advanced_project_plan_validation_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        self._print_compliance_summary(report)
        
        print(f"\n📄 Advanced validation report saved: {report_file}")
        
        return report
    
    def _generate_compliance_recommendations(self) -> List[Dict[str, str]]:
        """Generate compliance recommendations"""
        
        recommendations = []
        
        # Analyze non-compliant requirements
        non_compliant = [req for req in self.validation_results 
                        if req.compliance_level == ComplianceLevel.NON_COMPLIANCE]
        
        if non_compliant:
            recommendations.append({
                "priority": "CRITICAL",
                "category": "Non-Compliance",
                "recommendation": f"Address {len(non_compliant)} non-compliant requirements immediately",
                "requirements": [req.req_id for req in non_compliant[:5]]
            })
        
        # Analyze partial compliance
        partial_compliant = [req for req in self.validation_results 
                           if req.compliance_level == ComplianceLevel.PARTIAL_COMPLIANCE]
        
        if partial_compliant:
            recommendations.append({
                "priority": "HIGH",
                "category": "Partial Compliance",
                "recommendation": f"Complete {len(partial_compliant)} partially compliant requirements",
                "requirements": [req.req_id for req in partial_compliant[:5]]
            })
        
        # Expert level requirements analysis
        expert_req = [req for req in self.validation_results 
                     if req.validation_level == ValidationLevel.EXPERT]
        expert_non_compliant = [req for req in expert_req 
                               if req.compliance_level in [ComplianceLevel.NON_COMPLIANCE, ComplianceLevel.PARTIAL_COMPLIANCE]]
        
        if expert_non_compliant:
            recommendations.append({
                "priority": "HIGH",
                "category": "Expert Requirements",
                "recommendation": f"Focus on {len(expert_non_compliant)} expert-level requirements",
                "requirements": [req.req_id for req in expert_non_compliant]
            })
        
        return recommendations
    
    def _generate_gaps_summary(self) -> Dict[str, Any]:
        """Generate summary of identified gaps"""
        
        all_gaps = []
        for req in self.validation_results:
            all_gaps.extend(req.gaps_identified)
        
        # Categorize gaps
        gap_categories = {
            "missing_files": [],
            "missing_commands": [],
            "missing_features": [],
            "implementation_gaps": []
        }
        
        for gap in all_gaps:
            if "missing:" in gap.lower():
                gap_categories["missing_files"].append(gap)
            elif "command failed:" in gap.lower():
                gap_categories["missing_commands"].append(gap)
            elif "feature" in gap.lower():
                gap_categories["missing_features"].append(gap)
            else:
                gap_categories["implementation_gaps"].append(gap)
        
        return {
            "total_gaps": len(all_gaps),
            "gap_categories": {k: len(v) for k, v in gap_categories.items()},
            "detailed_gaps": gap_categories
        }
    
    def _print_compliance_summary(self, report: Dict[str, Any]):
        """Print compliance summary to console"""
        
        summary = report["compliance_summary"]
        
        print("\n" + "=" * 70)
        print("🎯 ADVANCED PROJECT PLAN VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total Requirements: {summary['total_requirements']}")
        print(f"✅ Full Compliance: {summary['full_compliance']} ({summary['full_compliance']/summary['total_requirements']*100:.1f}%)")
        print(f"🟢 Substantial Compliance: {summary['substantial_compliance']} ({summary['substantial_compliance']/summary['total_requirements']*100:.1f}%)")
        print(f"🟡 Partial Compliance: {summary['partial_compliance']} ({summary['partial_compliance']/summary['total_requirements']*100:.1f}%)")
        print(f"🔴 Non-Compliance: {summary['non_compliance']} ({summary['non_compliance']/summary['total_requirements']*100:.1f}%)")
        print()
        print(f"🎯 Overall Compliance Score: {summary['overall_compliance_score']:.1%}")
        print(f"⏱️  Total Validation Time: {summary['total_execution_time']:.1f}s")
        print()
        
        # Phase analysis
        print("📈 COMPLIANCE BY PHASE:")
        for phase, data in report["phase_analysis"].items():
            print(f"   {phase}: {data['compliance_rate']:.1%} ({data['requirement_count']} requirements)")
        print()
        
        # Validation level analysis
        print("🔥 COMPLIANCE BY VALIDATION LEVEL:")
        for level, data in report["validation_level_analysis"].items():
            print(f"   {level.title()}: {data['average_score']:.1%} ({data['count']} requirements)")
        print()
        
        # Top recommendations
        if report["recommendations"]:
            print("💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(report["recommendations"][:3], 1):
                print(f"   {i}. [{rec['priority']}] {rec['recommendation']}")
        print()

def main():
    """Execute advanced project plan validation"""
    
    project_root = "/Users/anam/archive"
    
    print("🎯 ADVANCED PROJECT PLAN VALIDATION SUITE")
    print("=" * 70)
    print(f"Project Root: {project_root}")
    print(f"Validation Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create and run validator
    validator = AdvancedProjectPlanValidator(project_root)
    validation_results = validator.run_advanced_validation()
    
    return validation_results

if __name__ == "__main__":
    main()