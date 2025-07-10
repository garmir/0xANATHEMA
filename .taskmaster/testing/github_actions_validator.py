#!/usr/bin/env python3
"""
GitHub Actions Validation and Assessment
Validates GitHub Actions functionality and contribution to project improvement
"""

import json
import os
import subprocess
# import yaml  # Not available, will parse manually
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

@dataclass
class WorkflowAnalysis:
    """Analysis of a single workflow"""
    name: str
    file_path: str
    triggers: List[str]
    jobs: List[str]
    dependencies: List[str]
    complexity_score: int
    automation_level: str
    contribution_score: int

@dataclass
class GitHubActionsReport:
    """Comprehensive GitHub Actions validation report"""
    validation_timestamp: datetime
    total_workflows: int
    functional_workflows: int
    workflow_analyses: List[WorkflowAnalysis]
    overall_automation_score: float
    project_contribution_score: float
    ci_cd_effectiveness: float
    recommendations: List[str]

class GitHubActionsValidator:
    """Validates GitHub Actions configuration and effectiveness"""
    
    def __init__(self):
        self.workflows_dir = Path('.github/workflows')
        self.scripts_dir = Path('.github/scripts')
        
    def analyze_workflow_file(self, workflow_path: Path) -> WorkflowAnalysis:
        """Analyze a single workflow file"""
        
        try:
            with open(workflow_path, 'r') as f:
                content = f.read()
            # Parse workflow manually since yaml not available
            workflow_data = self._parse_workflow_content(content)
        except Exception as e:
            return WorkflowAnalysis(
                name=workflow_path.stem,
                file_path=str(workflow_path),
                triggers=[],
                jobs=[],
                dependencies=[],
                complexity_score=0,
                automation_level="invalid",
                contribution_score=0
            )
        
        # Extract workflow information
        name = workflow_data.get('name', workflow_path.stem)
        
        # Analyze triggers
        triggers = workflow_data.get('on', [])
        if isinstance(triggers, str):
            triggers = [triggers]
        
        # Analyze jobs
        jobs = list(workflow_data.get('jobs', {}).keys())
        job_count = workflow_data.get('_job_count', len(jobs))
        
        # Analyze dependencies and complexity
        dependencies = []
        complexity_score = workflow_data.get('_step_count', 0)
        
        # Additional complexity from job count
        complexity_score += job_count * 2
        
        # Determine automation level
        automation_level = "basic"
        if 'schedule' in triggers:
            automation_level = "scheduled"
        if any('workflow_dispatch' in t for t in triggers):
            automation_level = "manual"
        if len(jobs) > 3 and complexity_score > 20:
            automation_level = "advanced"
        
        # Calculate contribution score based on workflow features
        contribution_score = 0
        
        # CI/CD features
        if 'push' in triggers or 'pull_request' in triggers:
            contribution_score += 25
        
        # Automation features
        if 'schedule' in triggers:
            contribution_score += 20
        
        # Parallel execution
        if len(jobs) > 1:
            contribution_score += 15
        
        # Advanced features
        if 'matrix' in str(workflow_data):
            contribution_score += 20
        
        # Task execution
        if 'task-master' in str(workflow_data) or 'claude' in str(workflow_data).lower():
            contribution_score += 20
        
        return WorkflowAnalysis(
            name=name,
            file_path=str(workflow_path),
            triggers=triggers,
            jobs=jobs,
            dependencies=dependencies,
            complexity_score=complexity_score,
            automation_level=automation_level,
            contribution_score=min(contribution_score, 100)
        )
    
    def validate_github_actions(self) -> GitHubActionsReport:
        """Validate all GitHub Actions workflows"""
        
        workflow_analyses = []
        total_workflows = 0
        functional_workflows = 0
        
        if self.workflows_dir.exists():
            for workflow_file in self.workflows_dir.glob('*.yml'):
                total_workflows += 1
                analysis = self.analyze_workflow_file(workflow_file)
                workflow_analyses.append(analysis)
                
                if analysis.contribution_score > 0:
                    functional_workflows += 1
        
        # Calculate overall scores
        if workflow_analyses:
            overall_automation_score = sum(a.complexity_score for a in workflow_analyses) / len(workflow_analyses)
            project_contribution_score = sum(a.contribution_score for a in workflow_analyses) / len(workflow_analyses)
        else:
            overall_automation_score = 0
            project_contribution_score = 0
        
        # Calculate CI/CD effectiveness
        ci_cd_effectiveness = 0
        if total_workflows > 0:
            ci_cd_effectiveness = (functional_workflows / total_workflows) * 100
        
        # Generate recommendations
        recommendations = self._generate_recommendations(workflow_analyses)
        
        return GitHubActionsReport(
            validation_timestamp=datetime.now(),
            total_workflows=total_workflows,
            functional_workflows=functional_workflows,
            workflow_analyses=workflow_analyses,
            overall_automation_score=overall_automation_score,
            project_contribution_score=project_contribution_score,
            ci_cd_effectiveness=ci_cd_effectiveness,
            recommendations=recommendations
        )
    
    def assess_project_improvement_contribution(self) -> Dict[str, Any]:
        """Assess how GitHub Actions contribute to project improvement"""
        
        contribution_assessment = {
            "automation_capabilities": {},
            "ci_cd_integration": {},
            "task_execution": {},
            "monitoring_and_reporting": {},
            "overall_contribution": {}
        }
        
        # Analyze automation capabilities
        automation_score = 0
        scheduled_workflows = 0
        manual_workflows = 0
        
        for workflow_file in self.workflows_dir.glob('*.yml'):
            try:
                with open(workflow_file, 'r') as f:
                    content = f.read()
                
                if 'schedule:' in content:
                    scheduled_workflows += 1
                    automation_score += 30
                
                if 'workflow_dispatch:' in content:
                    manual_workflows += 1
                    automation_score += 20
                
                if 'matrix:' in content:
                    automation_score += 25
                
            except Exception:
                continue
        
        contribution_assessment["automation_capabilities"] = {
            "scheduled_workflows": scheduled_workflows,
            "manual_workflows": manual_workflows,
            "automation_score": min(automation_score, 100),
            "assessment": "Excellent" if automation_score >= 80 else "Good" if automation_score >= 50 else "Basic"
        }
        
        # Analyze CI/CD integration
        ci_cd_score = 0
        has_ci = False
        has_cd = False
        
        for workflow_file in self.workflows_dir.glob('*.yml'):
            try:
                with open(workflow_file, 'r') as f:
                    content = f.read()
                
                if any(trigger in content for trigger in ['push:', 'pull_request:']):
                    has_ci = True
                    ci_cd_score += 40
                
                if 'deploy' in content.lower() or 'release' in content.lower():
                    has_cd = True
                    ci_cd_score += 30
                
                if 'test' in content.lower() or 'validation' in content.lower():
                    ci_cd_score += 30
                
            except Exception:
                continue
        
        contribution_assessment["ci_cd_integration"] = {
            "continuous_integration": has_ci,
            "continuous_deployment": has_cd,
            "ci_cd_score": min(ci_cd_score, 100),
            "assessment": "Excellent" if ci_cd_score >= 80 else "Good" if ci_cd_score >= 50 else "Basic"
        }
        
        # Analyze task execution capabilities
        task_execution_score = 0
        claude_integration = False
        task_master_integration = False
        parallel_execution = False
        
        for workflow_file in self.workflows_dir.glob('*.yml'):
            try:
                with open(workflow_file, 'r') as f:
                    content = f.read()
                
                if 'claude' in content.lower():
                    claude_integration = True
                    task_execution_score += 35
                
                if 'task-master' in content:
                    task_master_integration = True
                    task_execution_score += 35
                
                if 'matrix:' in content or 'parallel' in content.lower():
                    parallel_execution = True
                    task_execution_score += 30
                
            except Exception:
                continue
        
        contribution_assessment["task_execution"] = {
            "claude_integration": claude_integration,
            "task_master_integration": task_master_integration,
            "parallel_execution": parallel_execution,
            "task_execution_score": min(task_execution_score, 100),
            "assessment": "Excellent" if task_execution_score >= 80 else "Good" if task_execution_score >= 50 else "Basic"
        }
        
        # Analyze monitoring and reporting
        monitoring_score = 0
        artifact_uploads = 0
        reporting_workflows = 0
        
        for workflow_file in self.workflows_dir.glob('*.yml'):
            try:
                with open(workflow_file, 'r') as f:
                    content = f.read()
                
                if 'upload-artifact' in content:
                    artifact_uploads += 1
                    monitoring_score += 20
                
                if any(keyword in content.lower() for keyword in ['report', 'summary', 'metrics', 'assessment']):
                    reporting_workflows += 1
                    monitoring_score += 25
                
            except Exception:
                continue
        
        contribution_assessment["monitoring_and_reporting"] = {
            "artifact_uploads": artifact_uploads,
            "reporting_workflows": reporting_workflows,
            "monitoring_score": min(monitoring_score, 100),
            "assessment": "Excellent" if monitoring_score >= 80 else "Good" if monitoring_score >= 50 else "Basic"
        }
        
        # Calculate overall contribution
        scores = [
            contribution_assessment["automation_capabilities"]["automation_score"],
            contribution_assessment["ci_cd_integration"]["ci_cd_score"],
            contribution_assessment["task_execution"]["task_execution_score"],
            contribution_assessment["monitoring_and_reporting"]["monitoring_score"]
        ]
        
        overall_score = sum(scores) / len(scores)
        
        contribution_assessment["overall_contribution"] = {
            "overall_score": overall_score,
            "assessment": "Excellent" if overall_score >= 80 else "Good" if overall_score >= 60 else "Needs Improvement",
            "contributes_to_improvement": overall_score >= 60
        }
        
        return contribution_assessment
    
    def _generate_recommendations(self, analyses: List[WorkflowAnalysis]) -> List[str]:
        """Generate recommendations for improving GitHub Actions"""
        
        recommendations = []
        
        if not analyses:
            recommendations.append("No GitHub Actions workflows found - consider implementing CI/CD automation")
            return recommendations
        
        # Check for scheduled automation
        scheduled_count = sum(1 for a in analyses if 'schedule' in a.triggers)
        if scheduled_count == 0:
            recommendations.append("Consider adding scheduled workflows for continuous monitoring")
        
        # Check for parallel execution
        parallel_count = sum(1 for a in analyses if len(a.jobs) > 1)
        if parallel_count < len(analyses) * 0.5:
            recommendations.append("Implement parallel job execution to improve performance")
        
        # Check for task automation
        task_automation = sum(1 for a in analyses if a.contribution_score >= 50)
        if task_automation < len(analyses) * 0.7:
            recommendations.append("Enhance task automation and Claude integration")
        
        # Check complexity
        avg_complexity = sum(a.complexity_score for a in analyses) / len(analyses)
        if avg_complexity < 10:
            recommendations.append("Expand workflow complexity for more comprehensive automation")
        
        return recommendations
    
    def _parse_workflow_content(self, content: str) -> Dict[str, Any]:
        """Simple YAML-like parsing for workflow files"""
        lines = content.split('\n')
        workflow_data = {'jobs': {}}
        
        # Extract name
        for line in lines:
            if line.strip().startswith('name:'):
                workflow_data['name'] = line.split(':', 1)[1].strip().strip('"\'')
                break
        
        # Extract triggers (on:)
        triggers = []
        in_on_section = False
        for line in lines:
            if line.strip().startswith('on:'):
                in_on_section = True
                # Check if inline
                rest = line.split(':', 1)[1].strip()
                if rest:
                    triggers.append(rest)
            elif in_on_section and line.strip() and not line.startswith(' '):
                in_on_section = False
            elif in_on_section and line.strip().startswith('- '):
                triggers.append(line.strip()[2:])
            elif in_on_section and ':' in line and not line.strip().startswith('#'):
                trigger = line.strip().split(':')[0]
                if trigger:
                    triggers.append(trigger)
        
        workflow_data['on'] = triggers
        
        # Extract jobs
        in_jobs = False
        current_job = None
        job_count = 0
        step_count = 0
        
        for line in lines:
            if line.strip().startswith('jobs:'):
                in_jobs = True
                continue
            elif in_jobs and line.strip() and not line.startswith(' '):
                in_jobs = False
            elif in_jobs and line.strip().endswith(':') and not line.strip().startswith('#'):
                current_job = line.strip()[:-1]
                if current_job:
                    job_count += 1
                    workflow_data['jobs'][current_job] = {}
            elif in_jobs and '- name:' in line or '- uses:' in line:
                step_count += 1
        
        workflow_data['_job_count'] = job_count
        workflow_data['_step_count'] = step_count
        
        return workflow_data

def main():
    """Main GitHub Actions validation execution"""
    print("GitHub Actions Validator")
    print("=" * 40)
    
    validator = GitHubActionsValidator()
    
    try:
        # Validate workflows
        report = validator.validate_github_actions()
        
        print(f"Total Workflows: {report.total_workflows}")
        print(f"Functional Workflows: {report.functional_workflows}")
        print(f"CI/CD Effectiveness: {report.ci_cd_effectiveness:.1f}%")
        print(f"Automation Score: {report.overall_automation_score:.1f}")
        print(f"Contribution Score: {report.project_contribution_score:.1f}")
        
        print(f"\nWorkflow Analysis:")
        for analysis in report.workflow_analyses:
            print(f"  • {analysis.name}: {analysis.contribution_score}% contribution")
            print(f"    Triggers: {', '.join(analysis.triggers)}")
            print(f"    Jobs: {len(analysis.jobs)} | Complexity: {analysis.complexity_score}")
        
        # Assess project contribution
        contribution = validator.assess_project_improvement_contribution()
        
        print(f"\nProject Improvement Contribution:")
        for category, data in contribution.items():
            if isinstance(data, dict) and 'assessment' in data:
                print(f"  • {category.replace('_', ' ').title()}: {data['assessment']}")
        
        overall_contributes = contribution["overall_contribution"]["contributes_to_improvement"]
        print(f"\nContributes to Project Improvement: {'✅ Yes' if overall_contributes else '❌ No'}")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations:
                print(f"  • {rec}")
        
        # Save detailed report
        os.makedirs('.taskmaster/reports', exist_ok=True)
        
        detailed_report = {
            "github_actions_validation": asdict(report),
            "project_contribution_assessment": contribution,
            "validation_timestamp": datetime.now().isoformat()
        }
        
        with open('.taskmaster/reports/github_actions_validation.json', 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
        
        print(f"\n✅ GitHub Actions validation completed")
        print(f"Detailed report saved to: .taskmaster/reports/github_actions_validation.json")
        
        return overall_contributes
        
    except Exception as e:
        print(f"❌ GitHub Actions validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)