#!/usr/bin/env python3
"""
GitHub Actions Deployment Verification
Verify that all workflows are properly deployed and configured
"""

import os
import json
from pathlib import Path
from datetime import datetime

def verify_github_actions_deployment():
    """Verify GitHub Actions deployment"""
    print("🔍 GitHub Actions Deployment Verification")
    print("=" * 50)
    
    verification_results = {
        "timestamp": datetime.now().isoformat(),
        "workflows_found": [],
        "missing_workflows": [],
        "configuration_issues": [],
        "deployment_status": "unknown"
    }
    
    # Expected workflows
    expected_workflows = [
        "unified-development-acceleration.yml",
        "continuous-integration.yml", 
        "github-pull-assessment.yml",
        "claude-task-execution.yml",
        "scale-runners.yml",
        "results-aggregation.yml"
    ]
    
    workflows_dir = Path(".github/workflows")
    
    if not workflows_dir.exists():
        print("❌ .github/workflows directory not found")
        verification_results["deployment_status"] = "failed"
        return verification_results
    
    # Check for workflow files
    found_workflows = list(workflows_dir.glob("*.yml"))
    found_workflow_names = [f.name for f in found_workflows]
    
    print(f"📁 Found {len(found_workflows)} workflow files:")
    for workflow in found_workflows:
        print(f"  ✅ {workflow.name}")
        verification_results["workflows_found"].append(workflow.name)
    
    # Check for missing expected workflows
    for expected in expected_workflows:
        if expected not in found_workflow_names:
            print(f"  ⚠️ Missing: {expected}")
            verification_results["missing_workflows"].append(expected)
    
    # Verify workflow contents
    print("\n🔍 Workflow Configuration Check:")
    for workflow_file in found_workflows:
        try:
            with open(workflow_file, 'r') as f:
                content = f.read()
            
            # Basic validation
            issues = []
            if "name:" not in content:
                issues.append("Missing workflow name")
            if "on:" not in content:
                issues.append("Missing trigger configuration")
            if "jobs:" not in content:
                issues.append("Missing jobs section")
            
            if issues:
                print(f"  ⚠️ {workflow_file.name}: {', '.join(issues)}")
                verification_results["configuration_issues"].append({
                    "workflow": workflow_file.name,
                    "issues": issues
                })
            else:
                print(f"  ✅ {workflow_file.name}: Configuration looks good")
                
        except Exception as e:
            print(f"  ❌ {workflow_file.name}: Error reading file - {e}")
            verification_results["configuration_issues"].append({
                "workflow": workflow_file.name,
                "issues": [f"Read error: {e}"]
            })
    
    # Check for supporting scripts
    print("\n🔧 Supporting Scripts Check:")
    scripts_dir = Path(".github/scripts")
    expected_scripts = [
        "task-distributor.js",
        "workflow-monitor.py",
        "deployment-verification.py"
    ]
    
    if scripts_dir.exists():
        for script in expected_scripts:
            script_path = scripts_dir / script
            if script_path.exists():
                print(f"  ✅ {script}")
            else:
                print(f"  ⚠️ Missing: {script}")
    else:
        print("  ⚠️ .github/scripts directory not found")
    
    # Check requirements
    print("\n📦 Dependencies Check:")
    requirements_files = [
        "requirements.txt",
        ".github/workflows/requirements.txt"
    ]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            print(f"  ✅ {req_file}")
        else:
            print(f"  ⚠️ Missing: {req_file}")
    
    # Overall assessment
    print("\n📊 Deployment Assessment:")
    total_expected = len(expected_workflows)
    found_count = len([w for w in expected_workflows if w in found_workflow_names])
    config_issues = len(verification_results["configuration_issues"])
    
    success_rate = (found_count / total_expected) * 100
    
    print(f"  Workflows Found: {found_count}/{total_expected} ({success_rate:.1f}%)")
    print(f"  Configuration Issues: {config_issues}")
    
    if success_rate >= 90 and config_issues == 0:
        verification_results["deployment_status"] = "excellent"
        print("  🎉 Status: EXCELLENT - Full deployment successful!")
    elif success_rate >= 75 and config_issues <= 2:
        verification_results["deployment_status"] = "good"
        print("  ✅ Status: GOOD - Deployment mostly successful")
    elif success_rate >= 50:
        verification_results["deployment_status"] = "partial"
        print("  ⚠️ Status: PARTIAL - Some components missing")
    else:
        verification_results["deployment_status"] = "failed"
        print("  ❌ Status: FAILED - Major deployment issues")
    
    # Automation capabilities assessment
    print("\n🚀 Automation Capabilities:")
    automation_features = [
        ("Unified Development Acceleration", "unified-development-acceleration.yml" in found_workflow_names),
        ("Continuous Integration", "continuous-integration.yml" in found_workflow_names),
        ("GitHub Pull Assessment", "github-pull-assessment.yml" in found_workflow_names),
        ("Task Distribution", scripts_dir.exists() and (scripts_dir / "task-distributor.js").exists()),
        ("Workflow Monitoring", scripts_dir.exists() and (scripts_dir / "workflow-monitor.py").exists())
    ]
    
    for feature, available in automation_features:
        status = "✅ Available" if available else "❌ Missing"
        print(f"  {feature}: {status}")
    
    return verification_results

def main():
    """Main verification execution"""
    try:
        results = verify_github_actions_deployment()
        
        # Save results
        with open('.github/deployment-verification.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Verification results saved to: .github/deployment-verification.json")
        
        # Exit with appropriate code
        if results["deployment_status"] in ["excellent", "good"]:
            print("\n🎉 GitHub Actions deployment verification PASSED!")
            return 0
        else:
            print("\n⚠️ GitHub Actions deployment verification found issues")
            return 1
            
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())