#!/usr/bin/env python3
"""
GitHub Actions Workflow Monitor
Real-time monitoring and reporting for unified development acceleration
"""

import requests
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class GitHubActionsMonitor:
    """Monitor GitHub Actions workflows and generate performance reports"""
    
    def __init__(self, repo_owner: str, repo_name: str, token: Optional[str] = None):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {self.token}" if self.token else ""
        }
    
    def get_workflow_runs(self, workflow_name: str = None, per_page: int = 50) -> List[Dict]:
        """Get recent workflow runs"""
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
        params = {"per_page": per_page}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            runs = response.json().get("workflow_runs", [])
            
            # Filter by workflow name if specified
            if workflow_name:
                runs = [run for run in runs if workflow_name.lower() in run.get("name", "").lower()]
            
            return runs
        except requests.RequestException as e:
            print(f"❌ Error fetching workflow runs: {e}")
            return []
    
    def get_workflow_performance(self, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze workflow performance over specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        runs = self.get_workflow_runs(per_page=100)
        
        # Filter recent runs
        recent_runs = []
        for run in runs:
            run_time = datetime.fromisoformat(run.get("created_at", "").replace("Z", "+00:00"))
            if run_time.replace(tzinfo=None) > cutoff_time:
                recent_runs.append(run)
        
        # Analyze performance
        performance = {
            "analysis_period": f"Last {hours_back} hours",
            "total_runs": len(recent_runs),
            "successful_runs": len([r for r in recent_runs if r.get("conclusion") == "success"]),
            "failed_runs": len([r for r in recent_runs if r.get("conclusion") == "failure"]),
            "cancelled_runs": len([r for r in recent_runs if r.get("conclusion") == "cancelled"]),
            "in_progress_runs": len([r for r in recent_runs if r.get("status") == "in_progress"]),
            "workflows": {}
        }
        
        # Group by workflow
        for run in recent_runs:
            workflow_name = run.get("name", "Unknown")
            if workflow_name not in performance["workflows"]:
                performance["workflows"][workflow_name] = {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                    "avg_duration_minutes": 0,
                    "last_run": None
                }
            
            perf = performance["workflows"][workflow_name]
            perf["total_runs"] += 1
            
            if run.get("conclusion") == "success":
                perf["successful_runs"] += 1
            elif run.get("conclusion") == "failure":
                perf["failed_runs"] += 1
            
            # Update last run time
            if not perf["last_run"] or run.get("created_at", "") > perf["last_run"]:
                perf["last_run"] = run.get("created_at")
        
        # Calculate success rates
        performance["overall_success_rate"] = (
            performance["successful_runs"] / performance["total_runs"] * 100
            if performance["total_runs"] > 0 else 0
        )
        
        for workflow_name, workflow_perf in performance["workflows"].items():
            workflow_perf["success_rate"] = (
                workflow_perf["successful_runs"] / workflow_perf["total_runs"] * 100
                if workflow_perf["total_runs"] > 0 else 0
            )
        
        return performance
    
    def get_active_runners_status(self) -> Dict[str, Any]:
        """Get status of active GitHub Actions runners"""
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
        params = {"status": "in_progress"}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            active_runs = response.json().get("workflow_runs", [])
            
            return {
                "active_runs_count": len(active_runs),
                "active_workflows": [
                    {
                        "name": run.get("name"),
                        "run_id": run.get("id"),
                        "started_at": run.get("created_at"),
                        "html_url": run.get("html_url")
                    }
                    for run in active_runs[:10]  # Limit to 10 most recent
                ]
            }
        except requests.RequestException as e:
            print(f"❌ Error fetching active runners: {e}")
            return {"active_runs_count": 0, "active_workflows": []}
    
    def generate_monitoring_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        print(f"📊 Generating monitoring report for last {hours_back} hours...")
        
        performance = self.get_workflow_performance(hours_back)
        active_status = self.get_active_runners_status()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "repository": f"{self.repo_owner}/{self.repo_name}",
            "monitoring_period_hours": hours_back,
            "performance_metrics": performance,
            "active_runners": active_status,
            "system_health": {
                "overall_status": "healthy" if performance["overall_success_rate"] >= 80 else "warning" if performance["overall_success_rate"] >= 60 else "critical",
                "success_rate": performance["overall_success_rate"],
                "automation_activity": "high" if performance["total_runs"] >= 10 else "medium" if performance["total_runs"] >= 5 else "low"
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if performance["overall_success_rate"] < 70:
            report["recommendations"].append("Investigate workflow failures and optimize configurations")
        
        if performance["failed_runs"] > performance["successful_runs"]:
            report["recommendations"].append("High failure rate detected - review error logs and dependencies")
        
        if active_status["active_runs_count"] > 10:
            report["recommendations"].append("High runner utilization - consider optimizing task distribution")
        
        if performance["total_runs"] == 0:
            report["recommendations"].append("No recent workflow activity - verify automation triggers")
        
        return report
    
    def print_monitoring_summary(self, report: Dict[str, Any]):
        """Print formatted monitoring summary"""
        print("\n" + "="*60)
        print("🚀 GITHUB ACTIONS MONITORING DASHBOARD")
        print("="*60)
        
        print(f"📊 Repository: {report['repository']}")
        print(f"⏰ Period: Last {report['monitoring_period_hours']} hours")
        print(f"🕐 Generated: {report['timestamp']}")
        
        print("\n📈 PERFORMANCE METRICS")
        print("-"*30)
        perf = report["performance_metrics"]
        print(f"Total Runs: {perf['total_runs']}")
        print(f"Successful: {perf['successful_runs']} ({perf['overall_success_rate']:.1f}%)")
        print(f"Failed: {perf['failed_runs']}")
        print(f"In Progress: {perf['in_progress_runs']}")
        
        print("\n🔧 WORKFLOW BREAKDOWN")
        print("-"*30)
        for workflow, stats in perf["workflows"].items():
            print(f"• {workflow}:")
            print(f"  Runs: {stats['total_runs']}, Success: {stats['success_rate']:.1f}%")
        
        print("\n⚡ ACTIVE RUNNERS")
        print("-"*30)
        active = report["active_runners"]
        print(f"Active Workflows: {active['active_runs_count']}")
        for workflow in active["active_workflows"][:5]:
            print(f"• {workflow['name']} (ID: {workflow['run_id']})")
        
        print("\n🏥 SYSTEM HEALTH")
        print("-"*30)
        health = report["system_health"]
        status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}
        print(f"Status: {status_emoji.get(health['overall_status'], '❓')} {health['overall_status'].upper()}")
        print(f"Success Rate: {health['success_rate']:.1f}%")
        print(f"Activity Level: {health['automation_activity'].upper()}")
        
        if report["recommendations"]:
            print("\n💡 RECOMMENDATIONS")
            print("-"*30)
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"{i}. {rec}")
        
        print("\n" + "="*60)
    
    def save_monitoring_report(self, report: Dict[str, Any], filepath: str = None):
        """Save monitoring report to file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"monitoring_report_{timestamp}.json"
        
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"💾 Monitoring report saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")

def main():
    """Main monitoring execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Actions Workflow Monitor")
    parser.add_argument("--repo", default="garmir/0xANATHEMA", help="Repository (owner/name)")
    parser.add_argument("--hours", type=int, default=24, help="Hours to analyze (default: 24)")
    parser.add_argument("--output", help="Output file path for report")
    parser.add_argument("--continuous", action="store_true", help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=300, help="Continuous monitoring interval in seconds")
    
    args = parser.parse_args()
    
    # Parse repository
    if "/" in args.repo:
        repo_owner, repo_name = args.repo.split("/", 1)
    else:
        print("❌ Repository must be in format 'owner/name'")
        return
    
    # Create monitor
    monitor = GitHubActionsMonitor(repo_owner, repo_name)
    
    if args.continuous:
        print(f"🔄 Starting continuous monitoring every {args.interval} seconds...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                report = monitor.generate_monitoring_report(args.hours)
                monitor.print_monitoring_summary(report)
                
                if args.output:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"{args.output}_{timestamp}.json"
                    monitor.save_monitoring_report(report, output_file)
                
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
    else:
        # Single run
        report = monitor.generate_monitoring_report(args.hours)
        monitor.print_monitoring_summary(report)
        
        if args.output:
            monitor.save_monitoring_report(report, args.output)

if __name__ == "__main__":
    main()