#!/usr/bin/env python3
"""
Performance Analyzer for Autonomous Development System
Identifies bottlenecks and optimization opportunities
"""

import time
import psutil
import gc
import os
import sys
import threading
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import subprocess
from pathlib import Path

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage_percent: float
    disk_io_read: int
    disk_io_write: int
    network_sent: int
    network_recv: int
    process_count: int
    thread_count: int
    file_descriptors: int

@dataclass
class ProcessMetrics:
    """Process-specific metrics"""
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    memory_vms: int
    num_threads: int
    num_fds: int
    create_time: float

class PerformanceAnalyzer:
    """Main performance analysis engine"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: List[SystemMetrics] = []
        self.process_metrics: Dict[int, List[ProcessMetrics]] = {}
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Start background performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("🛑 Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 measurements to prevent memory growth
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network
        network = psutil.net_io_counters()
        
        # Process info
        processes = list(psutil.process_iter(['pid']))
        process_count = len(processes)
        
        # Thread count (approximate)
        thread_count = 0
        for proc in psutil.process_iter(['pid']):
            try:
                thread_count += proc.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # File descriptors (on Unix systems)
        fd_count = 0
        try:
            if hasattr(os, 'listdir') and os.path.exists('/proc/self/fd'):
                fd_count = len(os.listdir('/proc/self/fd'))
        except Exception:
            pass
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available,
            disk_usage_percent=(disk.used / disk.total) * 100,
            disk_io_read=disk_io.read_bytes if disk_io else 0,
            disk_io_write=disk_io.write_bytes if disk_io else 0,
            network_sent=network.bytes_sent if network else 0,
            network_recv=network.bytes_recv if network else 0,
            process_count=process_count,
            thread_count=thread_count,
            file_descriptors=fd_count
        )
    
    def collect_process_metrics(self, target_processes: Optional[List[str]] = None) -> Dict[str, ProcessMetrics]:
        """Collect metrics for specific processes"""
        target_processes = target_processes or ['python', 'node', 'task-master']
        process_metrics = {}
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                proc_info = proc.info
                if any(target in proc_info['name'].lower() for target in target_processes):
                    memory_info = proc.memory_info()
                    
                    metrics = ProcessMetrics(
                        pid=proc_info['pid'],
                        name=proc_info['name'],
                        cpu_percent=proc_info['cpu_percent'] or 0,
                        memory_percent=proc_info['memory_percent'] or 0,
                        memory_rss=memory_info.rss,
                        memory_vms=memory_info.vms,
                        num_threads=proc.num_threads(),
                        num_fds=proc.num_fds() if hasattr(proc, 'num_fds') else 0,
                        create_time=proc.create_time()
                    )
                    
                    process_metrics[f"{proc_info['name']}_{proc_info['pid']}"] = metrics
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        return process_metrics
    
    def analyze_current_performance(self) -> Dict[str, Any]:
        """Analyze current system performance"""
        if not self.metrics_history:
            return {"error": "No metrics collected yet"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics)
        
        # Identify bottlenecks
        bottlenecks = []
        recommendations = []
        
        if avg_cpu > 80:
            bottlenecks.append("High CPU usage")
            recommendations.append("Consider optimizing CPU-intensive operations")
            recommendations.append("Implement parallel processing for heavy tasks")
        
        if avg_memory > 80:
            bottlenecks.append("High memory usage")
            recommendations.append("Optimize memory usage and implement garbage collection")
            recommendations.append("Consider using memory-mapped files for large datasets")
        
        if avg_disk > 90:
            bottlenecks.append("High disk usage")
            recommendations.append("Clean up temporary files and implement disk space monitoring")
        
        # Thread count analysis
        recent_thread_count = recent_metrics[-1].thread_count
        if recent_thread_count > 500:
            bottlenecks.append("High thread count")
            recommendations.append("Optimize thread pool usage and reduce thread creation")
        
        return {
            "current_performance": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "disk_percent": avg_disk,
                "thread_count": recent_thread_count
            },
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "metrics_count": len(self.metrics_history)
        }
    
    def analyze_workflow_files(self) -> Dict[str, Any]:
        """Analyze workflow files for performance issues"""
        analysis = {
            "large_files": [],
            "potential_bottlenecks": [],
            "optimization_opportunities": []
        }
        
        # Find large Python files that might need optimization
        for file_path in Path('.').rglob('*.py'):
            try:
                size = file_path.stat().st_size
                if size > 50 * 1024:  # Files larger than 50KB
                    analysis["large_files"].append({
                        "file": str(file_path),
                        "size_kb": size // 1024,
                        "lines": self._count_lines(file_path)
                    })
            except Exception:
                continue
        
        # Look for potential performance issues in code
        performance_patterns = [
            ("time.sleep", "Blocking sleep calls"),
            ("subprocess.run", "Synchronous subprocess calls"),
            ("requests.get", "Synchronous HTTP requests"),
            ("open(", "File operations without context"),
            ("while True:", "Infinite loops"),
            ("for.*in.*range(", "Potentially inefficient loops")
        ]
        
        for file_path in Path('.').rglob('*.py'):
            if file_path.name.startswith('.') or 'venv' in str(file_path):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                for pattern, description in performance_patterns:
                    if pattern in content:
                        analysis["potential_bottlenecks"].append({
                            "file": str(file_path),
                            "pattern": pattern,
                            "description": description
                        })
            except Exception:
                continue
        
        # General optimization opportunities
        analysis["optimization_opportunities"] = [
            "Implement async/await for I/O operations",
            "Use connection pooling for external services", 
            "Add caching for expensive computations",
            "Optimize import statements",
            "Use generators for large datasets",
            "Implement batch processing",
            "Add performance monitoring decorators"
        ]
        
        return analysis
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_analysis = self.analyze_current_performance()
        file_analysis = self.analyze_workflow_files()
        process_metrics = self.collect_process_metrics()
        
        # System information
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        # Performance recommendations based on analysis
        recommendations = current_analysis.get("recommendations", [])
        recommendations.extend([
            "Replace synchronous operations with async alternatives",
            "Implement intelligent caching strategies",
            "Use thread/process pools for concurrent execution",
            "Optimize memory usage with generators and context managers",
            "Add performance monitoring and profiling",
            "Implement batch processing for similar operations"
        ])
        
        return {
            "timestamp": time.time(),
            "system_info": system_info,
            "current_performance": current_analysis,
            "file_analysis": file_analysis,
            "process_metrics": {k: asdict(v) for k, v in process_metrics.items()},
            "recommendations": list(set(recommendations)),  # Remove duplicates
            "monitoring_duration": len(self.metrics_history) * self.monitoring_interval
        }
    
    def save_report(self, filename: str = "performance_analysis.json"):
        """Save performance report to file"""
        report = self.generate_performance_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Performance report saved to {filename}")
        return report

class WorkflowPerformanceAnalyzer:
    """Specialized analyzer for autonomous workflow performance"""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
    
    def analyze_task_execution_time(self, tasks_file: str = ".taskmaster/tasks/tasks.json") -> Dict[str, Any]:
        """Analyze task execution patterns"""
        analysis = {
            "task_count": 0,
            "estimated_execution_time": 0,
            "complexity_distribution": {},
            "optimization_potential": []
        }
        
        try:
            if os.path.exists(tasks_file):
                with open(tasks_file, 'r') as f:
                    data = json.load(f)
                
                tasks = data.get('tags', {}).get('master', {}).get('tasks', [])
                analysis["task_count"] = len(tasks)
                
                # Analyze complexity
                complexity_counts = {}
                for task in tasks:
                    # Simple complexity estimation based on description length
                    desc_length = len(task.get('description', '')) + len(task.get('details', ''))
                    
                    if desc_length < 100:
                        complexity = "low"
                    elif desc_length < 300:
                        complexity = "medium"
                    else:
                        complexity = "high"
                    
                    complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
                
                analysis["complexity_distribution"] = complexity_counts
                
                # Estimate execution time (rough)
                total_time = 0
                for task in tasks:
                    priority = task.get('priority', 'medium')
                    desc_length = len(task.get('description', '')) + len(task.get('details', ''))
                    
                    # Simple time estimation
                    base_time = {"low": 5, "medium": 15, "high": 30}.get(priority, 15)
                    complexity_multiplier = max(1, desc_length / 100)
                    total_time += base_time * complexity_multiplier
                
                analysis["estimated_execution_time"] = total_time
                
                # Optimization potential
                if complexity_counts.get("high", 0) > 5:
                    analysis["optimization_potential"].append("High number of complex tasks - consider parallel execution")
                
                if analysis["task_count"] > 50:
                    analysis["optimization_potential"].append("Large number of tasks - implement batch processing")
        
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def analyze_github_actions_performance(self) -> Dict[str, Any]:
        """Analyze GitHub Actions workflow performance"""
        analysis = {
            "workflow_files": [],
            "optimization_opportunities": [],
            "estimated_execution_time": 0
        }
        
        workflow_dir = Path(".github/workflows")
        if workflow_dir.exists():
            for workflow_file in workflow_dir.glob("*.yml"):
                try:
                    content = workflow_file.read_text()
                    
                    # Simple analysis
                    job_count = content.count("jobs:")
                    step_count = content.count("- name:")
                    runner_count = content.count("runs-on:")
                    
                    file_analysis = {
                        "file": str(workflow_file),
                        "jobs": job_count,
                        "steps": step_count,
                        "runners": runner_count,
                        "parallel_potential": job_count > 1
                    }
                    
                    analysis["workflow_files"].append(file_analysis)
                    
                    # Estimate execution time (rough)
                    estimated_time = step_count * 2  # 2 minutes per step average
                    if job_count > 1:
                        estimated_time = estimated_time / job_count  # Parallel execution
                    
                    analysis["estimated_execution_time"] += estimated_time
                    
                except Exception:
                    continue
            
            # Optimization opportunities
            analysis["optimization_opportunities"] = [
                "Implement caching for dependencies",
                "Use matrix builds for parallel execution",
                "Optimize Docker images if used",
                "Implement conditional execution",
                "Use artifacts to share data between jobs"
            ]
        
        return analysis

def run_comprehensive_analysis():
    """Run comprehensive performance analysis"""
    print("🚀 Starting Comprehensive Performance Analysis")
    print("=" * 60)
    
    # Initialize analyzers
    analyzer = PerformanceAnalyzer(monitoring_interval=0.5)
    workflow_analyzer = WorkflowPerformanceAnalyzer()
    
    # Start monitoring
    analyzer.start_monitoring()
    
    try:
        # Let it collect some data
        print("📊 Collecting performance data...")
        time.sleep(5)
        
        # Generate reports
        print("\n📈 Analyzing current performance...")
        performance_report = analyzer.generate_performance_report()
        
        print("\n🔍 Analyzing workflow performance...")
        task_analysis = workflow_analyzer.analyze_task_execution_time()
        
        print("\n⚙️ Analyzing GitHub Actions performance...")
        actions_analysis = workflow_analyzer.analyze_github_actions_performance()
        
        # Combined report
        comprehensive_report = {
            "system_performance": performance_report,
            "task_analysis": task_analysis,
            "github_actions": actions_analysis,
            "summary": {
                "total_optimization_opportunities": (
                    len(performance_report.get("recommendations", [])) +
                    len(task_analysis.get("optimization_potential", [])) +
                    len(actions_analysis.get("optimization_opportunities", []))
                ),
                "critical_issues": [],
                "quick_wins": []
            }
        }
        
        # Identify critical issues and quick wins
        current_perf = performance_report.get("current_performance", {})
        
        if current_perf.get("cpu_percent", 0) > 80:
            comprehensive_report["summary"]["critical_issues"].append("High CPU usage")
        
        if current_perf.get("memory_percent", 0) > 80:
            comprehensive_report["summary"]["critical_issues"].append("High memory usage")
        
        # Quick wins
        comprehensive_report["summary"]["quick_wins"] = [
            "Replace time.sleep with async alternatives",
            "Implement caching for repeated computations",
            "Use connection pooling",
            "Optimize garbage collection settings",
            "Add performance monitoring decorators"
        ]
        
        # Save comprehensive report
        with open("comprehensive_performance_analysis.json", "w") as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        print(f"\n✅ Analysis complete! Report saved to comprehensive_performance_analysis.json")
        
        # Print summary
        print(f"\n📊 PERFORMANCE ANALYSIS SUMMARY")
        print(f"=" * 40)
        print(f"CPU Usage: {current_perf.get('cpu_percent', 0):.1f}%")
        print(f"Memory Usage: {current_perf.get('memory_percent', 0):.1f}%")
        print(f"Tasks Analyzed: {task_analysis.get('task_count', 0)}")
        print(f"GitHub Actions Files: {len(actions_analysis.get('workflow_files', []))}")
        print(f"Total Optimization Opportunities: {comprehensive_report['summary']['total_optimization_opportunities']}")
        
        if comprehensive_report["summary"]["critical_issues"]:
            print(f"\n🚨 Critical Issues:")
            for issue in comprehensive_report["summary"]["critical_issues"]:
                print(f"  - {issue}")
        
        print(f"\n💡 Quick Wins:")
        for win in comprehensive_report["summary"]["quick_wins"][:5]:
            print(f"  - {win}")
        
        return comprehensive_report
        
    finally:
        analyzer.stop_monitoring()

if __name__ == "__main__":
    run_comprehensive_analysis()