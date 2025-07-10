#!/usr/bin/env python3
"""
Simple Performance Analyzer - No external dependencies
Analyzes code structure and provides optimization recommendations
"""

import os
import sys
import time
import gc
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import re

class SimplePerformanceAnalyzer:
    """Lightweight performance analyzer"""
    
    def __init__(self):
        self.start_time = time.time()
        self.analysis_results = {}
    
    def analyze_file_structure(self) -> Dict[str, Any]:
        """Analyze project file structure for performance insights"""
        analysis = {
            "python_files": [],
            "large_files": [],
            "total_lines": 0,
            "file_count": 0
        }
        
        for file_path in Path('.').rglob('*.py'):
            if 'venv' in str(file_path) or '.git' in str(file_path):
                continue
                
            try:
                size = file_path.stat().st_size
                lines = self._count_lines(file_path)
                
                file_info = {
                    "path": str(file_path),
                    "size_bytes": size,
                    "lines": lines,
                    "is_large": size > 50000 or lines > 500
                }
                
                analysis["python_files"].append(file_info)
                analysis["total_lines"] += lines
                analysis["file_count"] += 1
                
                if file_info["is_large"]:
                    analysis["large_files"].append(file_info)
                    
            except Exception as e:
                continue
        
        return analysis
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    def analyze_code_patterns(self) -> Dict[str, Any]:
        """Analyze code for performance anti-patterns"""
        patterns = {
            "blocking_calls": {
                "patterns": [r'time\.sleep\(', r'input\(', r'requests\.get\(', r'subprocess\.run\('],
                "description": "Blocking operations that could be async",
                "files": []
            },
            "inefficient_loops": {
                "patterns": [r'while\s+True:', r'for.*in.*range\(\d+\)'],
                "description": "Potentially inefficient loop patterns",
                "files": []
            },
            "memory_intensive": {
                "patterns": [r'\.read\(\)', r'\.readlines\(\)', r'json\.load\('],
                "description": "Operations that load large amounts into memory",
                "files": []
            },
            "thread_usage": {
                "patterns": [r'threading\.Thread\(', r'Thread\(', r'ThreadPoolExecutor'],
                "description": "Threading usage (check for optimization)",
                "files": []
            },
            "process_usage": {
                "patterns": [r'multiprocessing\.', r'ProcessPoolExecutor', r'subprocess\.'],
                "description": "Process/subprocess usage",
                "files": []
            }
        }
        
        for file_path in Path('.').rglob('*.py'):
            if 'venv' in str(file_path) or '.git' in str(file_path):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                for pattern_type, pattern_info in patterns.items():
                    for pattern in pattern_info["patterns"]:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            patterns[pattern_type]["files"].append({
                                "file": str(file_path),
                                "matches": len(matches),
                                "examples": matches[:3]  # First 3 matches
                            })
                            
            except Exception:
                continue
        
        return patterns
    
    def analyze_imports(self) -> Dict[str, Any]:
        """Analyze import patterns for optimization opportunities"""
        analysis = {
            "heavy_imports": [],
            "unused_imports": [],
            "import_patterns": {},
            "recommendations": []
        }
        
        heavy_libraries = [
            'numpy', 'pandas', 'matplotlib', 'tensorflow', 'torch',
            'scipy', 'sklearn', 'opencv', 'PIL', 'cv2'
        ]
        
        for file_path in Path('.').rglob('*.py'):
            if 'venv' in str(file_path) or '.git' in str(file_path):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        for lib in heavy_libraries:
                            if lib in line:
                                analysis["heavy_imports"].append({
                                    "file": str(file_path),
                                    "line": line_num,
                                    "import": line,
                                    "library": lib
                                })
                        
                        # Count import patterns
                        if 'import *' in line:
                            pattern = 'wildcard_import'
                        elif line.startswith('from '):
                            pattern = 'from_import'
                        else:
                            pattern = 'direct_import'
                        
                        analysis["import_patterns"][pattern] = analysis["import_patterns"].get(pattern, 0) + 1
                        
            except Exception:
                continue
        
        # Generate recommendations
        if analysis["import_patterns"].get("wildcard_import", 0) > 0:
            analysis["recommendations"].append("Avoid wildcard imports (import *) - import specific functions instead")
        
        if analysis["heavy_imports"]:
            analysis["recommendations"].append("Consider lazy loading for heavy libraries")
            analysis["recommendations"].append("Move heavy imports inside functions if not used globally")
        
        return analysis
    
    def analyze_task_master_performance(self) -> Dict[str, Any]:
        """Analyze Task Master specific performance aspects"""
        analysis = {
            "task_files_found": False,
            "task_count": 0,
            "large_tasks": [],
            "complexity_distribution": {},
            "recommendations": []
        }
        
        tasks_file = Path(".taskmaster/tasks/tasks.json")
        if tasks_file.exists():
            try:
                with open(tasks_file, 'r') as f:
                    data = json.load(f)
                
                analysis["task_files_found"] = True
                tasks = data.get('tags', {}).get('master', {}).get('tasks', [])
                analysis["task_count"] = len(tasks)
                
                # Analyze task complexity
                complexity_counts = {"low": 0, "medium": 0, "high": 0}
                
                for task in tasks:
                    desc_length = len(task.get('description', '')) + len(task.get('details', ''))
                    dependencies = len(task.get('dependencies', []))
                    
                    # Simple complexity scoring
                    complexity_score = desc_length / 100 + dependencies * 0.5
                    
                    if complexity_score < 1:
                        complexity = "low"
                    elif complexity_score < 3:
                        complexity = "medium"
                    else:
                        complexity = "high"
                        analysis["large_tasks"].append({
                            "id": task.get('id'),
                            "title": task.get('title', '')[:50],
                            "complexity_score": complexity_score
                        })
                    
                    complexity_counts[complexity] += 1
                
                analysis["complexity_distribution"] = complexity_counts
                
                # Recommendations
                if analysis["task_count"] > 100:
                    analysis["recommendations"].append("Large number of tasks - consider batch processing")
                
                if len(analysis["large_tasks"]) > 10:
                    analysis["recommendations"].append("Many complex tasks - implement parallel execution")
                
                if complexity_counts["high"] > complexity_counts["low"]:
                    analysis["recommendations"].append("High complexity tasks dominate - break down into smaller subtasks")
                    
            except Exception as e:
                analysis["error"] = str(e)
        
        return analysis
    
    def analyze_github_actions(self) -> Dict[str, Any]:
        """Analyze GitHub Actions for performance optimization"""
        analysis = {
            "workflow_files": [],
            "total_jobs": 0,
            "total_steps": 0,
            "optimization_opportunities": []
        }
        
        workflow_dir = Path(".github/workflows")
        if workflow_dir.exists():
            for workflow_file in workflow_dir.glob("*.yml"):
                try:
                    content = workflow_file.read_text(encoding='utf-8')
                    
                    # Count jobs and steps
                    job_count = len(re.findall(r'^\s*[\w-]+:', content, re.MULTILINE)) - 1  # Subtract workflow name
                    step_count = len(re.findall(r'- name:', content))
                    
                    # Look for optimization opportunities
                    has_caching = 'cache:' in content or 'actions/cache' in content
                    has_matrix = 'matrix:' in content
                    has_parallel = job_count > 1
                    
                    file_analysis = {
                        "file": str(workflow_file),
                        "jobs": job_count,
                        "steps": step_count,
                        "has_caching": has_caching,
                        "has_matrix": has_matrix,
                        "has_parallel": has_parallel,
                        "estimated_runtime_minutes": step_count * 2  # Rough estimate
                    }
                    
                    analysis["workflow_files"].append(file_analysis)
                    analysis["total_jobs"] += job_count
                    analysis["total_steps"] += step_count
                    
                    # Optimization opportunities
                    if not has_caching:
                        analysis["optimization_opportunities"].append(f"{workflow_file.name}: Add dependency caching")
                    
                    if step_count > 10 and not has_matrix:
                        analysis["optimization_opportunities"].append(f"{workflow_file.name}: Consider matrix builds")
                    
                    if job_count == 1 and step_count > 5:
                        analysis["optimization_opportunities"].append(f"{workflow_file.name}: Consider splitting into parallel jobs")
                        
                except Exception:
                    continue
        
        return analysis
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
            "current_directory": os.getcwd(),
            "environment_variables": {
                "PATH": os.environ.get("PATH", "")[:200] + "...",  # Truncated
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "HOME": os.environ.get("HOME", ""),
            }
        }
        
        # Try to get memory info
        try:
            if sys.platform == 'darwin':  # macOS
                result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info["memory_info"] = "Available (vm_stat output truncated)"
            elif sys.platform.startswith('linux'):
                result = subprocess.run(['free', '-h'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info["memory_info"] = result.stdout.split('\n')[1]  # Memory line
        except Exception:
            info["memory_info"] = "Unable to determine"
        
        return info
    
    def generate_optimization_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive optimization recommendations"""
        recommendations = []
        
        # File structure recommendations
        file_analysis = analysis_results.get("file_structure", {})
        if len(file_analysis.get("large_files", [])) > 5:
            recommendations.append("🔧 Consider splitting large files into smaller modules")
        
        # Code pattern recommendations
        patterns = analysis_results.get("code_patterns", {})
        
        if patterns.get("blocking_calls", {}).get("files"):
            recommendations.append("⚡ Replace blocking calls with async alternatives (time.sleep, requests)")
        
        if patterns.get("inefficient_loops", {}).get("files"):
            recommendations.append("🔄 Optimize loop patterns - consider generators and vectorization")
        
        if patterns.get("memory_intensive", {}).get("files"):
            recommendations.append("💾 Implement streaming/chunked processing for large data operations")
        
        # Import recommendations
        imports = analysis_results.get("imports", {})
        if imports.get("heavy_imports"):
            recommendations.append("📦 Implement lazy loading for heavy libraries")
        
        # Task Master recommendations
        task_analysis = analysis_results.get("task_master", {})
        recommendations.extend(task_analysis.get("recommendations", []))
        
        # GitHub Actions recommendations
        actions_analysis = analysis_results.get("github_actions", {})
        if actions_analysis.get("optimization_opportunities"):
            recommendations.append("🚀 Optimize GitHub Actions workflows with caching and parallelization")
        
        # General recommendations
        recommendations.extend([
            "🎯 Implement performance monitoring decorators",
            "🗄️ Add intelligent caching for expensive operations",
            "🧵 Use thread/process pools for concurrent execution",
            "📊 Add performance profiling to identify bottlenecks",
            "🔍 Implement connection pooling for external services",
            "⚡ Use batch processing for similar operations",
            "🎛️ Optimize garbage collection settings",
            "📈 Add real-time performance monitoring"
        ])
        
        return recommendations
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run complete performance analysis"""
        print("🚀 Starting Simple Performance Analysis")
        print("=" * 50)
        
        analysis_results = {}
        
        print("📁 Analyzing file structure...")
        analysis_results["file_structure"] = self.analyze_file_structure()
        
        print("🔍 Analyzing code patterns...")
        analysis_results["code_patterns"] = self.analyze_code_patterns()
        
        print("📦 Analyzing imports...")
        analysis_results["imports"] = self.analyze_imports()
        
        print("📋 Analyzing Task Master performance...")
        analysis_results["task_master"] = self.analyze_task_master_performance()
        
        print("⚙️ Analyzing GitHub Actions...")
        analysis_results["github_actions"] = self.analyze_github_actions()
        
        print("💻 Getting system info...")
        analysis_results["system_info"] = self.get_system_info()
        
        print("💡 Generating recommendations...")
        analysis_results["recommendations"] = self.generate_optimization_recommendations(analysis_results)
        
        # Add analysis metadata
        analysis_results["metadata"] = {
            "analysis_time": time.time() - self.start_time,
            "timestamp": time.time(),
            "analyzer_version": "1.0.0"
        }
        
        return analysis_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print analysis summary"""
        print(f"\n📊 PERFORMANCE ANALYSIS SUMMARY")
        print(f"=" * 40)
        
        # File statistics
        file_stats = results.get("file_structure", {})
        print(f"Python Files: {file_stats.get('file_count', 0)}")
        print(f"Total Lines: {file_stats.get('total_lines', 0):,}")
        print(f"Large Files: {len(file_stats.get('large_files', []))}")
        
        # Task Master statistics
        task_stats = results.get("task_master", {})
        if task_stats.get("task_files_found"):
            print(f"Tasks: {task_stats.get('task_count', 0)}")
            complexity = task_stats.get("complexity_distribution", {})
            print(f"High Complexity Tasks: {complexity.get('high', 0)}")
        
        # GitHub Actions statistics
        actions_stats = results.get("github_actions", {})
        print(f"Workflow Files: {len(actions_stats.get('workflow_files', []))}")
        print(f"Total Jobs: {actions_stats.get('total_jobs', 0)}")
        print(f"Total Steps: {actions_stats.get('total_steps', 0)}")
        
        # Top recommendations
        recommendations = results.get("recommendations", [])
        print(f"\n💡 TOP OPTIMIZATION OPPORTUNITIES:")
        for i, rec in enumerate(recommendations[:8], 1):
            print(f"  {i}. {rec}")
        
        # Performance patterns found
        patterns = results.get("code_patterns", {})
        print(f"\n🔍 PERFORMANCE PATTERNS DETECTED:")
        for pattern_type, pattern_info in patterns.items():
            if pattern_info.get("files"):
                print(f"  • {pattern_info['description']}: {len(pattern_info['files'])} files")
        
        print(f"\nAnalysis completed in {results['metadata']['analysis_time']:.2f} seconds")

def main():
    """Main execution function"""
    analyzer = SimplePerformanceAnalyzer()
    
    try:
        results = analyzer.run_comprehensive_analysis()
        
        # Print summary
        analyzer.print_summary(results)
        
        # Save detailed results
        output_file = "simple_performance_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed analysis saved to {output_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

if __name__ == "__main__":
    main()