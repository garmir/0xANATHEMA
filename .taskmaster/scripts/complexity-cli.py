#!/usr/bin/env python3
"""
Command-line interface for the Task Complexity Analysis Engine
Integrates with task-master system for real-time complexity analysis
"""

import argparse
import json
import sys
from pathlib import Path
from task_complexity_analyzer import TaskComplexityAnalyzer, OptimizationEngine, ComplexityDashboard, OptimizationStrategy

def load_tasks_from_taskmaster() -> list:
    """Load tasks from task-master tasks.json file"""
    tasks_file = Path(".taskmaster/tasks/tasks.json")
    
    if not tasks_file.exists():
        print("❌ Error: .taskmaster/tasks/tasks.json not found")
        return []
    
    try:
        with open(tasks_file) as f:
            data = json.load(f)
        
        # Extract tasks from master context
        if "master" in data and "tasks" in data["master"]:
            return data["master"]["tasks"]
        else:
            print("❌ Error: Invalid tasks.json format")
            return []
            
    except Exception as e:
        print(f"❌ Error loading tasks: {e}")
        return []

def analyze_tasks(task_ids: list = None, strategy: str = "adaptive") -> bool:
    """Analyze task complexity and generate optimization plan"""
    
    print("🔍 Loading tasks from task-master...")
    tasks = load_tasks_from_taskmaster()
    
    if not tasks:
        print("❌ No tasks found to analyze")
        return False
    
    # Filter tasks if specific IDs provided
    if task_ids:
        tasks = [task for task in tasks if str(task.get('id')) in task_ids]
        if not tasks:
            print(f"❌ No tasks found with IDs: {task_ids}")
            return False
    
    print(f"📊 Analyzing {len(tasks)} tasks...")
    
    # Initialize analyzer components
    analyzer = TaskComplexityAnalyzer()
    optimizer = OptimizationEngine(analyzer)
    dashboard = ComplexityDashboard()
    
    try:
        # Convert strategy string to enum
        strategy_map = {
            "greedy": OptimizationStrategy.GREEDY,
            "dynamic": OptimizationStrategy.DYNAMIC_PROGRAMMING,
            "ml": OptimizationStrategy.MACHINE_LEARNING,
            "adaptive": OptimizationStrategy.ADAPTIVE
        }
        opt_strategy = strategy_map.get(strategy, OptimizationStrategy.ADAPTIVE)
        
        # Analyze tasks
        analyses = []
        for task in tasks:
            analysis = analyzer.analyze_task(task)
            analyses.append(analysis)
        
        # Generate optimization plan
        execution_plan = optimizer.optimize_execution_plan(tasks, opt_strategy)
        
        # Generate report
        report_path = dashboard.generate_complexity_report(analyses, execution_plan)
        
        # Display summary
        print("\n" + "="*60)
        print("COMPLEXITY ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📈 Tasks analyzed: {len(analyses)}")
        print(f"⚡ Optimization strategy: {strategy}")
        print(f"⏱️  Estimated total time: {execution_plan['estimated_total_time_ms']:.0f}ms")
        print(f"🔧 Optimization opportunities: {len(execution_plan.get('optimization_opportunities', []))}")
        print(f"⚠️  Bottlenecks identified: {len(execution_plan.get('bottlenecks', []))}")
        print(f"⚡ Parallelizable groups: {len(execution_plan.get('parallelization_groups', []))}")
        
        print(f"\n📋 Optimized execution order:")
        for i, task_info in enumerate(execution_plan['execution_order'][:10]):  # Show first 10
            print(f"   {i+1}. Task {task_info['task_id']} ({task_info['estimated_time_ms']:.0f}ms)")
        
        if len(execution_plan['execution_order']) > 10:
            print(f"   ... and {len(execution_plan['execution_order']) - 10} more tasks")
        
        print(f"\n📊 Full report saved: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_performance(max_tasks: int = 1000) -> bool:
    """Validate that analysis completes within 30 seconds for large task sets"""
    
    print(f"🚀 Performance validation with {max_tasks} synthetic tasks...")
    
    import time
    
    # Generate synthetic tasks
    synthetic_tasks = []
    for i in range(max_tasks):
        synthetic_tasks.append({
            "id": str(i),
            "title": f"Synthetic task {i}",
            "details": f"Processing data with size factor {i % 1000}",
            "dependencies": [str(i-1)] if i > 0 and i % 10 != 0 else []
        })
    
    # Initialize analyzer
    analyzer = TaskComplexityAnalyzer()
    optimizer = OptimizationEngine(analyzer)
    
    start_time = time.time()
    
    try:
        # Analyze all tasks
        analyses = []
        for task in synthetic_tasks:
            analysis = analyzer.analyze_task(task)
            analyses.append(analysis)
        
        # Generate optimization plan
        execution_plan = optimizer.optimize_execution_plan(synthetic_tasks)
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        print(f"⏱️  Analysis completed in {analysis_time:.2f} seconds")
        print(f"📊 Tasks processed: {len(analyses)}")
        print(f"⚡ Processing rate: {len(analyses)/analysis_time:.0f} tasks/second")
        
        if analysis_time <= 30.0:
            print("✅ Performance validation PASSED (< 30 seconds)")
            return True
        else:
            print("❌ Performance validation FAILED (> 30 seconds)")
            return False
            
    except Exception as e:
        print(f"❌ Performance validation error: {e}")
        return False

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Task Complexity Analysis and Optimization Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze                    # Analyze all tasks
  %(prog)s analyze --ids 1 2 3        # Analyze specific tasks
  %(prog)s analyze --strategy greedy  # Use greedy optimization
  %(prog)s validate                   # Run performance validation
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze task complexity')
    analyze_parser.add_argument('--ids', nargs='+', help='Specific task IDs to analyze')
    analyze_parser.add_argument('--strategy', 
                               choices=['greedy', 'dynamic', 'ml', 'adaptive'],
                               default='adaptive',
                               help='Optimization strategy to use')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Run performance validation')
    validate_parser.add_argument('--max-tasks', type=int, default=1000,
                                help='Maximum number of tasks for validation')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        success = analyze_tasks(args.ids, args.strategy)
    elif args.command == 'validate':
        success = validate_performance(args.max_tasks)
    else:
        parser.print_help()
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())