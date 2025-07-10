#!/usr/bin/env python3
"""
Example usage of the Recursive Todo Enhancement Engine

This script demonstrates how to use the various components of the enhancement engine
to analyze, optimize, and enhance todo lists for improved productivity.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the enhancement directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recursive_todo_enhancement_engine import (
    RecursiveTodoEnhancementEngine,
    Todo,
    TodoStatus,
    Priority,
    EnhancementType
)

def example_basic_enhancement():
    """Example of basic todo enhancement"""
    print("=" * 60)
    print("BASIC TODO ENHANCEMENT EXAMPLE")
    print("=" * 60)
    
    # Create sample todos
    todos = [
        Todo(
            id="1",
            title="Create API",
            description="API for users",
            priority=Priority.HIGH
        ),
        Todo(
            id="2",
            title="Database setup",
            description="Setup database",
            priority=Priority.MEDIUM,
            dependencies=["1"]
        ),
        Todo(
            id="3",
            title="Frontend",
            description="User interface",
            priority=Priority.LOW
        )
    ]
    
    print(f"Original todos: {len(todos)}")
    for todo in todos:
        print(f"  - {todo.id}: {todo.title}")
        print(f"    Quality score: {todo.quality_metrics.overall_score:.2f}")
    
    # Initialize enhancement engine
    engine = RecursiveTodoEnhancementEngine(
        taskmaster_dir="./example_taskmaster",
        enable_meta_learning=False
    )
    
    # Enhance todos
    enhanced_todos = engine.enhance_todos(
        todos=todos,
        enhancement_types=[
            EnhancementType.DESCRIPTION_ENHANCEMENT,
            EnhancementType.TIME_ESTIMATION,
            EnhancementType.TEST_STRATEGY
        ],
        recursive_depth=1
    )
    
    print(f"\nEnhanced todos: {len(enhanced_todos)}")
    for todo in enhanced_todos:
        print(f"  - {todo.id}: {todo.title}")
        print(f"    Quality score: {todo.quality_metrics.overall_score:.2f}")
        print(f"    Time estimate: {todo.time_estimate} minutes")
        print(f"    Test strategy: {todo.test_strategy[:50]}...")
        print(f"    Enhancements applied: {len(todo.enhancement_history)}")
        print()

def example_project_analysis():
    """Example of comprehensive project analysis"""
    print("=" * 60)
    print("PROJECT ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # Create a realistic project structure
    project_todos = [
        Todo(
            id="1",
            title="Setup development environment",
            description="Initialize project structure and dependencies",
            priority=Priority.HIGH,
            status=TodoStatus.PENDING
        ),
        Todo(
            id="2",
            title="Design system architecture",
            description="Create technical architecture document",
            priority=Priority.HIGH,
            status=TodoStatus.PENDING,
            dependencies=["1"]
        ),
        Todo(
            id="3",
            title="Implement user authentication system",
            description="Create secure authentication with JWT tokens",
            priority=Priority.HIGH,
            status=TodoStatus.IN_PROGRESS,
            dependencies=["2"]
        ),
        Todo(
            id="4",
            title="Build user management API endpoints",
            description="REST API for user CRUD operations",
            priority=Priority.MEDIUM,
            status=TodoStatus.PENDING,
            dependencies=["3"]
        ),
        Todo(
            id="5",
            title="Create frontend user interface",
            description="React components for user management",
            priority=Priority.MEDIUM,
            status=TodoStatus.PENDING,
            dependencies=["4"]
        ),
        Todo(
            id="6",
            title="Setup automated testing pipeline",
            description="CI/CD with automated testing",
            priority=Priority.LOW,
            status=TodoStatus.PENDING,
            dependencies=["2"]
        ),
        Todo(
            id="7",
            title="Write comprehensive documentation",
            description="Technical and user documentation",
            priority=Priority.LOW,
            status=TodoStatus.PENDING,
            dependencies=["5", "6"]
        )
    ]
    
    # Initialize engine
    engine = RecursiveTodoEnhancementEngine(
        taskmaster_dir="./example_taskmaster",
        enable_meta_learning=False
    )
    
    # Save todos to analysis
    engine.taskmaster_integration.save_tasks(project_todos)
    
    # Perform comprehensive analysis
    analysis = engine.analyze_project_todos()
    
    print("PROJECT OVERVIEW:")
    print(f"  Total todos: {analysis['project_overview']['total_todos']}")
    print(f"  By status: {analysis['project_overview']['by_status']}")
    print(f"  By priority: {analysis['project_overview']['by_priority']}")
    
    print("\nQUALITY ANALYSIS:")
    quality = analysis['quality_analysis']
    print(f"  Average quality score: {quality['average_score']:.2f}")
    print(f"  Quality range: {quality['min_score']:.2f} - {quality['max_score']:.2f}")
    print(f"  Recommendations: {len(quality['recommendations'])}")
    for rec in quality['recommendations']:
        print(f"    - {rec}")
    
    print("\nOPTIMIZATION OPPORTUNITIES:")
    opportunities = analysis['optimization_opportunities']
    print(f"  Found {len(opportunities)} optimization opportunities:")
    for opp in opportunities:
        print(f"    - {opp['type']}: {opp['suggestion']}")
    
    print("\nDEPENDENCY ANALYSIS:")
    deps = analysis['dependency_analysis']
    print(f"  Circular dependencies: {len(deps['circular_dependencies'])}")
    print(f"  Parallel opportunities: {len(deps['parallel_opportunities'])}")
    print(f"  Optimal order: {deps['optimal_order']}")
    
    print("\nDECOMPOSITION RECOMMENDATIONS:")
    decomp = analysis['decomposition_recommendations']
    print(f"  {len(decomp)} tasks recommended for decomposition:")
    for rec in decomp:
        print(f"    - {rec['todo_title']} (complexity: {rec['complexity_score']:.2f})")

def example_recursive_enhancement():
    """Example of recursive enhancement with multiple cycles"""
    print("=" * 60)
    print("RECURSIVE ENHANCEMENT EXAMPLE")
    print("=" * 60)
    
    # Create a complex todo that benefits from multiple enhancement cycles
    complex_todo = Todo(
        id="1",
        title="Build microservices platform",
        description="Create scalable microservices architecture",
        priority=Priority.HIGH
    )
    
    print(f"Original todo: {complex_todo.title}")
    print(f"Description: {complex_todo.description}")
    print(f"Initial quality: {complex_todo.quality_metrics.overall_score:.2f}")
    
    # Initialize engine
    engine = RecursiveTodoEnhancementEngine(
        taskmaster_dir="./example_taskmaster",
        enable_meta_learning=False
    )
    
    # Apply multiple enhancement cycles
    enhanced_todos = engine.enhance_todos(
        todos=[complex_todo],
        enhancement_types=[
            EnhancementType.DESCRIPTION_ENHANCEMENT,
            EnhancementType.TIME_ESTIMATION,
            EnhancementType.RESOURCE_PLANNING,
            EnhancementType.TEST_STRATEGY,
            EnhancementType.VALIDATION_CRITERIA
        ],
        recursive_depth=3
    )
    
    enhanced_todo = enhanced_todos[0]
    
    print(f"\nAfter {len(enhanced_todo.enhancement_history)} enhancements:")
    print(f"Enhanced description: {enhanced_todo.description}")
    print(f"Time estimate: {enhanced_todo.time_estimate} minutes")
    print(f"Resource requirements: {enhanced_todo.resource_requirements}")
    print(f"Test strategy: {enhanced_todo.test_strategy}")
    print(f"Validation criteria: {enhanced_todo.validation_criteria}")
    print(f"Final quality: {enhanced_todo.quality_metrics.overall_score:.2f}")
    
    print("\nEnhancement history:")
    for i, enhancement in enumerate(enhanced_todo.enhancement_history):
        print(f"  {i+1}. {enhancement.enhancement_type.value}")
        print(f"     Quality improvement: +{enhancement.quality_improvement:.2f}")
        print(f"     Suggestions: {len(enhancement.suggestions)}")

def example_task_decomposition():
    """Example of automatic task decomposition"""
    print("=" * 60)
    print("TASK DECOMPOSITION EXAMPLE")
    print("=" * 60)
    
    # Create complex todos that should be decomposed
    complex_todos = [
        Todo(
            id="1",
            title="Create comprehensive e-commerce platform",
            description="Build complete online shopping system with payment processing, inventory management, and user accounts",
            priority=Priority.HIGH
        ),
        Todo(
            id="2",
            title="Implement real-time analytics dashboard",
            description="Create dashboard with live data visualization, reporting, and monitoring capabilities",
            priority=Priority.MEDIUM
        ),
        Todo(
            id="3",
            title="Setup multi-region deployment infrastructure",
            description="Implement scalable cloud infrastructure with load balancing, auto-scaling, and disaster recovery",
            priority=Priority.LOW
        )
    ]
    
    print(f"Original complex todos: {len(complex_todos)}")
    for todo in complex_todos:
        print(f"  - {todo.id}: {todo.title}")
        print(f"    Subtasks: {len(todo.subtasks)}")
    
    # Initialize engine
    engine = RecursiveTodoEnhancementEngine(
        taskmaster_dir="./example_taskmaster",
        enable_meta_learning=False
    )
    
    # Save and decompose complex todos
    engine.taskmaster_integration.save_tasks(complex_todos)
    decomposed_todos = engine.auto_decompose_complex_todos(complexity_threshold=0.5)
    
    print(f"\nAfter decomposition: {len(decomposed_todos)} todos")
    for todo in decomposed_todos:
        print(f"  - {todo.id}: {todo.title}")
        print(f"    Subtasks: {len(todo.subtasks)}")
        
        for subtask in todo.subtasks:
            print(f"      - {subtask.id}: {subtask.title}")
            print(f"        Description: {subtask.description}")
            print(f"        Test strategy: {subtask.test_strategy}")
            print()

def example_dependency_optimization():
    """Example of dependency analysis and optimization"""
    print("=" * 60)
    print("DEPENDENCY OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # Create todos with complex dependencies (including circular)
    dependency_todos = [
        Todo(id="1", title="Setup database", dependencies=["2"]),
        Todo(id="2", title="Create API", dependencies=["3"]),
        Todo(id="3", title="Design schema", dependencies=["1"]),  # Creates circle
        Todo(id="4", title="Build frontend", dependencies=["2"]),
        Todo(id="5", title="Write tests", dependencies=["4"]),
        Todo(id="6", title="Deploy system", dependencies=["5"])
    ]
    
    print("Original dependencies:")
    for todo in dependency_todos:
        print(f"  {todo.id}: {todo.title} -> {todo.dependencies}")
    
    # Initialize engine
    engine = RecursiveTodoEnhancementEngine(
        taskmaster_dir="./example_taskmaster",
        enable_meta_learning=False
    )
    
    # Save and optimize dependencies
    engine.taskmaster_integration.save_tasks(dependency_todos)
    optimization_result = engine.optimize_dependencies()
    
    print(f"\nOptimization results:")
    print(f"  Circular dependencies found: {optimization_result['circular_dependencies_found']}")
    print(f"  Resolutions applied: {optimization_result['resolutions_applied']}")
    print(f"  Optimal order: {optimization_result['optimal_order']}")
    
    print(f"\nParallel opportunities: {len(optimization_result['parallel_opportunities'])}")
    for group in optimization_result['parallel_opportunities']:
        print(f"  Can work in parallel: {group}")
    
    # Load optimized todos
    optimized_todos = engine.taskmaster_integration.load_tasks()
    print("\nOptimized dependencies:")
    for todo in optimized_todos:
        print(f"  {todo.id}: {todo.title} -> {todo.dependencies}")

def example_batch_enhancement():
    """Example of batch enhancement by pattern"""
    print("=" * 60)
    print("BATCH ENHANCEMENT EXAMPLE")
    print("=" * 60)
    
    # Create todos with specific patterns
    pattern_todos = [
        Todo(id="1", title="Create user API", description="API for user operations"),
        Todo(id="2", title="Create product API", description="API for product management"),
        Todo(id="3", title="Create order API", description="API for order processing"),
        Todo(id="4", title="Setup database", description="Database configuration"),
        Todo(id="5", title="Create admin API", description="API for admin functions"),
        Todo(id="6", title="Write documentation", description="Technical documentation")
    ]
    
    print(f"Original todos: {len(pattern_todos)}")
    for todo in pattern_todos:
        print(f"  - {todo.id}: {todo.title}")
        print(f"    Enhancement history: {len(todo.enhancement_history)}")
    
    # Initialize engine
    engine = RecursiveTodoEnhancementEngine(
        taskmaster_dir="./example_taskmaster",
        enable_meta_learning=False
    )
    
    # Save todos
    engine.taskmaster_integration.save_tasks(pattern_todos)
    
    # Batch enhance all API-related todos
    enhanced_api_todos = engine.batch_enhance_by_pattern(
        pattern="API",
        enhancement_types=[
            EnhancementType.DESCRIPTION_ENHANCEMENT,
            EnhancementType.TIME_ESTIMATION,
            EnhancementType.TEST_STRATEGY,
            EnhancementType.VALIDATION_CRITERIA
        ]
    )
    
    print(f"\nEnhanced API todos: {len(enhanced_api_todos)}")
    for todo in enhanced_api_todos:
        print(f"  - {todo.id}: {todo.title}")
        print(f"    Enhancement history: {len(todo.enhancement_history)}")
        print(f"    Time estimate: {todo.time_estimate} minutes")
        print(f"    Test strategy: {todo.test_strategy[:50]}...")
        print(f"    Validation criteria: {len(todo.validation_criteria)} items")
        print()

def example_performance_monitoring():
    """Example of performance monitoring and reporting"""
    print("=" * 60)
    print("PERFORMANCE MONITORING EXAMPLE")
    print("=" * 60)
    
    # Create todos for performance testing
    performance_todos = [
        Todo(id="1", title="Simple task", description="Easy implementation"),
        Todo(id="2", title="Complex task", description="Complex implementation with multiple components"),
        Todo(id="3", title="API task", description="REST API implementation"),
        Todo(id="4", title="Database task", description="Database schema and operations"),
        Todo(id="5", title="UI task", description="User interface components")
    ]
    
    # Initialize engine
    engine = RecursiveTodoEnhancementEngine(
        taskmaster_dir="./example_taskmaster",
        enable_meta_learning=False
    )
    
    # Enhance todos to generate performance data
    print("Enhancing todos and collecting performance data...")
    enhanced_todos = engine.enhance_todos(
        todos=performance_todos,
        enhancement_types=[
            EnhancementType.DESCRIPTION_ENHANCEMENT,
            EnhancementType.TIME_ESTIMATION,
            EnhancementType.TEST_STRATEGY,
            EnhancementType.VALIDATION_CRITERIA
        ],
        recursive_depth=2
    )
    
    # Get performance report
    performance_report = engine.performance_monitor.get_performance_report()
    
    print("\nPERFORMANCE REPORT:")
    print(f"Overall performance:")
    overall = performance_report['overall_performance']
    print(f"  Total enhancements: {overall['total_enhancements']}")
    print(f"  Average time: {overall['average_time']:.2f} seconds")
    print(f"  Average improvement: {overall['average_improvement']:.2f}")
    print(f"  Total errors: {overall['total_errors']}")
    
    print("\nEnhancement type statistics:")
    for enhancement_type, stats in performance_report['enhancement_statistics'].items():
        print(f"  {enhancement_type}:")
        print(f"    Average time: {stats['average_time']:.2f} seconds")
        print(f"    Average improvement: {stats['average_improvement']:.2f}")
        print(f"    Success rate: {stats['success_rate']:.2f}")
        print(f"    Success/Error count: {stats['success_count']}/{stats['error_count']}")

def example_export_and_reporting():
    """Example of exporting enhanced todos and generating reports"""
    print("=" * 60)
    print("EXPORT AND REPORTING EXAMPLE")
    print("=" * 60)
    
    # Create a sample project
    project_todos = [
        Todo(id="1", title="Project setup", description="Initialize project"),
        Todo(id="2", title="Core implementation", description="Main functionality"),
        Todo(id="3", title="Testing", description="Test suite", dependencies=["2"]),
        Todo(id="4", title="Documentation", description="Technical docs", dependencies=["3"]),
        Todo(id="5", title="Deployment", description="Production deployment", dependencies=["4"])
    ]
    
    # Initialize engine
    engine = RecursiveTodoEnhancementEngine(
        taskmaster_dir="./example_taskmaster",
        enable_meta_learning=False
    )
    
    # Enhance project todos
    enhanced_todos = engine.enhance_todos(
        todos=project_todos,
        enhancement_types=[
            EnhancementType.DESCRIPTION_ENHANCEMENT,
            EnhancementType.TIME_ESTIMATION,
            EnhancementType.RESOURCE_PLANNING,
            EnhancementType.TEST_STRATEGY,
            EnhancementType.VALIDATION_CRITERIA
        ],
        recursive_depth=2
    )
    
    # Export enhanced todos
    export_file = "./enhanced_project_todos.json"
    export_success = engine.taskmaster_integration.export_enhanced_tasks(enhanced_todos, export_file)
    
    if export_success:
        print(f"Enhanced todos exported to: {export_file}")
        
        # Show export summary
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        summary = export_data['enhancement_summary']
        print(f"\nExport Summary:")
        print(f"  Total tasks: {summary['total_tasks']}")
        print(f"  Enhanced tasks: {summary['enhanced_tasks']}")
        print(f"  Enhancement types used: {dict(summary['enhancement_types'])}")
        if 'average_quality_improvement' in summary:
            print(f"  Average quality improvement: {summary['average_quality_improvement']:.2f}")
    
    # Generate comprehensive report
    report_file = "./comprehensive_enhancement_report.json"
    report_success = engine.export_enhancement_report(report_file)
    
    if report_success:
        print(f"\nComprehensive report exported to: {report_file}")
        
        # Show report highlights
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        print("\nReport Highlights:")
        print(f"  Project overview: {report_data['project_overview']['total_todos']} todos")
        print(f"  Average quality: {report_data['quality_analysis']['average_score']:.2f}")
        print(f"  Optimization opportunities: {len(report_data['optimization_opportunities'])}")
        print(f"  Decomposition recommendations: {len(report_data['decomposition_recommendations'])}")

def main():
    """Run all examples"""
    print("RECURSIVE TODO ENHANCEMENT ENGINE - EXAMPLES")
    print("=" * 80)
    print("This script demonstrates the various capabilities of the enhancement engine.")
    print("=" * 80)
    
    examples = [
        ("Basic Enhancement", example_basic_enhancement),
        ("Project Analysis", example_project_analysis),
        ("Recursive Enhancement", example_recursive_enhancement),
        ("Task Decomposition", example_task_decomposition),
        ("Dependency Optimization", example_dependency_optimization),
        ("Batch Enhancement", example_batch_enhancement),
        ("Performance Monitoring", example_performance_monitoring),
        ("Export and Reporting", example_export_and_reporting)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n{name}...")
            example_func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
        
        print("\n" + "-" * 60)
    
    print("\nAll examples completed!")
    print("Check the generated files:")
    print("  - ./enhanced_project_todos.json")
    print("  - ./comprehensive_enhancement_report.json")
    print("  - ./example_taskmaster/ directory")

if __name__ == "__main__":
    main()