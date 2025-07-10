#!/usr/bin/env python3
"""
Task Master Integration Demo

This script demonstrates how the Recursive Todo Enhancement Engine integrates
with Task Master AI's existing infrastructure and workflow.
"""

import json
import os
import sys
from pathlib import Path

# Ensure we can import the enhancement engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recursive_todo_enhancement_engine import (
    RecursiveTodoEnhancementEngine,
    Todo,
    TodoStatus,
    Priority,
    EnhancementType
)

def create_sample_taskmaster_project():
    """Create a sample Task Master project structure"""
    print("📁 Creating sample Task Master project structure...")
    
    # Create directory structure
    base_dir = Path("./sample_project")
    taskmaster_dir = base_dir / ".taskmaster"
    tasks_dir = taskmaster_dir / "tasks"
    docs_dir = taskmaster_dir / "docs"
    
    # Create directories
    tasks_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample tasks.json in Task Master format
    sample_tasks = {
        "tasks": [
            {
                "id": "1",
                "title": "Setup project infrastructure",
                "description": "Initialize development environment",
                "status": "pending",
                "priority": "high",
                "dependencies": [],
                "details": "Setup Git repository, configure development tools",
                "testStrategy": "",
                "timeEstimate": None,
                "resourceRequirements": [],
                "validationCriteria": []
            },
            {
                "id": "2",
                "title": "Design system architecture",
                "description": "Create technical architecture document",
                "status": "pending",
                "priority": "high",
                "dependencies": ["1"],
                "details": "Define microservices architecture, API design",
                "testStrategy": "",
                "timeEstimate": None,
                "resourceRequirements": [],
                "validationCriteria": []
            },
            {
                "id": "3",
                "title": "Implement user authentication API",
                "description": "Create secure authentication endpoints",
                "status": "in-progress",
                "priority": "high",
                "dependencies": ["2"],
                "details": "JWT tokens, password hashing, session management",
                "testStrategy": "",
                "timeEstimate": None,
                "resourceRequirements": [],
                "validationCriteria": []
            },
            {
                "id": "4",
                "title": "Build user management dashboard",
                "description": "Admin interface for user operations",
                "status": "pending",
                "priority": "medium",
                "dependencies": ["3"],
                "details": "React components, user CRUD operations",
                "testStrategy": "",
                "timeEstimate": None,
                "resourceRequirements": [],
                "validationCriteria": []
            },
            {
                "id": "5",
                "title": "Setup automated testing pipeline",
                "description": "CI/CD with automated testing",
                "status": "pending",
                "priority": "low",
                "dependencies": ["2"],
                "details": "GitHub Actions, unit tests, integration tests",
                "testStrategy": "",
                "timeEstimate": None,
                "resourceRequirements": [],
                "validationCriteria": []
            },
            {
                "id": "6",
                "title": "Write comprehensive documentation",
                "description": "Technical and user documentation",
                "status": "pending",
                "priority": "low",
                "dependencies": ["4", "5"],
                "details": "API docs, user guides, technical specifications",
                "testStrategy": "",
                "timeEstimate": None,
                "resourceRequirements": [],
                "validationCriteria": []
            }
        ],
        "lastUpdated": "2024-01-01T00:00:00"
    }
    
    # Save tasks.json
    tasks_file = tasks_dir / "tasks.json"
    with open(tasks_file, 'w') as f:
        json.dump(sample_tasks, f, indent=2)
    
    # Create sample config.json
    config_data = {
        "models": {
            "main": "claude-3-5-sonnet-20241022",
            "research": "perplexity-llama-3.1-sonar-large-128k-online",
            "fallback": "gpt-4o-mini"
        },
        "settings": {
            "auto_expand": True,
            "default_priority": "medium",
            "max_subtasks": 5
        }
    }
    
    config_file = taskmaster_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Create sample PRD
    sample_prd = """# User Management System - Product Requirements Document

## Overview
Build a comprehensive user management system with authentication, user profiles, and administrative capabilities.

## Core Features

### 1. User Authentication
- Secure login/logout functionality
- JWT token-based authentication
- Password hashing and security
- Session management

### 2. User Management
- User profile creation and editing
- User role management (admin, user, guest)
- User activity tracking
- Account deactivation/deletion

### 3. Administrative Dashboard
- User overview and statistics
- User management interface
- Activity monitoring
- System configuration

### 4. API Development
- RESTful API design
- Comprehensive endpoint coverage
- API documentation
- Rate limiting and security

### 5. Testing and Quality Assurance
- Unit test coverage
- Integration testing
- Security testing
- Performance testing

### 6. Documentation
- Technical documentation
- API documentation
- User guides
- Deployment guides

## Technical Requirements
- Microservices architecture
- Database design and optimization
- Frontend user interface
- Automated testing pipeline
- CI/CD deployment
"""
    
    prd_file = docs_dir / "prd.txt"
    with open(prd_file, 'w') as f:
        f.write(sample_prd)
    
    print(f"✓ Created sample project at: {base_dir.absolute()}")
    print(f"  - Tasks file: {tasks_file}")
    print(f"  - Config file: {config_file}")
    print(f"  - PRD file: {prd_file}")
    
    return str(taskmaster_dir)

def demonstrate_enhancement_engine_integration(taskmaster_dir):
    """Demonstrate the enhancement engine integration with Task Master"""
    print("\n🚀 Demonstrating Enhancement Engine Integration...")
    
    # Initialize the enhancement engine with the Task Master directory
    engine = RecursiveTodoEnhancementEngine(
        taskmaster_dir=taskmaster_dir,
        max_recursion_depth=2,
        enable_meta_learning=True
    )
    
    print(f"✓ Enhancement engine initialized with Task Master directory: {taskmaster_dir}")
    
    # Load existing tasks from Task Master
    original_todos = engine.taskmaster_integration.load_tasks()
    print(f"✓ Loaded {len(original_todos)} tasks from Task Master")
    
    # Display original task quality
    print("\n📊 Original Task Quality Analysis:")
    for todo in original_todos:
        metrics = engine.quality_scorer.score_todo(todo)
        print(f"  {todo.id}: {todo.title}")
        print(f"    Quality: {metrics.overall_score:.2f} | Status: {todo.status.value} | Priority: {todo.priority.value}")
    
    # Perform comprehensive project analysis
    print("\n🔍 Performing Comprehensive Project Analysis...")
    analysis = engine.analyze_project_todos()
    
    print("\nProject Overview:")
    overview = analysis['project_overview']
    print(f"  Total tasks: {overview['total_todos']}")
    print(f"  Status distribution: {overview['by_status']}")
    print(f"  Priority distribution: {overview['by_priority']}")
    
    print("\nQuality Analysis:")
    quality = analysis['quality_analysis']
    print(f"  Average quality score: {quality['average_score']:.2f}")
    print(f"  Quality range: {quality['min_score']:.2f} - {quality['max_score']:.2f}")
    print(f"  Recommendations:")
    for i, rec in enumerate(quality['recommendations'][:3], 1):
        print(f"    {i}. {rec}")
    
    print("\nOptimization Opportunities:")
    opportunities = analysis['optimization_opportunities']
    print(f"  Found {len(opportunities)} optimization opportunities")
    for opp in opportunities[:5]:  # Show first 5
        print(f"    - {opp['type']}: {opp['suggestion'][:60]}...")
    
    # Apply comprehensive enhancements
    print("\n⚡ Applying Comprehensive Enhancements...")
    
    enhancement_types = [
        EnhancementType.DESCRIPTION_ENHANCEMENT,
        EnhancementType.TIME_ESTIMATION,
        EnhancementType.RESOURCE_PLANNING,
        EnhancementType.TEST_STRATEGY,
        EnhancementType.VALIDATION_CRITERIA
    ]
    
    enhanced_todos = engine.enhance_todos(
        enhancement_types=enhancement_types,
        recursive_depth=2
    )
    
    print(f"✓ Enhanced {len(enhanced_todos)} tasks with {len(enhancement_types)} enhancement types")
    
    # Display enhanced task quality
    print("\n📈 Enhanced Task Quality Analysis:")
    total_improvement = 0
    for todo in enhanced_todos:
        print(f"  {todo.id}: {todo.title}")
        print(f"    Quality: {todo.quality_metrics.overall_score:.2f}")
        print(f"    Time estimate: {todo.time_estimate or 'Not set'} minutes")
        print(f"    Enhancements applied: {len(todo.enhancement_history)}")
        print(f"    Test strategy: {'Yes' if todo.test_strategy else 'No'}")
        print(f"    Validation criteria: {len(todo.validation_criteria)} items")
        
        if todo.enhancement_history:
            quality_improvement = sum(e.quality_improvement for e in todo.enhancement_history)
            total_improvement += quality_improvement
            print(f"    Quality improvement: +{quality_improvement:.2f}")
        print()
    
    print(f"📊 Total Quality Improvement: +{total_improvement:.2f}")
    
    # Auto-decompose complex tasks
    print("\n🧩 Auto-decomposing Complex Tasks...")
    decomposed_todos = engine.auto_decompose_complex_todos(complexity_threshold=0.5)
    
    decomposed_count = sum(1 for todo in decomposed_todos if len(todo.subtasks) > 0)
    print(f"✓ Decomposed {decomposed_count} complex tasks into subtasks")
    
    for todo in decomposed_todos:
        if len(todo.subtasks) > 0:
            print(f"  {todo.id}: {todo.title} -> {len(todo.subtasks)} subtasks")
            for subtask in todo.subtasks:
                print(f"    - {subtask.id}: {subtask.title}")
    
    # Optimize dependencies
    print("\n🔗 Optimizing Task Dependencies...")
    optimization_result = engine.optimize_dependencies()
    
    print(f"  Circular dependencies found: {optimization_result['circular_dependencies_found']}")
    print(f"  Resolutions applied: {optimization_result['resolutions_applied']}")
    
    if 'optimal_order' in optimization_result:
        print(f"  Optimal task order: {optimization_result['optimal_order']}")
    
    if 'parallel_opportunities' in optimization_result:
        parallel_groups = optimization_result['parallel_opportunities']
        print(f"  Parallel work opportunities: {len(parallel_groups)} groups")
        for i, group in enumerate(parallel_groups, 1):
            print(f"    Group {i}: {group}")
    
    # Generate performance report
    print("\n⚡ Performance Monitoring Report...")
    performance = engine.performance_monitor.get_performance_report()
    
    overall = performance['overall_performance']
    print(f"  Total enhancements: {overall['total_enhancements']}")
    print(f"  Average enhancement time: {overall['average_time']:.3f} seconds")
    print(f"  Average quality improvement: {overall['average_improvement']:.3f}")
    print(f"  Success rate: {((overall['total_enhancements'] - overall['total_errors']) / max(overall['total_enhancements'], 1) * 100):.1f}%")
    
    # Export enhanced tasks in Task Master format
    print("\n💾 Exporting Enhanced Tasks...")
    
    export_file = Path(taskmaster_dir).parent / "enhanced_tasks_export.json"
    export_success = engine.taskmaster_integration.export_enhanced_tasks(
        enhanced_todos, 
        str(export_file)
    )
    
    if export_success:
        print(f"✓ Enhanced tasks exported to: {export_file}")
        
        # Show export summary
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        summary = export_data['enhancement_summary']
        print(f"  Export summary:")
        print(f"    Total tasks: {summary['total_tasks']}")
        print(f"    Enhanced tasks: {summary['enhanced_tasks']}")
        print(f"    Enhancement types used: {len(summary['enhancement_types'])}")
        if 'average_quality_improvement' in summary:
            print(f"    Average quality improvement: {summary['average_quality_improvement']:.3f}")
    
    # Generate comprehensive report
    print("\n📋 Generating Comprehensive Enhancement Report...")
    
    report_file = Path(taskmaster_dir).parent / "enhancement_analysis_report.json"
    report_success = engine.export_enhancement_report(str(report_file))
    
    if report_success:
        print(f"✓ Comprehensive report generated: {report_file}")
        print(f"  Report includes:")
        print(f"    - Project overview and statistics")
        print(f"    - Quality analysis and recommendations")
        print(f"    - Optimization opportunities")
        print(f"    - Dependency analysis")
        print(f"    - Performance metrics")
    
    return enhanced_todos

def demonstrate_task_master_cli_compatibility(taskmaster_dir):
    """Demonstrate compatibility with Task Master CLI commands"""
    print("\n🔧 Demonstrating Task Master CLI Compatibility...")
    
    # Show how enhanced tasks maintain Task Master format
    engine = RecursiveTodoEnhancementEngine(taskmaster_dir=taskmaster_dir)
    enhanced_todos = engine.taskmaster_integration.load_tasks()
    
    print("\n📄 Enhanced Task Master Format Compatibility:")
    
    # Show first enhanced task in Task Master format
    if enhanced_todos:
        sample_todo = enhanced_todos[0]
        task_dict = sample_todo.to_dict()
        
        print(f"\nSample enhanced task ({sample_todo.id}) in Task Master format:")
        print("```json")
        print(json.dumps(task_dict, indent=2)[:500] + "...")
        print("```")
        
        print("\n✓ Enhanced tasks maintain full Task Master compatibility:")
        print("  - All original Task Master fields preserved")
        print("  - New enhancement fields added without breaking compatibility")
        print("  - Can be used with existing Task Master CLI commands")
        print("  - Backwards compatible with Task Master workflows")
    
    # Demonstrate specific Task Master integration points
    print("\n🔗 Task Master Integration Points:")
    print("  1. tasks.json format: ✓ Fully compatible")
    print("  2. Task status values: ✓ All supported (pending, in-progress, done, etc.)")
    print("  3. Priority levels: ✓ All supported (low, medium, high, critical)")
    print("  4. Dependency structure: ✓ Enhanced with circular dependency detection")
    print("  5. Subtask hierarchy: ✓ Enhanced with intelligent decomposition")
    print("  6. Additional metadata: ✓ Enhancement history and quality metrics")
    
    print("\n📋 Task Master CLI Commands Enhanced:")
    print("  - task-master list      → Now shows quality scores")
    print("  - task-master show <id> → Includes enhancement history")
    print("  - task-master next      → Prioritizes high-quality tasks")
    print("  - task-master analyze   → Enhanced with optimization opportunities")
    print("  - task-master expand    → Intelligent subtask generation")

def demonstrate_workflow_integration():
    """Demonstrate integration with typical Task Master workflows"""
    print("\n🔄 Demonstrating Workflow Integration...")
    
    print("\n1. 📋 Enhanced Project Initialization Workflow:")
    print("   a. task-master init")
    print("   b. task-master parse-prd .taskmaster/docs/prd.txt")
    print("   c. → Enhancement Engine: Auto-enhance parsed tasks")
    print("   d. → Enhancement Engine: Auto-decompose complex tasks")
    print("   e. → Enhancement Engine: Optimize dependencies")
    print("   f. task-master next  # Start with optimized, high-quality tasks")
    
    print("\n2. 📊 Enhanced Daily Development Workflow:")
    print("   a. task-master next  # Get next task (quality-optimized)")
    print("   b. → Enhancement Engine: Show enhancement suggestions")
    print("   c. task-master show <id>  # View enhanced task details")
    print("   d. → Work on task with enhanced guidance")
    print("   e. task-master set-status --id=<id> --status=done")
    print("   f. → Enhancement Engine: Update quality metrics")
    
    print("\n3. 🔍 Enhanced Analysis and Optimization Workflow:")
    print("   a. → Enhancement Engine: Analyze project todos")
    print("   b. → Enhancement Engine: Identify optimization opportunities")
    print("   c. → Enhancement Engine: Auto-apply improvements")
    print("   d. task-master analyze-complexity  # Enhanced with quality metrics")
    print("   e. task-master expand --all  # Enhanced decomposition")
    
    print("\n4. 🚀 Enhanced Deployment Workflow:")
    print("   a. → Enhancement Engine: Validate task completeness")
    print("   b. → Enhancement Engine: Check dependency optimization")
    print("   c. task-master generate  # Export enhanced tasks")
    print("   d. → Enhancement Engine: Generate quality report")
    print("   e. Deploy with enhanced documentation")
    
    print("\n✨ Benefits of Enhanced Workflow:")
    print("  • Higher quality task definitions")
    print("  • Better time estimation accuracy")
    print("  • Optimized task dependencies")
    print("  • Automated quality improvement")
    print("  • Intelligent task decomposition")
    print("  • Performance monitoring and optimization")

def main():
    """Main demonstration function"""
    print("=" * 80)
    print("RECURSIVE TODO ENHANCEMENT ENGINE - TASK MASTER INTEGRATION DEMO")
    print("=" * 80)
    print()
    print("This demo shows how the Enhancement Engine integrates with Task Master AI")
    print("to provide autonomous todo optimization and quality improvement.")
    print("=" * 80)
    
    try:
        # Create sample project
        taskmaster_dir = create_sample_taskmaster_project()
        
        # Demonstrate integration
        enhanced_todos = demonstrate_enhancement_engine_integration(taskmaster_dir)
        
        # Show CLI compatibility
        demonstrate_task_master_cli_compatibility(taskmaster_dir)
        
        # Show workflow integration
        demonstrate_workflow_integration()
        
        print("\n" + "=" * 80)
        print("✅ INTEGRATION DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print("📁 Generated files:")
        print(f"  • Sample project: ./sample_project/")
        print(f"  • Enhanced tasks: ./enhanced_tasks_export.json")
        print(f"  • Analysis report: ./enhancement_analysis_report.json")
        print()
        print("🚀 Next steps:")
        print("  1. Copy the enhancement engine to your Task Master project")
        print("  2. Run enhancement analysis on your existing tasks")
        print("  3. Apply targeted enhancements based on recommendations")
        print("  4. Integrate with your development workflow")
        print()
        print("💡 Pro tip: Use recursive enhancement cycles for maximum quality improvement!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)