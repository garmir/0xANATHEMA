#!/usr/bin/env python3
"""
Focused Enhancement Engine Runner
Runs the enhancement engine on current task-master todos with targeted analysis
"""

import json
from recursive_todo_enhancement_engine import RecursiveTodoEnhancer
from taskmaster_enhancement_integration import TaskMasterIntegration

def run_focused_enhancement():
    """Run focused enhancement on current todos"""
    print("🚀 Starting Focused Todo Enhancement Engine")
    print("=" * 60)
    
    # Initialize components
    enhancer = RecursiveTodoEnhancer()
    integration = TaskMasterIntegration()
    
    # Load current todos
    print("📋 Loading current todos from task-master...")
    todos = integration.load_taskmaster_todos()
    
    if not todos:
        print("❌ No todos found!")
        return
    
    # Focus on recent or in-progress todos
    active_todos = [
        todo for todo in todos 
        if todo.get('status') in ['pending', 'in-progress'] and 
        int(todo.get('id', '0')) >= 50  # Recent tasks
    ][:10]  # Limit to 10 for focused analysis
    
    print(f"🎯 Analyzing {len(active_todos)} active todos (IDs 50+)...")
    
    # Analyze quality
    print("\n📊 QUALITY ANALYSIS")
    print("-" * 40)
    
    quality_results = []
    for todo in active_todos:
        metrics = enhancer.analyze_todo_quality(todo)
        quality_results.append({
            'id': todo.get('id'),
            'content': todo.get('content', '')[:50] + '...',
            'overall_score': metrics.overall_score,
            'clarity': metrics.clarity_score,
            'actionability': metrics.actionability_score
        })
    
    # Sort by quality score (lowest first - needs most improvement)
    quality_results.sort(key=lambda x: x['overall_score'])
    
    for result in quality_results:
        print(f"Todo {result['id']}: {result['content']}")
        print(f"  📈 Overall: {result['overall_score']:.2f} | "
              f"✨ Clarity: {result['clarity']:.2f} | "
              f"⚡ Action: {result['actionability']:.2f}")
    
    # Generate enhancement suggestions
    print(f"\n🔧 ENHANCEMENT SUGGESTIONS")
    print("-" * 40)
    
    suggestions = enhancer.generate_enhancement_suggestions(active_todos)
    
    # Group suggestions by type
    by_type = {}
    for suggestion in suggestions:
        type_name = suggestion.type.value
        if type_name not in by_type:
            by_type[type_name] = []
        by_type[type_name].append(suggestion)
    
    for enhancement_type, type_suggestions in by_type.items():
        print(f"\n🏷️ {enhancement_type.replace('_', ' ').title()} ({len(type_suggestions)} suggestions)")
        
        # Show top 3 suggestions for this type
        top_suggestions = sorted(type_suggestions, key=lambda x: x.confidence, reverse=True)[:3]
        
        for suggestion in top_suggestions:
            print(f"  • Todo {suggestion.todo_id}: {suggestion.description}")
            print(f"    Confidence: {suggestion.confidence:.2f} | {suggestion.reasoning}")
    
    # Show recursive improvement
    print(f"\n🔄 RECURSIVE SELF-IMPROVEMENT")
    print("-" * 40)
    
    improvement_results = enhancer.recursive_self_improvement()
    print(f"Improvement depth: {improvement_results['depth']}")
    print(f"Performance score: {improvement_results['performance_gains']['overall_score']:.2f}")
    print(f"Improvements made: {len(improvement_results['improvements_made'])}")
    
    # Show strategy weights
    print(f"\n⚖️ ENHANCEMENT STRATEGY WEIGHTS")
    print("-" * 40)
    
    for enhancement_type, strategy in enhancer.enhancement_strategies.items():
        print(f"{enhancement_type.value}: weight={strategy['weight']:.2f}, success_rate={strategy['success_rate']:.2f}")
    
    # Save enhanced state
    enhancer.save_enhancement_state()
    
    # Summary
    print(f"\n📈 ENHANCEMENT SUMMARY")
    print("=" * 60)
    print(f"✅ Analyzed {len(active_todos)} todos")
    print(f"💡 Generated {len(suggestions)} enhancement suggestions")
    print(f"🎯 Average quality score: {sum(r['overall_score'] for r in quality_results) / len(quality_results):.2f}")
    print(f"🔝 Highest confidence suggestion: {max(suggestions, key=lambda x: x.confidence).confidence:.2f}")
    print(f"📊 Enhancement types identified: {len(by_type)}")
    
    # Show next steps
    print(f"\n🎯 RECOMMENDED NEXT STEPS")
    print("-" * 40)
    
    # Find todo with lowest quality score
    lowest_quality = min(quality_results, key=lambda x: x['overall_score'])
    print(f"1. Improve Todo {lowest_quality['id']} (quality score: {lowest_quality['overall_score']:.2f})")
    
    # Find highest confidence suggestion
    best_suggestion = max(suggestions, key=lambda x: x.confidence)
    print(f"2. Apply {best_suggestion.type.value} to Todo {best_suggestion.todo_id} (confidence: {best_suggestion.confidence:.2f})")
    
    print(f"3. Run autonomous enhancement cycle with task-master integration")
    
    print(f"\n🏁 Enhancement engine run completed successfully!")
    
    return {
        'todos_analyzed': len(active_todos),
        'suggestions_generated': len(suggestions),
        'quality_results': quality_results,
        'enhancement_suggestions': by_type,
        'improvement_results': improvement_results
    }

if __name__ == "__main__":
    run_focused_enhancement()