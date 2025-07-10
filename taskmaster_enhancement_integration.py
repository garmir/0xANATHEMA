#!/usr/bin/env python3
"""
Task-Master Enhancement Integration
Integrates the Recursive Todo Enhancement Engine with the existing task-master system
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile

# Import our enhancement engine
from recursive_todo_enhancement_engine import (
    RecursiveTodoEnhancer, 
    EnhancementSuggestion, 
    EnhancementType
)

class TaskMasterIntegration:
    """Integration layer between the enhancement engine and task-master system"""
    
    def __init__(self, tasks_file: str = ".taskmaster/tasks/tasks.json"):
        self.tasks_file = tasks_file
        self.enhancer = RecursiveTodoEnhancer()
        self.enhancer.load_enhancement_state()
        
        # Configuration for integration
        self.auto_apply_enabled = False
        self.auto_apply_threshold = 0.9
        self.enhancement_log_file = ".taskmaster/enhancement_log.json"
        
    def load_taskmaster_todos(self) -> List[Dict[str, Any]]:
        """Load todos from the task-master system"""
        try:
            with open(self.tasks_file, 'r') as f:
                data = json.load(f)
                
            # Extract todos from the task-master format
            todos = []
            
            # Handle task-master's nested structure
            if 'master' in data and 'tasks' in data['master']:
                for task in data['master']['tasks']:
                    # Convert task-master format to our format
                    todo = {
                        'id': str(task.get('id', '')),
                        'content': task.get('title', task.get('description', '')),
                        'status': task.get('status', 'pending'),
                        'priority': task.get('priority', 'medium'),
                        'dependencies': task.get('dependencies', []),
                        'details': task.get('details', ''),
                        'testStrategy': task.get('testStrategy', '')
                    }
                    todos.append(todo)
                    
                    # Add subtasks if they exist
                    for subtask in task.get('subtasks', []):
                        subtodo = {
                            'id': str(subtask.get('id', '')),
                            'content': subtask.get('title', subtask.get('description', '')),
                            'status': subtask.get('status', 'pending'),
                            'priority': subtask.get('priority', 'medium'),
                            'dependencies': subtask.get('dependencies', []),
                            'details': subtask.get('details', ''),
                            'testStrategy': subtask.get('testStrategy', ''),
                            'parent_id': str(task.get('id', ''))
                        }
                        todos.append(subtodo)
            
            return todos
            
        except Exception as e:
            print(f"Error loading todos from {self.tasks_file}: {e}")
            return []
    
    def convert_to_claude_todowrite_format(self, todos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert todos to Claude TodoWrite format"""
        claude_todos = []
        
        for todo in todos:
            claude_todo = {
                'id': todo.get('id', ''),
                'content': todo.get('content', ''),
                'status': self._map_status_to_claude(todo.get('status', 'pending')),
                'priority': todo.get('priority', 'medium')
            }
            claude_todos.append(claude_todo)
        
        return claude_todos
    
    def _map_status_to_claude(self, taskmaster_status: str) -> str:
        """Map task-master status to Claude TodoWrite status"""
        status_mapping = {
            'pending': 'pending',
            'in-progress': 'in_progress', 
            'done': 'completed',
            'completed': 'completed',
            'blocked': 'pending',
            'deferred': 'pending',
            'cancelled': 'completed'
        }
        return status_mapping.get(taskmaster_status, 'pending')
    
    def analyze_taskmaster_todos(self) -> Dict[str, Any]:
        """Analyze todos from task-master and generate enhancement report"""
        print("Loading todos from task-master system...")
        todos = self.load_taskmaster_todos()
        
        if not todos:
            return {"error": "No todos found or unable to load todos"}
        
        print(f"Analyzing {len(todos)} todos...")
        
        # Generate enhancement suggestions
        suggestions = self.enhancer.generate_enhancement_suggestions(todos)
        
        # Analyze quality metrics
        quality_analysis = {}
        for todo in todos:
            metrics = self.enhancer.analyze_todo_quality(todo)
            quality_analysis[todo['id']] = {
                'content': todo['content'][:50] + '...' if len(todo['content']) > 50 else todo['content'],
                'overall_score': metrics.overall_score,
                'clarity_score': metrics.clarity_score,
                'actionability_score': metrics.actionability_score,
                'specificity_score': metrics.specificity_score,
                'completeness_score': metrics.completeness_score
            }
        
        # Group suggestions by type
        suggestions_by_type = {}
        for suggestion in suggestions:
            type_name = suggestion.type.value
            if type_name not in suggestions_by_type:
                suggestions_by_type[type_name] = []
            suggestions_by_type[type_name].append({
                'todo_id': suggestion.todo_id,
                'description': suggestion.description,
                'confidence': suggestion.confidence,
                'reasoning': suggestion.reasoning,
                'suggested_change': suggestion.suggested_change
            })
        
        # Perform recursive self-improvement
        print("Performing recursive self-improvement...")
        improvement_results = self.enhancer.recursive_self_improvement()
        
        analysis_report = {
            'timestamp': datetime.now().isoformat(),
            'total_todos': len(todos),
            'quality_analysis': quality_analysis,
            'enhancement_suggestions': suggestions_by_type,
            'suggestions_count': len(suggestions),
            'improvement_results': improvement_results,
            'overall_quality_score': sum(q['overall_score'] for q in quality_analysis.values()) / len(quality_analysis) if quality_analysis else 0
        }
        
        # Save analysis report
        self._save_analysis_report(analysis_report)
        
        return analysis_report
    
    def _save_analysis_report(self, report: Dict[str, Any]) -> str:
        """Save analysis report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f".taskmaster/enhancement_reports/analysis_{timestamp}.json"
        
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis report saved to {report_file}")
        return report_file
    
    def apply_enhancement_suggestions(self, suggestions: List[EnhancementSuggestion], 
                                    auto_apply: bool = False) -> Dict[str, Any]:
        """Apply enhancement suggestions to the task-master system"""
        results = {
            'applied_count': 0,
            'skipped_count': 0,
            'errors': [],
            'applied_suggestions': []
        }
        
        todos = self.load_taskmaster_todos()
        if not todos:
            results['errors'].append("Could not load todos from task-master")
            return results
        
        for suggestion in suggestions:
            try:
                # Check if we should auto-apply
                should_apply = (auto_apply and suggestion.confidence >= self.auto_apply_threshold) or not auto_apply
                
                if should_apply:
                    success = self._apply_single_suggestion(suggestion, todos)
                    if success:
                        results['applied_count'] += 1
                        results['applied_suggestions'].append({
                            'type': suggestion.type.value,
                            'todo_id': suggestion.todo_id,
                            'description': suggestion.description
                        })
                    else:
                        results['errors'].append(f"Failed to apply suggestion for todo {suggestion.todo_id}")
                else:
                    results['skipped_count'] += 1
                    
            except Exception as e:
                results['errors'].append(f"Error applying suggestion {suggestion.id}: {str(e)}")
        
        # Log enhancement session
        self._log_enhancement_session(results)
        
        return results
    
    def _apply_single_suggestion(self, suggestion: EnhancementSuggestion, todos: List[Dict[str, Any]]) -> bool:
        """Apply a single enhancement suggestion"""
        # Find the target todo
        target_todo = None
        for todo in todos:
            if todo['id'] == suggestion.todo_id:
                target_todo = todo
                break
        
        if not target_todo:
            return False
        
        try:
            # Apply different types of enhancements
            if suggestion.type == EnhancementType.CLARITY_IMPROVEMENT:
                return self._apply_clarity_improvement(suggestion, target_todo)
            elif suggestion.type == EnhancementType.TASK_DECOMPOSITION:
                return self._apply_task_decomposition(suggestion, target_todo)
            elif suggestion.type == EnhancementType.PRIORITY_ADJUSTMENT:
                return self._apply_priority_adjustment(suggestion, target_todo)
            elif suggestion.type == EnhancementType.DEPENDENCY_ADDITION:
                return self._apply_dependency_addition(suggestion, target_todo)
            elif suggestion.type == EnhancementType.CONTEXT_ENRICHMENT:
                return self._apply_context_enrichment(suggestion, target_todo)
            else:
                print(f"Unknown enhancement type: {suggestion.type}")
                return False
                
        except Exception as e:
            print(f"Error applying suggestion: {e}")
            return False
    
    def _apply_clarity_improvement(self, suggestion: EnhancementSuggestion, todo: Dict[str, Any]) -> bool:
        """Apply clarity improvement to a todo via task-master CLI"""
        if 'content' in suggestion.suggested_change:
            new_content = suggestion.suggested_change['content']
            
            # Use task-master update-task command
            command = [
                'task-master', 'update-task', 
                f'--id={todo["id"]}',
                f'--prompt=Enhanced clarity: {new_content}'
            ]
            
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                print(f"Applied clarity improvement to todo {todo['id']}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error applying clarity improvement: {e}")
                return False
        
        return False
    
    def _apply_task_decomposition(self, suggestion: EnhancementSuggestion, todo: Dict[str, Any]) -> bool:
        """Apply task decomposition to a todo"""
        if 'subtasks' in suggestion.suggested_change:
            subtasks = suggestion.suggested_change['subtasks']
            
            # Use task-master expand command to create subtasks
            command = [
                'task-master', 'expand',
                f'--id={todo["id"]}',
                '--force'
            ]
            
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                print(f"Applied task decomposition to todo {todo['id']}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error applying task decomposition: {e}")
                return False
        
        return False
    
    def _apply_priority_adjustment(self, suggestion: EnhancementSuggestion, todo: Dict[str, Any]) -> bool:
        """Apply priority adjustment to a todo"""
        if 'priority' in suggestion.suggested_change:
            new_priority = suggestion.suggested_change['priority']
            
            # Use task-master update-task command for priority
            command = [
                'task-master', 'update-task',
                f'--id={todo["id"]}',
                f'--prompt=Adjusted priority to {new_priority} based on content analysis'
            ]
            
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                print(f"Applied priority adjustment to todo {todo['id']}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error applying priority adjustment: {e}")
                return False
        
        return False
    
    def _apply_dependency_addition(self, suggestion: EnhancementSuggestion, todo: Dict[str, Any]) -> bool:
        """Apply dependency addition to a todo"""
        if 'add_dependency' in suggestion.suggested_change:
            dependency_id = suggestion.suggested_change['add_dependency']
            
            # Use task-master add-dependency command
            command = [
                'task-master', 'add-dependency',
                f'--id={todo["id"]}',
                f'--depends-on={dependency_id}'
            ]
            
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                print(f"Added dependency to todo {todo['id']}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error adding dependency: {e}")
                return False
        
        return False
    
    def _apply_context_enrichment(self, suggestion: EnhancementSuggestion, todo: Dict[str, Any]) -> bool:
        """Apply context enrichment to a todo"""
        if 'content' in suggestion.suggested_change:
            enriched_content = suggestion.suggested_change['content']
            
            # Use task-master update-task command
            command = [
                'task-master', 'update-task',
                f'--id={todo["id"]}',
                f'--prompt=Enhanced with context: {enriched_content}'
            ]
            
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                print(f"Applied context enrichment to todo {todo['id']}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error applying context enrichment: {e}")
                return False
        
        return False
    
    def _log_enhancement_session(self, results: Dict[str, Any]) -> None:
        """Log enhancement session results"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_results': results,
            'enhancer_performance': self.enhancer._calculate_overall_performance()
        }
        
        # Load existing log
        log_data = []
        if os.path.exists(self.enhancement_log_file):
            try:
                with open(self.enhancement_log_file, 'r') as f:
                    log_data = json.load(f)
            except:
                log_data = []
        
        # Append new entry
        log_data.append(log_entry)
        
        # Keep only last 100 entries
        log_data = log_data[-100:]
        
        # Save log
        os.makedirs(os.path.dirname(self.enhancement_log_file), exist_ok=True)
        with open(self.enhancement_log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def generate_enhancement_report(self) -> str:
        """Generate a comprehensive enhancement report"""
        analysis = self.analyze_taskmaster_todos()
        
        report_lines = [
            "# Task-Master Enhancement Report",
            f"Generated: {analysis['timestamp']}",
            "",
            f"## Summary",
            f"- Total todos analyzed: {analysis['total_todos']}",
            f"- Enhancement suggestions: {analysis['suggestions_count']}",
            f"- Overall quality score: {analysis['overall_quality_score']:.2f}/1.0",
            ""
        ]
        
        # Quality Analysis Section
        report_lines.extend([
            "## Quality Analysis",
            ""
        ])
        
        for todo_id, quality in analysis['quality_analysis'].items():
            report_lines.extend([
                f"### Todo {todo_id}: {quality['content']}",
                f"- Overall Score: {quality['overall_score']:.2f}",
                f"- Clarity: {quality['clarity_score']:.2f}",
                f"- Actionability: {quality['actionability_score']:.2f}",
                f"- Specificity: {quality['specificity_score']:.2f}",
                f"- Completeness: {quality['completeness_score']:.2f}",
                ""
            ])
        
        # Enhancement Suggestions Section
        report_lines.extend([
            "## Enhancement Suggestions",
            ""
        ])
        
        for enhancement_type, suggestions in analysis['enhancement_suggestions'].items():
            report_lines.extend([
                f"### {enhancement_type.replace('_', ' ').title()}",
                ""
            ])
            
            for suggestion in suggestions:
                report_lines.extend([
                    f"- **Todo {suggestion['todo_id']}**: {suggestion['description']}",
                    f"  - Confidence: {suggestion['confidence']:.2f}",
                    f"  - Reasoning: {suggestion['reasoning']}",
                    ""
                ])
        
        # Improvement Results Section
        improvement = analysis['improvement_results']
        report_lines.extend([
            "## Recursive Self-Improvement Results",
            f"- Recursive depth: {improvement['depth']}",
            f"- Overall performance: {improvement['performance_gains']['overall_score']:.2f}",
            f"- Improvements made: {len(improvement['improvements_made'])}",
            ""
        ])
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f".taskmaster/enhancement_reports/report_{timestamp}.md"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Enhancement report saved to {report_file}")
        return report_file
    
    def enable_autonomous_enhancement(self, auto_apply_threshold: float = 0.9) -> None:
        """Enable autonomous enhancement mode"""
        self.auto_apply_enabled = True
        self.auto_apply_threshold = auto_apply_threshold
        print(f"Autonomous enhancement enabled with threshold {auto_apply_threshold}")
    
    def disable_autonomous_enhancement(self) -> None:
        """Disable autonomous enhancement mode"""
        self.auto_apply_enabled = False
        print("Autonomous enhancement disabled")
    
    def run_autonomous_enhancement_cycle(self) -> Dict[str, Any]:
        """Run a complete autonomous enhancement cycle"""
        print("Starting autonomous enhancement cycle...")
        
        # Analyze current todos
        analysis = self.analyze_taskmaster_todos()
        
        if 'error' in analysis:
            return analysis
        
        # Generate suggestions
        todos = self.load_taskmaster_todos()
        suggestions = self.enhancer.generate_enhancement_suggestions(todos)
        
        # Apply high-confidence suggestions automatically
        application_results = self.apply_enhancement_suggestions(
            suggestions, 
            auto_apply=self.auto_apply_enabled
        )
        
        # Save enhanced state
        self.enhancer.save_enhancement_state()
        
        cycle_results = {
            'timestamp': datetime.now().isoformat(),
            'todos_analyzed': analysis['total_todos'],
            'suggestions_generated': len(suggestions),
            'suggestions_applied': application_results['applied_count'],
            'suggestions_skipped': application_results['skipped_count'],
            'errors': application_results['errors'],
            'overall_quality_improvement': analysis['overall_quality_score'],
            'autonomous_mode': self.auto_apply_enabled
        }
        
        print(f"Autonomous enhancement cycle completed:")
        print(f"- Analyzed {cycle_results['todos_analyzed']} todos")
        print(f"- Generated {cycle_results['suggestions_generated']} suggestions")
        print(f"- Applied {cycle_results['suggestions_applied']} suggestions")
        print(f"- Skipped {cycle_results['suggestions_skipped']} suggestions")
        
        return cycle_results

def main():
    """Main execution function for testing integration"""
    print("Initializing Task-Master Enhancement Integration...")
    
    integration = TaskMasterIntegration()
    
    try:
        # Generate comprehensive enhancement report
        report_file = integration.generate_enhancement_report()
        print(f"Enhancement report generated: {report_file}")
        
        # Run autonomous enhancement cycle
        cycle_results = integration.run_autonomous_enhancement_cycle()
        print(f"Autonomous enhancement cycle results: {cycle_results}")
        
    except Exception as e:
        print(f"Error during integration test: {e}")
        return None

if __name__ == "__main__":
    main()