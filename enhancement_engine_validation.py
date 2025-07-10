#!/usr/bin/env python3
"""
Enhancement Engine Validation and Optimization Framework
Validates and optimizes the recursive todo enhancement engine performance
"""

import json
import os
import time
import statistics
from typing import Dict, List, Any, Tuple
from datetime import datetime
import tempfile

# Import our components
from recursive_todo_enhancement_engine import (
    RecursiveTodoEnhancer, 
    EnhancementSuggestion, 
    EnhancementType,
    TodoQualityMetrics
)
from taskmaster_enhancement_integration import TaskMasterIntegration

class EnhancementEngineValidator:
    """Comprehensive validation framework for the enhancement engine"""
    
    def __init__(self):
        self.enhancer = RecursiveTodoEnhancer()
        self.integration = TaskMasterIntegration()
        self.validation_results = {}
        
        # Test datasets
        self.test_todos = self._create_test_datasets()
        
        # Performance benchmarks
        self.performance_benchmarks = {
            'quality_analysis_time': 0.1,  # seconds per todo
            'suggestion_generation_time': 0.5,  # seconds for full analysis
            'accuracy_threshold': 0.8,  # minimum accuracy for suggestions
            'precision_threshold': 0.75,  # minimum precision for enhancements
            'recall_threshold': 0.7  # minimum recall for enhancement detection
        }
    
    def _create_test_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create comprehensive test datasets"""
        return {
            'clarity_test': [
                {
                    'id': 'clarity_1',
                    'content': 'do the thing',
                    'status': 'pending',
                    'priority': 'medium',
                    'expected_improvements': ['clarity_improvement']
                },
                {
                    'id': 'clarity_2', 
                    'content': 'implement user authentication system with JWT tokens and bcrypt password hashing',
                    'status': 'pending',
                    'priority': 'high',
                    'expected_improvements': []  # Already clear
                }
            ],
            'decomposition_test': [
                {
                    'id': 'decomp_1',
                    'content': 'implement complete user management system with authentication, authorization, profile management, and admin dashboard',
                    'status': 'pending',
                    'priority': 'high',
                    'expected_improvements': ['task_decomposition']
                },
                {
                    'id': 'decomp_2',
                    'content': 'fix bug in login function',
                    'status': 'pending', 
                    'priority': 'medium',
                    'expected_improvements': []  # Simple task
                }
            ],
            'dependency_test': [
                {
                    'id': 'dep_1',
                    'content': 'test user authentication system',
                    'status': 'pending',
                    'priority': 'medium',
                    'expected_improvements': ['dependency_addition']
                },
                {
                    'id': 'dep_2',
                    'content': 'implement user authentication system',
                    'status': 'pending',
                    'priority': 'high',
                    'expected_improvements': []
                }
            ],
            'priority_test': [
                {
                    'id': 'priority_1',
                    'content': 'urgent critical bug fix needed asap',
                    'status': 'pending',
                    'priority': 'medium',  # Should be high
                    'expected_improvements': ['priority_adjustment']
                },
                {
                    'id': 'priority_2',
                    'content': 'nice to have feature enhancement when time permits',
                    'status': 'pending',
                    'priority': 'high',  # Should be low
                    'expected_improvements': ['priority_adjustment']
                }
            ],
            'duplicate_test': [
                {
                    'id': 'dup_1',
                    'content': 'implement user login functionality',
                    'status': 'pending',
                    'priority': 'high',
                    'expected_improvements': ['duplicate_resolution']
                },
                {
                    'id': 'dup_2',
                    'content': 'create user authentication login feature',
                    'status': 'pending',
                    'priority': 'high',
                    'expected_improvements': ['duplicate_resolution']
                }
            ],
            'context_test': [
                {
                    'id': 'context_1',
                    'content': 'implement function',
                    'status': 'pending',
                    'priority': 'medium',
                    'expected_improvements': ['context_enrichment']
                }
            ],
            'performance_test': [
                # Large dataset for performance testing
                {
                    'id': f'perf_{i}',
                    'content': f'implement feature {i} with appropriate testing and documentation',
                    'status': 'pending',
                    'priority': 'medium'
                } for i in range(100)
            ]
        }
    
    def validate_quality_analysis(self) -> Dict[str, Any]:
        """Validate quality analysis functionality"""
        print("Validating quality analysis...")
        
        results = {
            'test_cases': 0,
            'passed': 0,
            'failed': 0,
            'performance_metrics': {},
            'detailed_results': []
        }
        
        # Test basic quality scoring
        test_cases = [
            {
                'todo': {'content': 'fix the thing', 'status': 'pending', 'priority': 'medium', 'id': 'test1'},
                'expected_score_range': (0.3, 0.7),  # Low quality
                'test_name': 'low_quality_todo'
            },
            {
                'todo': {'content': 'implement user authentication system using JWT tokens with bcrypt password hashing and proper error handling', 'status': 'pending', 'priority': 'high', 'id': 'test2'},
                'expected_score_range': (0.7, 1.0),  # High quality
                'test_name': 'high_quality_todo'
            },
            {
                'todo': {'content': '', 'status': 'pending', 'priority': 'medium', 'id': 'test3'},
                'expected_score_range': (0.0, 0.3),  # Empty content
                'test_name': 'empty_content_todo'
            }
        ]
        
        for test_case in test_cases:
            results['test_cases'] += 1
            
            start_time = time.time()
            metrics = self.enhancer.analyze_todo_quality(test_case['todo'])
            analysis_time = time.time() - start_time
            
            # Check if score is in expected range
            min_score, max_score = test_case['expected_score_range']
            score_valid = min_score <= metrics.overall_score <= max_score
            
            # Check performance
            performance_valid = analysis_time <= self.performance_benchmarks['quality_analysis_time']
            
            test_passed = score_valid and performance_valid
            
            if test_passed:
                results['passed'] += 1
            else:
                results['failed'] += 1
            
            results['detailed_results'].append({
                'test_name': test_case['test_name'],
                'passed': test_passed,
                'score': metrics.overall_score,
                'expected_range': test_case['expected_score_range'],
                'analysis_time': analysis_time,
                'metrics': {
                    'clarity': metrics.clarity_score,
                    'actionability': metrics.actionability_score,
                    'specificity': metrics.specificity_score,
                    'completeness': metrics.completeness_score
                }
            })
        
        # Calculate performance metrics
        times = [result['analysis_time'] for result in results['detailed_results']]
        results['performance_metrics'] = {
            'avg_analysis_time': statistics.mean(times),
            'max_analysis_time': max(times),
            'min_analysis_time': min(times),
            'performance_benchmark_met': max(times) <= self.performance_benchmarks['quality_analysis_time']
        }
        
        return results
    
    def validate_enhancement_suggestions(self) -> Dict[str, Any]:
        """Validate enhancement suggestion generation"""
        print("Validating enhancement suggestion generation...")
        
        results = {
            'test_categories': 0,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'accuracy_scores': {},
            'performance_metrics': {},
            'category_results': {}
        }
        
        for category, todos in self.test_todos.items():
            if category == 'performance_test':
                continue  # Skip for now, will test separately
            
            results['test_categories'] += 1
            category_result = {
                'tests': len(todos),
                'passed': 0,
                'failed': 0,
                'accuracy': 0.0,
                'suggestions_generated': 0,
                'expected_suggestions_found': 0
            }
            
            start_time = time.time()
            suggestions = self.enhancer.generate_enhancement_suggestions(todos)
            generation_time = time.time() - start_time
            
            category_result['suggestions_generated'] = len(suggestions)
            
            # Check if expected suggestions were generated
            for todo in todos:
                results['total_tests'] += 1
                expected_improvements = todo.get('expected_improvements', [])
                
                todo_suggestions = [s for s in suggestions if s.todo_id == todo['id']]
                found_types = [s.type.value for s in todo_suggestions]
                
                # Check if expected improvements were found
                expected_found = 0
                for expected in expected_improvements:
                    if expected in found_types:
                        expected_found += 1
                        category_result['expected_suggestions_found'] += 1
                
                # Test passes if all expected suggestions were found or no suggestions were expected
                test_passed = (
                    (len(expected_improvements) == 0 and len(todo_suggestions) >= 0) or
                    (expected_found == len(expected_improvements))
                )
                
                if test_passed:
                    results['passed_tests'] += 1
                    category_result['passed'] += 1
                else:
                    results['failed_tests'] += 1
                    category_result['failed'] += 1
            
            # Calculate category accuracy
            if category_result['tests'] > 0:
                category_result['accuracy'] = category_result['passed'] / category_result['tests']
            
            category_result['generation_time'] = generation_time
            results['category_results'][category] = category_result
        
        # Calculate overall accuracy
        if results['total_tests'] > 0:
            overall_accuracy = results['passed_tests'] / results['total_tests']
            results['overall_accuracy'] = overall_accuracy
            results['accuracy_benchmark_met'] = overall_accuracy >= self.performance_benchmarks['accuracy_threshold']
        
        return results
    
    def validate_performance(self) -> Dict[str, Any]:
        """Validate performance with large datasets"""
        print("Validating performance with large datasets...")
        
        performance_todos = self.test_todos['performance_test']
        
        results = {
            'dataset_size': len(performance_todos),
            'quality_analysis_performance': {},
            'suggestion_generation_performance': {},
            'memory_usage': {},
            'scalability_assessment': {}
        }
        
        # Test quality analysis performance
        start_time = time.time()
        quality_metrics = []
        for todo in performance_todos:
            metrics = self.enhancer.analyze_todo_quality(todo)
            quality_metrics.append(metrics.overall_score)
        quality_time = time.time() - start_time
        
        results['quality_analysis_performance'] = {
            'total_time': quality_time,
            'avg_time_per_todo': quality_time / len(performance_todos),
            'todos_per_second': len(performance_todos) / quality_time,
            'benchmark_met': (quality_time / len(performance_todos)) <= self.performance_benchmarks['quality_analysis_time']
        }
        
        # Test suggestion generation performance
        start_time = time.time()
        suggestions = self.enhancer.generate_enhancement_suggestions(performance_todos)
        suggestion_time = time.time() - start_time
        
        results['suggestion_generation_performance'] = {
            'total_time': suggestion_time,
            'suggestions_generated': len(suggestions),
            'suggestions_per_second': len(suggestions) / suggestion_time if suggestion_time > 0 else 0,
            'benchmark_met': suggestion_time <= self.performance_benchmarks['suggestion_generation_time']
        }
        
        # Assess scalability
        results['scalability_assessment'] = {
            'can_handle_1000_todos': results['quality_analysis_performance']['todos_per_second'] > 10,
            'can_handle_10000_todos': results['quality_analysis_performance']['todos_per_second'] > 100,
            'suggestion_efficiency': len(suggestions) / len(performance_todos) if performance_todos else 0
        }
        
        return results
    
    def validate_recursive_improvement(self) -> Dict[str, Any]:
        """Validate recursive self-improvement functionality"""
        print("Validating recursive self-improvement...")
        
        results = {
            'initial_performance': 0.0,
            'post_improvement_performance': 0.0,
            'improvement_cycles': 0,
            'performance_gain': 0.0,
            'strategy_updates': {},
            'convergence_achieved': False
        }
        
        # Get initial performance
        initial_performance = self.enhancer._calculate_overall_performance()
        results['initial_performance'] = initial_performance
        
        # Perform recursive improvement
        improvement_results = self.enhancer.recursive_self_improvement()
        results['improvement_cycles'] = improvement_results.get('depth', 0)
        results['strategy_updates'] = improvement_results.get('strategy_updates', {})
        
        # Get post-improvement performance
        post_performance = self.enhancer._calculate_overall_performance()
        results['post_improvement_performance'] = post_performance
        
        # Calculate performance gain
        if initial_performance > 0:
            results['performance_gain'] = (post_performance - initial_performance) / initial_performance
        else:
            results['performance_gain'] = post_performance
        
        # Check convergence
        results['convergence_achieved'] = abs(results['performance_gain']) < 0.01  # Less than 1% change
        
        return results
    
    def validate_integration(self) -> Dict[str, Any]:
        """Validate integration with task-master system"""
        print("Validating task-master integration...")
        
        results = {
            'can_load_todos': False,
            'can_generate_reports': False,
            'can_apply_suggestions': False,
            'data_integrity': False,
            'error_handling': False
        }
        
        try:
            # Test loading todos
            todos = self.integration.load_taskmaster_todos()
            results['can_load_todos'] = len(todos) > 0
            
            # Test report generation
            try:
                report_file = self.integration.generate_enhancement_report()
                results['can_generate_reports'] = os.path.exists(report_file)
            except Exception as e:
                print(f"Report generation test failed: {e}")
                results['can_generate_reports'] = False
            
            # Test data integrity
            if todos:
                sample_todo = todos[0]
                required_fields = ['id', 'content', 'status', 'priority']
                results['data_integrity'] = all(field in sample_todo for field in required_fields)
            
            # Test error handling
            try:
                # Try with invalid data
                invalid_integration = TaskMasterIntegration("nonexistent_file.json")
                invalid_todos = invalid_integration.load_taskmaster_todos()
                results['error_handling'] = len(invalid_todos) == 0  # Should handle gracefully
            except Exception:
                results['error_handling'] = False
            
        except Exception as e:
            print(f"Integration validation failed: {e}")
        
        return results
    
    def optimize_enhancement_strategies(self) -> Dict[str, Any]:
        """Optimize enhancement strategies based on validation results"""
        print("Optimizing enhancement strategies...")
        
        optimization_results = {
            'strategies_optimized': [],
            'performance_improvements': {},
            'configuration_updates': {}
        }
        
        # Run comprehensive analysis to gather optimization data
        sample_todos = []
        for category, todos in self.test_todos.items():
            if category != 'performance_test':
                sample_todos.extend(todos[:5])  # Sample from each category
        
        # Generate suggestions and analyze patterns
        suggestions = self.enhancer.generate_enhancement_suggestions(sample_todos)
        
        # Analyze suggestion patterns by type
        type_analysis = {}
        for suggestion in suggestions:
            type_name = suggestion.type.value
            if type_name not in type_analysis:
                type_analysis[type_name] = {
                    'count': 0,
                    'avg_confidence': 0.0,
                    'confidences': []
                }
            
            type_analysis[type_name]['count'] += 1
            type_analysis[type_name]['confidences'].append(suggestion.confidence)
        
        # Calculate averages and optimize weights
        for type_name, analysis in type_analysis.items():
            if analysis['confidences']:
                analysis['avg_confidence'] = statistics.mean(analysis['confidences'])
                
                # Optimize strategy weight based on confidence
                enhancement_type = EnhancementType(type_name)
                if enhancement_type in self.enhancer.enhancement_strategies:
                    old_weight = self.enhancer.enhancement_strategies[enhancement_type]['weight']
                    
                    # Adjust weight based on average confidence
                    confidence_factor = analysis['avg_confidence']
                    new_weight = old_weight * (0.8 + 0.4 * confidence_factor)
                    
                    self.enhancer.enhancement_strategies[enhancement_type]['weight'] = new_weight
                    
                    optimization_results['strategies_optimized'].append(type_name)
                    optimization_results['performance_improvements'][type_name] = {
                        'old_weight': old_weight,
                        'new_weight': new_weight,
                        'avg_confidence': analysis['avg_confidence'],
                        'suggestion_count': analysis['count']
                    }
        
        # Update configuration based on validation results
        config_updates = {
            'enhancement_threshold': max(0.5, min(0.8, statistics.mean([s.confidence for s in suggestions]) - 0.1)),
            'max_suggestions_per_todo': min(5, max(1, len(suggestions) // len(sample_todos) + 1))
        }
        
        for key, value in config_updates.items():
            self.enhancer.config[key] = value
        
        optimization_results['configuration_updates'] = config_updates
        
        # Save optimized state
        self.enhancer.save_enhancement_state()
        
        return optimization_results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of the entire system"""
        print("Running comprehensive validation of the enhancement engine...")
        
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {},
            'quality_analysis_validation': {},
            'enhancement_suggestions_validation': {},
            'performance_validation': {},
            'recursive_improvement_validation': {},
            'integration_validation': {},
            'optimization_results': {},
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # Run all validation tests
        validation_report['quality_analysis_validation'] = self.validate_quality_analysis()
        validation_report['enhancement_suggestions_validation'] = self.validate_enhancement_suggestions()
        validation_report['performance_validation'] = self.validate_performance()
        validation_report['recursive_improvement_validation'] = self.validate_recursive_improvement()
        validation_report['integration_validation'] = self.validate_integration()
        validation_report['optimization_results'] = self.optimize_enhancement_strategies()
        
        # Calculate overall score
        scores = []
        
        # Quality analysis score
        qa_val = validation_report['quality_analysis_validation']
        qa_score = qa_val['passed'] / qa_val['test_cases'] if qa_val['test_cases'] > 0 else 0
        scores.append(qa_score)
        
        # Enhancement suggestions score
        es_val = validation_report['enhancement_suggestions_validation']
        es_score = es_val.get('overall_accuracy', 0)
        scores.append(es_score)
        
        # Performance score
        perf_val = validation_report['performance_validation']
        perf_score = 1.0 if perf_val['quality_analysis_performance']['benchmark_met'] else 0.5
        scores.append(perf_score)
        
        # Integration score
        int_val = validation_report['integration_validation']
        int_score = sum(int_val.values()) / len(int_val) if int_val else 0
        scores.append(int_score)
        
        validation_report['overall_score'] = statistics.mean(scores)
        
        # Generate recommendations
        recommendations = []
        
        if qa_score < 0.8:
            recommendations.append("Improve quality analysis algorithms for better accuracy")
        
        if es_score < 0.8:
            recommendations.append("Enhance suggestion generation logic and pattern recognition")
        
        if not perf_val['quality_analysis_performance']['benchmark_met']:
            recommendations.append("Optimize performance for large-scale todo analysis")
        
        if int_score < 0.8:
            recommendations.append("Strengthen integration with task-master system")
        
        if validation_report['overall_score'] > 0.9:
            recommendations.append("System is performing excellently - ready for production use")
        elif validation_report['overall_score'] > 0.8:
            recommendations.append("System is performing well - minor optimizations recommended")
        else:
            recommendations.append("System needs significant improvements before production use")
        
        validation_report['recommendations'] = recommendations
        
        # Summary
        validation_report['validation_summary'] = {
            'total_tests_run': (
                qa_val['test_cases'] + 
                es_val['total_tests'] + 
                len(perf_val) + 
                len(int_val)
            ),
            'overall_score': validation_report['overall_score'],
            'performance_benchmark_met': perf_val['quality_analysis_performance']['benchmark_met'],
            'accuracy_benchmark_met': es_val.get('accuracy_benchmark_met', False),
            'system_ready_for_production': validation_report['overall_score'] > 0.8
        }
        
        return validation_report
    
    def save_validation_report(self, report: Dict[str, Any]) -> str:
        """Save validation report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f".taskmaster/validation_reports/enhancement_validation_{timestamp}.json"
        
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_file

def main():
    """Main execution function"""
    print("Starting Enhancement Engine Validation and Optimization...")
    
    validator = EnhancementEngineValidator()
    
    try:
        # Run comprehensive validation
        validation_report = validator.run_comprehensive_validation()
        
        # Save validation report
        report_file = validator.save_validation_report(validation_report)
        print(f"\nValidation report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("ENHANCEMENT ENGINE VALIDATION SUMMARY")
        print("="*70)
        
        summary = validation_report['validation_summary']
        print(f"Overall Score: {summary['overall_score']:.2f}/1.0")
        print(f"Total Tests Run: {summary['total_tests_run']}")
        print(f"Performance Benchmark Met: {summary['performance_benchmark_met']}")
        print(f"Accuracy Benchmark Met: {summary['accuracy_benchmark_met']}")
        print(f"Production Ready: {summary['system_ready_for_production']}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(validation_report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\nDetailed Results:")
        
        # Quality Analysis
        qa_results = validation_report['quality_analysis_validation']
        print(f"- Quality Analysis: {qa_results['passed']}/{qa_results['test_cases']} tests passed")
        print(f"  Avg Analysis Time: {qa_results['performance_metrics']['avg_analysis_time']:.3f}s")
        
        # Enhancement Suggestions
        es_results = validation_report['enhancement_suggestions_validation']
        print(f"- Enhancement Suggestions: {es_results['passed_tests']}/{es_results['total_tests']} tests passed")
        print(f"  Overall Accuracy: {es_results.get('overall_accuracy', 0):.2f}")
        
        # Performance
        perf_results = validation_report['performance_validation']
        print(f"- Performance: {perf_results['quality_analysis_performance']['todos_per_second']:.1f} todos/sec")
        
        # Integration
        int_results = validation_report['integration_validation']
        integration_score = sum(int_results.values()) / len(int_results) if int_results else 0
        print(f"- Integration: {integration_score:.2f} compatibility score")
        
        print("\n" + "="*70)
        
        return validation_report
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return None

if __name__ == "__main__":
    main()