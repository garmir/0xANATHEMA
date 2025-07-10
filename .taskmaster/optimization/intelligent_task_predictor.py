#!/usr/bin/env python3
"""
Intelligent Task Prediction and Auto-Generation System
AI-Powered analysis of project patterns and user behavior for task prediction
"""

import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import re
from collections import defaultdict, Counter
import statistics


@dataclass
class TaskPattern:
    """Identified task pattern"""
    pattern_id: str
    pattern_type: str  # "sequence", "dependency", "timing", "category"
    confidence: float
    frequency: int
    description: str
    example_tasks: List[str]


@dataclass
class PredictionScore:
    """Task prediction confidence score"""
    task_type: str
    confidence: float
    reasoning: str
    supporting_patterns: List[str]
    estimated_priority: str


@dataclass
class GeneratedTask:
    """Auto-generated task proposal"""
    title: str
    description: str
    details: str
    predicted_priority: str
    confidence_score: float
    suggested_dependencies: List[str]
    reasoning: str
    approval_required: bool


class PatternAnalyzer:
    """Analyzes historical tasks to identify patterns"""
    
    def __init__(self):
        self.identified_patterns = []
        self.task_categories = defaultdict(list)
        self.sequence_patterns = []
        self.timing_patterns = {}
        
    def analyze_task_history(self, tasks_data: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Analyze historical tasks to identify patterns"""
        self.identified_patterns = []
        
        # Categorize tasks
        self._categorize_tasks(tasks_data)
        
        # Analyze sequences
        self._analyze_task_sequences(tasks_data)
        
        # Analyze timing patterns
        self._analyze_timing_patterns(tasks_data)
        
        # Analyze dependency patterns
        self._analyze_dependency_patterns(tasks_data)
        
        return self.identified_patterns
    
    def _categorize_tasks(self, tasks_data: List[Dict[str, Any]]):
        """Categorize tasks by type and content"""
        category_keywords = {
            'implementation': ['implement', 'create', 'build', 'develop'],
            'testing': ['test', 'validate', 'verify', 'check'],
            'optimization': ['optimize', 'improve', 'enhance', 'performance'],
            'documentation': ['document', 'write', 'create guide', 'tutorial'],
            'integration': ['integrate', 'connect', 'combine', 'merge'],
            'monitoring': ['monitor', 'track', 'observe', 'dashboard'],
            'analysis': ['analyze', 'research', 'investigate', 'study']
        }
        
        for task in tasks_data:
            title = task.get('title', '').lower()
            description = task.get('description', '').lower()
            text = f"{title} {description}"
            
            for category, keywords in category_keywords.items():
                if any(keyword in text for keyword in keywords):
                    self.task_categories[category].append(task)
                    break
            else:
                self.task_categories['other'].append(task)
        
        # Create category patterns
        for category, tasks in self.task_categories.items():
            if len(tasks) >= 3:  # Minimum threshold for pattern
                pattern = TaskPattern(
                    pattern_id=f"category_{category}",
                    pattern_type="category",
                    confidence=min(len(tasks) / 10.0, 1.0),
                    frequency=len(tasks),
                    description=f"Tasks related to {category}",
                    example_tasks=[t.get('title', '')[:50] for t in tasks[:3]]
                )
                self.identified_patterns.append(pattern)
    
    def _analyze_task_sequences(self, tasks_data: List[Dict[str, Any]]):
        """Analyze common task sequences"""
        # Sort tasks by ID to analyze sequences
        sorted_tasks = sorted(tasks_data, key=lambda x: x.get('id', 0))
        
        # Look for 3-task sequences
        for i in range(len(sorted_tasks) - 2):
            task1 = sorted_tasks[i]
            task2 = sorted_tasks[i + 1]
            task3 = sorted_tasks[i + 2]
            
            # Extract key words from titles
            words1 = self._extract_key_words(task1.get('title', ''))
            words2 = self._extract_key_words(task2.get('title', ''))
            words3 = self._extract_key_words(task3.get('title', ''))
            
            if words1 and words2 and words3:
                sequence_pattern = f"{words1[0]} -> {words2[0]} -> {words3[0]}"
                
                pattern = TaskPattern(
                    pattern_id=f"sequence_{i}",
                    pattern_type="sequence",
                    confidence=0.6,  # Medium confidence for sequences
                    frequency=1,
                    description=f"Common sequence: {sequence_pattern}",
                    example_tasks=[task1.get('title', ''), task2.get('title', ''), task3.get('title', '')]
                )
                self.identified_patterns.append(pattern)
    
    def _analyze_timing_patterns(self, tasks_data: List[Dict[str, Any]]):
        """Analyze timing patterns in task completion"""
        # Simple timing analysis based on task IDs (proxy for time)
        completion_gaps = []
        
        completed_tasks = [t for t in tasks_data if t.get('status') == 'done']
        completed_tasks.sort(key=lambda x: x.get('id', 0))
        
        for i in range(1, len(completed_tasks)):
            gap = completed_tasks[i].get('id', 0) - completed_tasks[i-1].get('id', 0)
            completion_gaps.append(gap)
        
        if completion_gaps:
            avg_gap = statistics.mean(completion_gaps)
            
            pattern = TaskPattern(
                pattern_id="timing_completion",
                pattern_type="timing",
                confidence=0.5,
                frequency=len(completion_gaps),
                description=f"Average task completion interval: {avg_gap:.1f} tasks",
                example_tasks=[]
            )
            self.identified_patterns.append(pattern)
    
    def _analyze_dependency_patterns(self, tasks_data: List[Dict[str, Any]]):
        """Analyze dependency patterns between tasks"""
        dependency_counts = Counter()
        
        for task in tasks_data:
            dependencies = task.get('dependencies', [])
            if dependencies:
                dependency_counts[len(dependencies)] += 1
        
        if dependency_counts:
            most_common_dep_count = dependency_counts.most_common(1)[0]
            
            pattern = TaskPattern(
                pattern_id="dependency_common",
                pattern_type="dependency",
                confidence=0.7,
                frequency=most_common_dep_count[1],
                description=f"Most common dependency count: {most_common_dep_count[0]}",
                example_tasks=[]
            )
            self.identified_patterns.append(pattern)
    
    def _extract_key_words(self, text: str) -> List[str]:
        """Extract key words from task titles"""
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text.lower())
        key_words = [w for w in words if w not in stop_words and len(w) > 3]
        return key_words[:3]  # Return top 3 key words


class TrajectoryPredictor:
    """Predicts development trajectory using ML-like approaches"""
    
    def __init__(self):
        self.trajectory_model = None
        self.trend_analysis = {}
        
    def predict_trajectory(self, tasks_data: List[Dict[str, Any]], 
                          patterns: List[TaskPattern]) -> Dict[str, Any]:
        """Predict project development trajectory"""
        
        # Analyze task complexity trend
        complexity_trend = self._analyze_complexity_trend(tasks_data)
        
        # Analyze category distribution trend
        category_trend = self._analyze_category_trend(tasks_data, patterns)
        
        # Analyze priority trend
        priority_trend = self._analyze_priority_trend(tasks_data)
        
        # Generate trajectory prediction
        trajectory = {
            'complexity_trend': complexity_trend,
            'category_trend': category_trend,
            'priority_trend': priority_trend,
            'predicted_next_phase': self._predict_next_phase(complexity_trend, category_trend),
            'confidence': self._calculate_trajectory_confidence(tasks_data)
        }
        
        return trajectory
    
    def _analyze_complexity_trend(self, tasks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the trend in task complexity"""
        # Proxy complexity based on description length and keywords
        complexity_scores = []
        
        for task in sorted(tasks_data, key=lambda x: x.get('id', 0)):
            description = task.get('description', '') + task.get('details', '')
            
            # Simple complexity scoring
            complexity = len(description) / 100.0  # Length factor
            
            # Keyword complexity indicators
            complex_keywords = ['advanced', 'comprehensive', 'integration', 'optimization', 'ai', 'machine learning']
            for keyword in complex_keywords:
                if keyword in description.lower():
                    complexity += 0.5
            
            complexity_scores.append(min(complexity, 5.0))  # Cap at 5.0
        
        if len(complexity_scores) >= 3:
            # Simple trend analysis
            recent_avg = statistics.mean(complexity_scores[-5:]) if len(complexity_scores) >= 5 else statistics.mean(complexity_scores)
            overall_avg = statistics.mean(complexity_scores)
            
            trend = "increasing" if recent_avg > overall_avg else "stable" if abs(recent_avg - overall_avg) < 0.5 else "decreasing"
        else:
            trend = "stable"
            recent_avg = 2.0
            overall_avg = 2.0
        
        return {
            'trend': trend,
            'recent_average': recent_avg,
            'overall_average': overall_avg,
            'scores': complexity_scores
        }
    
    def _analyze_category_trend(self, tasks_data: List[Dict[str, Any]], 
                               patterns: List[TaskPattern]) -> Dict[str, Any]:
        """Analyze trending task categories"""
        category_patterns = [p for p in patterns if p.pattern_type == "category"]
        
        if not category_patterns:
            return {'trending_category': 'implementation', 'confidence': 0.5}
        
        # Find most frequent category
        top_category = max(category_patterns, key=lambda x: x.frequency)
        
        return {
            'trending_category': top_category.pattern_id.replace('category_', ''),
            'confidence': top_category.confidence,
            'frequency': top_category.frequency
        }
    
    def _analyze_priority_trend(self, tasks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze priority distribution trends"""
        priority_counts = Counter(task.get('priority', 'medium') for task in tasks_data)
        
        most_common = priority_counts.most_common(1)[0] if priority_counts else ('medium', 1)
        
        return {
            'dominant_priority': most_common[0],
            'distribution': dict(priority_counts)
        }
    
    def _predict_next_phase(self, complexity_trend: Dict, category_trend: Dict) -> str:
        """Predict the next development phase"""
        if complexity_trend['trend'] == 'increasing':
            if category_trend['trending_category'] == 'implementation':
                return 'optimization'
            elif category_trend['trending_category'] == 'testing':
                return 'deployment'
            else:
                return 'enhancement'
        else:
            if category_trend['trending_category'] == 'documentation':
                return 'maintenance'
            else:
                return 'feature_development'
    
    def _calculate_trajectory_confidence(self, tasks_data: List[Dict[str, Any]]) -> float:
        """Calculate confidence in trajectory prediction"""
        base_confidence = 0.5
        
        # More tasks = higher confidence
        task_count_factor = min(len(tasks_data) / 20.0, 0.3)
        
        # Completed tasks factor
        completed_count = sum(1 for task in tasks_data if task.get('status') == 'done')
        completion_factor = min(completed_count / len(tasks_data), 0.2) if tasks_data else 0
        
        return min(base_confidence + task_count_factor + completion_factor, 1.0)


class TaskAutoGenerator:
    """Auto-generates new tasks based on patterns and predictions"""
    
    def __init__(self):
        self.generation_templates = self._load_templates()
        self.confidence_threshold = 0.6
        
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load task generation templates"""
        return {
            'implementation': {
                'title_template': "Implement {feature} for {domain}",
                'description_template': "Create {feature} functionality that {purpose}",
                'common_features': ['advanced analytics', 'user interface', 'API endpoints', 'data processing'],
                'common_purposes': ['improves performance', 'enhances user experience', 'increases efficiency', 'provides insights']
            },
            'testing': {
                'title_template': "Create {test_type} tests for {component}",
                'description_template': "Develop comprehensive {test_type} testing suite for {component}",
                'test_types': ['unit', 'integration', 'performance', 'security'],
                'components': ['API', 'user interface', 'data layer', 'authentication system']
            },
            'optimization': {
                'title_template': "Optimize {component} for {aspect}",
                'description_template': "Improve {component} performance focusing on {aspect}",
                'aspects': ['speed', 'memory usage', 'scalability', 'reliability']
            },
            'documentation': {
                'title_template': "Create {doc_type} documentation for {feature}",
                'description_template': "Develop comprehensive {doc_type} covering {feature}",
                'doc_types': ['user guide', 'API documentation', 'technical specification', 'tutorial']
            }
        }
    
    def generate_tasks(self, patterns: List[TaskPattern], 
                      trajectory: Dict[str, Any], 
                      existing_tasks: List[Dict[str, Any]]) -> List[GeneratedTask]:
        """Generate new tasks based on analysis"""
        
        generated_tasks = []
        
        # Generate based on trending category
        trending_category = trajectory.get('category_trend', {}).get('trending_category', 'implementation')
        if trending_category in self.generation_templates:
            task = self._generate_category_task(trending_category, trajectory)
            if task:
                generated_tasks.append(task)
        
        # Generate based on predicted next phase
        next_phase = trajectory.get('predicted_next_phase', 'feature_development')
        if next_phase in self.generation_templates:
            task = self._generate_phase_task(next_phase, trajectory)
            if task:
                generated_tasks.append(task)
        
        # Generate complementary tasks
        complementary_task = self._generate_complementary_task(existing_tasks, patterns)
        if complementary_task:
            generated_tasks.append(complementary_task)
        
        return generated_tasks
    
    def _generate_category_task(self, category: str, trajectory: Dict[str, Any]) -> Optional[GeneratedTask]:
        """Generate task based on trending category"""
        template = self.generation_templates.get(category)
        if not template:
            return None
        
        import random
        
        if category == 'implementation':
            feature = random.choice(template['common_features'])
            purpose = random.choice(template['common_purposes'])
            title = f"Implement {feature} system"
            description = f"Create {feature} functionality that {purpose}"
            details = f"Build comprehensive {feature} with proper error handling, logging, and integration capabilities"
        
        elif category == 'testing':
            test_type = random.choice(template['test_types'])
            component = random.choice(template['components'])
            title = f"Create {test_type} tests for {component}"
            description = f"Develop comprehensive {test_type} testing suite for {component}"
            details = f"Implement {test_type} tests covering all major functionality and edge cases"
        
        elif category == 'optimization':
            aspect = random.choice(template['aspects'])
            title = f"Optimize system for {aspect}"
            description = f"Improve system performance focusing on {aspect}"
            details = f"Analyze and optimize current implementation to enhance {aspect} with measurable improvements"
        
        else:
            return None
        
        confidence = trajectory.get('confidence', 0.5) * 0.8  # Slightly lower for generated tasks
        
        return GeneratedTask(
            title=title,
            description=description,
            details=details,
            predicted_priority=self._predict_priority(trajectory),
            confidence_score=confidence,
            suggested_dependencies=[],
            reasoning=f"Generated based on trending category: {category}",
            approval_required=confidence < self.confidence_threshold
        )
    
    def _generate_phase_task(self, phase: str, trajectory: Dict[str, Any]) -> Optional[GeneratedTask]:
        """Generate task based on predicted next phase"""
        phase_templates = {
            'optimization': {
                'title': "Implement system-wide performance optimization",
                'description': "Optimize overall system performance and resource utilization",
                'details': "Conduct comprehensive performance analysis and implement optimizations across all system components"
            },
            'deployment': {
                'title': "Setup production deployment pipeline",
                'description': "Create automated deployment and monitoring infrastructure",
                'details': "Implement CI/CD pipeline with monitoring, logging, and rollback capabilities"
            },
            'enhancement': {
                'title': "Enhance user experience and interface",
                'description': "Improve user interface and interaction patterns",
                'details': "Redesign and enhance user interface based on usage analytics and feedback"
            },
            'maintenance': {
                'title': "Implement maintenance and monitoring tools",
                'description': "Create tools for ongoing system maintenance and health monitoring",
                'details': "Build comprehensive maintenance toolkit with automated health checks and alerts"
            }
        }
        
        template = phase_templates.get(phase)
        if not template:
            return None
        
        confidence = trajectory.get('confidence', 0.5) * 0.9
        
        return GeneratedTask(
            title=template['title'],
            description=template['description'],
            details=template['details'],
            predicted_priority=self._predict_priority(trajectory),
            confidence_score=confidence,
            suggested_dependencies=[],
            reasoning=f"Generated for predicted next phase: {phase}",
            approval_required=confidence < self.confidence_threshold
        )
    
    def _generate_complementary_task(self, existing_tasks: List[Dict[str, Any]], 
                                   patterns: List[TaskPattern]) -> Optional[GeneratedTask]:
        """Generate complementary task based on gaps in existing tasks"""
        
        # Find missing categories
        existing_categories = set()
        for task in existing_tasks:
            title = task.get('title', '').lower()
            if 'test' in title:
                existing_categories.add('testing')
            elif 'document' in title:
                existing_categories.add('documentation')
            elif 'monitor' in title:
                existing_categories.add('monitoring')
            elif 'optim' in title:
                existing_categories.add('optimization')
        
        all_categories = {'testing', 'documentation', 'monitoring', 'optimization'}
        missing_categories = all_categories - existing_categories
        
        if missing_categories:
            import random
            missing_category = random.choice(list(missing_categories))
            
            task_templates = {
                'testing': {
                    'title': "Implement comprehensive testing framework",
                    'description': "Create full testing suite for system validation",
                    'details': "Build comprehensive testing framework covering unit, integration, and performance tests"
                },
                'documentation': {
                    'title': "Create comprehensive system documentation",
                    'description': "Document system architecture and usage guidelines",
                    'details': "Create detailed documentation covering system design, API reference, and user guides"
                },
                'monitoring': {
                    'title': "Implement advanced monitoring and alerting",
                    'description': "Create comprehensive monitoring and alerting system",
                    'details': "Build monitoring dashboard with real-time metrics, alerts, and historical analysis"
                }
            }
            
            template = task_templates.get(missing_category)
            if template:
                return GeneratedTask(
                    title=template['title'],
                    description=template['description'],
                    details=template['details'],
                    predicted_priority='medium',
                    confidence_score=0.7,
                    suggested_dependencies=[],
                    reasoning=f"Generated to address missing {missing_category} capabilities",
                    approval_required=False
                )
        
        return None
    
    def _predict_priority(self, trajectory: Dict[str, Any]) -> str:
        """Predict task priority based on trajectory"""
        priority_dist = trajectory.get('priority_trend', {}).get('distribution', {})
        
        if not priority_dist:
            return 'medium'
        
        # Return most common priority
        return max(priority_dist.keys(), key=lambda x: priority_dist[x])


class IntelligentTaskSystem:
    """Main intelligent task prediction and generation system"""
    
    def __init__(self, tasks_file_path: str):
        self.tasks_file_path = Path(tasks_file_path)
        self.pattern_analyzer = PatternAnalyzer()
        self.trajectory_predictor = TrajectoryPredictor()
        self.task_generator = TaskAutoGenerator()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('.taskmaster/logs/task_predictor.log'),
                logging.StreamHandler()
            ]
        )
    
    def analyze_and_predict(self) -> Dict[str, Any]:
        """Perform complete analysis and prediction"""
        # Load task data
        tasks_data = self._load_tasks_data()
        
        if not tasks_data:
            return {"error": "No task data available"}
        
        # Analyze patterns
        patterns = self.pattern_analyzer.analyze_task_history(tasks_data)
        
        # Predict trajectory
        trajectory = self.trajectory_predictor.predict_trajectory(tasks_data, patterns)
        
        # Generate new tasks
        generated_tasks = self.task_generator.generate_tasks(patterns, trajectory, tasks_data)
        
        # Create comprehensive report
        analysis_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'input_data': {
                'total_tasks': len(tasks_data),
                'completed_tasks': sum(1 for t in tasks_data if t.get('status') == 'done'),
                'task_count_by_priority': self._count_by_priority(tasks_data)
            },
            'identified_patterns': [asdict(p) for p in patterns],
            'trajectory_prediction': trajectory,
            'generated_tasks': [asdict(t) for t in generated_tasks],
            'recommendations': self._generate_recommendations(patterns, trajectory, generated_tasks)
        }
        
        # Save analysis
        self._save_analysis(analysis_report)
        
        return analysis_report
    
    def _load_tasks_data(self) -> List[Dict[str, Any]]:
        """Load tasks data from file"""
        try:
            with open(self.tasks_file_path, 'r') as f:
                data = json.load(f)
            
            # Extract tasks from master context
            master_data = data.get('master', {})
            return master_data.get('tasks', [])
            
        except Exception as e:
            logging.error(f"Error loading tasks data: {e}")
            return []
    
    def _count_by_priority(self, tasks_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count tasks by priority"""
        counts = Counter(task.get('priority', 'medium') for task in tasks_data)
        return dict(counts)
    
    def _generate_recommendations(self, patterns: List[TaskPattern], 
                                trajectory: Dict[str, Any], 
                                generated_tasks: List[GeneratedTask]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Pattern-based recommendations
        if len(patterns) >= 5:
            recommendations.append("Strong patterns detected - consider creating task templates for efficiency")
        
        # Trajectory-based recommendations
        next_phase = trajectory.get('predicted_next_phase', '')
        if next_phase:
            recommendations.append(f"Focus on {next_phase.replace('_', ' ')} activities in upcoming sprints")
        
        # Generated tasks recommendations
        high_confidence_tasks = [t for t in generated_tasks if t.confidence_score > 0.7]
        if high_confidence_tasks:
            recommendations.append(f"Consider implementing {len(high_confidence_tasks)} high-confidence predicted tasks")
        
        # Complexity recommendations
        complexity_trend = trajectory.get('complexity_trend', {}).get('trend', '')
        if complexity_trend == 'increasing':
            recommendations.append("Task complexity is increasing - consider breaking down future tasks into smaller pieces")
        
        return recommendations
    
    def _save_analysis(self, analysis_report: Dict[str, Any]):
        """Save analysis report"""
        output_path = Path('.taskmaster/reports/intelligent_task_analysis.json')
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis_report, f, indent=2, default=str)
        
        logging.info(f"Analysis saved to {output_path}")


def main():
    """Main execution function"""
    print("Intelligent Task Prediction and Auto-Generation System")
    print("=" * 55)
    
    # Initialize system
    tasks_file = ".taskmaster/tasks/tasks.json"
    system = IntelligentTaskSystem(tasks_file)
    
    try:
        # Perform analysis and prediction
        print("Analyzing task patterns and predicting future tasks...")
        analysis_report = system.analyze_and_predict()
        
        if "error" in analysis_report:
            print(f"✗ Error: {analysis_report['error']}")
            return False
        
        # Display results
        print(f"✓ Analysis completed successfully")
        print(f"✓ Total tasks analyzed: {analysis_report['input_data']['total_tasks']}")
        print(f"✓ Patterns identified: {len(analysis_report['identified_patterns'])}")
        print(f"✓ Tasks generated: {len(analysis_report['generated_tasks'])}")
        
        # Show predictions
        trajectory = analysis_report['trajectory_prediction']
        print(f"✓ Predicted next phase: {trajectory.get('predicted_next_phase', 'unknown')}")
        print(f"✓ Trajectory confidence: {trajectory.get('confidence', 0):.1%}")
        
        # Show generated tasks
        generated_tasks = analysis_report['generated_tasks']
        if generated_tasks:
            print("\n📋 Generated Task Suggestions:")
            for i, task in enumerate(generated_tasks[:3], 1):
                print(f"  {i}. {task['title']} (confidence: {task['confidence_score']:.1%})")
        
        # Show recommendations
        recommendations = analysis_report['recommendations']
        if recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print(f"\n✓ Detailed analysis saved to: .taskmaster/reports/intelligent_task_analysis.json")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)