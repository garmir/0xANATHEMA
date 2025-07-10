#!/usr/bin/env python3
"""
Comprehensive Task Complexity Analyzer
Analyzes computational complexity of tasks and optimizations
"""

import json
import math
import time
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

@dataclass
class ComplexityAnalysis:
    """Complexity analysis result"""
    task_id: str
    time_complexity: str
    space_complexity: str
    computational_weight: float
    optimization_potential: float
    bottleneck_analysis: List[str]

class TaskComplexityAnalyzer:
    """Comprehensive complexity analyzer"""
    
    def analyze_all_tasks(self) -> List[ComplexityAnalysis]:
        """Analyze complexity of all tasks"""
        tasks_file = Path('.taskmaster/tasks/tasks.json')
        if not tasks_file.exists():
            return []
        
        with open(tasks_file, 'r') as f:
            data = json.load(f)
        
        tasks = data.get('master', {}).get('tasks', [])
        analyses = []
        
        for task in tasks:
            analysis = self._analyze_task_complexity(task)
            analyses.append(analysis)
        
        # Save analysis results
        self._save_analysis_results(analyses)
        return analyses
    
    def _analyze_task_complexity(self, task: Dict[str, Any]) -> ComplexityAnalysis:
        """Analyze individual task complexity"""
        title = task.get('title', '')
        details = task.get('details', '')
        
        # Analyze based on keywords and patterns
        time_complexity = self._determine_time_complexity(title, details)
        space_complexity = self._determine_space_complexity(title, details)
        weight = self._calculate_computational_weight(title, details)
        optimization = self._assess_optimization_potential(title, details)
        bottlenecks = self._identify_bottlenecks(title, details)
        
        return ComplexityAnalysis(
            task_id=str(task.get('id', '')),
            time_complexity=time_complexity,
            space_complexity=space_complexity,
            computational_weight=weight,
            optimization_potential=optimization,
            bottleneck_analysis=bottlenecks
        )
    
    def _determine_time_complexity(self, title: str, details: str) -> str:
        """Determine time complexity classification"""
        text = (title + ' ' + details).lower()
        
        if 'optimization' in text or 'algorithm' in text:
            if 'sqrt' in text or 'square root' in text:
                return 'O(√n)'
            elif 'log' in text:
                return 'O(log n)'
            elif 'recursive' in text:
                return 'O(n log n)'
            else:
                return 'O(n)'
        elif 'validation' in text or 'test' in text:
            return 'O(n)'
        elif 'generation' in text or 'creation' in text:
            return 'O(n)'
        else:
            return 'O(1)'
    
    def _determine_space_complexity(self, title: str, details: str) -> str:
        """Determine space complexity classification"""
        text = (title + ' ' + details).lower()
        
        if 'sqrt' in text or 'square root' in text:
            return 'O(√n)'
        elif 'tree' in text and 'log' in text:
            return 'O(log n)'
        elif 'recursive' in text or 'decomposition' in text:
            return 'O(n)'
        else:
            return 'O(1)'
    
    def _calculate_computational_weight(self, title: str, details: str) -> float:
        """Calculate computational weight (0.0 to 1.0)"""
        text = (title + ' ' + details).lower()
        
        weight = 0.1  # Base weight
        
        # Add weight based on complexity indicators
        if 'comprehensive' in text: weight += 0.3
        if 'advanced' in text: weight += 0.2
        if 'optimization' in text: weight += 0.2
        if 'analysis' in text: weight += 0.15
        if 'validation' in text: weight += 0.1
        if 'generation' in text: weight += 0.1
        
        return min(weight, 1.0)
    
    def _assess_optimization_potential(self, title: str, details: str) -> float:
        """Assess optimization potential (0.0 to 1.0)"""
        text = (title + ' ' + details).lower()
        
        potential = 0.0
        
        if 'optimization' not in text: potential += 0.3
        if 'performance' in text: potential += 0.2
        if 'efficiency' in text: potential += 0.2
        if 'memory' in text: potential += 0.15
        if 'time' in text: potential += 0.15
        
        return min(potential, 1.0)
    
    def _identify_bottlenecks(self, title: str, details: str) -> List[str]:
        """Identify potential bottlenecks"""
        bottlenecks = []
        text = (title + ' ' + details).lower()
        
        if 'recursive' in text:
            bottlenecks.append('Recursive depth limitations')
        if 'memory' in text:
            bottlenecks.append('Memory usage constraints')
        if 'dependency' in text:
            bottlenecks.append('Dependency resolution complexity')
        if 'validation' in text:
            bottlenecks.append('Validation overhead')
        if 'generation' in text:
            bottlenecks.append('Generation computational cost')
        
        return bottlenecks
    
    def _save_analysis_results(self, analyses: List[ComplexityAnalysis]):
        """Save analysis results"""
        os.makedirs('.taskmaster/reports', exist_ok=True)
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_tasks_analyzed': len(analyses),
            'complexity_analyses': [asdict(analysis) for analysis in analyses],
            'summary': self._generate_summary(analyses)
        }
        
        with open('.taskmaster/reports/task_complexity_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)
    
    def _generate_summary(self, analyses: List[ComplexityAnalysis]) -> Dict[str, Any]:
        """Generate analysis summary"""
        if not analyses:
            return {}
        
        avg_weight = sum(a.computational_weight for a in analyses) / len(analyses)
        avg_optimization = sum(a.optimization_potential for a in analyses) / len(analyses)
        
        complexity_distribution = {}
        for analysis in analyses:
            time_comp = analysis.time_complexity
            complexity_distribution[time_comp] = complexity_distribution.get(time_comp, 0) + 1
        
        return {
            'average_computational_weight': avg_weight,
            'average_optimization_potential': avg_optimization,
            'time_complexity_distribution': complexity_distribution,
            'high_weight_tasks': len([a for a in analyses if a.computational_weight > 0.7])
        }

def main():
    """Main execution"""
    analyzer = TaskComplexityAnalyzer()
    analyses = analyzer.analyze_all_tasks()
    print(f"Analyzed {len(analyses)} tasks")
    return len(analyses) > 0

if __name__ == "__main__":
    main()
