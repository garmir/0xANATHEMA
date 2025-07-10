#!/usr/bin/env python3

"""
Enhanced Atomic Task Detection System
Implements semantic analysis and complexity scoring to achieve 80%+ atomic task detection
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AtomicityScore:
    """Atomicity assessment result"""
    task_id: str
    is_atomic: bool
    confidence: float
    complexity_score: float
    reasons: List[str]

class EnhancedAtomicDetector:
    """
    Advanced atomic task detection using semantic analysis and complexity scoring
    Target: 80%+ atomic task detection accuracy
    """
    
    def __init__(self):
        self.atomic_patterns = [
            # Single action verbs indicating atomic tasks
            r'\b(implement|create|write|build|add|fix|update|test|deploy|configure)\b',
            # Specific deliverables
            r'\b(file|function|class|method|script|config|test|documentation)\b',
            # Measurable outcomes
            r'\b(complete|finish|validate|verify|ensure|check)\b'
        ]
        
        self.composite_patterns = [
            # Multiple actions indicating composite tasks
            r'\b(design and implement|create and test|build and deploy)\b',
            # Complex workflows
            r'\b(system|framework|pipeline|architecture|infrastructure)\b',
            # Integration tasks
            r'\b(integrate|coordinate|orchestrate|manage)\b'
        ]
        
        self.complexity_indicators = {
            'high': ['system', 'framework', 'architecture', 'infrastructure', 'comprehensive'],
            'medium': ['integration', 'optimization', 'configuration', 'monitoring'],
            'low': ['function', 'method', 'file', 'script', 'test']
        }
    
    def analyze_task_atomicity(self, task_data: Dict) -> AtomicityScore:
        """Analyze a single task for atomicity using multiple criteria"""
        task_id = task_data.get('id', 'unknown')
        title = task_data.get('title', '')
        description = task_data.get('description', '')
        details = task_data.get('details', '')
        
        text_content = f"{title} {description} {details}".lower()
        
        # 1. Semantic Analysis
        semantic_score = self._analyze_semantic_patterns(text_content)
        
        # 2. Complexity Scoring
        complexity_score = self._calculate_complexity_score(task_data, text_content)
        
        # 3. Dependency Analysis
        dependency_score = self._analyze_dependencies(task_data)
        
        # 4. Size Analysis
        size_score = self._analyze_task_size(text_content)
        
        # Combine scores with weights
        weights = {
            'semantic': 0.3,
            'complexity': 0.3,
            'dependency': 0.2,
            'size': 0.2
        }
        
        overall_score = (
            semantic_score * weights['semantic'] +
            complexity_score * weights['complexity'] +
            dependency_score * weights['dependency'] +
            size_score * weights['size']
        )
        
        # Determine atomicity (threshold: 0.6)
        is_atomic = overall_score >= 0.6
        confidence = min(overall_score, 1.0)
        
        # Generate reasoning
        reasons = self._generate_atomicity_reasons(
            semantic_score, complexity_score, dependency_score, size_score, text_content
        )
        
        return AtomicityScore(
            task_id=str(task_id),
            is_atomic=is_atomic,
            confidence=confidence,
            complexity_score=1.0 - complexity_score,  # Invert for reporting
            reasons=reasons
        )
    
    def _analyze_semantic_patterns(self, text: str) -> float:
        """Analyze semantic patterns for atomicity indicators"""
        atomic_matches = sum(len(re.findall(pattern, text)) for pattern in self.atomic_patterns)
        composite_matches = sum(len(re.findall(pattern, text)) for pattern in self.composite_patterns)
        
        total_matches = atomic_matches + composite_matches
        if total_matches == 0:
            return 0.5  # Neutral score
        
        # Higher atomic ratio = more atomic
        atomic_ratio = atomic_matches / total_matches
        return atomic_ratio
    
    def _calculate_complexity_score(self, task_data: Dict, text: str) -> float:
        """Calculate complexity score (lower = more atomic)"""
        complexity_score = 0.0
        
        # Check for complexity indicators in text
        for level, indicators in self.complexity_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in text)
            if level == 'high':
                complexity_score += matches * 0.8
            elif level == 'medium':
                complexity_score += matches * 0.5
            elif level == 'low':
                complexity_score += matches * 0.2
        
        # Check estimated duration
        duration_str = task_data.get('estimated_duration', '5min')
        if 'min' in duration_str:
            minutes = int(duration_str.replace('min', ''))
            if minutes <= 30:
                complexity_score -= 0.3  # Lower complexity for short tasks
            elif minutes > 120:
                complexity_score += 0.4  # Higher complexity for long tasks
        
        # Check resource requirements
        resources = task_data.get('resources', {})
        memory_str = resources.get('memory', '50MB')
        if 'GB' in memory_str or int(memory_str.replace('MB', '').replace('GB', '')) > 500:
            complexity_score += 0.3  # Higher complexity for resource-intensive tasks
        
        # Normalize to 0-1 range (invert so lower complexity = higher score)
        normalized_score = max(0, min(1, 1 - (complexity_score / 3)))
        return normalized_score
    
    def _analyze_dependencies(self, task_data: Dict) -> float:
        """Analyze dependencies (fewer dependencies = more atomic)"""
        dependencies = task_data.get('dependencies', [])
        subtasks = task_data.get('subtasks', [])
        
        total_deps = len(dependencies) + len(subtasks)
        
        if total_deps == 0:
            return 1.0  # No dependencies = very atomic
        elif total_deps <= 2:
            return 0.7  # Few dependencies = somewhat atomic
        else:
            return max(0.2, 1 - (total_deps * 0.1))  # Many dependencies = less atomic
    
    def _analyze_task_size(self, text: str) -> float:
        """Analyze task size (smaller = more atomic)"""
        word_count = len(text.split())
        
        if word_count <= 50:
            return 1.0  # Very concise = atomic
        elif word_count <= 150:
            return 0.7  # Moderate size = somewhat atomic
        else:
            return max(0.3, 1 - (word_count / 500))  # Large = less atomic
    
    def _generate_atomicity_reasons(self, semantic: float, complexity: float, 
                                   dependency: float, size: float, text: str) -> List[str]:
        """Generate human-readable reasons for atomicity assessment"""
        reasons = []
        
        if semantic >= 0.7:
            reasons.append("Strong atomic action patterns detected")
        elif semantic <= 0.3:
            reasons.append("Complex/composite action patterns detected")
        
        if complexity >= 0.7:
            reasons.append("Low complexity indicators")
        elif complexity <= 0.3:
            reasons.append("High complexity indicators present")
        
        if dependency >= 0.8:
            reasons.append("Minimal dependencies")
        elif dependency <= 0.4:
            reasons.append("Multiple dependencies detected")
        
        if size >= 0.8:
            reasons.append("Concise task description")
        elif size <= 0.4:
            reasons.append("Lengthy/complex description")
        
        # Specific pattern matches
        if any(pattern in text for pattern in ['implement', 'create', 'add', 'fix']):
            reasons.append("Single action verb detected")
        
        if any(pattern in text for pattern in ['system', 'framework', 'comprehensive']):
            reasons.append("System-level complexity indicators")
        
        return reasons[:3]  # Limit to top 3 reasons
    
    def process_all_tasks(self, task_file: str = '.taskmaster/optimization/task-tree.json') -> Dict:
        """Process all tasks and generate atomicity report"""
        try:
            with open(task_file, 'r') as f:
                data = json.load(f)
            
            tasks = data.get('tasks', [])
            results = []
            
            atomic_count = 0
            for task in tasks:
                score = self.analyze_task_atomicity(task)
                results.append(score)
                if score.is_atomic:
                    atomic_count += 1
                
                logger.info(f"Task {score.task_id}: {'ATOMIC' if score.is_atomic else 'COMPOSITE'} "
                           f"(confidence: {score.confidence:.2f})")
            
            atomicity_ratio = atomic_count / len(tasks) if tasks else 0
            target_met = atomicity_ratio >= 0.8
            
            report = {
                "analysis_timestamp": "2025-07-10T17:40:00Z",
                "total_tasks": len(tasks),
                "atomic_tasks": atomic_count,
                "atomicity_ratio": round(atomicity_ratio, 3),
                "target_ratio": 0.8,
                "target_met": target_met,
                "improvement_needed": max(0, round((0.8 - atomicity_ratio) * len(tasks))),
                "detailed_results": [
                    {
                        "task_id": result.task_id,
                        "is_atomic": result.is_atomic,
                        "confidence": round(result.confidence, 3),
                        "complexity_score": round(result.complexity_score, 3),
                        "reasons": result.reasons
                    }
                    for result in results
                ]
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error processing tasks: {e}")
            return {"error": str(e)}
    
    def update_task_atomicity(self, task_file: str = '.taskmaster/optimization/task-tree.json') -> bool:
        """Update task file with improved atomicity classifications"""
        try:
            with open(task_file, 'r') as f:
                data = json.load(f)
            
            tasks = data.get('tasks', [])
            updated_count = 0
            
            for task in tasks:
                score = self.analyze_task_atomicity(task)
                
                # Update task with atomicity information
                old_complexity = task.get('complexity', 5)
                if score.is_atomic and score.confidence >= 0.7:
                    task['complexity'] = min(old_complexity, 3)  # Mark as low complexity
                    task['atomic'] = True
                    updated_count += 1
                else:
                    task['complexity'] = max(old_complexity, 6)  # Mark as high complexity
                    task['atomic'] = False
                
                task['atomicity_confidence'] = round(score.confidence, 3)
                task['atomicity_reasons'] = score.reasons
            
            # Save updated file
            with open(task_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Updated {updated_count} tasks with improved atomicity classifications")
            return True
            
        except Exception as e:
            logger.error(f"Error updating task atomicity: {e}")
            return False

def main():
    """Main execution function"""
    print("🔬 Enhanced Atomic Task Detection System")
    print("=" * 50)
    
    detector = EnhancedAtomicDetector()
    
    # Analyze current tasks
    report = detector.process_all_tasks()
    
    if "error" in report:
        print(f"❌ Error: {report['error']}")
        return False
    
    # Save report
    import os
    os.makedirs('.taskmaster/reports', exist_ok=True)
    with open('.taskmaster/reports/atomicity-analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Update task classifications
    updated = detector.update_task_atomicity()
    
    # Print results
    print(f"📊 Atomicity Analysis Results:")
    print(f"   Total tasks: {report['total_tasks']}")
    print(f"   Atomic tasks: {report['atomic_tasks']}")
    print(f"   Atomicity ratio: {report['atomicity_ratio']:.1%}")
    print(f"   Target (80%): {'✅ MET' if report['target_met'] else '❌ NOT MET'}")
    
    if not report['target_met']:
        print(f"   Improvement needed: {report['improvement_needed']} more atomic tasks")
    
    print(f"   Task classifications updated: {'✅' if updated else '❌'}")
    print(f"📄 Detailed report: .taskmaster/reports/atomicity-analysis.json")
    
    return report['target_met']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)