#!/usr/bin/env python3

"""
Autonomy Scorer for Task-Master System
Calculates and tracks autonomy score to meet 95% target
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AutonomyMetrics:
    """Metrics for calculating autonomy score"""
    tasks_completed_autonomously: int = 0
    tasks_requiring_intervention: int = 0
    research_driven_solutions: int = 0
    error_recovery_successes: int = 0
    total_execution_time: float = 0.0
    intervention_time: float = 0.0
    success_rate: float = 0.0
    
class AutonomyScorer:
    """
    Calculates and tracks autonomy score for the Task-Master system
    Target: >= 95% autonomy score for production readiness
    """
    
    def __init__(self):
        self.metrics = AutonomyMetrics()
        self.target_score = 0.95
        self.scoring_history = []
        self.last_update = time.time()
        
    def calculate_current_autonomy_score(self) -> float:
        """Calculate current autonomy score based on system performance"""
        
        # Load existing metrics
        self._load_autonomy_metrics()
        
        # Calculate component scores
        scores = {
            'task_completion_autonomy': self._calculate_task_autonomy(),
            'research_effectiveness': self._calculate_research_effectiveness(),
            'error_recovery_capability': self._calculate_error_recovery(),
            'execution_efficiency': self._calculate_execution_efficiency(),
            'system_integration': self._calculate_system_integration(),
            'production_readiness': self._calculate_production_readiness()
        }
        
        # Weighted autonomy score
        weights = {
            'task_completion_autonomy': 0.35,
            'research_effectiveness': 0.20,
            'error_recovery_capability': 0.15,
            'execution_efficiency': 0.15,
            'system_integration': 0.10,
            'production_readiness': 0.05
        }
        
        autonomy_score = sum(scores[component] * weights[component] 
                           for component in scores)
        
        # Update metrics
        self.metrics.success_rate = autonomy_score
        
        # Save scoring history
        self._save_scoring_record(autonomy_score, scores)
        
        logger.info(f"🎯 Current autonomy score: {autonomy_score:.3f} (target: {self.target_score:.3f})")
        
        return autonomy_score
    
    def _calculate_task_autonomy(self) -> float:
        """Calculate task completion autonomy"""
        
        try:
            # Check task completion rates
            with open('.taskmaster/tasks/tasks.json', 'r') as f:
                tasks_data = json.load(f)
            
            tasks = tasks_data.get('master', {}).get('tasks', [])
            if not tasks:
                return 0.5  # Default moderate score
            
            total_tasks = len(tasks)
            completed_tasks = len([t for t in tasks if t.get('status') == 'done'])
            
            # Base completion rate
            completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            # Autonomy factors
            autonomy_indicators = self._analyze_autonomy_indicators(tasks)
            
            # Combined score
            task_autonomy = (completion_rate * 0.7) + (autonomy_indicators * 0.3)
            
            return min(1.0, task_autonomy)
            
        except Exception as e:
            logger.warning(f"Cannot calculate task autonomy: {e}")
            return 0.5
    
    def _analyze_autonomy_indicators(self, tasks: List[Dict]) -> float:
        """Analyze indicators of autonomous execution"""
        
        autonomy_score = 0.0
        total_indicators = 0
        
        # Check for autonomous workflow files
        autonomous_files = [
            '.taskmaster/autonomous-workflow-loop.py',
            '.taskmaster/claude-integration-wrapper.py',
            '.taskmaster/optimization/evolutionary-optimizer.py'
        ]
        
        files_present = sum(1 for f in autonomous_files if os.path.exists(f))
        autonomy_score += (files_present / len(autonomous_files)) * 0.3
        total_indicators += 0.3
        
        # Check for error recovery configuration
        recovery_configs = [
            '.taskmaster/config/error-recovery.json',
            '.taskmaster/config/autonomous-execution.json'
        ]
        
        configs_present = sum(1 for f in recovery_configs if os.path.exists(f))
        autonomy_score += (configs_present / len(recovery_configs)) * 0.2
        total_indicators += 0.2
        
        # Check for optimization results
        optimization_files = [
            '.taskmaster/optimization/evolutionary-optimization.json',
            '.taskmaster/artifacts/sqrt-space/sqrt-optimized.json'
        ]
        
        optimizations_present = sum(1 for f in optimization_files if os.path.exists(f))
        autonomy_score += (optimizations_present / len(optimization_files)) * 0.3
        total_indicators += 0.3
        
        # Check for validation systems
        validation_files = [
            '.taskmaster/reports/autonomous-system-validation.json',
            '.taskmaster/optimization/autonomous-system-validator.py'
        ]
        
        validations_present = sum(1 for f in validation_files if os.path.exists(f))
        autonomy_score += (validations_present / len(validation_files)) * 0.2
        total_indicators += 0.2
        
        return autonomy_score / total_indicators if total_indicators > 0 else 0.0
    
    def _calculate_research_effectiveness(self) -> float:
        """Calculate research-driven problem solving effectiveness"""
        
        research_score = 0.0
        
        # Check for research infrastructure
        if os.path.exists('.taskmaster/autonomous-workflow-loop.py'):
            with open('.taskmaster/autonomous-workflow-loop.py', 'r') as f:
                content = f.read().lower()
                if 'research' in content and 'perplexity' in content:
                    research_score += 0.4
        
        # Check for Claude integration
        if os.path.exists('.taskmaster/claude-integration-wrapper.py'):
            research_score += 0.3
        
        # Check for research model configuration
        try:
            import subprocess
            result = subprocess.run(['task-master', 'models'], capture_output=True, timeout=5)
            if result.returncode == 0 and 'research' in result.stdout.lower():
                research_score += 0.3
        except:
            pass
        
        return min(1.0, research_score)
    
    def _calculate_error_recovery(self) -> float:
        """Calculate error recovery and resilience capability"""
        
        recovery_score = 0.0
        
        # Check for error recovery configurations
        error_configs = [
            '.taskmaster/config/error-recovery.json',
            '.taskmaster/config/autonomous-execution.json'
        ]
        
        for config_file in error_configs:
            if os.path.exists(config_file):
                recovery_score += 0.25
        
        # Check for error handling in code
        code_files = [
            '.taskmaster/autonomous-workflow-loop.py',
            '.taskmaster/claude-integration-wrapper.py'
        ]
        
        for code_file in code_files:
            if os.path.exists(code_file):
                try:
                    with open(code_file, 'r') as f:
                        content = f.read().lower()
                        if 'try:' in content and 'except:' in content:
                            recovery_score += 0.15
                        if 'recovery' in content or 'retry' in content:
                            recovery_score += 0.1
                except:
                    pass
        
        return min(1.0, recovery_score)
    
    def _calculate_execution_efficiency(self) -> float:
        """Calculate execution efficiency metrics"""
        
        efficiency_score = 0.0
        
        # Check for optimization artifacts
        optimization_files = [
            '.taskmaster/artifacts/sqrt-space/sqrt-optimized.json',
            '.taskmaster/artifacts/pebbling/pebbling-strategy.json',
            '.taskmaster/optimization/evolutionary-optimization.json'
        ]
        
        optimizations_present = sum(1 for f in optimization_files if os.path.exists(f))
        efficiency_score += (optimizations_present / len(optimization_files)) * 0.6
        
        # Check memory optimization results
        try:
            with open('.taskmaster/artifacts/sqrt-space/sqrt-optimized.json', 'r') as f:
                memory_data = json.load(f)
                improvements = memory_data.get('improvements', {})
                reduction_percent = improvements.get('memory_reduction_percent', 0)
                
                if reduction_percent >= 50:  # 50% reduction is good
                    efficiency_score += 0.4
                elif reduction_percent >= 25:  # 25% reduction is acceptable
                    efficiency_score += 0.2
        except:
            pass
        
        return min(1.0, efficiency_score)
    
    def _calculate_system_integration(self) -> float:
        """Calculate system integration completeness"""
        
        integration_score = 0.0
        
        # Check core system files
        core_files = [
            '.taskmaster/tasks/tasks.json',
            '.taskmaster/optimization/task-tree.json',
            '.taskmaster/config.json'
        ]
        
        files_present = sum(1 for f in core_files if os.path.exists(f))
        integration_score += (files_present / len(core_files)) * 0.4
        
        # Check CLI integration
        try:
            import subprocess
            result = subprocess.run(['task-master', '--help'], capture_output=True, timeout=5)
            if result.returncode == 0:
                integration_score += 0.3
        except:
            pass
        
        # Check Claude Code integration
        try:
            import subprocess
            result = subprocess.run(['claude', '--version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                integration_score += 0.3
        except:
            pass
        
        return min(1.0, integration_score)
    
    def _calculate_production_readiness(self) -> float:
        """Calculate production deployment readiness"""
        
        production_score = 0.0
        
        # Check executable scripts
        executable_scripts = [
            '.taskmaster/start-autonomous-loop.sh',
            '.taskmaster/autonomous-workflow-loop.py'
        ]
        
        for script in executable_scripts:
            if os.path.exists(script) and os.access(script, os.X_OK):
                production_score += 0.25
        
        # Check documentation
        docs = [
            '.taskmaster/AUTONOMOUS-WORKFLOW-IMPLEMENTATION.md',
            'CLAUDE.md'
        ]
        
        docs_present = sum(1 for d in docs if os.path.exists(d))
        production_score += (docs_present / len(docs)) * 0.3
        
        # Check validation results
        if os.path.exists('.taskmaster/reports/autonomous-system-validation.json'):
            try:
                with open('.taskmaster/reports/autonomous-system-validation.json', 'r') as f:
                    validation_data = json.load(f)
                    if validation_data.get('autonomous_capable', False):
                        production_score += 0.2
            except:
                pass
        
        return min(1.0, production_score)
    
    def _load_autonomy_metrics(self) -> None:
        """Load existing autonomy metrics"""
        
        metrics_file = '.taskmaster/reports/autonomy-metrics.json'
        
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    
                # Update metrics from saved data
                saved_metrics = data.get('metrics', {})
                self.metrics.tasks_completed_autonomously = saved_metrics.get('tasks_completed_autonomously', 0)
                self.metrics.tasks_requiring_intervention = saved_metrics.get('tasks_requiring_intervention', 0)
                self.metrics.research_driven_solutions = saved_metrics.get('research_driven_solutions', 0)
                self.metrics.error_recovery_successes = saved_metrics.get('error_recovery_successes', 0)
                
            except Exception as e:
                logger.warning(f"Could not load autonomy metrics: {e}")
    
    def _save_scoring_record(self, score: float, components: Dict[str, float]) -> None:
        """Save autonomy scoring record"""
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'autonomy_score': score,
            'target_score': self.target_score,
            'target_met': score >= self.target_score,
            'component_scores': components,
            'metrics': {
                'tasks_completed_autonomously': self.metrics.tasks_completed_autonomously,
                'tasks_requiring_intervention': self.metrics.tasks_requiring_intervention,
                'research_driven_solutions': self.metrics.research_driven_solutions,
                'error_recovery_successes': self.metrics.error_recovery_successes,
                'success_rate': self.metrics.success_rate
            }
        }
        
        self.scoring_history.append(record)
        
        # Save to file
        os.makedirs('.taskmaster/reports', exist_ok=True)
        
        with open('.taskmaster/reports/autonomy-metrics.json', 'w') as f:
            json.dump({
                'current_score': score,
                'target_score': self.target_score,
                'target_met': score >= self.target_score,
                'latest_components': components,
                'metrics': record['metrics'],
                'scoring_history': self.scoring_history[-10:]  # Keep last 10 records
            }, f, indent=2)
    
    def update_task_completion(self, task_id: str, autonomous: bool, research_used: bool = False) -> None:
        """Update metrics when a task completes"""
        
        if autonomous:
            self.metrics.tasks_completed_autonomously += 1
        else:
            self.metrics.tasks_requiring_intervention += 1
        
        if research_used:
            self.metrics.research_driven_solutions += 1
        
        logger.info(f"📊 Updated autonomy metrics: {self.metrics.tasks_completed_autonomously} autonomous / {self.metrics.tasks_requiring_intervention} with intervention")

def main():
    """Test autonomy scorer"""
    print("🎯 Autonomy Scorer Test")
    print("=" * 30)
    
    scorer = AutonomyScorer()
    autonomy_score = scorer.calculate_current_autonomy_score()
    
    print(f"Current Autonomy Score: {autonomy_score:.3f}")
    print(f"Target Score: {scorer.target_score:.3f}")
    print(f"Target Met: {'✅ YES' if autonomy_score >= scorer.target_score else '❌ NO'}")
    
    return autonomy_score >= scorer.target_score

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)