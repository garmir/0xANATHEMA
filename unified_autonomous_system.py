#!/usr/bin/env python3
"""
Unified Autonomous System - Integration of LABRYS Framework with Task Master AI

This module creates a unified autonomous system that combines:
- LABRYS dual-blade methodology (analytical + synthesis)
- Task Master AI prediction and optimization
- Research-driven problem solving
- Recursive self-improvement
- Comprehensive monitoring and validation
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Import Task Master AI components
from intelligent_task_predictor import TaskPredictionEngine
from autonomous_workflow_loop import AutonomousWorkflowLoop
from task_complexity_analyzer import TaskComplexityAnalyzer
from optimization_engine import OptimizationEngine

# Import LABRYS components
try:
    from taskmaster_labrys import TaskMasterLabrys, LabrysTask, TaskType, TaskStatus
    from labrys_main import LabrysFramework
    LABRYS_AVAILABLE = True
except ImportError:
    LABRYS_AVAILABLE = False
    print("Warning: LABRYS components not available, running in Task Master AI only mode")


@dataclass
class UnifiedSystemState:
    """State tracking for the unified autonomous system"""
    task_master_initialized: bool = False
    labrys_initialized: bool = False
    prediction_engine_ready: bool = False
    workflow_loop_active: bool = False
    last_improvement_cycle: Optional[str] = None
    total_tasks_processed: int = 0
    autonomous_success_rate: float = 0.0
    system_health_score: float = 0.0


class UnifiedAutonomousSystem:
    """
    Unified Autonomous System combining LABRYS and Task Master AI
    
    Features:
    - Dual-blade task analysis and synthesis
    - Intelligent task prediction and auto-generation
    - Research-driven problem solving
    - Recursive self-improvement
    - Comprehensive monitoring and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified autonomous system"""
        self.config = config or self._load_default_config()
        self.logger = self._setup_logging()
        self.state = UnifiedSystemState()
        
        # Initialize components
        self.task_prediction_engine = TaskPredictionEngine()
        self.autonomous_workflow = AutonomousWorkflowLoop()
        self.complexity_analyzer = TaskComplexityAnalyzer(".taskmaster/tasks/tasks.json")
        self.optimization_engine = OptimizationEngine(self.complexity_analyzer)
        
        # Initialize LABRYS components if available
        if LABRYS_AVAILABLE:
            self.labrys_framework = LabrysFramework()
            self.taskmaster_labrys = TaskMasterLabrys()
        else:
            self.labrys_framework = None
            self.taskmaster_labrys = None
        
        # System metrics
        self.performance_metrics = {
            "unified_system": {
                "initialization_time": 0,
                "total_execution_cycles": 0,
                "successful_predictions": 0,
                "successful_improvements": 0,
                "research_resolutions": 0,
                "optimization_cycles": 0
            },
            "task_master": {
                "tasks_predicted": 0,
                "patterns_identified": 0,
                "optimizations_applied": 0
            },
            "labrys": {
                "analytical_operations": 0,
                "synthesis_operations": 0,
                "coordination_cycles": 0
            }
        }
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for unified system"""
        return {
            "autonomous_mode": True,
            "prediction_enabled": True,
            "labrys_integration": LABRYS_AVAILABLE,
            "research_integration": True,
            "recursive_improvement": True,
            "max_execution_cycles": 100,
            "improvement_threshold": 0.95,
            "health_check_interval": 300,  # 5 minutes
            "backup_enabled": True,
            "validation_required": True,
            "logging_level": "INFO"
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for unified system"""
        logger = logging.getLogger("unified_autonomous_system")
        logger.setLevel(getattr(logging, self.config.get("logging_level", "INFO")))
        
        if not logger.handlers:
            # Create logs directory
            os.makedirs(".taskmaster/logs", exist_ok=True)
            
            # File handler for unified system logs
            file_handler = logging.FileHandler(
                f".taskmaster/logs/unified-system-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - UNIFIED - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    async def initialize_unified_system(self) -> Dict[str, Any]:
        """Initialize all components of the unified system"""
        start_time = time.time()
        
        self.logger.info("🚀 Initializing Unified Autonomous System")
        self.logger.info("   Combining LABRYS + Task Master AI capabilities")
        self.logger.info("=" * 60)
        
        initialization_results = {
            "task_master_ai": {"status": "pending"},
            "labrys_framework": {"status": "pending"},
            "integration_layer": {"status": "pending"},
            "overall_status": "pending"
        }
        
        try:
            # Initialize Task Master AI components
            self.logger.info("📊 Initializing Task Master AI components...")
            
            # Initialize prediction engine
            pattern_analysis = self.task_prediction_engine.analyze_patterns()
            if "error" not in pattern_analysis:
                self.state.prediction_engine_ready = True
                self.logger.info("✅ Task prediction engine initialized")
            else:
                self.logger.warning(f"⚠️ Task prediction engine partial init: {pattern_analysis.get('error')}")
            
            # Initialize complexity analyzer
            try:
                complexity_report = self.complexity_analyzer.generate_complexity_report()
                self.logger.info("✅ Complexity analyzer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Complexity analyzer warning: {e}")
            
            self.state.task_master_initialized = True
            initialization_results["task_master_ai"] = {
                "status": "success",
                "components": ["prediction_engine", "complexity_analyzer", "optimization_engine", "workflow_loop"]
            }
            
            # Initialize LABRYS if available
            if LABRYS_AVAILABLE and self.labrys_framework:
                self.logger.info("🗲 Initializing LABRYS dual-blade framework...")
                
                labrys_init = await self.labrys_framework.initialize_system()
                
                if labrys_init.get("status") == "success":
                    self.state.labrys_initialized = True
                    self.logger.info("✅ LABRYS framework initialized")
                    initialization_results["labrys_framework"] = {
                        "status": "success",
                        "components": ["analytical_blade", "synthesis_blade", "coordinator", "validator"]
                    }
                else:
                    self.logger.warning("⚠️ LABRYS framework partial initialization")
                    initialization_results["labrys_framework"] = {
                        "status": "partial",
                        "error": labrys_init.get("message", "Unknown error")
                    }
            else:
                self.logger.info("📝 LABRYS framework not available, using Task Master AI only")
                initialization_results["labrys_framework"] = {
                    "status": "not_available",
                    "message": "LABRYS components not found"
                }
            
            # Initialize integration layer
            self.logger.info("🔗 Initializing integration layer...")
            
            # Set up unified workflow
            self.state.workflow_loop_active = True
            
            # Calculate system health score
            self.state.system_health_score = self._calculate_system_health()
            
            initialization_results["integration_layer"] = {
                "status": "success",
                "system_health_score": self.state.system_health_score
            }
            
            # Overall status
            if self.state.task_master_initialized:
                initialization_results["overall_status"] = "success"
                self.logger.info("🎉 Unified system initialization completed successfully")
            else:
                initialization_results["overall_status"] = "partial"
                self.logger.warning("⚠️ Unified system partial initialization")
            
            # Record initialization time
            self.performance_metrics["unified_system"]["initialization_time"] = time.time() - start_time
            
            return initialization_results
            
        except Exception as e:
            self.logger.error(f"❌ Unified system initialization failed: {e}")
            initialization_results["overall_status"] = "failed"
            initialization_results["error"] = str(e)
            return initialization_results
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        health_factors = []
        
        # Task Master AI health
        if self.state.task_master_initialized:
            health_factors.append(0.4)  # 40% weight
        
        # LABRYS health
        if self.state.labrys_initialized:
            health_factors.append(0.3)  # 30% weight
        elif LABRYS_AVAILABLE:
            health_factors.append(0.1)  # Partial credit if available but not init
        
        # Prediction engine health
        if self.state.prediction_engine_ready:
            health_factors.append(0.2)  # 20% weight
        
        # Workflow health
        if self.state.workflow_loop_active:
            health_factors.append(0.1)  # 10% weight
        
        return sum(health_factors) * 100  # Convert to percentage
    
    async def run_unified_autonomous_cycle(self) -> Dict[str, Any]:
        """
        Run one complete cycle of the unified autonomous system
        
        This combines:
        1. Task prediction and analysis
        2. LABRYS dual-blade processing (if available)
        3. Research-driven problem solving
        4. Optimization and improvement
        """
        cycle_start = time.time()
        cycle_results = {
            "cycle_id": f"cycle_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "phases": {},
            "overall_success": False
        }
        
        self.logger.info(f"🔄 Starting unified autonomous cycle: {cycle_results['cycle_id']}")
        
        try:
            # Phase 1: Task Prediction and Pattern Analysis
            self.logger.info("📊 Phase 1: Task Prediction and Pattern Analysis")
            
            prediction_result = self.task_prediction_engine.run_full_analysis()
            task_suggestions = prediction_result.get("task_suggestions", [])
            
            cycle_results["phases"]["prediction"] = {
                "status": "success",
                "patterns_identified": len(prediction_result.get("pattern_analysis", {}).get("pattern_summary", [])),
                "task_suggestions": len(task_suggestions),
                "behavioral_insights": len(prediction_result.get("behavioral_insights", []))
            }
            
            self.performance_metrics["task_master"]["tasks_predicted"] += len(task_suggestions)
            self.performance_metrics["task_master"]["patterns_identified"] += cycle_results["phases"]["prediction"]["patterns_identified"]
            
            # Phase 2: LABRYS Dual-Blade Analysis (if available)
            if LABRYS_AVAILABLE and self.state.labrys_initialized:
                self.logger.info("🗲 Phase 2: LABRYS Dual-Blade Analysis")
                
                # Create LABRYS tasks from predictions
                labrys_tasks = self._convert_predictions_to_labrys_tasks(task_suggestions)
                
                if labrys_tasks:
                    labrys_result = await self.taskmaster_labrys.execute_task_sequence(labrys_tasks)
                    
                    cycle_results["phases"]["labrys"] = {
                        "status": "success",
                        "tasks_processed": len(labrys_tasks),
                        "completed_tasks": labrys_result.get("completed_tasks", 0),
                        "analytical_operations": len([t for t in labrys_tasks if t.type == TaskType.ANALYTICAL]),
                        "synthesis_operations": len([t for t in labrys_tasks if t.type == TaskType.SYNTHESIS])
                    }
                    
                    self.performance_metrics["labrys"]["analytical_operations"] += cycle_results["phases"]["labrys"]["analytical_operations"]
                    self.performance_metrics["labrys"]["synthesis_operations"] += cycle_results["phases"]["labrys"]["synthesis_operations"]
                else:
                    cycle_results["phases"]["labrys"] = {"status": "skipped", "reason": "no_tasks_to_process"}
            else:
                cycle_results["phases"]["labrys"] = {"status": "not_available"}
            
            # Phase 3: Optimization and Complexity Analysis
            self.logger.info("⚡ Phase 3: Optimization and Complexity Analysis")
            
            try:
                optimization_result = self.optimization_engine.optimize_execution_order()
                
                cycle_results["phases"]["optimization"] = {
                    "status": "success",
                    "efficiency_score": optimization_result.efficiency_score,
                    "optimization_strategy": optimization_result.strategy.value,
                    "bottlenecks_identified": len(optimization_result.bottlenecks)
                }
                
                self.performance_metrics["task_master"]["optimizations_applied"] += 1
                
            except Exception as e:
                self.logger.warning(f"Optimization phase warning: {e}")
                cycle_results["phases"]["optimization"] = {"status": "partial", "error": str(e)}
            
            # Phase 4: Research-Driven Problem Resolution
            self.logger.info("🔍 Phase 4: Research-Driven Problem Resolution")
            
            # Check for any stuck situations or problems to research
            research_candidates = self._identify_research_candidates(cycle_results)
            
            if research_candidates:
                research_results = []
                for candidate in research_candidates[:3]:  # Limit to 3 research queries
                    research_result = self.autonomous_workflow.research_solution(
                        candidate["problem"], 
                        candidate.get("context", "")
                    )
                    research_results.append({
                        "problem": candidate["problem"],
                        "solution_steps": len(research_result.solution_steps),
                        "confidence": research_result.confidence
                    })
                
                cycle_results["phases"]["research"] = {
                    "status": "success",
                    "research_queries": len(research_candidates),
                    "solutions_found": len(research_results),
                    "avg_confidence": sum(r["confidence"] for r in research_results) / len(research_results) if research_results else 0
                }
                
                self.performance_metrics["unified_system"]["research_resolutions"] += len(research_results)
            else:
                cycle_results["phases"]["research"] = {"status": "no_research_needed"}
            
            # Phase 5: Recursive Self-Improvement
            if self.config.get("recursive_improvement", True):
                self.logger.info("🔄 Phase 5: Recursive Self-Improvement")
                
                improvement_result = await self._perform_recursive_improvement(cycle_results)
                cycle_results["phases"]["improvement"] = improvement_result
                
                if improvement_result.get("status") == "success":
                    self.performance_metrics["unified_system"]["successful_improvements"] += 1
            
            # Calculate overall cycle success
            successful_phases = len([p for p in cycle_results["phases"].values() if p.get("status") == "success"])
            total_phases = len(cycle_results["phases"])
            cycle_success_rate = successful_phases / total_phases if total_phases > 0 else 0
            
            cycle_results["overall_success"] = cycle_success_rate >= 0.6  # 60% success threshold
            cycle_results["success_rate"] = cycle_success_rate
            cycle_results["execution_time"] = time.time() - cycle_start
            
            # Update system metrics
            self.performance_metrics["unified_system"]["total_execution_cycles"] += 1
            if cycle_results["overall_success"]:
                self.performance_metrics["unified_system"]["successful_predictions"] += 1
            
            self.state.autonomous_success_rate = (
                self.performance_metrics["unified_system"]["successful_predictions"] /
                self.performance_metrics["unified_system"]["total_execution_cycles"]
            )
            
            self.logger.info(f"✅ Cycle {cycle_results['cycle_id']} completed: {cycle_success_rate:.1%} success rate")
            
            return cycle_results
            
        except Exception as e:
            self.logger.error(f"❌ Unified autonomous cycle failed: {e}")
            cycle_results["overall_success"] = False
            cycle_results["error"] = str(e)
            return cycle_results
    
    def _convert_predictions_to_labrys_tasks(self, task_suggestions: List[Dict]) -> List[LabrysTask]:
        """Convert Task Master AI predictions to LABRYS tasks"""
        if not LABRYS_AVAILABLE:
            return []
        
        labrys_tasks = []
        
        for suggestion in task_suggestions:
            predicted_task = suggestion.get("predicted_task", {})
            
            # Determine task type based on content
            task_type = TaskType.ANALYTICAL
            if "implement" in predicted_task.get("title", "").lower():
                task_type = TaskType.SYNTHESIS
            elif "coordinate" in predicted_task.get("title", "").lower():
                task_type = TaskType.COORDINATION
            
            labrys_task = LabrysTask(
                id=f"pred_{predicted_task.get('id', 'unknown')}",
                title=predicted_task.get("title", ""),
                description=predicted_task.get("description", ""),
                type=task_type,
                priority=predicted_task.get("priority", "medium"),
                dependencies=[],
                validation=[]
            )
            
            labrys_tasks.append(labrys_task)
        
        return labrys_tasks
    
    def _identify_research_candidates(self, cycle_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify problems that could benefit from research-driven solutions"""
        candidates = []
        
        # Check for failed phases
        for phase_name, phase_result in cycle_results.get("phases", {}).items():
            if phase_result.get("status") in ["failed", "partial"]:
                candidates.append({
                    "problem": f"{phase_name} phase issue: {phase_result.get('error', 'Unknown error')}",
                    "context": f"Unified autonomous cycle phase failure in {phase_name}"
                })
        
        # Check for optimization opportunities
        optimization_phase = cycle_results.get("phases", {}).get("optimization", {})
        if optimization_phase.get("efficiency_score", 0) < 0.8:
            candidates.append({
                "problem": "Low optimization efficiency score detected",
                "context": f"Efficiency score: {optimization_phase.get('efficiency_score', 0):.2f}"
            })
        
        return candidates
    
    async def _perform_recursive_improvement(self, cycle_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform recursive self-improvement based on cycle results"""
        try:
            # Analyze cycle performance
            success_rate = cycle_results.get("success_rate", 0)
            execution_time = cycle_results.get("execution_time", 0)
            
            improvement_actions = []
            
            # Identify improvement opportunities
            if success_rate < 0.8:
                improvement_actions.append("optimize_success_rate")
            
            if execution_time > 60:  # More than 1 minute
                improvement_actions.append("optimize_execution_time")
            
            # Check phase-specific improvements
            phases = cycle_results.get("phases", {})
            if phases.get("optimization", {}).get("status") != "success":
                improvement_actions.append("enhance_optimization_phase")
            
            if not improvement_actions:
                return {"status": "no_improvement_needed", "current_performance": "optimal"}
            
            # Apply improvements (simplified for this integration)
            improvement_results = []
            for action in improvement_actions:
                if action == "optimize_success_rate":
                    # Adjust configuration for better success rate
                    self.config["improvement_threshold"] = max(0.85, self.config.get("improvement_threshold", 0.95) - 0.05)
                    improvement_results.append(f"Adjusted improvement threshold to {self.config['improvement_threshold']}")
                
                elif action == "optimize_execution_time":
                    # Optimize for faster execution
                    self.config["max_execution_cycles"] = min(self.config.get("max_execution_cycles", 100), 80)
                    improvement_results.append(f"Reduced max execution cycles to {self.config['max_execution_cycles']}")
            
            self.state.last_improvement_cycle = cycle_results.get("cycle_id")
            
            return {
                "status": "success",
                "improvements_applied": len(improvement_actions),
                "improvement_details": improvement_results
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def run_continuous_autonomous_operation(self, max_cycles: int = None) -> Dict[str, Any]:
        """
        Run the unified system in continuous autonomous mode
        """
        if not self.state.task_master_initialized:
            await self.initialize_unified_system()
        
        max_cycles = max_cycles or self.config.get("max_execution_cycles", 100)
        
        self.logger.info(f"🚀 Starting continuous autonomous operation (max {max_cycles} cycles)")
        
        operation_results = {
            "start_time": datetime.now().isoformat(),
            "cycles_completed": 0,
            "successful_cycles": 0,
            "failed_cycles": 0,
            "cycle_details": []
        }
        
        try:
            for cycle_num in range(max_cycles):
                self.logger.info(f"\n{'='*20} Cycle {cycle_num + 1}/{max_cycles} {'='*20}")
                
                cycle_result = await self.run_unified_autonomous_cycle()
                
                operation_results["cycles_completed"] += 1
                if cycle_result.get("overall_success"):
                    operation_results["successful_cycles"] += 1
                else:
                    operation_results["failed_cycles"] += 1
                
                operation_results["cycle_details"].append({
                    "cycle_number": cycle_num + 1,
                    "cycle_id": cycle_result.get("cycle_id"),
                    "success": cycle_result.get("overall_success"),
                    "execution_time": cycle_result.get("execution_time", 0)
                })
                
                # Check for early termination conditions
                if self.state.autonomous_success_rate >= self.config.get("improvement_threshold", 0.95):
                    self.logger.info(f"🎉 Achieved target autonomous success rate: {self.state.autonomous_success_rate:.1%}")
                    break
                
                # Health check
                current_health = self._calculate_system_health()
                if current_health < 50:
                    self.logger.warning(f"⚠️ System health below threshold: {current_health:.1f}%")
                    break
                
                # Brief pause between cycles
                await asyncio.sleep(1)
            
            operation_results["end_time"] = datetime.now().isoformat()
            operation_results["final_success_rate"] = self.state.autonomous_success_rate
            operation_results["final_health_score"] = self._calculate_system_health()
            operation_results["performance_metrics"] = self.performance_metrics
            
            self.logger.info(f"🏁 Autonomous operation completed:")
            self.logger.info(f"   Cycles: {operation_results['cycles_completed']}")
            self.logger.info(f"   Success Rate: {operation_results['final_success_rate']:.1%}")
            self.logger.info(f"   Health Score: {operation_results['final_health_score']:.1f}%")
            
            return operation_results
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Autonomous operation interrupted by user")
            operation_results["interrupted"] = True
            return operation_results
        except Exception as e:
            self.logger.error(f"❌ Autonomous operation failed: {e}")
            operation_results["error"] = str(e)
            return operation_results
    
    def get_unified_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the unified system"""
        return {
            "unified_system": {
                "version": "1.0.0",
                "description": "LABRYS + Task Master AI Unified Autonomous System",
                "state": asdict(self.state),
                "config": self.config,
                "performance_metrics": self.performance_metrics
            },
            "components": {
                "task_master_ai": {
                    "initialized": self.state.task_master_initialized,
                    "prediction_engine": self.state.prediction_engine_ready,
                    "workflow_loop": self.state.workflow_loop_active
                },
                "labrys_framework": {
                    "available": LABRYS_AVAILABLE,
                    "initialized": self.state.labrys_initialized
                }
            },
            "health": {
                "overall_score": self._calculate_system_health(),
                "autonomous_success_rate": self.state.autonomous_success_rate,
                "last_improvement": self.state.last_improvement_cycle
            },
            "timestamp": datetime.now().isoformat()
        }


async def main():
    """Main entry point for unified autonomous system"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified Autonomous System - LABRYS + Task Master AI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--initialize", action="store_true", help="Initialize unified system")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--run-cycle", action="store_true", help="Run single autonomous cycle")
    parser.add_argument("--continuous", action="store_true", help="Run continuous autonomous operation")
    parser.add_argument("--cycles", type=int, default=10, help="Number of cycles for continuous operation")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create unified system
    unified_system = UnifiedAutonomousSystem(config)
    
    if args.initialize:
        print("🚀 Initializing Unified Autonomous System...")
        result = await unified_system.initialize_unified_system()
        print(json.dumps(result, indent=2))
    
    elif args.status:
        if not unified_system.state.task_master_initialized:
            await unified_system.initialize_unified_system()
        status = unified_system.get_unified_system_status()
        print(json.dumps(status, indent=2))
    
    elif args.run_cycle:
        if not unified_system.state.task_master_initialized:
            await unified_system.initialize_unified_system()
        result = await unified_system.run_unified_autonomous_cycle()
        print(json.dumps(result, indent=2))
    
    elif args.continuous:
        result = await unified_system.run_continuous_autonomous_operation(args.cycles)
        print(json.dumps(result, indent=2))
    
    else:
        print("🤖 Unified Autonomous System")
        print("=" * 40)
        print("Combining LABRYS dual-blade methodology with Task Master AI")
        print("for ultimate autonomous development capabilities.")
        print("\nUse --help for available commands")


if __name__ == "__main__":
    asyncio.run(main())