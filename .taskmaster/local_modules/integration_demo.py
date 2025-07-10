#!/usr/bin/env python3
"""
Task Master AI Local Modules Integration Demo
Demonstrates complete integration and usage of all local LLM modules
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
import logging

from core.api_abstraction import UnifiedModelAPI, TaskType, ModelConfigFactory
from core.recursive_prd_processor import RecursivePRDProcessor
from research.local_rag_system import LocalRAGSystem
from optimization.evolutionary_optimization import (
    EvolutionaryOptimizer, LocalLLMFitnessEvaluator, EvolutionConfig
)
from meta_learning.meta_learning_framework import MetaLearningEngine, LearningExperience
from failure_recovery.failure_detection_recovery import (
    FailureRecoverySystem, FailureType, SeverityLevel
)
from config.model_configuration import ModelConfigurationManager, DeploymentMode
from utils.performance_monitor import CachedPerformanceMonitor

logger = logging.getLogger(__name__)

class TaskMasterLocalSystem:
    """
    Complete Task Master AI system using local LLMs
    Integrates all components for autonomous operation
    """
    
    def __init__(self, data_dir: str = ".taskmaster/local_modules"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.api = None
        self.config_manager = None
        self.performance_monitor = None
        
        # Functional components
        self.prd_processor = None
        self.rag_system = None
        self.optimizer = None
        self.meta_engine = None
        self.recovery_system = None
        
        # System state
        self.initialized = False
        self.system_stats = {
            "start_time": None,
            "operations_completed": 0,
            "total_tasks_processed": 0,
            "research_queries": 0,
            "optimizations_run": 0,
            "failures_recovered": 0
        }
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing Task Master AI Local System...")
        start_time = time.time()
        
        try:
            # 1. Initialize configuration manager
            self.config_manager = ModelConfigurationManager()
            
            # 2. Initialize API with models
            self.api = UnifiedModelAPI()
            self.config_manager.configure_for_api(self.api)
            
            # 3. Initialize performance monitoring
            self.performance_monitor = CachedPerformanceMonitor()
            
            # 4. Initialize functional components
            self.prd_processor = RecursivePRDProcessor(self.api)
            self.rag_system = LocalRAGSystem(self.api)
            self.meta_engine = MetaLearningEngine(self.api)
            self.recovery_system = FailureRecoverySystem(self.api)
            
            # 5. Set up optimization capabilities
            self._setup_optimization()
            
            # 6. Load initial knowledge
            await self._load_initial_knowledge()
            
            self.initialized = True
            self.system_stats["start_time"] = start_time
            
            initialization_time = time.time() - start_time
            logger.info(f"System initialized in {initialization_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def _setup_optimization(self):
        """Setup optimization capabilities"""
        # Create a versatile fitness evaluator
        evaluation_criteria = """
        Evaluate solutions based on:
        1. Effectiveness in achieving goals
        2. Resource efficiency
        3. Implementation feasibility
        4. Maintainability and robustness
        5. Performance characteristics
        """
        
        fitness_evaluator = LocalLLMFitnessEvaluator(
            api=self.api,
            evaluation_criteria=evaluation_criteria,
            optimization_goal="Optimize system performance and efficiency"
        )
        
        config = EvolutionConfig(
            population_size=10,
            max_generations=15,
            mutation_rate=0.15,
            crossover_rate=0.8
        )
        
        self.optimizer = EvolutionaryOptimizer(
            api=self.api,
            fitness_evaluator=fitness_evaluator,
            config=config
        )
    
    async def _load_initial_knowledge(self):
        """Load initial knowledge into the RAG system"""
        initial_knowledge = [
            {
                "title": "Task Decomposition Best Practices",
                "content": """
                Effective task decomposition involves breaking complex tasks into smaller, 
                manageable subtasks. Key principles include: maintaining logical dependencies, 
                ensuring atomic task definitions, considering resource requirements, and 
                enabling parallel execution where possible.
                """,
                "source": "system_knowledge"
            },
            {
                "title": "Local LLM Performance Optimization",
                "content": """
                Local LLM performance can be optimized through: model quantization, 
                efficient prompt engineering, caching strategies, batch processing, 
                and intelligent model routing based on task complexity.
                """,
                "source": "system_knowledge"
            },
            {
                "title": "Autonomous System Design Patterns",
                "content": """
                Autonomous systems benefit from: self-monitoring capabilities, 
                adaptive behavior mechanisms, fault tolerance design, 
                continuous learning integration, and human oversight interfaces.
                """,
                "source": "system_knowledge"
            }
        ]
        
        for knowledge in initial_knowledge:
            self.rag_system.add_external_knowledge(
                knowledge["title"],
                knowledge["content"],
                knowledge["source"]
            )
        
        logger.info(f"Loaded {len(initial_knowledge)} initial knowledge entries")
    
    async def process_project_requirements(self, prd_content: str) -> Dict[str, Any]:
        """Process project requirements and generate task breakdown"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        with self.performance_monitor.monitored_operation("prd_processing", "full_workflow") as session:
            try:
                # Process PRD with recursive decomposition
                result = await self.prd_processor.process_prd(
                    prd_content, 
                    max_depth=3
                )
                
                # Record meta-learning experience
                experience = LearningExperience(
                    id=f"prd_processing_{int(time.time())}",
                    task_type="prd_processing",
                    context={
                        "prd_length": len(prd_content),
                        "complexity": "high" if len(prd_content) > 1000 else "medium"
                    },
                    action_taken={
                        "strategy": "recursive_decomposition",
                        "max_depth": 3
                    },
                    outcome={
                        "success": True,
                        "tasks_generated": result["total_tasks"]
                    },
                    performance_metrics={
                        "performance_score": 0.9,
                        "processing_time": result["processing_time"]
                    }
                )
                
                self.meta_engine.record_experience(experience)
                self.system_stats["total_tasks_processed"] += result["total_tasks"]
                
                session.add_metric("tasks_generated", result["total_tasks"], "count")
                session.add_metric("processing_time", result["processing_time"], "seconds")
                
                return result
                
            except Exception as e:
                await self.recovery_system.report_failure(
                    failure_type=FailureType.SYSTEM_ERROR,
                    description=f"PRD processing failed: {str(e)}",
                    context={"component": "prd_processor", "prd_length": len(prd_content)}
                )
                raise
    
    async def research_topic(self, query: str, context: str = "") -> Dict[str, Any]:
        """Conduct research on a topic using local RAG"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        with self.performance_monitor.monitored_operation("research", "query_processing") as session:
            try:
                # Perform research
                research_result = await self.rag_system.research_query(
                    query, 
                    context=context, 
                    research_type="comprehensive"
                )
                
                # Record experience
                experience = LearningExperience(
                    id=f"research_{int(time.time())}",
                    task_type="research",
                    context={"query_complexity": len(query), "has_context": bool(context)},
                    action_taken={"approach": "local_rag", "research_type": "comprehensive"},
                    outcome={"success": True, "sources_found": len(research_result["sources"])},
                    performance_metrics={
                        "performance_score": research_result.get("synthesis", {}).get("confidence_score", 0.7),
                        "execution_time": research_result["execution_time"]
                    }
                )
                
                self.meta_engine.record_experience(experience)
                self.system_stats["research_queries"] += 1
                
                session.add_metric("sources_found", len(research_result["sources"]), "count")
                session.add_metric("confidence_score", research_result.get("synthesis", {}).get("confidence_score", 0), "score")
                
                return research_result
                
            except Exception as e:
                await self.recovery_system.report_failure(
                    failure_type=FailureType.SYSTEM_ERROR,
                    description=f"Research failed: {str(e)}",
                    context={"component": "rag_system", "query": query}
                )
                raise
    
    async def optimize_configuration(self, 
                                   target_parameters: Dict[str, Any],
                                   parameter_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize system configuration using evolutionary algorithm"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        with self.performance_monitor.monitored_operation("optimization", "configuration") as session:
            try:
                # Run optimization
                optimization_result = await self.optimizer.optimize(
                    target_parameters,
                    parameter_ranges,
                    save_results=True
                )
                
                # Record experience
                experience = LearningExperience(
                    id=f"optimization_{int(time.time())}",
                    task_type="optimization",
                    context={"parameter_count": len(target_parameters)},
                    action_taken={"algorithm": "evolutionary", "generations": optimization_result["final_generation"]},
                    outcome={"success": True, "best_fitness": optimization_result["best_individual"]["fitness"]},
                    performance_metrics={
                        "performance_score": optimization_result["best_individual"]["fitness"],
                        "optimization_time": optimization_result["evolution_stats"]["optimization_time"]
                    }
                )
                
                self.meta_engine.record_experience(experience)
                self.system_stats["optimizations_run"] += 1
                
                session.add_metric("generations", optimization_result["final_generation"], "count")
                session.add_metric("best_fitness", optimization_result["best_individual"]["fitness"], "score")
                
                return optimization_result
                
            except Exception as e:
                await self.recovery_system.report_failure(
                    failure_type=FailureType.SYSTEM_ERROR,
                    description=f"Optimization failed: {str(e)}",
                    context={"component": "optimizer", "parameters": list(target_parameters.keys())}
                )
                raise
    
    async def autonomous_improvement_cycle(self) -> Dict[str, Any]:
        """Run autonomous improvement cycle using meta-learning"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        with self.performance_monitor.monitored_operation("meta_learning", "improvement_cycle") as session:
            try:
                # Get system performance data
                system_data = {
                    "uptime": time.time() - self.system_stats["start_time"],
                    "operations_completed": self.system_stats["operations_completed"],
                    "configuration": self.config_manager.get_configuration_summary(),
                    "performance": self.performance_monitor.get_combined_stats()
                }
                
                # Perform meta-improvement analysis
                meta_analysis = await self.meta_engine.meta_improvement_analysis(system_data)
                
                # Get recommendations for system optimization
                recommendations = await self.meta_engine.get_recommendations(
                    context={"system_health": "good", "performance_trend": "stable"},
                    task_type="system_optimization"
                )
                
                improvement_result = {
                    "meta_analysis": meta_analysis,
                    "recommendations": recommendations,
                    "system_data": system_data,
                    "timestamp": datetime.now().isoformat()
                }
                
                session.add_metric("recommendations_generated", len(recommendations.get("strategic_recommendations", [])), "count")
                session.add_metric("analysis_confidence", meta_analysis.get("confidence_score", 0), "score")
                
                return improvement_result
                
            except Exception as e:
                await self.recovery_system.report_failure(
                    failure_type=FailureType.SYSTEM_ERROR,
                    description=f"Meta-improvement cycle failed: {str(e)}",
                    context={"component": "meta_engine"}
                )
                raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "metrics": {},
            "recommendations": []
        }
        
        try:
            # Check API health
            api_health = await self.api.health_check()
            health_report["components"]["api"] = {
                "status": "healthy" if any(api_health.values()) else "unhealthy",
                "details": api_health
            }
            
            # Check configuration
            config_summary = self.config_manager.get_configuration_summary()
            health_report["components"]["configuration"] = {
                "status": "healthy" if config_summary["enabled_models"] > 0 else "unhealthy",
                "details": config_summary
            }
            
            # Check performance
            perf_stats = self.performance_monitor.get_combined_stats()
            health_report["components"]["performance"] = {
                "status": "healthy",
                "details": perf_stats
            }
            
            # Check recovery system
            recovery_status = self.recovery_system.get_recovery_status()
            health_report["components"]["recovery"] = {
                "status": "healthy" if recovery_status["recovery_rate"] > 0.8 else "warning",
                "details": recovery_status
            }
            
            # System metrics
            health_report["metrics"] = {
                "uptime": time.time() - self.system_stats["start_time"],
                "operations_completed": self.system_stats["operations_completed"],
                "total_tasks_processed": self.system_stats["total_tasks_processed"],
                "research_queries": self.system_stats["research_queries"],
                "optimizations_run": self.system_stats["optimizations_run"]
            }
            
            # Determine overall status
            component_statuses = [comp["status"] for comp in health_report["components"].values()]
            if "unhealthy" in component_statuses:
                health_report["overall_status"] = "unhealthy"
            elif "warning" in component_statuses:
                health_report["overall_status"] = "warning"
            
            return health_report
            
        except Exception as e:
            health_report["overall_status"] = "error"
            health_report["error"] = str(e)
            return health_report
    
    async def run_complete_workflow_demo(self) -> Dict[str, Any]:
        """Run a complete workflow demonstration"""
        logger.info("Starting complete workflow demonstration...")
        
        demo_results = {
            "start_time": time.time(),
            "steps": {},
            "overall_success": True
        }
        
        try:
            # Step 1: Process a sample PRD
            sample_prd = """
            # AI-Powered Task Management System
            
            ## Overview
            Create an intelligent task management system with local LLM capabilities.
            
            ## Core Features
            1. Automatic task decomposition and prioritization
            2. Intelligent research and knowledge synthesis
            3. Performance optimization and monitoring
            4. Autonomous failure recovery
            5. Continuous learning and improvement
            
            ## Technical Requirements
            - Local LLM integration for privacy
            - Vector database for knowledge storage
            - Real-time performance monitoring
            - Evolutionary optimization algorithms
            - Meta-learning capabilities
            """
            
            logger.info("Step 1: Processing PRD...")
            prd_result = await self.process_project_requirements(sample_prd)
            demo_results["steps"]["prd_processing"] = {
                "success": True,
                "tasks_generated": prd_result["total_tasks"],
                "processing_time": prd_result["processing_time"]
            }
            
            # Step 2: Research best practices
            logger.info("Step 2: Conducting research...")
            research_result = await self.research_topic(
                "Best practices for local LLM integration in autonomous systems",
                "Building an AI task management system"
            )
            demo_results["steps"]["research"] = {
                "success": True,
                "sources_found": len(research_result["sources"]),
                "confidence_score": research_result.get("synthesis", {}).get("confidence_score", 0)
            }
            
            # Step 3: Optimize system configuration
            logger.info("Step 3: Running optimization...")
            optimization_target = {
                "batch_size": 32,
                "timeout": 60,
                "cache_ttl": 3600,
                "max_concurrent": 4
            }
            
            parameter_ranges = {
                "batch_size": {"type": "int", "min": 1, "max": 100},
                "timeout": {"type": "int", "min": 10, "max": 300},
                "cache_ttl": {"type": "int", "min": 300, "max": 7200},
                "max_concurrent": {"type": "int", "min": 1, "max": 16}
            }
            
            optimization_result = await self.optimize_configuration(optimization_target, parameter_ranges)
            demo_results["steps"]["optimization"] = {
                "success": True,
                "final_generation": optimization_result["final_generation"],
                "best_fitness": optimization_result["best_individual"]["fitness"]
            }
            
            # Step 4: Run improvement cycle
            logger.info("Step 4: Running improvement cycle...")
            improvement_result = await self.autonomous_improvement_cycle()
            demo_results["steps"]["improvement"] = {
                "success": True,
                "recommendations_count": len(improvement_result["recommendations"].get("strategic_recommendations", [])),
                "analysis_confidence": improvement_result["meta_analysis"].get("confidence_score", 0)
            }
            
            # Step 5: Health check
            logger.info("Step 5: Performing health check...")
            health_result = await self.health_check()
            demo_results["steps"]["health_check"] = {
                "success": health_result["overall_status"] != "error",
                "overall_status": health_result["overall_status"],
                "healthy_components": sum(1 for comp in health_result["components"].values() if comp["status"] == "healthy")
            }
            
        except Exception as e:
            logger.error(f"Workflow demo failed: {e}")
            demo_results["overall_success"] = False
            demo_results["error"] = str(e)
        
        demo_results["end_time"] = time.time()
        demo_results["total_duration"] = demo_results["end_time"] - demo_results["start_time"]
        
        # Update system stats
        self.system_stats["operations_completed"] += 1
        
        logger.info(f"Workflow demonstration completed in {demo_results['total_duration']:.2f} seconds")
        return demo_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "initialized": self.initialized,
            "system_stats": self.system_stats.copy(),
            "deployment_mode": self.config_manager.deployment_mode.value if self.config_manager else "unknown",
            "model_count": len(self.config_manager.model_profiles) if self.config_manager else 0,
            "cache_stats": self.performance_monitor.cache.get_stats() if self.performance_monitor else {},
            "timestamp": datetime.now().isoformat()
        }

# Demonstration function
async def run_integration_demo():
    """Run the complete integration demonstration"""
    print("="*80)
    print("TASK MASTER AI LOCAL MODULES INTEGRATION DEMO")
    print("="*80)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize system
        system = TaskMasterLocalSystem()
        
        print("\n🚀 Initializing Task Master AI Local System...")
        initialization_success = await system.initialize()
        
        if not initialization_success:
            print("❌ System initialization failed!")
            return
        
        print("✅ System initialized successfully!")
        
        # Display system status
        status = system.get_system_status()
        print(f"\n📊 System Status:")
        print(f"  - Models configured: {status['model_count']}")
        print(f"  - Deployment mode: {status['deployment_mode']}")
        print(f"  - Cache size: {status['cache_stats'].get('size', 0)}")
        
        # Run complete workflow demonstration
        print("\n🔄 Running complete workflow demonstration...")
        demo_results = await system.run_complete_workflow_demo()
        
        # Display results
        print(f"\n📈 Demo Results:")
        print(f"  - Overall success: {'✅' if demo_results['overall_success'] else '❌'}")
        print(f"  - Total duration: {demo_results['total_duration']:.2f} seconds")
        
        for step_name, step_result in demo_results["steps"].items():
            status_emoji = "✅" if step_result["success"] else "❌"
            print(f"  - {step_name}: {status_emoji}")
        
        # Final health check
        print("\n🏥 Final health check...")
        health_report = await system.health_check()
        print(f"  - Overall status: {health_report['overall_status']}")
        print(f"  - Uptime: {health_report['metrics']['uptime']:.1f} seconds")
        print(f"  - Operations completed: {health_report['metrics']['operations_completed']}")
        
        # Save demo results
        results_file = Path(".taskmaster/local_modules/demo_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                "demo_results": demo_results,
                "health_report": health_report,
                "system_status": status
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        print("\n🎉 Integration demonstration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the integration demo
    asyncio.run(run_integration_demo())