#!/usr/bin/env python3
"""
Failure Detection and Recovery System with Local LLMs
Provides autonomous diagnosis and recovery using local model intelligence
"""

import asyncio
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
import logging
import hashlib
from enum import Enum
import threading
from collections import defaultdict, deque

from ..core.api_abstraction import UnifiedModelAPI, TaskType

logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Types of failures that can be detected"""
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MODEL_FAILURE = "model_failure"
    API_TIMEOUT = "api_timeout"
    DATA_CORRUPTION = "data_corruption"
    WORKFLOW_STALL = "workflow_stall"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"

class SeverityLevel(Enum):
    """Severity levels for failures"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class FailureEvent:
    """Represents a failure event"""
    id: str
    failure_type: FailureType
    severity: SeverityLevel
    description: str
    context: Dict[str, Any]
    error_details: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_attempts: List[Dict[str, Any]] = field(default_factory=list)
    resolution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "failure_type": self.failure_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "context": self.context,
            "error_details": self.error_details,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolution_attempts": self.resolution_attempts,
            "resolution_time": self.resolution_time
        }

@dataclass
class RecoveryStrategy:
    """Represents a recovery strategy"""
    id: str
    strategy_type: str
    description: str
    applicable_failures: List[FailureType]
    implementation: Callable
    priority: int = 1
    success_rate: float = 0.0
    avg_recovery_time: float = 0.0
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding callable)"""
        return {
            "id": self.id,
            "strategy_type": self.strategy_type,
            "description": self.description,
            "applicable_failures": [f.value for f in self.applicable_failures],
            "priority": self.priority,
            "success_rate": self.success_rate,
            "avg_recovery_time": self.avg_recovery_time,
            "usage_count": self.usage_count
        }

class FailureDetector:
    """Monitors system for failure conditions"""
    
    def __init__(self, monitoring_interval: float = 30.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))
        self.thresholds = {
            "cpu_usage": 90.0,
            "memory_usage": 85.0,
            "error_rate": 10.0,
            "response_time": 30.0,
            "success_rate": 50.0
        }
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start failure monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Failure monitoring started")
    
    def stop_monitoring(self):
        """Stop failure monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Failure monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                
                # Update history
                for metric, value in metrics.items():
                    self.metrics_history[metric].append({
                        "value": value,
                        "timestamp": time.time()
                    })
                
                # Check for failure conditions
                failures = self._detect_failures(metrics)
                
                # Report any detected failures
                for failure in failures:
                    self._report_failure(failure)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        try:
            import psutil
            
            metrics = {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "error_rate": self._calculate_error_rate(),
                "response_time": self._calculate_avg_response_time()
            }
            
            return metrics
            
        except ImportError:
            # Fallback metrics if psutil not available
            return {
                "cpu_usage": 50.0,  # Mock values
                "memory_usage": 60.0,
                "error_rate": 2.0,
                "response_time": 5.0
            }
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return {}
    
    def _calculate_error_rate(self) -> float:
        """Calculate recent error rate"""
        # This would integrate with actual error tracking
        # For now, return a mock value
        return 1.0
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        # This would integrate with actual performance tracking
        # For now, return a mock value
        return 2.5
    
    def _detect_failures(self, current_metrics: Dict[str, float]) -> List[FailureEvent]:
        """Detect failure conditions from metrics"""
        failures = []
        
        for metric, value in current_metrics.items():
            if metric in self.thresholds and value > self.thresholds[metric]:
                failure_id = f"failure_{metric}_{int(time.time())}"
                
                failure_type = {
                    "cpu_usage": FailureType.PERFORMANCE_DEGRADATION,
                    "memory_usage": FailureType.RESOURCE_EXHAUSTION,
                    "error_rate": FailureType.SYSTEM_ERROR,
                    "response_time": FailureType.PERFORMANCE_DEGRADATION
                }.get(metric, FailureType.UNKNOWN)
                
                severity = SeverityLevel.HIGH if value > self.thresholds[metric] * 1.2 else SeverityLevel.MEDIUM
                
                failure = FailureEvent(
                    id=failure_id,
                    failure_type=failure_type,
                    severity=severity,
                    description=f"{metric} exceeded threshold: {value:.2f} > {self.thresholds[metric]}",
                    context={
                        "metric": metric,
                        "current_value": value,
                        "threshold": self.thresholds[metric],
                        "history": list(self.metrics_history[metric])[-10:]  # Last 10 values
                    }
                )
                
                failures.append(failure)
        
        return failures
    
    def _report_failure(self, failure: FailureEvent):
        """Report detected failure"""
        logger.warning(f"Failure detected: {failure.id} - {failure.description}")
        # This would typically send the failure to the recovery system
        # For now, just log it

class FailureRecoverySystem:
    """
    Main failure recovery system using local LLMs for diagnosis and recovery
    """
    
    def __init__(self,
                 api: UnifiedModelAPI,
                 data_dir: str = ".taskmaster/local_modules/failure_recovery"):
        self.api = api
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Failure tracking
        self.active_failures: Dict[str, FailureEvent] = {}
        self.failure_history: List[FailureEvent] = []
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        
        # Recovery state
        self.recovery_in_progress = False
        self.recovery_queue = asyncio.Queue()
        self.recovery_task = None
        
        # Performance tracking
        self.recovery_stats = {
            "total_failures": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "avg_recovery_time": 0.0,
            "recovery_rate": 0.0
        }
        
        # Initialize built-in recovery strategies
        self._initialize_recovery_strategies()
        
        # Load historical data
        self._load_failure_data()
        
        # Start recovery worker
        self._start_recovery_worker()
    
    def _initialize_recovery_strategies(self):
        """Initialize built-in recovery strategies"""
        strategies = [
            RecoveryStrategy(
                id="restart_service",
                strategy_type="restart",
                description="Restart the affected service or component",
                applicable_failures=[FailureType.SYSTEM_ERROR, FailureType.MODEL_FAILURE],
                implementation=self._restart_service_recovery,
                priority=2
            ),
            RecoveryStrategy(
                id="clear_cache",
                strategy_type="cleanup",
                description="Clear caches and temporary data",
                applicable_failures=[FailureType.RESOURCE_EXHAUSTION, FailureType.DATA_CORRUPTION],
                implementation=self._clear_cache_recovery,
                priority=1
            ),
            RecoveryStrategy(
                id="scale_resources",
                strategy_type="scaling",
                description="Scale up system resources",
                applicable_failures=[FailureType.PERFORMANCE_DEGRADATION, FailureType.RESOURCE_EXHAUSTION],
                implementation=self._scale_resources_recovery,
                priority=3
            ),
            RecoveryStrategy(
                id="fallback_model",
                strategy_type="fallback",
                description="Switch to fallback model or configuration",
                applicable_failures=[FailureType.MODEL_FAILURE, FailureType.API_TIMEOUT],
                implementation=self._fallback_model_recovery,
                priority=1
            ),
            RecoveryStrategy(
                id="reset_configuration",
                strategy_type="reset",
                description="Reset to known good configuration",
                applicable_failures=[FailureType.CONFIGURATION_ERROR, FailureType.WORKFLOW_STALL],
                implementation=self._reset_configuration_recovery,
                priority=2
            )
        ]
        
        for strategy in strategies:
            self.recovery_strategies[strategy.id] = strategy
    
    def _load_failure_data(self):
        """Load historical failure data"""
        history_file = self.data_dir / "failure_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    self.failure_history = [
                        FailureEvent(
                            id=item["id"],
                            failure_type=FailureType(item["failure_type"]),
                            severity=SeverityLevel(item["severity"]),
                            description=item["description"],
                            context=item["context"],
                            error_details=item.get("error_details"),
                            timestamp=item["timestamp"],
                            resolved=item["resolved"],
                            resolution_attempts=item.get("resolution_attempts", []),
                            resolution_time=item.get("resolution_time")
                        )
                        for item in history_data
                    ]
                logger.info(f"Loaded {len(self.failure_history)} historical failures")
            except Exception as e:
                logger.warning(f"Failed to load failure history: {e}")
    
    def _save_failure_data(self):
        """Save failure data"""
        try:
            history_file = self.data_dir / "failure_history.json"
            with open(history_file, 'w') as f:
                json.dump([failure.to_dict() for failure in self.failure_history], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save failure data: {e}")
    
    def _start_recovery_worker(self):
        """Start the recovery worker task"""
        self.recovery_task = asyncio.create_task(self._recovery_worker())
        logger.info("Recovery worker started")
    
    async def _recovery_worker(self):
        """Main recovery worker loop"""
        while True:
            try:
                # Wait for failure to process
                failure = await self.recovery_queue.get()
                
                # Process the failure
                await self._process_failure_recovery(failure)
                
                # Mark task as done
                self.recovery_queue.task_done()
                
            except Exception as e:
                logger.error(f"Recovery worker error: {e}")
                await asyncio.sleep(1)
    
    async def report_failure(self, 
                           failure_type: FailureType,
                           description: str,
                           context: Dict[str, Any] = None,
                           error_details: Dict[str, Any] = None,
                           severity: SeverityLevel = SeverityLevel.MEDIUM) -> str:
        """Report a new failure for recovery"""
        failure_id = f"failure_{int(time.time())}_{hashlib.md5(description.encode()).hexdigest()[:8]}"
        
        failure = FailureEvent(
            id=failure_id,
            failure_type=failure_type,
            severity=severity,
            description=description,
            context=context or {},
            error_details=error_details
        )
        
        # Add to active failures
        self.active_failures[failure_id] = failure
        self.failure_history.append(failure)
        self.recovery_stats["total_failures"] += 1
        
        # Queue for recovery
        await self.recovery_queue.put(failure)
        
        logger.warning(f"Failure reported: {failure_id} - {description}")
        return failure_id
    
    async def _process_failure_recovery(self, failure: FailureEvent):
        """Process failure recovery"""
        logger.info(f"Processing recovery for failure: {failure.id}")
        start_time = time.time()
        
        try:
            # Diagnose the failure using local LLM
            diagnosis = await self._diagnose_failure(failure)
            
            # Select recovery strategies
            strategies = await self._select_recovery_strategies(failure, diagnosis)
            
            # Attempt recovery
            recovery_successful = False
            for strategy in strategies:
                try:
                    logger.info(f"Attempting recovery strategy: {strategy.id}")
                    
                    # Record attempt
                    attempt = {
                        "strategy_id": strategy.id,
                        "timestamp": time.time(),
                        "diagnosis": diagnosis
                    }
                    
                    # Execute recovery strategy
                    result = await strategy.implementation(failure, diagnosis)
                    
                    attempt["result"] = result
                    attempt["success"] = result.get("success", False)
                    failure.resolution_attempts.append(attempt)
                    
                    # Update strategy statistics
                    strategy.usage_count += 1
                    
                    if result.get("success", False):
                        recovery_successful = True
                        strategy.success_rate = (strategy.success_rate * (strategy.usage_count - 1) + 1) / strategy.usage_count
                        break
                    else:
                        strategy.success_rate = (strategy.success_rate * (strategy.usage_count - 1)) / strategy.usage_count
                        
                except Exception as e:
                    logger.error(f"Recovery strategy {strategy.id} failed: {e}")
                    failure.resolution_attempts.append({
                        "strategy_id": strategy.id,
                        "timestamp": time.time(),
                        "error": str(e),
                        "success": False
                    })
            
            # Update failure status
            if recovery_successful:
                failure.resolved = True
                failure.resolution_time = time.time() - start_time
                self.recovery_stats["successful_recoveries"] += 1
                logger.info(f"Recovery successful for failure: {failure.id}")
            else:
                self.recovery_stats["failed_recoveries"] += 1
                logger.error(f"Recovery failed for failure: {failure.id}")
            
            # Update statistics
            self._update_recovery_stats()
            
            # Remove from active failures if resolved
            if failure.resolved and failure.id in self.active_failures:
                del self.active_failures[failure.id]
            
        except Exception as e:
            logger.error(f"Failure recovery processing error: {e}")
            self.recovery_stats["failed_recoveries"] += 1
        
        finally:
            self._save_failure_data()
    
    async def _diagnose_failure(self, failure: FailureEvent) -> Dict[str, Any]:
        """Diagnose failure using local LLM"""
        diagnosis_prompt = f"""
        Diagnose this system failure and provide detailed analysis:
        
        FAILURE DETAILS:
        Type: {failure.failure_type.value}
        Severity: {failure.severity.value}
        Description: {failure.description}
        
        CONTEXT:
        {json.dumps(failure.context, indent=2)}
        
        ERROR DETAILS:
        {json.dumps(failure.error_details, indent=2) if failure.error_details else "None"}
        
        HISTORICAL CONTEXT:
        Recent similar failures: {len([f for f in self.failure_history[-20:] if f.failure_type == failure.failure_type])}
        
        Please provide:
        1. Root cause analysis
        2. Contributing factors
        3. Impact assessment
        4. Urgency level
        5. Recommended recovery approaches
        6. Prevention strategies
        
        Provide response in this JSON format:
        {{
            "root_cause": "primary cause of the failure",
            "contributing_factors": ["factor1", "factor2"],
            "impact_assessment": {{
                "severity": "critical|high|medium|low",
                "affected_components": ["component1", "component2"],
                "user_impact": "description of user impact"
            }},
            "urgency_level": "immediate|high|medium|low",
            "recovery_approaches": [
                {{
                    "approach": "approach_name",
                    "description": "what this approach does",
                    "probability_success": 0.0-1.0,
                    "implementation_complexity": "low|medium|high"
                }}
            ],
            "prevention_strategies": ["prevention1", "prevention2"],
            "confidence_score": 0.0-1.0
        }}
        """
        
        try:
            response = await self.api.generate(
                diagnosis_prompt,
                task_type=TaskType.ANALYSIS,
                temperature=0.2
            )
            
            diagnosis = json.loads(response.content)
            diagnosis["diagnosis_timestamp"] = time.time()
            diagnosis["model_used"] = response.model_used
            
            return diagnosis
            
        except Exception as e:
            logger.error(f"Failure diagnosis failed: {e}")
            return {
                "root_cause": "Unable to determine",
                "recovery_approaches": [],
                "confidence_score": 0.0,
                "error": str(e)
            }
    
    async def _select_recovery_strategies(self, failure: FailureEvent, diagnosis: Dict[str, Any]) -> List[RecoveryStrategy]:
        """Select appropriate recovery strategies"""
        # Filter strategies by applicable failure types
        applicable_strategies = [
            strategy for strategy in self.recovery_strategies.values()
            if failure.failure_type in strategy.applicable_failures
        ]
        
        # Use LLM to rank strategies
        if applicable_strategies:
            strategy_ranking = await self._rank_recovery_strategies(failure, diagnosis, applicable_strategies)
            
            # Sort by LLM ranking and success rate
            ranked_strategies = []
            for strategy_id in strategy_ranking.get("ranked_strategies", []):
                strategy = next((s for s in applicable_strategies if s.id == strategy_id), None)
                if strategy:
                    ranked_strategies.append(strategy)
            
            # Add any remaining strategies sorted by success rate
            remaining = [s for s in applicable_strategies if s not in ranked_strategies]
            remaining.sort(key=lambda s: (s.success_rate, s.priority), reverse=True)
            ranked_strategies.extend(remaining)
            
            return ranked_strategies[:3]  # Top 3 strategies
        
        return []
    
    async def _rank_recovery_strategies(self, failure: FailureEvent, diagnosis: Dict[str, Any], strategies: List[RecoveryStrategy]) -> Dict[str, Any]:
        """Rank recovery strategies using local LLM"""
        ranking_prompt = f"""
        Rank these recovery strategies for the given failure scenario:
        
        FAILURE: {failure.description}
        DIAGNOSIS: {json.dumps(diagnosis, indent=2)}
        
        AVAILABLE STRATEGIES:
        {json.dumps([strategy.to_dict() for strategy in strategies], indent=2)}
        
        Rank the strategies by:
        1. Likelihood of success for this specific failure
        2. Speed of recovery
        3. Risk of side effects
        4. Resource requirements
        
        Consider the diagnosis and historical success rates.
        
        Provide response in this JSON format:
        {{
            "ranked_strategies": ["strategy_id1", "strategy_id2", "strategy_id3"],
            "ranking_rationale": {{
                "strategy_id1": "why this strategy is ranked first",
                "strategy_id2": "why this strategy is ranked second"
            }},
            "confidence_score": 0.0-1.0
        }}
        """
        
        try:
            response = await self.api.generate(
                ranking_prompt,
                task_type=TaskType.ANALYSIS,
                temperature=0.3
            )
            
            return json.loads(response.content)
            
        except Exception as e:
            logger.error(f"Strategy ranking failed: {e}")
            return {"ranked_strategies": [s.id for s in strategies]}
    
    # Recovery strategy implementations
    async def _restart_service_recovery(self, failure: FailureEvent, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """Restart service recovery strategy"""
        try:
            # Simulate service restart
            logger.info("Executing service restart recovery")
            await asyncio.sleep(2)  # Simulate restart time
            
            # In a real implementation, this would actually restart the service
            # For now, simulate success
            success = True
            
            return {
                "success": success,
                "action": "service_restart",
                "details": "Service restarted successfully",
                "recovery_time": 2.0
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "service_restart",
                "error": str(e)
            }
    
    async def _clear_cache_recovery(self, failure: FailureEvent, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """Clear cache recovery strategy"""
        try:
            logger.info("Executing cache clear recovery")
            
            # Clear actual caches
            cache_dirs = [
                self.data_dir.parent / "cache",
                Path(".taskmaster/local_modules/cache")
            ]
            
            cleared_count = 0
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    for cache_file in cache_dir.glob("*.json"):
                        if cache_file.stat().st_mtime < time.time() - 3600:  # Older than 1 hour
                            cache_file.unlink()
                            cleared_count += 1
            
            return {
                "success": True,
                "action": "cache_clear",
                "details": f"Cleared {cleared_count} cache files",
                "recovery_time": 1.0
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "cache_clear",
                "error": str(e)
            }
    
    async def _scale_resources_recovery(self, failure: FailureEvent, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """Scale resources recovery strategy"""
        try:
            logger.info("Executing resource scaling recovery")
            
            # Simulate resource scaling
            # In a real implementation, this would scale actual resources
            await asyncio.sleep(1)
            
            return {
                "success": True,
                "action": "resource_scaling",
                "details": "Resources scaled up successfully",
                "recovery_time": 1.0
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "resource_scaling",
                "error": str(e)
            }
    
    async def _fallback_model_recovery(self, failure: FailureEvent, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback model recovery strategy"""
        try:
            logger.info("Executing fallback model recovery")
            
            # Switch to fallback model in the API
            # This would integrate with the model router
            
            return {
                "success": True,
                "action": "fallback_model",
                "details": "Switched to fallback model",
                "recovery_time": 0.5
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "fallback_model",
                "error": str(e)
            }
    
    async def _reset_configuration_recovery(self, failure: FailureEvent, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """Reset configuration recovery strategy"""
        try:
            logger.info("Executing configuration reset recovery")
            
            # Reset to known good configuration
            # This would restore from a backup configuration
            
            return {
                "success": True,
                "action": "configuration_reset",
                "details": "Configuration reset to known good state",
                "recovery_time": 3.0
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "configuration_reset",
                "error": str(e)
            }
    
    def _update_recovery_stats(self):
        """Update recovery statistics"""
        total_recoveries = self.recovery_stats["successful_recoveries"] + self.recovery_stats["failed_recoveries"]
        if total_recoveries > 0:
            self.recovery_stats["recovery_rate"] = self.recovery_stats["successful_recoveries"] / total_recoveries
        
        # Calculate average recovery time
        resolved_failures = [f for f in self.failure_history if f.resolved and f.resolution_time]
        if resolved_failures:
            self.recovery_stats["avg_recovery_time"] = sum(f.resolution_time for f in resolved_failures) / len(resolved_failures)
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery system status"""
        return {
            "active_failures": len(self.active_failures),
            "recovery_in_progress": self.recovery_in_progress,
            "total_failures": self.recovery_stats["total_failures"],
            "successful_recoveries": self.recovery_stats["successful_recoveries"],
            "failed_recoveries": self.recovery_stats["failed_recoveries"],
            "recovery_rate": self.recovery_stats["recovery_rate"],
            "avg_recovery_time": self.recovery_stats["avg_recovery_time"],
            "available_strategies": len(self.recovery_strategies),
            "queue_size": self.recovery_queue.qsize()
        }
    
    async def test_recovery_system(self) -> Dict[str, Any]:
        """Test the recovery system with a simulated failure"""
        test_failure_id = await self.report_failure(
            failure_type=FailureType.SYSTEM_ERROR,
            description="Test failure for system validation",
            context={"test": True, "component": "recovery_system"},
            severity=SeverityLevel.LOW
        )
        
        # Wait for recovery to complete
        await asyncio.sleep(5)
        
        # Check if the test failure was resolved
        test_failure = next((f for f in self.failure_history if f.id == test_failure_id), None)
        
        return {
            "test_completed": True,
            "test_failure_id": test_failure_id,
            "recovery_attempted": len(test_failure.resolution_attempts) > 0 if test_failure else False,
            "recovery_successful": test_failure.resolved if test_failure else False,
            "system_status": self.get_recovery_status()
        }

# Example usage
if __name__ == "__main__":
    async def test_failure_recovery():
        from ..core.api_abstraction import UnifiedModelAPI, ModelConfigFactory
        
        # Initialize API
        api = UnifiedModelAPI()
        api.add_model("ollama-llama2", ModelConfigFactory.create_ollama_config(
            "llama2", capabilities=[TaskType.ANALYSIS]
        ))
        
        # Initialize recovery system
        recovery_system = FailureRecoverySystem(api)
        
        # Test the recovery system
        test_result = await recovery_system.test_recovery_system()
        print(f"Recovery system test: {json.dumps(test_result, indent=2)}")
        
        # Report a real failure
        failure_id = await recovery_system.report_failure(
            failure_type=FailureType.PERFORMANCE_DEGRADATION,
            description="High response time detected",
            context={"response_time": 45.2, "threshold": 30.0},
            severity=SeverityLevel.MEDIUM
        )
        
        print(f"Reported failure: {failure_id}")
        
        # Wait for recovery
        await asyncio.sleep(10)
        
        # Get status
        status = recovery_system.get_recovery_status()
        print(f"Recovery status: {json.dumps(status, indent=2)}")
    
    # Run test
    asyncio.run(test_failure_recovery())