#!/usr/bin/env python3
"""
Phased Deployment Strategy with Real-Time Monitoring

Comprehensive deployment system featuring:
- Multi-phase rollout with validation gates
- Real-time health monitoring and alerting
- Automated rollback capabilities
- Progressive traffic shifting
- Comprehensive deployment validation
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
import hashlib
import shutil
import signal
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import uuid
import statistics
from collections import defaultdict, deque
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentPhase(Enum):
    """Deployment phases in order"""
    PREPARATION = "preparation"
    VALIDATION = "validation"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    PROGRESSIVE_ROLLOUT = "progressive_rollout"
    FULL_DEPLOYMENT = "full_deployment"
    MONITORING = "monitoring"
    COMPLETION = "completion"

class DeploymentStatus(Enum):
    """Deployment status types"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"

class HealthCheckStatus(Enum):
    """Health check results"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class HealthMetrics:
    """System health metrics"""
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    response_time_ms: float = 0.0
    error_rate_percent: float = 0.0
    throughput_requests_per_second: float = 0.0
    autonomy_score: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ValidationResult:
    """Validation checkpoint result"""
    checkpoint_name: str
    status: bool
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    execution_time_seconds: float = 0.0

@dataclass
class DeploymentPhaseConfig:
    """Configuration for a deployment phase"""
    phase: DeploymentPhase
    name: str
    description: str
    traffic_percentage: float = 0.0
    duration_minutes: int = 30
    validation_checks: List[str] = field(default_factory=list)
    rollback_on_failure: bool = True
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    health_check_interval_seconds: int = 30
    max_error_rate_percent: float = 5.0
    max_response_time_ms: float = 5000.0
    min_autonomy_score: float = 0.9

@dataclass
class Alert:
    """System alert"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    component: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    resolved: bool = False

class HealthMonitor:
    """Real-time health monitoring system"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.metrics_history = deque(maxlen=1000)
        self.alerts = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.health_checks = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_usage_percent': 80.0,
            'memory_usage_percent': 85.0,
            'disk_usage_percent': 90.0,
            'response_time_ms': 5000.0,
            'error_rate_percent': 5.0,
            'autonomy_score': 0.9
        }
        
        self.callbacks = []
    
    def register_health_check(self, name: str, check_function: Callable[[], Dict[str, Any]]):
        """Register a custom health check function"""
        self.health_checks[name] = check_function
        logger.info(f"Health check registered: {name}")
    
    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """Register callback for alert notifications"""
        self.callbacks.append(callback)
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self._collect_health_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _collect_health_metrics(self) -> HealthMetrics:
        """Collect comprehensive health metrics"""
        metrics = HealthMetrics()
        
        try:
            # System metrics (simplified - would use actual system calls)
            metrics.cpu_usage_percent = self._get_cpu_usage()
            metrics.memory_usage_percent = self._get_memory_usage()
            metrics.disk_usage_percent = self._get_disk_usage()
            
            # Application metrics
            app_metrics = self._collect_application_metrics()
            metrics.response_time_ms = app_metrics.get('response_time_ms', 0)
            metrics.error_rate_percent = app_metrics.get('error_rate_percent', 0)
            metrics.throughput_requests_per_second = app_metrics.get('throughput_rps', 0)
            metrics.autonomy_score = app_metrics.get('autonomy_score', 0.95)
            metrics.active_tasks = app_metrics.get('active_tasks', 0)
            metrics.completed_tasks = app_metrics.get('completed_tasks', 0)
            metrics.failed_tasks = app_metrics.get('failed_tasks', 0)
            
            # Custom health checks
            for name, check_func in self.health_checks.items():
                try:
                    custom_metrics = check_func()
                    # Merge custom metrics
                    for key, value in custom_metrics.items():
                        setattr(metrics, key, value)
                except Exception as e:
                    logger.warning(f"Custom health check {name} failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to collect health metrics: {e}")
        
        return metrics
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage (simplified)"""
        try:
            # Simulate CPU usage (would use actual system monitoring)
            import random
            return random.uniform(10, 30)  # Simulate normal CPU usage
        except:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage (simplified)"""
        try:
            # Simulate memory usage
            import random
            return random.uniform(40, 60)  # Simulate normal memory usage
        except:
            return 0.0
    
    def _get_disk_usage(self) -> float:
        """Get disk usage percentage (simplified)"""
        try:
            # Get actual disk usage for current directory
            import shutil
            total, used, free = shutil.disk_usage(os.getcwd())
            return (used / total) * 100
        except:
            return 0.0
    
    def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics"""
        # Simulate application metrics (would integrate with actual app)
        import random
        
        return {
            'response_time_ms': random.uniform(100, 500),
            'error_rate_percent': random.uniform(0, 2),
            'throughput_rps': random.uniform(50, 200),
            'autonomy_score': random.uniform(0.92, 0.98),
            'active_tasks': random.randint(5, 15),
            'completed_tasks': random.randint(50, 100),
            'failed_tasks': random.randint(0, 3)
        }
    
    def _check_alerts(self, metrics: HealthMetrics):
        """Check metrics against alert thresholds"""
        current_time = time.time()
        
        # Check each threshold
        for metric_name, threshold in self.alert_thresholds.items():
            metric_value = getattr(metrics, metric_name, 0)
            
            # Determine if alert should be triggered
            should_alert = False
            severity = AlertSeverity.WARNING
            
            if metric_name == 'autonomy_score':
                should_alert = metric_value < threshold
                severity = AlertSeverity.ERROR if metric_value < 0.85 else AlertSeverity.WARNING
            else:
                should_alert = metric_value > threshold
                if metric_value > threshold * 1.2:
                    severity = AlertSeverity.CRITICAL
                elif metric_value > threshold * 1.1:
                    severity = AlertSeverity.ERROR
            
            if should_alert:
                # Check if we already have a recent alert for this metric
                recent_alerts = [
                    alert for alert in self.alerts[-10:]
                    if alert.component == metric_name and 
                    current_time - alert.timestamp < 300  # 5 minutes
                ]
                
                if not recent_alerts:
                    alert = Alert(
                        alert_id=str(uuid.uuid4())[:8],
                        severity=severity,
                        title=f"High {metric_name.replace('_', ' ').title()}",
                        message=f"{metric_name} is {metric_value:.2f}, threshold: {threshold}",
                        component=metric_name,
                        metrics=asdict(metrics)
                    )
                    
                    self.alerts.append(alert)
                    self._notify_alert(alert)
    
    def _notify_alert(self, alert: Alert):
        """Notify registered callbacks about alert"""
        logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.title} - {alert.message}")
        
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def get_current_health_status(self) -> HealthCheckStatus:
        """Get overall system health status"""
        if not self.metrics_history:
            return HealthCheckStatus.UNKNOWN
        
        latest_metrics = self.metrics_history[-1]
        
        # Check critical thresholds
        critical_issues = 0
        warning_issues = 0
        
        # CPU check
        if latest_metrics.cpu_usage_percent > 90:
            critical_issues += 1
        elif latest_metrics.cpu_usage_percent > 80:
            warning_issues += 1
        
        # Memory check
        if latest_metrics.memory_usage_percent > 95:
            critical_issues += 1
        elif latest_metrics.memory_usage_percent > 85:
            warning_issues += 1
        
        # Error rate check
        if latest_metrics.error_rate_percent > 10:
            critical_issues += 1
        elif latest_metrics.error_rate_percent > 5:
            warning_issues += 1
        
        # Autonomy score check
        if latest_metrics.autonomy_score < 0.85:
            critical_issues += 1
        elif latest_metrics.autonomy_score < 0.9:
            warning_issues += 1
        
        # Determine overall status
        if critical_issues > 0:
            return HealthCheckStatus.UNHEALTHY
        elif warning_issues > 2:
            return HealthCheckStatus.DEGRADED
        elif warning_issues > 0:
            return HealthCheckStatus.DEGRADED
        else:
            return HealthCheckStatus.HEALTHY
    
    def get_metrics_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get metrics summary for the last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        return {
            'timeframe_minutes': minutes,
            'sample_count': len(recent_metrics),
            'avg_cpu_usage': statistics.mean(m.cpu_usage_percent for m in recent_metrics),
            'avg_memory_usage': statistics.mean(m.memory_usage_percent for m in recent_metrics),
            'avg_response_time': statistics.mean(m.response_time_ms for m in recent_metrics),
            'avg_error_rate': statistics.mean(m.error_rate_percent for m in recent_metrics),
            'avg_autonomy_score': statistics.mean(m.autonomy_score for m in recent_metrics),
            'max_response_time': max(m.response_time_ms for m in recent_metrics),
            'max_error_rate': max(m.error_rate_percent for m in recent_metrics),
            'min_autonomy_score': min(m.autonomy_score for m in recent_metrics)
        }

class DeploymentValidator:
    """Validates deployment phases and system state"""
    
    def __init__(self, workspace_path: str = ".taskmaster"):
        self.workspace_path = Path(workspace_path)
        self.validation_results = []
    
    def validate_pre_deployment(self) -> ValidationResult:
        """Validate system readiness for deployment"""
        logger.info("Running pre-deployment validation...")
        
        start_time = time.time()
        checks_passed = 0
        total_checks = 5
        issues = []
        
        try:
            # Check 1: Workspace structure
            if self._validate_workspace_structure():
                checks_passed += 1
            else:
                issues.append("Workspace structure incomplete")
            
            # Check 2: Required files present
            if self._validate_required_files():
                checks_passed += 1
            else:
                issues.append("Required files missing")
            
            # Check 3: Dependencies available
            if self._validate_dependencies():
                checks_passed += 1
            else:
                issues.append("Dependencies not satisfied")
            
            # Check 4: Configuration valid
            if self._validate_configuration():
                checks_passed += 1
            else:
                issues.append("Configuration invalid")
            
            # Check 5: System resources
            if self._validate_system_resources():
                checks_passed += 1
            else:
                issues.append("Insufficient system resources")
            
            success = checks_passed == total_checks
            message = f"Pre-deployment validation: {checks_passed}/{total_checks} checks passed"
            if issues:
                message += f". Issues: {', '.join(issues)}"
            
            result = ValidationResult(
                checkpoint_name="pre_deployment",
                status=success,
                message=message,
                metrics={
                    'checks_passed': checks_passed,
                    'total_checks': total_checks,
                    'issues': issues
                },
                execution_time_seconds=time.time() - start_time
            )
            
            self.validation_results.append(result)
            return result
            
        except Exception as e:
            return ValidationResult(
                checkpoint_name="pre_deployment",
                status=False,
                message=f"Pre-deployment validation failed: {str(e)}",
                execution_time_seconds=time.time() - start_time
            )
    
    def validate_canary_deployment(self, health_monitor: HealthMonitor) -> ValidationResult:
        """Validate canary deployment health"""
        logger.info("Validating canary deployment...")
        
        start_time = time.time()
        
        try:
            # Get recent health metrics
            health_summary = health_monitor.get_metrics_summary(minutes=2)
            current_status = health_monitor.get_current_health_status()
            
            # Check canary-specific criteria
            success_criteria = {
                'error_rate_below_threshold': health_summary.get('avg_error_rate', 0) < 5.0,
                'response_time_acceptable': health_summary.get('avg_response_time', 0) < 3000.0,
                'autonomy_score_maintained': health_summary.get('avg_autonomy_score', 0) > 0.9,
                'overall_health_good': current_status in [HealthCheckStatus.HEALTHY, HealthCheckStatus.DEGRADED]
            }
            
            success = all(success_criteria.values())
            
            return ValidationResult(
                checkpoint_name="canary_deployment",
                status=success,
                message=f"Canary validation: {'PASSED' if success else 'FAILED'}",
                metrics={
                    'success_criteria': success_criteria,
                    'health_summary': health_summary,
                    'current_health_status': current_status.value
                },
                execution_time_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                checkpoint_name="canary_deployment",
                status=False,
                message=f"Canary validation failed: {str(e)}",
                execution_time_seconds=time.time() - start_time
            )
    
    def validate_full_deployment(self, health_monitor: HealthMonitor) -> ValidationResult:
        """Validate full deployment success"""
        logger.info("Validating full deployment...")
        
        start_time = time.time()
        
        try:
            # Extended validation for full deployment
            health_summary = health_monitor.get_metrics_summary(minutes=5)
            current_status = health_monitor.get_current_health_status()
            
            # Stricter criteria for full deployment
            success_criteria = {
                'error_rate_low': health_summary.get('avg_error_rate', 0) < 2.0,
                'response_time_good': health_summary.get('avg_response_time', 0) < 2000.0,
                'autonomy_score_high': health_summary.get('avg_autonomy_score', 0) > 0.95,
                'system_healthy': current_status == HealthCheckStatus.HEALTHY,
                'no_critical_alerts': len([a for a in health_monitor.alerts[-10:] 
                                         if a.severity == AlertSeverity.CRITICAL]) == 0
            }
            
            success = all(success_criteria.values())
            
            return ValidationResult(
                checkpoint_name="full_deployment",
                status=success,
                message=f"Full deployment validation: {'PASSED' if success else 'FAILED'}",
                metrics={
                    'success_criteria': success_criteria,
                    'health_summary': health_summary,
                    'current_health_status': current_status.value
                },
                execution_time_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                checkpoint_name="full_deployment",
                status=False,
                message=f"Full deployment validation failed: {str(e)}",
                execution_time_seconds=time.time() - start_time
            )
    
    def _validate_workspace_structure(self) -> bool:
        """Validate workspace directory structure"""
        required_dirs = [
            "scripts", "reports", "logs", "checkpoints", 
            "config", "monitoring", "validation"
        ]
        
        for directory in required_dirs:
            if not (self.workspace_path / directory).exists():
                logger.warning(f"Missing directory: {directory}")
                return False
        
        return True
    
    def _validate_required_files(self) -> bool:
        """Validate required files are present"""
        required_files = [
            "scripts/unified-execution-framework.py",
            "scripts/performance-optimizer.py",
            "scripts/intelligent-task-prioritizer.py"
        ]
        
        for file_path in required_files:
            if not (self.workspace_path / file_path).exists():
                logger.warning(f"Missing required file: {file_path}")
                return False
        
        return True
    
    def _validate_dependencies(self) -> bool:
        """Validate system dependencies"""
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                logger.warning("Python version too old (< 3.8)")
                return False
            
            # Check required modules are importable
            required_modules = ['json', 'time', 'threading', 'pathlib']
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    logger.warning(f"Required module not available: {module}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Dependency validation failed: {e}")
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate system configuration"""
        try:
            # Check if configuration files are valid JSON
            config_files = [
                "config/unified_framework_config.json"
            ]
            
            for config_file in config_files:
                config_path = self.workspace_path / config_file
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in config file: {config_file}")
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Configuration validation failed: {e}")
            return False
    
    def _validate_system_resources(self) -> bool:
        """Validate sufficient system resources"""
        try:
            # Check disk space
            total, used, free = shutil.disk_usage(self.workspace_path)
            free_gb = free / (1024**3)
            
            if free_gb < 1.0:  # Need at least 1GB free
                logger.warning(f"Insufficient disk space: {free_gb:.1f}GB free")
                return False
            
            # Check if we can create test files
            test_file = self.workspace_path / "test_write_access.tmp"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()
            except Exception:
                logger.warning("Cannot write to workspace directory")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Resource validation failed: {e}")
            return False

class PhasedDeploymentManager:
    """Main phased deployment management system"""
    
    def __init__(self, workspace_path: str = ".taskmaster"):
        self.workspace_path = Path(workspace_path)
        self.deployment_id = str(uuid.uuid4())[:8]
        
        # Initialize components
        self.health_monitor = HealthMonitor()
        self.validator = DeploymentValidator(workspace_path)
        
        # Deployment state
        self.current_phase = DeploymentPhase.PREPARATION
        self.deployment_status = DeploymentStatus.PENDING
        self.phase_start_time = None
        self.deployment_start_time = None
        
        # Configuration
        self.phases_config = self._create_default_phases_config()
        self.rollback_data = {}
        
        # Callbacks for phase transitions
        self.phase_callbacks = {}
        
        # Deployment history
        self.deployment_history = []
        
        # Initialize workspace
        self._initialize_deployment_workspace()
        
        logger.info(f"Phased deployment manager initialized (ID: {self.deployment_id})")
    
    def _create_default_phases_config(self) -> Dict[DeploymentPhase, DeploymentPhaseConfig]:
        """Create default configuration for deployment phases"""
        return {
            DeploymentPhase.PREPARATION: DeploymentPhaseConfig(
                phase=DeploymentPhase.PREPARATION,
                name="Preparation",
                description="Pre-deployment validation and setup",
                duration_minutes=10,
                validation_checks=["pre_deployment"],
                success_criteria={"validation_passed": True}
            ),
            
            DeploymentPhase.VALIDATION: DeploymentPhaseConfig(
                phase=DeploymentPhase.VALIDATION,
                name="System Validation",
                description="Comprehensive system validation before deployment",
                duration_minutes=15,
                validation_checks=["system_health", "component_readiness"],
                success_criteria={"all_validations_passed": True}
            ),
            
            DeploymentPhase.CANARY: DeploymentPhaseConfig(
                phase=DeploymentPhase.CANARY,
                name="Canary Deployment",
                description="Deploy to small subset (5% traffic)",
                traffic_percentage=5.0,
                duration_minutes=20,
                validation_checks=["canary_health"],
                success_criteria={
                    "error_rate_below": 5.0,
                    "response_time_below": 3000.0,
                    "autonomy_score_above": 0.9
                }
            ),
            
            DeploymentPhase.BLUE_GREEN: DeploymentPhaseConfig(
                phase=DeploymentPhase.BLUE_GREEN,
                name="Blue-Green Switch",
                description="Switch traffic to new deployment (50% traffic)",
                traffic_percentage=50.0,
                duration_minutes=30,
                validation_checks=["blue_green_health"],
                success_criteria={
                    "error_rate_below": 3.0,
                    "response_time_below": 2500.0,
                    "autonomy_score_above": 0.92
                }
            ),
            
            DeploymentPhase.PROGRESSIVE_ROLLOUT: DeploymentPhaseConfig(
                phase=DeploymentPhase.PROGRESSIVE_ROLLOUT,
                name="Progressive Rollout",
                description="Gradually increase traffic to 100%",
                traffic_percentage=100.0,
                duration_minutes=45,
                validation_checks=["progressive_health"],
                success_criteria={
                    "error_rate_below": 2.0,
                    "response_time_below": 2000.0,
                    "autonomy_score_above": 0.95
                }
            ),
            
            DeploymentPhase.FULL_DEPLOYMENT: DeploymentPhaseConfig(
                phase=DeploymentPhase.FULL_DEPLOYMENT,
                name="Full Deployment",
                description="Complete deployment with full traffic",
                traffic_percentage=100.0,
                duration_minutes=30,
                validation_checks=["full_deployment"],
                success_criteria={
                    "error_rate_below": 1.0,
                    "response_time_below": 1500.0,
                    "autonomy_score_above": 0.95,
                    "system_stable": True
                }
            ),
            
            DeploymentPhase.MONITORING: DeploymentPhaseConfig(
                phase=DeploymentPhase.MONITORING,
                name="Post-Deployment Monitoring",
                description="Extended monitoring and validation",
                traffic_percentage=100.0,
                duration_minutes=60,
                validation_checks=["extended_monitoring"],
                success_criteria={
                    "sustained_performance": True,
                    "no_critical_alerts": True
                }
            ),
            
            DeploymentPhase.COMPLETION: DeploymentPhaseConfig(
                phase=DeploymentPhase.COMPLETION,
                name="Deployment Completion",
                description="Finalize deployment and cleanup",
                duration_minutes=5,
                success_criteria={"deployment_finalized": True}
            )
        }
    
    def _initialize_deployment_workspace(self):
        """Initialize deployment workspace"""
        deployment_dir = self.workspace_path / "deployments" / self.deployment_id
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create deployment subdirectories
        subdirs = ["logs", "configs", "backups", "reports"]
        for subdir in subdirs:
            (deployment_dir / subdir).mkdir(exist_ok=True)
        
        # Setup logging for this deployment
        log_file = deployment_dir / "logs" / "deployment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Deployment workspace initialized: {deployment_dir}")
    
    def register_phase_callback(self, phase: DeploymentPhase, callback: Callable[[DeploymentPhase, bool], None]):
        """Register callback for phase completion"""
        if phase not in self.phase_callbacks:
            self.phase_callbacks[phase] = []
        self.phase_callbacks[phase].append(callback)
    
    def execute_deployment(self, deployment_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute complete phased deployment"""
        logger.info(f"Starting phased deployment (ID: {self.deployment_id})")
        
        self.deployment_start_time = time.time()
        self.deployment_status = DeploymentStatus.IN_PROGRESS
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
        
        try:
            # Execute each phase in sequence
            phases_order = [
                DeploymentPhase.PREPARATION,
                DeploymentPhase.VALIDATION,
                DeploymentPhase.CANARY,
                DeploymentPhase.BLUE_GREEN,
                DeploymentPhase.PROGRESSIVE_ROLLOUT,
                DeploymentPhase.FULL_DEPLOYMENT,
                DeploymentPhase.MONITORING,
                DeploymentPhase.COMPLETION
            ]
            
            for phase in phases_order:
                result = self._execute_phase(phase)
                
                if not result['success']:
                    logger.error(f"Phase {phase.value} failed: {result['error']}")
                    
                    # Attempt rollback if configured
                    if self.phases_config[phase].rollback_on_failure:
                        rollback_result = self._execute_rollback(phase)
                        if rollback_result['success']:
                            self.deployment_status = DeploymentStatus.ROLLED_BACK
                        else:
                            self.deployment_status = DeploymentStatus.FAILED
                    else:
                        self.deployment_status = DeploymentStatus.FAILED
                    
                    return self._generate_deployment_report(success=False, failed_phase=phase)
                
                # Execute phase callbacks
                self._execute_phase_callbacks(phase, True)
            
            # Deployment completed successfully
            self.deployment_status = DeploymentStatus.COMPLETED
            logger.info("Phased deployment completed successfully")
            
            return self._generate_deployment_report(success=True)
            
        except Exception as e:
            logger.error(f"Deployment failed with exception: {e}")
            self.deployment_status = DeploymentStatus.FAILED
            return self._generate_deployment_report(success=False, error=str(e))
        
        finally:
            # Stop health monitoring
            self.health_monitor.stop_monitoring()
    
    def _execute_phase(self, phase: DeploymentPhase) -> Dict[str, Any]:
        """Execute a single deployment phase"""
        phase_config = self.phases_config[phase]
        
        logger.info(f"Executing phase: {phase_config.name}")
        self.current_phase = phase
        self.phase_start_time = time.time()
        
        try:
            # Phase-specific execution logic
            if phase == DeploymentPhase.PREPARATION:
                result = self._execute_preparation_phase()
            elif phase == DeploymentPhase.VALIDATION:
                result = self._execute_validation_phase()
            elif phase == DeploymentPhase.CANARY:
                result = self._execute_canary_phase()
            elif phase == DeploymentPhase.BLUE_GREEN:
                result = self._execute_blue_green_phase()
            elif phase == DeploymentPhase.PROGRESSIVE_ROLLOUT:
                result = self._execute_progressive_rollout_phase()
            elif phase == DeploymentPhase.FULL_DEPLOYMENT:
                result = self._execute_full_deployment_phase()
            elif phase == DeploymentPhase.MONITORING:
                result = self._execute_monitoring_phase()
            elif phase == DeploymentPhase.COMPLETION:
                result = self._execute_completion_phase()
            else:
                result = {"success": False, "error": f"Unknown phase: {phase}"}
            
            # Record phase execution
            phase_duration = time.time() - self.phase_start_time
            self.deployment_history.append({
                'phase': phase.value,
                'start_time': self.phase_start_time,
                'duration_seconds': phase_duration,
                'success': result['success'],
                'metrics': result.get('metrics', {})
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Phase {phase.value} execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_preparation_phase(self) -> Dict[str, Any]:
        """Execute preparation phase"""
        logger.info("Running preparation phase...")
        
        # Run pre-deployment validation
        validation_result = self.validator.validate_pre_deployment()
        
        if not validation_result.status:
            return {
                "success": False,
                "error": f"Pre-deployment validation failed: {validation_result.message}",
                "validation_result": asdict(validation_result)
            }
        
        # Create backup of current state
        self._create_deployment_backup()
        
        # Initialize deployment artifacts
        self._initialize_deployment_artifacts()
        
        return {
            "success": True,
            "metrics": {
                "validation_passed": True,
                "backup_created": True,
                "artifacts_initialized": True
            }
        }
    
    def _execute_validation_phase(self) -> Dict[str, Any]:
        """Execute validation phase"""
        logger.info("Running validation phase...")
        
        # Wait a moment for system to stabilize
        time.sleep(5)
        
        # Check system health
        health_status = self.health_monitor.get_current_health_status()
        if health_status == HealthCheckStatus.UNHEALTHY:
            return {
                "success": False,
                "error": "System unhealthy before deployment",
                "health_status": health_status.value
            }
        
        # Additional validation checks would go here
        
        return {
            "success": True,
            "metrics": {
                "health_status": health_status.value,
                "system_ready": True
            }
        }
    
    def _execute_canary_phase(self) -> Dict[str, Any]:
        """Execute canary deployment phase"""
        logger.info("Running canary deployment phase...")
        
        # Simulate canary deployment (5% traffic)
        self._simulate_traffic_shift(5.0)
        
        # Monitor for configured duration
        phase_config = self.phases_config[DeploymentPhase.CANARY]
        monitor_duration = min(phase_config.duration_minutes * 60, 300)  # Max 5 minutes for demo
        
        logger.info(f"Monitoring canary deployment for {monitor_duration} seconds...")
        time.sleep(monitor_duration)
        
        # Validate canary health
        validation_result = self.validator.validate_canary_deployment(self.health_monitor)
        
        if not validation_result.status:
            return {
                "success": False,
                "error": f"Canary validation failed: {validation_result.message}",
                "validation_result": asdict(validation_result)
            }
        
        return {
            "success": True,
            "metrics": {
                "traffic_percentage": 5.0,
                "monitoring_duration": monitor_duration,
                "canary_healthy": True
            }
        }
    
    def _execute_blue_green_phase(self) -> Dict[str, Any]:
        """Execute blue-green deployment phase"""
        logger.info("Running blue-green deployment phase...")
        
        # Shift to 50% traffic
        self._simulate_traffic_shift(50.0)
        
        # Monitor blue-green deployment
        time.sleep(30)  # Shortened for demo
        
        # Check health metrics
        health_summary = self.health_monitor.get_metrics_summary(minutes=1)
        
        return {
            "success": True,
            "metrics": {
                "traffic_percentage": 50.0,
                "avg_response_time": health_summary.get('avg_response_time', 0),
                "avg_error_rate": health_summary.get('avg_error_rate', 0)
            }
        }
    
    def _execute_progressive_rollout_phase(self) -> Dict[str, Any]:
        """Execute progressive rollout phase"""
        logger.info("Running progressive rollout phase...")
        
        # Gradually increase traffic: 50% -> 75% -> 100%
        traffic_steps = [75.0, 100.0]
        
        for traffic_percent in traffic_steps:
            self._simulate_traffic_shift(traffic_percent)
            logger.info(f"Traffic shifted to {traffic_percent}%")
            time.sleep(15)  # Monitor each step
        
        return {
            "success": True,
            "metrics": {
                "final_traffic_percentage": 100.0,
                "progressive_steps_completed": len(traffic_steps)
            }
        }
    
    def _execute_full_deployment_phase(self) -> Dict[str, Any]:
        """Execute full deployment phase"""
        logger.info("Running full deployment phase...")
        
        # Ensure 100% traffic
        self._simulate_traffic_shift(100.0)
        
        # Extended monitoring for full deployment
        time.sleep(30)
        
        # Validate full deployment
        validation_result = self.validator.validate_full_deployment(self.health_monitor)
        
        if not validation_result.status:
            return {
                "success": False,
                "error": f"Full deployment validation failed: {validation_result.message}",
                "validation_result": asdict(validation_result)
            }
        
        return {
            "success": True,
            "metrics": {
                "traffic_percentage": 100.0,
                "full_deployment_validated": True
            }
        }
    
    def _execute_monitoring_phase(self) -> Dict[str, Any]:
        """Execute post-deployment monitoring phase"""
        logger.info("Running post-deployment monitoring phase...")
        
        # Extended monitoring period
        monitoring_duration = 60  # 1 minute for demo
        
        start_time = time.time()
        alert_count = len(self.health_monitor.alerts)
        
        time.sleep(monitoring_duration)
        
        # Check for new critical alerts
        new_alerts = len(self.health_monitor.alerts) - alert_count
        critical_alerts = [
            a for a in self.health_monitor.alerts[-new_alerts:]
            if a.severity == AlertSeverity.CRITICAL
        ]
        
        return {
            "success": len(critical_alerts) == 0,
            "metrics": {
                "monitoring_duration": monitoring_duration,
                "new_alerts": new_alerts,
                "critical_alerts": len(critical_alerts),
                "system_stable": len(critical_alerts) == 0
            }
        }
    
    def _execute_completion_phase(self) -> Dict[str, Any]:
        """Execute deployment completion phase"""
        logger.info("Running deployment completion phase...")
        
        # Finalize deployment
        self._finalize_deployment()
        
        # Cleanup temporary resources
        self._cleanup_deployment_resources()
        
        return {
            "success": True,
            "metrics": {
                "deployment_finalized": True,
                "resources_cleaned": True
            }
        }
    
    def _simulate_traffic_shift(self, percentage: float):
        """Simulate traffic shifting to new deployment"""
        logger.info(f"Shifting {percentage}% traffic to new deployment")
        # In a real implementation, this would configure load balancers,
        # update routing rules, etc.
        time.sleep(2)  # Simulate configuration time
    
    def _create_deployment_backup(self):
        """Create backup of current deployment state"""
        backup_dir = self.workspace_path / "deployments" / self.deployment_id / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # Backup key configuration files
        config_files = [
            "config/unified_framework_config.json",
            "scripts/unified-execution-framework.py"
        ]
        
        for config_file in config_files:
            source_path = self.workspace_path / config_file
            if source_path.exists():
                backup_path = backup_dir / f"backup_{config_file.replace('/', '_')}"
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, backup_path)
        
        self.rollback_data['backup_dir'] = str(backup_dir)
        logger.info(f"Deployment backup created: {backup_dir}")
    
    def _initialize_deployment_artifacts(self):
        """Initialize deployment artifacts"""
        artifacts_dir = self.workspace_path / "deployments" / self.deployment_id / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Create deployment manifest
        manifest = {
            'deployment_id': self.deployment_id,
            'timestamp': time.time(),
            'version': '1.0.0',
            'components': ['unified-execution-framework', 'task-prioritizer', 'performance-optimizer']
        }
        
        with open(artifacts_dir / "deployment_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info("Deployment artifacts initialized")
    
    def _execute_rollback(self, failed_phase: DeploymentPhase) -> Dict[str, Any]:
        """Execute deployment rollback"""
        logger.warning(f"Executing rollback from phase: {failed_phase.value}")
        self.deployment_status = DeploymentStatus.ROLLING_BACK
        
        try:
            # Restore traffic to 0%
            self._simulate_traffic_shift(0.0)
            
            # Restore backup files if available
            if 'backup_dir' in self.rollback_data:
                backup_dir = Path(self.rollback_data['backup_dir'])
                if backup_dir.exists():
                    # Restore configuration files
                    for backup_file in backup_dir.glob("backup_*"):
                        original_path = str(backup_file.name).replace('backup_', '').replace('_', '/')
                        target_path = self.workspace_path / original_path
                        if target_path.parent.exists():
                            shutil.copy2(backup_file, target_path)
            
            # Wait for system to stabilize
            time.sleep(10)
            
            # Verify rollback success
            health_status = self.health_monitor.get_current_health_status()
            
            rollback_success = health_status != HealthCheckStatus.UNHEALTHY
            
            return {
                "success": rollback_success,
                "metrics": {
                    "health_status_after_rollback": health_status.value,
                    "rollback_completed": True
                }
            }
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_phase_callbacks(self, phase: DeploymentPhase, success: bool):
        """Execute registered callbacks for phase completion"""
        if phase in self.phase_callbacks:
            for callback in self.phase_callbacks[phase]:
                try:
                    callback(phase, success)
                except Exception as e:
                    logger.error(f"Phase callback failed: {e}")
    
    def _finalize_deployment(self):
        """Finalize deployment process"""
        # Update deployment status
        self.deployment_status = DeploymentStatus.COMPLETED
        
        # Save final deployment report
        report = self._generate_deployment_report(success=True)
        report_path = (self.workspace_path / "deployments" / self.deployment_id / 
                      "reports" / "final_deployment_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Deployment finalized successfully")
    
    def _cleanup_deployment_resources(self):
        """Cleanup temporary deployment resources"""
        # Remove temporary files
        temp_dir = self.workspace_path / "deployments" / self.deployment_id / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        logger.info("Deployment resources cleaned up")
    
    def _generate_deployment_report(self, success: bool, failed_phase: DeploymentPhase = None, error: str = None) -> Dict[str, Any]:
        """Generate comprehensive deployment report"""
        total_duration = time.time() - self.deployment_start_time if self.deployment_start_time else 0
        
        report = {
            "deployment_id": self.deployment_id,
            "status": self.deployment_status.value,
            "success": success,
            "start_time": self.deployment_start_time,
            "total_duration_seconds": total_duration,
            "failed_phase": failed_phase.value if failed_phase else None,
            "error": error,
            "phases_executed": len(self.deployment_history),
            "phase_history": self.deployment_history,
            "health_summary": self.health_monitor.get_metrics_summary(minutes=10),
            "alerts_generated": len(self.health_monitor.alerts),
            "critical_alerts": len([a for a in self.health_monitor.alerts if a.severity == AlertSeverity.CRITICAL]),
            "validation_results": [asdict(vr) for vr in self.validator.validation_results],
            "final_health_status": self.health_monitor.get_current_health_status().value
        }
        
        return report


def main():
    """Demo deployment execution"""
    # Initialize deployment manager
    deployment_manager = PhasedDeploymentManager()
    
    # Register alert callback
    def alert_handler(alert: Alert):
        print(f"🚨 ALERT: {alert.title} - {alert.message}")
    
    deployment_manager.health_monitor.register_alert_callback(alert_handler)
    
    # Register phase callback
    def phase_completed(phase: DeploymentPhase, success: bool):
        status = "✅ COMPLETED" if success else "❌ FAILED"
        print(f"Phase {phase.value}: {status}")
    
    for phase in DeploymentPhase:
        deployment_manager.register_phase_callback(phase, phase_completed)
    
    # Execute deployment
    print("🚀 Starting phased deployment...")
    result = deployment_manager.execute_deployment()
    
    # Print results
    print("\n" + "="*80)
    print("PHASED DEPLOYMENT SUMMARY")
    print("="*80)
    print(f"Deployment ID: {result['deployment_id']}")
    print(f"Status: {result['status']}")
    print(f"Success: {'✅ YES' if result['success'] else '❌ NO'}")
    print(f"Total Duration: {result['total_duration_seconds']:.1f}s")
    print(f"Phases Executed: {result['phases_executed']}")
    
    if result['failed_phase']:
        print(f"Failed Phase: {result['failed_phase']}")
        print(f"Error: {result['error']}")
    
    print(f"Final Health Status: {result['final_health_status']}")
    print(f"Alerts Generated: {result['alerts_generated']}")
    print(f"Critical Alerts: {result['critical_alerts']}")
    
    # Phase execution summary
    print("\nPhase Execution Details:")
    for phase_data in result['phase_history']:
        print(f"  {phase_data['phase']}: {phase_data['duration_seconds']:.1f}s "
              f"({'✅' if phase_data['success'] else '❌'})")
    
    print("="*80)


if __name__ == "__main__":
    main()