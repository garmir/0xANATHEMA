#!/usr/bin/env python3
"""
Advanced Alerting and Optimization Engine for Task Master AI

This module provides intelligent alerting, anomaly detection, and automated
optimization recommendations based on real-time performance metrics and
historical analysis.
"""

import json
import time
import statistics
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert categories"""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    TASK = "task"
    GITHUB = "github"
    SECURITY = "security"
    OPTIMIZATION = "optimization"


@dataclass
class AlertRule:
    """Configuration for an alert rule"""
    id: str
    name: str
    category: AlertCategory
    severity: AlertSeverity
    metric_path: str  # e.g., "system.cpu_percent"
    condition: str    # e.g., ">", "<", "==", "!=", "trend_up", "trend_down"
    threshold: float
    duration_seconds: int = 0  # Alert only if condition persists
    cooldown_seconds: int = 300  # Minimum time between alerts
    enabled: bool = True
    description: str = ""
    remediation_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.remediation_suggestions is None:
            self.remediation_suggestions = []


@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    metric_value: Any
    threshold: float
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    resolution_message: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    id: str
    timestamp: datetime
    category: str
    priority: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    current_state: str
    target_state: str
    implementation_steps: List[str]
    expected_impact: str
    effort_estimate: str  # "low", "medium", "high"
    confidence_score: float  # 0.0 to 1.0
    relevant_metrics: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AnomalyDetector:
    """Advanced anomaly detection using statistical methods"""
    
    def __init__(self, window_size: int = 50, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity  # Standard deviations from mean
        self.metric_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines = {}
    
    def add_data_point(self, metric_path: str, value: float, timestamp: datetime):
        """Add a data point for anomaly detection"""
        if not isinstance(value, (int, float)):
            return
        
        self.metric_windows[metric_path].append((timestamp, value))
        
        # Update baseline if we have enough data
        if len(self.metric_windows[metric_path]) >= 10:
            values = [v for _, v in self.metric_windows[metric_path]]
            self.baselines[metric_path] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'updated': timestamp
            }
    
    def detect_anomaly(self, metric_path: str, value: float) -> Tuple[bool, float]:
        """Detect if a value is anomalous. Returns (is_anomaly, anomaly_score)"""
        if metric_path not in self.baselines:
            return False, 0.0
        
        baseline = self.baselines[metric_path]
        if baseline['std'] == 0:
            return False, 0.0
        
        # Calculate z-score
        z_score = abs(value - baseline['mean']) / baseline['std']
        is_anomaly = z_score > self.sensitivity
        
        # Normalize anomaly score to 0-1 range
        anomaly_score = min(z_score / (self.sensitivity * 2), 1.0)
        
        return is_anomaly, anomaly_score
    
    def detect_trend_anomaly(self, metric_path: str, trend_window: int = 10) -> Tuple[bool, str, float]:
        """Detect trend-based anomalies. Returns (is_anomaly, trend_direction, trend_strength)"""
        if metric_path not in self.metric_windows:
            return False, "none", 0.0
        
        window = list(self.metric_windows[metric_path])
        if len(window) < trend_window:
            return False, "none", 0.0
        
        # Get recent values
        recent_values = [v for _, v in window[-trend_window:]]
        
        # Calculate trend using linear regression slope
        n = len(recent_values)
        x_values = list(range(n))
        
        # Calculate slope
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(recent_values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, recent_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return False, "none", 0.0
        
        slope = numerator / denominator
        
        # Determine trend strength and direction
        trend_strength = abs(slope) / (y_mean + 1e-10)  # Avoid division by zero
        
        if trend_strength > 0.1:  # Significant trend threshold
            trend_direction = "up" if slope > 0 else "down"
            return True, trend_direction, min(trend_strength, 1.0)
        
        return False, "none", trend_strength


class AlertEngine:
    """Advanced alerting engine with rule-based and anomaly detection"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self.condition_start_times: Dict[str, datetime] = {}
        self.anomaly_detector = AnomalyDetector()
        self.notification_callbacks: List[Callable[[Alert], None]] = []
        
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alerting rules"""
        
        # System resource rules
        self.add_rule(AlertRule(
            id="high_cpu_usage",
            name="High CPU Usage",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            metric_path="system.cpu_percent",
            condition=">",
            threshold=80.0,
            duration_seconds=60,
            cooldown_seconds=300,
            description="CPU usage is critically high",
            remediation_suggestions=[
                "Identify CPU-intensive processes",
                "Consider scaling resources",
                "Review task scheduling",
                "Optimize CPU-bound algorithms"
            ]
        ))
        
        self.add_rule(AlertRule(
            id="critical_cpu_usage",
            name="Critical CPU Usage",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.CRITICAL,
            metric_path="system.cpu_percent",
            condition=">",
            threshold=95.0,
            duration_seconds=30,
            cooldown_seconds=180,
            description="CPU usage is at critical levels",
            remediation_suggestions=[
                "Immediate investigation required",
                "Kill non-essential processes",
                "Emergency resource scaling",
                "Check for runaway processes"
            ]
        ))
        
        self.add_rule(AlertRule(
            id="high_memory_usage",
            name="High Memory Usage",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            metric_path="system.memory_percent",
            condition=">",
            threshold=85.0,
            duration_seconds=120,
            cooldown_seconds=300,
            description="Memory usage is approaching limits",
            remediation_suggestions=[
                "Review memory-intensive processes",
                "Implement garbage collection",
                "Consider memory optimization",
                "Monitor for memory leaks"
            ]
        ))
        
        # Task performance rules
        self.add_rule(AlertRule(
            id="task_failure_rate",
            name="High Task Failure Rate",
            category=AlertCategory.TASK,
            severity=AlertSeverity.ERROR,
            metric_path="tasks.failure_rate",
            condition=">",
            threshold=0.1,
            duration_seconds=300,
            cooldown_seconds=600,
            description="Task failure rate is unacceptably high",
            remediation_suggestions=[
                "Review recent task failures",
                "Check task dependencies",
                "Validate input data",
                "Review error logs"
            ]
        ))
        
        # GitHub Actions rules
        self.add_rule(AlertRule(
            id="github_failure_rate",
            name="GitHub Actions Failure Rate",
            category=AlertCategory.GITHUB,
            severity=AlertSeverity.WARNING,
            metric_path="github.success_rate",
            condition="<",
            threshold=0.9,
            duration_seconds=180,
            cooldown_seconds=300,
            description="GitHub Actions success rate is low",
            remediation_suggestions=[
                "Review failed workflow runs",
                "Check for infrastructure issues",
                "Validate deployment configurations",
                "Review recent code changes"
            ]
        ))
        
        # Performance rules
        self.add_rule(AlertRule(
            id="degraded_performance",
            name="Performance Degradation",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            metric_path="performance.health_score",
            condition="<",
            threshold=70.0,
            duration_seconds=300,
            cooldown_seconds=600,
            description="Overall system performance has degraded",
            remediation_suggestions=[
                "Run performance analysis",
                "Check resource utilization",
                "Review recent changes",
                "Consider optimization strategies"
            ]
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add or update an alert rule"""
        self.rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """Remove an alert rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def add_notification_callback(self, callback: Callable[[Alert], None]):
        """Add a notification callback function"""
        self.notification_callbacks.append(callback)
    
    def evaluate_metrics(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate metrics against all rules and return new alerts"""
        new_alerts = []
        current_time = datetime.now()
        
        # Flatten metrics for easy access
        flat_metrics = self._flatten_dict(metrics)
        
        # Add metrics to anomaly detector
        for metric_path, value in flat_metrics.items():
            if isinstance(value, (int, float)):
                self.anomaly_detector.add_data_point(metric_path, value, current_time)
        
        # Evaluate each rule
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check if metric exists
            if rule.metric_path not in flat_metrics:
                continue
            
            metric_value = flat_metrics[rule.metric_path]
            
            # Evaluate condition
            condition_met = self._evaluate_condition(rule, metric_value)
            
            if condition_met:
                # Check duration requirement
                if rule.duration_seconds > 0:
                    if rule.id not in self.condition_start_times:
                        self.condition_start_times[rule.id] = current_time
                        continue
                    
                    duration = (current_time - self.condition_start_times[rule.id]).total_seconds()
                    if duration < rule.duration_seconds:
                        continue
                else:
                    self.condition_start_times[rule.id] = current_time
                
                # Check cooldown
                if rule.id in self.last_alert_times:
                    time_since_last = (current_time - self.last_alert_times[rule.id]).total_seconds()
                    if time_since_last < rule.cooldown_seconds:
                        continue
                
                # Create alert
                alert = self._create_alert(rule, metric_value, current_time)
                new_alerts.append(alert)
                
                # Update tracking
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
                self.last_alert_times[rule.id] = current_time
                
                # Reset condition start time
                if rule.id in self.condition_start_times:
                    del self.condition_start_times[rule.id]
                
                # Trigger notifications
                for callback in self.notification_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Error in notification callback: {e}")
            
            else:
                # Condition not met, reset duration tracking
                if rule.id in self.condition_start_times:
                    del self.condition_start_times[rule.id]
        
        # Check for anomalies
        anomaly_alerts = self._check_anomalies(flat_metrics, current_time)
        new_alerts.extend(anomaly_alerts)
        
        return new_alerts
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _evaluate_condition(self, rule: AlertRule, value: Any) -> bool:
        """Evaluate a rule condition"""
        if not isinstance(value, (int, float)) and rule.condition not in ["==", "!="]:
            return False
        
        try:
            if rule.condition == ">":
                return float(value) > rule.threshold
            elif rule.condition == "<":
                return float(value) < rule.threshold
            elif rule.condition == ">=":
                return float(value) >= rule.threshold
            elif rule.condition == "<=":
                return float(value) <= rule.threshold
            elif rule.condition == "==":
                return value == rule.threshold
            elif rule.condition == "!=":
                return value != rule.threshold
            elif rule.condition == "trend_up":
                is_anomaly, direction, strength = self.anomaly_detector.detect_trend_anomaly(rule.metric_path)
                return is_anomaly and direction == "up" and strength > rule.threshold
            elif rule.condition == "trend_down":
                is_anomaly, direction, strength = self.anomaly_detector.detect_trend_anomaly(rule.metric_path)
                return is_anomaly and direction == "down" and strength > rule.threshold
            else:
                logger.warning(f"Unknown condition: {rule.condition}")
                return False
        except (ValueError, TypeError) as e:
            logger.error(f"Error evaluating condition for rule {rule.id}: {e}")
            return False
    
    def _create_alert(self, rule: AlertRule, metric_value: Any, timestamp: datetime) -> Alert:
        """Create an alert from a rule"""
        alert_id = f"{rule.id}_{int(timestamp.timestamp())}"
        
        # Format message
        message = f"{rule.description}. Current value: {metric_value}, Threshold: {rule.threshold}"
        
        return Alert(
            id=alert_id,
            rule_id=rule.id,
            timestamp=timestamp,
            severity=rule.severity,
            category=rule.category,
            title=rule.name,
            message=message,
            metric_value=metric_value,
            threshold=rule.threshold,
            metadata={
                'metric_path': rule.metric_path,
                'remediation_suggestions': rule.remediation_suggestions
            }
        )
    
    def _check_anomalies(self, flat_metrics: Dict[str, Any], timestamp: datetime) -> List[Alert]:
        """Check for statistical anomalies"""
        anomaly_alerts = []
        
        for metric_path, value in flat_metrics.items():
            if not isinstance(value, (int, float)):
                continue
            
            is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(metric_path, value)
            
            if is_anomaly and anomaly_score > 0.8:  # High confidence anomaly
                # Check cooldown for anomaly alerts
                anomaly_rule_id = f"anomaly_{metric_path}"
                if anomaly_rule_id in self.last_alert_times:
                    time_since_last = (timestamp - self.last_alert_times[anomaly_rule_id]).total_seconds()
                    if time_since_last < 600:  # 10 minute cooldown for anomalies
                        continue
                
                alert_id = f"anomaly_{metric_path}_{int(timestamp.timestamp())}"
                
                severity = AlertSeverity.WARNING if anomaly_score < 0.9 else AlertSeverity.ERROR
                
                alert = Alert(
                    id=alert_id,
                    rule_id=anomaly_rule_id,
                    timestamp=timestamp,
                    severity=severity,
                    category=AlertCategory.PERFORMANCE,
                    title=f"Anomaly Detected: {metric_path}",
                    message=f"Statistical anomaly detected in {metric_path}. Value: {value}, Anomaly Score: {anomaly_score:.2f}",
                    metric_value=value,
                    threshold=anomaly_score,
                    metadata={
                        'anomaly_score': anomaly_score,
                        'metric_path': metric_path,
                        'type': 'statistical_anomaly'
                    }
                )
                
                anomaly_alerts.append(alert)
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
                self.last_alert_times[anomaly_rule_id] = timestamp
        
        return anomaly_alerts
    
    def resolve_alert(self, alert_id: str, resolution_message: str = ""):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_timestamp = datetime.now()
            alert.resolution_message = resolution_message
            del self.active_alerts[alert_id]
            logger.info(f"Resolved alert: {alert.title}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of active alerts by severity"""
        active_alerts = self.get_active_alerts()
        summary = {
            'total': len(active_alerts),
            'critical': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'error': len([a for a in active_alerts if a.severity == AlertSeverity.ERROR]),
            'warning': len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
            'info': len([a for a in active_alerts if a.severity == AlertSeverity.INFO])
        }
        return summary


class OptimizationEngine:
    """AI-powered optimization recommendation engine"""
    
    def __init__(self):
        self.historical_metrics = deque(maxlen=1000)
        self.recommendations_cache = []
        self.optimization_patterns = {}
        
    def add_metrics(self, metrics: Dict[str, Any]):
        """Add metrics for optimization analysis"""
        self.historical_metrics.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
    
    def generate_recommendations(self, current_metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on current and historical metrics"""
        recommendations = []
        
        # Flatten current metrics
        flat_metrics = self._flatten_dict(current_metrics)
        
        # System resource optimization
        recommendations.extend(self._analyze_system_resources(flat_metrics))
        
        # Task performance optimization
        recommendations.extend(self._analyze_task_performance(flat_metrics))
        
        # Memory optimization
        recommendations.extend(self._analyze_memory_usage(flat_metrics))
        
        # GitHub Actions optimization
        recommendations.extend(self._analyze_github_actions(flat_metrics))
        
        # Performance trend optimization
        recommendations.extend(self._analyze_performance_trends())
        
        # Cache recommendations
        self.recommendations_cache = recommendations
        
        return recommendations
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _analyze_system_resources(self, metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze system resource usage for optimization opportunities"""
        recommendations = []
        
        # CPU optimization
        cpu_usage = metrics.get('system.cpu_percent', 0)
        if cpu_usage > 70:
            priority = "high" if cpu_usage > 85 else "medium"
            confidence = min((cpu_usage - 70) / 30, 1.0)
            
            recommendations.append(OptimizationRecommendation(
                id=f"cpu_optimization_{int(time.time())}",
                timestamp=datetime.now(),
                category="system_resources",
                priority=priority,
                title="CPU Usage Optimization",
                description=f"CPU usage is at {cpu_usage:.1f}%, indicating potential optimization opportunities",
                current_state=f"CPU usage: {cpu_usage:.1f}%",
                target_state="CPU usage: <70%",
                implementation_steps=[
                    "Profile CPU-intensive processes using top or htop",
                    "Identify bottleneck algorithms and optimize",
                    "Implement task prioritization and scheduling",
                    "Consider parallel processing for CPU-bound tasks",
                    "Enable CPU affinity for critical processes"
                ],
                expected_impact=f"Reduce CPU usage by 15-25% (target: {max(cpu_usage * 0.75, 60):.1f}%)",
                effort_estimate="medium",
                confidence_score=confidence,
                relevant_metrics=["system.cpu_percent", "system.load_average"]
            ))
        
        # Memory optimization
        memory_usage = metrics.get('system.memory_percent', 0)
        if memory_usage > 75:
            priority = "high" if memory_usage > 90 else "medium"
            confidence = min((memory_usage - 75) / 25, 1.0)
            
            recommendations.append(OptimizationRecommendation(
                id=f"memory_optimization_{int(time.time())}",
                timestamp=datetime.now(),
                category="memory_management",
                priority=priority,
                title="Memory Usage Optimization",
                description=f"Memory usage is at {memory_usage:.1f}%, requiring optimization",
                current_state=f"Memory usage: {memory_usage:.1f}%",
                target_state="Memory usage: <75%",
                implementation_steps=[
                    "Run memory profiler to identify memory leaks",
                    "Implement object pooling for frequently allocated objects",
                    "Optimize data structures and algorithms",
                    "Enable garbage collection tuning",
                    "Consider caching strategies with TTL"
                ],
                expected_impact=f"Reduce memory usage by 10-20% (target: {max(memory_usage * 0.8, 65):.1f}%)",
                effort_estimate="medium",
                confidence_score=confidence,
                relevant_metrics=["system.memory_percent", "system.memory_available_gb"]
            ))
        
        return recommendations
    
    def _analyze_task_performance(self, metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze task performance for optimization"""
        recommendations = []
        
        # Task completion rate
        completion_rate = metrics.get('tasks.completion_rate', 1.0)
        if completion_rate < 0.95:
            priority = "high" if completion_rate < 0.9 else "medium"
            confidence = min((0.95 - completion_rate) / 0.1, 1.0)
            
            recommendations.append(OptimizationRecommendation(
                id=f"task_completion_optimization_{int(time.time())}",
                timestamp=datetime.now(),
                category="task_performance",
                priority=priority,
                title="Task Completion Rate Optimization",
                description=f"Task completion rate is {completion_rate:.1%}, below optimal threshold",
                current_state=f"Completion rate: {completion_rate:.1%}",
                target_state="Completion rate: >95%",
                implementation_steps=[
                    "Analyze failed task patterns and root causes",
                    "Implement retry mechanisms with exponential backoff",
                    "Add input validation and error handling",
                    "Optimize task dependency resolution",
                    "Implement circuit breaker pattern for external dependencies"
                ],
                expected_impact=f"Improve completion rate to {min(completion_rate + 0.05, 0.98):.1%}",
                effort_estimate="medium",
                confidence_score=confidence,
                relevant_metrics=["tasks.completion_rate", "tasks.failure_rate"]
            ))
        
        # Average execution time
        avg_execution_time = metrics.get('tasks.avg_execution_time', 0)
        if avg_execution_time > 120:  # More than 2 minutes
            priority = "medium" if avg_execution_time < 300 else "high"
            confidence = min((avg_execution_time - 120) / 180, 1.0)
            
            recommendations.append(OptimizationRecommendation(
                id=f"task_execution_optimization_{int(time.time())}",
                timestamp=datetime.now(),
                category="task_performance",
                priority=priority,
                title="Task Execution Time Optimization",
                description=f"Average task execution time is {avg_execution_time:.1f}s, indicating optimization potential",
                current_state=f"Avg execution time: {avg_execution_time:.1f}s",
                target_state="Avg execution time: <120s",
                implementation_steps=[
                    "Profile slow-executing tasks",
                    "Implement caching for repeated operations",
                    "Optimize database queries and I/O operations",
                    "Enable parallel execution where possible",
                    "Implement task batching strategies"
                ],
                expected_impact=f"Reduce execution time by 30-40% (target: {max(avg_execution_time * 0.7, 90):.1f}s)",
                effort_estimate="medium",
                confidence_score=confidence,
                relevant_metrics=["tasks.avg_execution_time", "tasks.execution_variance"]
            ))
        
        return recommendations
    
    def _analyze_memory_usage(self, metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze memory usage patterns for optimization"""
        recommendations = []
        
        memory_percent = metrics.get('system.memory_percent', 0)
        memory_available = metrics.get('system.memory_available_gb', 0)
        
        if memory_percent > 80 or memory_available < 2.0:
            recommendations.append(OptimizationRecommendation(
                id=f"memory_pattern_optimization_{int(time.time())}",
                timestamp=datetime.now(),
                category="memory_optimization",
                priority="high",
                title="Memory Pattern Optimization",
                description="Memory usage patterns indicate need for advanced optimization strategies",
                current_state=f"Memory: {memory_percent:.1f}% used, {memory_available:.1f}GB available",
                target_state="Memory: <75% used, >3GB available",
                implementation_steps=[
                    "Implement memory-aware task scheduling",
                    "Add memory pressure monitoring",
                    "Implement intelligent garbage collection",
                    "Optimize data structure selection",
                    "Enable memory compaction strategies"
                ],
                expected_impact="Improve memory efficiency by 20-30%",
                effort_estimate="high",
                confidence_score=0.8,
                relevant_metrics=["system.memory_percent", "system.memory_available_gb"]
            ))
        
        return recommendations
    
    def _analyze_github_actions(self, metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze GitHub Actions performance"""
        recommendations = []
        
        success_rate = metrics.get('github.success_rate', 1.0)
        if success_rate < 0.95:
            priority = "medium" if success_rate > 0.9 else "high"
            
            recommendations.append(OptimizationRecommendation(
                id=f"github_optimization_{int(time.time())}",
                timestamp=datetime.now(),
                category="ci_cd",
                priority=priority,
                title="GitHub Actions Optimization",
                description=f"GitHub Actions success rate is {success_rate:.1%}, below optimal threshold",
                current_state=f"Success rate: {success_rate:.1%}",
                target_state="Success rate: >95%",
                implementation_steps=[
                    "Analyze failed workflow patterns",
                    "Implement workflow retry strategies",
                    "Optimize dependency caching",
                    "Review workflow trigger conditions",
                    "Implement parallel job execution"
                ],
                expected_impact=f"Improve success rate to {min(success_rate + 0.05, 0.98):.1%}",
                effort_estimate="medium",
                confidence_score=0.7,
                relevant_metrics=["github.success_rate", "github.workflow_duration"]
            ))
        
        return recommendations
    
    def _analyze_performance_trends(self) -> List[OptimizationRecommendation]:
        """Analyze performance trends for predictive optimization"""
        recommendations = []
        
        if len(self.historical_metrics) < 10:
            return recommendations
        
        # Analyze health score trend
        recent_health_scores = []
        for entry in list(self.historical_metrics)[-10:]:
            health_score = self._get_nested_value(entry['metrics'], 'performance.health_score')
            if health_score:
                recent_health_scores.append(health_score)
        
        if len(recent_health_scores) >= 5:
            # Calculate trend
            trend_slope = self._calculate_trend_slope(recent_health_scores)
            
            if trend_slope < -2:  # Declining health score
                recommendations.append(OptimizationRecommendation(
                    id=f"performance_trend_optimization_{int(time.time())}",
                    timestamp=datetime.now(),
                    category="performance_trends",
                    priority="high",
                    title="Performance Degradation Trend",
                    description=f"System health score is declining (trend: {trend_slope:.2f})",
                    current_state=f"Health score trending down: {trend_slope:.2f} points/interval",
                    target_state="Stable or improving health score trend",
                    implementation_steps=[
                        "Investigate recent system changes",
                        "Run comprehensive performance audit",
                        "Implement proactive monitoring",
                        "Review resource allocation strategies",
                        "Consider system optimization overhaul"
                    ],
                    expected_impact="Stabilize and improve health score trend",
                    effort_estimate="high",
                    confidence_score=0.9,
                    relevant_metrics=["performance.health_score", "system.cpu_percent", "system.memory_percent"]
                ))
        
        return recommendations
    
    def _get_nested_value(self, dictionary: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = path.split('.')
        value = dictionary
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def get_cached_recommendations(self) -> List[OptimizationRecommendation]:
        """Get cached optimization recommendations"""
        return self.recommendations_cache


class NotificationManager:
    """Manage different types of notifications"""
    
    def __init__(self):
        self.notification_channels = {}
        self.notification_history = []
    
    def add_console_notification(self):
        """Add console notification channel"""
        def console_notifier(alert: Alert):
            severity_icons = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨"
            }
            
            icon = severity_icons.get(alert.severity, "📢")
            print(f"{icon} [{alert.severity.value.upper()}] {alert.title}: {alert.message}")
            
            if alert.metadata and 'remediation_suggestions' in alert.metadata:
                print("   Remediation suggestions:")
                for suggestion in alert.metadata['remediation_suggestions']:
                    print(f"   • {suggestion}")
        
        self.notification_channels['console'] = console_notifier
    
    def add_file_notification(self, file_path: str):
        """Add file-based notification channel"""
        def file_notifier(alert: Alert):
            try:
                with open(file_path, 'a') as f:
                    f.write(f"{alert.timestamp.isoformat()} [{alert.severity.value.upper()}] {alert.title}: {alert.message}\n")
            except Exception as e:
                logger.error(f"Error writing to notification file: {e}")
        
        self.notification_channels['file'] = file_notifier
    
    def send_notification(self, alert: Alert):
        """Send notification through all configured channels"""
        for channel_name, notifier in self.notification_channels.items():
            try:
                notifier(alert)
                self.notification_history.append({
                    'timestamp': datetime.now(),
                    'channel': channel_name,
                    'alert_id': alert.id,
                    'status': 'sent'
                })
            except Exception as e:
                logger.error(f"Error sending notification via {channel_name}: {e}")
                self.notification_history.append({
                    'timestamp': datetime.now(),
                    'channel': channel_name,
                    'alert_id': alert.id,
                    'status': 'failed',
                    'error': str(e)
                })


class AlertingOptimizationEngine:
    """Main engine combining alerting and optimization"""
    
    def __init__(self):
        self.alert_engine = AlertEngine()
        self.optimization_engine = OptimizationEngine()
        self.notification_manager = NotificationManager()
        
        # Setup default notifications
        self.notification_manager.add_console_notification()
        self.notification_manager.add_file_notification(".taskmaster/alerts.log")
        
        # Connect alert engine to notification manager
        self.alert_engine.add_notification_callback(self.notification_manager.send_notification)
        
        self.running = False
    
    def start(self):
        """Start the alerting and optimization engine"""
        self.running = True
        logger.info("Alerting and Optimization Engine started")
    
    def stop(self):
        """Stop the alerting and optimization engine"""
        self.running = False
        logger.info("Alerting and Optimization Engine stopped")
    
    def process_metrics(self, metrics: Dict[str, Any]) -> Tuple[List[Alert], List[OptimizationRecommendation]]:
        """Process metrics through both alerting and optimization engines"""
        if not self.running:
            return [], []
        
        # Process alerts
        new_alerts = self.alert_engine.evaluate_metrics(metrics)
        
        # Add metrics to optimization engine
        self.optimization_engine.add_metrics(metrics)
        
        # Generate optimization recommendations
        recommendations = self.optimization_engine.generate_recommendations(metrics)
        
        return new_alerts, recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        return {
            'running': self.running,
            'active_alerts': len(self.alert_engine.get_active_alerts()),
            'alert_summary': self.alert_engine.get_alert_summary(),
            'total_rules': len(self.alert_engine.rules),
            'cached_recommendations': len(self.optimization_engine.get_cached_recommendations()),
            'notification_channels': list(self.notification_manager.notification_channels.keys())
        }


def main():
    """Main function for testing the alerting and optimization engine"""
    # Create engine
    engine = AlertingOptimizationEngine()
    engine.start()
    
    try:
        print("Alerting and Optimization Engine Test")
        print("=" * 50)
        
        # Simulate some metrics
        test_metrics = {
            'system': {
                'cpu_percent': 85.5,  # High CPU - should trigger alert
                'memory_percent': 67.2,
                'disk_usage_percent': 78.1
            },
            'tasks': {
                'completion_rate': 0.87,  # Low completion rate - should trigger alert
                'avg_execution_time': 145.3,
                'failure_rate': 0.13
            },
            'github': {
                'success_rate': 0.92,
                'total_runs': 25,
                'workflow_duration': 180.5
            },
            'performance': {
                'health_score': 72.3,
                'response_time_avg': 165.2,
                'throughput_rps': 11.8
            }
        }
        
        # Process metrics
        alerts, recommendations = engine.process_metrics(test_metrics)
        
        print(f"\nGenerated {len(alerts)} alerts and {len(recommendations)} recommendations")
        
        # Display alerts
        if alerts:
            print("\n🚨 ALERTS:")
            for alert in alerts:
                print(f"  • {alert.title} ({alert.severity.value}): {alert.message}")
        
        # Display recommendations
        if recommendations:
            print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"  • {rec.title} ({rec.priority} priority)")
                print(f"    {rec.description}")
                print(f"    Expected Impact: {rec.expected_impact}")
        
        # Show engine status
        status = engine.get_status()
        print(f"\n📊 ENGINE STATUS:")
        print(f"  Active Alerts: {status['active_alerts']}")
        print(f"  Total Rules: {status['total_rules']}")
        print(f"  Cached Recommendations: {status['cached_recommendations']}")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()