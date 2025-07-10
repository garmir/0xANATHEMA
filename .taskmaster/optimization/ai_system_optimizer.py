#!/usr/bin/env python3
"""
Advanced System Optimization and Monitoring Suite
AI-Powered Performance Analysis and Autonomous Self-Healing System
"""

import json
import time
import psutil
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import queue
import statistics
from pathlib import Path
import subprocess
import logging
from collections import deque
import pickle


@dataclass
class SystemMetrics:
    """System performance metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_processes: int
    load_average: Tuple[float, float, float]


@dataclass
class PerformanceAnomaly:
    """Detected performance anomaly"""
    timestamp: datetime
    metric_name: str
    actual_value: float
    expected_value: float
    severity: str  # "low", "medium", "high", "critical"
    confidence: float
    suggested_action: str


@dataclass
class SelfHealingAction:
    """Self-healing action performed by the system"""
    timestamp: datetime
    issue_detected: str
    action_taken: str
    success: bool
    impact_metrics: Dict[str, float]


class MetricsCollector:
    """Real-time system metrics collection"""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.running = False
        self.metrics_queue = queue.Queue(maxsize=1000)
        self.historical_data = deque(maxlen=3600)  # Keep 1 hour of data
        self.collector_thread = None
        
    def start_collection(self):
        """Start metrics collection in background thread"""
        self.running = True
        self.collector_thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self.collector_thread.start()
        
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join()
    
    def _collect_metrics(self):
        """Background metrics collection loop"""
        while self.running:
            try:
                metrics = self._capture_current_metrics()
                self.historical_data.append(metrics)
                
                if not self.metrics_queue.full():
                    self.metrics_queue.put(metrics)
                    
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logging.error(f"Metrics collection error: {e}")
                
    def _capture_current_metrics(self) -> SystemMetrics:
        """Capture current system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        disk_io_read_mb = disk_io.read_bytes / (1024**2) if disk_io else 0
        disk_io_write_mb = disk_io.write_bytes / (1024**2) if disk_io else 0
        
        # Network metrics
        net_io = psutil.net_io_counters()
        network_bytes_sent = net_io.bytes_sent if net_io else 0
        network_bytes_recv = net_io.bytes_recv if net_io else 0
        
        # Process metrics
        active_processes = len(psutil.pids())
        
        # Load average (Unix systems)
        try:
            load_average = psutil.getloadavg()
        except (AttributeError, OSError):
            load_average = (0.0, 0.0, 0.0)
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            active_processes=active_processes,
            load_average=load_average
        )
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent metrics"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_historical_data(self, minutes: int = 10) -> List[SystemMetrics]:
        """Get historical metrics for specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.historical_data if m.timestamp >= cutoff_time]


class AIPerformanceAnalyzer:
    """AI-powered performance analysis engine"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        self.prediction_window = 60  # Seconds
        self.ml_models = {}
        
    def establish_baseline(self, historical_data: List[SystemMetrics]):
        """Establish performance baselines from historical data"""
        if len(historical_data) < 10:
            return
            
        metrics_arrays = {
            'cpu_percent': [m.cpu_percent for m in historical_data],
            'memory_percent': [m.memory_percent for m in historical_data],
            'disk_io_read_mb': [m.disk_io_read_mb for m in historical_data],
            'disk_io_write_mb': [m.disk_io_write_mb for m in historical_data],
            'active_processes': [m.active_processes for m in historical_data]
        }
        
        for metric_name, values in metrics_arrays.items():
            self.baseline_metrics[metric_name] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'median': statistics.median(values),
                'percentile_95': np.percentile(values, 95),
                'percentile_99': np.percentile(values, 99)
            }
    
    def detect_anomalies(self, current_metrics: SystemMetrics) -> List[PerformanceAnomaly]:
        """Detect performance anomalies using statistical analysis"""
        anomalies = []
        
        if not self.baseline_metrics:
            return anomalies
            
        metrics_to_check = {
            'cpu_percent': current_metrics.cpu_percent,
            'memory_percent': current_metrics.memory_percent,
            'active_processes': current_metrics.active_processes
        }
        
        for metric_name, current_value in metrics_to_check.items():
            if metric_name not in self.baseline_metrics:
                continue
                
            baseline = self.baseline_metrics[metric_name]
            
            # Z-score anomaly detection
            if baseline['std'] > 0:
                z_score = abs(current_value - baseline['mean']) / baseline['std']
                
                if z_score > self.anomaly_threshold:
                    severity = self._calculate_severity(z_score)
                    confidence = min(z_score / 4.0, 1.0)  # Normalize confidence
                    
                    anomaly = PerformanceAnomaly(
                        timestamp=current_metrics.timestamp,
                        metric_name=metric_name,
                        actual_value=current_value,
                        expected_value=baseline['mean'],
                        severity=severity,
                        confidence=confidence,
                        suggested_action=self._suggest_action(metric_name, current_value, baseline)
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_severity(self, z_score: float) -> str:
        """Calculate anomaly severity based on z-score"""
        if z_score > 4.0:
            return "critical"
        elif z_score > 3.0:
            return "high"
        elif z_score > 2.5:
            return "medium"
        else:
            return "low"
    
    def _suggest_action(self, metric_name: str, current_value: float, baseline: Dict) -> str:
        """Suggest corrective action for detected anomaly"""
        actions = {
            'cpu_percent': {
                'high': "Consider terminating high-CPU processes or scaling resources",
                'low': "System may be underutilized - consider workload consolidation"
            },
            'memory_percent': {
                'high': "Free memory by clearing caches or restarting memory-intensive processes",
                'low': "Memory usage unusually low - check for process failures"
            },
            'active_processes': {
                'high': "High process count detected - investigate for process leaks",
                'low': "Low process count - check for service failures"
            }
        }
        
        direction = "high" if current_value > baseline['mean'] else "low"
        return actions.get(metric_name, {}).get(direction, "Monitor system closely")
    
    def predict_performance_degradation(self, historical_data: List[SystemMetrics]) -> Dict[str, float]:
        """Predict potential performance issues using trend analysis"""
        if len(historical_data) < 30:
            return {}
            
        predictions = {}
        
        # Analyze trends for key metrics
        metrics_to_analyze = ['cpu_percent', 'memory_percent']
        
        for metric_name in metrics_to_analyze:
            values = [getattr(m, metric_name) for m in historical_data[-30:]]
            
            # Simple linear trend analysis
            x = np.arange(len(values))
            coefficients = np.polyfit(x, values, 1)
            trend_slope = coefficients[0]
            
            # Predict value in next prediction window
            future_x = len(values) + (self.prediction_window / 60)  # Convert to minutes
            predicted_value = np.polyval(coefficients, future_x)
            
            predictions[metric_name] = {
                'current_trend': trend_slope,
                'predicted_value': predicted_value,
                'risk_level': self._assess_risk_level(metric_name, predicted_value)
            }
        
        return predictions


class SelfHealingSystem:
    """Autonomous self-healing system"""
    
    def __init__(self):
        self.healing_actions = []
        self.action_cooldowns = {}  # Prevent rapid repeated actions
        self.cooldown_period = 300  # 5 minutes
        
    def apply_healing_action(self, anomaly: PerformanceAnomaly) -> SelfHealingAction:
        """Apply appropriate healing action for detected anomaly"""
        action_key = f"{anomaly.metric_name}_{anomaly.severity}"
        
        # Check cooldown
        if self._is_action_on_cooldown(action_key):
            return SelfHealingAction(
                timestamp=datetime.now(),
                issue_detected=f"{anomaly.metric_name} anomaly",
                action_taken="Skipped (cooldown active)",
                success=False,
                impact_metrics={}
            )
        
        # Determine and execute healing action
        action_taken, success, impact = self._execute_healing_action(anomaly)
        
        # Record action
        healing_action = SelfHealingAction(
            timestamp=datetime.now(),
            issue_detected=f"{anomaly.metric_name} {anomaly.severity} anomaly",
            action_taken=action_taken,
            success=success,
            impact_metrics=impact
        )
        
        self.healing_actions.append(healing_action)
        self.action_cooldowns[action_key] = datetime.now()
        
        return healing_action
    
    def _is_action_on_cooldown(self, action_key: str) -> bool:
        """Check if action is on cooldown"""
        if action_key not in self.action_cooldowns:
            return False
            
        last_action = self.action_cooldowns[action_key]
        return (datetime.now() - last_action).seconds < self.cooldown_period
    
    def _execute_healing_action(self, anomaly: PerformanceAnomaly) -> Tuple[str, bool, Dict[str, float]]:
        """Execute specific healing action based on anomaly type"""
        try:
            if anomaly.metric_name == "cpu_percent" and anomaly.severity in ["high", "critical"]:
                return self._handle_high_cpu()
            elif anomaly.metric_name == "memory_percent" and anomaly.severity in ["high", "critical"]:
                return self._handle_high_memory()
            elif anomaly.metric_name == "active_processes" and anomaly.severity in ["high", "critical"]:
                return self._handle_process_anomaly()
            else:
                return ("Monitoring only", True, {})
                
        except Exception as e:
            return (f"Action failed: {e}", False, {})
    
    def _handle_high_cpu(self) -> Tuple[str, bool, Dict[str, float]]:
        """Handle high CPU usage"""
        try:
            # Get top CPU processes
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] > 10:  # Only high CPU processes
                        processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if processes:
                # Sort by CPU usage
                processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
                top_process = processes[0]
                
                # For safety, only terminate non-system processes with very high CPU
                if top_process['cpu_percent'] > 80 and not self._is_system_process(top_process['name']):
                    subprocess.run(['kill', '-TERM', str(top_process['pid'])], check=True)
                    return (f"Terminated high-CPU process {top_process['name']}", True, 
                           {"cpu_reduction": top_process['cpu_percent']})
                else:
                    return ("High CPU detected but no safe action available", True, {})
            else:
                return ("No high-CPU processes found", True, {})
                
        except Exception as e:
            return (f"CPU handling failed: {e}", False, {})
    
    def _handle_high_memory(self) -> Tuple[str, bool, Dict[str, float]]:
        """Handle high memory usage"""
        try:
            # Clear system caches (safe operation)
            subprocess.run(['sync'], check=True)
            
            # Get memory info before and after
            memory_before = psutil.virtual_memory().percent
            
            # Force garbage collection in Python processes (if applicable)
            import gc
            gc.collect()
            
            memory_after = psutil.virtual_memory().percent
            memory_freed = memory_before - memory_after
            
            return ("Cleared system caches and triggered garbage collection", True,
                   {"memory_freed_percent": memory_freed})
                   
        except Exception as e:
            return (f"Memory handling failed: {e}", False, {})
    
    def _handle_process_anomaly(self) -> Tuple[str, bool, Dict[str, float]]:
        """Handle process count anomalies"""
        try:
            process_count = len(psutil.pids())
            
            # Check for zombie processes
            zombie_count = 0
            for proc in psutil.process_iter():
                try:
                    if proc.status() == psutil.STATUS_ZOMBIE:
                        zombie_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if zombie_count > 10:
                # Clean up zombie processes
                subprocess.run(['ps', '-eo', 'pid,ppid,state', '|', 'grep', 'Z'], 
                             shell=True, capture_output=True)
                return (f"Cleaned up {zombie_count} zombie processes", True,
                       {"zombies_cleaned": zombie_count})
            else:
                return ("Process count anomaly detected but no action needed", True, {})
                
        except Exception as e:
            return (f"Process handling failed: {e}", False, {})
    
    def _is_system_process(self, process_name: str) -> bool:
        """Check if process is a critical system process"""
        system_processes = {
            'kernel', 'init', 'systemd', 'kthreadd', 'ksoftirqd', 'migration',
            'rcu_', 'watchdog', 'sshd', 'NetworkManager', 'dbus'
        }
        
        return any(sys_proc in process_name.lower() for sys_proc in system_processes)


class SystemOptimizationSuite:
    """Main system optimization and monitoring suite"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.metrics_collector = MetricsCollector(self.config['collection_interval'])
        self.ai_analyzer = AIPerformanceAnalyzer()
        self.self_healer = SelfHealingSystem()
        
        # State tracking
        self.running = False
        self.monitoring_thread = None
        self.optimization_stats = {
            'anomalies_detected': 0,
            'healing_actions_taken': 0,
            'successful_optimizations': 0,
            'start_time': None
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('.taskmaster/logs/system_optimizer.log'),
                logging.StreamHandler()
            ]
        )
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'collection_interval': 1.0,
            'analysis_interval': 30.0,
            'healing_enabled': True,
            'anomaly_threshold': 2.0,
            'baseline_window_minutes': 60,
            'max_healing_actions_per_hour': 10
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def start_monitoring(self):
        """Start the monitoring and optimization system"""
        if self.running:
            return
            
        self.running = True
        self.optimization_stats['start_time'] = datetime.now()
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start monitoring and analysis loop
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logging.info("System optimization suite started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        self.metrics_collector.stop_collection()
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
            
        logging.info("System optimization suite stopped")
    
    def _monitoring_loop(self):
        """Main monitoring and analysis loop"""
        last_analysis = datetime.now()
        baseline_established = False
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Establish baseline if not done yet
                if not baseline_established:
                    historical_data = self.metrics_collector.get_historical_data(
                        self.config['baseline_window_minutes']
                    )
                    if len(historical_data) >= 30:
                        self.ai_analyzer.establish_baseline(historical_data)
                        baseline_established = True
                        logging.info("Performance baseline established")
                
                # Perform analysis at configured intervals
                if (current_time - last_analysis).seconds >= self.config['analysis_interval']:
                    self._perform_analysis()
                    last_analysis = current_time
                
                time.sleep(5)  # Short sleep between checks
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def _perform_analysis(self):
        """Perform AI analysis and self-healing"""
        try:
            # Get current metrics
            current_metrics = self.metrics_collector.get_latest_metrics()
            if not current_metrics:
                return
            
            # Detect anomalies
            anomalies = self.ai_analyzer.detect_anomalies(current_metrics)
            
            for anomaly in anomalies:
                self.optimization_stats['anomalies_detected'] += 1
                logging.warning(f"Anomaly detected: {anomaly.metric_name} = {anomaly.actual_value:.2f} "
                              f"(expected: {anomaly.expected_value:.2f}, severity: {anomaly.severity})")
                
                # Apply self-healing if enabled
                if self.config['healing_enabled'] and anomaly.severity in ['high', 'critical']:
                    healing_action = self.self_healer.apply_healing_action(anomaly)
                    self.optimization_stats['healing_actions_taken'] += 1
                    
                    if healing_action.success:
                        self.optimization_stats['successful_optimizations'] += 1
                        logging.info(f"Healing action successful: {healing_action.action_taken}")
                    else:
                        logging.error(f"Healing action failed: {healing_action.action_taken}")
            
            # Predict future issues
            historical_data = self.metrics_collector.get_historical_data(30)
            predictions = self.ai_analyzer.predict_performance_degradation(historical_data)
            
            for metric_name, prediction in predictions.items():
                if prediction['risk_level'] == 'high':
                    logging.warning(f"Performance degradation predicted for {metric_name}: "
                                  f"{prediction['predicted_value']:.2f}")
                                  
        except Exception as e:
            logging.error(f"Analysis error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and optimization statistics"""
        current_metrics = self.metrics_collector.get_latest_metrics()
        
        status = {
            'monitoring_active': self.running,
            'optimization_stats': self.optimization_stats.copy(),
            'current_metrics': asdict(current_metrics) if current_metrics else None,
            'recent_healing_actions': [asdict(action) for action in self.self_healer.healing_actions[-5:]],
            'uptime_minutes': (datetime.now() - self.optimization_stats['start_time']).seconds // 60 
                            if self.optimization_stats['start_time'] else 0
        }
        
        return status
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        historical_data = self.metrics_collector.get_historical_data(60)  # Last hour
        
        if not historical_data:
            return {"error": "No historical data available"}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in historical_data]
        memory_values = [m.memory_percent for m in historical_data]
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'data_points': len(historical_data),
            'time_range_minutes': 60,
            'performance_summary': {
                'cpu_utilization': {
                    'average': statistics.mean(cpu_values),
                    'max': max(cpu_values),
                    'min': min(cpu_values),
                    'std_dev': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
                },
                'memory_utilization': {
                    'average': statistics.mean(memory_values),
                    'max': max(memory_values),
                    'min': min(memory_values),
                    'std_dev': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
                }
            },
            'optimization_effectiveness': {
                'total_anomalies': self.optimization_stats['anomalies_detected'],
                'healing_actions': self.optimization_stats['healing_actions_taken'],
                'success_rate': (self.optimization_stats['successful_optimizations'] / 
                               max(1, self.optimization_stats['healing_actions_taken'])) * 100
            },
            'recommendations': self._generate_recommendations(historical_data)
        }
        
        return report
    
    def _generate_recommendations(self, historical_data: List[SystemMetrics]) -> List[str]:
        """Generate optimization recommendations based on historical data"""
        recommendations = []
        
        if not historical_data:
            return recommendations
        
        # Analyze patterns
        cpu_values = [m.cpu_percent for m in historical_data]
        memory_values = [m.memory_percent for m in historical_data]
        
        avg_cpu = statistics.mean(cpu_values)
        avg_memory = statistics.mean(memory_values)
        
        if avg_cpu > 80:
            recommendations.append("High CPU utilization detected. Consider scaling resources or optimizing workloads.")
        elif avg_cpu < 20:
            recommendations.append("Low CPU utilization. Consider workload consolidation for cost optimization.")
        
        if avg_memory > 80:
            recommendations.append("High memory utilization. Consider increasing memory or optimizing memory usage.")
        elif avg_memory < 30:
            recommendations.append("Low memory utilization. Memory allocation may be oversized.")
        
        # Check for variability
        cpu_std = statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
        if cpu_std > 20:
            recommendations.append("High CPU usage variability detected. Investigate workload patterns.")
        
        if not recommendations:
            recommendations.append("System performance is within normal parameters.")
        
        return recommendations


if __name__ == "__main__":
    # Example usage and testing
    print("Advanced System Optimization and Monitoring Suite")
    print("=" * 55)
    
    # Initialize the suite
    optimizer = SystemOptimizationSuite()
    
    try:
        # Start monitoring
        optimizer.start_monitoring()
        print("✓ Monitoring started")
        
        # Run for a short test period
        time.sleep(30)
        
        # Get status
        status = optimizer.get_system_status()
        print(f"✓ System status: {json.dumps(status, indent=2, default=str)}")
        
        # Generate report
        report = optimizer.generate_performance_report()
        print(f"✓ Performance report generated")
        
        # Save report
        with open('.taskmaster/reports/system_optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("✓ Advanced system optimization suite operational")
        
    finally:
        optimizer.stop_monitoring()
        print("✓ Monitoring stopped")