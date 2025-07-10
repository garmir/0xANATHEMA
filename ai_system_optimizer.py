#!/usr/bin/env python3
"""
AI-Powered System Optimization and Monitoring Suite

This module provides comprehensive system optimization with AI-powered performance analysis,
autonomous self-healing capabilities, and advanced monitoring for Task Master AI.
"""

import os
import sys
import time
import json
import threading
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import sqlite3
import logging
from enum import Enum
import statistics
import pickle
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class SystemState(Enum):
    """System health states"""
    OPTIMAL = "optimal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SystemMetrics:
    """System performance metrics snapshot"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: List[float]
    process_count: int
    thread_count: int
    file_descriptors: int
    temperature: Optional[float] = None
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert metrics to feature vector for ML analysis"""
        return np.array([
            self.cpu_percent,
            self.memory_percent,
            self.disk_usage_percent,
            self.disk_io_read_mb,
            self.disk_io_write_mb,
            self.network_bytes_sent / 1024 / 1024,  # Convert to MB
            self.network_bytes_recv / 1024 / 1024,
            self.load_average[0] if self.load_average else 0,
            self.process_count,
            self.thread_count,
            self.file_descriptors
        ])


@dataclass
class PerformanceAlert:
    """Performance alert information"""
    id: str
    timestamp: str
    level: AlertLevel
    category: str
    message: str
    metrics: Dict[str, Any]
    suggested_action: str
    auto_resolved: bool = False


@dataclass
class OptimizationAction:
    """System optimization action"""
    id: str
    timestamp: str
    action_type: str
    description: str
    parameters: Dict[str, Any]
    success: bool
    impact_score: float
    execution_time: float


class AIPerformanceAnalyzer:
    """AI-powered performance analysis engine"""
    
    def __init__(self, model_dir: str = ".taskmaster/ai-models"):
        """Initialize AI performance analyzer"""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.anomaly_detector = None
        self.performance_predictor = None
        self.scaler = StandardScaler()
        
        self.feature_names = [
            'cpu_percent', 'memory_percent', 'disk_usage_percent',
            'disk_io_read_mb', 'disk_io_write_mb', 'network_sent_mb',
            'network_recv_mb', 'load_average', 'process_count',
            'thread_count', 'file_descriptors'
        ]
        
        self.training_data = []
        self.min_training_samples = 100
        self.retrain_interval = 3600  # 1 hour
        self.last_training = 0
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            anomaly_path = self.model_dir / "anomaly_detector.pkl"
            predictor_path = self.model_dir / "performance_predictor.pkl"
            scaler_path = self.model_dir / "scaler.pkl"
            
            if anomaly_path.exists():
                with open(anomaly_path, 'rb') as f:
                    self.anomaly_detector = pickle.load(f)
            
            if predictor_path.exists():
                with open(predictor_path, 'rb') as f:
                    self.performance_predictor = pickle.load(f)
            
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            if self.anomaly_detector:
                with open(self.model_dir / "anomaly_detector.pkl", 'wb') as f:
                    pickle.dump(self.anomaly_detector, f)
            
            if self.performance_predictor:
                with open(self.model_dir / "performance_predictor.pkl", 'wb') as f:
                    pickle.dump(self.performance_predictor, f)
            
            with open(self.model_dir / "scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
                
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def add_training_data(self, metrics: SystemMetrics, performance_score: float = None):
        """Add new metrics data for training"""
        feature_vector = metrics.to_feature_vector()
        
        self.training_data.append({
            'features': feature_vector,
            'timestamp': metrics.timestamp,
            'performance_score': performance_score
        })
        
        # Limit training data size
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-5000:]
        
        # Auto-retrain periodically
        if (time.time() - self.last_training > self.retrain_interval and 
            len(self.training_data) >= self.min_training_samples):
            self.train_models()
    
    def train_models(self):
        """Train anomaly detection and performance prediction models"""
        if len(self.training_data) < self.min_training_samples:
            print(f"Insufficient training data: {len(self.training_data)} < {self.min_training_samples}")
            return
        
        print("Training AI models...")
        
        # Prepare training data
        features = np.array([d['features'] for d in self.training_data])
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train anomaly detector
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.anomaly_detector.fit(features_scaled)
        
        # Train performance predictor if we have performance scores
        performance_scores = [d['performance_score'] for d in self.training_data 
                            if d['performance_score'] is not None]
        
        if len(performance_scores) > 50:
            valid_indices = [i for i, d in enumerate(self.training_data) 
                           if d['performance_score'] is not None]
            
            if valid_indices:
                X = features_scaled[valid_indices]
                y = np.array(performance_scores)
                
                if len(X) > 10:
                    self.performance_predictor = RandomForestRegressor(
                        n_estimators=100,
                        random_state=42
                    )
                    self.performance_predictor.fit(X, y)
        
        self.last_training = time.time()
        self._save_models()
        print("AI models training completed")
    
    def detect_anomalies(self, metrics: SystemMetrics) -> Tuple[bool, float]:
        """Detect if current metrics indicate an anomaly"""
        if not self.anomaly_detector:
            return False, 0.0
        
        try:
            features = metrics.to_feature_vector().reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Get anomaly score (-1 for anomaly, 1 for normal)
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
            
            # Convert to confidence score (0-1)
            confidence = max(0, min(1, (anomaly_score + 0.5) / 1.0))
            
            return is_anomaly, confidence
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return False, 0.0
    
    def predict_performance(self, metrics: SystemMetrics) -> Optional[float]:
        """Predict system performance score"""
        if not self.performance_predictor:
            return None
        
        try:
            features = metrics.to_feature_vector().reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            predicted_score = self.performance_predictor.predict(features_scaled)[0]
            return max(0, min(1, predicted_score))
            
        except Exception as e:
            print(f"Error in performance prediction: {e}")
            return None
    
    def analyze_bottlenecks(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Analyze system bottlenecks using feature importance"""
        bottlenecks = []
        
        # Rule-based bottleneck detection
        if metrics.cpu_percent > 90:
            bottlenecks.append({
                'type': 'cpu',
                'severity': 'critical',
                'value': metrics.cpu_percent,
                'threshold': 90,
                'description': 'CPU usage critically high'
            })
        elif metrics.cpu_percent > 80:
            bottlenecks.append({
                'type': 'cpu',
                'severity': 'warning',
                'value': metrics.cpu_percent,
                'threshold': 80,
                'description': 'CPU usage high'
            })
        
        if metrics.memory_percent > 95:
            bottlenecks.append({
                'type': 'memory',
                'severity': 'critical',
                'value': metrics.memory_percent,
                'threshold': 95,
                'description': 'Memory usage critically high'
            })
        elif metrics.memory_percent > 85:
            bottlenecks.append({
                'type': 'memory',
                'severity': 'warning',
                'value': metrics.memory_percent,
                'threshold': 85,
                'description': 'Memory usage high'
            })
        
        if metrics.disk_usage_percent > 95:
            bottlenecks.append({
                'type': 'disk',
                'severity': 'critical',
                'value': metrics.disk_usage_percent,
                'threshold': 95,
                'description': 'Disk usage critically high'
            })
        elif metrics.disk_usage_percent > 90:
            bottlenecks.append({
                'type': 'disk',
                'severity': 'warning',
                'value': metrics.disk_usage_percent,
                'threshold': 90,
                'description': 'Disk usage high'
            })
        
        # Load average analysis
        if metrics.load_average and len(metrics.load_average) > 0:
            cpu_cores = psutil.cpu_count()
            load_ratio = metrics.load_average[0] / cpu_cores
            
            if load_ratio > 2.0:
                bottlenecks.append({
                    'type': 'load',
                    'severity': 'critical',
                    'value': load_ratio,
                    'threshold': 2.0,
                    'description': 'System load extremely high'
                })
            elif load_ratio > 1.5:
                bottlenecks.append({
                    'type': 'load',
                    'severity': 'warning',
                    'value': load_ratio,
                    'threshold': 1.5,
                    'description': 'System load high'
                })
        
        return bottlenecks


class AutonomousSelfHealer:
    """Autonomous self-healing system"""
    
    def __init__(self, optimization_db: str = ".taskmaster/optimization.db"):
        """Initialize self-healing system"""
        self.db_path = optimization_db
        self.action_history = []
        self.healing_strategies = self._init_healing_strategies()
        self.cooldown_periods = {}  # Action type -> last execution time
        self.min_cooldown = 300  # 5 minutes between same actions
        
        self._init_database()
    
    def _init_database(self):
        """Initialize optimization database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_actions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                action_type TEXT,
                description TEXT,
                parameters TEXT,
                success BOOLEAN,
                impact_score REAL,
                execution_time REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                trigger_metric TEXT,
                intervention_type TEXT,
                success BOOLEAN,
                before_value REAL,
                after_value REAL,
                effectiveness_score REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _init_healing_strategies(self) -> Dict[str, Dict]:
        """Initialize self-healing strategies"""
        return {
            'high_cpu': {
                'actions': ['restart_high_cpu_processes', 'adjust_cpu_governor', 'kill_runaway_processes'],
                'thresholds': {'warning': 80, 'critical': 90},
                'cooldown': 300
            },
            'high_memory': {
                'actions': ['clear_cache', 'restart_memory_intensive_processes', 'garbage_collect'],
                'thresholds': {'warning': 85, 'critical': 95},
                'cooldown': 180
            },
            'high_disk': {
                'actions': ['cleanup_temp_files', 'rotate_logs', 'compress_old_files'],
                'thresholds': {'warning': 90, 'critical': 95},
                'cooldown': 600
            },
            'high_load': {
                'actions': ['rebalance_workload', 'pause_non_critical_tasks', 'scale_resources'],
                'thresholds': {'warning': 1.5, 'critical': 2.0},
                'cooldown': 240
            },
            'service_failure': {
                'actions': ['restart_service', 'fallback_service', 'service_health_check'],
                'thresholds': {'warning': 1, 'critical': 3},
                'cooldown': 120
            }
        }
    
    def diagnose_and_heal(self, metrics: SystemMetrics, 
                         bottlenecks: List[Dict[str, Any]]) -> List[OptimizationAction]:
        """Diagnose issues and apply healing actions"""
        actions_taken = []
        
        for bottleneck in bottlenecks:
            if bottleneck['severity'] not in ['warning', 'critical']:
                continue
            
            # Determine healing strategy
            strategy_key = f"high_{bottleneck['type']}"
            if strategy_key not in self.healing_strategies:
                continue
            
            strategy = self.healing_strategies[strategy_key]
            
            # Check cooldown
            if self._is_on_cooldown(strategy_key):
                continue
            
            # Select appropriate action
            action_name = self._select_healing_action(strategy, bottleneck)
            if not action_name:
                continue
            
            # Execute healing action
            action = self._execute_healing_action(action_name, bottleneck, metrics)
            if action:
                actions_taken.append(action)
                self._record_action(action)
                self.cooldown_periods[strategy_key] = time.time()
        
        return actions_taken
    
    def _is_on_cooldown(self, action_type: str) -> bool:
        """Check if action type is on cooldown"""
        if action_type not in self.cooldown_periods:
            return False
        
        last_execution = self.cooldown_periods[action_type]
        return time.time() - last_execution < self.min_cooldown
    
    def _select_healing_action(self, strategy: Dict, bottleneck: Dict) -> Optional[str]:
        """Select most appropriate healing action"""
        actions = strategy['actions']
        
        # For now, select first action that hasn't been used recently
        for action in actions:
            if not self._is_on_cooldown(f"{action}_{bottleneck['type']}"):
                return action
        
        return actions[0] if actions else None
    
    def _execute_healing_action(self, action_name: str, bottleneck: Dict, 
                               metrics: SystemMetrics) -> Optional[OptimizationAction]:
        """Execute specific healing action"""
        start_time = time.time()
        action_id = f"{action_name}_{int(start_time)}"
        
        try:
            success = False
            impact_score = 0.0
            
            if action_name == 'restart_high_cpu_processes':
                success, impact_score = self._restart_high_cpu_processes()
            elif action_name == 'clear_cache':
                success, impact_score = self._clear_system_cache()
            elif action_name == 'cleanup_temp_files':
                success, impact_score = self._cleanup_temp_files()
            elif action_name == 'rotate_logs':
                success, impact_score = self._rotate_logs()
            elif action_name == 'adjust_cpu_governor':
                success, impact_score = self._adjust_cpu_governor()
            elif action_name == 'garbage_collect':
                success, impact_score = self._force_garbage_collection()
            elif action_name == 'rebalance_workload':
                success, impact_score = self._rebalance_workload()
            else:
                print(f"Unknown healing action: {action_name}")
                return None
            
            execution_time = time.time() - start_time
            
            action = OptimizationAction(
                id=action_id,
                timestamp=datetime.now().isoformat(),
                action_type=action_name,
                description=f"Healing action for {bottleneck['type']} bottleneck",
                parameters={
                    'bottleneck_type': bottleneck['type'],
                    'bottleneck_value': bottleneck['value'],
                    'threshold': bottleneck['threshold']
                },
                success=success,
                impact_score=impact_score,
                execution_time=execution_time
            )
            
            return action
            
        except Exception as e:
            print(f"Error executing healing action {action_name}: {e}")
            return None
    
    def _restart_high_cpu_processes(self) -> Tuple[bool, float]:
        """Restart processes with high CPU usage"""
        try:
            # Find processes with high CPU usage
            high_cpu_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    if proc.info['cpu_percent'] > 50:  # More than 50% CPU
                        high_cpu_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not high_cpu_processes:
                return True, 0.1
            
            # Restart non-critical processes
            restarted = 0
            for proc in high_cpu_processes[:3]:  # Limit to 3 processes
                try:
                    proc_name = proc.info['name']
                    if self._is_safe_to_restart(proc_name):
                        proc.terminate()
                        restarted += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            impact_score = min(1.0, restarted * 0.3)
            return restarted > 0, impact_score
            
        except Exception as e:
            print(f"Error restarting high CPU processes: {e}")
            return False, 0.0
    
    def _clear_system_cache(self) -> Tuple[bool, float]:
        """Clear system caches to free memory"""
        try:
            if sys.platform == 'linux':
                # Linux: clear page cache, dentries and inodes
                subprocess.run(['sudo', 'sysctl', 'vm.drop_caches=3'], 
                              check=True, capture_output=True)
                return True, 0.5
            elif sys.platform == 'darwin':
                # macOS: purge memory
                subprocess.run(['sudo', 'purge'], 
                              check=True, capture_output=True)
                return True, 0.4
            else:
                return False, 0.0
                
        except subprocess.CalledProcessError:
            return False, 0.0
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False, 0.0
    
    def _cleanup_temp_files(self) -> Tuple[bool, float]:
        """Clean up temporary files to free disk space"""
        try:
            temp_dirs = ['/tmp', '/var/tmp', '/Users/*/Library/Caches']
            cleaned_mb = 0
            
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    try:
                        # Find and remove files older than 7 days
                        result = subprocess.run([
                            'find', temp_dir, '-type', 'f', '-mtime', '+7', '-delete'
                        ], capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            cleaned_mb += 10  # Estimate
                    except Exception:
                        continue
            
            impact_score = min(1.0, cleaned_mb / 1000)  # MB to score
            return cleaned_mb > 0, impact_score
            
        except Exception as e:
            print(f"Error cleaning temp files: {e}")
            return False, 0.0
    
    def _rotate_logs(self) -> Tuple[bool, float]:
        """Rotate and compress old log files"""
        try:
            log_dirs = ['.taskmaster/logs', '/var/log']
            rotated_files = 0
            
            for log_dir in log_dirs:
                if os.path.exists(log_dir):
                    for log_file in Path(log_dir).glob('*.log'):
                        if log_file.stat().st_size > 100 * 1024 * 1024:  # > 100MB
                            try:
                                # Compress and rotate
                                subprocess.run([
                                    'gzip', '-c', str(log_file)
                                ], stdout=open(f"{log_file}.gz", 'wb'))
                                
                                # Truncate original
                                with open(log_file, 'w') as f:
                                    f.write('')
                                
                                rotated_files += 1
                            except Exception:
                                continue
            
            impact_score = min(1.0, rotated_files * 0.2)
            return rotated_files > 0, impact_score
            
        except Exception as e:
            print(f"Error rotating logs: {e}")
            return False, 0.0
    
    def _adjust_cpu_governor(self) -> Tuple[bool, float]:
        """Adjust CPU governor for better performance"""
        try:
            if sys.platform == 'linux':
                # Set CPU governor to performance mode
                subprocess.run([
                    'sudo', 'cpupower', 'frequency-set', '-g', 'performance'
                ], check=True, capture_output=True)
                return True, 0.3
            else:
                return False, 0.0
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, 0.0
        except Exception as e:
            print(f"Error adjusting CPU governor: {e}")
            return False, 0.0
    
    def _force_garbage_collection(self) -> Tuple[bool, float]:
        """Force garbage collection in Python processes"""
        try:
            import gc
            gc.collect()
            
            # Also send SIGUSR1 to Python processes to trigger GC
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            signaled = 0
            for proc in python_processes[:5]:  # Limit to 5 processes
                try:
                    proc.send_signal(10)  # SIGUSR1
                    signaled += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            impact_score = min(1.0, signaled * 0.1)
            return signaled > 0, impact_score
            
        except Exception as e:
            print(f"Error forcing garbage collection: {e}")
            return False, 0.0
    
    def _rebalance_workload(self) -> Tuple[bool, float]:
        """Rebalance workload distribution"""
        try:
            # This is a placeholder for workload rebalancing
            # In a real implementation, this would distribute tasks across cores/nodes
            
            # For now, just adjust process priorities
            adjusted = 0
            for proc in psutil.process_iter(['pid', 'name', 'nice']):
                try:
                    if proc.info['nice'] < 5:  # Normal priority
                        proc.nice(5)  # Lower priority slightly
                        adjusted += 1
                        if adjusted >= 5:  # Limit adjustments
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            impact_score = min(1.0, adjusted * 0.1)
            return adjusted > 0, impact_score
            
        except Exception as e:
            print(f"Error rebalancing workload: {e}")
            return False, 0.0
    
    def _is_safe_to_restart(self, process_name: str) -> bool:
        """Check if process is safe to restart"""
        unsafe_processes = [
            'kernel', 'init', 'systemd', 'launchd', 'WindowServer',
            'loginwindow', 'Dock', 'Finder', 'ssh', 'sshd'
        ]
        
        return not any(unsafe in process_name.lower() for unsafe in unsafe_processes)
    
    def _record_action(self, action: OptimizationAction):
        """Record optimization action in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO optimization_actions
                (id, timestamp, action_type, description, parameters, 
                 success, impact_score, execution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                action.id, action.timestamp, action.action_type,
                action.description, json.dumps(action.parameters),
                action.success, action.impact_score, action.execution_time
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error recording action: {e}")


class SystemOptimizationSuite:
    """Main system optimization and monitoring suite"""
    
    def __init__(self, config_file: str = ".taskmaster/optimization-config.json"):
        """Initialize optimization suite"""
        self.config_file = config_file
        self.config = self._load_config()
        
        self.ai_analyzer = AIPerformanceAnalyzer()
        self.self_healer = AutonomousSelfHealer()
        
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = self.config.get('monitoring_interval', 30)
        
        self.alerts = []
        self.system_state = SystemState.UNKNOWN
        
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load optimization configuration"""
        default_config = {
            'monitoring_interval': 30,
            'ai_training_enabled': True,
            'auto_healing_enabled': True,
            'alert_thresholds': {
                'cpu_warning': 80,
                'cpu_critical': 90,
                'memory_warning': 85,
                'memory_critical': 95,
                'disk_warning': 90,
                'disk_critical': 95
            },
            'healing_cooldowns': {
                'cpu': 300,
                'memory': 180,
                'disk': 600,
                'load': 240
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
        except Exception as e:
            print(f"Error loading config: {e}")
        
        return default_config
    
    def _setup_logging(self):
        """Setup optimization logging"""
        log_dir = Path(".taskmaster/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "optimization.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("OptimizationSuite")
    
    def start_monitoring(self):
        """Start continuous system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("System optimization monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=30)
        
        self.logger.info("System optimization monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Analyze with AI
                if self.config['ai_training_enabled']:
                    self.ai_analyzer.add_training_data(metrics)
                
                # Detect anomalies
                is_anomaly, confidence = self.ai_analyzer.detect_anomalies(metrics)
                
                # Analyze bottlenecks
                bottlenecks = self.ai_analyzer.analyze_bottlenecks(metrics)
                
                # Update system state
                self._update_system_state(metrics, bottlenecks, is_anomaly)
                
                # Generate alerts
                alerts = self._generate_alerts(metrics, bottlenecks, is_anomaly, confidence)
                self.alerts.extend(alerts)
                
                # Auto-healing if enabled
                if self.config['auto_healing_enabled'] and bottlenecks:
                    actions = self.self_healer.diagnose_and_heal(metrics, bottlenecks)
                    if actions:
                        self.logger.info(f"Executed {len(actions)} healing actions")
                
                # Cleanup old alerts
                self._cleanup_old_alerts()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_read_mb = disk_io.read_bytes / 1024 / 1024 if disk_io else 0
            disk_io_write_mb = disk_io.write_bytes / 1024 / 1024 if disk_io else 0
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_bytes_sent = network_io.bytes_sent if network_io else 0
            network_bytes_recv = network_io.bytes_recv if network_io else 0
            
            # Load average
            load_average = list(os.getloadavg()) if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            # Process and thread counts
            process_count = len(psutil.pids())
            thread_count = sum(p.num_threads() for p in psutil.process_iter() 
                              if hasattr(p, 'num_threads'))
            
            # File descriptors (Linux/macOS)
            try:
                if sys.platform == 'linux':
                    with open('/proc/sys/fs/file-nr', 'r') as f:
                        file_descriptors = int(f.read().split()[0])
                else:
                    file_descriptors = 0
            except:
                file_descriptors = 0
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage_percent=disk_usage_percent,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                load_average=load_average,
                process_count=process_count,
                thread_count=thread_count,
                file_descriptors=file_descriptors
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            # Return minimal metrics
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0, memory_percent=0, disk_usage_percent=0,
                disk_io_read_mb=0, disk_io_write_mb=0,
                network_bytes_sent=0, network_bytes_recv=0,
                load_average=[0, 0, 0], process_count=0,
                thread_count=0, file_descriptors=0
            )
    
    def _update_system_state(self, metrics: SystemMetrics, bottlenecks: List[Dict],
                           is_anomaly: bool):
        """Update overall system state"""
        critical_bottlenecks = [b for b in bottlenecks if b['severity'] == 'critical']
        warning_bottlenecks = [b for b in bottlenecks if b['severity'] == 'warning']
        
        if critical_bottlenecks or is_anomaly:
            self.system_state = SystemState.CRITICAL
        elif warning_bottlenecks:
            self.system_state = SystemState.DEGRADED
        else:
            self.system_state = SystemState.OPTIMAL
    
    def _generate_alerts(self, metrics: SystemMetrics, bottlenecks: List[Dict],
                        is_anomaly: bool, anomaly_confidence: float) -> List[PerformanceAlert]:
        """Generate performance alerts"""
        alerts = []
        
        # Bottleneck alerts
        for bottleneck in bottlenecks:
            alert_level = AlertLevel.CRITICAL if bottleneck['severity'] == 'critical' else AlertLevel.WARNING
            
            alert = PerformanceAlert(
                id=f"bottleneck_{bottleneck['type']}_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                level=alert_level,
                category='bottleneck',
                message=f"{bottleneck['description']}: {bottleneck['value']:.1f}%",
                metrics=asdict(metrics),
                suggested_action=f"Consider optimizing {bottleneck['type']} usage"
            )
            alerts.append(alert)
        
        # Anomaly alerts
        if is_anomaly and anomaly_confidence > 0.7:
            alert = PerformanceAlert(
                id=f"anomaly_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                level=AlertLevel.WARNING,
                category='anomaly',
                message=f"Anomalous system behavior detected (confidence: {anomaly_confidence:.2f})",
                metrics=asdict(metrics),
                suggested_action="Review system metrics and recent changes"
            )
            alerts.append(alert)
        
        return alerts
    
    def _cleanup_old_alerts(self):
        """Remove old alerts to prevent memory buildup"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        cutoff_str = cutoff_time.isoformat()
        
        self.alerts = [alert for alert in self.alerts 
                      if alert.timestamp > cutoff_str]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        recent_alerts = [alert for alert in self.alerts[-10:]]  # Last 10 alerts
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_state': self.system_state.value,
            'monitoring_active': self.monitoring_active,
            'recent_alerts': [asdict(alert) for alert in recent_alerts],
            'ai_models_trained': {
                'anomaly_detector': self.ai_analyzer.anomaly_detector is not None,
                'performance_predictor': self.ai_analyzer.performance_predictor is not None
            },
            'training_data_size': len(self.ai_analyzer.training_data),
            'last_training': self.ai_analyzer.last_training
        }
    
    def run_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostic"""
        self.logger.info("Running system diagnostic...")
        
        # Collect current metrics
        metrics = self._collect_system_metrics()
        
        # Analyze with AI
        is_anomaly, anomaly_confidence = self.ai_analyzer.detect_anomalies(metrics)
        predicted_performance = self.ai_analyzer.predict_performance(metrics)
        bottlenecks = self.ai_analyzer.analyze_bottlenecks(metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(bottlenecks, metrics)
        
        diagnostic_report = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': asdict(metrics),
            'system_state': self.system_state.value,
            'anomaly_detected': is_anomaly,
            'anomaly_confidence': anomaly_confidence,
            'predicted_performance': predicted_performance,
            'bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'ai_analysis': {
                'models_available': self.ai_analyzer.anomaly_detector is not None,
                'training_samples': len(self.ai_analyzer.training_data)
            }
        }
        
        self.logger.info("System diagnostic completed")
        return diagnostic_report
    
    def _generate_recommendations(self, bottlenecks: List[Dict], 
                                 metrics: SystemMetrics) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'cpu':
                recommendations.append(
                    "Consider identifying and optimizing CPU-intensive processes"
                )
                recommendations.append(
                    "Review task parallelization and load balancing"
                )
            elif bottleneck['type'] == 'memory':
                recommendations.append(
                    "Monitor memory leaks and optimize memory usage"
                )
                recommendations.append(
                    "Consider increasing system memory or enabling swap"
                )
            elif bottleneck['type'] == 'disk':
                recommendations.append(
                    "Clean up unnecessary files and optimize disk usage"
                )
                recommendations.append(
                    "Consider disk space expansion or data archiving"
                )
            elif bottleneck['type'] == 'load':
                recommendations.append(
                    "Distribute workload across multiple cores/systems"
                )
                recommendations.append(
                    "Review and optimize concurrent task execution"
                )
        
        # General recommendations
        if not bottlenecks:
            recommendations.append("System performance appears optimal")
            recommendations.append("Continue regular monitoring and maintenance")
        
        return recommendations


def main():
    """Main function for running optimization suite"""
    suite = SystemOptimizationSuite()
    
    try:
        print("Starting AI-Powered System Optimization Suite...")
        
        # Start monitoring
        suite.start_monitoring()
        
        # Run initial diagnostic
        print("Running initial system diagnostic...")
        diagnostic = suite.run_diagnostic()
        print(f"System State: {diagnostic['system_state']}")
        print(f"Bottlenecks Found: {len(diagnostic['bottlenecks'])}")
        
        # Show status
        status = suite.get_system_status()
        print(f"Monitoring Active: {status['monitoring_active']}")
        print(f"AI Models Trained: {status['ai_models_trained']}")
        
        print("Optimization suite running. Press Ctrl+C to stop...")
        
        # Keep running
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\\nShutting down optimization suite...")
        suite.stop_monitoring()


if __name__ == "__main__":
    main()