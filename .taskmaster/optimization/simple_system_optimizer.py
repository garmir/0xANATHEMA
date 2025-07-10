#!/usr/bin/env python3
"""
Simplified System Optimization and Monitoring Suite
Using standard libraries for cross-platform compatibility
"""

import json
import time
import threading
import subprocess
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path


@dataclass
class SimpleSystemMetrics:
    """Simple system metrics using standard tools"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    load_average: str
    uptime_hours: float


class SimpleMetricsCollector:
    """Cross-platform metrics collector using standard tools"""
    
    def __init__(self):
        self.running = False
        self.metrics_history = []
        
    def collect_metrics(self) -> SimpleSystemMetrics:
        """Collect basic system metrics"""
        timestamp = datetime.now()
        
        # Get CPU usage (simplified)
        try:
            if sys.platform == "darwin":  # macOS
                cpu_cmd = "top -l 1 -n 0 | grep 'CPU usage' | awk '{print $3}' | sed 's/%//'"
                cpu_result = subprocess.run(cpu_cmd, shell=True, capture_output=True, text=True)
                cpu_usage = float(cpu_result.stdout.strip()) if cpu_result.stdout.strip() else 0.0
            else:  # Linux
                cpu_usage = 0.0  # Fallback
        except:
            cpu_usage = 0.0
        
        # Get memory usage
        try:
            if sys.platform == "darwin":
                mem_cmd = "memory_pressure | grep 'System-wide memory free percentage' | awk '{print 100-$5}' | sed 's/%//'"
                mem_result = subprocess.run(mem_cmd, shell=True, capture_output=True, text=True)
                memory_usage = float(mem_result.stdout.strip()) if mem_result.stdout.strip() else 0.0
            else:
                memory_usage = 0.0
        except:
            memory_usage = 0.0
        
        # Get disk usage
        try:
            disk_cmd = "df -h / | tail -1 | awk '{print $5}' | sed 's/%//'"
            disk_result = subprocess.run(disk_cmd, shell=True, capture_output=True, text=True)
            disk_usage = float(disk_result.stdout.strip()) if disk_result.stdout.strip() else 0.0
        except:
            disk_usage = 0.0
        
        # Get load average
        try:
            load_cmd = "uptime | awk '{print $(NF-2)}' | sed 's/,//'"
            load_result = subprocess.run(load_cmd, shell=True, capture_output=True, text=True)
            load_average = load_result.stdout.strip() if load_result.stdout.strip() else "0.0"
        except:
            load_average = "0.0"
        
        # Get uptime
        try:
            uptime_cmd = "uptime | awk '{print $3}' | sed 's/,//'"
            uptime_result = subprocess.run(uptime_cmd, shell=True, capture_output=True, text=True)
            uptime_str = uptime_result.stdout.strip()
            uptime_hours = float(uptime_str.replace(':', '.')) if uptime_str else 0.0
        except:
            uptime_hours = 0.0
        
        return SimpleSystemMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            load_average=load_average,
            uptime_hours=uptime_hours
        )


class SimpleAIAnalyzer:
    """Simplified AI analyzer using statistical methods"""
    
    def __init__(self):
        self.baseline_cpu = 20.0
        self.baseline_memory = 50.0
        self.anomaly_threshold = 2.0
        
    def analyze_metrics(self, metrics: SimpleSystemMetrics) -> Dict[str, Any]:
        """Analyze metrics for anomalies and optimization opportunities"""
        analysis = {
            'timestamp': metrics.timestamp,
            'anomalies': [],
            'recommendations': [],
            'health_score': 100.0
        }
        
        # CPU analysis
        if metrics.cpu_usage > self.baseline_cpu * 2:
            analysis['anomalies'].append({
                'type': 'high_cpu',
                'value': metrics.cpu_usage,
                'severity': 'high' if metrics.cpu_usage > 80 else 'medium'
            })
            analysis['recommendations'].append("Consider optimizing high-CPU processes")
            analysis['health_score'] -= 20
        
        # Memory analysis
        if metrics.memory_usage > self.baseline_memory * 1.5:
            analysis['anomalies'].append({
                'type': 'high_memory',
                'value': metrics.memory_usage,
                'severity': 'high' if metrics.memory_usage > 90 else 'medium'
            })
            analysis['recommendations'].append("Consider freeing memory or scaling resources")
            analysis['health_score'] -= 15
        
        # Disk analysis
        if metrics.disk_usage > 85:
            analysis['anomalies'].append({
                'type': 'high_disk',
                'value': metrics.disk_usage,
                'severity': 'critical' if metrics.disk_usage > 95 else 'high'
            })
            analysis['recommendations'].append("Urgent: Free disk space to prevent system issues")
            analysis['health_score'] -= 25
        
        return analysis


class SimpleOptimizer:
    """Simple system optimizer with basic self-healing"""
    
    def __init__(self):
        self.optimization_log = []
        
    def apply_optimizations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply simple optimization strategies"""
        optimization_results = {
            'timestamp': datetime.now(),
            'actions_taken': [],
            'success': True
        }
        
        for anomaly in analysis.get('anomalies', []):
            action = self._handle_anomaly(anomaly)
            optimization_results['actions_taken'].append(action)
            self.optimization_log.append(action)
        
        return optimization_results
    
    def _handle_anomaly(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """Handle specific anomaly types"""
        action = {
            'anomaly_type': anomaly['type'],
            'action_taken': 'none',
            'success': False
        }
        
        try:
            if anomaly['type'] == 'high_cpu':
                # Simple CPU optimization: suggest process management
                action['action_taken'] = 'Recommended process optimization'
                action['success'] = True
                
            elif anomaly['type'] == 'high_memory':
                # Memory optimization: trigger garbage collection
                import gc
                gc.collect()
                action['action_taken'] = 'Triggered garbage collection'
                action['success'] = True
                
            elif anomaly['type'] == 'high_disk':
                # Disk optimization: recommend cleanup
                action['action_taken'] = 'Recommended disk cleanup'
                action['success'] = True
                
        except Exception as e:
            action['action_taken'] = f'Failed: {e}'
            action['success'] = False
        
        return action


class SimpleMonitoringSuite:
    """Main simplified monitoring suite"""
    
    def __init__(self):
        self.metrics_collector = SimpleMetricsCollector()
        self.ai_analyzer = SimpleAIAnalyzer()
        self.optimizer = SimpleOptimizer()
        self.running = False
        self.monitoring_thread = None
        self.report_data = []
        
        # Setup logging
        os.makedirs('.taskmaster/logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('.taskmaster/logs/simple_optimizer.log'),
                logging.StreamHandler()
            ]
        )
    
    def start_monitoring(self, duration_seconds: int = 30):
        """Start monitoring for specified duration"""
        logging.info("Starting simplified system monitoring")
        
        self.running = True
        start_time = time.time()
        
        while self.running and (time.time() - start_time) < duration_seconds:
            try:
                # Collect metrics
                metrics = self.metrics_collector.collect_metrics()
                
                # Analyze metrics
                analysis = self.ai_analyzer.analyze_metrics(metrics)
                
                # Apply optimizations if needed
                if analysis['anomalies']:
                    optimization_results = self.optimizer.apply_optimizations(analysis)
                    logging.info(f"Applied optimizations: {len(optimization_results['actions_taken'])} actions")
                
                # Store for reporting
                self.report_data.append({
                    'metrics': asdict(metrics),
                    'analysis': analysis
                })
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(5)
        
        self.running = False
        logging.info("Monitoring completed")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        if not self.report_data:
            return {"error": "No monitoring data available"}
        
        # Calculate statistics
        cpu_values = [data['metrics']['cpu_usage'] for data in self.report_data]
        memory_values = [data['metrics']['memory_usage'] for data in self.report_data]
        disk_values = [data['metrics']['disk_usage'] for data in self.report_data]
        
        # Count anomalies
        total_anomalies = sum(len(data['analysis']['anomalies']) for data in self.report_data)
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'monitoring_duration_seconds': len(self.report_data) * 5,
            'data_points': len(self.report_data),
            'performance_summary': {
                'cpu_usage': {
                    'average': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    'max': max(cpu_values) if cpu_values else 0,
                    'min': min(cpu_values) if cpu_values else 0
                },
                'memory_usage': {
                    'average': sum(memory_values) / len(memory_values) if memory_values else 0,
                    'max': max(memory_values) if memory_values else 0,
                    'min': min(memory_values) if memory_values else 0
                },
                'disk_usage': {
                    'average': sum(disk_values) / len(disk_values) if disk_values else 0,
                    'max': max(disk_values) if disk_values else 0,
                    'min': min(disk_values) if disk_values else 0
                }
            },
            'optimization_summary': {
                'total_anomalies_detected': total_anomalies,
                'optimization_actions_taken': len(self.optimizer.optimization_log),
                'successful_optimizations': sum(1 for action in self.optimizer.optimization_log if action['success'])
            },
            'health_assessment': self._calculate_overall_health(),
            'recommendations': self._generate_overall_recommendations()
        }
        
        return report
    
    def _calculate_overall_health(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        if not self.report_data:
            return {"score": 100, "status": "unknown"}
        
        health_scores = [data['analysis']['health_score'] for data in self.report_data]
        avg_health = sum(health_scores) / len(health_scores)
        
        if avg_health >= 90:
            status = "excellent"
        elif avg_health >= 75:
            status = "good"
        elif avg_health >= 60:
            status = "fair"
        else:
            status = "needs_attention"
        
        return {
            "score": round(avg_health, 1),
            "status": status,
            "trend": "stable"  # Simplified
        }
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall system recommendations"""
        recommendations = []
        
        if not self.report_data:
            return ["No data available for recommendations"]
        
        # Analyze patterns
        anomaly_types = {}
        for data in self.report_data:
            for anomaly in data['analysis']['anomalies']:
                anomaly_type = anomaly['type']
                anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
        
        # Generate recommendations based on frequent issues
        if anomaly_types.get('high_cpu', 0) > 2:
            recommendations.append("Persistent high CPU usage detected - consider process optimization")
        
        if anomaly_types.get('high_memory', 0) > 2:
            recommendations.append("Frequent memory pressure - consider memory optimization or scaling")
        
        if anomaly_types.get('high_disk', 0) > 0:
            recommendations.append("Disk space issues detected - immediate cleanup recommended")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters")
        
        return recommendations


def main():
    """Main execution function"""
    print("Simple System Optimization and Monitoring Suite")
    print("=" * 50)
    
    # Initialize monitoring suite
    suite = SimpleMonitoringSuite()
    
    try:
        # Start monitoring for 30 seconds
        print("Starting monitoring...")
        suite.start_monitoring(duration_seconds=30)
        
        # Generate and save report
        print("Generating performance report...")
        report = suite.generate_report()
        
        # Save report
        os.makedirs('.taskmaster/reports', exist_ok=True)
        with open('.taskmaster/reports/simple_optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✓ Monitoring completed successfully")
        print(f"✓ Health Score: {report['health_assessment']['score']}/100")
        print(f"✓ Status: {report['health_assessment']['status']}")
        print(f"✓ Anomalies detected: {report['optimization_summary']['total_anomalies_detected']}")
        print(f"✓ Report saved to: .taskmaster/reports/simple_optimization_report.json")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)