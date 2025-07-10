#!/usr/bin/env python3
"""
Golden Signals Monitoring Dashboard
Implements SRE best practices with the Four Golden Signals:
1. Latency - How long requests take
2. Traffic - How much demand is being placed on the system  
3. Errors - Rate of requests that fail
4. Saturation - How "full" the service is
"""

import asyncio
import json
import time
import os
import statistics
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

@dataclass
class LatencyMetric:
    """Latency measurement for a specific operation"""
    operation_name: str
    duration_ms: float
    timestamp: datetime
    success: bool
    
@dataclass 
class TrafficMetric:
    """Traffic measurement for system demand"""
    endpoint: str
    requests_per_second: float
    concurrent_users: int
    timestamp: datetime

@dataclass
class ErrorMetric:
    """Error rate measurement"""
    error_type: str
    error_count: int
    total_requests: int
    error_rate: float
    timestamp: datetime

@dataclass
class SaturationMetric:
    """Resource saturation measurement"""
    resource_type: str  # cpu, memory, disk, network
    utilization_percent: float
    capacity_remaining: float
    timestamp: datetime

class GoldenSignalsCollector:
    """Collects the Four Golden Signals metrics"""
    
    def __init__(self):
        self.latency_history: List[LatencyMetric] = []
        self.traffic_history: List[TrafficMetric] = []
        self.error_history: List[ErrorMetric] = []
        self.saturation_history: List[SaturationMetric] = []
        self.active_operations = {}
        self.request_counts = {}
        self.error_counts = {}
        
    def start_operation(self, operation_name: str) -> str:
        """Start timing an operation"""
        operation_id = f"{operation_name}_{time.time()}"
        self.active_operations[operation_id] = {
            "name": operation_name,
            "start_time": time.time()
        }
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True):
        """End timing an operation and record latency"""
        if operation_id not in self.active_operations:
            return
            
        operation = self.active_operations.pop(operation_id)
        duration_ms = (time.time() - operation["start_time"]) * 1000
        
        latency = LatencyMetric(
            operation_name=operation["name"],
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            success=success
        )
        
        self.latency_history.append(latency)
        
        # Keep only last 1000 measurements
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-1000:]
    
    def record_traffic(self, endpoint: str, requests_per_second: float, concurrent_users: int = 1):
        """Record traffic metrics"""
        traffic = TrafficMetric(
            endpoint=endpoint,
            requests_per_second=requests_per_second,
            concurrent_users=concurrent_users,
            timestamp=datetime.now()
        )
        
        self.traffic_history.append(traffic)
        
        # Keep only last 500 measurements  
        if len(self.traffic_history) > 500:
            self.traffic_history = self.traffic_history[-500:]
    
    def record_error(self, error_type: str, total_requests: int):
        """Record error metrics"""
        # Track error counts
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Track request counts  
        endpoint = error_type.split(":")[0] if ":" in error_type else "general"
        if endpoint not in self.request_counts:
            self.request_counts[endpoint] = 0
        self.request_counts[endpoint] = total_requests
        
        error_count = self.error_counts[error_type]
        error_rate = (error_count / total_requests) if total_requests > 0 else 0
        
        error = ErrorMetric(
            error_type=error_type,
            error_count=error_count,
            total_requests=total_requests,
            error_rate=error_rate,
            timestamp=datetime.now()
        )
        
        self.error_history.append(error)
        
        # Keep only last 500 measurements
        if len(self.error_history) > 500:
            self.error_history = self.error_history[-500:]
    
    def collect_saturation_metrics(self):
        """Collect current system saturation metrics using stdlib"""
        try:
            # CPU saturation (simplified)
            if sys.platform == 'darwin':  # macOS
                try:
                    result = subprocess.run(['top', '-l', '1', '-s', '0', '-n', '0'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'CPU usage:' in line:
                                # Parse something like "CPU usage: 12.34% user, 5.67% sys, 81.99% idle"
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if 'idle' in part and i > 0:
                                        idle_percent = float(parts[i-1].rstrip('%'))
                                        cpu_percent = 100 - idle_percent
                                        break
                                else:
                                    cpu_percent = 20  # Default estimate
                            
                    else:
                        cpu_percent = 20  # Default estimate
                except:
                    cpu_percent = 20  # Default estimate
            else:
                cpu_percent = 25  # Default estimate for other platforms
            
            cpu_saturation = SaturationMetric(
                resource_type="cpu",
                utilization_percent=cpu_percent,
                capacity_remaining=100 - cpu_percent,
                timestamp=datetime.now()
            )
            self.saturation_history.append(cpu_saturation)
            
            # Memory saturation (simplified estimation)
            try:
                if sys.platform == 'darwin':  # macOS
                    result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5)
                    # Very simplified memory calculation - actual implementation would be more complex
                    memory_percent = 45  # Default estimate
                else:
                    memory_percent = 50  # Default estimate
            except:
                memory_percent = 50  # Default estimate
                
            memory_saturation = SaturationMetric(
                resource_type="memory", 
                utilization_percent=memory_percent,
                capacity_remaining=100 - memory_percent,
                timestamp=datetime.now()
            )
            self.saturation_history.append(memory_saturation)
            
            # Disk saturation
            try:
                if sys.platform == 'darwin':  # macOS
                    result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            parts = lines[1].split()
                            if len(parts) >= 5:
                                use_percent = parts[4].rstrip('%')
                                disk_percent = float(use_percent)
                            else:
                                disk_percent = 30  # Default
                        else:
                            disk_percent = 30  # Default
                    else:
                        disk_percent = 30  # Default
                else:
                    disk_percent = 35  # Default estimate
            except:
                disk_percent = 35  # Default estimate
                
            disk_saturation = SaturationMetric(
                resource_type="disk",
                utilization_percent=disk_percent,
                capacity_remaining=100 - disk_percent,
                timestamp=datetime.now()
            )
            self.saturation_history.append(disk_saturation)
            
            # Keep only last 200 measurements per resource
            if len(self.saturation_history) > 800:  # 200 * 4 resources
                self.saturation_history = self.saturation_history[-800:]
                
        except Exception as e:
            logging.error(f"Failed to collect saturation metrics: {e}")
            # Add fallback metrics
            for resource_type, default_util in [("cpu", 25), ("memory", 50), ("disk", 35)]:
                fallback_metric = SaturationMetric(
                    resource_type=resource_type,
                    utilization_percent=default_util,
                    capacity_remaining=100 - default_util,
                    timestamp=datetime.now()
                )
                self.saturation_history.append(fallback_metric)

class GoldenSignalsAnalyzer:
    """Analyzes Golden Signals metrics and provides insights"""
    
    def __init__(self, collector: GoldenSignalsCollector):
        self.collector = collector
    
    def analyze_latency(self, time_window_minutes: int = 5) -> Dict[str, Any]:
        """Analyze latency patterns"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_latencies = [
            metric for metric in self.collector.latency_history
            if metric.timestamp >= cutoff_time
        ]
        
        if not recent_latencies:
            return {"status": "no_data", "window_minutes": time_window_minutes}
        
        # Calculate percentiles
        durations = [m.duration_ms for m in recent_latencies]
        success_rates = [m.success for m in recent_latencies]
        
        analysis = {
            "window_minutes": time_window_minutes,
            "total_operations": len(recent_latencies),
            "success_rate": sum(success_rates) / len(success_rates),
            "latency_p50": statistics.median(durations),
            "latency_p95": statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations),
            "latency_p99": statistics.quantiles(durations, n=100)[98] if len(durations) >= 100 else max(durations),
            "avg_latency": statistics.mean(durations),
            "max_latency": max(durations),
            "min_latency": min(durations)
        }
        
        # Determine status
        if analysis["latency_p95"] > 5000:  # 5 seconds
            analysis["status"] = "critical"
            analysis["alert"] = "P95 latency exceeds 5 seconds"
        elif analysis["latency_p95"] > 2000:  # 2 seconds
            analysis["status"] = "warning"
            analysis["alert"] = "P95 latency exceeds 2 seconds"
        else:
            analysis["status"] = "healthy"
        
        return analysis
    
    def analyze_traffic(self, time_window_minutes: int = 5) -> Dict[str, Any]:
        """Analyze traffic patterns"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_traffic = [
            metric for metric in self.collector.traffic_history
            if metric.timestamp >= cutoff_time
        ]
        
        if not recent_traffic:
            return {"status": "no_data", "window_minutes": time_window_minutes}
        
        # Aggregate by endpoint
        endpoint_stats = {}
        total_rps = 0
        total_users = 0
        
        for metric in recent_traffic:
            if metric.endpoint not in endpoint_stats:
                endpoint_stats[metric.endpoint] = {
                    "requests": [],
                    "users": []
                }
            endpoint_stats[metric.endpoint]["requests"].append(metric.requests_per_second)
            endpoint_stats[metric.endpoint]["users"].append(metric.concurrent_users)
            total_rps += metric.requests_per_second
            total_users += metric.concurrent_users
        
        analysis = {
            "window_minutes": time_window_minutes,
            "total_rps": total_rps / len(recent_traffic) if recent_traffic else 0,
            "total_concurrent_users": total_users / len(recent_traffic) if recent_traffic else 0,
            "endpoint_breakdown": {}
        }
        
        for endpoint, stats in endpoint_stats.items():
            analysis["endpoint_breakdown"][endpoint] = {
                "avg_rps": statistics.mean(stats["requests"]),
                "max_rps": max(stats["requests"]),
                "avg_users": statistics.mean(stats["users"])
            }
        
        # Determine status based on load
        if analysis["total_rps"] > 1000:
            analysis["status"] = "high_load"
        elif analysis["total_rps"] > 100:
            analysis["status"] = "moderate_load"
        else:
            analysis["status"] = "low_load"
        
        return analysis
    
    def analyze_errors(self, time_window_minutes: int = 5) -> Dict[str, Any]:
        """Analyze error patterns"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_errors = [
            metric for metric in self.collector.error_history
            if metric.timestamp >= cutoff_time
        ]
        
        if not recent_errors:
            return {"status": "no_data", "window_minutes": time_window_minutes}
        
        # Calculate overall error rate
        total_errors = sum(m.error_count for m in recent_errors)
        total_requests = sum(m.total_requests for m in recent_errors)
        overall_error_rate = (total_errors / total_requests) if total_requests > 0 else 0
        
        # Group by error type
        error_breakdown = {}
        for metric in recent_errors:
            if metric.error_type not in error_breakdown:
                error_breakdown[metric.error_type] = {
                    "count": 0,
                    "rates": []
                }
            error_breakdown[metric.error_type]["count"] += metric.error_count
            error_breakdown[metric.error_type]["rates"].append(metric.error_rate)
        
        analysis = {
            "window_minutes": time_window_minutes,
            "overall_error_rate": overall_error_rate,
            "total_errors": total_errors,
            "total_requests": total_requests,
            "error_breakdown": {}
        }
        
        for error_type, stats in error_breakdown.items():
            analysis["error_breakdown"][error_type] = {
                "count": stats["count"],
                "avg_rate": statistics.mean(stats["rates"]),
                "max_rate": max(stats["rates"])
            }
        
        # Determine status
        if overall_error_rate > 0.05:  # 5%
            analysis["status"] = "critical"
            analysis["alert"] = f"Error rate {overall_error_rate:.2%} exceeds 5%"
        elif overall_error_rate > 0.01:  # 1%
            analysis["status"] = "warning"
            analysis["alert"] = f"Error rate {overall_error_rate:.2%} exceeds 1%"
        else:
            analysis["status"] = "healthy"
        
        return analysis
    
    def analyze_saturation(self, time_window_minutes: int = 5) -> Dict[str, Any]:
        """Analyze resource saturation"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_saturation = [
            metric for metric in self.collector.saturation_history
            if metric.timestamp >= cutoff_time
        ]
        
        if not recent_saturation:
            return {"status": "no_data", "window_minutes": time_window_minutes}
        
        # Group by resource type
        resource_stats = {}
        for metric in recent_saturation:
            if metric.resource_type not in resource_stats:
                resource_stats[metric.resource_type] = []
            resource_stats[metric.resource_type].append(metric.utilization_percent)
        
        analysis = {
            "window_minutes": time_window_minutes,
            "resources": {}
        }
        
        max_utilization = 0
        critical_resource = None
        
        for resource_type, utilizations in resource_stats.items():
            avg_util = statistics.mean(utilizations)
            max_util = max(utilizations)
            
            analysis["resources"][resource_type] = {
                "avg_utilization": avg_util,
                "max_utilization": max_util,
                "current_utilization": utilizations[-1] if utilizations else 0
            }
            
            if max_util > max_utilization:
                max_utilization = max_util
                critical_resource = resource_type
        
        # Determine overall status
        if max_utilization > 90:
            analysis["status"] = "critical"
            analysis["alert"] = f"{critical_resource} utilization {max_utilization:.1f}% exceeds 90%"
        elif max_utilization > 70:
            analysis["status"] = "warning"
            analysis["alert"] = f"{critical_resource} utilization {max_utilization:.1f}% exceeds 70%"
        else:
            analysis["status"] = "healthy"
        
        analysis["max_utilization"] = max_utilization
        analysis["critical_resource"] = critical_resource
        
        return analysis

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard"""
    
    collector = None
    analyzer = None
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_dashboard_html()
        elif self.path == '/api/metrics':
            self.serve_metrics()
        elif self.path == '/api/analysis':
            self.serve_analysis()
        elif self.path == '/api/alerts':
            self.serve_alerts()
        else:
            self.send_error(404)
    
    def serve_dashboard_html(self):
        """Serve dashboard HTML"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        html = self.get_dashboard_html()
        self.wfile.write(html.encode())
    
    def get_dashboard_html(self):
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Golden Signals Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .signals-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
        .signal-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .signal-title { font-size: 20px; font-weight: bold; margin-bottom: 15px; }
        .metric { margin: 10px 0; }
        .metric-label { font-weight: bold; display: inline-block; width: 150px; }
        .metric-value { color: #333; }
        .status-healthy { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .alert { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; border-radius: 4px; margin: 10px 0; }
        .refresh-button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .timestamp { font-size: 12px; color: #666; text-align: right; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Golden Signals Dashboard</h1>
            <p>Four Golden Signals of Monitoring: Latency • Traffic • Errors • Saturation</p>
            <button class="refresh-button" onclick="refreshData()">🔄 Refresh</button>
            <div id="last-updated" class="timestamp"></div>
        </div>
        
        <div class="signals-grid">
            <div class="signal-card">
                <div class="signal-title">⚡ Latency</div>
                <div id="latency-content">Loading...</div>
            </div>
            
            <div class="signal-card">
                <div class="signal-title">📈 Traffic</div>
                <div id="traffic-content">Loading...</div>
            </div>
            
            <div class="signal-card">
                <div class="signal-title">❌ Errors</div>
                <div id="errors-content">Loading...</div>
            </div>
            
            <div class="signal-card">
                <div class="signal-title">🔋 Saturation</div>
                <div id="saturation-content">Loading...</div>
            </div>
        </div>
        
        <div id="alerts-section" style="margin-top: 30px;"></div>
    </div>

    <script>
        async function fetchData(endpoint) {
            try {
                const response = await fetch(endpoint);
                return await response.json();
            } catch (error) {
                console.error('Fetch error:', error);
                return null;
            }
        }
        
        function formatPercent(value) {
            return (value * 100).toFixed(1) + '%';
        }
        
        function formatLatency(ms) {
            if (ms > 1000) return (ms / 1000).toFixed(2) + 's';
            return ms.toFixed(1) + 'ms';
        }
        
        function getStatusClass(status) {
            return 'status-' + status;
        }
        
        async function refreshData() {
            const analysis = await fetchData('/api/analysis');
            if (!analysis) return;
            
            // Update latency
            const latency = analysis.latency;
            document.getElementById('latency-content').innerHTML = `
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value $${getStatusClass(latency.status || 'no_data')}">${(latency.status || 'no_data').toUpperCase()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">P50 Latency:</span>
                    <span class="metric-value">${formatLatency(latency.latency_p50 || 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">P95 Latency:</span>
                    <span class="metric-value">${formatLatency(latency.latency_p95 || 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value">${formatPercent(latency.success_rate || 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Operations:</span>
                    <span class="metric-value">${latency.total_operations || 0}</span>
                </div>
            `;
            
            // Update traffic
            const traffic = analysis.traffic;
            document.getElementById('traffic-content').innerHTML = `
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value $${getStatusClass(traffic.status || 'no_data')}">${(traffic.status || 'no_data').toUpperCase()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Requests/sec:</span>
                    <span class="metric-value">${(traffic.total_rps || 0).toFixed(1)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Concurrent Users:</span>
                    <span class="metric-value">${(traffic.total_concurrent_users || 0).toFixed(0)}</span>
                </div>
            `;
            
            // Update errors
            const errors = analysis.errors;
            document.getElementById('errors-content').innerHTML = `
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value $${getStatusClass(errors.status || 'no_data')}">${(errors.status || 'no_data').toUpperCase()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Error Rate:</span>
                    <span class="metric-value">${formatPercent(errors.overall_error_rate || 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Errors:</span>
                    <span class="metric-value">${errors.total_errors || 0}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Requests:</span>
                    <span class="metric-value">${errors.total_requests || 0}</span>
                </div>
            `;
            
            // Update saturation
            const saturation = analysis.saturation;
            document.getElementById('saturation-content').innerHTML = `
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value $${getStatusClass(saturation.status || 'no_data')}">${(saturation.status || 'no_data').toUpperCase()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Utilization:</span>
                    <span class="metric-value">${(saturation.max_utilization || 0).toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Critical Resource:</span>
                    <span class="metric-value">${saturation.critical_resource || 'None'}</span>
                </div>
            `;
            
            // Update alerts
            const alerts = await fetchData('/api/alerts');
            let alertsHtml = '';
            if (alerts && alerts.length > 0) {
                alertsHtml = '<h3>🚨 Active Alerts</h3>';
                alerts.forEach(alert => {
                    alertsHtml += `<div class="alert">${alert}</div>`;
                });
            }
            document.getElementById('alerts-section').innerHTML = alertsHtml;
            
            // Update timestamp
            document.getElementById('last-updated').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
        }
        
        // Initial load
        refreshData();
        
        // Auto-refresh every 10 seconds
        setInterval(refreshData, 10000);
    </script>
</body>
</html>
        """
    
    def serve_json(self, data):
        """Helper to serve JSON responses"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        json_str = json.dumps(data, default=str, indent=2)
        self.wfile.write(json_str.encode())
    
    def serve_metrics(self):
        """Serve metrics data"""
        if not self.collector:
            self.send_error(500)
            return
            
        metrics = {
            "latency": [asdict(m) for m in self.collector.latency_history[-100:]],
            "traffic": [asdict(m) for m in self.collector.traffic_history[-100:]],
            "errors": [asdict(m) for m in self.collector.error_history[-100:]],
            "saturation": [asdict(m) for m in self.collector.saturation_history[-100:]]
        }
        self.serve_json(metrics)
    
    def serve_analysis(self):
        """Serve analysis data"""
        if not self.analyzer:
            self.send_error(500)
            return
            
        analysis = {
            "latency": self.analyzer.analyze_latency(),
            "traffic": self.analyzer.analyze_traffic(),
            "errors": self.analyzer.analyze_errors(),
            "saturation": self.analyzer.analyze_saturation()
        }
        self.serve_json(analysis)
    
    def serve_alerts(self):
        """Serve alerts data"""
        if not self.analyzer:
            self.send_error(500)
            return
            
        analysis = {
            "latency": self.analyzer.analyze_latency(),
            "traffic": self.analyzer.analyze_traffic(),
            "errors": self.analyzer.analyze_errors(),
            "saturation": self.analyzer.analyze_saturation()
        }
        
        alerts = []
        for signal_name, signal_data in analysis.items():
            if signal_data.get("alert"):
                alerts.append(f"{signal_name.upper()}: {signal_data['alert']}")
        
        self.serve_json(alerts)

class GoldenSignalsDashboard:
    """Web dashboard for Golden Signals monitoring"""
    
    def __init__(self, collector: GoldenSignalsCollector, analyzer: GoldenSignalsAnalyzer):
        self.collector = collector
        self.analyzer = analyzer
        self.server = None
        
    def start_server(self, port=8080):
        """Start the HTTP server"""
        # Set class variables for the handler
        DashboardHandler.collector = self.collector
        DashboardHandler.analyzer = self.analyzer
        
        self.server = HTTPServer(('localhost', port), DashboardHandler)
        print(f"✅ Golden Signals Dashboard started on http://localhost:{port}")
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        server_thread.start()
        return server_thread
    
    def stop_server(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
    async def dashboard_html(self, request):
        """Serve dashboard HTML"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Golden Signals Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .signals-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
        .signal-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .signal-title { font-size: 20px; font-weight: bold; margin-bottom: 15px; }
        .metric { margin: 10px 0; }
        .metric-label { font-weight: bold; display: inline-block; width: 150px; }
        .metric-value { color: #333; }
        .status-healthy { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .alert { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; border-radius: 4px; margin: 10px 0; }
        .refresh-button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .timestamp { font-size: 12px; color: #666; text-align: right; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Golden Signals Dashboard</h1>
            <p>Four Golden Signals of Monitoring: Latency • Traffic • Errors • Saturation</p>
            <button class="refresh-button" onclick="refreshData()">🔄 Refresh</button>
            <div id="last-updated" class="timestamp"></div>
        </div>
        
        <div class="signals-grid">
            <div class="signal-card">
                <div class="signal-title">⚡ Latency</div>
                <div id="latency-content">Loading...</div>
            </div>
            
            <div class="signal-card">
                <div class="signal-title">📈 Traffic</div>
                <div id="traffic-content">Loading...</div>
            </div>
            
            <div class="signal-card">
                <div class="signal-title">❌ Errors</div>
                <div id="errors-content">Loading...</div>
            </div>
            
            <div class="signal-card">
                <div class="signal-title">🔋 Saturation</div>
                <div id="saturation-content">Loading...</div>
            </div>
        </div>
        
        <div id="alerts-section" style="margin-top: 30px;"></div>
    </div>

    <script>
        async function fetchData(endpoint) {
            try {
                const response = await fetch(endpoint);
                return await response.json();
            } catch (error) {
                console.error('Fetch error:', error);
                return null;
            }
        }
        
        function formatPercent(value) {
            return (value * 100).toFixed(1) + '%';
        }
        
        function formatLatency(ms) {
            if (ms > 1000) return (ms / 1000).toFixed(2) + 's';
            return ms.toFixed(1) + 'ms';
        }
        
        function getStatusClass(status) {
            return 'status-' + status;
        }
        
        async function refreshData() {
            const analysis = await fetchData('/api/analysis');
            if (!analysis) return;
            
            // Update latency
            const latency = analysis.latency;
            document.getElementById('latency-content').innerHTML = `
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value ${getStatusClass(latency.status)}">${latency.status.toUpperCase()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">P50 Latency:</span>
                    <span class="metric-value">${formatLatency(latency.latency_p50 || 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">P95 Latency:</span>
                    <span class="metric-value">${formatLatency(latency.latency_p95 || 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value">${formatPercent(latency.success_rate || 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Operations:</span>
                    <span class="metric-value">${latency.total_operations || 0}</span>
                </div>
            `;
            
            // Update traffic
            const traffic = analysis.traffic;
            document.getElementById('traffic-content').innerHTML = `
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value ${getStatusClass(traffic.status || 'no_data')}">${(traffic.status || 'no_data').toUpperCase()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Requests/sec:</span>
                    <span class="metric-value">${(traffic.total_rps || 0).toFixed(1)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Concurrent Users:</span>
                    <span class="metric-value">${(traffic.total_concurrent_users || 0).toFixed(0)}</span>
                </div>
            `;
            
            // Update errors
            const errors = analysis.errors;
            document.getElementById('errors-content').innerHTML = `
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value ${getStatusClass(errors.status || 'no_data')}">${(errors.status || 'no_data').toUpperCase()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Error Rate:</span>
                    <span class="metric-value">${formatPercent(errors.overall_error_rate || 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Errors:</span>
                    <span class="metric-value">${errors.total_errors || 0}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Requests:</span>
                    <span class="metric-value">${errors.total_requests || 0}</span>
                </div>
            `;
            
            // Update saturation
            const saturation = analysis.saturation;
            document.getElementById('saturation-content').innerHTML = `
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value ${getStatusClass(saturation.status || 'no_data')}">${(saturation.status || 'no_data').toUpperCase()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Utilization:</span>
                    <span class="metric-value">${(saturation.max_utilization || 0).toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Critical Resource:</span>
                    <span class="metric-value">${saturation.critical_resource || 'None'}</span>
                </div>
            `;
            
            // Update alerts
            const alerts = await fetchData('/api/alerts');
            let alertsHtml = '';
            if (alerts && alerts.length > 0) {
                alertsHtml = '<h3>🚨 Active Alerts</h3>';
                alerts.forEach(alert => {
                    alertsHtml += `<div class="alert">${alert}</div>`;
                });
            }
            document.getElementById('alerts-section').innerHTML = alertsHtml;
            
            // Update timestamp
            document.getElementById('last-updated').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
        }
        
        // Initial load
        refreshData();
        
        // Auto-refresh every 10 seconds
        setInterval(refreshData, 10000);
    </script>
</body>
</html>
        """
    
    async def get_metrics(self, request):
        """Get raw metrics data"""
        metrics = {
            "latency": [asdict(m) for m in self.collector.latency_history[-100:]],
            "traffic": [asdict(m) for m in self.collector.traffic_history[-100:]],
            "errors": [asdict(m) for m in self.collector.error_history[-100:]],
            "saturation": [asdict(m) for m in self.collector.saturation_history[-100:]]
        }
        return web.json_response(metrics, default=str)
    
    async def get_analysis(self, request):
        """Get analyzed Golden Signals data"""
        analysis = {
            "latency": self.analyzer.analyze_latency(),
            "traffic": self.analyzer.analyze_traffic(),
            "errors": self.analyzer.analyze_errors(),
            "saturation": self.analyzer.analyze_saturation()
        }
        return web.json_response(analysis, default=str)
    
    async def get_alerts(self, request):
        """Get active alerts"""
        analysis = {
            "latency": self.analyzer.analyze_latency(),
            "traffic": self.analyzer.analyze_traffic(),
            "errors": self.analyzer.analyze_errors(),
            "saturation": self.analyzer.analyze_saturation()
        }
        
        alerts = []
        for signal_name, signal_data in analysis.items():
            if signal_data.get("alert"):
                alerts.append(f"{signal_name.upper()}: {signal_data['alert']}")
        
        return web.json_response(alerts)

class TaskMasterGoldenSignalsIntegration:
    """Integration with Task Master system for automatic metrics collection"""
    
    def __init__(self, collector: GoldenSignalsCollector):
        self.collector = collector
        self.is_running = False
        self.background_thread = None
        
    def start_monitoring(self):
        """Start background monitoring of Task Master operations"""
        if self.is_running:
            return
            
        self.is_running = True
        self.background_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.background_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            try:
                # Collect saturation metrics
                self.collector.collect_saturation_metrics()
                
                # Simulate task master operations
                self._simulate_task_master_operations()
                
                time.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _simulate_task_master_operations(self):
        """Simulate Task Master operations for demonstration"""
        # Simulate task execution latency
        operation_id = self.collector.start_operation("task_execution")
        time.sleep(0.1)  # Simulate work
        self.collector.end_operation(operation_id, success=True)
        
        # Record traffic for task operations
        self.collector.record_traffic("task_master_api", 5.2, 3)
        
        # Occasionally record errors
        import random
        if random.random() < 0.05:  # 5% chance of error
            self.collector.record_error("task_execution_error", 100)

def main():
    """Main execution function"""
    print("🎯 Starting Golden Signals Monitoring Dashboard")
    print("=" * 60)
    
    # Initialize components
    collector = GoldenSignalsCollector()
    analyzer = GoldenSignalsAnalyzer(collector)
    dashboard = GoldenSignalsDashboard(collector, analyzer)
    integration = TaskMasterGoldenSignalsIntegration(collector)
    
    # Start monitoring
    integration.start_monitoring()
    
    # Start web server
    server_thread = dashboard.start_server(8080)
    
    print("🌐 Dashboard available at: http://localhost:8080")
    print("📊 Monitoring Four Golden Signals:")
    print("   • Latency - Request response times")
    print("   • Traffic - System demand and load")  
    print("   • Errors - Failure rates and types")
    print("   • Saturation - Resource utilization")
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Golden Signals Dashboard...")
        integration.stop_monitoring()
        dashboard.stop_server()

if __name__ == "__main__":
    main()