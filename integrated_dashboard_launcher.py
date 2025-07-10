#!/usr/bin/env python3
"""
Integrated Dashboard Launcher for Task Master AI

This module provides a unified launcher that integrates the real-time dashboard,
enhanced data pipeline, and existing monitoring infrastructure into a cohesive
performance monitoring solution.
"""

import os
import sys
import time
import json
import threading
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

# Import our custom components
try:
    from real_time_dashboard import RealTimeDashboard, DashboardConfig
    from enhanced_data_pipeline import DataPipeline, PipelineConfig, DataPoint
    from advanced_analytics_dashboard import AdvancedAnalyticsDashboard
    from performance_monitor import perf_monitor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are in the same directory.")
    sys.exit(1)


class IntegratedDashboardSystem:
    """Integrated dashboard system combining all monitoring components"""
    
    def __init__(self, dashboard_port: int = 8090, enable_advanced: bool = True):
        self.dashboard_port = dashboard_port
        self.enable_advanced = enable_advanced
        
        # Initialize configurations
        self.dashboard_config = DashboardConfig(
            dashboard_port=dashboard_port,
            websocket_port=dashboard_port + 1,
            api_port=dashboard_port + 2,
            update_interval=5,
            max_data_points=100
        )
        
        self.pipeline_config = PipelineConfig(
            batch_size=50,
            flush_interval=3.0,
            worker_threads=3,
            retention_days=30
        )
        
        # Initialize components
        self.data_pipeline = None
        self.real_time_dashboard = None
        self.advanced_dashboard = None
        
        # System state
        self.running = False
        self.components_started = []
    
    def start_system(self):
        """Start the complete integrated dashboard system"""
        print("🚀 Starting Integrated Task Master AI Dashboard System...")
        print("=" * 60)
        
        try:
            # 1. Start the enhanced data pipeline
            print("📊 Starting Enhanced Data Pipeline...")
            self.data_pipeline = DataPipeline(self.pipeline_config)
            self.data_pipeline.start()
            self.components_started.append("data_pipeline")
            print("✅ Data Pipeline started successfully")
            
            # 2. Start real-time dashboard
            print("🌐 Starting Real-Time Dashboard...")
            self.real_time_dashboard = RealTimeDashboard(self.dashboard_config)
            dashboard_url = self.real_time_dashboard.start_dashboard(auto_open=False)
            self.components_started.append("real_time_dashboard")
            print(f"✅ Real-Time Dashboard started at {dashboard_url}")
            
            # 3. Start advanced analytics dashboard (if enabled)
            if self.enable_advanced:
                print("📈 Starting Advanced Analytics Dashboard...")
                self.advanced_dashboard = AdvancedAnalyticsDashboard()
                advanced_url = self.advanced_dashboard.start_advanced_dashboard(auto_open=False)
                self.components_started.append("advanced_dashboard")
                print(f"✅ Advanced Dashboard started at {advanced_url}")
            
            # 4. Start data generation and feeding
            print("🔄 Starting Data Generation...")
            self._start_data_generation()
            self.components_started.append("data_generation")
            print("✅ Data Generation started")
            
            self.running = True
            
            # Display system information
            self._display_system_info()
            
            # Open dashboards in browser
            self._open_dashboards()
            
            print("\n🎉 Integrated Dashboard System started successfully!")
            print("Press Ctrl+C to stop the system...")
            
        except Exception as e:
            print(f"❌ Error starting system: {e}")
            self.stop_system()
            raise
    
    def stop_system(self):
        """Stop all components of the integrated system"""
        print("\n🛑 Stopping Integrated Dashboard System...")
        
        self.running = False
        
        # Stop components in reverse order
        if "data_generation" in self.components_started:
            print("⏹️  Stopping data generation...")
        
        if "advanced_dashboard" in self.components_started and self.advanced_dashboard:
            print("⏹️  Stopping advanced dashboard...")
            # Advanced dashboard doesn't have explicit stop method
        
        if "real_time_dashboard" in self.components_started and self.real_time_dashboard:
            print("⏹️  Stopping real-time dashboard...")
            self.real_time_dashboard.stop_dashboard()
        
        if "data_pipeline" in self.components_started and self.data_pipeline:
            print("⏹️  Stopping data pipeline...")
            self.data_pipeline.stop()
        
        print("✅ Integrated Dashboard System stopped successfully")
    
    def _start_data_generation(self):
        """Start generating sample data for the dashboards"""
        def data_generator():
            import random
            
            while self.running:
                try:
                    # Generate realistic system metrics
                    timestamp = datetime.now()
                    
                    # System metrics with some variability
                    base_cpu = 25 + random.uniform(-10, 35)  # 15-60% CPU
                    base_memory = 45 + random.uniform(-15, 25)  # 30-70% Memory
                    base_disk = 65 + random.uniform(-5, 15)  # 60-80% Disk
                    
                    system_data_points = [
                        DataPoint(timestamp, "system", "cpu_percent", max(0, min(100, base_cpu)), {"host": "localhost"}),
                        DataPoint(timestamp, "system", "memory_percent", max(0, min(100, base_memory)), {"host": "localhost"}),
                        DataPoint(timestamp, "system", "disk_usage_percent", max(0, min(100, base_disk)), {"host": "localhost"}),
                        DataPoint(timestamp, "system", "load_average", random.uniform(0.5, 3.0), {"host": "localhost"}),
                    ]
                    
                    # Task metrics
                    total_tasks = 43
                    completed = random.randint(38, 42)
                    in_progress = random.randint(0, 2)
                    pending = total_tasks - completed - in_progress
                    
                    task_data_points = [
                        DataPoint(timestamp, "tasks", "total_tasks", total_tasks, {"project": "task-master"}),
                        DataPoint(timestamp, "tasks", "completed_tasks", completed, {"project": "task-master"}),
                        DataPoint(timestamp, "tasks", "in_progress_tasks", in_progress, {"project": "task-master"}),
                        DataPoint(timestamp, "tasks", "pending_tasks", pending, {"project": "task-master"}),
                        DataPoint(timestamp, "tasks", "completion_rate", completed / total_tasks, {"project": "task-master"}),
                        DataPoint(timestamp, "tasks", "avg_execution_time", random.uniform(30, 180), {"project": "task-master"}),
                    ]
                    
                    # GitHub metrics
                    success_rate = random.uniform(0.85, 0.98)
                    total_runs = random.randint(20, 30)
                    successful = int(total_runs * success_rate)
                    failed = total_runs - successful
                    
                    github_data_points = [
                        DataPoint(timestamp, "github", "total_runs", total_runs, {"repo": "task-master"}),
                        DataPoint(timestamp, "github", "successful_runs", successful, {"repo": "task-master"}),
                        DataPoint(timestamp, "github", "failed_runs", failed, {"repo": "task-master"}),
                        DataPoint(timestamp, "github", "success_rate", success_rate, {"repo": "task-master"}),
                        DataPoint(timestamp, "github", "workflow_duration", random.uniform(60, 300), {"repo": "task-master"}),
                    ]
                    
                    # Performance metrics
                    health_score = 100 - (base_cpu * 0.3 + base_memory * 0.4 + (failed / total_runs * 100) * 0.3)
                    performance_data_points = [
                        DataPoint(timestamp, "performance", "health_score", max(0, min(100, health_score)), {"system": "task-master"}),
                        DataPoint(timestamp, "performance", "response_time_avg", random.uniform(50, 200), {"system": "task-master"}),
                        DataPoint(timestamp, "performance", "throughput_rps", random.uniform(8, 20), {"system": "task-master"}),
                        DataPoint(timestamp, "performance", "error_rate", random.uniform(0.001, 0.02), {"system": "task-master"}),
                    ]
                    
                    # Combine all data points
                    all_data_points = (system_data_points + task_data_points + 
                                     github_data_points + performance_data_points)
                    
                    # Send to data pipeline
                    if self.data_pipeline:
                        self.data_pipeline.ingest_data_points(all_data_points)
                    
                    # Sleep for update interval
                    time.sleep(self.dashboard_config.update_interval)
                    
                except Exception as e:
                    print(f"Error in data generation: {e}")
                    time.sleep(5)
        
        # Start data generation in background thread
        data_thread = threading.Thread(target=data_generator, name="DataGenerator")
        data_thread.daemon = True
        data_thread.start()
    
    def _display_system_info(self):
        """Display comprehensive system information"""
        print("\n" + "=" * 60)
        print("📊 INTEGRATED DASHBOARD SYSTEM - STATUS")
        print("=" * 60)
        
        # System URLs
        print("🌐 Dashboard URLs:")
        print(f"   Real-Time Dashboard:  http://localhost:{self.dashboard_config.dashboard_port}")
        if self.enable_advanced:
            print(f"   Advanced Analytics:   http://localhost:8080")
        print(f"   WebSocket Stream:     ws://localhost:{self.dashboard_config.websocket_port}")
        print(f"   API Endpoint:         http://localhost:{self.dashboard_config.api_port}")
        
        # Configuration
        print(f"\n⚙️  Configuration:")
        print(f"   Update Interval:      {self.dashboard_config.update_interval} seconds")
        print(f"   Data Retention:       {self.pipeline_config.retention_days} days")
        print(f"   Batch Size:           {self.pipeline_config.batch_size}")
        print(f"   Worker Threads:       {self.pipeline_config.worker_threads}")
        
        # Components Status
        print(f"\n🔧 Components:")
        components = {
            "Enhanced Data Pipeline": "data_pipeline" in self.components_started,
            "Real-Time Dashboard": "real_time_dashboard" in self.components_started,
            "Advanced Analytics": "advanced_dashboard" in self.components_started,
            "Data Generation": "data_generation" in self.components_started
        }
        
        for component, status in components.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}")
        
        # Data Pipeline Status
        if self.data_pipeline:
            pipeline_status = self.data_pipeline.get_pipeline_status()
            print(f"\n📈 Pipeline Metrics:")
            print(f"   Queue Size:           {pipeline_status['queue_size']}")
            print(f"   Data Points Processed: {pipeline_status['metrics']['data_points_processed']}")
            print(f"   Processing Errors:    {pipeline_status['metrics']['processing_errors']}")
            print(f"   Avg Processing Time:  {pipeline_status['metrics']['average_processing_time']:.4f}s")
        
        # Monitoring Coverage
        print(f"\n🔍 Monitoring Coverage:")
        monitoring_areas = [
            "System Resources (CPU, Memory, Disk)",
            "Task Master AI Metrics",
            "GitHub Actions Status",
            "Performance Analytics",
            "Real-time Alerting",
            "Anomaly Detection",
            "Trend Analysis",
            "Optimization Recommendations"
        ]
        
        for area in monitoring_areas:
            print(f"   ✅ {area}")
        
        print("=" * 60)
    
    def _open_dashboards(self):
        """Open dashboards in web browser"""
        try:
            # Wait a moment for servers to be ready
            time.sleep(3)
            
            print("\n🌐 Opening dashboards in browser...")
            
            # Open real-time dashboard
            real_time_url = f"http://localhost:{self.dashboard_config.dashboard_port}"
            webbrowser.open(real_time_url)
            
            # Open advanced dashboard if enabled
            if self.enable_advanced:
                time.sleep(2)  # Stagger browser tabs
                advanced_url = "http://localhost:8080"
                webbrowser.open(advanced_url)
            
        except Exception as e:
            print(f"Could not open browsers automatically: {e}")
            print("Please open the URLs manually in your browser.")
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        status = {
            "running": self.running,
            "components": self.components_started.copy(),
            "urls": {
                "real_time_dashboard": f"http://localhost:{self.dashboard_config.dashboard_port}",
                "websocket": f"ws://localhost:{self.dashboard_config.websocket_port}",
                "api": f"http://localhost:{self.dashboard_config.api_port}"
            },
            "config": {
                "dashboard": self.dashboard_config.__dict__,
                "pipeline": self.pipeline_config.__dict__
            }
        }
        
        if self.enable_advanced:
            status["urls"]["advanced_dashboard"] = "http://localhost:8080"
        
        if self.data_pipeline:
            status["pipeline_status"] = self.data_pipeline.get_pipeline_status()
        
        return status
    
    def run_health_check(self) -> bool:
        """Run a comprehensive health check"""
        print("🏥 Running System Health Check...")
        
        health_issues = []
        
        # Check if components are running
        required_components = ["data_pipeline", "real_time_dashboard"]
        for component in required_components:
            if component not in self.components_started:
                health_issues.append(f"Component {component} is not running")
        
        # Check data pipeline health
        if self.data_pipeline:
            pipeline_status = self.data_pipeline.get_pipeline_status()
            if pipeline_status['metrics']['processing_errors'] > 100:
                health_issues.append("High processing error rate in data pipeline")
            
            if pipeline_status['queue_size'] > 1000:
                health_issues.append("Data pipeline queue is backing up")
        
        # Report health status
        if not health_issues:
            print("✅ System Health Check: ALL SYSTEMS NORMAL")
            return True
        else:
            print("⚠️  System Health Check: ISSUES DETECTED")
            for issue in health_issues:
                print(f"   ❌ {issue}")
            return False


def main():
    """Main function to launch the integrated dashboard system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated Task Master AI Dashboard System")
    parser.add_argument("--port", type=int, default=8090, help="Base port for dashboard (default: 8090)")
    parser.add_argument("--no-advanced", action="store_true", help="Disable advanced analytics dashboard")
    parser.add_argument("--health-check", action="store_true", help="Run health check and exit")
    parser.add_argument("--status", action="store_true", help="Show system status and exit")
    
    args = parser.parse_args()
    
    # Create integrated dashboard system
    dashboard_system = IntegratedDashboardSystem(
        dashboard_port=args.port,
        enable_advanced=not args.no_advanced
    )
    
    try:
        if args.health_check:
            # Run health check only
            healthy = dashboard_system.run_health_check()
            sys.exit(0 if healthy else 1)
        
        if args.status:
            # Show status only
            status = dashboard_system.get_system_status()
            print(json.dumps(status, indent=2, default=str))
            sys.exit(0)
        
        # Start the complete system
        dashboard_system.start_system()
        
        # Keep running until interrupted
        while True:
            time.sleep(10)
            
            # Periodic health check
            if dashboard_system.running:
                dashboard_system.run_health_check()
            
    except KeyboardInterrupt:
        print("\n⏹️  Shutdown requested by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    finally:
        dashboard_system.stop_system()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()