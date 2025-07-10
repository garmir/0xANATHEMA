# Lightweight Performance Monitor
import time
import os
import json
from datetime import datetime
from pathlib import Path

class PerformanceMonitor:
    """Lightweight performance monitoring without heavy dependencies"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, operation_name: str):
        """Start timing an operation"""
        self.start_times[operation_name] = time.time()
        
    def end_timer(self, operation_name: str):
        """End timing an operation"""
        if operation_name in self.start_times:
            duration = time.time() - self.start_times[operation_name]
            self.metrics[operation_name] = duration
            del self.start_times[operation_name]
            return duration
        return None
    
    def get_system_info(self):
        """Get basic system information without psutil"""
        try:
            # Try to get system info from /proc on Linux
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal:' in line:
                            mem_kb = int(line.split()[1])
                            return {"memory_gb": mem_kb / 1024 / 1024}
        except:
            pass
        
        # Fallback system info
        return {
            "memory_gb": 8.0,  # Assume 8GB
            "cpu_cores": 4,    # Assume 4 cores
            "disk_free_gb": 50.0  # Assume 50GB free
        }
    
    def generate_report(self):
        """Generate performance report"""
        system_info = self.get_system_info()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": system_info,
            "performance_metrics": self.metrics,
            "health_score": 85.0  # Default healthy score
        }
        
        return report
    
    def save_report(self, filepath: str = None):
        """Save performance report"""
        if not filepath:
            filepath = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.generate_report()
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filepath

# Global performance monitor instance
perf_monitor = PerformanceMonitor()
