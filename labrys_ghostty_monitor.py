#!/usr/bin/env python3
"""
LABRYS Ghostty Monitor Spawner
Uses dual-blade methodology to spawn and manage monitoring windows
"""

import os
import sys
import json
import asyncio
import subprocess
from datetime import datetime
from typing import Dict, List, Any

# Add LABRYS paths
sys.path.append(os.path.join(os.path.dirname(__file__), '.labrys'))
from analytical.self_analysis_engine import SelfAnalysisEngine
from synthesis.self_synthesis_engine import SelfSynthesisEngine

class LabrysGhosttyMonitor:
    """
    LABRYS-powered Ghostty monitoring window manager
    """
    
    def __init__(self):
        self.labrys_root = os.path.join(os.path.dirname(__file__), '.labrys')
        self.analysis_engine = SelfAnalysisEngine(self.labrys_root)
        self.synthesis_engine = SelfSynthesisEngine(self.labrys_root)
        
        # Monitoring configuration
        self.monitoring_windows = {}
        self.ghostty_path = "/Applications/Ghostty.app/Contents/MacOS/ghostty"
        
    async def spawn_monitoring_window(self):
        """Use LABRYS to spawn optimized monitoring window"""
        print("🗲 LABRYS Ghostty Monitor Spawner")
        print("   Using dual-blade methodology for optimal monitoring")
        print("   " + "="*50)
        
        # ANALYTICAL BLADE: Analyze monitoring requirements
        print("\n⚡ Left Blade: Analyzing monitoring requirements...")
        monitor_analysis = await self._analyze_monitoring_needs()
        
        # SYNTHESIS BLADE: Generate optimal window configuration
        print("⚡ Right Blade: Synthesizing optimal window configuration...")
        window_config = await self._synthesize_window_config(monitor_analysis)
        
        # COORDINATION: Launch monitoring window
        print("🔄 Coordinating window launch...")
        window_process = await self._launch_ghostty_window(window_config)
        
        if window_process:
            print("✅ LABRYS monitoring window spawned successfully!")
            print(f"   Window PID: {window_process.pid}")
            print(f"   Configuration: {window_config['title']}")
            
            # Track the window
            self.monitoring_windows[window_process.pid] = {
                'process': window_process,
                'config': window_config,
                'spawn_time': datetime.now().isoformat()
            }
            
            return window_process
        else:
            print("❌ Failed to spawn monitoring window")
            return None
    
    async def _analyze_monitoring_needs(self) -> Dict[str, Any]:
        """Analytical blade: Analyze what monitoring is needed"""
        
        # Analyze current LABRYS processes
        current_processes = await self._get_current_labrys_processes()
        
        # Analyze system load and requirements
        analysis = {
            'active_processes': len(current_processes),
            'monitoring_priority': 'high',
            'refresh_interval': 5,  # seconds
            'window_size': 'large',
            'display_mode': 'comprehensive',
            'processes_to_monitor': current_processes,
            'monitoring_types': [
                'process_health',
                'resource_usage',
                'task_completion',
                'guardian_status',
                'system_metrics'
            ]
        }
        
        print(f"   📊 Analysis complete: {analysis['active_processes']} processes to monitor")
        return analysis
    
    async def _synthesize_window_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesis blade: Generate optimal window configuration"""
        
        # Generate window configuration based on analysis
        config = {
            'title': f"LABRYS Monitor - {analysis['active_processes']} Processes",
            'geometry': '120x40',  # columns x rows
            'position': '+100+100',  # x+y offset
            'font_size': '12',
            'background_color': '#1a1a1a',
            'foreground_color': '#00ff00',
            'scrollback_lines': '10000',
            'command': self._generate_monitoring_command(analysis),
            'window_class': 'LabrysMonitor',
            'working_directory': os.path.dirname(__file__)
        }
        
        print(f"   🔧 Synthesized configuration: {config['geometry']} window")
        return config
    
    def _generate_monitoring_command(self, analysis: Dict[str, Any]) -> str:
        """Generate the command to run in the monitoring window"""
        
        # Create a comprehensive monitoring script
        monitor_script = f"""
#!/bin/bash
cd "{os.path.dirname(__file__)}"
source venv/bin/activate

echo "🗲 LABRYS Process Monitor"
echo "   Real-time monitoring of {analysis['active_processes']} processes"
echo "   " + "="*50

while true; do
    clear
    echo "🗲 LABRYS Process Monitor - $(date)"
    echo "   " + "="*50
    
    # Health check
    python3 check_labrys_health.py
    
    echo ""
    echo "📊 Process Guardian Status:"
    if ps aux | grep -q "labrys_process_guardian"; then
        echo "   🛡️  Guardian: ACTIVE"
    else
        echo "   ⚠️  Guardian: NOT RUNNING"
    fi
    
    echo ""
    echo "📋 PID Scenarios:"
    if [ -d ".labrys/pid_scenarios" ]; then
        for scenario in .labrys/pid_scenarios/*/; do
            if [ -f "$scenario/current_status.json" ]; then
                pid=$(basename "$scenario" | sed 's/pid_//')
                health=$(python3 -c "import json; data=json.load(open('$scenario/current_status.json')); print(f'Health: {{data[\"health_score\"]:.1f}} - {{data[\"process_stats\"][\"status\"]}}')")
                echo "   PID $pid: $health"
            fi
        done
    fi
    
    echo ""
    echo "🔄 Refreshing in {analysis['refresh_interval']} seconds... (Ctrl+C to exit)"
    sleep {analysis['refresh_interval']}
done
"""
        
        # Write monitoring script
        script_path = os.path.join(os.path.dirname(__file__), 'labrys_monitor_script.sh')
        with open(script_path, 'w') as f:
            f.write(monitor_script)
        
        os.chmod(script_path, 0o755)
        
        return f"bash {script_path}"
    
    async def _get_current_labrys_processes(self) -> List[Dict[str, Any]]:
        """Get current LABRYS processes for monitoring"""
        import psutil
        
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                info = proc.info
                cmdline = info.get('cmdline', [])
                if cmdline and self._is_labrys_process(cmdline):
                    processes.append({
                        'pid': info['pid'],
                        'name': info['name'],
                        'cmdline': ' '.join(cmdline)
                    })
            except:
                continue
        
        return processes
    
    def _is_labrys_process(self, cmdline: List[str]) -> bool:
        """Check if process is LABRYS-related"""
        cmdline_str = ' '.join(cmdline).lower()
        indicators = ['labrys', 'recursive', 'guardian', 'self_test']
        return any(indicator in cmdline_str for indicator in indicators)
    
    async def _launch_ghostty_window(self, config: Dict[str, Any]) -> subprocess.Popen:
        """Launch Ghostty window with configuration"""
        
        try:
            # Build Ghostty command
            ghostty_cmd = [
                self.ghostty_path,
                '--title', config['title'],
                '--geometry', config['geometry'],
                '--working-directory', config['working_directory'],
                '--command', config['command']
            ]
            
            # Launch Ghostty
            process = subprocess.Popen(
                ghostty_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=config['working_directory']
            )
            
            # Give it a moment to start
            await asyncio.sleep(1)
            
            return process
            
        except Exception as e:
            print(f"❌ Error launching Ghostty: {e}")
            return None
    
    async def spawn_multiple_windows(self, window_configs: List[str] = None):
        """Spawn multiple monitoring windows for different aspects"""
        
        if not window_configs:
            window_configs = [
                'health_monitor',
                'process_guardian',
                'task_completion',
                'system_metrics'
            ]
        
        spawned_windows = []
        
        for config_type in window_configs:
            print(f"\n🪟 Spawning {config_type} window...")
            
            # Customize analysis for specific window type
            analysis = await self._analyze_monitoring_needs()
            analysis['window_type'] = config_type
            analysis['title_suffix'] = config_type.replace('_', ' ').title()
            
            # Generate specialized configuration
            window_config = await self._synthesize_specialized_config(analysis)
            
            # Launch window
            window_process = await self._launch_ghostty_window(window_config)
            
            if window_process:
                spawned_windows.append(window_process)
                print(f"   ✅ {config_type} window spawned (PID: {window_process.pid})")
            else:
                print(f"   ❌ Failed to spawn {config_type} window")
        
        return spawned_windows
    
    async def _synthesize_specialized_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specialized window configuration"""
        
        window_type = analysis.get('window_type', 'general')
        base_config = await self._synthesize_window_config(analysis)
        
        # Customize based on window type
        if window_type == 'health_monitor':
            base_config.update({
                'title': f"LABRYS Health Monitor - {analysis['active_processes']} Processes",
                'geometry': '100x30',
                'position': '+50+50',
                'command': self._generate_health_monitor_command()
            })
        
        elif window_type == 'process_guardian':
            base_config.update({
                'title': 'LABRYS Process Guardian',
                'geometry': '80x25',
                'position': '+400+50',
                'command': self._generate_guardian_monitor_command()
            })
        
        elif window_type == 'task_completion':
            base_config.update({
                'title': 'LABRYS Task Completion Tracker',
                'geometry': '90x20',
                'position': '+50+400',
                'command': self._generate_task_monitor_command()
            })
        
        elif window_type == 'system_metrics':
            base_config.update({
                'title': 'LABRYS System Metrics',
                'geometry': '70x15',
                'position': '+400+400',
                'command': self._generate_metrics_monitor_command()
            })
        
        return base_config
    
    def _generate_health_monitor_command(self) -> str:
        """Generate health monitoring command"""
        script_content = f"""
#!/bin/bash
cd "{os.path.dirname(__file__)}"
source venv/bin/activate

while true; do
    clear
    echo "🗲 LABRYS Health Monitor - $(date)"
    echo "   " + "="*50
    python3 check_labrys_health.py
    echo ""
    echo "⏸️  Next update in 10 seconds..."
    sleep 10
done
"""
        script_path = os.path.join(os.path.dirname(__file__), 'health_monitor_script.sh')
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        return f"bash {script_path}"
    
    def _generate_guardian_monitor_command(self) -> str:
        """Generate guardian monitoring command"""
        script_content = f"""
#!/bin/bash
cd "{os.path.dirname(__file__)}"

while true; do
    clear
    echo "🛡️  LABRYS Process Guardian Monitor - $(date)"
    echo "   " + "="*50
    
    if ps aux | grep -q "labrys_process_guardian" | grep -v grep; then
        echo "✅ Guardian Status: ACTIVE"
        echo ""
        echo "📊 Guardian Activity:"
        if [ -f "labrys_guardian_report.json" ]; then
            echo "   Last report: $(stat -f %Sm labrys_guardian_report.json)"
        fi
        
        echo ""
        echo "🔧 Recent Maintenance:"
        # Show last few lines of any maintenance logs
        if [ -f ".labrys/maintenance.log" ]; then
            tail -5 .labrys/maintenance.log
        else
            echo "   No maintenance actions logged"
        fi
    else
        echo "❌ Guardian Status: NOT RUNNING"
        echo "   Consider starting guardian process"
    fi
    
    echo ""
    echo "⏸️  Next update in 15 seconds..."
    sleep 15
done
"""
        script_path = os.path.join(os.path.dirname(__file__), 'guardian_monitor_script.sh')
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        return f"bash {script_path}"
    
    def _generate_task_monitor_command(self) -> str:
        """Generate task completion monitoring command"""
        script_content = f"""
#!/bin/bash
cd "{os.path.dirname(__file__)}"

while true; do
    clear
    echo "📋 LABRYS Task Completion Monitor - $(date)"
    echo "   " + "="*50
    
    echo "🎯 Active Scenarios:"
    if [ -d ".labrys/pid_scenarios" ]; then
        for scenario in .labrys/pid_scenarios/*/; do
            if [ -f "$scenario/current_status.json" ]; then
                pid=$(basename "$scenario" | sed 's/pid_//')
                echo "   PID $pid: Monitoring active"
            fi
        done
    else
        echo "   No active scenarios"
    fi
    
    echo ""
    echo "🏁 Completed Tasks:"
    if [ -f "labrys_self_test_results.json" ]; then
        echo "   Self-test: COMPLETED"
    fi
    if [ -f "recursive_improvement_results.json" ]; then
        echo "   Recursive improvement: COMPLETED"
    fi
    
    echo ""
    echo "⏸️  Next update in 20 seconds..."
    sleep 20
done
"""
        script_path = os.path.join(os.path.dirname(__file__), 'task_monitor_script.sh')
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        return f"bash {script_path}"
    
    def _generate_metrics_monitor_command(self) -> str:
        """Generate metrics monitoring command"""
        script_content = f"""
#!/bin/bash
cd "{os.path.dirname(__file__)}"

while true; do
    clear
    echo "📊 LABRYS System Metrics - $(date)"
    echo "   " + "="*50
    
    echo "💾 System Resources:"
    echo "   CPU: $(top -l 1 | grep "CPU usage" | awk '{{print $3}}' | sed 's/%//')"
    echo "   Memory: $(memory_pressure | grep "System-wide memory free percentage" | awk '{{print $5}}')"
    
    echo ""
    echo "🗲 LABRYS Processes:"
    ps aux | grep -E "(labrys|recursive)" | grep -v grep | wc -l | xargs echo "   Active processes:"
    
    echo ""
    echo "📈 Performance:"
    if [ -f "labrys_self_test_results.json" ]; then
        echo "   Test execution time: $(grep -o '"total_execution_time": [0-9.]*' labrys_self_test_results.json | cut -d' ' -f2)s"
    fi
    
    echo ""
    echo "⏸️  Next update in 30 seconds..."
    sleep 30
done
"""
        script_path = os.path.join(os.path.dirname(__file__), 'metrics_monitor_script.sh')
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        return f"bash {script_path}"

async def main():
    """Main function to spawn monitoring windows"""
    monitor = LabrysGhosttyMonitor()
    
    print("🗲 LABRYS Ghostty Monitor Spawner")
    print("   Choose monitoring configuration:")
    print("   1. Single comprehensive window")
    print("   2. Multiple specialized windows")
    print("   3. Health monitor only")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        await monitor.spawn_monitoring_window()
    elif choice == '2':
        await monitor.spawn_multiple_windows()
    elif choice == '3':
        await monitor.spawn_multiple_windows(['health_monitor'])
    else:
        print("Invalid choice, spawning comprehensive window...")
        await monitor.spawn_monitoring_window()

if __name__ == "__main__":
    asyncio.run(main())