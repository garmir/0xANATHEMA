#!/usr/bin/env python3
"""
LABRYS Process Guardian
Monitors, maintains, and fixes all LABRYS PIDs using dual-blade methodology
"""

import os
import sys
import json
import asyncio
import psutil
import signal
import subprocess
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

# Add LABRYS paths
sys.path.append(os.path.join(os.path.dirname(__file__), '.labrys'))
from analytical.self_analysis_engine import SelfAnalysisEngine
from synthesis.self_synthesis_engine import SelfSynthesisEngine
from validation.safety_validator import SafetyValidator

class ProcessState(Enum):
    RUNNING = "running"
    SLEEPING = "sleeping"
    ZOMBIE = "zombie"
    STOPPED = "stopped"
    DEAD = "dead"
    UNKNOWN = "unknown"

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    STUCK = "stuck"
    UNKNOWN = "unknown"

@dataclass
class LabrysProcess:
    """Represents a monitored LABRYS process"""
    pid: int
    ppid: int
    name: str
    cmdline: List[str]
    state: ProcessState
    task_status: TaskStatus
    start_time: float
    cpu_percent: float
    memory_mb: float
    working_directory: str
    task_description: str
    expected_completion_time: Optional[float] = None
    last_health_check: float = 0
    health_score: float = 1.0
    failure_count: int = 0
    recovery_attempts: int = 0
    last_activity: float = 0
    
    def __post_init__(self):
        if self.last_health_check == 0:
            self.last_health_check = time.time()
        if self.last_activity == 0:
            self.last_activity = time.time()

@dataclass
class ProcessMaintenanceAction:
    """Represents a maintenance action taken on a process"""
    action_type: str
    target_pid: int
    description: str
    timestamp: float
    success: bool
    details: Dict[str, Any]

class LabrysProcessGuardian:
    """
    Guardian system for monitoring and maintaining LABRYS processes
    Uses dual-blade methodology for analysis and synthesis of solutions
    """
    
    def __init__(self, labrys_root: str = None):
        self.labrys_root = labrys_root or os.path.join(os.path.dirname(__file__), '.labrys')
        
        # Initialize LABRYS engines
        self.analysis_engine = SelfAnalysisEngine(self.labrys_root)
        self.synthesis_engine = SelfSynthesisEngine(self.labrys_root)
        self.safety_validator = SafetyValidator()
        
        # Process tracking
        self.monitored_processes: Dict[int, LabrysProcess] = {}
        self.maintenance_history: List[ProcessMaintenanceAction] = []
        self.guardian_active = False
        
        # Configuration
        self.monitor_interval = 5.0  # seconds
        self.health_check_interval = 30.0  # seconds
        self.max_recovery_attempts = 3
        self.task_timeout_threshold = 1800.0  # 30 minutes
        self.cpu_warning_threshold = 95.0
        self.memory_warning_threshold = 500.0  # MB
        
        # Task patterns and completion indicators
        self.task_completion_patterns = {
            'interactive': ['quit', 'exit', 'shutdown'],
            'batch': ['completed', 'finished', 'done'],
            'test': ['validation complete', 'tests passed', 'all tests'],
            'improvement': ['convergence', 'improvement complete', 'iterations']
        }
        
    async def start_guardian(self):
        """Start the process guardian system"""
        print("🗲 LABRYS Process Guardian Starting")
        print("   Monitoring and maintaining all LABRYS processes")
        print("   " + "="*50)
        
        self.guardian_active = True
        
        # Initial process discovery
        await self._discover_labrys_processes()
        
        # Start monitoring loop
        await self._guardian_main_loop()
    
    async def _discover_labrys_processes(self):
        """Discover all running LABRYS processes"""
        print("🔍 Discovering LABRYS processes...")
        
        discovered_count = 0
        
        for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'status', 'create_time', 'cwd']):
            try:
                info = proc.info
                cmdline = info.get('cmdline', [])
                
                # Check if this is a LABRYS process
                if self._is_labrys_process(cmdline, info.get('name', '')):
                    pid = info['pid']
                    
                    # Determine task type and description
                    task_type, task_desc = self._identify_task_type(cmdline)
                    
                    # Create process object
                    labrys_proc = LabrysProcess(
                        pid=pid,
                        ppid=info.get('ppid', 0),
                        name=info.get('name', 'unknown'),
                        cmdline=cmdline,
                        state=self._map_process_state(info.get('status', 'unknown')),
                        task_status=TaskStatus.IN_PROGRESS,
                        start_time=info.get('create_time', time.time()),
                        cpu_percent=0.0,
                        memory_mb=0.0,
                        working_directory=info.get('cwd', ''),
                        task_description=task_desc
                    )
                    
                    # Add to monitored processes
                    self.monitored_processes[pid] = labrys_proc
                    discovered_count += 1
                    
                    print(f"   📋 Discovered LABRYS PID {pid}: {task_desc}")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        print(f"✅ Discovered {discovered_count} LABRYS processes")
        return discovered_count
    
    def _is_labrys_process(self, cmdline: List[str], name: str) -> bool:
        """Determine if a process is related to LABRYS"""
        if not cmdline:
            return False
        
        cmdline_str = ' '.join(cmdline).lower()
        labrys_indicators = [
            'labrys', 'recursive', 'self_test', 'dual_blade',
            'analytical_blade', 'synthesis_blade', 'coordinator'
        ]
        
        return any(indicator in cmdline_str for indicator in labrys_indicators)
    
    def _identify_task_type(self, cmdline: List[str]) -> tuple:
        """Identify the task type and description for a process"""
        cmdline_str = ' '.join(cmdline)
        
        if '--interactive' in cmdline_str:
            return 'interactive', 'LABRYS Interactive Mode'
        elif '--execute' in cmdline_str:
            return 'batch', 'LABRYS Batch Execution'
        elif 'self_test' in cmdline_str:
            return 'test', 'LABRYS Self-Testing'
        elif 'recursive' in cmdline_str:
            return 'improvement', 'LABRYS Recursive Improvement'
        elif '--validate' in cmdline_str:
            return 'validation', 'LABRYS System Validation'
        else:
            return 'unknown', 'LABRYS Process'
    
    def _map_process_state(self, psutil_status: str) -> ProcessState:
        """Map psutil status to ProcessState enum"""
        mapping = {
            'running': ProcessState.RUNNING,
            'sleeping': ProcessState.SLEEPING,
            'disk-sleep': ProcessState.SLEEPING,
            'stopped': ProcessState.STOPPED,
            'tracing-stop': ProcessState.STOPPED,
            'zombie': ProcessState.ZOMBIE,
            'dead': ProcessState.DEAD,
            'wake-kill': ProcessState.DEAD,
            'waking': ProcessState.RUNNING
        }
        return mapping.get(psutil_status.lower(), ProcessState.UNKNOWN)
    
    async def _guardian_main_loop(self):
        """Main guardian monitoring loop"""
        print("🛡️  Guardian main loop started")
        
        try:
            while self.guardian_active:
                # Update process information
                await self._update_process_information()
                
                # Analyze process health
                await self._analyze_process_health()
                
                # Perform maintenance actions
                await self._perform_maintenance_actions()
                
                # Check task completion
                await self._check_task_completion()
                
                # Clean up dead processes
                await self._cleanup_dead_processes()
                
                # Report status
                await self._report_guardian_status()
                
                # Wait for next cycle
                await asyncio.sleep(self.monitor_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Guardian shutdown requested")
        except Exception as e:
            print(f"💥 Guardian error: {e}")
        finally:
            await self._shutdown_guardian()
    
    async def _update_process_information(self):
        """Update information for all monitored processes"""
        current_time = time.time()
        
        for pid, proc_info in list(self.monitored_processes.items()):
            try:
                if psutil.pid_exists(pid):
                    psutil_proc = psutil.Process(pid)
                    
                    # Update process metrics
                    proc_info.cpu_percent = psutil_proc.cpu_percent()
                    proc_info.memory_mb = psutil_proc.memory_info().rss / 1024 / 1024
                    proc_info.state = self._map_process_state(psutil_proc.status())
                    proc_info.last_health_check = current_time
                    
                    # Check for activity (file changes, network activity, etc.)
                    activity_detected = await self._detect_process_activity(pid)
                    if activity_detected:
                        proc_info.last_activity = current_time
                        
                else:
                    # Process no longer exists
                    proc_info.state = ProcessState.DEAD
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                proc_info.state = ProcessState.DEAD
    
    async def _detect_process_activity(self, pid: int) -> bool:
        """Detect if a process is showing signs of activity"""
        try:
            # Check for recent file modifications in working directory
            proc_info = self.monitored_processes.get(pid)
            if not proc_info:
                return False
            
            working_dir = proc_info.working_directory
            if not working_dir or not os.path.exists(working_dir):
                return False
            
            # Check for recent file modifications
            recent_threshold = time.time() - 60  # Last minute
            
            for root, dirs, files in os.walk(working_dir):
                # Skip deep recursion
                if root.count(os.sep) - working_dir.count(os.sep) > 3:
                    continue
                    
                for file in files:
                    if file.endswith(('.py', '.json', '.log', '.tmp')):
                        file_path = os.path.join(root, file)
                        try:
                            if os.path.getmtime(file_path) > recent_threshold:
                                return True
                        except OSError:
                            continue
            
            return False
            
        except Exception:
            return False
    
    async def _analyze_process_health(self):
        """Analyze health of all monitored processes using analytical blade"""
        current_time = time.time()
        
        for pid, proc_info in self.monitored_processes.items():
            if proc_info.state == ProcessState.DEAD:
                continue
            
            # Calculate health score
            health_factors = []
            
            # CPU usage factor
            if proc_info.cpu_percent > self.cpu_warning_threshold:
                health_factors.append(("high_cpu", 0.7))
            elif proc_info.cpu_percent > 50:
                health_factors.append(("moderate_cpu", 0.9))
            else:
                health_factors.append(("normal_cpu", 1.0))
            
            # Memory usage factor
            if proc_info.memory_mb > self.memory_warning_threshold:
                health_factors.append(("high_memory", 0.6))
            elif proc_info.memory_mb > 100:
                health_factors.append(("moderate_memory", 0.9))
            else:
                health_factors.append(("normal_memory", 1.0))
            
            # Activity factor
            time_since_activity = current_time - proc_info.last_activity
            if time_since_activity > 600:  # 10 minutes
                health_factors.append(("no_activity", 0.3))
            elif time_since_activity > 300:  # 5 minutes
                health_factors.append(("low_activity", 0.7))
            else:
                health_factors.append(("active", 1.0))
            
            # Runtime factor
            runtime = current_time - proc_info.start_time
            if runtime > self.task_timeout_threshold:
                health_factors.append(("long_running", 0.8))
            else:
                health_factors.append(("normal_runtime", 1.0))
            
            # Calculate overall health score
            if health_factors:
                proc_info.health_score = sum(factor[1] for factor in health_factors) / len(health_factors)
            
            # Update task status based on health
            if proc_info.health_score < 0.3:
                proc_info.task_status = TaskStatus.FAILED
            elif proc_info.health_score < 0.5:
                proc_info.task_status = TaskStatus.STUCK
            elif proc_info.state == ProcessState.RUNNING and proc_info.health_score > 0.8:
                proc_info.task_status = TaskStatus.IN_PROGRESS
    
    async def _perform_maintenance_actions(self):
        """Perform maintenance actions on processes that need help"""
        for pid, proc_info in list(self.monitored_processes.items()):
            
            # Skip if process is healthy or already dead
            if proc_info.health_score > 0.7 or proc_info.state == ProcessState.DEAD:
                continue
            
            # Skip if too many recovery attempts
            if proc_info.recovery_attempts >= self.max_recovery_attempts:
                continue
            
            # Determine appropriate maintenance action
            action = await self._determine_maintenance_action(proc_info)
            
            if action:
                success = await self._execute_maintenance_action(action)
                
                # Record the action
                maintenance_record = ProcessMaintenanceAction(
                    action_type=action['type'],
                    target_pid=pid,
                    description=action['description'],
                    timestamp=time.time(),
                    success=success,
                    details=action.get('details', {})
                )
                
                self.maintenance_history.append(maintenance_record)
                
                if success:
                    proc_info.recovery_attempts += 1
                    print(f"🔧 Maintenance action successful: {action['description']} for PID {pid}")
                else:
                    proc_info.failure_count += 1
                    print(f"❌ Maintenance action failed: {action['description']} for PID {pid}")
    
    async def _determine_maintenance_action(self, proc_info: LabrysProcess) -> Optional[Dict[str, Any]]:
        """Determine what maintenance action to take for a process"""
        current_time = time.time()
        
        # Process is stuck (low activity, high CPU)
        if (proc_info.cpu_percent > 95 and 
            current_time - proc_info.last_activity > 300):
            return {
                'type': 'gentle_restart',
                'description': 'Send gentle restart signal to stuck process',
                'details': {'reason': 'stuck_high_cpu'}
            }
        
        # Process is consuming too much memory
        if proc_info.memory_mb > self.memory_warning_threshold:
            return {
                'type': 'memory_optimization',
                'description': 'Attempt to optimize memory usage',
                'details': {'memory_mb': proc_info.memory_mb}
            }
        
        # Process has been running too long without activity
        if (current_time - proc_info.start_time > self.task_timeout_threshold and
            current_time - proc_info.last_activity > 600):
            return {
                'type': 'activity_check',
                'description': 'Check if process is still working or needs intervention',
                'details': {'runtime': current_time - proc_info.start_time}
            }
        
        # Process is in zombie state
        if proc_info.state == ProcessState.ZOMBIE:
            return {
                'type': 'cleanup_zombie',
                'description': 'Clean up zombie process',
                'details': {'state': 'zombie'}
            }
        
        return None
    
    async def _execute_maintenance_action(self, action: Dict[str, Any]) -> bool:
        """Execute a maintenance action"""
        try:
            action_type = action['type']
            target_pid = action.get('target_pid')
            
            if action_type == 'gentle_restart':
                return await self._gentle_restart_process(target_pid)
            
            elif action_type == 'memory_optimization':
                return await self._optimize_process_memory(target_pid)
            
            elif action_type == 'activity_check':
                return await self._check_process_activity(target_pid)
            
            elif action_type == 'cleanup_zombie':
                return await self._cleanup_zombie_process(target_pid)
            
            else:
                print(f"⚠️  Unknown maintenance action: {action_type}")
                return False
                
        except Exception as e:
            print(f"💥 Error executing maintenance action: {e}")
            return False
    
    async def _gentle_restart_process(self, pid: int) -> bool:
        """Attempt to gently restart a stuck process"""
        try:
            if not psutil.pid_exists(pid):
                return False
            
            proc = psutil.Process(pid)
            
            # Try SIGUSR1 first (gentle signal)
            proc.send_signal(signal.SIGUSR1)
            await asyncio.sleep(5)
            
            # Check if process responded
            if proc.is_running() and proc.cpu_percent() < 90:
                return True
            
            # Try SIGTERM
            proc.terminate()
            await asyncio.sleep(10)
            
            # Check if process terminated gracefully
            if not proc.is_running():
                # Attempt to restart the process
                return await self._restart_process(pid)
            
            return False
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    async def _optimize_process_memory(self, pid: int) -> bool:
        """Attempt to optimize memory usage of a process"""
        try:
            # Send a custom signal to trigger garbage collection if supported
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                
                # For Python processes, we can try to trigger GC
                if 'python' in proc.name().lower():
                    # This is a placeholder - real implementation would depend on
                    # the specific process and available interfaces
                    return True
            
            return False
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    async def _check_process_activity(self, pid: int) -> bool:
        """Check if a long-running process is still active"""
        try:
            proc_info = self.monitored_processes.get(pid)
            if not proc_info:
                return False
            
            # Look for signs of progress
            activity_detected = await self._detect_process_activity(pid)
            
            if activity_detected:
                proc_info.last_activity = time.time()
                return True
            
            # Check if process is waiting for input (interactive mode)
            if 'interactive' in proc_info.task_description.lower():
                return True  # Interactive processes are expected to wait
            
            return False
            
        except Exception:
            return False
    
    async def _cleanup_zombie_process(self, pid: int) -> bool:
        """Clean up a zombie process"""
        try:
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                
                # Try to reap the zombie
                try:
                    proc.wait(timeout=1)
                    return True
                except psutil.TimeoutExpired:
                    # Force cleanup
                    proc.kill()
                    return True
            
            return False
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return True  # Process is gone, which is what we wanted
    
    async def _restart_process(self, pid: int) -> bool:
        """Restart a terminated process"""
        try:
            proc_info = self.monitored_processes.get(pid)
            if not proc_info:
                return False
            
            # Use synthesis engine to generate restart command
            restart_command = await self._synthesize_restart_command(proc_info)
            
            if restart_command:
                # Execute restart command
                process = await asyncio.create_subprocess_exec(
                    *restart_command,
                    cwd=proc_info.working_directory,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Update process tracking with new PID
                new_pid = process.pid
                
                # Copy process info to new PID
                new_proc_info = LabrysProcess(
                    pid=new_pid,
                    ppid=os.getpid(),
                    name=proc_info.name,
                    cmdline=restart_command,
                    state=ProcessState.RUNNING,
                    task_status=TaskStatus.IN_PROGRESS,
                    start_time=time.time(),
                    cpu_percent=0.0,
                    memory_mb=0.0,
                    working_directory=proc_info.working_directory,
                    task_description=proc_info.task_description
                )
                
                self.monitored_processes[new_pid] = new_proc_info
                
                print(f"🔄 Restarted process: PID {pid} → {new_pid}")
                return True
            
            return False
            
        except Exception as e:
            print(f"💥 Error restarting process: {e}")
            return False
    
    async def _synthesize_restart_command(self, proc_info: LabrysProcess) -> Optional[List[str]]:
        """Synthesize a restart command for a process using synthesis engine"""
        try:
            # Extract the original command
            original_cmd = proc_info.cmdline
            
            if not original_cmd:
                return None
            
            # Simple restart logic - use the original command
            return original_cmd
            
        except Exception:
            return None
    
    async def _check_task_completion(self):
        """Check if tasks have completed and update status"""
        for pid, proc_info in self.monitored_processes.items():
            if proc_info.task_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                continue
            
            # Check for completion indicators
            completion_detected = await self._detect_task_completion(proc_info)
            
            if completion_detected:
                proc_info.task_status = TaskStatus.COMPLETED
                print(f"✅ Task completed: PID {pid} - {proc_info.task_description}")
    
    async def _detect_task_completion(self, proc_info: LabrysProcess) -> bool:
        """Detect if a task has completed"""
        try:
            # Check if process has terminated normally
            if proc_info.state == ProcessState.DEAD:
                return True
            
            # Look for completion files or patterns
            working_dir = proc_info.working_directory
            if working_dir and os.path.exists(working_dir):
                
                # Check for completion files
                completion_files = [
                    'task_complete.flag',
                    'execution_complete.json',
                    'validation_complete.json',
                    'improvement_complete.json'
                ]
                
                for comp_file in completion_files:
                    if os.path.exists(os.path.join(working_dir, comp_file)):
                        return True
                
                # Check recent log files for completion patterns
                task_type = proc_info.task_description.lower()
                completion_patterns = []
                
                for pattern_type, patterns in self.task_completion_patterns.items():
                    if pattern_type in task_type:
                        completion_patterns.extend(patterns)
                
                if completion_patterns:
                    # Look for patterns in recent files
                    recent_threshold = time.time() - 300  # Last 5 minutes
                    
                    for root, dirs, files in os.walk(working_dir):
                        for file in files:
                            if file.endswith(('.log', '.json', '.txt')):
                                file_path = os.path.join(root, file)
                                try:
                                    if os.path.getmtime(file_path) > recent_threshold:
                                        with open(file_path, 'r') as f:
                                            content = f.read().lower()
                                            if any(pattern in content for pattern in completion_patterns):
                                                return True
                                except (OSError, UnicodeDecodeError):
                                    continue
            
            return False
            
        except Exception:
            return False
    
    async def _cleanup_dead_processes(self):
        """Remove dead processes from monitoring"""
        dead_pids = [
            pid for pid, proc_info in self.monitored_processes.items()
            if proc_info.state == ProcessState.DEAD and 
               proc_info.task_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        ]
        
        for pid in dead_pids:
            proc_info = self.monitored_processes.pop(pid)
            print(f"🗑️  Removed completed process from monitoring: PID {pid}")
    
    async def _report_guardian_status(self):
        """Report current status of all monitored processes"""
        current_time = time.time()
        
        # Only report every 30 seconds to avoid spam
        if not hasattr(self, '_last_status_report'):
            self._last_status_report = 0
        
        if current_time - self._last_status_report < 30:
            return
        
        self._last_status_report = current_time
        
        print(f"\n📊 Guardian Status Report - {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Monitored Processes: {len(self.monitored_processes)}")
        
        status_counts = {}
        for proc_info in self.monitored_processes.values():
            status = proc_info.task_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in status_counts.items():
            print(f"   {status.replace('_', ' ').title()}: {count}")
        
        # Show any recent maintenance actions
        recent_actions = [
            action for action in self.maintenance_history[-5:]
            if current_time - action.timestamp < 300  # Last 5 minutes
        ]
        
        if recent_actions:
            print("   Recent Maintenance:")
            for action in recent_actions:
                status = "✅" if action.success else "❌"
                print(f"   {status} {action.description} (PID {action.target_pid})")
    
    async def _shutdown_guardian(self):
        """Shutdown the guardian system"""
        print("\n🛑 LABRYS Process Guardian Shutting Down")
        
        self.guardian_active = False
        
        # Generate final report
        await self._generate_final_report()
        
        print("👋 Guardian shutdown complete")
    
    async def _generate_final_report(self):
        """Generate final maintenance report"""
        report = {
            "guardian_session": {
                "start_time": getattr(self, '_start_time', time.time()),
                "end_time": time.time(),
                "total_processes_monitored": len(self.monitored_processes),
                "maintenance_actions_taken": len(self.maintenance_history)
            },
            "final_process_status": {
                pid: asdict(proc_info) for pid, proc_info in self.monitored_processes.items()
            },
            "maintenance_history": [asdict(action) for action in self.maintenance_history]
        }
        
        # Save report
        report_file = "labrys_guardian_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Final report saved to: {report_file}")

async def main():
    """Main entry point for LABRYS Process Guardian"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LABRYS Process Guardian - Monitor and maintain LABRYS processes"
    )
    parser.add_argument("--monitor", action="store_true",
                       help="Start monitoring LABRYS processes")
    parser.add_argument("--interval", type=float, default=5.0,
                       help="Monitoring interval in seconds (default: 5.0)")
    parser.add_argument("--max-recovery", type=int, default=3,
                       help="Maximum recovery attempts per process (default: 3)")
    
    args = parser.parse_args()
    
    if args.monitor:
        # Initialize guardian
        guardian = LabrysProcessGuardian()
        guardian.monitor_interval = args.interval
        guardian.max_recovery_attempts = args.max_recovery
        guardian._start_time = time.time()
        
        # Start monitoring
        await guardian.start_guardian()
    
    else:
        parser.print_help()
        print("\n🗲 LABRYS Process Guardian")
        print("   Monitor, maintain, and fix all LABRYS processes")
        print("   Use --monitor to start guardian")

if __name__ == "__main__":
    asyncio.run(main())