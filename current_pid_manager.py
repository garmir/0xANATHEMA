#!/usr/bin/env python3
"""
LABRYS Current PID Manager
Uses actual running LABRYS PIDs as monitoring scenarios
"""

import os
import sys
import json
import psutil
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any

# Add LABRYS paths
sys.path.append(os.path.join(os.path.dirname(__file__), '.labrys'))

class CurrentPIDManager:
    """
    Manages current LABRYS PIDs as real monitoring scenarios
    """
    
    def __init__(self):
        self.active_pids = {}
        self.pid_scenarios = {}
        self.monitoring_active = True
        
    async def discover_and_manage_current_pids(self):
        """Discover current LABRYS PIDs and set up management scenarios"""
        print("🗲 LABRYS Current PID Manager")
        print("   Using actual running PIDs as scenarios")
        print("   " + "="*50)
        
        # Discover current LABRYS processes
        current_pids = await self._discover_current_labrys_pids()
        
        if not current_pids:
            print("❌ No LABRYS PIDs found to manage")
            return
        
        # Set up scenarios for each PID
        for pid_info in current_pids:
            await self._setup_pid_scenario(pid_info)
        
        # Start continuous monitoring and maintenance
        await self._start_continuous_management()
    
    async def _discover_current_labrys_pids(self) -> List[Dict[str, Any]]:
        """Discover all current LABRYS processes"""
        print("🔍 Discovering current LABRYS PIDs...")
        
        labrys_pids = []
        
        for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'status', 'create_time', 'cwd']):
            try:
                info = proc.info
                cmdline = info.get('cmdline', [])
                
                if self._is_labrys_process(cmdline, info.get('name', '')):
                    pid_info = {
                        'pid': info['pid'],
                        'ppid': info.get('ppid', 0),
                        'name': info.get('name', 'unknown'),
                        'cmdline': cmdline,
                        'status': info.get('status', 'unknown'),
                        'start_time': info.get('create_time', time.time()),
                        'working_dir': info.get('cwd', ''),
                        'task_type': self._identify_task_type(cmdline),
                        'task_description': self._generate_task_description(cmdline)
                    }
                    
                    labrys_pids.append(pid_info)
                    print(f"   📋 Found PID {info['pid']}: {pid_info['task_description']}")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        print(f"✅ Discovered {len(labrys_pids)} active LABRYS PIDs")
        return labrys_pids
    
    def _is_labrys_process(self, cmdline: List[str], name: str) -> bool:
        """Check if process is LABRYS-related"""
        if not cmdline:
            return False
        
        cmdline_str = ' '.join(cmdline).lower()
        indicators = ['labrys', 'recursive', 'self_test', 'guardian', 'analytical', 'synthesis']
        return any(indicator in cmdline_str for indicator in indicators)
    
    def _identify_task_type(self, cmdline: List[str]) -> str:
        """Identify the task type for a process"""
        cmdline_str = ' '.join(cmdline).lower()
        
        if 'guardian' in cmdline_str:
            return 'guardian'
        elif '--interactive' in cmdline_str:
            return 'interactive'
        elif 'self_test' in cmdline_str:
            return 'testing'
        elif 'recursive' in cmdline_str:
            return 'improvement'
        elif 'analytical' in cmdline_str:
            return 'analysis'
        elif 'synthesis' in cmdline_str:
            return 'synthesis'
        else:
            return 'labrys_process'
    
    def _generate_task_description(self, cmdline: List[str]) -> str:
        """Generate human-readable task description"""
        cmdline_str = ' '.join(cmdline)
        
        if 'labrys_process_guardian.py' in cmdline_str:
            return 'Process Guardian - Monitoring and maintaining LABRYS processes'
        elif 'labrys_main.py --interactive' in cmdline_str:
            return 'Interactive LABRYS Framework - User interface and command processing'
        elif 'labrys_self_test.py' in cmdline_str:
            return 'Self-Testing System - Recursive validation and testing'
        elif 'recursive_labrys_improvement.py' in cmdline_str:
            return 'Recursive Improvement - Self-enhancement and optimization'
        elif '/bin/zsh' in cmdline_str and 'labrys' in cmdline_str:
            return 'Shell Wrapper - Parent process for LABRYS execution'
        else:
            return 'LABRYS Process - Framework component'
    
    async def _setup_pid_scenario(self, pid_info: Dict[str, Any]):
        """Set up a monitoring scenario for a specific PID"""
        pid = pid_info['pid']
        
        print(f"\n🎯 Setting up scenario for PID {pid}")
        
        # Create scenario directory
        scenario_dir = f".labrys/pid_scenarios/pid_{pid}"
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Determine task completion criteria
        completion_criteria = self._determine_completion_criteria(pid_info)
        
        # Create scenario configuration
        scenario_config = {
            'pid': pid,
            'scenario_type': pid_info['task_type'],
            'task_description': pid_info['task_description'],
            'start_time': datetime.now().isoformat(),
            'process_start_time': datetime.fromtimestamp(pid_info['start_time']).isoformat(),
            'completion_criteria': completion_criteria,
            'monitoring_active': True,
            'maintenance_history': [],
            'health_checks': [],
            'expected_behaviors': self._define_expected_behaviors(pid_info),
            'intervention_thresholds': self._define_intervention_thresholds(pid_info)
        }
        
        # Save scenario configuration
        config_file = os.path.join(scenario_dir, 'scenario_config.json')
        with open(config_file, 'w') as f:
            json.dump(scenario_config, f, indent=2)
        
        # Add to active scenarios
        self.pid_scenarios[pid] = {
            'config': scenario_config,
            'directory': scenario_dir,
            'last_health_check': time.time(),
            'health_score': 1.0,
            'intervention_count': 0
        }
        
        print(f"   📝 Created scenario: {scenario_config['task_description']}")
        print(f"   📁 Directory: {scenario_dir}")
        print(f"   🎯 Completion criteria: {len(completion_criteria['criteria'])} checks")
    
    def _determine_completion_criteria(self, pid_info: Dict[str, Any]) -> Dict[str, Any]:
        """Determine completion criteria for different process types"""
        task_type = pid_info['task_type']
        
        criteria = {
            'type': task_type,
            'criteria': [],
            'timeout_minutes': 60,  # Default timeout
            'heartbeat_required': False
        }
        
        if task_type == 'interactive':
            criteria.update({
                'criteria': [
                    'Process responds to signals',
                    'Memory usage stable',
                    'No zombie state',
                    'User input responsiveness'
                ],
                'timeout_minutes': 0,  # No timeout for interactive
                'heartbeat_required': True
            })
        
        elif task_type == 'guardian':
            criteria.update({
                'criteria': [
                    'Monitoring other processes',
                    'Generating health reports',
                    'Performing maintenance actions',
                    'Memory usage under control'
                ],
                'timeout_minutes': 0,  # Guardian runs indefinitely
                'heartbeat_required': True
            })
        
        elif task_type == 'testing':
            criteria.update({
                'criteria': [
                    'Test completion files created',
                    'Test results documented',
                    'All tests executed',
                    'Validation reports generated'
                ],
                'timeout_minutes': 30,
                'heartbeat_required': False
            })
        
        elif task_type == 'improvement':
            criteria.update({
                'criteria': [
                    'Improvement iterations completed',
                    'Convergence achieved or max iterations',
                    'Results files generated',
                    'Safety validations passed'
                ],
                'timeout_minutes': 60,
                'heartbeat_required': False
            })
        
        else:
            criteria.update({
                'criteria': [
                    'Process runs without errors',
                    'Expected output files created',
                    'Resource usage within limits'
                ],
                'timeout_minutes': 30,
                'heartbeat_required': False
            })
        
        return criteria
    
    def _define_expected_behaviors(self, pid_info: Dict[str, Any]) -> Dict[str, Any]:
        """Define expected behaviors for monitoring"""
        task_type = pid_info['task_type']
        
        behaviors = {
            'cpu_usage': {'min': 0, 'max': 100, 'typical': 10},
            'memory_mb': {'min': 0, 'max': 500, 'typical': 50},
            'file_activity': True,
            'network_activity': False,
            'response_to_signals': True
        }
        
        if task_type == 'interactive':
            behaviors.update({
                'cpu_usage': {'min': 0, 'max': 100, 'typical': 1},  # Usually idle
                'memory_mb': {'min': 10, 'max': 100, 'typical': 30},
                'file_activity': True,
                'user_interaction': True
            })
        
        elif task_type == 'guardian':
            behaviors.update({
                'cpu_usage': {'min': 0, 'max': 50, 'typical': 5},
                'memory_mb': {'min': 20, 'max': 200, 'typical': 40},
                'file_activity': True,
                'monitoring_activity': True
            })
        
        elif task_type in ['testing', 'improvement']:
            behaviors.update({
                'cpu_usage': {'min': 10, 'max': 100, 'typical': 30},
                'memory_mb': {'min': 20, 'max': 300, 'typical': 60},
                'file_activity': True,
                'progress_indicators': True
            })
        
        return behaviors
    
    def _define_intervention_thresholds(self, pid_info: Dict[str, Any]) -> Dict[str, Any]:
        """Define when to intervene with maintenance actions"""
        return {
            'high_cpu_threshold': 95.0,  # % CPU
            'high_memory_threshold': 500.0,  # MB
            'no_activity_threshold': 600,  # seconds
            'zombie_tolerance': 0,  # immediate intervention
            'stuck_threshold': 1800,  # 30 minutes
            'max_interventions': 3
        }
    
    async def _start_continuous_management(self):
        """Start continuous monitoring and management of all PIDs"""
        print(f"\n🛡️  Starting continuous management of {len(self.pid_scenarios)} PIDs")
        print("   Press Ctrl+C to stop monitoring")
        
        cycle_count = 0
        
        try:
            while self.monitoring_active:
                cycle_count += 1
                current_time = time.time()
                
                print(f"\n📊 Management Cycle {cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Check each PID scenario
                for pid, scenario in list(self.pid_scenarios.items()):
                    await self._manage_pid_scenario(pid, scenario, current_time)
                
                # Clean up completed/dead scenarios
                await self._cleanup_completed_scenarios()
                
                # Generate status report
                await self._generate_status_report(cycle_count)
                
                # Wait for next cycle
                print("   ⏸️  Waiting 30 seconds for next cycle...")
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n💥 Management error: {e}")
        finally:
            await self._shutdown_management()
    
    async def _manage_pid_scenario(self, pid: int, scenario: Dict[str, Any], current_time: float):
        """Manage a specific PID scenario"""
        try:
            # Check if process still exists
            if not psutil.pid_exists(pid):
                print(f"   💀 PID {pid} no longer exists - marking for cleanup")
                scenario['config']['monitoring_active'] = False
                return
            
            # Get current process info
            proc = psutil.Process(pid)
            current_stats = {
                'cpu_percent': proc.cpu_percent(),
                'memory_mb': proc.memory_info().rss / 1024 / 1024,
                'status': proc.status(),
                'num_threads': proc.num_threads(),
                'create_time': proc.create_time()
            }
            
            # Perform health check
            health_score = await self._perform_health_check(pid, scenario, current_stats)
            scenario['health_score'] = health_score
            
            # Check if intervention needed
            intervention_needed = await self._check_intervention_needed(pid, scenario, current_stats)
            
            if intervention_needed:
                success = await self._perform_intervention(pid, scenario, intervention_needed)
                scenario['intervention_count'] += 1
                
                # Log intervention
                intervention_log = {
                    'timestamp': datetime.now().isoformat(),
                    'intervention_type': intervention_needed,
                    'success': success,
                    'health_score_before': health_score
                }
                scenario['config']['maintenance_history'].append(intervention_log)
            
            # Update scenario files
            await self._update_scenario_files(pid, scenario, current_stats)
            
            # Check completion criteria
            completion_status = await self._check_completion_criteria(pid, scenario)
            if completion_status:
                print(f"   ✅ PID {pid} task completion detected: {completion_status}")
                scenario['config']['monitoring_active'] = False
            
        except psutil.NoSuchProcess:
            print(f"   👻 PID {pid} disappeared during management")
            scenario['config']['monitoring_active'] = False
        except Exception as e:
            print(f"   ⚠️  Error managing PID {pid}: {e}")
    
    async def _perform_health_check(self, pid: int, scenario: Dict[str, Any], stats: Dict[str, Any]) -> float:
        """Perform health check and return health score (0.0 - 1.0)"""
        health_factors = []
        expected = scenario['config']['expected_behaviors']
        
        # CPU usage check
        cpu_pct = stats['cpu_percent']
        if cpu_pct <= expected['cpu_usage']['max']:
            health_factors.append(1.0)
        else:
            health_factors.append(0.3)
        
        # Memory usage check
        memory_mb = stats['memory_mb']
        if memory_mb <= expected['memory_mb']['max']:
            health_factors.append(1.0)
        else:
            health_factors.append(0.5)
        
        # Status check
        if stats['status'] in ['running', 'sleeping']:
            health_factors.append(1.0)
        elif stats['status'] == 'zombie':
            health_factors.append(0.0)
        else:
            health_factors.append(0.7)
        
        # Calculate overall health score
        health_score = sum(health_factors) / len(health_factors) if health_factors else 0.0
        
        # Log health check
        health_log = {
            'timestamp': datetime.now().isoformat(),
            'health_score': health_score,
            'cpu_percent': cpu_pct,
            'memory_mb': memory_mb,
            'status': stats['status']
        }
        scenario['config']['health_checks'].append(health_log)
        
        # Keep only last 10 health checks
        if len(scenario['config']['health_checks']) > 10:
            scenario['config']['health_checks'] = scenario['config']['health_checks'][-10:]
        
        return health_score
    
    async def _check_intervention_needed(self, pid: int, scenario: Dict[str, Any], stats: Dict[str, Any]) -> str:
        """Check if intervention is needed and return intervention type"""
        thresholds = scenario['config']['intervention_thresholds']
        
        # Check CPU usage
        if stats['cpu_percent'] > thresholds['high_cpu_threshold']:
            return 'high_cpu'
        
        # Check memory usage
        if stats['memory_mb'] > thresholds['high_memory_threshold']:
            return 'high_memory'
        
        # Check for zombie state
        if stats['status'] == 'zombie':
            return 'zombie_cleanup'
        
        # Check if stuck (would need activity tracking)
        if scenario['health_score'] < 0.3:
            return 'health_recovery'
        
        return None
    
    async def _perform_intervention(self, pid: int, scenario: Dict[str, Any], intervention_type: str) -> bool:
        """Perform maintenance intervention"""
        print(f"   🔧 Performing intervention '{intervention_type}' on PID {pid}")
        
        try:
            if intervention_type == 'high_cpu':
                # Send gentle signal to reduce CPU usage
                proc = psutil.Process(pid)
                proc.send_signal(1)  # SIGHUP
                await asyncio.sleep(5)
                return True
            
            elif intervention_type == 'high_memory':
                # Log memory issue (real intervention would be more complex)
                print(f"      📝 Logged high memory usage for PID {pid}")
                return True
            
            elif intervention_type == 'zombie_cleanup':
                # Attempt to clean up zombie
                proc = psutil.Process(pid)
                proc.kill()
                return True
            
            elif intervention_type == 'health_recovery':
                # Generic health recovery
                print(f"      🏥 Attempting health recovery for PID {pid}")
                return True
            
            return False
            
        except Exception as e:
            print(f"      ❌ Intervention failed: {e}")
            return False
    
    async def _update_scenario_files(self, pid: int, scenario: Dict[str, Any], stats: Dict[str, Any]):
        """Update scenario files with current status"""
        scenario_dir = scenario['directory']
        
        # Update status file
        status_file = os.path.join(scenario_dir, 'current_status.json')
        status_data = {
            'pid': pid,
            'timestamp': datetime.now().isoformat(),
            'health_score': scenario['health_score'],
            'intervention_count': scenario['intervention_count'],
            'process_stats': stats,
            'monitoring_active': scenario['config']['monitoring_active']
        }
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        # Update config file
        config_file = os.path.join(scenario_dir, 'scenario_config.json')
        with open(config_file, 'w') as f:
            json.dump(scenario['config'], f, indent=2)
    
    async def _check_completion_criteria(self, pid: int, scenario: Dict[str, Any]) -> str:
        """Check if completion criteria are met"""
        criteria = scenario['config']['completion_criteria']
        
        # For now, simple checks
        if scenario['config']['task_description'] == 'Interactive LABRYS Framework - User interface and command processing':
            # Interactive processes don't complete automatically
            return None
        
        if scenario['config']['task_description'] == 'Process Guardian - Monitoring and maintaining LABRYS processes':
            # Guardian processes run indefinitely
            return None
        
        # Check for completion files
        scenario_dir = scenario['directory']
        completion_files = ['task_complete.flag', 'execution_complete.json', 'test_complete.json']
        
        for comp_file in completion_files:
            if os.path.exists(os.path.join(scenario_dir, comp_file)):
                return f"Completion file found: {comp_file}"
        
        return None
    
    async def _cleanup_completed_scenarios(self):
        """Remove completed scenarios from active monitoring"""
        completed_pids = [
            pid for pid, scenario in self.pid_scenarios.items()
            if not scenario['config']['monitoring_active']
        ]
        
        for pid in completed_pids:
            scenario = self.pid_scenarios.pop(pid)
            print(f"   🗑️  Removed completed scenario: PID {pid}")
    
    async def _generate_status_report(self, cycle: int):
        """Generate status report for all scenarios"""
        active_count = len(self.pid_scenarios)
        healthy_count = sum(1 for s in self.pid_scenarios.values() if s['health_score'] > 0.7)
        
        print(f"   📈 Status: {healthy_count}/{active_count} PIDs healthy")
        
        # Save detailed report every 5 cycles
        if cycle % 5 == 0:
            report_data = {
                'cycle': cycle,
                'timestamp': datetime.now().isoformat(),
                'active_scenarios': active_count,
                'healthy_scenarios': healthy_count,
                'scenarios': {
                    pid: {
                        'health_score': scenario['health_score'],
                        'intervention_count': scenario['intervention_count'],
                        'task_description': scenario['config']['task_description']
                    }
                    for pid, scenario in self.pid_scenarios.items()
                }
            }
            
            report_file = 'current_pid_management_report.json'
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
    
    async def _shutdown_management(self):
        """Shutdown management system"""
        print("\n🔚 Shutting down PID management...")
        
        self.monitoring_active = False
        
        # Generate final report
        final_report = {
            'shutdown_time': datetime.now().isoformat(),
            'final_scenario_count': len(self.pid_scenarios),
            'scenarios': {
                pid: scenario['config'] for pid, scenario in self.pid_scenarios.items()
            }
        }
        
        with open('final_pid_management_report.json', 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print("📋 Final management report saved")
        print("👋 PID management shutdown complete")

async def main():
    """Main function"""
    manager = CurrentPIDManager()
    await manager.discover_and_manage_current_pids()

if __name__ == "__main__":
    asyncio.run(main())