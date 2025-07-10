#!/usr/bin/env python3
"""
Quick LABRYS Health Check
Monitor current status of all LABRYS processes
"""

import os
import sys
import psutil
import json
from datetime import datetime

def check_labrys_processes():
    """Check health of all LABRYS processes"""
    print("🗲 LABRYS Process Health Check")
    print("   " + "="*40)
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    labrys_processes = []
    
    for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'status', 'create_time', 'cpu_percent', 'memory_info']):
        try:
            info = proc.info
            cmdline = info.get('cmdline', [])
            
            if cmdline and is_labrys_process(cmdline, info.get('name', '')):
                # Get process details
                try:
                    memory_mb = info['memory_info'].rss / 1024 / 1024
                except:
                    memory_mb = 0
                
                process_info = {
                    'pid': info['pid'],
                    'ppid': info.get('ppid', 0),
                    'name': info.get('name', 'unknown'),
                    'cmdline': ' '.join(cmdline),
                    'status': info.get('status', 'unknown'),
                    'cpu_percent': info.get('cpu_percent', 0),
                    'memory_mb': memory_mb,
                    'runtime_minutes': (datetime.now().timestamp() - info.get('create_time', 0)) / 60
                }
                
                labrys_processes.append(process_info)
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    # Display results
    if not labrys_processes:
        print("❌ No LABRYS processes found")
        return
    
    print(f"✅ Found {len(labrys_processes)} LABRYS processes:")
    print()
    
    for i, proc in enumerate(labrys_processes, 1):
        print(f"🔸 Process {i}:")
        print(f"   PID: {proc['pid']}")
        print(f"   Status: {proc['status']}")
        print(f"   CPU: {proc['cpu_percent']:.1f}%")
        print(f"   Memory: {proc['memory_mb']:.1f} MB")
        print(f"   Runtime: {proc['runtime_minutes']:.1f} minutes")
        print(f"   Command: {proc['cmdline'][:80]}{'...' if len(proc['cmdline']) > 80 else ''}")
        
        # Health assessment
        health_status = assess_health(proc)
        print(f"   Health: {health_status}")
        print()
    
    # Check for guardian process
    guardian_running = any('guardian' in proc['cmdline'].lower() for proc in labrys_processes)
    
    if guardian_running:
        print("🛡️  Process Guardian: ACTIVE")
    else:
        print("⚠️  Process Guardian: NOT DETECTED")
    
    print()
    print("📊 Summary:")
    
    # Status summary
    status_counts = {}
    for proc in labrys_processes:
        status = proc['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    for status, count in status_counts.items():
        print(f"   {status}: {count}")
    
    # Overall assessment
    healthy_count = sum(1 for proc in labrys_processes if 'healthy' in assess_health(proc).lower())
    
    if healthy_count == len(labrys_processes):
        print("✅ Overall Status: ALL PROCESSES HEALTHY")
    elif healthy_count > len(labrys_processes) / 2:
        print("⚠️  Overall Status: MOSTLY HEALTHY")
    else:
        print("❌ Overall Status: MULTIPLE ISSUES DETECTED")

def is_labrys_process(cmdline, name):
    """Check if process is LABRYS-related"""
    cmdline_str = ' '.join(cmdline).lower()
    indicators = ['labrys', 'recursive', 'self_test', 'guardian', 'dual_blade']
    return any(indicator in cmdline_str for indicator in indicators)

def assess_health(proc):
    """Assess health of a process"""
    issues = []
    
    # Check CPU usage
    if proc['cpu_percent'] > 95:
        issues.append("high CPU")
    
    # Check memory usage
    if proc['memory_mb'] > 500:
        issues.append("high memory")
    
    # Check runtime
    if proc['runtime_minutes'] > 60:
        issues.append("long running")
    
    # Check status
    if proc['status'] in ['zombie', 'stopped']:
        issues.append(f"bad status ({proc['status']})")
    
    if not issues:
        return "🟢 Healthy"
    elif len(issues) == 1:
        return f"🟡 Warning: {issues[0]}"
    else:
        return f"🔴 Issues: {', '.join(issues)}"

if __name__ == "__main__":
    try:
        check_labrys_processes()
    except KeyboardInterrupt:
        print("\n👋 Health check interrupted")
    except Exception as e:
        print(f"💥 Error during health check: {e}")