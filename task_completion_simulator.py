#!/usr/bin/env python3
"""
LABRYS Task Completion Simulator
Creates task scenarios to test guardian maintenance capabilities
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime

# Add LABRYS paths
sys.path.append(os.path.join(os.path.dirname(__file__), '.labrys'))

async def simulate_task_completion_scenarios():
    """Simulate various task completion scenarios"""
    print("🎯 LABRYS Task Completion Simulator")
    print("   Testing guardian maintenance capabilities")
    print("   " + "="*50)
    
    scenarios = [
        {
            'name': 'Analysis Task',
            'type': 'analytical',
            'duration': 30,
            'completion_indicator': 'analysis_complete.json'
        },
        {
            'name': 'Synthesis Task', 
            'type': 'synthesis',
            'duration': 45,
            'completion_indicator': 'synthesis_complete.json'
        },
        {
            'name': 'Validation Task',
            'type': 'validation', 
            'duration': 20,
            'completion_indicator': 'validation_complete.json'
        }
    ]
    
    # Create scenario directories
    scenarios_dir = '.labrys/scenarios'
    os.makedirs(scenarios_dir, exist_ok=True)
    
    for scenario in scenarios:
        print(f"\n🚀 Starting scenario: {scenario['name']}")
        
        # Create scenario directory
        scenario_dir = os.path.join(scenarios_dir, scenario['type'])
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Create task file
        task_file = os.path.join(scenario_dir, 'task_status.json')
        task_data = {
            'task_name': scenario['name'],
            'task_type': scenario['type'],
            'status': 'in_progress',
            'start_time': datetime.now().isoformat(),
            'expected_duration': scenario['duration'],
            'completion_indicator': scenario['completion_indicator']
        }
        
        with open(task_file, 'w') as f:
            json.dump(task_data, f, indent=2)
        
        print(f"   📝 Created task file: {task_file}")
        
        # Simulate task progress
        await simulate_task_progress(scenario, scenario_dir)
        
        # Create completion indicator
        completion_file = os.path.join(scenario_dir, scenario['completion_indicator'])
        completion_data = {
            'task_name': scenario['name'],
            'status': 'completed',
            'completion_time': datetime.now().isoformat(),
            'success': True,
            'results': f"Task {scenario['name']} completed successfully"
        }
        
        with open(completion_file, 'w') as f:
            json.dump(completion_data, f, indent=2)
        
        print(f"   ✅ Task completed: {scenario['name']}")
        
        # Update task status
        task_data['status'] = 'completed'
        task_data['completion_time'] = datetime.now().isoformat()
        
        with open(task_file, 'w') as f:
            json.dump(task_data, f, indent=2)

async def simulate_task_progress(scenario, scenario_dir):
    """Simulate task making progress"""
    duration = scenario['duration']
    steps = 5
    step_duration = duration / steps
    
    for step in range(1, steps + 1):
        await asyncio.sleep(step_duration)
        
        # Create progress file
        progress_file = os.path.join(scenario_dir, f'progress_{step}.log')
        progress_data = {
            'step': step,
            'total_steps': steps,
            'progress_percent': (step / steps) * 100,
            'timestamp': datetime.now().isoformat(),
            'message': f"Step {step} of {steps} completed for {scenario['name']}"
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"   📊 Progress: {progress_data['progress_percent']:.0f}% ({step}/{steps})")

async def create_long_running_task():
    """Create a long-running task to test guardian maintenance"""
    print("\n⏳ Creating long-running task for guardian testing...")
    
    long_task_dir = '.labrys/long_running'
    os.makedirs(long_task_dir, exist_ok=True)
    
    # Create a task that will run for a while
    task_data = {
        'task_name': 'Long Running Analysis',
        'task_type': 'long_running',
        'status': 'in_progress',
        'start_time': datetime.now().isoformat(),
        'expected_duration': 300,  # 5 minutes
        'heartbeat_file': 'heartbeat.json'
    }
    
    task_file = os.path.join(long_task_dir, 'task_status.json')
    with open(task_file, 'w') as f:
        json.dump(task_data, f, indent=2)
    
    # Create heartbeat updates every 10 seconds
    for i in range(30):  # 5 minutes worth of heartbeats
        heartbeat_data = {
            'heartbeat': i + 1,
            'timestamp': datetime.now().isoformat(),
            'status': 'running',
            'message': f'Heartbeat {i + 1} - task is active'
        }
        
        heartbeat_file = os.path.join(long_task_dir, 'heartbeat.json')
        with open(heartbeat_file, 'w') as f:
            json.dump(heartbeat_data, f, indent=2)
        
        if i % 6 == 0:  # Print every minute
            print(f"   💓 Heartbeat {i + 1}/30 - Task is active")
        
        await asyncio.sleep(10)
    
    # Complete the long-running task
    completion_data = {
        'task_name': 'Long Running Analysis',
        'status': 'completed',
        'completion_time': datetime.now().isoformat(),
        'total_heartbeats': 30,
        'success': True
    }
    
    completion_file = os.path.join(long_task_dir, 'task_complete.flag')
    with open(completion_file, 'w') as f:
        json.dump(completion_data, f, indent=2)
    
    print("   ✅ Long-running task completed successfully")

async def test_error_scenarios():
    """Test error scenarios for guardian to handle"""
    print("\n🚨 Testing error scenarios...")
    
    error_dir = '.labrys/error_scenarios'
    os.makedirs(error_dir, exist_ok=True)
    
    # Scenario 1: Task that appears stuck
    stuck_task_data = {
        'task_name': 'Stuck Task Simulation',
        'status': 'stuck',
        'start_time': (datetime.now().timestamp() - 1800),  # Started 30 minutes ago
        'last_activity': (datetime.now().timestamp() - 600),  # No activity for 10 minutes
        'error_type': 'appears_stuck'
    }
    
    with open(os.path.join(error_dir, 'stuck_task.json'), 'w') as f:
        json.dump(stuck_task_data, f, indent=2)
    
    print("   🚧 Created stuck task scenario")
    
    # Scenario 2: Task that failed
    failed_task_data = {
        'task_name': 'Failed Task Simulation',
        'status': 'failed',
        'start_time': datetime.now().isoformat(),
        'error_message': 'Simulated failure for testing',
        'error_type': 'task_failure'
    }
    
    with open(os.path.join(error_dir, 'failed_task.json'), 'w') as f:
        json.dump(failed_task_data, f, indent=2)
    
    print("   ❌ Created failed task scenario")

async def main():
    """Main simulation function"""
    print("Starting LABRYS task completion simulation...")
    
    try:
        # Run task completion scenarios
        await simulate_task_completion_scenarios()
        
        # Create long-running task (this will take 5 minutes)
        await create_long_running_task()
        
        # Test error scenarios
        await test_error_scenarios()
        
        print("\n🎉 All task scenarios completed!")
        print("   Guardian should have monitored and maintained all tasks")
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted")
    except Exception as e:
        print(f"💥 Simulation error: {e}")

if __name__ == "__main__":
    asyncio.run(main())