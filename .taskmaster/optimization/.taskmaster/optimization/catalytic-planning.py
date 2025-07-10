#!/usr/bin/env python3
import json

def create_catalytic_execution_plan(pebbling_strategy):
    """Generate catalytic execution plan with memory reuse"""
    
    plan = {
        'execution_type': 'catalytic',
        'workspace_size': '10GB',
        'reuse_factor': 0.8,
        'memory_efficiency': 'high',
        'execution_phases': [],
        'resource_reuse_map': {},
        'total_execution_time': 0
    }
    
    # Create execution phases with memory reuse
    pebbles = pebbling_strategy.get('pebbles', [])
    reused_memory = 0
    
    for i, pebble in enumerate(pebbles):
        phase = {
            'phase_id': i + 1,
            'task_id': pebble['node_id'],
            'memory_allocated': pebble['memory_footprint'],
            'memory_reused': int(reused_memory * 0.8),  # 80% reuse factor
            'net_memory': pebble['memory_footprint'] - int(reused_memory * 0.8),
            'execution_time': pebble['allocation_time'],
            'catalytic_efficiency': 0.8 if i > 0 else 1.0
        }
        
        plan['execution_phases'].append(phase)
        plan['total_execution_time'] += phase['execution_time']
        
        # Update reused memory pool
        reused_memory = pebble['memory_footprint']
    
    # Calculate overall efficiency
    total_allocated = sum(p['memory_allocated'] for p in plan['execution_phases'])
    total_net = sum(p['net_memory'] for p in plan['execution_phases'])
    plan['memory_savings'] = f"{((total_allocated - total_net) / total_allocated * 100):.1f}%"
    
    return plan

if __name__ == "__main__":
    with open('pebbling-strategy.json', 'r') as f:
        strategy = json.load(f)
    
    plan = create_catalytic_execution_plan(strategy)
    
    with open('catalytic-execution.json', 'w') as f:
        json.dump(plan, f, indent=2)
    
    print(f"✅ Created catalytic execution plan: {plan['memory_savings']} memory savings")
