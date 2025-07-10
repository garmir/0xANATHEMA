#!/usr/bin/env python3
import json
import random

def generate_pebbling_strategy(task_tree):
    """Generate pebbling strategy for resource allocation"""
    nodes = task_tree.get('nodes', [])
    
    strategy = {
        'strategy_type': 'branching-program',
        'memory_minimization': True,
        'pebbles': [],
        'allocation_order': [],
        'resource_conflicts': [],
        'timing_constraints': {}
    }
    
    # Generate pebbling sequence
    for i, node in enumerate(nodes):
        pebble = {
            'node_id': node['id'],
            'pebble_id': f"P{i+1}",
            'allocation_time': i * 5,  # 5 minute intervals
            'resource_requirement': node['resources'],
            'priority': random.choice(['high', 'medium', 'low']),
            'memory_footprint': int(node['resources']['memory'].replace('MB', ''))
        }
        strategy['pebbles'].append(pebble)
        strategy['allocation_order'].append(node['id'])
    
    # Sort by memory footprint for optimal allocation
    strategy['pebbles'].sort(key=lambda x: x['memory_footprint'])
    
    # Calculate total memory usage
    total_memory = sum(p['memory_footprint'] for p in strategy['pebbles'])
    strategy['total_memory_required'] = f"{total_memory}MB"
    strategy['peak_memory'] = f"{max(p['memory_footprint'] for p in strategy['pebbles'])}MB"
    
    return strategy

if __name__ == "__main__":
    with open('tree-optimized.json', 'r') as f:
        task_tree = json.load(f)
    
    strategy = generate_pebbling_strategy(task_tree)
    
    with open('pebbling-strategy.json', 'w') as f:
        json.dump(strategy, f, indent=2)
    
    print(f"✅ Generated pebbling strategy: {strategy['total_memory_required']} total memory")
