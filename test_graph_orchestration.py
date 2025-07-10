#!/usr/bin/env python3
"""
Test script for graph-based orchestration system
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

# Import the graph orchestration system
try:
    from graph_based_orchestration import (
        GraphBasedOrchestrator, ExecutableWorkflowGraph, GraphNode, GraphEdge,
        GraphNodeType, AgentRole, GraphState, create_research_planning_graph,
        create_conditional_workflow_graph
    )
    from multi_agent_orchestration import (
        ResearchAgent, PlanningAgent, ExecutionAgent, ValidationAgent,
        InMemoryMessageBus
    )
    print("✅ Successfully imported graph orchestration components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

async def test_simple_workflow():
    """Test basic workflow execution"""
    print("\n🔄 Testing Simple Linear Workflow")
    print("-" * 40)
    
    try:
        # Create communication system
        communication = InMemoryMessageBus()
        
        # Create orchestrator
        orchestrator = GraphBasedOrchestrator(communication)
        
        # Register basic agents (mock agents for testing)
        class MockAgent:
            def __init__(self, agent_id, role, max_tasks=5):
                self.agent_id = agent_id
                self.role = role
                self.current_tasks = []
                self.capabilities = type('obj', (object,), {
                    'max_concurrent_tasks': max_tasks,
                    'processing_time_estimate': 1.0
                })()
        
        agents = [
            MockAgent("research-test", AgentRole.RESEARCH),
            MockAgent("planning-test", AgentRole.PLANNING),
            MockAgent("execution-test", AgentRole.EXECUTION),
            MockAgent("validation-test", AgentRole.VALIDATION)
        ]
        
        for agent in agents:
            orchestrator.agents[agent.agent_id] = agent
        
        print(f"   Registered {len(agents)} mock agents")
        
        # Create simple graph
        graph = ExecutableWorkflowGraph("test_linear")
        
        # Add nodes
        start_node = GraphNode("start", GraphNodeType.START, "Start")
        research_node = GraphNode("research", GraphNodeType.AGENT, "Research", AgentRole.RESEARCH)
        end_node = GraphNode("end", GraphNodeType.END, "End")
        
        graph.add_node(start_node)
        graph.add_node(research_node) 
        graph.add_node(end_node)
        
        # Add edges
        graph.add_edge(GraphEdge("start", "research"))
        graph.add_edge(GraphEdge("research", "end"))
        
        print(f"   Created graph with {len(graph.nodes)} nodes")
        
        # Validate graph
        is_valid = graph.validate()
        print(f"   Graph validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        if is_valid:
            orchestrator.set_graph(graph)
            
            # Create initial state
            initial_state = GraphState(data={
                "topic": "Graph Testing",
                "test_mode": True
            })
            
            # Execute with timeout protection
            start_time = datetime.now()
            try:
                # Use asyncio.wait_for for timeout protection
                result_state = await asyncio.wait_for(
                    orchestrator.execute_graph(initial_state), 
                    timeout=5.0
                )
                execution_time = (datetime.now() - start_time).total_seconds()
                
                print(f"   ✅ Execution completed in {execution_time:.2f}s")
                print(f"   Execution path: {' -> '.join(result_state.execution_history)}")
                print(f"   Final state keys: {list(result_state.data.keys())}")
                
                return True
                
            except asyncio.TimeoutError:
                print(f"   ⏱️ Execution timed out after 5 seconds")
                return False
                
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

async def test_conditional_routing():
    """Test conditional routing logic"""
    print("\n🔀 Testing Conditional Routing")
    print("-" * 40)
    
    try:
        # Create a simple conditional workflow
        graph = ExecutableWorkflowGraph("test_conditional")
        
        # Add nodes
        start_node = GraphNode("start", GraphNodeType.START, "Start")
        decision_node = GraphNode("decision", GraphNodeType.CONDITIONAL, "Decision")
        path_a_node = GraphNode("path_a", GraphNodeType.AGENT, "Path A", AgentRole.RESEARCH)
        path_b_node = GraphNode("path_b", GraphNodeType.AGENT, "Path B", AgentRole.PLANNING)
        end_node = GraphNode("end", GraphNodeType.END, "End")
        
        graph.add_node(start_node)
        graph.add_node(decision_node)
        graph.add_node(path_a_node)
        graph.add_node(path_b_node)
        graph.add_node(end_node)
        
        # Add edges with conditions
        graph.add_edge(GraphEdge("start", "decision"))
        
        # Conditional edges
        condition_a = lambda state: state.get("priority", 0) > 5
        condition_b = lambda state: state.get("priority", 0) <= 5
        
        graph.add_conditional_edge("decision", "path_a", condition_a)
        graph.add_conditional_edge("decision", "path_b", condition_b) 
        
        graph.add_edge(GraphEdge("path_a", "end"))
        graph.add_edge(GraphEdge("path_b", "end"))
        
        print(f"   Created conditional graph with {len(graph.nodes)} nodes")
        
        # Test routing strategies
        from graph_based_orchestration import ConditionalRoutingStrategy, PriorityRoutingStrategy
        
        routing_strategy = ConditionalRoutingStrategy()
        
        # Test high priority path
        high_priority_state = GraphState(data={"priority": 8})
        high_priority_edges = [
            GraphEdge("decision", "path_a", condition_a),
            GraphEdge("decision", "path_b", condition_b)
        ]
        
        next_nodes_high = routing_strategy.select_next_nodes("decision", high_priority_state, high_priority_edges)
        print(f"   High priority (8) routes to: {next_nodes_high}")
        
        # Test low priority path
        low_priority_state = GraphState(data={"priority": 3})
        next_nodes_low = routing_strategy.select_next_nodes("decision", low_priority_state, high_priority_edges)
        print(f"   Low priority (3) routes to: {next_nodes_low}")
        
        # Validate expected routing
        expected_high = ["path_a"]
        expected_low = ["path_b"]
        
        routing_correct = (next_nodes_high == expected_high and next_nodes_low == expected_low)
        print(f"   Routing logic: {'✅ Correct' if routing_correct else '❌ Incorrect'}")
        
        return routing_correct
        
    except Exception as e:
        print(f"   ❌ Conditional routing test failed: {e}")
        return False

async def test_graph_validation():
    """Test graph validation logic"""
    print("\n✅ Testing Graph Validation")
    print("-" * 40)
    
    try:
        # Test valid graph
        valid_graph = ExecutableWorkflowGraph("valid_test")
        valid_graph.add_node(GraphNode("start", GraphNodeType.START, "Start"))
        valid_graph.add_node(GraphNode("middle", GraphNodeType.AGENT, "Middle", AgentRole.RESEARCH))
        valid_graph.add_node(GraphNode("end", GraphNodeType.END, "End"))
        valid_graph.add_edge(GraphEdge("start", "middle"))
        valid_graph.add_edge(GraphEdge("middle", "end"))
        
        is_valid = valid_graph.validate()
        print(f"   Valid graph validation: {'✅ Passed' if is_valid else '❌ Failed'}")
        
        # Test invalid graph (no start node)
        invalid_graph = ExecutableWorkflowGraph("invalid_test")
        invalid_graph.add_node(GraphNode("middle", GraphNodeType.AGENT, "Middle", AgentRole.RESEARCH))
        invalid_graph.add_node(GraphNode("end", GraphNodeType.END, "End"))
        invalid_graph.add_edge(GraphEdge("middle", "end"))
        
        is_invalid = not invalid_graph.validate()
        print(f"   Invalid graph rejection: {'✅ Passed' if is_invalid else '❌ Failed'}")
        
        return is_valid and is_invalid
        
    except Exception as e:
        print(f"   ❌ Graph validation test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Graph-Based Orchestration Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Run tests
    test_results.append(await test_simple_workflow())
    test_results.append(await test_conditional_routing())
    test_results.append(await test_graph_validation())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
    
    # Save test results
    results_data = {
        "test_execution_timestamp": datetime.now().isoformat(),
        "test_suite": "Graph-Based Orchestration",
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests/total_tests)*100 if total_tests > 0 else 0
        },
        "individual_tests": {
            "simple_workflow": test_results[0] if len(test_results) > 0 else False,
            "conditional_routing": test_results[1] if len(test_results) > 1 else False,
            "graph_validation": test_results[2] if len(test_results) > 2 else False
        },
        "features_tested": [
            "Graph creation and validation",
            "Agent registration and management",
            "Conditional routing strategies",
            "Workflow execution with timeout protection",
            "State management and propagation"
        ]
    }
    
    # Ensure reports directory exists
    Path(".taskmaster/reports").mkdir(parents=True, exist_ok=True)
    
    with open(".taskmaster/reports/graph-orchestration-test-results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n📄 Test results saved to: .taskmaster/reports/graph-orchestration-test-results.json")
    
    if passed_tests == total_tests:
        print("✅ All tests passed! Graph-based orchestration is working correctly.")
    else:
        print("⚠️ Some tests failed. Check implementation for issues.")

if __name__ == "__main__":
    asyncio.run(main())