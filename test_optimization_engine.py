#!/usr/bin/env python3
"""
Comprehensive tests for Optimization Engine module
"""

import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from optimization_engine import (
    OptimizationEngine,
    OptimizationStrategy,
    ExecutionPlan,
    ResourceConstraints
)
from task_complexity_analyzer import TaskComplexityAnalyzer, TaskComplexity, ComplexityClass


class TestOptimizationEngine(unittest.TestCase):
    """Test suite for OptimizationEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test task data
        self.test_tasks = {
            "tags": {
                "master": {
                    "tasks": [
                        {
                            "id": "1",
                            "title": "Fast task",
                            "description": "Quick linear operation",
                            "details": "Simple file read",
                            "dependencies": []
                        },
                        {
                            "id": "2",
                            "title": "Slow task",
                            "description": "Complex recursive algorithm",
                            "details": "Exponential backtracking with recursive calls",
                            "dependencies": ["1"]
                        },
                        {
                            "id": "3",
                            "title": "Medium task",
                            "description": "Quadratic matrix operation",
                            "details": "Nested loop matrix multiplication",
                            "dependencies": []
                        },
                        {
                            "id": "4",
                            "title": "Parallel task",
                            "description": "Highly parallelizable operation",
                            "details": "Independent parallel processing batch concurrent",
                            "dependencies": ["1"]
                        },
                        {
                            "id": "5",
                            "title": "CPU intensive task",
                            "description": "Complex computation",
                            "details": "Intensive algorithm compute calculate optimization",
                            "dependencies": ["3"]
                        }
                    ]
                }
            }
        }
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_tasks, self.temp_file)
        self.temp_file.close()
        
        # Initialize analyzer and engine
        self.analyzer = TaskComplexityAnalyzer(self.temp_file.name)
        self.engine = OptimizationEngine(self.analyzer, self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_file.name)
    
    def test_analyze_dependencies(self):
        """Test dependency graph analysis"""
        dependency_graph = self.engine.analyze_dependencies()
        
        self.assertEqual(dependency_graph["1"], [])
        self.assertEqual(dependency_graph["2"], ["1"])
        self.assertEqual(dependency_graph["3"], [])
        self.assertEqual(dependency_graph["4"], ["1"])
        self.assertEqual(dependency_graph["5"], ["3"])
    
    def test_get_task_complexities(self):
        """Test task complexity retrieval"""
        complexities = self.engine.get_task_complexities()
        
        self.assertEqual(len(complexities), 5)
        self.assertIn("1", complexities)
        self.assertIn("2", complexities)
        self.assertIn("3", complexities)
        self.assertIn("4", complexities)
        self.assertIn("5", complexities)
        
        # Verify complexity types
        self.assertIsInstance(complexities["1"], TaskComplexity)
        self.assertEqual(complexities["2"].time_complexity, ComplexityClass.EXPONENTIAL)
        self.assertEqual(complexities["3"].time_complexity, ComplexityClass.QUADRATIC)
    
    def test_optimize_greedy_shortest_first(self):
        """Test greedy shortest-first optimization"""
        constraints = ResourceConstraints(
            max_cpu_cores=4,
            max_memory_gb=8.0,
            max_disk_gb=100.0,
            max_parallel_tasks=4,
            priority_weights={'high': 1.0, 'medium': 0.6, 'low': 0.3}
        )
        
        plan = self.engine.optimize_execution_order(
            OptimizationStrategy.GREEDY_SHORTEST_FIRST, constraints
        )
        
        # Verify plan structure
        self.assertEqual(plan.strategy, OptimizationStrategy.GREEDY_SHORTEST_FIRST)
        self.assertEqual(len(plan.task_order), 5)
        self.assertIsInstance(plan.parallel_groups, list)
        self.assertIsInstance(plan.resource_allocation, dict)
        self.assertGreater(plan.estimated_total_time, 0)
        self.assertGreaterEqual(plan.efficiency_score, 0)
        
        # Dependencies should be respected
        task_order = plan.task_order
        task1_index = task_order.index("1")
        task2_index = task_order.index("2")
        task4_index = task_order.index("4")
        
        self.assertLess(task1_index, task2_index)  # 1 must come before 2
        self.assertLess(task1_index, task4_index)  # 1 must come before 4
    
    def test_optimize_greedy_resource_aware(self):
        """Test greedy resource-aware optimization"""
        constraints = ResourceConstraints(
            max_cpu_cores=2,
            max_memory_gb=4.0,
            max_disk_gb=50.0,
            max_parallel_tasks=2,
            priority_weights={'high': 1.0, 'medium': 0.6, 'low': 0.3}
        )
        
        plan = self.engine.optimize_execution_order(
            OptimizationStrategy.GREEDY_RESOURCE_AWARE, constraints
        )
        
        self.assertEqual(plan.strategy, OptimizationStrategy.GREEDY_RESOURCE_AWARE)
        self.assertEqual(len(plan.task_order), 5)
        
        # Should respect resource constraints
        for task_id, allocation in plan.resource_allocation.items():
            self.assertLessEqual(allocation['cpu_cores'], constraints.max_cpu_cores)
            self.assertLessEqual(allocation['memory_gb'], constraints.max_memory_gb)
    
    def test_optimize_critical_path(self):
        """Test critical path optimization"""
        plan = self.engine.optimize_execution_order(OptimizationStrategy.CRITICAL_PATH)
        
        self.assertEqual(plan.strategy, OptimizationStrategy.CRITICAL_PATH)
        self.assertEqual(len(plan.task_order), 5)
        
        # Should identify critical path metadata
        self.assertIn('critical_path_distances', plan.optimization_metadata)
        distances = plan.optimization_metadata['critical_path_distances']
        self.assertIsInstance(distances, dict)
        self.assertEqual(len(distances), 5)
    
    def test_optimize_adaptive_scheduling(self):
        """Test adaptive scheduling optimization"""
        plan = self.engine.optimize_execution_order(OptimizationStrategy.ADAPTIVE_SCHEDULING)
        
        self.assertEqual(plan.strategy, OptimizationStrategy.ADAPTIVE_SCHEDULING)
        self.assertEqual(len(plan.task_order), 5)
        
        # Should have evaluated multiple strategies
        self.assertIn('strategies_evaluated', plan.optimization_metadata)
        strategies = plan.optimization_metadata['strategies_evaluated']
        self.assertGreater(len(strategies), 1)
    
    def test_create_parallel_groups(self):
        """Test parallel group creation"""
        complexities = self.engine.get_task_complexities()
        constraints = ResourceConstraints(
            max_cpu_cores=8,
            max_memory_gb=16.0,
            max_disk_gb=100.0,
            max_parallel_tasks=4,
            priority_weights={'high': 1.0, 'medium': 0.6, 'low': 0.3}
        )
        
        task_order = ["1", "3", "2", "4", "5"]
        parallel_groups = self.engine._create_parallel_groups(task_order, complexities, constraints)
        
        self.assertIsInstance(parallel_groups, list)
        self.assertGreater(len(parallel_groups), 0)
        
        # Check that all tasks are included
        all_tasks_in_groups = []
        for group in parallel_groups:
            all_tasks_in_groups.extend(group)
        
        self.assertEqual(set(all_tasks_in_groups), set(task_order))
        
        # Tasks with high parallelization potential should be grouped
        for group in parallel_groups:
            if "4" in group:  # Task 4 has high parallelization potential
                # Could be in a group with other parallelizable tasks
                pass
    
    def test_calculate_resource_allocation(self):
        """Test resource allocation calculation"""
        complexities = self.engine.get_task_complexities()
        constraints = ResourceConstraints(
            max_cpu_cores=8,
            max_memory_gb=16.0,
            max_disk_gb=100.0,
            max_parallel_tasks=4,
            priority_weights={'high': 1.0, 'medium': 0.6, 'low': 0.3}
        )
        
        task_order = ["1", "2", "3", "4", "5"]
        allocation = self.engine._calculate_resource_allocation(task_order, complexities, constraints)
        
        self.assertEqual(len(allocation), 5)
        
        for task_id, alloc in allocation.items():
            self.assertIn('cpu_cores', alloc)
            self.assertIn('memory_gb', alloc)
            self.assertIn('priority', alloc)
            self.assertIn('estimated_duration', alloc)
            self.assertIn('parallelization_potential', alloc)
            
            # Check constraints
            self.assertLessEqual(alloc['cpu_cores'], constraints.max_cpu_cores)
            self.assertLessEqual(alloc['memory_gb'], constraints.max_memory_gb)
    
    def test_calculate_efficiency_score(self):
        """Test efficiency score calculation"""
        complexities = self.engine.get_task_complexities()
        constraints = ResourceConstraints(
            max_cpu_cores=4,
            max_memory_gb=8.0,
            max_disk_gb=100.0,
            max_parallel_tasks=4,
            priority_weights={'high': 1.0, 'medium': 0.6, 'low': 0.3}
        )
        
        # Good task order (respects dependencies)
        good_order = ["1", "3", "2", "4", "5"]
        good_score = self.engine._calculate_efficiency_score(good_order, complexities, constraints)
        
        # Bad task order (violates dependencies)
        bad_order = ["2", "4", "5", "1", "3"]
        bad_score = self.engine._calculate_efficiency_score(bad_order, complexities, constraints)
        
        self.assertGreaterEqual(good_score, 0)
        self.assertLessEqual(good_score, 1)
        self.assertGreaterEqual(bad_score, 0)
        self.assertLessEqual(bad_score, 1)
        
        # Good order should have better efficiency
        self.assertGreaterEqual(good_score, bad_score)
    
    def test_identify_execution_bottlenecks(self):
        """Test bottleneck identification"""
        complexities = self.engine.get_task_complexities()
        constraints = ResourceConstraints(
            max_cpu_cores=2,
            max_memory_gb=4.0,
            max_disk_gb=50.0,
            max_parallel_tasks=2,
            priority_weights={'high': 1.0, 'medium': 0.6, 'low': 0.3}
        )
        
        task_order = ["1", "2", "3", "4", "5"]
        bottlenecks = self.engine._identify_execution_bottlenecks(task_order, complexities, constraints)
        
        self.assertIsInstance(bottlenecks, list)
        
        # Should identify exponential task as time bottleneck
        bottleneck_text = ' '.join(bottlenecks)
        if complexities["2"].estimated_runtime_seconds > 100:  # If exponential task is slow
            self.assertIn('bottleneck', bottleneck_text.lower())
    
    def test_generate_execution_script(self):
        """Test execution script generation"""
        plan = self.engine.optimize_execution_order(OptimizationStrategy.GREEDY_SHORTEST_FIRST)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            script_file = os.path.join(temp_dir, "test_execution.sh")
            generated_file = self.engine.generate_execution_script(plan, script_file)
            
            self.assertEqual(generated_file, script_file)
            self.assertTrue(os.path.exists(script_file))
            
            # Check script content
            with open(script_file, 'r') as f:
                content = f.read()
            
            self.assertIn('#!/bin/bash', content)
            self.assertIn('task-master set-status', content)
            self.assertIn(plan.strategy.value, content)
            
            # Check if file is executable
            self.assertTrue(os.access(script_file, os.X_OK))
    
    def test_save_optimization_report(self):
        """Test optimization report saving"""
        plan = self.engine.optimize_execution_order(OptimizationStrategy.ADAPTIVE_SCHEDULING)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_file = os.path.join(temp_dir, "test_report.json")
            saved_file = self.engine.save_optimization_report(plan, report_file)
            
            self.assertEqual(saved_file, report_file)
            self.assertTrue(os.path.exists(report_file))
            
            # Check report content
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            self.assertIn('timestamp', report)
            self.assertIn('execution_plan', report)
            self.assertIn('system_resources', report)
            self.assertIn('optimization_summary', report)
            self.assertIn('recommendations', report)
            
            # Check optimization summary
            summary = report['optimization_summary']
            self.assertEqual(summary['strategy_used'], plan.strategy.value)
            self.assertEqual(summary['total_tasks'], len(plan.task_order))
            self.assertEqual(summary['efficiency_score'], plan.efficiency_score)
    
    def test_generate_optimization_recommendations(self):
        """Test optimization recommendations generation"""
        # Create plan with low efficiency
        plan = ExecutionPlan(
            strategy=OptimizationStrategy.GREEDY_SHORTEST_FIRST,
            task_order=["1", "2", "3", "4", "5"],
            parallel_groups=[["1"], ["2"], ["3"], ["4"], ["5"]],  # Low parallelization
            resource_allocation={},
            estimated_total_time=7200,  # 2 hours
            efficiency_score=0.5,  # Low efficiency
            bottlenecks=["Time bottleneck: task 2"],
            optimization_metadata={}
        )
        
        recommendations = self.engine._generate_optimization_recommendations(plan)
        
        self.assertIsInstance(recommendations, list)
        
        # Should recommend addressing low efficiency
        rec_text = ' '.join(recommendations)
        self.assertIn('efficiency', rec_text.lower())
        
        # Should recommend addressing bottlenecks
        self.assertIn('bottleneck', rec_text.lower())
        
        # Should recommend parallelization improvements
        self.assertIn('parallel', rec_text.lower())
    
    def test_optimization_history(self):
        """Test optimization history tracking"""
        self.assertEqual(len(self.engine.optimization_history), 0)
        
        # Run multiple optimizations
        plan1 = self.engine.optimize_execution_order(OptimizationStrategy.GREEDY_SHORTEST_FIRST)
        plan2 = self.engine.optimize_execution_order(OptimizationStrategy.CRITICAL_PATH)
        
        self.assertEqual(len(self.engine.optimization_history), 2)
        self.assertEqual(self.engine.optimization_history[0], plan1)
        self.assertEqual(self.engine.optimization_history[1], plan2)


class TestOptimizationStrategy(unittest.TestCase):
    """Test OptimizationStrategy enum"""
    
    def test_strategy_values(self):
        """Test optimization strategy enum values"""
        self.assertEqual(OptimizationStrategy.GREEDY_SHORTEST_FIRST.value, "greedy_shortest")
        self.assertEqual(OptimizationStrategy.GREEDY_RESOURCE_AWARE.value, "greedy_resource")
        self.assertEqual(OptimizationStrategy.DYNAMIC_PROGRAMMING.value, "dynamic_programming")
        self.assertEqual(OptimizationStrategy.MACHINE_LEARNING.value, "machine_learning")
        self.assertEqual(OptimizationStrategy.CRITICAL_PATH.value, "critical_path")
        self.assertEqual(OptimizationStrategy.ADAPTIVE_SCHEDULING.value, "adaptive_scheduling")


class TestExecutionPlan(unittest.TestCase):
    """Test ExecutionPlan dataclass"""
    
    def test_execution_plan_creation(self):
        """Test ExecutionPlan object creation"""
        plan = ExecutionPlan(
            strategy=OptimizationStrategy.GREEDY_SHORTEST_FIRST,
            task_order=["1", "2", "3"],
            parallel_groups=[["1"], ["2", "3"]],
            resource_allocation={"1": {"cpu_cores": 1, "memory_gb": 0.5}},
            estimated_total_time=300.0,
            efficiency_score=0.85,
            bottlenecks=["Memory bottleneck"],
            optimization_metadata={"test": "data"}
        )
        
        self.assertEqual(plan.strategy, OptimizationStrategy.GREEDY_SHORTEST_FIRST)
        self.assertEqual(plan.task_order, ["1", "2", "3"])
        self.assertEqual(plan.parallel_groups, [["1"], ["2", "3"]])
        self.assertEqual(plan.resource_allocation, {"1": {"cpu_cores": 1, "memory_gb": 0.5}})
        self.assertEqual(plan.estimated_total_time, 300.0)
        self.assertEqual(plan.efficiency_score, 0.85)
        self.assertEqual(plan.bottlenecks, ["Memory bottleneck"])
        self.assertEqual(plan.optimization_metadata, {"test": "data"})


class TestResourceConstraints(unittest.TestCase):
    """Test ResourceConstraints dataclass"""
    
    def test_resource_constraints_creation(self):
        """Test ResourceConstraints object creation"""
        constraints = ResourceConstraints(
            max_cpu_cores=8,
            max_memory_gb=16.0,
            max_disk_gb=500.0,
            max_parallel_tasks=4,
            priority_weights={'high': 1.0, 'medium': 0.6, 'low': 0.3}
        )
        
        self.assertEqual(constraints.max_cpu_cores, 8)
        self.assertEqual(constraints.max_memory_gb, 16.0)
        self.assertEqual(constraints.max_disk_gb, 500.0)
        self.assertEqual(constraints.max_parallel_tasks, 4)
        self.assertEqual(constraints.priority_weights, {'high': 1.0, 'medium': 0.6, 'low': 0.3})


class TestOptimizationEngineIntegration(unittest.TestCase):
    """Integration tests for OptimizationEngine with real scenarios"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create realistic task scenario
        self.realistic_tasks = {
            "tags": {
                "master": {
                    "tasks": [
                        {
                            "id": "1",
                            "title": "Setup database",
                            "description": "Initialize database schema",
                            "details": "Linear database setup file operations",
                            "dependencies": []
                        },
                        {
                            "id": "2",
                            "title": "Data migration",
                            "description": "Migrate user data",
                            "details": "Large dataset memory intensive cache store processing",
                            "dependencies": ["1"]
                        },
                        {
                            "id": "3",
                            "title": "API setup",
                            "description": "Configure REST API",
                            "details": "Network api configuration parallel independent",
                            "dependencies": ["1"]
                        },
                        {
                            "id": "4",
                            "title": "Frontend build",
                            "description": "Build React application",
                            "details": "Parallel concurrent build process independent",
                            "dependencies": []
                        },
                        {
                            "id": "5",
                            "title": "Integration tests",
                            "description": "Run full test suite",
                            "details": "Complex algorithm compute analysis testing",
                            "dependencies": ["2", "3", "4"]
                        },
                        {
                            "id": "6",
                            "title": "Deployment",
                            "description": "Deploy to production",
                            "details": "Network deployment api operations",
                            "dependencies": ["5"]
                        }
                    ]
                }
            }
        }
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.realistic_tasks, self.temp_file)
        self.temp_file.close()
        
        # Initialize analyzer and engine
        self.analyzer = TaskComplexityAnalyzer(self.temp_file.name)
        self.engine = OptimizationEngine(self.analyzer, self.temp_file.name)
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        os.unlink(self.temp_file.name)
    
    def test_realistic_optimization_scenario(self):
        """Test optimization with realistic development scenario"""
        # Test different strategies on realistic data
        strategies = [
            OptimizationStrategy.GREEDY_SHORTEST_FIRST,
            OptimizationStrategy.GREEDY_RESOURCE_AWARE,
            OptimizationStrategy.CRITICAL_PATH,
            OptimizationStrategy.ADAPTIVE_SCHEDULING
        ]
        
        plans = {}
        for strategy in strategies:
            plan = self.engine.optimize_execution_order(strategy)
            plans[strategy] = plan
            
            # Verify basic plan validity
            self.assertEqual(len(plan.task_order), 6)
            self.assertGreaterEqual(plan.efficiency_score, 0)
            self.assertLessEqual(plan.efficiency_score, 1)
        
        # Adaptive should select best strategy
        adaptive_plan = plans[OptimizationStrategy.ADAPTIVE_SCHEDULING]
        self.assertIn('strategies_evaluated', adaptive_plan.optimization_metadata)
        
        # Check dependency constraints are satisfied
        for strategy, plan in plans.items():
            task_positions = {task: i for i, task in enumerate(plan.task_order)}
            
            # Task 2 depends on 1
            self.assertLess(task_positions["1"], task_positions["2"])
            # Task 3 depends on 1
            self.assertLess(task_positions["1"], task_positions["3"])
            # Task 5 depends on 2, 3, 4
            self.assertLess(task_positions["2"], task_positions["5"])
            self.assertLess(task_positions["3"], task_positions["5"])
            self.assertLess(task_positions["4"], task_positions["5"])
            # Task 6 depends on 5
            self.assertLess(task_positions["5"], task_positions["6"])
    
    def test_resource_constrained_optimization(self):
        """Test optimization under tight resource constraints"""
        tight_constraints = ResourceConstraints(
            max_cpu_cores=2,
            max_memory_gb=4.0,
            max_disk_gb=20.0,
            max_parallel_tasks=2,
            priority_weights={'high': 1.0, 'medium': 0.6, 'low': 0.3}
        )
        
        plan = self.engine.optimize_execution_order(
            OptimizationStrategy.GREEDY_RESOURCE_AWARE, tight_constraints
        )
        
        # Should still produce valid plan
        self.assertEqual(len(plan.task_order), 6)
        
        # Resource allocation should respect constraints
        for task_id, allocation in plan.resource_allocation.items():
            self.assertLessEqual(allocation['cpu_cores'], tight_constraints.max_cpu_cores)
            self.assertLessEqual(allocation['memory_gb'], tight_constraints.max_memory_gb)
        
        # Parallel groups should respect parallel task limit
        for group in plan.parallel_groups:
            self.assertLessEqual(len(group), tight_constraints.max_parallel_tasks)


if __name__ == '__main__':
    unittest.main()