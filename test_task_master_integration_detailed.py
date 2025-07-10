#!/usr/bin/env python3
"""
Detailed Task Master AI Integration Tests

This test validates the Task Master AI components that are already implemented
and working correctly in the project.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Import existing Task Master components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from task_complexity_analyzer import TaskComplexityAnalyzer, ComplexityClass
    from optimization_engine import OptimizationEngine, OptimizationStrategy
    from complexity_dashboard import ComplexityDashboard
    TASK_MASTER_AVAILABLE = True
except ImportError as e:
    print(f"Task Master components not available: {e}")
    TASK_MASTER_AVAILABLE = False


class TestTaskMasterIntegrationDetailed(unittest.TestCase):
    """Detailed tests for existing Task Master AI integration"""
    
    def setUp(self):
        """Set up test environment"""
        if not TASK_MASTER_AVAILABLE:
            self.skipTest("Task Master AI components not available")
            
        self.test_dir = tempfile.mkdtemp()
        self.tasks_file = os.path.join(self.test_dir, "tasks.json")
        
        # Create comprehensive test tasks
        test_tasks = {
            "tags": {
                "master": {
                    "tasks": [
                        {
                            "id": "1",
                            "title": "Database setup",
                            "description": "Initialize PostgreSQL database",
                            "details": "Linear database initialization with file operations",
                            "dependencies": []
                        },
                        {
                            "id": "2", 
                            "title": "Authentication system",
                            "description": "JWT-based authentication",
                            "details": "Complex recursive algorithm with exponential token generation",
                            "dependencies": ["1"]
                        },
                        {
                            "id": "3",
                            "title": "Data processing pipeline", 
                            "description": "ETL pipeline for analytics",
                            "details": "Memory intensive large dataset processing with cache operations",
                            "dependencies": ["1"]
                        },
                        {
                            "id": "4",
                            "title": "API endpoints",
                            "description": "REST API development",
                            "details": "Parallel concurrent independent API development with network operations",
                            "dependencies": ["2"]
                        },
                        {
                            "id": "5",
                            "title": "Machine learning model",
                            "description": "Recommendation engine",
                            "details": "Complex algorithm compute intensive matrix operations quadratic",
                            "dependencies": ["3"]
                        }
                    ]
                }
            }
        }
        
        with open(self.tasks_file, 'w') as f:
            json.dump(test_tasks, f)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_task_complexity_analyzer_comprehensive(self):
        """Test comprehensive task complexity analysis functionality"""
        print("\n🧪 DETAILED TEST: Task Complexity Analyzer")
        
        with patch('psutil.cpu_count', return_value=8), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.cpu_percent', return_value=25.0):
            
            # Mock system resources
            mock_memory.return_value.available = 8589934592
            mock_memory.return_value.percent = 45.0
            mock_disk.return_value.free = 107374182400
            
            analyzer = TaskComplexityAnalyzer(self.tasks_file)
            
            # Test individual task analysis
            complexities = analyzer.analyze_all_tasks()
            self.assertEqual(len(complexities), 5)
            
            # Verify specific complexity classifications
            complexity_map = {c.task_id: c for c in complexities}
            
            # Database setup should be linear
            self.assertEqual(complexity_map["1"].time_complexity, ComplexityClass.LINEAR)
            
            # Authentication with recursive algorithm should be exponential
            self.assertEqual(complexity_map["2"].time_complexity, ComplexityClass.EXPONENTIAL)
            
            # ML model with matrix operations should be quadratic
            self.assertEqual(complexity_map["5"].time_complexity, ComplexityClass.QUADRATIC)
            
            # Test resource intensiveness
            self.assertTrue(complexity_map["3"].memory_intensive)  # Large dataset processing
            self.assertTrue(complexity_map["5"].cpu_intensive)     # Complex algorithm
            self.assertTrue(complexity_map["4"].network_dependent) # API operations
            
            # Test parallelization potential
            self.assertGreater(complexity_map["4"].parallelization_potential, 0.7)  # Parallel API development
            
            print(f"✅ Analyzed {len(complexities)} tasks with correct complexity classification")
            
            # Test comprehensive report generation
            report = analyzer.generate_complexity_report()
            
            # Verify report structure
            required_sections = [
                'analysis_timestamp', 'system_resources', 'summary',
                'complexity_distribution', 'resource_requirements',
                'optimization_opportunities', 'detailed_analysis'
            ]
            
            for section in required_sections:
                self.assertIn(section, report)
            
            # Verify summary statistics
            summary = report['summary']
            self.assertEqual(summary['total_tasks'], 5)
            self.assertGreater(summary['cpu_intensive_tasks'], 0)
            self.assertGreater(summary['memory_intensive_tasks'], 0)
            
            print("✅ Comprehensive complexity analysis report validated")
    
    def test_optimization_engine_comprehensive(self):
        """Test comprehensive optimization engine functionality"""
        print("\n🧪 DETAILED TEST: Optimization Engine")
        
        with patch('psutil.cpu_count', return_value=8), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.cpu_percent', return_value=25.0):
            
            # Mock system resources
            mock_memory.return_value.available = 8589934592
            mock_memory.return_value.percent = 45.0
            mock_disk.return_value.free = 107374182400
            
            analyzer = TaskComplexityAnalyzer(self.tasks_file)
            engine = OptimizationEngine(analyzer, self.tasks_file)
            
            # Test dependency analysis
            dependencies = engine.analyze_dependencies()
            
            # Verify dependencies are correctly identified
            self.assertEqual(dependencies["2"], ["1"])  # Auth depends on DB
            self.assertEqual(dependencies["3"], ["1"])  # Pipeline depends on DB  
            self.assertEqual(dependencies["4"], ["2"])  # API depends on Auth
            self.assertEqual(dependencies["5"], ["3"])  # ML depends on Pipeline
            
            print("✅ Dependency analysis validated")
            
            # Test all optimization strategies
            strategies_to_test = [
                OptimizationStrategy.GREEDY_SHORTEST_FIRST,
                OptimizationStrategy.GREEDY_RESOURCE_AWARE, 
                OptimizationStrategy.CRITICAL_PATH,
                OptimizationStrategy.ADAPTIVE_SCHEDULING
            ]
            
            optimization_results = {}
            
            for strategy in strategies_to_test:
                plan = engine.optimize_execution_order(strategy)
                
                # Verify plan structure
                self.assertEqual(len(plan.task_order), 5)
                self.assertGreater(plan.efficiency_score, 0)
                self.assertLessEqual(plan.efficiency_score, 1.0)
                self.assertGreater(plan.estimated_total_time, 0)
                self.assertGreater(len(plan.parallel_groups), 0)
                
                # Verify dependency constraints are satisfied
                task_positions = {task: i for i, task in enumerate(plan.task_order)}
                
                # Task 1 should come before tasks 2 and 3
                self.assertLess(task_positions["1"], task_positions["2"])
                self.assertLess(task_positions["1"], task_positions["3"])
                
                # Task 2 should come before task 4
                self.assertLess(task_positions["2"], task_positions["4"])
                
                # Task 3 should come before task 5
                self.assertLess(task_positions["3"], task_positions["5"])
                
                optimization_results[strategy.value] = plan.efficiency_score
                
                print(f"✅ {strategy.value} optimization validated (efficiency: {plan.efficiency_score:.3f})")
            
            # Test adaptive scheduling selects reasonable strategy
            adaptive_plan = engine.optimize_execution_order(OptimizationStrategy.ADAPTIVE_SCHEDULING)
            self.assertGreater(adaptive_plan.efficiency_score, 0.5)
            
            print("✅ All optimization strategies validated")
            
            # Test execution script generation
            script_file = os.path.join(self.test_dir, "test_execution.sh")
            generated_script = engine.generate_execution_script(adaptive_plan, script_file)
            
            self.assertTrue(os.path.exists(generated_script))
            
            with open(generated_script, 'r') as f:
                script_content = f.read()
            
            # Verify script contains proper elements
            self.assertIn("#!/bin/bash", script_content)
            self.assertIn("task-master set-status", script_content)
            self.assertIn(adaptive_plan.strategy.value, script_content)
            
            print("✅ Execution script generation validated")
    
    def test_complexity_dashboard_comprehensive(self):
        """Test comprehensive complexity dashboard functionality"""
        print("\n🧪 DETAILED TEST: Complexity Dashboard")
        
        with patch('psutil.cpu_count', return_value=8), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.cpu_percent', return_value=25.0):
            
            # Mock system resources
            mock_memory.return_value.available = 8589934592
            mock_memory.return_value.percent = 45.0
            mock_disk.return_value.free = 107374182400
            
            dashboard = ComplexityDashboard(self.tasks_file)
            
            # Test dashboard generation
            dashboard_file = dashboard.generate_dashboard()
            self.assertTrue(os.path.exists(dashboard_file))
            
            # Test HTML content
            with open(dashboard_file, 'r') as f:
                html_content = f.read()
            
            # Verify essential HTML elements
            required_elements = [
                'Task Master AI - Complexity Dashboard',
                'Complexity Distribution',
                'Resource Requirements', 
                'Optimization Results',
                'Execution Timeline',
                'Bottlenecks',
                'System Resources',
                'dashboard.css',
                'dashboard.js',
                'Chart.js',
                'D3.js'
            ]
            
            for element in required_elements:
                self.assertIn(element, html_content)
            
            print("✅ Dashboard HTML generation validated")
            
            # Test CSS file generation
            css_file = os.path.join(dashboard.dashboard_dir, "dashboard.css")
            self.assertTrue(os.path.exists(css_file))
            
            with open(css_file, 'r') as f:
                css_content = f.read()
            
            # Verify CSS contains styling
            self.assertIn(".dashboard-grid", css_content)
            self.assertIn(".card", css_content)
            self.assertIn("chart-card", css_content)
            
            print("✅ Dashboard CSS generation validated")
            
            # Test JavaScript file generation  
            js_file = os.path.join(dashboard.dashboard_dir, "dashboard.js")
            self.assertTrue(os.path.exists(js_file))
            
            with open(js_file, 'r') as f:
                js_content = f.read()
            
            # Verify JavaScript contains functions
            required_functions = [
                'initializeComplexityChart',
                'initializeResourceChart',
                'initializeTimeline',
                'makeTableSortable'
            ]
            
            for func in required_functions:
                self.assertIn(func, js_content)
            
            print("✅ Dashboard JavaScript generation validated")
            
            # Test data file generation
            complexity_file = os.path.join(dashboard.dashboard_dir, "complexity_data.json")
            optimization_file = os.path.join(dashboard.dashboard_dir, "optimization_data.json")
            
            self.assertTrue(os.path.exists(complexity_file))
            self.assertTrue(os.path.exists(optimization_file))
            
            # Verify data file contents
            with open(complexity_file, 'r') as f:
                complexity_data = json.load(f)
            
            self.assertIn("summary", complexity_data)
            self.assertIn("complexity_distribution", complexity_data)
            
            print("✅ Dashboard data file generation validated")
            
            # Clean up dashboard directory
            import shutil
            if os.path.exists(dashboard.dashboard_dir):
                shutil.rmtree(dashboard.dashboard_dir)
    
    def test_integration_end_to_end(self):
        """Test end-to-end integration of all Task Master components"""
        print("\n🧪 DETAILED TEST: End-to-End Integration")
        
        with patch('psutil.cpu_count', return_value=8), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.cpu_percent', return_value=25.0):
            
            # Mock system resources
            mock_memory.return_value.available = 8589934592
            mock_memory.return_value.percent = 45.0
            mock_disk.return_value.free = 107374182400
            
            # Test complete workflow
            analyzer = TaskComplexityAnalyzer(self.tasks_file)
            engine = OptimizationEngine(analyzer, self.tasks_file)
            dashboard = ComplexityDashboard(self.tasks_file)
            
            # Step 1: Analyze complexity
            complexity_report = analyzer.generate_complexity_report()
            self.assertEqual(complexity_report['summary']['total_tasks'], 5)
            
            # Step 2: Optimize execution
            optimization_plan = engine.optimize_execution_order(OptimizationStrategy.ADAPTIVE_SCHEDULING)
            self.assertEqual(len(optimization_plan.task_order), 5)
            
            # Step 3: Generate dashboard
            dashboard_file = dashboard.generate_dashboard()
            self.assertTrue(os.path.exists(dashboard_file))
            
            # Step 4: Generate execution script
            script_file = os.path.join(self.test_dir, "integrated_execution.sh")
            execution_script = engine.generate_execution_script(optimization_plan, script_file)
            self.assertTrue(os.path.exists(execution_script))
            
            # Verify integration produces coherent results
            with open(execution_script, 'r') as f:
                script_content = f.read()
            
            # Script should contain all tasks in optimized order
            for task_id in optimization_plan.task_order:
                self.assertIn(f"Task {task_id}", script_content)
            
            print("✅ End-to-end Task Master integration validated")
            
            # Clean up
            import shutil
            if os.path.exists(dashboard.dashboard_dir):
                shutil.rmtree(dashboard.dashboard_dir)


def run_detailed_integration_tests():
    """Run detailed Task Master AI integration tests"""
    
    print("🔬 RUNNING DETAILED TASK MASTER AI INTEGRATION TESTS")
    print("=" * 70)
    
    if not TASK_MASTER_AVAILABLE:
        print("❌ Task Master AI components not available - skipping tests")
        return
    
    # Create and run test suite
    test_suite = unittest.TestSuite()
    
    test_methods = [
        'test_task_complexity_analyzer_comprehensive',
        'test_optimization_engine_comprehensive', 
        'test_complexity_dashboard_comprehensive',
        'test_integration_end_to_end'
    ]
    
    for test_method in test_methods:
        test_suite.addTest(TestTaskMasterIntegrationDetailed(test_method))
    
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("DETAILED INTEGRATION TEST RESULTS")
    print("=" * 70)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success = total - failures - errors
    success_rate = (success / total * 100) if total > 0 else 0
    
    print(f"📊 Tests Run: {total}")
    print(f"✅ Successful: {success}")
    print(f"❌ Failures: {failures}")
    print(f"🚨 Errors: {errors}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎯 ASSESSMENT: Task Master AI integration is EXCELLENT")
    elif success_rate >= 80:
        print("⚡ ASSESSMENT: Task Master AI integration is GOOD") 
    else:
        print("⚠️  ASSESSMENT: Task Master AI integration needs improvement")
    
    return result


if __name__ == "__main__":
    run_detailed_integration_tests()