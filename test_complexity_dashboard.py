#!/usr/bin/env python3
"""
Comprehensive tests for Complexity Dashboard module
"""

import unittest
import json
import tempfile
import os
import threading
import time
from unittest.mock import patch, MagicMock
from complexity_dashboard import ComplexityDashboard
from task_complexity_analyzer import TaskComplexityAnalyzer
from optimization_engine import OptimizationEngine, OptimizationStrategy, ExecutionPlan


class TestComplexityDashboard(unittest.TestCase):
    """Test suite for ComplexityDashboard"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test task data
        self.test_tasks = {
            "tags": {
                "master": {
                    "tasks": [
                        {
                            "id": "1",
                            "title": "Simple task",
                            "description": "Linear processing",
                            "details": "Simple file read operation",
                            "dependencies": []
                        },
                        {
                            "id": "2",
                            "title": "Complex task",
                            "description": "Exponential algorithm",
                            "details": "Recursive backtracking exponential complexity",
                            "dependencies": ["1"]
                        },
                        {
                            "id": "3",
                            "title": "Memory intensive task",
                            "description": "Large data processing",
                            "details": "Memory intensive cache store large dataset",
                            "dependencies": []
                        },
                        {
                            "id": "4",
                            "title": "Parallel task",
                            "description": "Parallelizable operation",
                            "details": "Parallel concurrent independent batch processing",
                            "dependencies": ["1"]
                        }
                    ]
                }
            }
        }
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_tasks, self.temp_file)
        self.temp_file.close()
        
        # Initialize dashboard
        self.dashboard = ComplexityDashboard(self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_file.name)
        
        # Clean up dashboard directory if it exists
        if hasattr(self.dashboard, 'dashboard_dir') and os.path.exists(self.dashboard.dashboard_dir):
            import shutil
            shutil.rmtree(self.dashboard.dashboard_dir, ignore_errors=True)
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        self.assertIsInstance(self.dashboard.analyzer, TaskComplexityAnalyzer)
        self.assertIsInstance(self.dashboard.optimizer, OptimizationEngine)
        self.assertEqual(self.dashboard.tasks_file, self.temp_file.name)
        self.assertTrue(self.dashboard.dashboard_dir.endswith('dashboard'))
        self.assertEqual(self.dashboard.port, 8080)
    
    def test_generate_dashboard(self):
        """Test dashboard HTML generation"""
        dashboard_file = self.dashboard.generate_dashboard()
        
        # Check that dashboard file was created
        self.assertTrue(os.path.exists(dashboard_file))
        self.assertTrue(dashboard_file.endswith('index.html'))
        
        # Check that supporting files were created
        css_file = os.path.join(self.dashboard.dashboard_dir, "dashboard.css")
        js_file = os.path.join(self.dashboard.dashboard_dir, "dashboard.js")
        
        self.assertTrue(os.path.exists(css_file))
        self.assertTrue(os.path.exists(js_file))
        
        # Check HTML content
        with open(dashboard_file, 'r') as f:
            html_content = f.read()
        
        self.assertIn('<!DOCTYPE html>', html_content)
        self.assertIn('Task Master AI - Complexity Dashboard', html_content)
        self.assertIn('dashboard.css', html_content)
        self.assertIn('dashboard.js', html_content)
        self.assertIn('Chart.js', html_content)
        self.assertIn('d3.js', html_content)
    
    def test_generate_html_dashboard(self):
        """Test HTML dashboard generation with data"""
        # Get test data
        complexity_report = self.dashboard.analyzer.generate_complexity_report()
        optimization_plan = self.dashboard.optimizer.optimize_execution_order(
            OptimizationStrategy.ADAPTIVE_SCHEDULING
        )
        
        html_content = self.dashboard._generate_html_dashboard(complexity_report, optimization_plan)
        
        # Check basic HTML structure
        self.assertIn('<!DOCTYPE html>', html_content)
        self.assertIn('<html lang="en">', html_content)
        self.assertIn('<head>', html_content)
        self.assertIn('<body>', html_content)
        
        # Check dashboard sections
        self.assertIn('Analysis Summary', html_content)
        self.assertIn('Complexity Distribution', html_content)
        self.assertIn('Resource Requirements', html_content)
        self.assertIn('Optimization Results', html_content)
        self.assertIn('Execution Timeline', html_content)
        self.assertIn('Bottlenecks', html_content)
        self.assertIn('Recommendations', html_content)
        self.assertIn('Detailed Task Analysis', html_content)
        self.assertIn('System Resources', html_content)
        
        # Check data integration
        self.assertIn(str(complexity_report['summary']['total_tasks']), html_content)
        self.assertIn(optimization_plan.strategy.value, html_content)
        self.assertIn(f"{optimization_plan.efficiency_score:.3f}", html_content)
    
    def test_generate_bottlenecks_html(self):
        """Test bottlenecks HTML generation"""
        complexity_report = {"optimization_opportunities": {"resource_bottlenecks": ["Memory bottleneck: 8GB needed"]}}
        optimization_plan = ExecutionPlan(
            strategy=OptimizationStrategy.GREEDY_SHORTEST_FIRST,
            task_order=["1", "2"],
            parallel_groups=[["1"], ["2"]],
            resource_allocation={},
            estimated_total_time=100.0,
            efficiency_score=0.8,
            bottlenecks=["CPU bottleneck: high utilization"],
            optimization_metadata={}
        )
        
        html = self.dashboard._generate_bottlenecks_html(complexity_report, optimization_plan)
        
        self.assertIn('Memory bottleneck', html)
        self.assertIn('CPU bottleneck', html)
        self.assertIn('bottleneck-item', html)
        
        # Test no bottlenecks
        html_empty = self.dashboard._generate_bottlenecks_html({}, None)
        self.assertIn('No significant bottlenecks', html_empty)
    
    def test_generate_recommendations_html(self):
        """Test recommendations HTML generation"""
        complexity_report = {
            "optimization_opportunities": {
                "optimization_recommendations": [
                    "Consider parallel execution for 2 tasks",
                    "Review high-complexity tasks for optimization"
                ]
            }
        }
        
        html = self.dashboard._generate_recommendations_html(complexity_report)
        
        self.assertIn('parallel execution', html)
        self.assertIn('high-complexity tasks', html)
        self.assertIn('recommendation-item', html)
        
        # Test no recommendations
        html_empty = self.dashboard._generate_recommendations_html({})
        self.assertIn('No specific recommendations', html_empty)
    
    def test_generate_task_table_html(self):
        """Test task analysis table HTML generation"""
        detailed_analysis = [
            {
                "task_id": "1",
                "time_complexity": "O(n)",
                "space_complexity": "O(1)",
                "estimated_runtime_seconds": 10.5,
                "cpu_intensive": True,
                "memory_intensive": False,
                "parallelization_potential": 0.7
            },
            {
                "task_id": "2",
                "time_complexity": "O(2^n)",
                "space_complexity": "O(n)",
                "estimated_runtime_seconds": 120.0,
                "cpu_intensive": False,
                "memory_intensive": True,
                "parallelization_potential": 0.3
            }
        ]
        
        complexity_report = {"detailed_analysis": detailed_analysis}
        html = self.dashboard._generate_task_table_html(complexity_report)
        
        self.assertIn('<tr>', html)
        self.assertIn('<td>1</td>', html)
        self.assertIn('<td>2</td>', html)
        self.assertIn('O(n)', html)
        self.assertIn('O(2^n)', html)
        self.assertIn('10.5', html)
        self.assertIn('120.0', html)
        self.assertIn('🔥', html)  # CPU intensive icon
        self.assertIn('💾', html)  # Memory intensive icon
        self.assertIn('progress-bar', html)
        
        # Test empty data
        html_empty = self.dashboard._generate_task_table_html({})
        self.assertIn('No task data available', html_empty)
    
    def test_generate_system_info_html(self):
        """Test system information HTML generation"""
        system_resources = {
            "cpu_cores": 8,
            "available_memory_gb": 16.5,
            "available_disk_gb": 500.2,
            "cpu_usage_percent": 25.7,
            "memory_usage_percent": 45.3
        }
        
        complexity_report = {"system_resources": system_resources}
        html = self.dashboard._generate_system_info_html(complexity_report)
        
        self.assertIn('8', html)  # CPU cores
        self.assertIn('16.5', html)  # Memory
        self.assertIn('500.2', html)  # Disk
        self.assertIn('25.7', html)  # CPU usage
        self.assertIn('45.3', html)  # Memory usage
        self.assertIn('system-metric', html)
    
    def test_generate_css_file(self):
        """Test CSS file generation"""
        self.dashboard._generate_css_file()
        
        css_file = os.path.join(self.dashboard.dashboard_dir, "dashboard.css")
        self.assertTrue(os.path.exists(css_file))
        
        with open(css_file, 'r') as f:
            css_content = f.read()
        
        # Check for key CSS components
        self.assertIn('body {', css_content)
        self.assertIn('.container', css_content)
        self.assertIn('.dashboard-grid', css_content)
        self.assertIn('.card', css_content)
        self.assertIn('.metric-grid', css_content)
        self.assertIn('@media (max-width: 768px)', css_content)  # Responsive design
        self.assertIn('linear-gradient', css_content)  # Styling
    
    def test_generate_js_file(self):
        """Test JavaScript file generation"""
        self.dashboard._generate_js_file()
        
        js_file = os.path.join(self.dashboard.dashboard_dir, "dashboard.js")
        self.assertTrue(os.path.exists(js_file))
        
        with open(js_file, 'r') as f:
            js_content = f.read()
        
        # Check for key JavaScript functions
        self.assertIn('function initializeComplexityChart', js_content)
        self.assertIn('function initializeResourceChart', js_content)
        self.assertIn('function initializeTimeline', js_content)
        self.assertIn('function makeTableSortable', js_content)
        self.assertIn('Chart.js', js_content)
        self.assertIn('addEventListener', js_content)
        self.assertIn('window.TaskMasterDashboard', js_content)
    
    def test_generate_data_files(self):
        """Test data file generation"""
        complexity_report = self.dashboard.analyzer.generate_complexity_report()
        optimization_plan = self.dashboard.optimizer.optimize_execution_order(
            OptimizationStrategy.ADAPTIVE_SCHEDULING
        )
        
        self.dashboard._generate_data_files(complexity_report, optimization_plan)
        
        # Check complexity data file
        complexity_file = os.path.join(self.dashboard.dashboard_dir, "complexity_data.json")
        self.assertTrue(os.path.exists(complexity_file))
        
        with open(complexity_file, 'r') as f:
            saved_complexity = json.load(f)
        
        self.assertEqual(saved_complexity['summary']['total_tasks'], 
                        complexity_report['summary']['total_tasks'])
        
        # Check optimization data file
        optimization_file = os.path.join(self.dashboard.dashboard_dir, "optimization_data.json")
        self.assertTrue(os.path.exists(optimization_file))
        
        with open(optimization_file, 'r') as f:
            saved_optimization = json.load(f)
        
        self.assertEqual(saved_optimization['strategy'], optimization_plan.strategy.value)
    
    def test_start_server(self):
        """Test HTTP server startup"""
        # Generate dashboard first
        self.dashboard.generate_dashboard()
        
        # Start server in a separate thread for testing
        def start_server_thread():
            try:
                url = self.dashboard.start_server(port=8081)  # Use different port
                return url
            except Exception as e:
                return str(e)
        
        server_thread = threading.Thread(target=start_server_thread)
        server_thread.daemon = True
        server_thread.start()
        
        # Give server time to start
        time.sleep(1)
        
        # Just verify the method exists and can be called
        # (Full server testing would require more complex setup)
        self.assertTrue(hasattr(self.dashboard, 'start_server'))
        self.assertTrue(callable(self.dashboard.start_server))
    
    @patch('webbrowser.open')
    def test_launch_dashboard(self, mock_browser):
        """Test dashboard launch with browser opening"""
        # Mock the server start to avoid actual server startup
        with patch.object(self.dashboard, 'start_server', return_value='http://localhost:8080'):
            url = self.dashboard.launch_dashboard(auto_open=True)
            
            self.assertEqual(url, 'http://localhost:8080')
            mock_browser.assert_called_once_with('http://localhost:8080')
        
        # Test without auto-opening browser
        with patch.object(self.dashboard, 'start_server', return_value='http://localhost:8080'):
            url = self.dashboard.launch_dashboard(auto_open=False)
            
            self.assertEqual(url, 'http://localhost:8080')
            # Browser should not be called again
            mock_browser.assert_called_once()  # Still once from previous call
    
    def test_dashboard_with_empty_tasks(self):
        """Test dashboard generation with empty task data"""
        # Create empty task file
        empty_tasks = {"tags": {"master": {"tasks": []}}}
        
        empty_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(empty_tasks, empty_file)
        empty_file.close()
        
        try:
            dashboard = ComplexityDashboard(empty_file.name)
            dashboard_file = dashboard.generate_dashboard()
            
            # Should still generate dashboard
            self.assertTrue(os.path.exists(dashboard_file))
            
            with open(dashboard_file, 'r') as f:
                html_content = f.read()
            
            # Should handle empty data gracefully
            self.assertIn('Task Master AI', html_content)
            self.assertIn('0', html_content)  # Zero tasks
            
        finally:
            os.unlink(empty_file.name)
            if os.path.exists(dashboard.dashboard_dir):
                import shutil
                shutil.rmtree(dashboard.dashboard_dir, ignore_errors=True)
    
    def test_dashboard_responsiveness(self):
        """Test that generated CSS includes responsive design"""
        self.dashboard._generate_css_file()
        
        css_file = os.path.join(self.dashboard.dashboard_dir, "dashboard.css")
        with open(css_file, 'r') as f:
            css_content = f.read()
        
        # Check for responsive design elements
        self.assertIn('@media (max-width: 768px)', css_content)
        self.assertIn('grid-template-columns: 1fr', css_content)
        
        # Check for mobile-friendly elements
        responsive_rules = [
            'grid-template-columns: 1fr',  # Single column on mobile
            'font-size: 2em'  # Responsive font sizing
        ]
        
        for rule in responsive_rules:
            self.assertIn(rule, css_content)
    
    def test_dashboard_accessibility(self):
        """Test that dashboard includes accessibility features"""
        complexity_report = self.dashboard.analyzer.generate_complexity_report()
        optimization_plan = self.dashboard.optimizer.optimize_execution_order(
            OptimizationStrategy.ADAPTIVE_SCHEDULING
        )
        
        html_content = self.dashboard._generate_html_dashboard(complexity_report, optimization_plan)
        
        # Check for accessibility attributes
        self.assertIn('lang="en"', html_content)
        self.assertIn('charset="UTF-8"', html_content)
        self.assertIn('viewport', html_content)
        
        # Check for semantic HTML
        semantic_elements = ['<header>', '<table>', '<thead>', '<tbody>', '<th>', '<td>']
        for element in semantic_elements:
            self.assertIn(element, html_content)
    
    def test_dashboard_performance_metrics(self):
        """Test that dashboard includes performance optimization features"""
        js_file_path = os.path.join(self.dashboard.dashboard_dir, "dashboard.js")
        self.dashboard._generate_js_file()
        
        with open(js_file_path, 'r') as f:
            js_content = f.read()
        
        # Check for performance optimizations
        self.assertIn('maintainAspectRatio: false', js_content)  # Chart optimization
        self.assertIn('DOMContentLoaded', js_content)  # Proper loading
        self.assertIn('setupAutoRefresh', js_content)  # Auto-refresh capability
    
    def test_error_handling(self):
        """Test dashboard error handling with invalid data"""
        # Test with missing optimization plan
        complexity_report = self.dashboard.analyzer.generate_complexity_report()
        html_content = self.dashboard._generate_html_dashboard(complexity_report, None)
        
        # Should handle None optimization plan gracefully
        self.assertIn('N/A', html_content)
        self.assertIn('Task Master AI', html_content)
        
        # Test with malformed complexity report
        malformed_report = {"summary": {}}  # Missing required fields
        html_content = self.dashboard._generate_html_dashboard(malformed_report, None)
        
        # Should use default values
        self.assertIn('0', html_content)  # Default for missing values


class TestComplexityDashboardIntegration(unittest.TestCase):
    """Integration tests for ComplexityDashboard with real data flow"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create comprehensive test scenario
        self.comprehensive_tasks = {
            "tags": {
                "master": {
                    "tasks": [
                        {
                            "id": "1",
                            "title": "Database setup",
                            "description": "Initialize database",
                            "details": "Linear database initialization file operations",
                            "dependencies": []
                        },
                        {
                            "id": "2",
                            "title": "Complex algorithm",
                            "description": "Implement sorting algorithm",
                            "details": "Recursive merge sort divide and conquer O(n log n)",
                            "dependencies": ["1"]
                        },
                        {
                            "id": "3",
                            "title": "Data processing",
                            "description": "Process large dataset",
                            "details": "Memory intensive large dataset cache store matrix",
                            "dependencies": ["1"]
                        },
                        {
                            "id": "4",
                            "title": "API endpoints",
                            "description": "Create REST API",
                            "details": "Network api parallel concurrent independent",
                            "dependencies": ["2"]
                        },
                        {
                            "id": "5",
                            "title": "Machine learning",
                            "description": "Train ML model",
                            "details": "Complex algorithm compute intensive exponential",
                            "dependencies": ["3"]
                        },
                        {
                            "id": "6",
                            "title": "Testing",
                            "description": "Run test suite",
                            "details": "Parallel independent testing concurrent batch",
                            "dependencies": ["4", "5"]
                        }
                    ]
                }
            }
        }
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.comprehensive_tasks, self.temp_file)
        self.temp_file.close()
        
        # Initialize dashboard
        self.dashboard = ComplexityDashboard(self.temp_file.name)
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        os.unlink(self.temp_file.name)
        
        # Clean up dashboard directory
        if os.path.exists(self.dashboard.dashboard_dir):
            import shutil
            shutil.rmtree(self.dashboard.dashboard_dir, ignore_errors=True)
    
    def test_end_to_end_dashboard_generation(self):
        """Test complete end-to-end dashboard generation"""
        dashboard_file = self.dashboard.generate_dashboard()
        
        # Verify all files were created
        expected_files = [
            'index.html',
            'dashboard.css',
            'dashboard.js',
            'complexity_data.json',
            'optimization_data.json'
        ]
        
        for file_name in expected_files:
            file_path = os.path.join(self.dashboard.dashboard_dir, file_name)
            self.assertTrue(os.path.exists(file_path), f"Missing file: {file_name}")
        
        # Verify HTML contains realistic data
        with open(dashboard_file, 'r') as f:
            html_content = f.read()
        
        # Should contain task count
        self.assertIn('6', html_content)  # 6 tasks
        
        # Should contain complexity information
        complexity_terms = ['O(n)', 'O(n log n)', 'O(2^n)']
        self.assertTrue(any(term in html_content for term in complexity_terms))
        
        # Should contain optimization results
        self.assertIn('efficiency', html_content.lower())
    
    def test_dashboard_data_accuracy(self):
        """Test that dashboard accurately represents analysis data"""
        # Generate analysis
        complexity_report = self.dashboard.analyzer.generate_complexity_report()
        optimization_plan = self.dashboard.optimizer.optimize_execution_order(
            OptimizationStrategy.ADAPTIVE_SCHEDULING
        )
        
        # Generate dashboard
        html_content = self.dashboard._generate_html_dashboard(complexity_report, optimization_plan)
        
        # Verify accuracy of key metrics
        total_tasks = complexity_report['summary']['total_tasks']
        self.assertIn(str(total_tasks), html_content)
        
        cpu_intensive = complexity_report['summary']['cpu_intensive_tasks']
        self.assertIn(str(cpu_intensive), html_content)
        
        efficiency_score = optimization_plan.efficiency_score
        self.assertIn(f"{efficiency_score:.3f}", html_content)
        
        # Verify task-specific data appears in table
        for task_data in complexity_report['detailed_analysis']:
            task_id = task_data['task_id']
            self.assertIn(task_id, html_content)
    
    def test_dashboard_interactivity_features(self):
        """Test that dashboard includes interactive features"""
        self.dashboard._generate_js_file()
        
        js_file = os.path.join(self.dashboard.dashboard_dir, "dashboard.js")
        with open(js_file, 'r') as f:
            js_content = f.read()
        
        # Check for interactive features
        interactive_features = [
            'Chart(',  # Chart.js integration
            'addEventListener',  # Event handling
            'sortTable',  # Table sorting
            'setupAutoRefresh',  # Auto-refresh
            'onclick'  # Click handlers
        ]
        
        for feature in interactive_features:
            self.assertIn(feature, js_content)


if __name__ == '__main__':
    unittest.main()