#!/usr/bin/env python3
"""
Comprehensive tests for Task Complexity Analyzer module
"""

import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from task_complexity_analyzer import (
    TaskComplexityAnalyzer, 
    TaskComplexity, 
    ComplexityClass, 
    SystemResources
)


class TestTaskComplexityAnalyzer(unittest.TestCase):
    """Test suite for TaskComplexityAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary test data
        self.test_tasks = {
            "tags": {
                "master": {
                    "tasks": [
                        {
                            "id": "1",
                            "title": "Simple file read task",
                            "description": "Read a single file",
                            "details": "Simple linear file reading operation",
                            "dependencies": []
                        },
                        {
                            "id": "2", 
                            "title": "Complex recursive algorithm",
                            "description": "Implement recursive backtracking",
                            "details": "Recursive permutation generation with exponential complexity",
                            "dependencies": ["1"]
                        },
                        {
                            "id": "3",
                            "title": "Matrix multiplication",
                            "description": "Multiply large matrices",
                            "details": "Nested loop matrix multiplication O(n²) complexity",
                            "dependencies": []
                        },
                        {
                            "id": "4",
                            "title": "Binary search implementation", 
                            "description": "Implement efficient search",
                            "details": "Logarithmic binary search algorithm",
                            "dependencies": []
                        }
                    ]
                }
            }
        }
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_tasks, self.temp_file)
        self.temp_file.close()
        
        # Initialize analyzer
        self.analyzer = TaskComplexityAnalyzer(self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_file.name)
    
    def test_load_tasks_valid_file(self):
        """Test loading tasks from valid JSON file"""
        self.assertEqual(self.analyzer.tasks_data, self.test_tasks)
        self.assertEqual(len(self.analyzer.tasks_data['tags']['master']['tasks']), 4)
    
    def test_load_tasks_missing_file(self):
        """Test loading tasks from non-existent file"""
        analyzer = TaskComplexityAnalyzer("nonexistent.json")
        expected = {"tags": {"master": {"tasks": []}}}
        self.assertEqual(analyzer.tasks_data, expected)
    
    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.cpu_percent')
    def test_get_system_resources(self, mock_cpu_percent, mock_disk, mock_memory, mock_cpu_count):
        """Test system resource detection"""
        # Mock system data
        mock_cpu_count.return_value = 8
        mock_memory.return_value = MagicMock(available=8589934592, percent=45.0)  # 8GB
        mock_disk.return_value = MagicMock(free=107374182400)  # 100GB
        mock_cpu_percent.return_value = 25.5
        
        analyzer = TaskComplexityAnalyzer(self.temp_file.name)
        resources = analyzer.system_resources
        
        self.assertEqual(resources.cpu_cores, 8)
        self.assertAlmostEqual(resources.available_memory_gb, 8.0, places=1)
        self.assertAlmostEqual(resources.available_disk_gb, 100.0, places=1)
        self.assertEqual(resources.cpu_usage_percent, 25.5)
        self.assertEqual(resources.memory_usage_percent, 45.0)
    
    def test_determine_time_complexity_exponential(self):
        """Test detection of exponential time complexity"""
        task_text = "recursive backtracking permutation exponential algorithm"
        task = {"id": "test"}
        
        complexity = self.analyzer._determine_time_complexity(task_text, task)
        self.assertEqual(complexity, ComplexityClass.EXPONENTIAL)
    
    def test_determine_time_complexity_quadratic(self):
        """Test detection of quadratic time complexity"""
        task_text = "nested loop matrix multiplication quadratic o(n²)"
        task = {"id": "test"}
        
        complexity = self.analyzer._determine_time_complexity(task_text, task)
        self.assertEqual(complexity, ComplexityClass.QUADRATIC)
    
    def test_determine_time_complexity_logarithmic(self):
        """Test detection of logarithmic time complexity"""
        task_text = "binary search logarithmic divide tree"
        task = {"id": "test"}
        
        complexity = self.analyzer._determine_time_complexity(task_text, task)
        self.assertEqual(complexity, ComplexityClass.LOGARITHMIC)
    
    def test_determine_time_complexity_linear(self):
        """Test detection of linear time complexity"""
        task_text = "iterate through array linear scan process each"
        task = {"id": "test"}
        
        complexity = self.analyzer._determine_time_complexity(task_text, task)
        self.assertEqual(complexity, ComplexityClass.LINEAR)
    
    def test_determine_space_complexity(self):
        """Test space complexity determination"""
        # High space complexity
        task_text = "cache store all memory intensive large data matrix"
        complexity = self.analyzer._determine_space_complexity(task_text, {})
        self.assertEqual(complexity, ComplexityClass.LINEAR)
        
        # Recursive space complexity
        task_text = "recursive call stack depth"
        complexity = self.analyzer._determine_space_complexity(task_text, {})
        self.assertEqual(complexity, ComplexityClass.LOGARITHMIC)
        
        # Default constant space
        task_text = "simple operation"
        complexity = self.analyzer._determine_space_complexity(task_text, {})
        self.assertEqual(complexity, ComplexityClass.CONSTANT)
    
    def test_count_io_operations(self):
        """Test I/O operation counting"""
        task_text = "read file write database save load api fetch"
        io_count = self.analyzer._count_io_operations(task_text)
        self.assertEqual(io_count, 6)
        
        # Minimum 1 I/O operation
        task_text = "simple calculation"
        io_count = self.analyzer._count_io_operations(task_text)
        self.assertEqual(io_count, 1)
    
    def test_assess_parallelization(self):
        """Test parallelization potential assessment"""
        # High parallelization
        task_text = "parallel concurrent independent batch async"
        potential = self.analyzer._assess_parallelization(task_text, {})
        self.assertEqual(potential, 0.9)
        
        # Low parallelization
        task_text = "sequential ordered dependent serial step-by-step"
        potential = self.analyzer._assess_parallelization(task_text, {})
        self.assertEqual(potential, 0.2)
        
        # Medium parallelization
        task_text = "process analyze generate"
        potential = self.analyzer._assess_parallelization(task_text, {})
        self.assertEqual(potential, 0.6)
    
    def test_resource_intensity_detection(self):
        """Test CPU and memory intensity detection"""
        # CPU intensive
        cpu_text = "compute calculate algorithm optimization complex"
        self.assertTrue(self.analyzer._is_cpu_intensive(cpu_text))
        
        # Memory intensive
        memory_text = "large cache store memory dataset matrix buffer"
        self.assertTrue(self.analyzer._is_memory_intensive(memory_text))
        
        # Network dependent
        network_text = "api network download upload fetch request web"
        self.assertTrue(self.analyzer._is_network_dependent(network_text))
    
    def test_analyze_task_complexity(self):
        """Test complete task complexity analysis"""
        task = self.test_tasks['tags']['master']['tasks'][1]  # Recursive task
        complexity = self.analyzer.analyze_task_complexity(task)
        
        self.assertEqual(complexity.task_id, "2")
        self.assertEqual(complexity.time_complexity, ComplexityClass.EXPONENTIAL)
        self.assertGreater(complexity.estimated_runtime_seconds, 0)
        self.assertIsInstance(complexity.resource_requirements, dict)
        
        # Test caching
        complexity2 = self.analyzer.analyze_task_complexity(task)
        self.assertEqual(complexity, complexity2)
    
    def test_analyze_all_tasks(self):
        """Test analysis of all tasks"""
        complexities = self.analyzer.analyze_all_tasks()
        
        self.assertEqual(len(complexities), 4)
        
        # Check that each task has appropriate complexity
        task_complexities = {c.task_id: c for c in complexities}
        
        # Linear task
        self.assertEqual(task_complexities["1"].time_complexity, ComplexityClass.LINEAR)
        
        # Exponential task
        self.assertEqual(task_complexities["2"].time_complexity, ComplexityClass.EXPONENTIAL)
        
        # Quadratic task  
        self.assertEqual(task_complexities["3"].time_complexity, ComplexityClass.QUADRATIC)
        
        # Logarithmic task
        self.assertEqual(task_complexities["4"].time_complexity, ComplexityClass.LOGARITHMIC)
    
    def test_generate_complexity_report(self):
        """Test complexity report generation"""
        report = self.analyzer.generate_complexity_report()
        
        # Check report structure
        self.assertIn('analysis_timestamp', report)
        self.assertIn('system_resources', report)
        self.assertIn('summary', report)
        self.assertIn('complexity_distribution', report)
        self.assertIn('resource_requirements', report)
        self.assertIn('optimization_opportunities', report)
        self.assertIn('detailed_analysis', report)
        
        # Check summary
        summary = report['summary']
        self.assertEqual(summary['total_tasks'], 4)
        self.assertGreaterEqual(summary['cpu_intensive_tasks'], 0)
        self.assertGreaterEqual(summary['memory_intensive_tasks'], 0)
        
        # Check complexity distribution
        dist = report['complexity_distribution']
        self.assertIn('O(n)', dist)  # Linear complexity
        self.assertIn('O(2^n)', dist)  # Exponential complexity
        
        # Check detailed analysis
        detailed = report['detailed_analysis']
        self.assertEqual(len(detailed), 4)
        self.assertIn('task_id', detailed[0])
        self.assertIn('time_complexity', detailed[0])
    
    def test_identify_bottlenecks(self):
        """Test bottleneck identification"""
        complexities = self.analyzer.analyze_all_tasks()
        bottlenecks = self.analyzer._identify_bottlenecks(complexities)
        
        self.assertIsInstance(bottlenecks, list)
        # Should identify high complexity tasks as bottlenecks
        bottleneck_text = ' '.join(bottlenecks)
        self.assertIn('bottleneck', bottleneck_text.lower())
    
    def test_generate_recommendations(self):
        """Test optimization recommendations"""
        complexities = self.analyzer.analyze_all_tasks()
        recommendations = self.analyzer._generate_recommendations(complexities)
        
        self.assertIsInstance(recommendations, list)
        # Should have recommendations for parallelization, etc.
        if recommendations:
            recommendation_text = ' '.join(recommendations)
            self.assertTrue(any(keyword in recommendation_text.lower() 
                              for keyword in ['parallel', 'memory', 'complexity']))
    
    def test_save_report(self):
        """Test saving analysis report"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_report.json")
            saved_file = self.analyzer.save_report(output_file)
            
            self.assertEqual(saved_file, output_file)
            self.assertTrue(os.path.exists(output_file))
            
            # Verify saved content
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            self.assertIn('analysis_timestamp', saved_data)
            self.assertIn('summary', saved_data)
    
    def test_estimate_runtime(self):
        """Test runtime estimation"""
        task = {"id": "test"}
        
        # Test different complexity classes
        runtime_constant = self.analyzer._estimate_runtime(task, ComplexityClass.CONSTANT, 1)
        runtime_exponential = self.analyzer._estimate_runtime(task, ComplexityClass.EXPONENTIAL, 1)
        
        self.assertGreater(runtime_exponential, runtime_constant)
        
        # Test I/O overhead
        runtime_high_io = self.analyzer._estimate_runtime(task, ComplexityClass.LINEAR, 10)
        runtime_low_io = self.analyzer._estimate_runtime(task, ComplexityClass.LINEAR, 1)
        
        self.assertGreater(runtime_high_io, runtime_low_io)
    
    def test_calculate_resource_requirements(self):
        """Test resource requirement calculation"""
        task = {"id": "test", "details": "file processing task"}
        
        requirements = self.analyzer._calculate_resource_requirements(
            task, ComplexityClass.QUADRATIC, ComplexityClass.LINEAR, True, True
        )
        
        self.assertIn('cpu_cores', requirements)
        self.assertIn('memory_gb', requirements)
        self.assertIn('disk_gb', requirements)
        self.assertIn('estimated_duration_seconds', requirements)
        
        # CPU intensive should require more cores
        self.assertGreater(requirements['cpu_cores'], 1)
        
        # Memory intensive should require more memory
        self.assertGreater(requirements['memory_gb'], 0.5)


class TestComplexityClasses(unittest.TestCase):
    """Test complexity class enumerations"""
    
    def test_complexity_class_values(self):
        """Test that complexity classes have correct values"""
        self.assertEqual(ComplexityClass.CONSTANT.value, "O(1)")
        self.assertEqual(ComplexityClass.LOGARITHMIC.value, "O(log n)")
        self.assertEqual(ComplexityClass.LINEAR.value, "O(n)")
        self.assertEqual(ComplexityClass.LINEARITHMIC.value, "O(n log n)")
        self.assertEqual(ComplexityClass.QUADRATIC.value, "O(n²)")
        self.assertEqual(ComplexityClass.CUBIC.value, "O(n³)")
        self.assertEqual(ComplexityClass.EXPONENTIAL.value, "O(2^n)")
        self.assertEqual(ComplexityClass.FACTORIAL.value, "O(n!)")


class TestSystemResources(unittest.TestCase):
    """Test SystemResources dataclass"""
    
    def test_system_resources_creation(self):
        """Test SystemResources object creation"""
        resources = SystemResources(
            cpu_cores=8,
            available_memory_gb=16.0,
            available_disk_gb=500.0,
            network_bandwidth_mbps=1000.0,
            cpu_usage_percent=25.5,
            memory_usage_percent=45.0
        )
        
        self.assertEqual(resources.cpu_cores, 8)
        self.assertEqual(resources.available_memory_gb, 16.0)
        self.assertEqual(resources.available_disk_gb, 500.0)
        self.assertEqual(resources.network_bandwidth_mbps, 1000.0)
        self.assertEqual(resources.cpu_usage_percent, 25.5)
        self.assertEqual(resources.memory_usage_percent, 45.0)


class TestTaskComplexity(unittest.TestCase):
    """Test TaskComplexity dataclass"""
    
    def test_task_complexity_creation(self):
        """Test TaskComplexity object creation"""
        complexity = TaskComplexity(
            task_id="test_1",
            time_complexity=ComplexityClass.LINEAR,
            space_complexity=ComplexityClass.CONSTANT,
            io_operations=3,
            parallelization_potential=0.7,
            cpu_intensive=True,
            memory_intensive=False,
            network_dependent=False,
            file_operations=2,
            estimated_runtime_seconds=120.5,
            resource_requirements={'cpu_cores': 2, 'memory_gb': 1.0}
        )
        
        self.assertEqual(complexity.task_id, "test_1")
        self.assertEqual(complexity.time_complexity, ComplexityClass.LINEAR)
        self.assertEqual(complexity.space_complexity, ComplexityClass.CONSTANT)
        self.assertEqual(complexity.io_operations, 3)
        self.assertEqual(complexity.parallelization_potential, 0.7)
        self.assertTrue(complexity.cpu_intensive)
        self.assertFalse(complexity.memory_intensive)
        self.assertFalse(complexity.network_dependent)
        self.assertEqual(complexity.file_operations, 2)
        self.assertEqual(complexity.estimated_runtime_seconds, 120.5)
        self.assertEqual(complexity.resource_requirements, {'cpu_cores': 2, 'memory_gb': 1.0})


if __name__ == '__main__':
    unittest.main()