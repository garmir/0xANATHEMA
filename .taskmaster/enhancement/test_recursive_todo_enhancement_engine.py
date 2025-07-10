#!/usr/bin/env python3
"""
Comprehensive test suite for the Recursive Todo Enhancement Engine

Tests all major components and functionality:
- Core engine functionality
- Todo analysis and quality scoring
- Task decomposition and enhancement
- Dependency analysis and optimization
- Meta-learning and performance monitoring
- Task Master integration
"""

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import the main engine and components
from recursive_todo_enhancement_engine import (
    RecursiveTodoEnhancementEngine,
    Todo,
    TodoStatus,
    Priority,
    EnhancementType,
    QualityMetrics,
    LocalLLMAdapter,
    TodoAnalyzer,
    TaskDecomposer,
    DependencyAnalyzer,
    EnhancementGenerator,
    QualityScorer,
    TaskMasterIntegration,
    MetaLearningSystem,
    PerformanceMonitor
)

class TestTodo(unittest.TestCase):
    """Test Todo data structure and conversion methods"""
    
    def setUp(self):
        self.sample_todo = Todo(
            id="1",
            title="Test Todo",
            description="Test description",
            status=TodoStatus.PENDING,
            priority=Priority.HIGH,
            dependencies=["2", "3"],
            details="Test details"
        )
    
    def test_todo_creation(self):
        """Test basic todo creation"""
        self.assertEqual(self.sample_todo.id, "1")
        self.assertEqual(self.sample_todo.title, "Test Todo")
        self.assertEqual(self.sample_todo.status, TodoStatus.PENDING)
        self.assertEqual(self.sample_todo.priority, Priority.HIGH)
        self.assertEqual(len(self.sample_todo.dependencies), 2)
    
    def test_todo_to_dict(self):
        """Test todo to dictionary conversion"""
        todo_dict = self.sample_todo.to_dict()
        
        self.assertEqual(todo_dict['id'], "1")
        self.assertEqual(todo_dict['title'], "Test Todo")
        self.assertEqual(todo_dict['status'], "pending")
        self.assertEqual(todo_dict['priority'], "high")
        self.assertEqual(todo_dict['dependencies'], ["2", "3"])
        self.assertIn('qualityMetrics', todo_dict)
        self.assertIn('enhancementHistory', todo_dict)
    
    def test_todo_from_dict(self):
        """Test todo creation from dictionary"""
        todo_dict = {
            'id': '1',
            'title': 'Test Todo',
            'description': 'Test description',
            'status': 'pending',
            'priority': 'high',
            'dependencies': ['2', '3'],
            'details': 'Test details'
        }
        
        todo = Todo.from_dict(todo_dict)
        
        self.assertEqual(todo.id, "1")
        self.assertEqual(todo.title, "Test Todo")
        self.assertEqual(todo.status, TodoStatus.PENDING)
        self.assertEqual(todo.priority, Priority.HIGH)
        self.assertEqual(todo.dependencies, ["2", "3"])
    
    def test_todo_with_subtasks(self):
        """Test todo with subtasks"""
        subtask = Todo(id="1.1", title="Subtask", description="Subtask description")
        self.sample_todo.subtasks.append(subtask)
        
        todo_dict = self.sample_todo.to_dict()
        self.assertEqual(len(todo_dict['subtasks']), 1)
        self.assertEqual(todo_dict['subtasks'][0]['id'], "1.1")
        
        # Test conversion back
        converted_todo = Todo.from_dict(todo_dict)
        self.assertEqual(len(converted_todo.subtasks), 1)
        self.assertEqual(converted_todo.subtasks[0].id, "1.1")

class TestLocalLLMAdapter(unittest.TestCase):
    """Test Local LLM Adapter functionality"""
    
    def setUp(self):
        self.llm_adapter = LocalLLMAdapter()
    
    def test_analyze_text_basic(self):
        """Test basic text analysis"""
        text = "Create a REST API for user authentication"
        analysis = self.llm_adapter.analyze_text(text)
        
        self.assertIn('clarity', analysis)
        self.assertIn('completeness', analysis)
        self.assertIn('actionability', analysis)
        self.assertIn('specificity', analysis)
        self.assertIn('suggestions', analysis)
        
        # Check that scores are between 0 and 1
        self.assertGreaterEqual(analysis['clarity'], 0.0)
        self.assertLessEqual(analysis['clarity'], 1.0)
    
    def test_analyze_empty_text(self):
        """Test analysis of empty text"""
        analysis = self.llm_adapter.analyze_text("")
        
        self.assertEqual(analysis['clarity'], 0.0)
        self.assertEqual(analysis['completeness'], 0.0)
        self.assertEqual(analysis['actionability'], 0.0)
        self.assertEqual(analysis['specificity'], 0.0)
    
    def test_analyze_complex_text(self):
        """Test analysis of complex technical text"""
        text = "Implement a microservices architecture with Docker containers, API Gateway, and service discovery using Consul"
        analysis = self.llm_adapter.analyze_text(text)
        
        # Complex text should have higher specificity
        self.assertGreater(analysis['specificity'], 0.3)
        self.assertGreater(analysis['actionability'], 0.3)
    
    def test_suggestions_generation(self):
        """Test that suggestions are generated"""
        text = "Fix the bug"
        analysis = self.llm_adapter.analyze_text(text)
        
        self.assertIsInstance(analysis['suggestions'], list)
        self.assertGreater(len(analysis['suggestions']), 0)

class TestTodoAnalyzer(unittest.TestCase):
    """Test Todo Analyzer functionality"""
    
    def setUp(self):
        self.llm_adapter = LocalLLMAdapter()
        self.analyzer = TodoAnalyzer(self.llm_adapter)
        self.sample_todo = Todo(
            id="1",
            title="Implement user authentication API",
            description="Create REST API endpoints for user login and registration",
            details="Use JWT tokens, bcrypt for password hashing, and validate input data"
        )
    
    def test_analyze_todo_quality(self):
        """Test todo quality analysis"""
        metrics = self.analyzer.analyze_todo(self.sample_todo)
        
        self.assertIsInstance(metrics, QualityMetrics)
        self.assertGreaterEqual(metrics.overall_score, 0.0)
        self.assertLessEqual(metrics.overall_score, 1.0)
        self.assertGreaterEqual(metrics.clarity_score, 0.0)
        self.assertLessEqual(metrics.clarity_score, 1.0)
    
    def test_analyze_empty_todo(self):
        """Test analysis of empty todo"""
        empty_todo = Todo(id="1", title="", description="")
        metrics = self.analyzer.analyze_todo(empty_todo)
        
        self.assertEqual(metrics.overall_score, 0.0)
        self.assertEqual(metrics.clarity_score, 0.0)
    
    def test_testability_assessment(self):
        """Test testability assessment"""
        todo_with_tests = Todo(
            id="1",
            title="Test todo",
            description="Test description",
            test_strategy="Unit tests and integration tests",
            validation_criteria=["Tests pass", "Code coverage > 80%"]
        )
        
        score = self.analyzer._assess_testability(todo_with_tests)
        self.assertGreater(score, 0.5)
    
    def test_feasibility_assessment(self):
        """Test feasibility assessment"""
        feasible_todo = Todo(
            id="1",
            title="Simple todo",
            description="Easy to implement",
            time_estimate=60,
            resource_requirements=["Developer"]
        )
        
        score = self.analyzer._assess_feasibility(feasible_todo)
        self.assertGreater(score, 0.7)
    
    def test_find_optimization_opportunities(self):
        """Test finding optimization opportunities"""
        todos = [
            Todo(id="1", title="Create API", description="Create REST API"),
            Todo(id="2", title="Build API", description="Build REST API"),  # Similar to first
            Todo(id="3", title="Complex task with multiple requirements", description="Very complex implementation")
        ]
        
        opportunities = self.analyzer.find_optimization_opportunities(todos)
        
        self.assertIsInstance(opportunities, list)
        # Should find at least the redundant tasks
        redundant_found = any(op['type'] == 'redundant_tasks' for op in opportunities)
        self.assertTrue(redundant_found)
    
    def test_similarity_calculation(self):
        """Test similarity calculation between texts"""
        text1 = "Create user authentication system"
        text2 = "Create user authentication API"
        
        similarity = self.analyzer._calculate_similarity(text1, text2)
        self.assertGreater(similarity, 0.5)  # Should be similar
        
        text3 = "Deploy application to production"
        similarity2 = self.analyzer._calculate_similarity(text1, text3)
        self.assertLess(similarity2, 0.3)  # Should be different

class TestDependencyAnalyzer(unittest.TestCase):
    """Test Dependency Analyzer functionality"""
    
    def setUp(self):
        self.analyzer = DependencyAnalyzer()
        self.sample_todos = [
            Todo(id="1", title="Task 1", dependencies=["2"]),
            Todo(id="2", title="Task 2", dependencies=["3"]),
            Todo(id="3", title="Task 3", dependencies=[])
        ]
    
    def test_build_dependency_graph(self):
        """Test building dependency graph"""
        graph = self.analyzer.build_dependency_graph(self.sample_todos)
        
        self.assertIn("1", graph)
        self.assertIn("2", graph)
        self.assertIn("3", graph)
        self.assertEqual(graph["1"], ["2"])
        self.assertEqual(graph["2"], ["3"])
        self.assertEqual(graph["3"], [])
    
    def test_detect_circular_dependencies(self):
        """Test circular dependency detection"""
        # Create circular dependency
        circular_todos = [
            Todo(id="1", title="Task 1", dependencies=["2"]),
            Todo(id="2", title="Task 2", dependencies=["3"]),
            Todo(id="3", title="Task 3", dependencies=["1"])  # Creates cycle
        ]
        
        self.analyzer.build_dependency_graph(circular_todos)
        cycles = self.analyzer.detect_circular_dependencies()
        
        self.assertGreater(len(cycles), 0)
        self.assertIn("1", cycles[0])
        self.assertIn("2", cycles[0])
        self.assertIn("3", cycles[0])
    
    def test_no_circular_dependencies(self):
        """Test when no circular dependencies exist"""
        self.analyzer.build_dependency_graph(self.sample_todos)
        cycles = self.analyzer.detect_circular_dependencies()
        
        self.assertEqual(len(cycles), 0)
    
    def test_optimize_task_order(self):
        """Test task order optimization"""
        ordered_tasks = self.analyzer.optimize_task_order(self.sample_todos)
        
        self.assertIsInstance(ordered_tasks, list)
        self.assertEqual(len(ordered_tasks), 3)
        
        # Task 3 should come before Task 2, which should come before Task 1
        self.assertLess(ordered_tasks.index("3"), ordered_tasks.index("2"))
        self.assertLess(ordered_tasks.index("2"), ordered_tasks.index("1"))
    
    def test_identify_parallel_opportunities(self):
        """Test identification of parallel work opportunities"""
        # Create todos that can be worked on in parallel
        parallel_todos = [
            Todo(id="1", title="Task 1", dependencies=[]),
            Todo(id="2", title="Task 2", dependencies=[]),
            Todo(id="3", title="Task 3", dependencies=["1", "2"])
        ]
        
        self.analyzer.build_dependency_graph(parallel_todos)
        parallel_groups = self.analyzer.identify_parallel_opportunities(parallel_todos)
        
        self.assertIsInstance(parallel_groups, list)
        # Should find that tasks 1 and 2 can be done in parallel
        self.assertGreater(len(parallel_groups), 0)

class TestTaskDecomposer(unittest.TestCase):
    """Test Task Decomposer functionality"""
    
    def setUp(self):
        self.llm_adapter = LocalLLMAdapter()
        self.decomposer = TaskDecomposer(self.llm_adapter)
    
    def test_decompose_api_task(self):
        """Test decomposition of API task"""
        api_todo = Todo(
            id="1",
            title="Create user authentication API",
            description="Implement REST API for user authentication with JWT tokens"
        )
        
        subtasks = self.decomposer.decompose_task(api_todo)
        
        self.assertIsInstance(subtasks, list)
        self.assertGreater(len(subtasks), 0)
        self.assertLessEqual(len(subtasks), 5)  # Should not exceed max_subtasks
        
        # Check subtask structure
        for subtask in subtasks:
            self.assertIsInstance(subtask, Todo)
            self.assertTrue(subtask.id.startswith("1."))
            self.assertNotEqual(subtask.title, "")
    
    def test_decompose_database_task(self):
        """Test decomposition of database task"""
        db_todo = Todo(
            id="2",
            title="Design user database schema",
            description="Create database tables for user management system"
        )
        
        subtasks = self.decomposer.decompose_task(db_todo)
        
        self.assertIsInstance(subtasks, list)
        self.assertGreater(len(subtasks), 0)
        
        # Should contain database-specific subtasks
        subtask_titles = [subtask.title for subtask in subtasks]
        self.assertTrue(any('schema' in title.lower() for title in subtask_titles))
    
    def test_decompose_frontend_task(self):
        """Test decomposition of frontend task"""
        ui_todo = Todo(
            id="3",
            title="Create user interface for login",
            description="Build frontend components for user authentication"
        )
        
        subtasks = self.decomposer.decompose_task(ui_todo)
        
        self.assertIsInstance(subtasks, list)
        self.assertGreater(len(subtasks), 0)
        
        # Should contain UI-specific subtasks
        subtask_titles = [subtask.title for subtask in subtasks]
        self.assertTrue(any('component' in title.lower() for title in subtask_titles))
    
    def test_decompose_already_decomposed_task(self):
        """Test decomposition of task that already has subtasks"""
        todo_with_subtasks = Todo(
            id="4",
            title="Test todo",
            description="Test description",
            subtasks=[Todo(id="4.1", title="Existing subtask")]
        )
        
        subtasks = self.decomposer.decompose_task(todo_with_subtasks)
        
        # Should return existing subtasks
        self.assertEqual(len(subtasks), 1)
        self.assertEqual(subtasks[0].id, "4.1")
    
    def test_identify_patterns(self):
        """Test pattern identification in tasks"""
        api_todo = Todo(
            id="1",
            title="Create REST API service",
            description="Implement microservice with database integration"
        )
        
        patterns = self.decomposer._identify_patterns(api_todo)
        
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        
        # Should identify API pattern
        pattern_titles = [pattern['title'] for pattern in patterns]
        self.assertTrue(any('api' in title.lower() for title in pattern_titles))

class TestEnhancementGenerator(unittest.TestCase):
    """Test Enhancement Generator functionality"""
    
    def setUp(self):
        self.llm_adapter = LocalLLMAdapter()
        self.generator = EnhancementGenerator(self.llm_adapter)
        self.sample_todo = Todo(
            id="1",
            title="Implement authentication",
            description="Basic auth implementation"
        )
    
    def test_enhance_todo_single_type(self):
        """Test enhancing todo with single enhancement type"""
        enhanced_todo = self.generator.enhance_todo(
            self.sample_todo,
            [EnhancementType.DESCRIPTION_ENHANCEMENT]
        )
        
        self.assertIsInstance(enhanced_todo, Todo)
        self.assertEqual(enhanced_todo.id, self.sample_todo.id)
        self.assertEqual(enhanced_todo.title, self.sample_todo.title)
        self.assertGreater(len(enhanced_todo.enhancement_history), 0)
    
    def test_enhance_todo_multiple_types(self):
        """Test enhancing todo with multiple enhancement types"""
        enhancement_types = [
            EnhancementType.DESCRIPTION_ENHANCEMENT,
            EnhancementType.TIME_ESTIMATION,
            EnhancementType.TEST_STRATEGY
        ]
        
        enhanced_todo = self.generator.enhance_todo(self.sample_todo, enhancement_types)
        
        self.assertEqual(len(enhanced_todo.enhancement_history), len(enhancement_types))
        
        # Check that enhancements were applied
        self.assertIsNotNone(enhanced_todo.time_estimate)
        self.assertNotEqual(enhanced_todo.test_strategy, "")
    
    def test_description_enhancement(self):
        """Test description enhancement specifically"""
        basic_todo = Todo(id="1", title="Create API", description="")
        
        enhanced_dict = self.generator._enhance_description(basic_todo.to_dict())
        
        self.assertNotEqual(enhanced_dict['description'], "")
        self.assertGreater(len(enhanced_dict['description']), 10)
    
    def test_time_estimation_enhancement(self):
        """Test time estimation enhancement"""
        todo_dict = {"id": "1", "title": "Create complex API system", "description": ""}
        
        enhanced_dict = self.generator._add_time_estimation(todo_dict)
        
        self.assertIsNotNone(enhanced_dict.get('timeEstimate'))
        self.assertGreater(enhanced_dict['timeEstimate'], 0)
    
    def test_resource_planning_enhancement(self):
        """Test resource planning enhancement"""
        todo_dict = {"id": "1", "title": "Create API endpoints", "description": ""}
        
        enhanced_dict = self.generator._add_resource_planning(todo_dict)
        
        self.assertIsNotNone(enhanced_dict.get('resourceRequirements'))
        self.assertGreater(len(enhanced_dict['resourceRequirements']), 0)
    
    def test_test_strategy_enhancement(self):
        """Test test strategy enhancement"""
        todo_dict = {"id": "1", "title": "Create database schema", "description": ""}
        
        enhanced_dict = self.generator._add_test_strategy(todo_dict)
        
        self.assertIsNotNone(enhanced_dict.get('testStrategy'))
        self.assertNotEqual(enhanced_dict['testStrategy'], "")
    
    def test_validation_criteria_enhancement(self):
        """Test validation criteria enhancement"""
        todo_dict = {"id": "1", "title": "Implement user interface", "description": ""}
        
        enhanced_dict = self.generator._add_validation_criteria(todo_dict)
        
        self.assertIsNotNone(enhanced_dict.get('validationCriteria'))
        self.assertGreater(len(enhanced_dict['validationCriteria']), 0)

class TestQualityScorer(unittest.TestCase):
    """Test Quality Scorer functionality"""
    
    def setUp(self):
        self.llm_adapter = LocalLLMAdapter()
        self.scorer = QualityScorer(self.llm_adapter)
    
    def test_score_single_todo(self):
        """Test scoring a single todo"""
        todo = Todo(
            id="1",
            title="Implement comprehensive user authentication system",
            description="Create secure authentication with JWT tokens, password hashing, and input validation",
            details="Use bcrypt for password hashing, implement proper session management",
            test_strategy="Unit tests for auth functions, integration tests for login flow",
            validation_criteria=["All tests pass", "Security audit complete"]
        )
        
        metrics = self.scorer.score_todo(todo)
        
        self.assertIsInstance(metrics, QualityMetrics)
        self.assertGreater(metrics.overall_score, 0.0)
        self.assertLessEqual(metrics.overall_score, 1.0)
        
        # Well-defined todo should have decent scores
        self.assertGreater(metrics.clarity_score, 0.3)
        self.assertGreater(metrics.completeness_score, 0.3)
    
    def test_score_todo_list(self):
        """Test scoring a list of todos"""
        todos = [
            Todo(id="1", title="Good task", description="Well-defined task with clear requirements"),
            Todo(id="2", title="Bad task", description=""),
            Todo(id="3", title="Average task", description="Some description")
        ]
        
        results = self.scorer.score_todo_list(todos)
        
        self.assertIn('average_score', results)
        self.assertIn('median_score', results)
        self.assertIn('min_score', results)
        self.assertIn('max_score', results)
        self.assertIn('individual_scores', results)
        self.assertIn('recommendations', results)
        
        # Should have scores for all todos
        self.assertEqual(len(results['individual_scores']), 3)
    
    def test_score_empty_todo_list(self):
        """Test scoring empty todo list"""
        results = self.scorer.score_todo_list([])
        
        self.assertEqual(results['average_score'], 0.0)
        self.assertEqual(results['median_score'], 0.0)
        self.assertEqual(len(results['individual_scores']), 0)
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        todos = [
            Todo(id="1", title="", description=""),  # Low quality
            Todo(id="2", title="Good task", description="Well-defined task")
        ]
        
        # Score todos first
        for todo in todos:
            todo.quality_metrics = self.scorer.score_todo(todo)
        
        scores = {
            todo.id: todo.quality_metrics.to_dict() for todo in todos
        }
        
        recommendations = self.scorer._generate_recommendations(todos, scores)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

class TestTaskMasterIntegration(unittest.TestCase):
    """Test Task Master Integration functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.integration = TaskMasterIntegration(self.temp_dir)
        
        # Create sample tasks file
        self.sample_tasks_data = {
            "tasks": [
                {
                    "id": "1",
                    "title": "Test Task 1",
                    "description": "Test description 1",
                    "status": "pending",
                    "priority": "high",
                    "dependencies": []
                },
                {
                    "id": "2",
                    "title": "Test Task 2",
                    "description": "Test description 2",
                    "status": "in-progress",
                    "priority": "medium",
                    "dependencies": ["1"]
                }
            ]
        }
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_load_tasks_success(self):
        """Test successful task loading"""
        # Create tasks file
        tasks_file = Path(self.temp_dir) / "tasks" / "tasks.json"
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(tasks_file, 'w') as f:
            json.dump(self.sample_tasks_data, f)
        
        todos = self.integration.load_tasks()
        
        self.assertEqual(len(todos), 2)
        self.assertEqual(todos[0].id, "1")
        self.assertEqual(todos[1].id, "2")
        self.assertEqual(todos[0].title, "Test Task 1")
        self.assertEqual(todos[1].status, TodoStatus.IN_PROGRESS)
    
    def test_load_tasks_no_file(self):
        """Test loading tasks when file doesn't exist"""
        todos = self.integration.load_tasks()
        self.assertEqual(len(todos), 0)
    
    def test_save_tasks_success(self):
        """Test successful task saving"""
        todos = [
            Todo(id="1", title="Test Task", description="Test description"),
            Todo(id="2", title="Test Task 2", description="Test description 2")
        ]
        
        success = self.integration.save_tasks(todos)
        
        self.assertTrue(success)
        self.assertTrue(self.integration.tasks_file.exists())
        
        # Verify saved data
        with open(self.integration.tasks_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(len(saved_data['tasks']), 2)
        self.assertEqual(saved_data['tasks'][0]['id'], "1")
    
    def test_get_task_by_id(self):
        """Test getting specific task by ID"""
        # Setup tasks file
        tasks_file = Path(self.temp_dir) / "tasks" / "tasks.json"
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(tasks_file, 'w') as f:
            json.dump(self.sample_tasks_data, f)
        
        # Test getting existing task
        todo = self.integration.get_task_by_id("1")
        self.assertIsNotNone(todo)
        self.assertEqual(todo.id, "1")
        self.assertEqual(todo.title, "Test Task 1")
        
        # Test getting non-existing task
        todo = self.integration.get_task_by_id("999")
        self.assertIsNone(todo)
    
    def test_update_task(self):
        """Test updating a specific task"""
        # Setup tasks file
        tasks_file = Path(self.temp_dir) / "tasks" / "tasks.json"
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(tasks_file, 'w') as f:
            json.dump(self.sample_tasks_data, f)
        
        # Update task
        updated_todo = Todo(
            id="1",
            title="Updated Task",
            description="Updated description",
            status=TodoStatus.DONE
        )
        
        success = self.integration.update_task("1", updated_todo)
        self.assertTrue(success)
        
        # Verify update
        todo = self.integration.get_task_by_id("1")
        self.assertEqual(todo.title, "Updated Task")
        self.assertEqual(todo.status, TodoStatus.DONE)
    
    def test_export_enhanced_tasks(self):
        """Test exporting enhanced tasks"""
        todos = [
            Todo(id="1", title="Test Task", description="Test description")
        ]
        
        output_file = os.path.join(self.temp_dir, "enhanced_tasks.json")
        success = self.integration.export_enhanced_tasks(todos, output_file)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_file))
        
        # Verify export content
        with open(output_file, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIn('enhanced_tasks', exported_data)
        self.assertIn('export_timestamp', exported_data)
        self.assertIn('enhancement_summary', exported_data)
        self.assertEqual(len(exported_data['enhanced_tasks']), 1)

class TestMetaLearningSystem(unittest.TestCase):
    """Test Meta Learning System functionality"""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        self.meta_learning = MetaLearningSystem(self.temp_file.name)
    
    def tearDown(self):
        os.unlink(self.temp_file.name)
    
    def test_record_enhancement_outcome(self):
        """Test recording enhancement outcomes"""
        context = {'has_api': True, 'complexity_level': 'high'}
        
        self.meta_learning.record_enhancement_outcome(
            EnhancementType.DESCRIPTION_ENHANCEMENT,
            0.3,
            context
        )
        
        # Check that outcome was recorded
        effectiveness = self.meta_learning.get_enhancement_effectiveness(
            EnhancementType.DESCRIPTION_ENHANCEMENT,
            context
        )
        
        self.assertEqual(effectiveness, 0.3)
    
    def test_get_enhancement_effectiveness_default(self):
        """Test getting effectiveness for unknown enhancement"""
        context = {'has_api': False, 'complexity_level': 'low'}
        
        effectiveness = self.meta_learning.get_enhancement_effectiveness(
            EnhancementType.TIME_ESTIMATION,
            context
        )
        
        self.assertEqual(effectiveness, 0.5)  # Default effectiveness
    
    def test_recommend_enhancement_strategy(self):
        """Test enhancement strategy recommendation"""
        todo = Todo(
            id="1",
            title="Create REST API",
            description="Build API endpoints for user management"
        )
        
        recommendations = self.meta_learning.recommend_enhancement_strategy(todo)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)
        
        for rec in recommendations:
            self.assertIsInstance(rec, EnhancementType)
    
    def test_context_extraction(self):
        """Test context extraction from todos"""
        api_todo = Todo(
            id="1",
            title="Create REST API",
            description="Build API with database integration"
        )
        
        context = self.meta_learning._extract_context(api_todo)
        
        self.assertIsInstance(context, dict)
        self.assertIn('has_api', context)
        self.assertIn('has_database', context)
        self.assertIn('complexity_level', context)
        
        self.assertTrue(context['has_api'])
        self.assertTrue(context['has_database'])
    
    def test_context_to_key(self):
        """Test context to key conversion"""
        context = {'has_api': True, 'complexity_level': 'high'}
        
        key = self.meta_learning._context_to_key(context)
        
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 8)  # MD5 hash truncated to 8 chars
        
        # Same context should produce same key
        key2 = self.meta_learning._context_to_key(context)
        self.assertEqual(key, key2)

class TestPerformanceMonitor(unittest.TestCase):
    """Test Performance Monitor functionality"""
    
    def setUp(self):
        self.monitor = PerformanceMonitor()
    
    def test_record_metrics(self):
        """Test recording various metrics"""
        # Record some metrics
        self.monitor.record_enhancement_time(EnhancementType.DESCRIPTION_ENHANCEMENT, 1.5)
        self.monitor.record_quality_improvement(EnhancementType.DESCRIPTION_ENHANCEMENT, 0.3)
        self.monitor.record_success(EnhancementType.DESCRIPTION_ENHANCEMENT)
        
        # Verify metrics were recorded
        times = self.monitor.metrics['enhancement_times'][EnhancementType.DESCRIPTION_ENHANCEMENT.value]
        self.assertEqual(len(times), 1)
        self.assertEqual(times[0], 1.5)
        
        improvements = self.monitor.metrics['quality_improvements'][EnhancementType.DESCRIPTION_ENHANCEMENT.value]
        self.assertEqual(len(improvements), 1)
        self.assertEqual(improvements[0], 0.3)
        
        successes = self.monitor.metrics['success_rates'][EnhancementType.DESCRIPTION_ENHANCEMENT.value]
        self.assertEqual(successes, 1)
    
    def test_record_errors(self):
        """Test recording errors"""
        self.monitor.record_error(EnhancementType.TIME_ESTIMATION)
        
        errors = self.monitor.metrics['error_counts'][EnhancementType.TIME_ESTIMATION.value]
        self.assertEqual(errors, 1)
    
    def test_get_performance_report(self):
        """Test generating performance report"""
        # Record some metrics
        self.monitor.record_enhancement_time(EnhancementType.DESCRIPTION_ENHANCEMENT, 1.0)
        self.monitor.record_enhancement_time(EnhancementType.DESCRIPTION_ENHANCEMENT, 2.0)
        self.monitor.record_quality_improvement(EnhancementType.DESCRIPTION_ENHANCEMENT, 0.2)
        self.monitor.record_quality_improvement(EnhancementType.DESCRIPTION_ENHANCEMENT, 0.4)
        self.monitor.record_success(EnhancementType.DESCRIPTION_ENHANCEMENT)
        self.monitor.record_success(EnhancementType.DESCRIPTION_ENHANCEMENT)
        
        report = self.monitor.get_performance_report()
        
        self.assertIn('enhancement_statistics', report)
        self.assertIn('overall_performance', report)
        
        desc_stats = report['enhancement_statistics'][EnhancementType.DESCRIPTION_ENHANCEMENT.value]
        self.assertEqual(desc_stats['average_time'], 1.5)
        self.assertEqual(desc_stats['average_improvement'], 0.3)
        self.assertEqual(desc_stats['success_count'], 2)
        self.assertEqual(desc_stats['success_rate'], 1.0)
        
        overall = report['overall_performance']
        self.assertEqual(overall['total_enhancements'], 2)
        self.assertEqual(overall['average_time'], 1.5)
        self.assertEqual(overall['average_improvement'], 0.3)

class TestRecursiveTodoEnhancementEngine(unittest.TestCase):
    """Test the main Recursive Todo Enhancement Engine"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.engine = RecursiveTodoEnhancementEngine(
            taskmaster_dir=self.temp_dir,
            max_recursion_depth=2,
            enable_meta_learning=False  # Disable for testing
        )
        
        # Create sample tasks
        self.sample_todos = [
            Todo(
                id="1",
                title="Create user authentication",
                description="Basic auth implementation",
                priority=Priority.HIGH
            ),
            Todo(
                id="2",
                title="Design database schema",
                description="User management database",
                priority=Priority.MEDIUM,
                dependencies=["1"]
            )
        ]
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine.llm_adapter)
        self.assertIsNotNone(self.engine.todo_analyzer)
        self.assertIsNotNone(self.engine.dependency_analyzer)
        self.assertIsNotNone(self.engine.task_decomposer)
        self.assertIsNotNone(self.engine.enhancement_generator)
        self.assertIsNotNone(self.engine.quality_scorer)
        self.assertIsNotNone(self.engine.taskmaster_integration)
        self.assertIsNotNone(self.engine.performance_monitor)
    
    def test_enhance_todos_basic(self):
        """Test basic todo enhancement"""
        enhanced_todos = self.engine.enhance_todos(
            todos=self.sample_todos,
            enhancement_types=[EnhancementType.DESCRIPTION_ENHANCEMENT],
            recursive_depth=1
        )
        
        self.assertEqual(len(enhanced_todos), 2)
        
        # Check that enhancements were applied
        for todo in enhanced_todos:
            self.assertGreater(len(todo.enhancement_history), 0)
            self.assertIsNotNone(todo.quality_metrics)
    
    def test_enhance_todos_multiple_cycles(self):
        """Test multiple enhancement cycles"""
        enhanced_todos = self.engine.enhance_todos(
            todos=self.sample_todos,
            enhancement_types=[
                EnhancementType.DESCRIPTION_ENHANCEMENT,
                EnhancementType.TIME_ESTIMATION
            ],
            recursive_depth=2
        )
        
        self.assertEqual(len(enhanced_todos), 2)
        
        # Check that multiple enhancements were applied
        for todo in enhanced_todos:
            self.assertGreaterEqual(len(todo.enhancement_history), 2)
    
    def test_enhance_todos_empty_list(self):
        """Test enhancement with empty todo list"""
        enhanced_todos = self.engine.enhance_todos(todos=[])
        
        self.assertEqual(len(enhanced_todos), 0)
    
    def test_analyze_project_todos(self):
        """Test project todo analysis"""
        # Save sample todos first
        self.engine.taskmaster_integration.save_tasks(self.sample_todos)
        
        report = self.engine.analyze_project_todos()
        
        self.assertIn('project_overview', report)
        self.assertIn('quality_analysis', report)
        self.assertIn('optimization_opportunities', report)
        self.assertIn('dependency_analysis', report)
        self.assertIn('decomposition_recommendations', report)
        self.assertIn('performance_metrics', report)
        
        # Check project overview
        overview = report['project_overview']
        self.assertEqual(overview['total_todos'], 2)
        self.assertIn('by_status', overview)
        self.assertIn('by_priority', overview)
    
    def test_auto_decompose_complex_todos(self):
        """Test automatic decomposition of complex todos"""
        complex_todo = Todo(
            id="3",
            title="Create comprehensive microservices architecture",
            description="Build scalable microservices with API Gateway, service discovery, and monitoring",
            details="Implement with Docker, Kubernetes, and include CI/CD pipeline"
        )
        
        todos = self.sample_todos + [complex_todo]
        self.engine.taskmaster_integration.save_tasks(todos)
        
        result_todos = self.engine.auto_decompose_complex_todos(complexity_threshold=0.5)
        
        self.assertEqual(len(result_todos), 3)
        
        # Find the complex todo and check if it was decomposed
        complex_result = next(t for t in result_todos if t.id == "3")
        self.assertGreater(len(complex_result.subtasks), 0)
    
    def test_optimize_dependencies(self):
        """Test dependency optimization"""
        # Create todos with circular dependency
        circular_todos = [
            Todo(id="1", title="Task 1", dependencies=["2"]),
            Todo(id="2", title="Task 2", dependencies=["3"]),
            Todo(id="3", title="Task 3", dependencies=["1"])
        ]
        
        self.engine.taskmaster_integration.save_tasks(circular_todos)
        
        result = self.engine.optimize_dependencies()
        
        self.assertIn('circular_dependencies_found', result)
        self.assertIn('resolutions_applied', result)
        self.assertIn('optimized_order', result)
        self.assertIn('parallel_opportunities', result)
    
    def test_batch_enhance_by_pattern(self):
        """Test batch enhancement by pattern matching"""
        api_todos = [
            Todo(id="1", title="Create user API", description="User management API"),
            Todo(id="2", title="Create product API", description="Product management API"),
            Todo(id="3", title="Setup database", description="Database configuration")
        ]
        
        self.engine.taskmaster_integration.save_tasks(api_todos)
        
        enhanced_todos = self.engine.batch_enhance_by_pattern(
            "API",
            [EnhancementType.DESCRIPTION_ENHANCEMENT, EnhancementType.TEST_STRATEGY]
        )
        
        self.assertEqual(len(enhanced_todos), 2)  # Only API todos should be enhanced
        
        for todo in enhanced_todos:
            self.assertIn("API", todo.title)
            self.assertGreater(len(todo.enhancement_history), 0)
    
    def test_export_enhancement_report(self):
        """Test exporting enhancement report"""
        self.engine.taskmaster_integration.save_tasks(self.sample_todos)
        
        report_file = os.path.join(self.temp_dir, "enhancement_report.json")
        success = self.engine.export_enhancement_report(report_file)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(report_file))
        
        # Verify report content
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        self.assertIn('project_overview', report_data)
        self.assertIn('quality_analysis', report_data)
    
    def test_count_by_status(self):
        """Test counting todos by status"""
        todos = [
            Todo(id="1", title="Task 1", status=TodoStatus.PENDING),
            Todo(id="2", title="Task 2", status=TodoStatus.PENDING),
            Todo(id="3", title="Task 3", status=TodoStatus.DONE)
        ]
        
        counts = self.engine._count_by_status(todos)
        
        self.assertEqual(counts['pending'], 2)
        self.assertEqual(counts['done'], 1)
    
    def test_count_by_priority(self):
        """Test counting todos by priority"""
        todos = [
            Todo(id="1", title="Task 1", priority=Priority.HIGH),
            Todo(id="2", title="Task 2", priority=Priority.HIGH),
            Todo(id="3", title="Task 3", priority=Priority.LOW)
        ]
        
        counts = self.engine._count_by_priority(todos)
        
        self.assertEqual(counts['high'], 2)
        self.assertEqual(counts['low'], 1)

class TestIntegrationScenarios(unittest.TestCase):
    """Test real-world integration scenarios"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.engine = RecursiveTodoEnhancementEngine(
            taskmaster_dir=self.temp_dir,
            enable_meta_learning=False
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_full_workflow_scenario(self):
        """Test complete workflow from raw todos to enhanced todos"""
        # Start with basic todos
        raw_todos = [
            Todo(id="1", title="Create API", description="API for users"),
            Todo(id="2", title="Database", description="Store user data"),
            Todo(id="3", title="Frontend", description="User interface")
        ]
        
        # Save initial todos
        self.engine.taskmaster_integration.save_tasks(raw_todos)
        
        # Analyze project
        initial_analysis = self.engine.analyze_project_todos()
        initial_quality = initial_analysis['quality_analysis']['average_score']
        
        # Enhance todos
        enhanced_todos = self.engine.enhance_todos(
            enhancement_types=[
                EnhancementType.DESCRIPTION_ENHANCEMENT,
                EnhancementType.TIME_ESTIMATION,
                EnhancementType.TEST_STRATEGY,
                EnhancementType.VALIDATION_CRITERIA
            ],
            recursive_depth=2
        )
        
        # Analyze after enhancement
        final_analysis = self.engine.analyze_project_todos()
        final_quality = final_analysis['quality_analysis']['average_score']
        
        # Verify improvements
        self.assertGreater(final_quality, initial_quality)
        
        # Verify all todos were enhanced
        for todo in enhanced_todos:
            self.assertGreater(len(todo.enhancement_history), 0)
            self.assertIsNotNone(todo.time_estimate)
            self.assertNotEqual(todo.test_strategy, "")
            self.assertGreater(len(todo.validation_criteria), 0)
    
    def test_complex_project_scenario(self):
        """Test scenario with complex project structure"""
        # Create a realistic project structure
        complex_todos = [
            Todo(
                id="1",
                title="Setup project infrastructure",
                description="Initialize development environment",
                priority=Priority.HIGH
            ),
            Todo(
                id="2",
                title="Design system architecture",
                description="Create technical architecture document",
                priority=Priority.HIGH,
                dependencies=["1"]
            ),
            Todo(
                id="3",
                title="Implement authentication service",
                description="Create microservice for user authentication",
                priority=Priority.HIGH,
                dependencies=["2"]
            ),
            Todo(
                id="4",
                title="Build user management API",
                description="REST API for user operations",
                priority=Priority.MEDIUM,
                dependencies=["3"]
            ),
            Todo(
                id="5",
                title="Create frontend application",
                description="React application for user interface",
                priority=Priority.MEDIUM,
                dependencies=["4"]
            ),
            Todo(
                id="6",
                title="Setup CI/CD pipeline",
                description="Automated deployment pipeline",
                priority=Priority.LOW,
                dependencies=["2"]
            ),
            Todo(
                id="7",
                title="Write documentation",
                description="Technical and user documentation",
                priority=Priority.LOW,
                dependencies=["5", "6"]
            )
        ]
        
        # Save complex project
        self.engine.taskmaster_integration.save_tasks(complex_todos)
        
        # Analyze dependencies
        dependency_analysis = self.engine.optimize_dependencies()
        
        # Verify optimal order respects dependencies
        optimal_order = dependency_analysis['optimal_order']
        self.assertIn("1", optimal_order)
        self.assertIn("7", optimal_order)
        self.assertLess(optimal_order.index("1"), optimal_order.index("2"))
        self.assertLess(optimal_order.index("2"), optimal_order.index("3"))
        
        # Auto-decompose complex tasks
        decomposed_todos = self.engine.auto_decompose_complex_todos(complexity_threshold=0.5)
        
        # Verify some tasks were decomposed
        decomposed_count = sum(1 for todo in decomposed_todos if len(todo.subtasks) > 0)
        self.assertGreater(decomposed_count, 0)
        
        # Enhance all todos
        final_todos = self.engine.enhance_todos(recursive_depth=2)
        
        # Verify comprehensive enhancement
        for todo in final_todos:
            self.assertGreater(todo.quality_metrics.overall_score, 0)
            if len(todo.enhancement_history) > 0:
                self.assertIsNotNone(todo.time_estimate)
    
    def test_error_handling_scenario(self):
        """Test error handling in various scenarios"""
        # Test with malformed todo data
        malformed_todo = Todo(id="", title="", description="")
        
        # Should not crash
        enhanced_todo = self.engine._enhance_single_todo(
            malformed_todo,
            [EnhancementType.DESCRIPTION_ENHANCEMENT]
        )
        
        # Should return the original todo if enhancement fails
        self.assertEqual(enhanced_todo.id, malformed_todo.id)
        
        # Test with non-existent task master directory
        non_existent_engine = RecursiveTodoEnhancementEngine(
            taskmaster_dir="/non/existent/path",
            enable_meta_learning=False
        )
        
        # Should handle gracefully
        todos = non_existent_engine.taskmaster_integration.load_tasks()
        self.assertEqual(len(todos), 0)

def run_comprehensive_tests():
    """Run all tests and provide detailed results"""
    print("=" * 80)
    print("COMPREHENSIVE TEST SUITE FOR RECURSIVE TODO ENHANCEMENT ENGINE")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTodo,
        TestLocalLLMAdapter,
        TestTodoAnalyzer,
        TestDependencyAnalyzer,
        TestTaskDecomposer,
        TestEnhancementGenerator,
        TestQualityScorer,
        TestTaskMasterIntegration,
        TestMetaLearningSystem,
        TestPerformanceMonitor,
        TestRecursiveTodoEnhancementEngine,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)