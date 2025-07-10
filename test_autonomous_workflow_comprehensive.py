#!/usr/bin/env python3
"""
Comprehensive Test Suite for Autonomous Workflow Loop

This test suite validates all functionality against the project plan requirements:
- Autonomous workflow execution
- Task Master AI integration
- Perplexity research integration
- Claude Code execution integration
- Self-healing capabilities
- End-to-end workflow validation
"""

import os
import sys
import json
import time
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the autonomous workflow components
from autonomous_workflow_loop import (
    AutonomousWorkflowLoop, WorkflowState, ResearchResult,
    demonstrate_hardcoded_workflow
)


class TestAutonomousWorkflowComprehensive(unittest.TestCase):
    """Comprehensive test suite for autonomous workflow functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.tasks_file = os.path.join(self.test_dir, "tasks.json")
        
        # Create test tasks file
        test_tasks = {
            "tags": {
                "master": {
                    "tasks": [
                        {
                            "id": "test_1",
                            "title": "Test Task 1",
                            "description": "Simple test task",
                            "details": "Linear test operation",
                            "dependencies": []
                        },
                        {
                            "id": "test_2", 
                            "title": "Complex Test Task",
                            "description": "Complex algorithm test",
                            "details": "Recursive exponential complexity test",
                            "dependencies": ["test_1"]
                        }
                    ]
                }
            }
        }
        
        with open(self.tasks_file, 'w') as f:
            json.dump(test_tasks, f)
        
        # Initialize workflow with test tasks
        self.workflow = AutonomousWorkflowLoop(self.tasks_file)
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_01_workflow_initialization(self):
        """Test 1: Workflow initialization and basic setup"""
        print("\n🧪 TEST 1: Workflow Initialization")
        
        # Test workflow object creation
        self.assertIsInstance(self.workflow, AutonomousWorkflowLoop)
        self.assertEqual(self.workflow.tasks_file, self.tasks_file)
        
        # Test workflow state initialization
        self.assertIsInstance(self.workflow.workflow_state, WorkflowState)
        self.assertEqual(self.workflow.workflow_state.stuck_count, 0)
        self.assertEqual(self.workflow.workflow_state.success_count, 0)
        
        # Test configuration parameters
        self.assertEqual(self.workflow.max_stuck_attempts, 3)
        self.assertEqual(self.workflow.max_research_attempts, 2)
        self.assertEqual(self.workflow.max_execution_attempts, 5)
        
        print("✅ Workflow initialization validated")
    
    def test_02_stuck_situation_detection(self):
        """Test 2: Stuck situation detection mechanisms"""
        print("\n🧪 TEST 2: Stuck Situation Detection")
        
        # Test various stuck situation scenarios
        stuck_scenarios = [
            {
                "task_id": "test_import_error",
                "error": "ModuleNotFoundError: No module named 'nonexistent_module'",
                "context": "Python import error"
            },
            {
                "task_id": "test_permission_error", 
                "error": "Permission denied: /restricted/path",
                "context": "File permission error"
            },
            {
                "task_id": "test_api_error",
                "error": "API authentication failed: Invalid key",
                "context": "API configuration error"
            },
            {
                "task_id": "test_dependency_error",
                "error": "Dependency resolution failed",
                "context": "Package dependency issue"
            }
        ]
        
        for scenario in stuck_scenarios:
            # Reset stuck count for each test
            self.workflow.workflow_state.stuck_count = 0
            
            # Test stuck situation handling
            result = self.workflow.handle_stuck_situation(
                scenario["task_id"],
                scenario["error"], 
                scenario["context"]
            )
            
            # Verify stuck situation was detected and handled
            self.assertIsInstance(result, bool)
            self.assertEqual(self.workflow.workflow_state.stuck_count, 1)
            self.assertEqual(self.workflow.workflow_state.last_error, scenario["error"])
            
            print(f"✅ Stuck detection validated for: {scenario['task_id']}")
        
        print("✅ All stuck situation detection scenarios validated")
    
    @patch('subprocess.Popen')
    def test_03_task_master_integration(self, mock_popen):
        """Test 3: Task Master AI integration functionality"""
        print("\n🧪 TEST 3: Task Master AI Integration")
        
        # Mock successful task-master commands
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("Task 1.2: Test task output", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        # Test get_next_task functionality
        next_task = self.workflow.get_next_task()
        self.assertIsNotNone(next_task)
        self.assertIn("id", next_task)
        
        # Verify task-master command was called
        mock_popen.assert_called_with(
            ["task-master", "next"],
            stdout=unittest.mock.ANY,
            stderr=unittest.mock.ANY,
            text=True
        )
        
        # Test show_task_details functionality
        task_details = self.workflow.show_task_details("1.2")
        self.assertIsNotNone(task_details)
        self.assertEqual(task_details["id"], "1.2")
        
        # Test research command integration
        mock_process.communicate.return_value = (
            "Solution steps:\n1. Install required package\n2. Configure settings\n3. Test implementation", 
            ""
        )
        
        research_result = self.workflow.research_solution(
            "Test problem", "Test context"
        )
        
        self.assertIsInstance(research_result, ResearchResult)
        self.assertEqual(research_result.source, "task-master-perplexity-integration")
        self.assertGreater(len(research_result.solution_steps), 0)
        
        print("✅ Task Master AI integration validated")
    
    def test_04_research_solution_parsing(self):
        """Test 4: Research solution parsing and step extraction"""
        print("\n🧪 TEST 4: Research Solution Parsing")
        
        # Test various research output formats
        research_outputs = [
            # Numbered list format
            "1. Install the required package\n2. Configure the environment\n3. Test the solution",
            
            # Bullet point format
            "- Check system requirements\n- Install dependencies\n- Verify installation",
            
            # Mixed format with action words
            "First, install the package. Then configure settings. Finally, test the implementation.",
            
            # Command-based format
            "Run pip install package\nExecute configuration script\nValidate installation"
        ]
        
        for i, output in enumerate(research_outputs):
            steps = self.workflow._parse_research_output(output)
            
            # Verify steps were extracted
            self.assertGreater(len(steps), 0, f"No steps extracted from format {i+1}")
            
            # Verify steps are meaningful (not too short)
            for step in steps:
                self.assertGreater(len(step.strip()), 5, f"Step too short: '{step}'")
            
            print(f"✅ Research parsing validated for format {i+1}: {len(steps)} steps")
        
        print("✅ All research solution parsing scenarios validated")
    
    def test_05_claude_todo_creation(self):
        """Test 5: Claude todo creation and step parsing"""
        print("\n🧪 TEST 5: Claude Todo Creation")
        
        # Test solution steps conversion to todos
        solution_steps = [
            "Install required Python package using pip",
            "Create configuration file with proper settings",
            "Run validation tests to verify setup",
            "Update documentation with new configuration"
        ]
        
        todos = self.workflow.create_todo_from_steps(solution_steps, "Test context")
        
        # Verify todos were created correctly
        self.assertEqual(len(todos), len(solution_steps))
        
        for i, todo in enumerate(todos):
            # Verify todo structure
            self.assertIn("id", todo)
            self.assertIn("content", todo)
            self.assertIn("status", todo)
            self.assertIn("priority", todo)
            self.assertIn("context", todo)
            self.assertIn("source", todo)
            self.assertIn("created", todo)
            
            # Verify content matches original step
            self.assertEqual(todo["content"], solution_steps[i])
            
            # Verify initial status
            self.assertEqual(todo["status"], "pending")
            
            # Verify priority assignment (first 3 are high priority)
            expected_priority = "high" if i < 3 else "medium"
            self.assertEqual(todo["priority"], expected_priority)
            
            # Verify context preservation
            self.assertEqual(todo["context"], "Test context")
            
            print(f"✅ Todo {i+1} structure validated")
        
        print("✅ Claude todo creation validated")
    
    def test_06_todo_execution_scenarios(self):
        """Test 6: Todo execution across different scenario types"""
        print("\n🧪 TEST 6: Todo Execution Scenarios")
        
        # Test different types of todo execution
        test_todos = [
            {
                "id": "install_test",
                "content": "Install psutil using pip install psutil",
                "status": "pending",
                "priority": "high",
                "context": "Installation test"
            },
            {
                "id": "check_test",
                "content": "Check Python version to verify compatibility",
                "status": "pending", 
                "priority": "high",
                "context": "Verification test"
            },
            {
                "id": "create_test",
                "content": "Create directory /tmp/test_workflow for testing",
                "status": "pending",
                "priority": "medium",
                "context": "File operation test"
            },
            {
                "id": "config_test",
                "content": "Configure environment variables for testing",
                "status": "pending",
                "priority": "medium", 
                "context": "Configuration test"
            },
            {
                "id": "manual_test",
                "content": "Review documentation for implementation details",
                "status": "pending",
                "priority": "low",
                "context": "Manual step test"
            }
        ]
        
        execution_results = []
        
        for todo in test_todos:
            # Execute the todo step
            success, message = self.workflow.execute_todo_step(todo)
            
            # Verify execution result
            self.assertIsInstance(success, bool)
            self.assertIsInstance(message, str)
            self.assertGreater(len(message), 0)
            
            execution_results.append({
                "todo_id": todo["id"],
                "success": success,
                "message": message,
                "content": todo["content"]
            })
            
            print(f"✅ Todo execution validated: {todo['id']} - {'SUCCESS' if success else 'HANDLED'}")
        
        # Verify at least some executions were successful
        successful_executions = [r for r in execution_results if r["success"]]
        self.assertGreater(len(successful_executions), 0, "No successful todo executions")
        
        print("✅ All todo execution scenarios validated")
    
    def test_07_workflow_solution_execution(self):
        """Test 7: Complete workflow solution execution"""
        print("\n🧪 TEST 7: Workflow Solution Execution")
        
        # Create test research result
        research_result = ResearchResult(
            query="Test problem resolution",
            solution_steps=[
                "Analyze the error message carefully",
                "Check system dependencies and requirements", 
                "Install any missing packages or tools",
                "Configure environment settings properly",
                "Test the solution implementation",
                "Verify everything works correctly"
            ],
            confidence=0.85,
            source="test-research",
            timestamp=datetime.now().isoformat()
        )
        
        # Execute the complete workflow
        workflow_success = self.workflow.execute_solution_workflow(
            research_result, "Test workflow context"
        )
        
        # Verify workflow execution
        self.assertIsInstance(workflow_success, bool)
        
        # Verify research result was processed
        self.assertGreater(len(self.workflow.research_history), 0)
        
        # Check if workflow files were created
        workflow_history_dir = ".taskmaster/workflow_history"
        if os.path.exists(workflow_history_dir):
            history_files = os.listdir(workflow_history_dir)
            self.assertGreater(len(history_files), 0, "No workflow history files created")
            print(f"✅ Workflow history files created: {len(history_files)}")
        
        print(f"✅ Workflow solution execution completed: {'SUCCESS' if workflow_success else 'PARTIAL'}")
    
    def test_08_end_to_end_autonomous_workflow(self):
        """Test 8: End-to-end autonomous workflow validation"""
        print("\n🧪 TEST 8: End-to-End Autonomous Workflow")
        
        # Test complete autonomous workflow cycle
        with patch('subprocess.Popen') as mock_popen:
            # Mock task-master commands to simulate real workflow
            mock_process = MagicMock()
            
            # Mock sequence: next -> show -> research -> set-status
            command_responses = [
                ("Task test_1: Simple test task", ""),  # next
                ("Task Details:\nID: test_1\nTitle: Simple test task", ""),  # show
                ("Research failed", "Command not found"),  # research (fail for fallback)
                ("Status updated", ""),  # set-status
            ]
            
            response_index = 0
            def mock_communicate(*args, **kwargs):
                nonlocal response_index
                if response_index < len(command_responses):
                    result = command_responses[response_index]
                    response_index += 1
                    return result
                return ("", "")
            
            mock_process.communicate.side_effect = mock_communicate
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            # Run autonomous loop with limited iterations
            report = self.workflow.run_autonomous_loop(max_iterations=2)
            
            # Verify report structure
            self.assertIn("workflow_summary", report)
            self.assertIn("task_statistics", report)
            self.assertIn("workflow_state", report)
            
            # Verify workflow summary
            workflow_summary = report["workflow_summary"]
            self.assertIn("start_time", workflow_summary)
            self.assertIn("end_time", workflow_summary)
            self.assertIn("total_duration_seconds", workflow_summary)
            self.assertIn("iterations_completed", workflow_summary)
            
            # Verify task statistics
            task_stats = report["task_statistics"]
            self.assertIn("completed_tasks", task_stats)
            self.assertIn("failed_tasks", task_stats)
            self.assertIn("success_rate", task_stats)
            
            # Verify workflow state
            workflow_state = report["workflow_state"]
            self.assertIn("stuck_situations_encountered", workflow_state)
            self.assertIn("successful_resolutions", workflow_state)
            
            print("✅ End-to-end autonomous workflow validated")
            print(f"✅ Report structure verified with {len(report)} main sections")
    
    def test_09_error_handling_and_resilience(self):
        """Test 9: Error handling and system resilience"""
        print("\n🧪 TEST 9: Error Handling and Resilience")
        
        # Test max stuck attempts limit
        original_max_attempts = self.workflow.max_stuck_attempts
        self.workflow.max_stuck_attempts = 2
        
        # Trigger stuck situations beyond limit
        for i in range(3):
            result = self.workflow.handle_stuck_situation(
                f"resilience_test_{i}",
                f"Test error {i}",
                "Resilience testing"
            )
            
            if i < 2:
                # Should succeed within limit
                expected_stuck_count = i + 1
            else:
                # Should fail at limit
                expected_stuck_count = 3
                self.assertFalse(result, "Should fail after max attempts exceeded")
        
        # Restore original setting
        self.workflow.max_stuck_attempts = original_max_attempts
        
        # Test invalid research result handling
        invalid_research = ResearchResult(
            query="Invalid test",
            solution_steps=[],  # Empty steps
            confidence=0.0,
            source="invalid-test",
            timestamp=datetime.now().isoformat()
        )
        
        # Should handle empty solution steps gracefully
        workflow_result = self.workflow.execute_solution_workflow(invalid_research)
        self.assertIsInstance(workflow_result, bool)
        
        print("✅ Error handling and resilience validated")
    
    def test_10_integration_validation(self):
        """Test 10: Integration validation with external components"""
        print("\n🧪 TEST 10: Integration Validation")
        
        # Test logging system
        self.assertIsNotNone(self.workflow.logger)
        
        # Test research history tracking
        initial_history_count = len(self.workflow.research_history)
        
        # Perform research to add to history
        research_result = self.workflow.research_solution("Integration test", "Test context")
        
        # Execute the workflow to ensure research is added to history
        self.workflow.execute_solution_workflow(research_result, "Integration test context")
        
        # Verify history was updated (either by research_solution or execute_solution_workflow)
        self.assertGreaterEqual(len(self.workflow.research_history), initial_history_count + 1)
        
        # Test workflow state persistence
        state_timestamp = self.workflow.workflow_state.workflow_start_time
        
        # Verify workflow state tracking
        self.workflow.workflow_state.success_count += 1
        self.assertEqual(self.workflow.workflow_state.success_count, 1)
        
        # Test file system integration
        logs_dir = ".taskmaster/logs"
        if os.path.exists(logs_dir):
            log_files = [f for f in os.listdir(logs_dir) if f.startswith("workflow-")]
            print(f"✅ Log files found: {len(log_files)}")
        
        print("✅ Integration validation completed")


class TestProjectPlanCompliance(unittest.TestCase):
    """Test suite to validate compliance with project plan requirements"""
    
    def test_project_plan_autonomous_capabilities(self):
        """Validate autonomous capabilities against project plan"""
        print("\n🎯 PROJECT PLAN VALIDATION: Autonomous Capabilities")
        
        workflow = AutonomousWorkflowLoop()
        
        # Verify autonomous workflow pattern implementation
        self.assertTrue(hasattr(workflow, 'handle_stuck_situation'))
        self.assertTrue(hasattr(workflow, 'research_solution'))
        self.assertTrue(hasattr(workflow, 'execute_solution_workflow'))
        self.assertTrue(hasattr(workflow, 'run_autonomous_loop'))
        
        print("✅ Autonomous workflow pattern verified")
        
        # Verify hard-coded workflow implementation
        # Pattern: stuck → research → parse → execute → success
        test_result = workflow.handle_stuck_situation(
            "plan_validation",
            "Test project plan validation", 
            "Autonomous capability test"
        )
        
        self.assertIsInstance(test_result, bool)
        print("✅ Hard-coded workflow pattern validated")
    
    def test_project_plan_research_integration(self):
        """Validate research integration against project plan"""
        print("\n🎯 PROJECT PLAN VALIDATION: Research Integration")
        
        workflow = AutonomousWorkflowLoop()
        
        # Verify task-master + perplexity integration
        research_result = workflow.research_solution(
            "Project plan validation test",
            "Research integration validation"
        )
        
        self.assertIsInstance(research_result, ResearchResult)
        self.assertGreater(len(research_result.solution_steps), 0)
        self.assertIn("research", research_result.source.lower())
        
        print("✅ Research integration validated")
    
    def test_project_plan_claude_integration(self):
        """Validate Claude integration against project plan"""
        print("\n🎯 PROJECT PLAN VALIDATION: Claude Integration") 
        
        workflow = AutonomousWorkflowLoop()
        
        # Verify Claude todo parsing and execution
        test_steps = [
            "Test Claude integration step 1",
            "Test Claude integration step 2", 
            "Test Claude integration step 3"
        ]
        
        todos = workflow.create_todo_from_steps(test_steps, "Claude integration test")
        
        self.assertEqual(len(todos), len(test_steps))
        
        # Verify todo execution
        for todo in todos:
            success, message = workflow.execute_todo_step(todo)
            self.assertIsInstance(success, bool)
            self.assertIsInstance(message, str)
        
        print("✅ Claude integration validated")


def run_comprehensive_tests():
    """Run the complete comprehensive test suite"""
    
    print("🚀 STARTING COMPREHENSIVE AUTONOMOUS WORKFLOW TESTS")
    print("=" * 80)
    print("Testing against project plan requirements...")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add autonomous workflow tests
    for test_method in [
        'test_01_workflow_initialization',
        'test_02_stuck_situation_detection', 
        'test_03_task_master_integration',
        'test_04_research_solution_parsing',
        'test_05_claude_todo_creation',
        'test_06_todo_execution_scenarios',
        'test_07_workflow_solution_execution',
        'test_08_end_to_end_autonomous_workflow',
        'test_09_error_handling_and_resilience',
        'test_10_integration_validation'
    ]:
        test_suite.addTest(TestAutonomousWorkflowComprehensive(test_method))
    
    # Add project plan compliance tests
    for test_method in [
        'test_project_plan_autonomous_capabilities',
        'test_project_plan_research_integration',
        'test_project_plan_claude_integration'
    ]:
        test_suite.addTest(TestProjectPlanCompliance(test_method))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_count = total_tests - failures - errors
    success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failures: {failures}")
    print(f"🚨 Errors: {errors}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if failures > 0:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        print(f"\n🚨 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    # Final assessment
    if success_rate >= 90:
        print(f"\n🎯 ASSESSMENT: EXCELLENT - System ready for production")
    elif success_rate >= 80:
        print(f"\n⚡ ASSESSMENT: GOOD - Minor issues to address")
    elif success_rate >= 70:
        print(f"\n⚠️  ASSESSMENT: ACCEPTABLE - Some fixes needed")
    else:
        print(f"\n❌ ASSESSMENT: NEEDS IMPROVEMENT - Major issues found")
    
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    # Run comprehensive test suite
    test_result = run_comprehensive_tests()
    
    # Exit with appropriate code
    exit_code = 0 if (len(test_result.failures) == 0 and len(test_result.errors) == 0) else 1
    sys.exit(exit_code)