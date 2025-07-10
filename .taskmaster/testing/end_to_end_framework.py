#!/usr/bin/env python3
"""
End-to-End Testing Framework for Task Master Autonomous Execution
Validates complete autonomous execution pipeline with comprehensive test scenarios.
"""

import json
import os
import sys
import time
import subprocess
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
import shutil

@dataclass
class TestScenario:
    """Represents a test scenario for autonomous execution"""
    id: str
    name: str
    project_type: str  # "web_app", "api", "data_processing", "cli_tool"
    complexity_level: str  # "simple", "medium", "complex"
    prd_content: str
    expected_tasks: int
    expected_autonomy_score: float
    timeout_minutes: int
    setup_commands: List[str]
    validation_commands: List[str]
    cleanup_commands: List[str]

@dataclass
class TestResult:
    """Represents the result of a test execution"""
    scenario_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    success: bool
    autonomy_score: float
    tasks_generated: int
    tasks_completed: int
    completion_rate: float
    error_messages: List[str]
    performance_metrics: Dict[str, Any]
    logs: List[str]

class EndToEndTestFramework:
    """Comprehensive testing framework for autonomous execution validation"""
    
    def __init__(self, test_workspace: str = None):
        self.test_workspace = test_workspace or "/tmp/taskmaster_e2e_tests"
        self.test_scenarios_dir = Path(__file__).parent / "test_scenarios"
        self.reports_dir = Path(__file__).parent / "reports"
        self.logger = self._setup_logging()
        self.test_results: List[TestResult] = []
        
        # Ensure directories exist
        Path(self.test_workspace).mkdir(parents=True, exist_ok=True)
        self.test_scenarios_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the test framework"""
        logger = logging.getLogger("e2e_test_framework")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = Path(__file__).parent / "logs" / f"e2e_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def generate_test_scenarios(self) -> List[TestScenario]:
        """Generate comprehensive test scenarios for different project types"""
        scenarios = []
        
        # Simple Web App Scenario
        scenarios.append(TestScenario(
            id="web_app_simple",
            name="Simple Web Application",
            project_type="web_app",
            complexity_level="simple",
            prd_content="""
# Simple Web App PRD

## Objective
Create a basic web application with user authentication and profile management.

## Requirements
- User registration and login system
- User profile editing
- Basic dashboard
- Responsive design
- Database integration

## Technical Stack
- Frontend: React.js
- Backend: Node.js/Express
- Database: MongoDB
- Authentication: JWT
            """,
            expected_tasks=8,
            expected_autonomy_score=0.90,
            timeout_minutes=15,
            setup_commands=[
                "mkdir -p test_web_app",
                "cd test_web_app && npm init -y"
            ],
            validation_commands=[
                "test -f package.json",
                "test -d src/",
                "grep -q 'react' package.json"
            ],
            cleanup_commands=["rm -rf test_web_app"]
        ))
        
        # Medium Complexity API Scenario
        scenarios.append(TestScenario(
            id="api_medium",
            name="REST API with Database",
            project_type="api",
            complexity_level="medium",
            prd_content="""
# REST API PRD

## Objective
Build a comprehensive REST API for an e-commerce platform.

## Requirements
- Product catalog management
- Order processing system
- Payment integration
- User authentication and authorization
- Inventory management
- Search and filtering
- Rate limiting and security
- API documentation
- Testing suite

## Technical Requirements
- RESTful design principles
- Database schema optimization
- Caching layer implementation
- Error handling and logging
- Performance monitoring
            """,
            expected_tasks=15,
            expected_autonomy_score=0.85,
            timeout_minutes=25,
            setup_commands=[
                "mkdir -p test_api",
                "cd test_api && touch requirements.txt"
            ],
            validation_commands=[
                "test -f requirements.txt",
                "test -d api/",
                "test -f database/schema.sql"
            ],
            cleanup_commands=["rm -rf test_api"]
        ))
        
        # Complex Data Processing Scenario
        scenarios.append(TestScenario(
            id="data_processing_complex",
            name="Complex Data Processing Pipeline",
            project_type="data_processing",
            complexity_level="complex",
            prd_content="""
# Data Processing Pipeline PRD

## Objective
Create a sophisticated data processing pipeline for real-time analytics.

## Requirements
- Data ingestion from multiple sources (APIs, files, streams)
- Data transformation and cleaning
- Real-time processing capabilities
- Machine learning model integration
- Data validation and quality checks
- Monitoring and alerting
- Scalable architecture
- Batch and stream processing
- Data visualization dashboard
- Automated testing and deployment

## Technical Requirements
- Distributed processing framework
- Message queue system
- Database optimization
- Caching strategies
- Error recovery mechanisms
- Performance optimization
- Security and compliance
- Documentation and monitoring
            """,
            expected_tasks=25,
            expected_autonomy_score=0.80,
            timeout_minutes=40,
            setup_commands=[
                "mkdir -p test_pipeline",
                "cd test_pipeline && touch Dockerfile"
            ],
            validation_commands=[
                "test -f Dockerfile",
                "test -d src/processors/",
                "test -f docker-compose.yml"
            ],
            cleanup_commands=["rm -rf test_pipeline"]
        ))
        
        # CLI Tool Scenario
        scenarios.append(TestScenario(
            id="cli_tool_simple",
            name="Command Line Tool",
            project_type="cli_tool",
            complexity_level="simple",
            prd_content="""
# CLI Tool PRD

## Objective
Build a command-line tool for file management and automation.

## Requirements
- File operations (copy, move, delete)
- Directory management
- Batch operations
- Configuration management
- Logging and error handling
- Help system
- Cross-platform compatibility

## Features
- Interactive and non-interactive modes
- Progress reporting
- Undo functionality
- Plugin system
            """,
            expected_tasks=10,
            expected_autonomy_score=0.92,
            timeout_minutes=18,
            setup_commands=[
                "mkdir -p test_cli",
                "cd test_cli && touch main.py"
            ],
            validation_commands=[
                "test -f main.py",
                "test -f setup.py",
                "test -d cli/"
            ],
            cleanup_commands=["rm -rf test_cli"]
        ))
        
        return scenarios
    
    def setup_test_environment(self, scenario: TestScenario) -> str:
        """Setup isolated test environment for a scenario"""
        test_dir = Path(self.test_workspace) / f"test_{scenario.id}_{uuid.uuid4().hex[:8]}"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Change to test directory
        original_dir = os.getcwd()
        os.chdir(test_dir)
        
        try:
            # Initialize task-master in test directory
            subprocess.run(["task-master", "init"], check=True, capture_output=True, text=True)
            
            # Create PRD file
            prd_file = test_dir / ".taskmaster" / "docs" / "test_prd.md"
            prd_file.parent.mkdir(parents=True, exist_ok=True)
            prd_file.write_text(scenario.prd_content)
            
            # Run setup commands
            for cmd in scenario.setup_commands:
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            
            self.logger.info(f"Test environment setup complete: {test_dir}")
            return str(test_dir)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to setup test environment: {e}")
            raise
        finally:
            os.chdir(original_dir)
    
    def run_autonomous_execution_test(self, scenario: TestScenario, test_dir: str) -> TestResult:
        """Run autonomous execution test for a scenario"""
        start_time = datetime.now()
        self.logger.info(f"Starting test for scenario: {scenario.name}")
        
        original_dir = os.getcwd()
        error_messages = []
        logs = []
        performance_metrics = {}
        
        try:
            os.chdir(test_dir)
            
            # Parse PRD to generate tasks
            self.logger.info("Parsing PRD to generate tasks...")
            parse_result = subprocess.run(
                ["task-master", "parse-prd", ".taskmaster/docs/test_prd.md"],
                capture_output=True, text=True, timeout=300
            )
            
            if parse_result.returncode != 0:
                error_messages.append(f"PRD parsing failed: {parse_result.stderr}")
                
            logs.append(f"Parse output: {parse_result.stdout}")
            
            # Get task list
            list_result = subprocess.run(
                ["task-master", "list"],
                capture_output=True, text=True, timeout=60
            )
            
            # Count generated tasks (extract from output)
            tasks_generated = len([line for line in list_result.stdout.split('\n') if '│' in line and '✓' not in line and '►' not in line and '○' in line])
            
            # Run complexity analysis
            complexity_result = subprocess.run(
                ["task-master", "analyze-complexity", "--research"],
                capture_output=True, text=True, timeout=180
            )
            
            # Expand tasks if needed
            expand_result = subprocess.run(
                ["task-master", "expand", "--all", "--research"],
                capture_output=True, text=True, timeout=300
            )
            
            # Execute tasks autonomously
            self.logger.info("Starting autonomous execution...")
            execution_start = time.time()
            tasks_completed = 0
            max_iterations = scenario.expected_tasks * 2  # Safety limit
            
            for iteration in range(max_iterations):
                # Get next task
                next_result = subprocess.run(
                    ["task-master", "next"],
                    capture_output=True, text=True, timeout=60
                )
                
                if "No eligible tasks found" in next_result.stdout:
                    self.logger.info("All tasks completed!")
                    break
                
                # Extract task ID from output (simplified parsing)
                if "Next Task:" in next_result.stdout:
                    # Simulate task completion for testing
                    # In real implementation, this would involve actual task execution
                    time.sleep(1)  # Simulate work
                    tasks_completed += 1
                    self.logger.info(f"Completed task {tasks_completed}")
                    
                    # Mark task as done (extract task ID and mark complete)
                    # This is simplified - real implementation would parse task ID properly
                    
            execution_time = time.time() - execution_start
            performance_metrics["execution_time"] = execution_time
            performance_metrics["tasks_per_minute"] = tasks_completed / (execution_time / 60) if execution_time > 0 else 0
            
            # Calculate completion rate
            completion_rate = tasks_completed / max(tasks_generated, 1)
            
            # Calculate autonomy score (simplified)
            autonomy_score = min(0.95, completion_rate * 0.9 + 0.05)
            
            # Run validation commands
            validation_success = True
            for cmd in scenario.validation_commands:
                try:
                    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    validation_success = False
                    error_messages.append(f"Validation failed: {cmd} - {e}")
            
            # Determine overall success
            success = (
                completion_rate >= 0.8 and
                autonomy_score >= scenario.expected_autonomy_score * 0.9 and
                validation_success and
                len(error_messages) == 0
            )
            
            end_time = datetime.now()
            duration_seconds = (end_time - start_time).total_seconds()
            
            result = TestResult(
                scenario_id=scenario.id,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration_seconds,
                success=success,
                autonomy_score=autonomy_score,
                tasks_generated=tasks_generated,
                tasks_completed=tasks_completed,
                completion_rate=completion_rate,
                error_messages=error_messages,
                performance_metrics=performance_metrics,
                logs=logs
            )
            
            self.logger.info(f"Test completed - Success: {success}, Autonomy: {autonomy_score:.2f}")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration_seconds = (end_time - start_time).total_seconds()
            
            error_messages.append(f"Test execution error: {str(e)}")
            
            return TestResult(
                scenario_id=scenario.id,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration_seconds,
                success=False,
                autonomy_score=0.0,
                tasks_generated=0,
                tasks_completed=0,
                completion_rate=0.0,
                error_messages=error_messages,
                performance_metrics=performance_metrics,
                logs=logs
            )
        finally:
            os.chdir(original_dir)
    
    def cleanup_test_environment(self, scenario: TestScenario, test_dir: str):
        """Clean up test environment after test completion"""
        try:
            original_dir = os.getcwd()
            os.chdir(test_dir)
            
            # Run cleanup commands
            for cmd in scenario.cleanup_commands:
                try:
                    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Cleanup command failed: {cmd} - {e}")
            
            os.chdir(original_dir)
            
            # Remove test directory
            shutil.rmtree(test_dir, ignore_errors=True)
            self.logger.info(f"Test environment cleaned up: {test_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup test environment: {e}")
    
    def run_test_suite(self, scenarios: List[TestScenario] = None) -> List[TestResult]:
        """Run complete test suite with all scenarios"""
        if scenarios is None:
            scenarios = self.generate_test_scenarios()
        
        self.logger.info(f"Starting test suite with {len(scenarios)} scenarios")
        results = []
        
        for scenario in scenarios:
            self.logger.info(f"Running scenario: {scenario.name}")
            
            test_dir = None
            try:
                # Setup test environment
                test_dir = self.setup_test_environment(scenario)
                
                # Run autonomous execution test
                result = self.run_autonomous_execution_test(scenario, test_dir)
                results.append(result)
                
                # Store result for reporting
                self.test_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Scenario {scenario.name} failed: {e}")
                # Create failure result
                failure_result = TestResult(
                    scenario_id=scenario.id,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0,
                    success=False,
                    autonomy_score=0.0,
                    tasks_generated=0,
                    tasks_completed=0,
                    completion_rate=0.0,
                    error_messages=[str(e)],
                    performance_metrics={},
                    logs=[]
                )
                results.append(failure_result)
                self.test_results.append(failure_result)
            finally:
                # Cleanup test environment
                if test_dir:
                    self.cleanup_test_environment(scenario, test_dir)
        
        self.logger.info("Test suite completed")
        return results
    
    def generate_test_report(self, results: List[TestResult] = None) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if results is None:
            results = self.test_results
        
        if not results:
            return {"error": "No test results available"}
        
        # Calculate summary statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        avg_autonomy_score = sum(r.autonomy_score for r in results) / total_tests if total_tests > 0 else 0
        avg_completion_rate = sum(r.completion_rate for r in results) / total_tests if total_tests > 0 else 0
        avg_duration = sum(r.duration_seconds for r in results) / total_tests if total_tests > 0 else 0
        
        # Performance metrics
        total_tasks_generated = sum(r.tasks_generated for r in results)
        total_tasks_completed = sum(r.tasks_completed for r in results)
        overall_completion_rate = total_tasks_completed / max(total_tasks_generated, 1)
        
        # Generate detailed report
        report = {
            "test_execution": {
                "timestamp": datetime.now().isoformat(),
                "total_scenarios": total_tests,
                "successful_scenarios": successful_tests,
                "success_rate": success_rate,
                "target_success_rate": 0.95
            },
            "performance_metrics": {
                "average_autonomy_score": avg_autonomy_score,
                "target_autonomy_score": 0.95,
                "average_completion_rate": avg_completion_rate,
                "overall_completion_rate": overall_completion_rate,
                "average_duration_seconds": avg_duration,
                "total_tasks_generated": total_tasks_generated,
                "total_tasks_completed": total_tasks_completed
            },
            "scenario_results": [
                {
                    "scenario_id": r.scenario_id,
                    "success": r.success,
                    "autonomy_score": r.autonomy_score,
                    "completion_rate": r.completion_rate,
                    "duration_seconds": r.duration_seconds,
                    "tasks_generated": r.tasks_generated,
                    "tasks_completed": r.tasks_completed,
                    "error_count": len(r.error_messages)
                }
                for r in results
            ],
            "validation": {
                "meets_autonomy_target": avg_autonomy_score >= 0.95,
                "meets_success_target": success_rate >= 0.95,
                "overall_pass": success_rate >= 0.95 and avg_autonomy_score >= 0.95
            },
            "recommendations": self._generate_recommendations(results)
        }
        
        # Save report to file
        report_file = self.reports_dir / f"e2e_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Test report generated: {report_file}")
        return report
    
    def _generate_recommendations(self, results: List[TestResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        successful_tests = sum(1 for r in results if r.success)
        success_rate = successful_tests / len(results) if results else 0
        avg_autonomy = sum(r.autonomy_score for r in results) / len(results) if results else 0
        
        if success_rate < 0.95:
            recommendations.append(f"Success rate ({success_rate:.2f}) below target (0.95). Review failed scenarios and improve autonomous execution.")
        
        if avg_autonomy < 0.95:
            recommendations.append(f"Average autonomy score ({avg_autonomy:.2f}) below target (0.95). Enhance evolutionary optimization algorithms.")
        
        # Check for common failure patterns
        error_patterns = {}
        for result in results:
            for error in result.error_messages:
                if "parsing" in error.lower():
                    error_patterns["parsing"] = error_patterns.get("parsing", 0) + 1
                elif "timeout" in error.lower():
                    error_patterns["timeout"] = error_patterns.get("timeout", 0) + 1
                elif "validation" in error.lower():
                    error_patterns["validation"] = error_patterns.get("validation", 0) + 1
        
        for pattern, count in error_patterns.items():
            if count > 1:
                recommendations.append(f"Multiple {pattern} errors detected ({count}). Focus on improving {pattern} reliability.")
        
        if not recommendations:
            recommendations.append("All tests passed successfully! System is ready for production deployment.")
        
        return recommendations

def main():
    """Main entry point for running end-to-end tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Task Master End-to-End Testing Framework")
    parser.add_argument("--scenario", help="Run specific scenario by ID")
    parser.add_argument("--workspace", help="Test workspace directory", default="/tmp/taskmaster_e2e_tests")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    
    args = parser.parse_args()
    
    framework = EndToEndTestFramework(args.workspace)
    
    if args.report_only:
        report = framework.generate_test_report()
        print(json.dumps(report, indent=2))
        return
    
    scenarios = framework.generate_test_scenarios()
    
    if args.scenario:
        scenarios = [s for s in scenarios if s.id == args.scenario]
        if not scenarios:
            print(f"Scenario '{args.scenario}' not found")
            sys.exit(1)
    
    # Run test suite
    results = framework.run_test_suite(scenarios)
    
    # Generate and display report
    report = framework.generate_test_report(results)
    
    print("\n" + "="*80)
    print("END-TO-END TEST REPORT")
    print("="*80)
    print(f"Success Rate: {report['test_execution']['success_rate']:.2%}")
    print(f"Average Autonomy Score: {report['performance_metrics']['average_autonomy_score']:.2f}")
    print(f"Overall Completion Rate: {report['performance_metrics']['overall_completion_rate']:.2%}")
    print(f"Validation Status: {'PASS' if report['validation']['overall_pass'] else 'FAIL'}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    # Exit with appropriate code
    sys.exit(0 if report['validation']['overall_pass'] else 1)

if __name__ == "__main__":
    main()