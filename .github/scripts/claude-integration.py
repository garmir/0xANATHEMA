#!/usr/bin/env python3
"""
Claude API Integration for GitHub Actions
Provides enhanced Claude integration with task execution, error handling, and result reporting
"""

import os
import sys
import json
import time
import asyncio
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import anthropic
import httpx

@dataclass
class TaskExecutionResult:
    task_id: str
    success: bool
    execution_time: float
    claude_response: str
    error_message: str = ""
    artifacts_created: List[str] = None
    tests_passed: bool = False
    complexity_score: int = 1
    runner_id: str = ""
    timestamp: str = ""

@dataclass
class ClaudePromptTemplate:
    task_execution: str
    error_recovery: str
    test_validation: str
    code_review: str

class ClaudeTaskExecutor:
    """Enhanced Claude integration for GitHub Actions task execution"""
    
    def __init__(self, api_key: str = None, runner_id: str = "unknown"):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
            
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.runner_id = runner_id
        self.logger = self._setup_logging()
        self.templates = self._load_prompt_templates()
        self.execution_history = []
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"ClaudeExecutor-{self.runner_id}")
        logger.setLevel(logging.INFO)
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'[%(asctime)s] [Runner-{self.runner_id}] %(levelname)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_prompt_templates(self) -> ClaudePromptTemplate:
        """Load optimized prompt templates for different task types"""
        return ClaudePromptTemplate(
            task_execution="""You are a GitHub Actions runner executing a Task Master AI task with full autonomous capabilities.

EXECUTION CONTEXT:
- Runner ID: {runner_id}
- Task ID: {task_id}
- Environment: Ubuntu Linux with Node.js, Python, git, task-master CLI
- Available APIs: task-master, git, npm, pip, curl, jq
- Working Directory: {working_dir}
- Execution Time Limit: {time_limit} minutes

TASK DETAILS:
Title: {title}
Description: {description}
Details: {details}
Priority: {priority}
Dependencies: {dependencies}
Test Strategy: {test_strategy}

AUTONOMOUS CAPABILITIES:
- You can execute shell commands by describing them
- You can read/write files by describing the operations  
- You can use task-master CLI for research if stuck
- You can install packages and dependencies as needed
- You have access to the autonomous research loop if errors occur

EXECUTION REQUIREMENTS:
1. Analyze the task and create an execution plan
2. Execute the plan step by step
3. Handle any errors using the autonomous research loop
4. Validate results using the test strategy
5. Report final status with detailed execution log

Start your response with either "SUCCESS:" or "FAILURE:" followed by:
- Execution plan
- Step-by-step progress log
- Any errors encountered and how they were resolved
- Validation results
- Files created/modified
- Final status summary

Execute this task completely and autonomously.""",

            error_recovery="""AUTONOMOUS ERROR RECOVERY MODE

You encountered an error during task execution:
ERROR: {error_details}
TASK: {task_id} - {title}
CONTEXT: {execution_context}

Use the autonomous research loop to resolve this error:

1. RESEARCH: Analyze the error and research solutions using task-master with Perplexity
2. SOLUTION: Generate step-by-step solution from research findings  
3. EXECUTE: Apply the solution and validate it works
4. RETRY: Retry the original task execution

Available commands:
- task-master add-task --prompt="Research solution for: [ERROR]" --research
- task-master parse-prd --append [research-results.md]
- task-master next (to execute research solutions)

Continue execution until the original task succeeds or determine if manual intervention is needed.""",

            test_validation="""TASK VALIDATION AND TESTING

Task: {task_id} - {title}
Test Strategy: {test_strategy}
Implementation Details: {implementation_summary}

Execute the test strategy to validate the task implementation:

1. Review the test strategy requirements
2. Execute each test case or validation step
3. Verify all success criteria are met
4. Report any test failures or issues
5. Provide final validation status

Return results in format:
VALIDATION: [PASSED/FAILED]
Test Results:
- [Test 1]: [PASS/FAIL] - [Details]
- [Test 2]: [PASS/FAIL] - [Details]
Summary: [Overall assessment]""",

            code_review="""CODE REVIEW AND QUALITY ASSESSMENT

Task: {task_id} - {title}
Implementation: {code_changes}

Perform automated code review:

1. Check code quality and best practices
2. Verify security considerations
3. Assess performance implications
4. Validate error handling
5. Check documentation completeness

Return review in format:
REVIEW: [APPROVED/NEEDS_WORK]
Quality Score: [1-10]
Issues Found:
- [Category]: [Description] - [Severity]
Recommendations:
- [Recommendation 1]
- [Recommendation 2]"""
        )
    
    async def execute_task_with_claude(self, task_data: Dict[str, Any]) -> TaskExecutionResult:
        """Execute a single task using Claude with full error handling"""
        task_id = str(task_data.get('id', 'unknown'))
        start_time = time.time()
        
        self.logger.info(f"🎯 Starting execution of task {task_id}: {task_data.get('title', 'Unknown')}")
        
        try:
            # Prepare execution context
            context = {
                'runner_id': self.runner_id,
                'task_id': task_id,
                'title': task_data.get('title', 'Unknown'),
                'description': task_data.get('description', ''),
                'details': task_data.get('details', ''),
                'priority': task_data.get('priority', 'medium'),
                'dependencies': task_data.get('dependencies', []),
                'test_strategy': task_data.get('testStrategy', ''),
                'working_dir': os.getcwd(),
                'time_limit': 30
            }
            
            # Execute main task
            result = await self._execute_with_retries(context)
            
            execution_time = time.time() - start_time
            
            # Validate results if test strategy exists
            if context['test_strategy'] and result.success:
                validation_result = await self._validate_task_results(context, result)
                result.tests_passed = validation_result
            
            result.execution_time = execution_time
            result.timestamp = datetime.now().isoformat()
            
            self.execution_history.append(result)
            
            if result.success:
                self.logger.info(f"✅ Task {task_id} completed successfully in {execution_time:.2f}s")
            else:
                self.logger.error(f"❌ Task {task_id} failed after {execution_time:.2f}s")
                
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"💥 Unexpected error executing task {task_id}: {e}")
            
            return TaskExecutionResult(
                task_id=task_id,
                success=False,
                execution_time=execution_time,
                claude_response="",
                error_message=str(e),
                runner_id=self.runner_id,
                timestamp=datetime.now().isoformat()
            )
    
    async def _execute_with_retries(self, context: Dict[str, Any], max_retries: int = 3) -> TaskExecutionResult:
        """Execute task with automatic retry and error recovery"""
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"🔄 Execution attempt {attempt + 1}/{max_retries}")
                
                # Format prompt
                prompt = self.templates.task_execution.format(**context)
                
                # Execute with Claude
                response = await self._call_claude_async(prompt, max_tokens=4000)
                
                # Parse response
                success = response.strip().startswith("SUCCESS:")
                
                result = TaskExecutionResult(
                    task_id=context['task_id'],
                    success=success,
                    execution_time=0,  # Will be set by caller
                    claude_response=response,
                    runner_id=self.runner_id,
                    complexity_score=self._calculate_response_complexity(response)
                )
                
                if success:
                    return result
                else:
                    # Task failed, try error recovery
                    if attempt < max_retries - 1:
                        self.logger.warning(f"⚠️ Task failed, attempting error recovery...")
                        recovery_result = await self._attempt_error_recovery(context, response)
                        if recovery_result:
                            continue  # Retry with recovered state
                    
                    return result
                    
            except Exception as e:
                self.logger.error(f"💥 Execution attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return TaskExecutionResult(
                        task_id=context['task_id'],
                        success=False,
                        execution_time=0,
                        claude_response="",
                        error_message=str(e),
                        runner_id=self.runner_id
                    )
                
                # Wait before retry
                await asyncio.sleep(5 * (attempt + 1))
    
    async def _attempt_error_recovery(self, context: Dict[str, Any], error_response: str) -> bool:
        """Attempt to recover from task execution error using autonomous research loop"""
        
        self.logger.info("🔧 Attempting autonomous error recovery...")
        
        try:
            # Extract error details from Claude response
            error_details = self._extract_error_details(error_response)
            
            # Use error recovery template
            recovery_context = {
                **context,
                'error_details': error_details,
                'execution_context': error_response[:500]  # First 500 chars
            }
            
            recovery_prompt = self.templates.error_recovery.format(**recovery_context)
            
            # Execute recovery with Claude
            recovery_response = await self._call_claude_async(recovery_prompt, max_tokens=3000)
            
            # Check if recovery was successful
            recovery_success = "RECOVERED" in recovery_response.upper() or "RESOLVED" in recovery_response.upper()
            
            if recovery_success:
                self.logger.info("✅ Error recovery successful")
                return True
            else:
                self.logger.warning("⚠️ Error recovery unsuccessful")
                return False
                
        except Exception as e:
            self.logger.error(f"💥 Error recovery failed: {e}")
            return False
    
    async def _validate_task_results(self, context: Dict[str, Any], result: TaskExecutionResult) -> bool:
        """Validate task results using test strategy"""
        
        if not context['test_strategy']:
            return True
            
        self.logger.info("🔍 Validating task results...")
        
        try:
            validation_context = {
                'task_id': context['task_id'],
                'title': context['title'],
                'test_strategy': context['test_strategy'],
                'implementation_summary': result.claude_response[:1000]
            }
            
            validation_prompt = self.templates.test_validation.format(**validation_context)
            validation_response = await self._call_claude_async(validation_prompt, max_tokens=2000)
            
            # Parse validation results
            validation_passed = "VALIDATION: PASSED" in validation_response
            
            if validation_passed:
                self.logger.info("✅ Task validation passed")
            else:
                self.logger.warning("⚠️ Task validation failed")
                
            return validation_passed
            
        except Exception as e:
            self.logger.error(f"💥 Validation failed: {e}")
            return False
    
    async def _call_claude_async(self, prompt: str, max_tokens: int = 3000) -> str:
        """Make async call to Claude API with rate limiting"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except anthropic.RateLimitError as e:
            self.logger.warning("⏳ Rate limit hit, waiting 60 seconds...")
            await asyncio.sleep(60)
            return await self._call_claude_async(prompt, max_tokens)
            
        except anthropic.APIError as e:
            self.logger.error(f"💥 Claude API error: {e}")
            raise
    
    def _extract_error_details(self, response: str) -> str:
        """Extract error details from Claude response"""
        lines = response.split('\n')
        error_lines = [line for line in lines if 'error' in line.lower() or 'fail' in line.lower()]
        return '\n'.join(error_lines[:5])  # First 5 error lines
    
    def _calculate_response_complexity(self, response: str) -> int:
        """Calculate complexity score based on response content"""
        complexity = 1
        
        # Length factor
        complexity += len(response) // 500
        
        # Technical terms
        technical_terms = ['algorithm', 'implementation', 'configuration', 'integration', 'optimization']
        complexity += sum(1 for term in technical_terms if term in response.lower())
        
        # Code blocks
        complexity += response.count('```') // 2
        
        return min(complexity, 10)
    
    async def execute_multiple_tasks(self, task_list: List[Dict[str, Any]]) -> List[TaskExecutionResult]:
        """Execute multiple tasks concurrently with resource management"""
        
        self.logger.info(f"🚀 Executing {len(task_list)} tasks concurrently")
        
        # Limit concurrency to prevent resource exhaustion
        semaphore = asyncio.Semaphore(3)
        
        async def execute_with_semaphore(task_data):
            async with semaphore:
                return await self.execute_task_with_claude(task_data)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in task_list],
            return_exceptions=True
        )
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"💥 Task {i} failed with exception: {result}")
                final_results.append(TaskExecutionResult(
                    task_id=str(task_list[i].get('id', i)),
                    success=False,
                    execution_time=0,
                    claude_response="",
                    error_message=str(result),
                    runner_id=self.runner_id,
                    timestamp=datetime.now().isoformat()
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def generate_execution_report(self, results: List[TaskExecutionResult]) -> Dict[str, Any]:
        """Generate comprehensive execution report"""
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        report = {
            'runner_id': self.runner_id,
            'timestamp': datetime.now().isoformat(),
            'execution_summary': {
                'total_tasks': len(results),
                'successful_tasks': len(successful),
                'failed_tasks': len(failed),
                'success_rate': len(successful) / len(results) if results else 0,
                'total_execution_time': sum(r.execution_time for r in results),
                'average_execution_time': sum(r.execution_time for r in results) / len(results) if results else 0
            },
            'task_results': [asdict(result) for result in results],
            'performance_metrics': {
                'fastest_task': min(results, key=lambda x: x.execution_time).task_id if results else None,
                'slowest_task': max(results, key=lambda x: x.execution_time).task_id if results else None,
                'average_complexity': sum(r.complexity_score for r in results) / len(results) if results else 0,
                'tests_passed_count': sum(1 for r in results if r.tests_passed)
            },
            'error_analysis': {
                'common_errors': self._analyze_common_errors(failed),
                'recovery_attempts': len([r for r in results if 'recovery' in r.claude_response.lower()]),
                'timeout_failures': len([r for r in failed if 'timeout' in r.error_message.lower()])
            }
        }
        
        return report
    
    def _analyze_common_errors(self, failed_results: List[TaskExecutionResult]) -> Dict[str, int]:
        """Analyze common error patterns"""
        error_patterns = {}
        
        for result in failed_results:
            error_text = result.error_message.lower()
            
            if 'permission' in error_text:
                error_patterns['permission_errors'] = error_patterns.get('permission_errors', 0) + 1
            elif 'network' in error_text or 'connection' in error_text:
                error_patterns['network_errors'] = error_patterns.get('network_errors', 0) + 1
            elif 'timeout' in error_text:
                error_patterns['timeout_errors'] = error_patterns.get('timeout_errors', 0) + 1
            elif 'dependency' in error_text:
                error_patterns['dependency_errors'] = error_patterns.get('dependency_errors', 0) + 1
            else:
                error_patterns['other_errors'] = error_patterns.get('other_errors', 0) + 1
        
        return error_patterns

async def main():
    """Main entry point for CLI usage"""
    if len(sys.argv) < 2:
        print("Usage: python claude-integration.py <task_id> [task_id2] ...")
        sys.exit(1)
    
    task_ids = sys.argv[1:]
    runner_id = os.getenv('RUNNER_ID', 'local')
    
    # Load tasks data
    try:
        with open('.taskmaster/tasks/tasks.json', 'r') as f:
            tasks_data = json.load(f)
        
        task_map = {str(task['id']): task for task in tasks_data['master']['tasks']}
        
        # Get tasks to execute
        tasks_to_execute = []
        for task_id in task_ids:
            if task_id in task_map:
                tasks_to_execute.append(task_map[task_id])
            else:
                print(f"⚠️ Task {task_id} not found")
        
        if not tasks_to_execute:
            print("❌ No valid tasks to execute")
            sys.exit(1)
        
        # Execute tasks
        executor = ClaudeTaskExecutor(runner_id=runner_id)
        results = await executor.execute_multiple_tasks(tasks_to_execute)
        
        # Generate report
        report = executor.generate_execution_report(results)
        
        # Save results
        output_file = f"claude_execution_results_{runner_id}.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Execution complete. Results saved to {output_file}")
        print(f"📊 Success rate: {report['execution_summary']['success_rate']:.1%}")
        
        # Exit with appropriate code
        sys.exit(0 if report['execution_summary']['failed_tasks'] == 0 else 1)
        
    except Exception as e:
        print(f"💥 Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())