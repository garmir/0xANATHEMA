# Claude Code & MCP Integration Patterns

## Overview

This document provides comprehensive integration patterns for connecting Task-Master with Claude Code and MCP (Model Context Protocol) to achieve autonomous execution workflows.

## MCP Server Configuration

### Basic Setup

```json
// .mcp.json - Project-specific MCP configuration
{
  "mcpServers": {
    "task-master-ai": {
      "command": "npx",
      "args": ["-y", "--package=task-master-ai", "task-master-ai"],
      "env": {
        "ANTHROPIC_API_KEY": "your_anthropic_key",
        "PERPLEXITY_API_KEY": "your_perplexity_key",
        "OPENAI_API_KEY": "your_openai_key",
        "GOOGLE_API_KEY": "your_google_key"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"]
    },
    "git": {
      "command": "npx", 
      "args": ["-y", "@modelcontextprotocol/server-git"]
    }
  }
}
```

### Advanced Configuration with Error Handling

```json
// .mcp.json - Production configuration
{
  "mcpServers": {
    "task-master-ai": {
      "command": "npx",
      "args": ["-y", "--package=task-master-ai", "task-master-ai"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}",
        "TASKMASTER_HOME": "${PWD}/.taskmaster",
        "TASKMASTER_LOG_LEVEL": "info",
        "TASKMASTER_RETRY_COUNT": "3",
        "TASKMASTER_TIMEOUT": "300"
      },
      "settings": {
        "retryOnFailure": true,
        "maxRetries": 3,
        "timeoutMs": 300000
      }
    }
  }
}
```

## Claude Code Configuration

### Tool Allowlist Configuration

```json
// .claude/settings.json
{
  "allowedTools": [
    "Edit",
    "MultiEdit", 
    "Read",
    "Write",
    "Bash(task-master *)",
    "Bash(git add *)",
    "Bash(git commit *)",
    "Bash(npm run *)",
    "Bash(python3 .taskmaster/scripts/*)",
    "mcp__task_master_ai__*",
    "mcp__filesystem__*",
    "mcp__git__*"
  ],
  "hooks": {
    "beforeEdit": "task-master update-subtask --id=${CURRENT_TASK_ID} --prompt='Starting edit: ${FILE_PATH}'",
    "afterCommit": "task-master update-subtask --id=${CURRENT_TASK_ID} --prompt='Committed changes: ${COMMIT_MESSAGE}'"
  },
  "preferences": {
    "autoSave": true,
    "verboseLogging": false,
    "parallelToolCalls": true
  }
}
```

### Custom Slash Commands

```markdown
<!-- .claude/commands/taskmaster-workflow.md -->
Execute complete Task-Master workflow: $ARGUMENTS

This command executes the full autonomous workflow loop.

Steps:
1. Get next available task: task-master next
2. If task available:
   - Show task details: task-master show <id>
   - Mark as in-progress: task-master set-status --id=<id> --status=in-progress
   - Implement the task following the details
   - Run tests and validation
   - Mark as done: task-master set-status --id=<id> --status=done
3. Continue with next task until all complete
4. Generate final report: task-master complexity-report
```

```markdown
<!-- .claude/commands/taskmaster-debug.md -->
Debug Task-Master execution issues: $ARGUMENTS

Steps:
1. Check task status: task-master list
2. Validate dependencies: task-master validate-dependencies
3. Check logs: cat .taskmaster/logs/latest.log
4. Verify configuration: cat .taskmaster/config.json
5. Test MCP connection: using MCP tools
6. Generate debug report with findings
```

## Integration Patterns

### Pattern 1: Sequential Execution

```python
#!/usr/bin/env python3
"""
Sequential Task Execution Pattern
Implements one-task-at-a-time execution with full validation
"""

import subprocess
import json
import time
from pathlib import Path

class SequentialTaskExecutor:
    def __init__(self, workspace_dir: str = ".taskmaster"):
        self.workspace = Path(workspace_dir)
        self.current_task = None
    
    def get_next_task(self):
        """Get next available task from Task-Master"""
        result = subprocess.run(
            ["task-master", "next", "--json"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            try:
                task_data = json.loads(result.stdout)
                return task_data.get('id')
            except json.JSONDecodeError:
                return None
        return None
    
    def execute_task_with_claude(self, task_id: str):
        """Execute task using Claude Code"""
        # Mark task as in-progress
        subprocess.run([
            "task-master", "set-status", 
            f"--id={task_id}", "--status=in-progress"
        ])
        
        # Get task details
        task_details = subprocess.run([
            "task-master", "show", task_id
        ], capture_output=True, text=True)
        
        # Execute with Claude Code
        claude_prompt = f"""
        Please implement Task {task_id}.
        
        Task Details:
        {task_details.stdout}
        
        Follow these steps:
        1. Read and understand the task requirements
        2. Plan the implementation approach
        3. Implement the solution
        4. Test the implementation
        5. Update task with implementation notes: task-master update-subtask --id={task_id} --prompt="implementation completed"
        """
        
        # Call Claude Code (simplified - in practice use MCP)
        claude_result = subprocess.run([
            "claude", "-p", claude_prompt
        ], capture_output=True, text=True)
        
        return claude_result.returncode == 0
    
    def validate_task_completion(self, task_id: str) -> bool:
        """Validate task completion"""
        # Run project tests
        test_result = subprocess.run(
            ["npm", "run", "test"], 
            capture_output=True
        )
        
        # Run linting
        lint_result = subprocess.run(
            ["npm", "run", "lint"],
            capture_output=True
        )
        
        return test_result.returncode == 0 and lint_result.returncode == 0
    
    def run_workflow(self):
        """Execute complete workflow"""
        while True:
            task_id = self.get_next_task()
            
            if not task_id:
                print("🎉 All tasks completed!")
                break
            
            print(f"🚀 Executing task {task_id}")
            
            # Execute task
            if self.execute_task_with_claude(task_id):
                # Validate completion
                if self.validate_task_completion(task_id):
                    subprocess.run([
                        "task-master", "set-status",
                        f"--id={task_id}", "--status=done"
                    ])
                    print(f"✅ Task {task_id} completed")
                else:
                    print(f"❌ Task {task_id} failed validation")
                    break
            else:
                print(f"❌ Task {task_id} execution failed")
                break

if __name__ == "__main__":
    executor = SequentialTaskExecutor()
    executor.run_workflow()
```

### Pattern 2: Parallel Execution with MCP

```python
#!/usr/bin/env python3
"""
Parallel Task Execution Pattern using MCP
Implements concurrent task execution for independent tasks
"""

import asyncio
import json
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TaskContext:
    id: str
    dependencies: List[str]
    status: str
    details: str

class ParallelMCPExecutor:
    def __init__(self):
        self.mcp_client = None  # Initialize MCP client
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
    async def initialize_mcp(self):
        """Initialize MCP connection"""
        # Connect to task-master-ai MCP server
        # Implementation depends on MCP client library
        pass
    
    async def get_available_tasks(self) -> List[TaskContext]:
        """Get all tasks that can be executed (dependencies met)"""
        # Use MCP to call get_tasks
        tasks_result = await self.mcp_client.call_tool(
            "get_tasks",
            {"filter": "ready"}
        )
        
        return [TaskContext(**task) for task in tasks_result]
    
    async def execute_task_parallel(self, task: TaskContext):
        """Execute single task in parallel"""
        try:
            # Mark as in-progress via MCP
            await self.mcp_client.call_tool(
                "set_task_status",
                {"id": task.id, "status": "in-progress"}
            )
            
            # Execute task via MCP
            result = await self.mcp_client.call_tool(
                "execute_task_autonomous",
                {
                    "id": task.id,
                    "details": task.details,
                    "validation_required": True
                }
            )
            
            if result["success"]:
                await self.mcp_client.call_tool(
                    "set_task_status",
                    {"id": task.id, "status": "done"}
                )
                print(f"✅ Task {task.id} completed successfully")
            else:
                print(f"❌ Task {task.id} failed: {result['error']}")
                
        except Exception as e:
            print(f"⚠️ Task {task.id} error: {e}")
    
    async def run_parallel_workflow(self):
        """Execute workflow with parallel task execution"""
        await self.initialize_mcp()
        
        while True:
            # Get available tasks
            available_tasks = await self.get_available_tasks()
            
            if not available_tasks:
                # Check if any tasks are still running
                if not self.active_tasks:
                    print("🎉 All tasks completed!")
                    break
                else:
                    # Wait for running tasks
                    await asyncio.sleep(5)
                    continue
            
            # Start new tasks (limit concurrency)
            max_concurrent = 3
            tasks_to_start = available_tasks[:max_concurrent - len(self.active_tasks)]
            
            for task in tasks_to_start:
                if task.id not in self.active_tasks:
                    async_task = asyncio.create_task(
                        self.execute_task_parallel(task)
                    )
                    self.active_tasks[task.id] = async_task
                    print(f"🚀 Started task {task.id}")
            
            # Clean up completed tasks
            completed_tasks = []
            for task_id, async_task in self.active_tasks.items():
                if async_task.done():
                    completed_tasks.append(task_id)
            
            for task_id in completed_tasks:
                del self.active_tasks[task_id]
            
            await asyncio.sleep(2)

# Usage
async def main():
    executor = ParallelMCPExecutor()
    await executor.run_parallel_workflow()

if __name__ == "__main__":
    asyncio.run(main())
```

### Pattern 3: Intelligent Task Routing

```python
#!/usr/bin/env python3
"""
Intelligent Task Routing Pattern
Routes tasks to appropriate execution strategies based on complexity
"""

from enum import Enum
from typing import Dict, Any, Callable
import re

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium" 
    COMPLEX = "complex"
    CRITICAL = "critical"

class IntelligentTaskRouter:
    def __init__(self):
        self.complexity_patterns = {
            TaskComplexity.SIMPLE: [
                r"update.*documentation",
                r"add.*comment",
                r"fix.*typo",
                r"rename.*variable"
            ],
            TaskComplexity.MEDIUM: [
                r"implement.*function",
                r"add.*test",
                r"refactor.*component",
                r"optimize.*query"
            ],
            TaskComplexity.COMPLEX: [
                r"design.*architecture",
                r"implement.*system",
                r"integrate.*service",
                r"migrate.*database"
            ],
            TaskComplexity.CRITICAL: [
                r"security.*fix",
                r"production.*issue",
                r"data.*recovery",
                r"performance.*critical"
            ]
        }
        
        self.execution_strategies = {
            TaskComplexity.SIMPLE: self.execute_simple_task,
            TaskComplexity.MEDIUM: self.execute_medium_task,
            TaskComplexity.COMPLEX: self.execute_complex_task,
            TaskComplexity.CRITICAL: self.execute_critical_task
        }
    
    def classify_task_complexity(self, task_description: str) -> TaskComplexity:
        """Classify task complexity based on description"""
        description_lower = task_description.lower()
        
        for complexity, patterns in self.complexity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, description_lower):
                    return complexity
        
        # Default to medium complexity
        return TaskComplexity.MEDIUM
    
    def execute_simple_task(self, task_id: str) -> bool:
        """Execute simple task with minimal validation"""
        # Use fast, automated execution
        return self.mcp_execute_autonomous(task_id, validation_level="basic")
    
    def execute_medium_task(self, task_id: str) -> bool:
        """Execute medium task with standard validation"""
        # Use standard Claude Code execution with testing
        return self.claude_execute_with_tests(task_id)
    
    def execute_complex_task(self, task_id: str) -> bool:
        """Execute complex task with comprehensive validation"""
        # Break into subtasks, execute with reviews
        self.break_into_subtasks(task_id)
        return self.execute_with_peer_review(task_id)
    
    def execute_critical_task(self, task_id: str) -> bool:
        """Execute critical task with maximum validation"""
        # Human oversight required
        return self.execute_with_human_oversight(task_id)
    
    def route_and_execute(self, task_id: str, task_description: str) -> bool:
        """Route task to appropriate execution strategy"""
        complexity = self.classify_task_complexity(task_description)
        strategy = self.execution_strategies[complexity]
        
        print(f"🎯 Task {task_id} classified as {complexity.value}")
        print(f"🚀 Executing with {strategy.__name__}")
        
        return strategy(task_id)
```

## Error Handling and Resilience

### Retry Logic with Exponential Backoff

```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1, max_delay=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                    print(f"🔄 Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            
        return wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=3)
def mcp_call_with_retry(tool_name, params):
    return mcp_client.call_tool(tool_name, params)
```

### Circuit Breaker Pattern

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
mcp_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)

def safe_mcp_call(tool_name, params):
    return mcp_circuit_breaker.call(mcp_client.call_tool, tool_name, params)
```

## Monitoring and Observability

### Integration Metrics Collection

```python
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class IntegrationMetrics:
    timestamp: float
    task_id: str
    execution_time: float
    success: bool
    tool_calls: int
    memory_usage: float
    error_message: str = None

class MetricsCollector:
    def __init__(self, metrics_file: str = ".taskmaster/metrics/integration.jsonl"):
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def record_task_execution(self, task_id: str, execution_time: float, 
                            success: bool, tool_calls: int, memory_usage: float,
                            error_message: str = None):
        """Record task execution metrics"""
        metrics = IntegrationMetrics(
            timestamp=time.time(),
            task_id=task_id,
            execution_time=execution_time,
            success=success,
            tool_calls=tool_calls,
            memory_usage=memory_usage,
            error_message=error_message
        )
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        if not self.metrics_file.exists():
            return {}
        
        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line.strip()))
        
        if not metrics:
            return {}
        
        total_tasks = len(metrics)
        successful_tasks = sum(1 for m in metrics if m['success'])
        avg_execution_time = sum(m['execution_time'] for m in metrics) / total_tasks
        avg_tool_calls = sum(m['tool_calls'] for m in metrics) / total_tasks
        
        return {
            'total_tasks': total_tasks,
            'success_rate': successful_tasks / total_tasks,
            'avg_execution_time': avg_execution_time,
            'avg_tool_calls': avg_tool_calls,
            'total_time': sum(m['execution_time'] for m in metrics)
        }

# Usage
metrics = MetricsCollector()
start_time = time.time()

try:
    # Execute task
    result = execute_task(task_id)
    execution_time = time.time() - start_time
    
    metrics.record_task_execution(
        task_id=task_id,
        execution_time=execution_time,
        success=True,
        tool_calls=5,
        memory_usage=get_memory_usage()
    )
except Exception as e:
    execution_time = time.time() - start_time
    metrics.record_task_execution(
        task_id=task_id,
        execution_time=execution_time,
        success=False,
        tool_calls=3,
        memory_usage=get_memory_usage(),
        error_message=str(e)
    )
```

## Best Practices Summary

### 1. Configuration Management
- Use environment variables for sensitive data
- Version control configuration templates
- Test configurations in isolation

### 2. Error Handling
- Implement retry logic with exponential backoff
- Use circuit breakers for external services
- Log errors with context for debugging

### 3. Performance Optimization
- Cache MCP responses where appropriate
- Use parallel execution for independent tasks
- Monitor resource usage and optimize accordingly

### 4. Security Considerations
- Never commit API keys to version control
- Use least-privilege access for tools
- Validate all inputs and outputs

### 5. Testing and Validation
- Test integration patterns in isolation
- Validate task completion with automated tests
- Monitor success rates and performance metrics

This integration framework provides robust, scalable patterns for connecting Task-Master with Claude Code and MCP to achieve autonomous execution workflows.