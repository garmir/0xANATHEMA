# Task-Master Integration Best Practices

## Overview

This document outlines best practices for integrating Task-Master with various development workflows, ensuring optimal performance, reliability, and autonomous execution capabilities.

## General Best Practices

### 1. Project Structure Organization

```
project-root/
├── .taskmaster/
│   ├── config.json              # AI models and settings
│   ├── docs/                    # PRD files and documentation
│   │   ├── prd.txt             # Main project requirements
│   │   └── sub-prds/           # Recursive decomposition results
│   ├── scripts/                 # Automation scripts
│   │   ├── claude-flow-integration.py
│   │   ├── catalytic-workspace-system.py
│   │   └── end-to-end-testing-framework.py
│   ├── tasks/                   # Task management
│   │   ├── tasks.json          # Main task database
│   │   └── *.md                # Individual task files
│   ├── artifacts/               # Optimization artifacts
│   │   ├── sqrt-space/         # Space optimization results
│   │   ├── tree-eval/          # Tree evaluation artifacts
│   │   └── pebbling/           # Resource allocation strategies
│   ├── logs/                    # Execution logs
│   ├── reports/                 # Analysis reports
│   └── templates/               # Reusable templates
├── .claude/
│   ├── settings.json           # Claude Code configuration
│   └── commands/               # Custom slash commands
├── .mcp.json                   # MCP server configuration
├── .env                        # Environment variables (API keys)
└── CLAUDE.md                   # Auto-loaded context
```

### 2. Configuration Management

#### API Key Security
```bash
# Never commit API keys to version control
echo ".env" >> .gitignore

# Use environment variables
cat > .env << EOF
ANTHROPIC_API_KEY=your_key_here
PERPLEXITY_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
EOF

# Load in scripts
source .env
export ANTHROPIC_API_KEY
export PERPLEXITY_API_KEY
```

#### Model Configuration Best Practices
```json
{
  "models": {
    "main": {
      "provider": "anthropic",
      "modelId": "claude-3-5-sonnet-20241022",
      "maxTokens": 8000,
      "temperature": 0.1
    },
    "research": {
      "provider": "perplexity",
      "modelId": "sonar-pro",
      "maxTokens": 4000,
      "temperature": 0.1
    },
    "fallback": {
      "provider": "openai",
      "modelId": "gpt-4o-mini",
      "maxTokens": 4000,
      "temperature": 0.2
    }
  },
  "global": {
    "logLevel": "info",
    "defaultSubtasks": 6,
    "defaultPriority": "medium",
    "responseLanguage": "English"
  }
}
```

### 3. Task Decomposition Best Practices

#### Optimal Task Granularity
- **Atomic Tasks**: 2-4 hours of work maximum
- **Clear Dependencies**: Explicitly define task dependencies
- **Measurable Outcomes**: Each task should have clear success criteria

```bash
# Good task decomposition
task-master expand --id=1 --research --max-depth=3

# Avoid over-decomposition
# Bad: Tasks that take < 30 minutes
# Good: Tasks that take 2-4 hours
```

#### PRD Writing Guidelines
```markdown
# Good PRD Structure

## Objective
Single, clear goal statement

## Core Features
Numbered list of main features (5-10 items max)

## Technical Requirements
Specific technologies and frameworks

## Performance Requirements
Quantifiable metrics and benchmarks

## Success Criteria
Measurable outcomes for completion
```

### 4. Claude Code Integration Best Practices

#### Tool Allowlist Configuration
```json
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
    "Bash(pytest *)",
    "mcp__task_master_ai__*"
  ],
  "hooks": {
    "beforeEdit": "task-master update-subtask --id=${CURRENT_TASK_ID} --prompt='Starting implementation'",
    "afterCommit": "task-master update-subtask --id=${CURRENT_TASK_ID} --prompt='Changes committed'"
  }
}
```

#### Custom Slash Commands Best Practices
```markdown
<!-- .claude/commands/taskmaster-workflow.md -->
Execute complete task workflow: $ARGUMENTS

This command implements the standard Task-Master workflow pattern.

Steps:
1. Get task details: task-master show $ARGUMENTS
2. Mark as in-progress: task-master set-status --id=$ARGUMENTS --status=in-progress
3. Implement the solution following task requirements
4. Run tests and validation
5. Update with implementation notes: task-master update-subtask --id=$ARGUMENTS --prompt="Implementation completed"
6. Mark as done: task-master set-status --id=$ARGUMENTS --status=done
7. Get next task: task-master next
```

### 5. MCP Integration Best Practices

#### Server Configuration
```json
{
  "mcpServers": {
    "task-master-ai": {
      "command": "npx",
      "args": ["-y", "--package=task-master-ai", "task-master-ai"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}",
        "TASKMASTER_HOME": "${PWD}/.taskmaster",
        "TASKMASTER_LOG_LEVEL": "info"
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

#### Error Handling and Resilience
```python
# Implement retry logic for MCP calls
import time
import random

def retry_mcp_call(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
```

### 6. Performance Optimization Best Practices

#### Memory Management
```python
# Use catalytic workspace for memory-intensive operations
def setup_catalytic_workspace():
    subprocess.run([
        "python3", ".taskmaster/scripts/catalytic-workspace-system.py"
    ])

# Monitor memory usage
import psutil

def check_memory_usage():
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        print("⚠️ High memory usage detected")
        # Trigger cleanup or optimization
```

#### Complexity Optimization
```bash
# Regular complexity analysis
task-master analyze-complexity --research

# Apply optimizations
python3 .taskmaster/scripts/task-complexity-analyzer.py

# Validate improvements
task-master complexity-report
```

### 7. Testing and Validation Best Practices

#### Comprehensive Testing Strategy
```python
# Test framework integration
def validate_task_completion(task_id: str) -> bool:
    """Validate task completion with multiple checks"""
    
    # 1. Run unit tests
    unit_test_result = subprocess.run(["npm", "run", "test:unit"])
    
    # 2. Run integration tests
    integration_test_result = subprocess.run(["npm", "run", "test:integration"])
    
    # 3. Run linting
    lint_result = subprocess.run(["npm", "run", "lint"])
    
    # 4. Check code coverage
    coverage_result = subprocess.run(["npm", "run", "test:coverage"])
    
    # 5. Validate task-specific criteria
    task_validation = validate_task_specific_criteria(task_id)
    
    return all([
        unit_test_result.returncode == 0,
        integration_test_result.returncode == 0,
        lint_result.returncode == 0,
        coverage_result.returncode == 0,
        task_validation
    ])
```

#### End-to-End Testing
```bash
# Regular E2E testing
python3 .taskmaster/scripts/end-to-end-testing-framework.py

# Automated validation
task-master validate-dependencies
```

### 8. Monitoring and Observability Best Practices

#### Metrics Collection
```python
import json
import time
from pathlib import Path

class TaskMetricsCollector:
    def __init__(self):
        self.metrics_file = Path(".taskmaster/metrics/execution.jsonl")
        self.metrics_file.parent.mkdir(exist_ok=True)
    
    def record_task_execution(self, task_id: str, success: bool, 
                            execution_time: float, resource_usage: dict):
        metric = {
            "timestamp": time.time(),
            "task_id": task_id,
            "success": success,
            "execution_time": execution_time,
            "resource_usage": resource_usage
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric) + '\n')
```

#### Health Monitoring
```bash
# System health check script
#!/bin/bash

check_system_health() {
    echo "🔍 Checking system health..."
    
    # Check Task-Master status
    task-master list > /dev/null || echo "❌ Task-Master not responding"
    
    # Check MCP connections
    if [ -f ".mcp.json" ]; then
        echo "✅ MCP configuration found"
    fi
    
    # Check API keys
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "⚠️ ANTHROPIC_API_KEY not set"
    fi
    
    # Check disk space
    DISK_USAGE=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ $DISK_USAGE -gt 90 ]; then
        echo "⚠️ Disk usage high: $DISK_USAGE%"
    fi
    
    echo "✅ System health check complete"
}
```

### 9. Autonomous Execution Best Practices

#### Autonomous Loop Implementation
```python
def autonomous_execution_loop():
    """Implement safe autonomous execution loop"""
    
    max_iterations = 100  # Safety limit
    iteration = 0
    
    while iteration < max_iterations:
        # Get next task
        next_task = get_next_task()
        if not next_task:
            break
        
        # Safety checks
        if not is_task_safe_for_autonomous_execution(next_task):
            print(f"⚠️ Task {next_task['id']} requires human oversight")
            break
        
        # Execute task
        try:
            result = execute_task_autonomous(next_task)
            if not result['success']:
                print(f"❌ Task {next_task['id']} failed")
                break
        except Exception as e:
            print(f"💥 Autonomous execution error: {e}")
            break
        
        iteration += 1
    
    print(f"🏁 Autonomous execution completed: {iteration} tasks")
```

#### Safety Mechanisms
```python
def is_task_safe_for_autonomous_execution(task: dict) -> bool:
    """Determine if task is safe for autonomous execution"""
    
    unsafe_keywords = [
        'delete',
        'drop',
        'remove production',
        'modify database',
        'security critical',
        'payment',
        'user data'
    ]
    
    description = task.get('description', '').lower()
    details = task.get('details', '').lower()
    
    for keyword in unsafe_keywords:
        if keyword in description or keyword in details:
            return False
    
    return True
```

### 10. Troubleshooting Best Practices

#### Common Issues and Solutions

```bash
# Issue: Task-Master commands failing
# Solution: Check API keys and model configuration
task-master models --list
cat .taskmaster/config.json

# Issue: MCP connection problems
# Solution: Restart MCP servers and check configuration
cat .mcp.json
npm list -g task-master-ai

# Issue: High memory usage
# Solution: Enable catalytic workspace
python3 .taskmaster/scripts/catalytic-workspace-system.py

# Issue: Claude Code tool restrictions
# Solution: Update allowlist
cat .claude/settings.json
```

#### Debug Mode
```bash
# Enable debug logging
export TASKMASTER_LOG_LEVEL=debug
export DEBUG=true

# Run with verbose output
task-master --debug list
task-master --verbose next
```

### 11. Security Best Practices

#### API Key Management
- Use environment variables, never hard-code keys
- Rotate keys regularly
- Use least-privilege access
- Monitor API usage for anomalies

#### Code Security
```bash
# Security scanning
npm audit
poetry run bandit -r .
pip-audit

# Dependency checking
npm outdated
poetry show --outdated
```

### 12. Documentation Best Practices

#### Task Documentation
```markdown
# Task Documentation Template

## Task ID: [ID]
## Title: [Clear, descriptive title]

### Description
Brief description of what needs to be done.

### Implementation Details
Step-by-step implementation plan.

### Test Strategy
How to validate the implementation.

### Dependencies
List of prerequisite tasks.

### Success Criteria
Measurable outcomes for completion.
```

#### Progress Tracking
```bash
# Regular status updates
task-master update-subtask --id=<id> --prompt="Progress update: 50% complete, implementing core logic"

# Final completion notes
task-master update-subtask --id=<id> --prompt="Implementation complete: all tests passing, documentation updated"
```

## Summary

Following these best practices ensures:
- **Reliability**: Robust error handling and recovery
- **Performance**: Optimized execution and resource usage
- **Security**: Safe handling of sensitive data and operations
- **Maintainability**: Clear documentation and organized structure
- **Scalability**: Patterns that work for projects of any size

Regular review and updates of these practices help maintain optimal performance and adapt to evolving project needs.