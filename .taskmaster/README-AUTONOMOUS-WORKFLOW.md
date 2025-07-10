# Autonomous Workflow Loop Documentation

## Overview

This directory contains a hard-coded autonomous workflow loop that implements the pattern:

**Get Stuck → Research with task-master + perplexity → Parse solutions to Claude → Execute until success**

## Files

### 1. `autonomous-workflow-loop.sh`
Main bash script that implements the autonomous workflow loop.

**Usage:**
```bash
# Make executable
chmod +x .taskmaster/autonomous-workflow-loop.sh

# Run with defaults (50 iterations, 0.95 success threshold)
./.taskmaster/autonomous-workflow-loop.sh

# Run with custom parameters
./.taskmaster/autonomous-workflow-loop.sh --max-iterations 100 --threshold 0.90

# Show help
./.taskmaster/autonomous-workflow-loop.sh --help
```

**Features:**
- Automatically executes `task-master next` to get available tasks
- When stuck, uses `task-master research` with perplexity to find solutions
- Parses research output into actionable steps
- Simulates Claude execution for task completion
- Validates autonomy score after each iteration
- Continues until success threshold reached or max iterations

### 2. `claude-integration-wrapper.py`
Python wrapper that provides programmatic interface for Claude integration.

**Usage:**
```bash
# Run autonomous workflow
python3 .taskmaster/claude-integration-wrapper.py

# Custom parameters
python3 .taskmaster/claude-integration-wrapper.py --max-iterations 25 --threshold 0.90

# Show help
python3 .taskmaster/claude-integration-wrapper.py --help
```

**Features:**
- Object-oriented interface for task-master operations
- Automated research solution generation
- Claude prompt creation with research insights
- Comprehensive logging and error handling
- Validation and success measurement
- Structured output and reporting

## Workflow Pattern

### 1. **Task Execution Phase**
```bash
task-master next  # Get next available task
```

### 2. **Research Phase (When Stuck)**
```bash
# Automatically triggered when execution fails
task-master research "Task X failed with error Y. Research comprehensive solution..."
```

### 3. **Solution Parsing Phase**
- Research output is parsed into actionable steps
- Claude prompts are generated with specific implementation guidance
- Solution steps are saved for execution tracking

### 4. **Execution Phase**
- Solutions are executed systematically
- Task status is updated (in-progress → done)
- Validation checks are performed

### 5. **Success Validation**
```bash
python3 .taskmaster/optimization/comprehensive-validator.py
```
- Autonomy score is measured
- Continue loop if below threshold
- Exit successfully when threshold reached

## Configuration

### Environment Variables
The workflow expects these environment variables (automatically set by setup):
```bash
export TASKMASTER_HOME="/Users/anam/archive/.taskmaster"
export TASKMASTER_DOCS="/Users/anam/archive/.taskmaster/docs"
export TASKMASTER_LOGS="/Users/anam/archive/.taskmaster/logs"
```

### Default Parameters
- **Max Iterations**: 50
- **Success Threshold**: 0.95 (95% autonomy score)
- **Retry Attempts**: 3 per task
- **Research Timeout**: 120 seconds

## Logging

### Log Locations
- **Workflow Logs**: `.taskmaster/logs/autonomous-workflow-YYYYMMDD-HHMMSS.log`
- **Claude Integration**: `.taskmaster/logs/claude-integration.log`
- **Research Solutions**: `/tmp/research-solution-{task_id}.txt`
- **Claude Prompts**: `.taskmaster/logs/claude-prompt-{task_id}.md`

### Log Format
```
🚀 Starting Autonomous Workflow Loop
🔄 ITERATION 1/50
📋 Current Task ID: 28
🔍 STUCK! Researching solution for task 28...
🧠 Researching with task-master + perplexity...
💡 Solution research saved to /tmp/research-solution-28.txt
⚡ Executing solution steps for task 28...
✅ Task 28 completed successfully!
🎯 Current autonomy score: 0.67
```

## Integration Points

### Task-Master CLI Integration
- `task-master next` - Get next available task
- `task-master show {id}` - Get task details
- `task-master research "{prompt}"` - Research solutions
- `task-master set-status --id={id} --status={status}` - Update task status

### Claude Code Integration
- Automatic prompt generation with research context
- Structured implementation guidance
- Error handling and retry logic
- Success validation and reporting

### Perplexity Research Integration
- Automatic research query generation
- Context-aware solution discovery
- Implementation-focused guidance
- Common pattern recognition

## Usage Examples

### Basic Autonomous Execution
```bash
# Start the autonomous workflow
./.taskmaster/autonomous-workflow-loop.sh

# Monitor progress
tail -f .taskmaster/logs/autonomous-workflow-*.log
```

### Advanced Usage with Python Interface
```python
from claude_integration_wrapper import ClaudeIntegrationWrapper

wrapper = ClaudeIntegrationWrapper()

# Execute single task with research
task = wrapper.get_next_task()
if task:
    research = wrapper.research_solution(task["id"], "Initial analysis needed")
    prompt = wrapper.create_claude_prompt(task["id"], research)
    print(prompt)

# Run full autonomous workflow
result = wrapper.execute_autonomous_workflow(max_iterations=30, success_threshold=0.92)
print(f"Completed {len(result['completed_tasks'])} tasks")
```

## Success Criteria

The workflow considers itself successful when:

1. **Autonomy Score ≥ Threshold** (default 0.95)
2. **All Critical Checks Pass**
3. **System Integration Validated**
4. **Tasks Completed Successfully**

## Troubleshooting

### Common Issues

1. **task-master commands fail**
   - Check environment variables are set
   - Verify .taskmaster directory structure
   - Ensure proper permissions

2. **Research timeouts**
   - Check perplexity API key configuration
   - Reduce research prompt complexity
   - Increase timeout in wrapper

3. **Validation failures**
   - Check comprehensive-validator.py exists
   - Verify optimization files in artifacts directory
   - Review validation logs for specific errors

### Debug Mode
```bash
# Enable debug logging
export TASKMASTER_DEBUG=true

# Run with verbose output
./.taskmaster/autonomous-workflow-loop.sh 2>&1 | tee debug.log
```

## Architecture

```
┌─────────────────────────────────────────┐
│        Autonomous Workflow Loop         │
├─────────────────────────────────────────┤
│ 1. task-master next                     │
│ 2. Research (when stuck)                │
│ 3. Parse to Claude                      │
│ 4. Execute solution                     │
│ 5. Validate success                     │
│ 6. Repeat until threshold               │
└─────────────────────────────────────────┘
           │
           ├─ Task-Master CLI
           ├─ Perplexity Research
           ├─ Claude Code Integration
           └─ Validation Framework
```

This autonomous workflow loop provides a robust, self-improving system that can handle complex task execution with intelligent problem-solving capabilities.