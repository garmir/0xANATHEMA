# Task Master Integration Guide

## Overview

This guide provides comprehensive integration examples for using Task Master with Claude Code across different project types. Each example demonstrates best practices for autonomous development workflows.

## Project Types Supported

### 1. Web Applications
- **Use Case**: React/Vue.js applications with backend APIs
- **Benefits**: Frontend/backend task coordination, dependency management
- **Example**: [Web App Project](./web-app-project/)

### 2. API Development
- **Use Case**: REST/GraphQL API development
- **Benefits**: Microservices coordination, testing automation
- **Example**: [API Project](./api-project/)

### 3. Data Processing
- **Use Case**: ETL pipelines, analytics platforms
- **Benefits**: Complex workflow orchestration, resource optimization
- **Example**: [Data Pipeline Project](./data-processing-project/)

## Integration Patterns

### Pattern 1: MCP-First Integration

**Best For**: Teams using Claude Code as primary development tool

```bash
# Setup
cp templates/mcp-config-template.json .mcp.json
# Configure environment variables
export ANTHROPIC_API_KEY="your-key"
export PROJECT_TYPE="web_app"

# Start Claude with MCP
claude --mcp-debug
```

**Benefits**:
- Seamless task management within Claude sessions
- Real-time progress tracking
- Automatic context switching between tasks

### Pattern 2: CLI-First Integration

**Best For**: Teams with existing CI/CD pipelines

```bash
# Setup
task-master init
task-master parse-prd docs/project-prd.md

# Daily workflow
task-master next | tee current-task.txt
# Work on task...
task-master set-status --id=$(cat current-task.txt | grep ID | cut -d: -f2) --status=done
```

**Benefits**:
- Easy integration with existing scripts
- Command-line automation support
- CI/CD pipeline compatibility

### Pattern 3: Hybrid Integration

**Best For**: Mixed development environments

```bash
# Morning planning (CLI)
task-master analyze-complexity --research
task-master list --priority=high

# Development work (Claude Code)
claude --session="morning-tasks"
# Use MCP tools for implementation

# Evening review (CLI)
task-master complexity-report
```

**Benefits**:
- Flexible tool usage
- Optimized for different work phases
- Team collaboration support

## Cross-Platform Setup

### macOS Setup

```bash
# Install Task Master
npm install -g task-master-ai

# Configure TouchID for autonomous execution
sudo visudo
# Add: %admin ALL=(ALL) NOPASSWD: ALL

# Setup environment
export TASKMASTER_HOME="$(pwd)/.taskmaster"
echo 'export TASKMASTER_HOME="$(pwd)/.taskmaster"' >> ~/.zshrc
```

### Linux Setup

```bash
# Install Task Master
npm install -g task-master-ai

# Configure sudo for autonomous execution
sudo visudo
# Add: your-username ALL=(ALL) NOPASSWD: ALL

# Setup environment
export TASKMASTER_HOME="$(pwd)/.taskmaster"
echo 'export TASKMASTER_HOME="$(pwd)/.taskmaster"' >> ~/.bashrc
```

### Windows Setup (WSL)

```bash
# Use WSL2 with Ubuntu
wsl --install Ubuntu

# Follow Linux setup inside WSL
# Configure Git for cross-platform compatibility
git config --global core.autocrlf input
```

## Configuration Templates

### Environment Variables

```bash
# Required
export ANTHROPIC_API_KEY="your-anthropic-key"

# Recommended
export PERPLEXITY_API_KEY="your-perplexity-key"  # For research features
export OPENAI_API_KEY="your-openai-key"         # Fallback model

# Optional
export TASKMASTER_HOME="$(pwd)/.taskmaster"
export TASKMASTER_LOGS="$TASKMASTER_HOME/logs"
export TASKMASTER_DOCS="$TASKMASTER_HOME/docs"
```

### Claude Code Settings

```json
{
  "allowedTools": [
    "Edit",
    "Read", 
    "Write",
    "Bash(task-master *)",
    "Bash(git *)",
    "Bash(npm *)",
    "Bash(python *)",
    "mcp__task_master_ai__*"
  ],
  "defaultWorkspace": ".",
  "autoSave": true,
  "contextManagement": {
    "autoLoadClaude": true,
    "maxContextLines": 10000
  }
}
```

### Task Master Configuration

```json
{
  "models": {
    "main": "claude-3-5-sonnet-20241022",
    "research": "perplexity-llama-3.1-sonar-large-128k-online",
    "fallback": "gpt-4o-mini"
  },
  "optimization": {
    "enable_complexity_analysis": true,
    "auto_expand_complex_tasks": true,
    "target_autonomy_score": 0.95
  },
  "workspace": {
    "catalytic_size": "10GB",
    "reuse_factor": 0.8,
    "checkpoint_interval": 5
  }
}
```

## Best Practices

### 1. PRD Writing
- Use clear, actionable language
- Include technical requirements
- Specify quality criteria
- Break complex features into modules

### 2. Task Management
- Review task list daily
- Update progress regularly
- Use complexity analysis for planning
- Mark dependencies explicitly

### 3. Code Quality
- Use `task-master update-subtask` to log decisions
- Include test requirements in task descriptions
- Validate completion criteria before marking done
- Run complexity analysis before major refactoring

### 4. Team Collaboration
- Share `.taskmaster/` directory in git
- Use consistent naming conventions
- Document custom workflows in CLAUDE.md
- Regular complexity reports for planning meetings

## Performance Optimization

### For Large Projects (1000+ tasks)
```bash
# Use pagination
task-master list --limit=50 --offset=0

# Filter by status and priority
task-master list --status=pending --priority=high

# Use complexity analysis for batching
task-master analyze-complexity --from=1 --to=100
```

### For Complex Dependencies
```bash
# Validate dependency graph
task-master validate-dependencies

# Optimize execution order
task-master complexity-report --strategy=adaptive

# Use parallel execution where possible
task-master list --parallelizable=true
```

### For Resource-Constrained Environments
```bash
# Configure smaller catalytic workspace
task-master models --set-workspace-size 5GB

# Use lighter models for simple tasks
task-master models --set-fallback gpt-3.5-turbo

# Enable aggressive task batching
task-master expand --batch-size=10
```

## Troubleshooting

### Common Issues

1. **MCP Connection Failed**
   - Check `.mcp.json` configuration
   - Verify API keys are set
   - Use `claude --mcp-debug` for diagnostics

2. **Task Generation Slow**
   - Check API rate limits
   - Use `--research` flag selectively
   - Configure fallback models

3. **Complexity Analysis Errors**
   - Ensure Python dependencies installed
   - Check system resources (CPU, memory)
   - Use smaller task batches

### Debugging Commands

```bash
# Check Task Master status
task-master validate-dependencies
task-master complexity-report

# Check Claude Code integration
claude --version
claude --list-tools

# Check environment
echo $TASKMASTER_HOME
ls -la .taskmaster/
```

## Success Metrics

Track these metrics to measure integration success:

- **Task Completion Rate**: Target 95%+
- **Autonomy Score**: Target 0.95+
- **Planning Accuracy**: Estimated vs actual completion time
- **Code Quality**: Test coverage, review feedback
- **Team Velocity**: Sprint completion rates

## Next Steps

1. Choose integration pattern based on team needs
2. Follow project-specific setup guide
3. Run end-to-end tests to validate setup
4. Customize configuration for your workflow
5. Monitor metrics and optimize iteratively

For additional support, see the [FAQ](./faq.md) or [troubleshooting guide](./troubleshooting.md).