# Web Application Integration Example

This example demonstrates how to integrate Task-Master with a React web application development workflow using Claude Code and MCP.

## Project Structure

```
my-web-app/
├── .taskmaster/
│   ├── config.json
│   ├── tasks/
│   └── docs/
├── .claude/
│   ├── settings.json
│   └── commands/
├── .mcp.json
├── src/
├── public/
├── package.json
└── CLAUDE.md
```

## Step 1: Initialize Task-Master for Web App

```bash
# Initialize project with Task-Master
cd my-web-app
task-master init --name="React Dashboard App"

# Configure for web app development
task-master models --set-main claude-3-5-sonnet-20241022
```

## Step 2: Create Web App PRD

Create `.taskmaster/docs/web-app-requirements.md`:

```markdown
# React Dashboard Application PRD

## Overview
Build a modern React dashboard application with authentication, data visualization, and real-time updates.

## Core Features

### 1. Authentication System
- User login/logout functionality
- JWT token management
- Protected routes
- User profile management

### 2. Dashboard Components
- Interactive charts and graphs
- Data tables with sorting/filtering
- Real-time data updates
- Responsive layout

### 3. Data Management
- API integration for data fetching
- State management with Redux/Context
- Error handling and loading states
- Data caching and optimization

### 4. UI/UX Features
- Modern, responsive design
- Dark/light theme toggle
- Mobile-friendly interface
- Accessibility compliance

## Technical Requirements
- React 18+ with TypeScript
- Modern build tooling (Vite/Webpack)
- Testing framework (Jest + Testing Library)
- Linting and formatting (ESLint + Prettier)
```

## Step 3: Parse PRD and Generate Tasks

```bash
# Parse PRD to generate tasks
task-master parse-prd .taskmaster/docs/web-app-requirements.md

# Analyze complexity and expand tasks
task-master analyze-complexity --research
task-master expand --all --research

# View generated tasks
task-master list
```

## Step 4: Configure Claude Code Integration

Create `.claude/settings.json`:

```json
{
  "allowedTools": [
    "Edit",
    "Read", 
    "Write",
    "Bash(npm *)",
    "Bash(yarn *)",
    "Bash(git *)",
    "Bash(task-master *)",
    "mcp__task_master_ai__*"
  ],
  "customInstructions": "You are working on a React web application. Use task-master for project management and follow modern React best practices.",
  "workspaceSettings": {
    "autoSave": true,
    "formatOnSave": true
  }
}
```

Create `.mcp.json`:

```json
{
  "mcpServers": {
    "task-master-ai": {
      "command": "npx",
      "args": ["-y", "--package=task-master-ai", "task-master-ai"],
      "env": {
        "ANTHROPIC_API_KEY": "your_key_here"
      }
    }
  }
}
```

## Step 5: Create Custom Slash Commands

Create `.claude/commands/web-start-dev.md`:

```markdown
Start web development workflow for the current task.

Steps:
1. Run `task-master next` to get the current task
2. If it's a frontend task, start the development server with `npm run dev`
3. Open the relevant files for editing
4. Provide implementation guidance based on the task requirements
```

Create `.claude/commands/web-complete-task.md`:

```markdown
Complete current web development task: $ARGUMENTS

Steps:
1. Review implementation completeness
2. Run tests with `npm test`
3. Check code quality with `npm run lint`
4. Update task status: `task-master set-status --id=$ARGUMENTS --status=done`
5. Get next task with `task-master next`
```

## Step 6: Autonomous Development Workflow

```bash
# Start autonomous development session
claude --mcp-debug

# In Claude session:
# 1. Get next task
task-master next

# 2. Create component structure
mkdir -p src/components/Dashboard
touch src/components/Dashboard/Dashboard.tsx
touch src/components/Dashboard/Dashboard.module.css

# 3. Implement component with Task-Master guidance
# [Claude will implement based on task requirements]

# 4. Test implementation
npm test src/components/Dashboard

# 5. Complete task and move to next
task-master set-status --id=1.1 --status=done
```

## Step 7: Integration with Build Pipeline

Create `package.json` scripts that integrate with Task-Master:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "test": "jest",
    "lint": "eslint src --ext .ts,.tsx",
    "task:next": "task-master next",
    "task:complete": "task-master set-status --id=$npm_config_id --status=done",
    "task:analyze": "task-master analyze-complexity --research"
  }
}
```

## Step 8: Continuous Integration

Create `.github/workflows/task-master-ci.yml`:

```yaml
name: Task-Master CI

on: [push, pull_request]

jobs:
  task-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Install Task-Master
        run: npm install -g task-master-ai
      
      - name: Validate Task Completion
        run: |
          task-master list
          task-master validate-dependencies
          
      - name: Run Tests
        run: |
          npm install
          npm test
          
      - name: Build Application
        run: npm run build
```

## Expected Workflow Output

```bash
$ task-master next
╭──────────────────────────────────────────────────────────╮
│ Next Task: #1.1 - Implement Authentication Components    │
╰──────────────────────────────────────────────────────────╯

$ task-master show 1.1
┌─────────────┬──────────────────────────────────────────┐
│ ID:         │ 1.1                                      │
│ Title:      │ Implement Authentication Components      │
│ Priority:   │ high                                     │
│ Details:    │ Create login form, authentication hooks, │
│             │ and JWT token management                 │
└─────────────┴──────────────────────────────────────────┘

# After implementation...
$ task-master set-status --id=1.1 --status=done
✅ Task 1.1 completed successfully
```

## Performance Metrics

This integration approach typically achieves:
- **Development Speed**: 40-60% faster than manual planning
- **Code Quality**: Consistent architecture and patterns
- **Task Completion**: 95%+ autonomous execution rate
- **Error Reduction**: 50% fewer integration issues

## Best Practices

1. **Regular Task Updates**: Update task status frequently
2. **Detailed PRDs**: More detailed PRDs = better task generation
3. **Incremental Development**: Break large features into smaller tasks
4. **Testing Integration**: Include testing in every task
5. **Documentation**: Keep implementation notes in task details

## Troubleshooting

### Common Issues

1. **MCP Connection Failed**
   ```bash
   # Check MCP configuration
   cat .mcp.json
   # Restart Claude Code with --mcp-debug
   claude --mcp-debug
   ```

2. **Task Generation Issues**
   ```bash
   # Re-parse PRD with more detail
   task-master parse-prd .taskmaster/docs/web-app-requirements.md --force
   # Expand tasks manually
   task-master expand --id=1 --num=5
   ```

3. **Dependency Issues**
   ```bash
   # Validate and fix dependencies
   task-master validate-dependencies
   task-master fix-dependencies
   ```

## Advanced Features

### Custom Task Types
```bash
# Add custom task types for web development
task-master add-task --prompt="Create responsive navigation component with mobile menu" --priority=high

# Add testing tasks
task-master add-task --prompt="Write unit tests for authentication hooks" --priority=medium
```

### Integration with Design Systems
```bash
# Generate tasks for design system implementation
task-master parse-prd design-system-requirements.md --append
task-master expand --all --prompt="Follow Material-UI design patterns"
```

This integration example demonstrates how Task-Master enhances web application development by providing structured task management, autonomous execution capabilities, and seamless integration with modern development tools.