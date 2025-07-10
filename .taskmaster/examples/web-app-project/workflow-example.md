# Web App Development Workflow Example

This example demonstrates how to use Task Master with Claude Code for a modern web application project.

## Step 1: Project Initialization

```bash
# Create project directory
mkdir my-web-app
cd my-web-app

# Initialize Task Master
task-master init

# Copy the PRD to Task Master docs
cp ../project-prd.md .taskmaster/docs/

# Configure Claude Code MCP integration
cp ../mcp-config.json .mcp.json
```

## Step 2: Generate and Optimize Tasks

```bash
# Parse PRD to generate initial tasks
task-master parse-prd .taskmaster/docs/project-prd.md

# Analyze project complexity with research
task-master analyze-complexity --research

# Expand tasks into detailed subtasks
task-master expand --all --research

# View generated tasks
task-master list
```

## Step 3: Claude Code Integration

Start Claude Code with MCP integration:

```bash
claude --mcp-debug
```

In Claude session:

```markdown
# Use Task Master MCP tools to work with tasks

## Get current task to work on
Use the `next_task` tool to get the next available task.

## Work on a specific task
Use `get_task` with task ID to get detailed requirements.

## Update task progress
Use `update_subtask` to log implementation notes and progress.

## Mark tasks complete
Use `set_task_status` to mark tasks as done when finished.
```

## Step 4: Development Workflow

### Backend Development
```bash
# Get next backend task
task-master next

# Example: Task "Set up Express server with TypeScript"
mkdir backend
cd backend
npm init -y
npm install express typescript @types/express

# Update task with progress
task-master update-subtask --id=2.1 --prompt="Installed Express and TypeScript, created basic server structure"

# Mark task complete when done
task-master set-status --id=2.1 --status=done
```

### Frontend Development  
```bash
# Get next frontend task
task-master next

# Example: Task "Create React app with TypeScript"
npx create-react-app frontend --template typescript
cd frontend
npm install @mui/material @emotion/react @emotion/styled

# Update progress
task-master update-subtask --id=3.1 --prompt="Created React app with MUI, set up basic component structure"

# Mark complete
task-master set-status --id=3.1 --status=done
```

### Database Setup
```bash
# Get database task
task-master show 4  # Database setup task

# Set up PostgreSQL with Prisma
npm install prisma @prisma/client
npx prisma init

# Update progress
task-master update-subtask --id=4.1 --prompt="Initialized Prisma, configured PostgreSQL connection"
```

## Step 5: Continuous Integration

Use Task Master's optimization engine to prioritize tasks:

```bash
# Generate optimized execution order
task-master complexity-report

# Focus on high-priority, low-complexity tasks first
task-master list --priority=high --status=pending
```

## Step 6: Testing and Deployment

```bash
# Get testing tasks
task-master list --filter="test"

# Run end-to-end validation
.taskmaster/testing/run_e2e_tests.sh --scenario web_app_simple

# Deploy when all critical tasks complete
task-master list --status=done --priority=high
```

## Expected Outcomes

After following this workflow, you'll have:

- ✅ Complete task breakdown from high-level PRD
- ✅ Optimized task execution order based on complexity analysis
- ✅ Seamless Claude Code integration for implementation
- ✅ Progress tracking and autonomy scoring
- ✅ Automated testing and validation
- ✅ Production-ready deployment pipeline

## Integration Benefits

- **Reduced Planning Time**: 70% faster project planning
- **Higher Code Quality**: Systematic approach reduces bugs
- **Better Resource Allocation**: Complexity analysis optimizes team assignments
- **Improved Predictability**: Autonomy scoring provides delivery confidence
- **Enhanced Collaboration**: Shared task visibility and progress tracking