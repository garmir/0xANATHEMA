# Task-Master Project Templates and Configuration Examples

## Template Overview

This document provides comprehensive templates and configuration examples for integrating Task-Master with various project types and development workflows.

## Project Initialization Templates

### 1. Web Application Template

```bash
#!/bin/bash
# Web Application Project Template

PROJECT_NAME="$1"
FRAMEWORK="${2:-react}" # react, vue, angular, svelte

if [ -z "$PROJECT_NAME" ]; then
    echo "Usage: $0 <project_name> [framework]"
    exit 1
fi

echo "🚀 Initializing Web Application: $PROJECT_NAME ($FRAMEWORK)"

# Create project directory
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Initialize Task-Master
task-master init -y

# Configure for web development
cat > .taskmaster/config.json << EOF
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
    }
  },
  "global": {
    "logLevel": "info",
    "defaultSubtasks": 6,
    "defaultPriority": "medium",
    "projectName": "$PROJECT_NAME",
    "projectType": "web-application",
    "framework": "$FRAMEWORK"
  }
}
EOF

# Create web application PRD template
cat > .taskmaster/docs/prd.txt << EOF
# $PROJECT_NAME - Web Application Development

## Objective
Build a modern web application with responsive design, performance optimization, and comprehensive testing.

## Core Features
1. User interface with $FRAMEWORK framework
2. State management and data flow
3. API integration and data handling
4. Authentication and authorization
5. Responsive design for all devices
6. Performance optimization
7. Comprehensive testing suite
8. Deployment and CI/CD pipeline

## Technical Requirements
- Frontend: $FRAMEWORK with TypeScript
- Styling: CSS-in-JS or Tailwind CSS
- Testing: Jest + Testing Library
- Build: Vite or Webpack
- Deployment: Vercel, Netlify, or AWS

## Performance Requirements
- First Contentful Paint < 1.5s
- Largest Contentful Paint < 2.5s
- Cumulative Layout Shift < 0.1
- First Input Delay < 100ms
- 95+ Lighthouse score

## Success Criteria
- All components tested with 90%+ coverage
- Cross-browser compatibility
- Responsive design validated
- Performance benchmarks met
- Accessibility WCAG 2.1 AA compliant
EOF

# Create Claude Code configuration
mkdir -p .claude
cat > .claude/settings.json << EOF
{
  "allowedTools": [
    "Edit",
    "MultiEdit",
    "Read",
    "Write",
    "Bash(npm *)",
    "Bash(yarn *)",
    "Bash(git *)",
    "mcp__task_master_ai__*"
  ],
  "hooks": {
    "beforeCommit": "npm run lint && npm run test",
    "afterEdit": "npm run type-check"
  },
  "preferences": {
    "testFramework": "jest",
    "linting": "eslint",
    "formatting": "prettier"
  }
}
EOF

# Create custom slash commands
mkdir -p .claude/commands
cat > .claude/commands/web-dev-cycle.md << 'EOF'
Complete web development cycle: $ARGUMENTS

Steps:
1. Get next task: task-master next
2. Implement component or feature
3. Write comprehensive tests
4. Run linting and formatting: npm run lint:fix
5. Test functionality: npm run test
6. Update task: task-master update-subtask --id=$ARGUMENTS --prompt="Feature implemented and tested"
7. Mark complete: task-master set-status --id=$ARGUMENTS --status=done
EOF

echo "✅ Web Application template created successfully!"
echo "📋 Next steps:"
echo "   1. task-master parse-prd .taskmaster/docs/prd.txt"
echo "   2. task-master expand --all --research"
echo "   3. npm create $FRAMEWORK@latest ."
```

### 2. API Development Template

```bash
#!/bin/bash
# API Development Project Template

PROJECT_NAME="$1"
FRAMEWORK="${2:-fastapi}" # fastapi, express, django, flask

if [ -z "$PROJECT_NAME" ]; then
    echo "Usage: $0 <project_name> [framework]"
    exit 1
fi

echo "🔌 Initializing API Project: $PROJECT_NAME ($FRAMEWORK)"

mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Initialize Task-Master
task-master init -y

# Configure for API development
cat > .taskmaster/config.json << EOF
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
    }
  },
  "global": {
    "logLevel": "info",
    "defaultSubtasks": 8,
    "defaultPriority": "high",
    "projectName": "$PROJECT_NAME",
    "projectType": "api",
    "framework": "$FRAMEWORK"
  }
}
EOF

# Create API PRD template
cat > .taskmaster/docs/prd.txt << EOF
# $PROJECT_NAME - API Development

## Objective
Build a robust, scalable API with comprehensive documentation, testing, and monitoring.

## Core Features
1. RESTful API endpoints with proper HTTP methods
2. Authentication and authorization system
3. Data validation and serialization
4. Database integration and ORM
5. API documentation (OpenAPI/Swagger)
6. Rate limiting and security measures
7. Logging and monitoring
8. Comprehensive testing suite
9. Docker containerization
10. CI/CD pipeline

## Technical Requirements
- Framework: $FRAMEWORK
- Database: PostgreSQL with Redis caching
- Authentication: JWT tokens
- Documentation: OpenAPI 3.0
- Testing: pytest or jest
- Deployment: Docker + Kubernetes

## Performance Requirements
- Response time < 100ms for cached data
- Handle 1000+ concurrent requests
- 99.9% uptime SLA
- Sub-second database queries

## Security Requirements
- HTTPS only
- Input validation and sanitization
- SQL injection prevention
- Rate limiting per client
- Security headers implementation
EOF

# Create API-specific Claude Code configuration
mkdir -p .claude
cat > .claude/settings.json << EOF
{
  "allowedTools": [
    "Edit",
    "MultiEdit", 
    "Read",
    "Write",
    "Bash(poetry *)",
    "Bash(pytest *)",
    "Bash(docker *)",
    "Bash(kubectl *)",
    "mcp__task_master_ai__*"
  ],
  "hooks": {
    "beforeCommit": "poetry run pytest && poetry run black . && poetry run flake8",
    "afterEdit": "poetry run mypy ."
  },
  "preferences": {
    "testFramework": "pytest",
    "linting": "flake8,mypy",
    "formatting": "black",
    "apiDocs": "swagger"
  }
}
EOF

# Create API development commands
mkdir -p .claude/commands
cat > .claude/commands/api-endpoint-cycle.md << 'EOF'
Create API endpoint with full testing: $ARGUMENTS

Steps:
1. Get task details: task-master show $ARGUMENTS
2. Implement endpoint with proper validation
3. Add comprehensive tests (unit + integration)
4. Update OpenAPI documentation
5. Test security (authentication, validation)
6. Run performance test
7. Update task: task-master update-subtask --id=$ARGUMENTS --prompt="Endpoint implemented with tests"
8. Mark complete: task-master set-status --id=$ARGUMENTS --status=done
EOF

echo "✅ API template created successfully!"
```

### 3. Data Science/ML Template

```bash
#!/bin/bash
# Data Science/ML Project Template

PROJECT_NAME="$1"
ML_TYPE="${2:-classification}" # classification, regression, clustering, nlp

echo "📊 Initializing ML Project: $PROJECT_NAME ($ML_TYPE)"

mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Initialize Task-Master
task-master init -y

# Configure for ML development
cat > .taskmaster/config.json << EOF
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
    }
  },
  "global": {
    "logLevel": "info",
    "defaultSubtasks": 10,
    "defaultPriority": "medium",
    "projectName": "$PROJECT_NAME",
    "projectType": "machine-learning",
    "mlType": "$ML_TYPE"
  }
}
EOF

# Create ML PRD template
cat > .taskmaster/docs/prd.txt << EOF
# $PROJECT_NAME - Machine Learning Project

## Objective
Build an end-to-end machine learning pipeline for $ML_TYPE with automated training, evaluation, and deployment.

## Core Features
1. Data ingestion and preprocessing pipeline
2. Exploratory data analysis (EDA)
3. Feature engineering and selection
4. Model training with hyperparameter tuning
5. Model evaluation and validation
6. Model deployment and serving
7. Monitoring and performance tracking
8. Data drift detection
9. Automated retraining pipeline
10. Model explainability and interpretability

## Technical Requirements
- Data Processing: Pandas, NumPy, Spark
- ML Libraries: scikit-learn, XGBoost, TensorFlow/PyTorch
- Experiment Tracking: MLflow or Weights & Biases
- Deployment: MLflow Model Registry + FastAPI
- Monitoring: Prometheus + Grafana
- Data Validation: Great Expectations

## Performance Requirements
- Model accuracy > 85% (adjust based on problem)
- Training time < 4 hours
- Inference latency < 10ms
- Data processing: 1M+ records/hour
- Model deployment: < 5 minutes

## Success Criteria
- Model passes validation tests
- A/B testing shows improvement
- Production monitoring in place
- Documentation complete
- Reproducible experiments
EOF

# Create ML-specific configuration
mkdir -p .claude
cat > .claude/settings.json << EOF
{
  "allowedTools": [
    "Edit",
    "MultiEdit",
    "Read", 
    "Write",
    "Bash(poetry *)",
    "Bash(jupyter *)",
    "Bash(mlflow *)",
    "Bash(python *.py)",
    "mcp__task_master_ai__*"
  ],
  "hooks": {
    "beforeCommit": "poetry run pytest tests/ && poetry run black .",
    "afterEdit": "poetry run mypy . --ignore-missing-imports"
  },
  "preferences": {
    "notebook": "jupyter",
    "tracking": "mlflow",
    "testing": "pytest"
  }
}
EOF

# Create ML development commands
mkdir -p .claude/commands
cat > .claude/commands/ml-experiment-cycle.md << 'EOF'
Run ML experiment cycle: $ARGUMENTS

Steps:
1. Get experiment task: task-master show $ARGUMENTS
2. Load and validate data
3. Perform EDA and feature engineering
4. Train model with hyperparameter tuning
5. Evaluate model performance
6. Log experiment to MLflow
7. Update task: task-master update-subtask --id=$ARGUMENTS --prompt="Experiment completed with results"
8. Mark complete: task-master set-status --id=$ARGUMENTS --status=done
EOF

echo "✅ ML template created successfully!"
```

## Configuration Templates

### 1. MCP Configuration Template

```json
{
  "mcpServers": {
    "task-master-ai": {
      "command": "npx",
      "args": ["-y", "--package=task-master-ai", "task-master-ai"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "GOOGLE_API_KEY": "${GOOGLE_API_KEY}",
        "TASKMASTER_HOME": "${PWD}/.taskmaster",
        "TASKMASTER_LOG_LEVEL": "info",
        "TASKMASTER_RETRY_COUNT": "3",
        "TASKMASTER_TIMEOUT": "300"
      },
      "settings": {
        "retryOnFailure": true,
        "maxRetries": 3,
        "timeoutMs": 300000,
        "healthCheckInterval": 30000
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"],
      "env": {
        "ALLOWED_PATHS": "${PWD}"
      }
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git"],
      "env": {
        "GIT_REPOSITORY_PATH": "${PWD}"
      }
    }
  },
  "global": {
    "debug": false,
    "logLevel": "info",
    "connectionTimeout": 10000,
    "requestTimeout": 30000
  }
}
```

### 2. Claude Code Settings Template

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
    "Bash(poetry run *)",
    "Bash(python3 .taskmaster/scripts/*)",
    "mcp__task_master_ai__*",
    "mcp__filesystem__*",
    "mcp__git__*"
  ],
  "hooks": {
    "beforeEdit": "task-master update-subtask --id=${CURRENT_TASK_ID} --prompt='Starting edit: ${FILE_PATH}'",
    "afterEdit": "echo 'File edited: ${FILE_PATH}'",
    "beforeCommit": "npm run lint && npm run test",
    "afterCommit": "task-master update-subtask --id=${CURRENT_TASK_ID} --prompt='Committed changes: ${COMMIT_MESSAGE}'"
  },
  "preferences": {
    "autoSave": true,
    "verboseLogging": false,
    "parallelToolCalls": true,
    "maxConcurrentTasks": 3
  },
  "security": {
    "allowFileSystem": true,
    "allowNetwork": true,
    "allowShell": true,
    "restrictedPaths": [
      "/etc",
      "/usr",
      "/System"
    ]
  }
}
```

### 3. Environment Configuration Template

```bash
#!/bin/bash
# Environment Configuration Template

# API Keys (set these in your .env file)
export ANTHROPIC_API_KEY="your_anthropic_key_here"
export PERPLEXITY_API_KEY="your_perplexity_key_here"
export OPENAI_API_KEY="your_openai_key_here"
export GOOGLE_API_KEY="your_google_key_here"

# Task-Master Configuration
export TASKMASTER_HOME="$(pwd)/.taskmaster"
export TASKMASTER_DOCS="$TASKMASTER_HOME/docs"
export TASKMASTER_LOGS="$TASKMASTER_HOME/logs"
export TASKMASTER_CATALYTIC="$TASKMASTER_HOME/catalytic-workspace"

# Project-specific settings
export PROJECT_NAME="$(basename $(pwd))"
export PROJECT_TYPE="web-application" # web-application, api, machine-learning
export ENVIRONMENT="development" # development, staging, production

# Development tools
export NODE_ENV="development"
export PYTHON_ENV="development"
export DEBUG="true"

# Claude Code settings
export CLAUDE_CODE_HOME="$(pwd)/.claude"
export CLAUDE_MCP_CONFIG="$(pwd)/.mcp.json"

# Logging
export LOG_LEVEL="info"
export LOG_FORMAT="json"

# Performance settings
export MAX_CONCURRENT_TASKS="3"
export TASK_TIMEOUT="300"
export MEMORY_LIMIT="4GB"

# Create necessary directories
mkdir -p "$TASKMASTER_HOME"/{docs,logs,scripts,artifacts,catalytic-workspace}
mkdir -p "$CLAUDE_CODE_HOME"/{commands,settings}

echo "✅ Environment configured for $PROJECT_NAME ($PROJECT_TYPE)"
```

## Best Practices Configuration

### 1. Git Integration Template

```bash
#!/bin/bash
# Git Integration Configuration

# Configure git hooks for Task-Master integration
mkdir -p .git/hooks

# Pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Task-Master Pre-commit Hook

echo "🔍 Running pre-commit checks..."

# Run linting
if command -v npm &> /dev/null; then
    npm run lint || exit 1
fi

if command -v poetry &> /dev/null; then
    poetry run black . || exit 1
    poetry run flake8 . || exit 1
fi

# Run tests
if command -v npm &> /dev/null && [ -f "package.json" ]; then
    npm run test || exit 1
fi

if command -v poetry &> /dev/null && [ -f "pyproject.toml" ]; then
    poetry run pytest || exit 1
fi

# Update Task-Master with commit info
CURRENT_TASK=$(task-master next --json 2>/dev/null | jq -r '.id // empty')
if [ -n "$CURRENT_TASK" ]; then
    task-master update-subtask --id="$CURRENT_TASK" --prompt="Pre-commit checks passed, ready to commit"
fi

echo "✅ Pre-commit checks passed"
EOF

# Post-commit hook
cat > .git/hooks/post-commit << 'EOF'
#!/bin/bash
# Task-Master Post-commit Hook

COMMIT_MSG=$(git log -1 --pretty=format:'%s')
COMMIT_HASH=$(git log -1 --pretty=format:'%h')

echo "📝 Commit completed: $COMMIT_HASH"

# Update Task-Master with commit info
CURRENT_TASK=$(task-master next --json 2>/dev/null | jq -r '.id // empty')
if [ -n "$CURRENT_TASK" ]; then
    task-master update-subtask --id="$CURRENT_TASK" --prompt="Committed: $COMMIT_MSG ($COMMIT_HASH)"
fi
EOF

chmod +x .git/hooks/pre-commit .git/hooks/post-commit

echo "✅ Git hooks configured for Task-Master integration"
```

### 2. CI/CD Configuration Template

```yaml
# .github/workflows/taskmaster-integration.yml
name: Task-Master CI/CD Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  PERPLEXITY_API_KEY: ${{ secrets.PERPLEXITY_API_KEY }}

jobs:
  taskmaster-validation:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install Task-Master
      run: |
        npm install -g task-master-ai
        task-master --version
        
    - name: Validate Task-Master Configuration
      run: |
        task-master validate-dependencies
        task-master complexity-report
        
    - name: Run Task-Master Tests
      run: |
        if [ -f ".taskmaster/scripts/end-to-end-testing-framework.py" ]; then
          python3 .taskmaster/scripts/end-to-end-testing-framework.py
        fi
        
    - name: Generate Task Report
      run: |
        task-master list --json > task-report.json
        task-master complexity-report --json > complexity-report.json
        
    - name: Upload Reports
      uses: actions/upload-artifact@v3
      with:
        name: taskmaster-reports
        path: |
          task-report.json
          complexity-report.json
          .taskmaster/logs/
```

### 3. Docker Configuration Template

```dockerfile
# Dockerfile.taskmaster
FROM python:3.9-slim

# Install Node.js for Task-Master
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Task-Master
RUN npm install -g task-master-ai

# Set working directory
WORKDIR /app

# Copy Task-Master configuration
COPY .taskmaster/ .taskmaster/
COPY .claude/ .claude/
COPY .mcp.json .mcp.json

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create entrypoint script
RUN cat > entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Initialize Task-Master if needed
if [ ! -f ".taskmaster/state.json" ]; then
    task-master init -y
fi

# Run autonomous execution if requested
if [ "$1" = "autonomous" ]; then
    echo "🚀 Starting autonomous execution..."
    python3 .taskmaster/scripts/claude-flow-integration.py
else
    exec "$@"
fi
EOF

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["autonomous"]
```

## Usage Examples

### Quick Start Script

```bash
#!/bin/bash
# Quick start script for any project type

PROJECT_TYPE="$1"
PROJECT_NAME="$2"

if [ -z "$PROJECT_TYPE" ] || [ -z "$PROJECT_NAME" ]; then
    echo "Usage: $0 <project_type> <project_name>"
    echo "Project types: web, api, ml, data"
    exit 1
fi

case "$PROJECT_TYPE" in
    "web")
        bash web-app-template.sh "$PROJECT_NAME"
        ;;
    "api")
        bash api-template.sh "$PROJECT_NAME"
        ;;
    "ml")
        bash ml-template.sh "$PROJECT_NAME"
        ;;
    "data")
        bash data-processing-template.sh "$PROJECT_NAME"
        ;;
    *)
        echo "❌ Unknown project type: $PROJECT_TYPE"
        exit 1
        ;;
esac

cd "$PROJECT_NAME"

echo "🔧 Setting up development environment..."
bash setup-environment.sh

echo "📋 Parsing initial PRD..."
task-master parse-prd .taskmaster/docs/prd.txt --research

echo "🧠 Analyzing complexity..."
task-master analyze-complexity --research

echo "📈 Expanding tasks..."
task-master expand --all --research

echo "🎉 Project $PROJECT_NAME ($PROJECT_TYPE) is ready!"
echo "📋 Next steps:"
echo "   1. Review tasks: task-master list"
echo "   2. Start development: task-master next"
echo "   3. Use Claude Code with: claude"
```

This comprehensive template system provides everything needed to quickly bootstrap Task-Master integration with any project type while following best practices for development workflows.