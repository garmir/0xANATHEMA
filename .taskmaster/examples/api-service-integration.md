# API Service Integration Example

This example demonstrates Task-Master integration with FastAPI/REST API development workflows.

## Project Setup

```bash
# Initialize API project
mkdir my-api-service && cd my-api-service
task-master init --name="User Management API"

# Create virtual environment
python -m venv venv
source venv/bin/activate
```

## PRD for API Service

Create `.taskmaster/docs/api-requirements.md`:

```markdown
# User Management API - PRD

## Overview
Build a RESTful API service for user management with authentication, CRUD operations, and data validation.

## API Endpoints

### Authentication
- POST /auth/login - User authentication
- POST /auth/register - User registration  
- POST /auth/refresh - Token refresh
- DELETE /auth/logout - User logout

### User Management
- GET /users - List users (paginated)
- GET /users/{id} - Get user details
- PUT /users/{id} - Update user
- DELETE /users/{id} - Delete user

### Profile Management
- GET /profile - Get current user profile
- PUT /profile - Update profile
- POST /profile/avatar - Upload avatar

## Technical Requirements
- FastAPI framework with Pydantic models
- PostgreSQL database with SQLAlchemy ORM
- JWT authentication with refresh tokens
- Input validation and error handling
- API documentation with OpenAPI
- Rate limiting and security headers
- Comprehensive testing suite
```

## Configuration Templates

Create `.taskmaster/templates/api-config.json`:

```json
{
  "project_type": "api_service",
  "framework": "fastapi",
  "database": "postgresql",
  "authentication": "jwt",
  "testing_framework": "pytest",
  "deployment": "docker",
  "monitoring": "prometheus"
}
```

## Autonomous API Development

```bash
# Parse PRD and generate tasks
task-master parse-prd .taskmaster/docs/api-requirements.md

# Analyze and expand with API-specific context
task-master expand --all --prompt="Focus on FastAPI best practices and async patterns"

# Start development workflow
task-master next
```

## Integration with Development Tools

Create `pyproject.toml`:

```toml
[tool.taskmaster]
project_type = "api_service"
auto_test = true
auto_format = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src --cov-report=term-missing"

[tool.black]
line-length = 100
target-version = ['py39']
```

## Expected Output

```bash
$ task-master next
╭──────────────────────────────────────────────────────────╮
│ Next Task: #2.1 - Implement User Model and Database      │
╰──────────────────────────────────────────────────────────╯

# Implementation would create:
# - src/models/user.py
# - src/database.py  
# - alembic migrations
# - tests/test_user_model.py
```