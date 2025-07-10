# Rollback Instructions

## Quick Rollback

To quickly disable the recursive todo processing system:

1. **Disable Workflow**:
   ```bash
   # Rename the workflow file to disable it
   mv .github/workflows/recursive-todo-processing.yml .github/workflows/recursive-todo-processing.yml.disabled
   ```

2. **Remove Deployment**:
   ```bash
   # Remove deployment artifacts
   rm -rf .taskmaster/deployment/
   ```

## Complete Rollback

To completely remove the system:

1. **Remove All Files**:
   ```bash
   rm .github/workflows/recursive-todo-processing.yml
   rm -rf .github/scripts/
   rm -rf .taskmaster/
   ```

2. **Clean Git History** (if needed):
   ```bash
   git rm .github/workflows/recursive-todo-processing.yml
   git rm -r .github/scripts/
   git commit -m "Rollback: Remove recursive todo processing system"
   ```

## Partial Rollback

To rollback specific components:

- **Disable automatic triggers**: Edit the workflow file and remove the `on:` triggers
- **Reduce processing scope**: Modify environment configuration
- **Disable recursive improvements**: Set `enableRecursive: false` in configuration

## Recovery

If rollback was performed in error:

1. **Restore from this deployment**:
   ```bash
   git checkout HEAD~1 -- .github/workflows/recursive-todo-processing.yml
   git checkout HEAD~1 -- .github/scripts/
   ```

2. **Re-run deployment**:
   ```bash
   node .github/scripts/deploy-system.js --environment production
   ```

---

Created during deployment on 2025-07-10T20:15:07.649Z
