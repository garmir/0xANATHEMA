# Stuck Prompt Resolution Command

Executes the hard-coded atomic task breakdown workflow rule when prompts get stuck or become too complex.

## Command Usage
Manually trigger atomic task breakdown for stuck prompt resolution.

## Execution Steps
1. Run task-master research for breakdown analysis
2. Recursively expand current task to atomic levels using task-master expand --id=<current_task> --research --force
3. Use task-master next to get the next atomic task (5-15 minute execution window)
4. Execute the atomic task as a simple, focused prompt
5. Mark task complete and continue with next atomic task

## When to Use
- Prompts are getting too complex or overwhelming
- Task execution has stalled or failed multiple times
- Need to break down work into manageable atomic pieces
- Want to reduce cognitive load and simplify execution

## Expected Outcome
- Complex tasks broken into 5-15 minute atomic components
- Clear, simple execution prompts generated
- Reduced complexity and cognitive load
- Smooth task completion workflow

This command implements the hard-coded workflow rule for automatic stuck prompt resolution.