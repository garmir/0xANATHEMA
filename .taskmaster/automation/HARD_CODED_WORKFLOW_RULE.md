
# HARD-CODED WORKFLOW RULE: Atomic Task Breakdown for Stuck Prompts

## Rule Definition
When prompts get stuck or become too complex:
1. Run 'task-master research' for breakdown analysis
2. Recursively expand tasks to atomic levels using task-master expand
3. Use 'task-master next' to get atomic tasks for simple execution
4. Execute atomic tasks as prompts with reduced complexity

## Trigger Conditions
- Prompt complexity too high
- Task execution stalled
- Multiple failed attempts
- User requests atomic breakdown

## Execution Protocol
```python
from atomic_task_breakdown_workflow import StuckPromptDetector

detector = StuckPromptDetector("/Users/anam/archive")
results = detector.detect_and_resolve_stuck_prompt("manual_trigger")
```

## Success Criteria
- Complex tasks broken into atomic components
- Each atomic task executable in 5-15 minutes
- Clear, simple execution prompts generated
- Reduced cognitive load for task completion

This rule is now HARD-CODED into the Task-Master workflow system.
