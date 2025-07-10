# Recursive Task Breakdown Workflow Rule

**Created**: July 10, 2025  
**Purpose**: Hard-coded workflow rule for breaking down complex tasks to atomic levels  
**Trigger**: When tasks stall due to prompt complexity or API failures

## Workflow Rule Definition

### **Rule ID**: `RECURSIVE_BREAKDOWN_001`
### **Priority**: Critical (always execute when triggered)
### **Auto-trigger Conditions**:
1. Task has been in-progress for >10 minutes without completion
2. API calls fail with parsing errors
3. Prompt complexity exceeds manageable scope
4. Manual invocation via `task-master research --atomic-breakdown`

## Atomic Breakdown Process

### **Step 1: Task Analysis**
**Action**: Analyze current task for complexity indicators
**Atomic Components**:
- Identify main objective
- List all required actions
- Assess dependencies
- Determine scope boundaries

### **Step 2: Decomposition to Atomic Level**
**Action**: Break task into smallest possible units
**Atomic Components**:
- Single action items (one verb, one object)
- Single concept research
- Single implementation step
- Single validation check

### **Step 3: Create Atomic Subtasks**
**Action**: Generate individual subtasks for each atomic component
**Atomic Components**:
- Write subtask title (max 50 characters)
- Write subtask description (max 100 characters)
- Define single success criterion
- Set atomic execution time (<5 minutes)

### **Step 4: Sequential Execution**
**Action**: Execute atomic tasks in dependency order
**Atomic Components**:
- Queue atomic tasks
- Execute one at a time
- Validate completion
- Move to next atomic task

### **Step 5: Rollback Mechanism**
**Action**: Handle failures with automatic re-breakdown
**Atomic Components**:
- Detect failure/stall
- Re-analyze failed atomic task
- Further decompose if needed
- Resume execution

## Implementation of Current Task (45.3)

Applying this rule to the current task "Research State-of-the-Art Recursive and Meta-Improvement Approaches":

### **Atomic Task 1**: Research ROME Framework Details
**Action**: Find specific implementation details of ROME framework
**Duration**: 3 minutes
**Output**: 2-3 sentence summary with key features

### **Atomic Task 2**: Research LADDER Framework Implementation 
**Action**: Find implementation patterns for LADDER self-improvement
**Duration**: 3 minutes
**Output**: Implementation approach description

### **Atomic Task 3**: Research RAG Enhancement Techniques
**Action**: Identify specific RAG improvement methods
**Duration**: 3 minutes
**Output**: List of 3-5 enhancement techniques

### **Atomic Task 4**: Research QIRO Algorithm Applications
**Action**: Find quantum-inspired optimization applications
**Duration**: 3 minutes
**Output**: Application examples and benefits

### **Atomic Task 5**: Research Meta-Learning Methods
**Action**: Identify meta-learning approaches for workflows
**Duration**: 3 minutes
**Output**: 3-5 specific methodologies

### **Atomic Task 6**: Research Feedback Loop Optimization
**Action**: Find automated feedback loop techniques
**Duration**: 3 minutes
**Output**: Optimization strategies list

### **Atomic Task 7**: Research Knowledge Synthesis Methods
**Action**: Identify knowledge integration techniques
**Duration**: 3 minutes
**Output**: Synthesis methodology descriptions

### **Atomic Task 8**: Compare with Task-Master Current State
**Action**: Identify gaps and opportunities
**Duration**: 5 minutes
**Output**: Gap analysis summary

## Execution Protocol

### **Manual Execution Process**:
1. Start with Atomic Task 1
2. Execute as simple prompt
3. Document output
4. Move to Atomic Task 2
5. Continue until all atomic tasks complete
6. Synthesize final result

### **Automated Triggers** (Future Implementation):
```bash
# Add to .taskmaster/config.json
{
  "workflow_rules": {
    "recursive_breakdown": {
      "enabled": true,
      "trigger_conditions": [
        "task_stall_time > 600", 
        "api_parse_failures > 2",
        "prompt_complexity_score > 8"
      ],
      "max_recursion_depth": 3,
      "atomic_task_max_duration": 300
    }
  }
}
```

## Success Criteria

### **Immediate Success**:
- ✅ Current task (45.3) completed via atomic breakdown
- ✅ All research areas covered in atomic detail
- ✅ Synthesis report generated
- ✅ Integration with Task-Master workflows

### **Long-term Success**:
- ✅ Workflow rule documented and accessible
- ✅ Atomic breakdown reduces prompt complexity
- ✅ Failure recovery through re-breakdown
- ✅ Improved task completion rates

## Application Example

**Before Atomic Breakdown**:
```
Complex Task: "Research comprehensive state-of-the-art approaches across 8 methodologies and integrate findings with current system analysis"
Result: API parsing failure, task stall
```

**After Atomic Breakdown**:
```
Atomic Task 1: "Research ROME framework key features"
Atomic Task 2: "Research LADDER implementation approach"  
Atomic Task 3: "Research RAG enhancement methods"
[...8 atomic tasks total...]
Result: All tasks complete, synthesis successful
```

## Integration with Task-Master

### **Command Integration**:
```bash
# Manual trigger
task-master research --atomic-breakdown --id=45.3

# Auto-detection (future)
task-master config --enable-atomic-breakdown

# Status check
task-master workflow-rules --status
```

### **Monitoring Integration**:
- Track atomic task completion rates
- Monitor time-to-completion improvements
- Measure prompt complexity reduction
- Document workflow effectiveness

---

**Status**: Workflow Rule Established  
**Implementation**: Manual execution for current task  
**Future**: Automated triggers and monitoring