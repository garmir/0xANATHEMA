#!/usr/bin/env node
/**
 * Recursive Todo Extraction System
 * Extracts and analyzes all todos from task history with recursive depth tracking
 */

const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');

class TodoExtractor {
    constructor(options = {}) {
        this.options = {
            depth: parseInt(options.depth) || 5,
            includeHistory: options.includeHistory || false,
            includeSubtasks: options.includeSubtasks || false,
            includeDependencies: options.includeDependencies || false,
            outputPath: options.output || '.taskmaster/extraction/todos.json',
            ...options
        };
        
        this.extractedTodos = new Map();
        this.todoHierarchy = new Map();
        this.dependencies = new Map();
        this.statistics = {
            totalTodos: 0,
            completedTodos: 0,
            pendingTodos: 0,
            blockedTodos: 0,
            maxDepth: 0,
            categories: new Map()
        };
    }

    async extractAllTodos() {
        console.log('🔍 Starting recursive todo extraction...');
        
        try {
            // Extract from Task Master AI
            await this.extractFromTaskMaster();
            
            // Extract from git history if requested
            if (this.options.includeHistory) {
                await this.extractFromGitHistory();
            }
            
            // Extract from code comments and documentation
            await this.extractFromCodebase();
            
            // Build hierarchy and analyze dependencies
            await this.buildTodoHierarchy();
            
            // Generate comprehensive analysis
            await this.analyzeExtractedTodos();
            
            // Save results
            await this.saveResults();
            
            console.log(`✅ Extraction complete: ${this.statistics.totalTodos} todos extracted`);
            
        } catch (error) {
            console.error('❌ Todo extraction failed:', error);
            throw error;
        }
    }

    async extractFromTaskMaster() {
        console.log('📋 Extracting from Task Master AI...');
        
        try {
            // Get current tasks
            const tasksOutput = execSync('task-master list --format=json', { 
                encoding: 'utf8',
                cwd: process.cwd()
            });
            
            const tasks = JSON.parse(tasksOutput);
            
            for (const task of tasks) {
                const todo = {
                    id: `tm-${task.id}`,
                    title: task.title,
                    description: task.description,
                    status: task.status,
                    priority: task.priority,
                    source: 'taskmaster',
                    parentId: task.parentId || null,
                    dependencies: task.dependencies || [],
                    subtasks: task.subtasks || [],
                    depth: 0,
                    createdAt: task.createdAt || new Date().toISOString(),
                    updatedAt: task.updatedAt || new Date().toISOString(),
                    category: this.categorizeTask(task),
                    metadata: {
                        testStrategy: task.testStrategy,
                        details: task.details,
                        tags: this.extractTags(task)
                    }
                };
                
                this.extractedTodos.set(todo.id, todo);
                this.updateStatistics(todo);
            }
            
            // Extract subtasks recursively if requested
            if (this.options.includeSubtasks) {
                await this.extractSubtasksRecursively();
            }
            
        } catch (error) {
            console.warn('⚠️ Could not extract from Task Master AI:', error.message);
        }
    }

    async extractFromGitHistory() {
        console.log('📜 Extracting from git history...');
        
        try {
            // Get git log with todo-related commits
            const gitLogOutput = execSync(
                'git log --oneline --grep="TODO\\|FIXME\\|HACK\\|XXX" --all --since="6 months ago"',
                { encoding: 'utf8', cwd: process.cwd() }
            );
            
            const commits = gitLogOutput.split('\n').filter(line => line.trim());
            
            for (const commit of commits) {
                const [hash, ...messageParts] = commit.split(' ');
                const message = messageParts.join(' ');
                
                const todo = {
                    id: `git-${hash}`,
                    title: message.substring(0, 100),
                    description: message,
                    status: 'historical',
                    priority: 'low',
                    source: 'git',
                    parentId: null,
                    dependencies: [],
                    subtasks: [],
                    depth: 0,
                    createdAt: this.getCommitDate(hash),
                    updatedAt: this.getCommitDate(hash),
                    category: 'maintenance',
                    metadata: {
                        commitHash: hash,
                        tags: this.extractTagsFromMessage(message)
                    }
                };
                
                this.extractedTodos.set(todo.id, todo);
                this.updateStatistics(todo);
            }
            
        } catch (error) {
            console.warn('⚠️ Could not extract from git history:', error.message);
        }
    }

    async extractFromCodebase() {
        console.log('💻 Extracting from codebase...');
        
        try {
            // Find files with TODO/FIXME comments
            const grepOutput = execSync(
                'grep -r -n "\\(TODO\\|FIXME\\|HACK\\|XXX\\)" --include="*.js" --include="*.ts" --include="*.py" --include="*.md" . || true',
                { encoding: 'utf8', cwd: process.cwd() }
            );
            
            const matches = grepOutput.split('\n').filter(line => line.trim());
            
            for (const match of matches) {
                const [filePath, lineNumber, content] = match.split(':', 3);
                if (!filePath || !content) continue;
                
                const todoText = content.trim();
                const todo = {
                    id: `code-${Buffer.from(filePath + lineNumber + content).toString('base64').substring(0, 8)}`,
                    title: todoText.substring(0, 100),
                    description: todoText,
                    status: 'pending',
                    priority: this.prioritizeFromComment(todoText),
                    source: 'code',
                    parentId: null,
                    dependencies: [],
                    subtasks: [],
                    depth: 0,
                    createdAt: new Date().toISOString(),
                    updatedAt: new Date().toISOString(),
                    category: 'code-maintenance',
                    metadata: {
                        filePath: filePath,
                        lineNumber: parseInt(lineNumber),
                        language: this.detectLanguage(filePath),
                        tags: this.extractTagsFromComment(todoText)
                    }
                };
                
                this.extractedTodos.set(todo.id, todo);
                this.updateStatistics(todo);
            }
            
        } catch (error) {
            console.warn('⚠️ Could not extract from codebase:', error.message);
        }
    }

    async extractSubtasksRecursively(parentId = null, currentDepth = 0) {
        if (currentDepth >= this.options.depth) {
            return;
        }
        
        for (const [todoId, todo] of this.extractedTodos) {
            if (todo.parentId === parentId && todo.subtasks && todo.subtasks.length > 0) {
                for (const subtask of todo.subtasks) {
                    const subtaskTodo = {
                        id: `sub-${todoId}-${subtask.id || Date.now()}`,
                        title: subtask.title || 'Untitled Subtask',
                        description: subtask.description || '',
                        status: subtask.status || 'pending',
                        priority: subtask.priority || todo.priority,
                        source: 'subtask',
                        parentId: todoId,
                        dependencies: subtask.dependencies || [],
                        subtasks: subtask.subtasks || [],
                        depth: currentDepth + 1,
                        createdAt: new Date().toISOString(),
                        updatedAt: new Date().toISOString(),
                        category: todo.category,
                        metadata: {
                            parentTodo: todoId,
                            inheritedFromParent: true
                        }
                    };
                    
                    this.extractedTodos.set(subtaskTodo.id, subtaskTodo);
                    this.updateStatistics(subtaskTodo);
                    
                    // Recurse for nested subtasks
                    await this.extractSubtasksRecursively(subtaskTodo.id, currentDepth + 1);
                }
            }
        }
    }

    async buildTodoHierarchy() {
        console.log('🔗 Building todo hierarchy...');
        
        // Build parent-child relationships
        for (const [todoId, todo] of this.extractedTodos) {
            if (!this.todoHierarchy.has(todoId)) {
                this.todoHierarchy.set(todoId, {
                    todo: todo,
                    children: [],
                    parents: [],
                    depth: todo.depth || 0
                });
            }
            
            // Add children
            for (const [childId, childTodo] of this.extractedTodos) {
                if (childTodo.parentId === todoId) {
                    this.todoHierarchy.get(todoId).children.push(childId);
                    
                    if (!this.todoHierarchy.has(childId)) {
                        this.todoHierarchy.set(childId, {
                            todo: childTodo,
                            children: [],
                            parents: [],
                            depth: childTodo.depth || 0
                        });
                    }
                    this.todoHierarchy.get(childId).parents.push(todoId);
                }
            }
            
            // Build dependency graph if requested
            if (this.options.includeDependencies) {
                this.buildDependencyGraph(todo);
            }
        }
        
        // Calculate actual depths
        this.calculateTodoDepths();
    }

    buildDependencyGraph(todo) {
        if (!this.dependencies.has(todo.id)) {
            this.dependencies.set(todo.id, {
                dependsOn: [],
                dependents: [],
                criticalPath: false,
                blockingCount: 0
            });
        }
        
        // Process dependencies
        for (const depId of todo.dependencies) {
            this.dependencies.get(todo.id).dependsOn.push(depId);
            
            // Find the dependency todo and add this as a dependent
            for (const [checkId, checkTodo] of this.extractedTodos) {
                if (checkId === depId || checkTodo.id === depId) {
                    if (!this.dependencies.has(checkId)) {
                        this.dependencies.set(checkId, {
                            dependsOn: [],
                            dependents: [],
                            criticalPath: false,
                            blockingCount: 0
                        });
                    }
                    this.dependencies.get(checkId).dependents.push(todo.id);
                    break;
                }
            }
        }
    }

    calculateTodoDepths() {
        const calculateDepth = (todoId, visited = new Set()) => {
            if (visited.has(todoId)) {
                return 0; // Cycle detected
            }
            
            visited.add(todoId);
            const hierarchy = this.todoHierarchy.get(todoId);
            
            if (!hierarchy || hierarchy.parents.length === 0) {
                return 0;
            }
            
            let maxParentDepth = 0;
            for (const parentId of hierarchy.parents) {
                const parentDepth = calculateDepth(parentId, new Set(visited));
                maxParentDepth = Math.max(maxParentDepth, parentDepth);
            }
            
            return maxParentDepth + 1;
        };
        
        for (const [todoId, hierarchy] of this.todoHierarchy) {
            const depth = calculateDepth(todoId);
            hierarchy.depth = depth;
            hierarchy.todo.depth = depth;
            this.statistics.maxDepth = Math.max(this.statistics.maxDepth, depth);
        }
    }

    async analyzeExtractedTodos() {
        console.log('📊 Analyzing extracted todos...');
        
        // Analyze patterns and categories
        const patterns = {
            highPriorityCount: 0,
            blockedCount: 0,
            orphanedCount: 0,
            deeplyNestedCount: 0,
            recentCount: 0
        };
        
        const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
        
        for (const [todoId, todo] of this.extractedTodos) {
            // High priority todos
            if (todo.priority === 'high' || todo.priority === 'critical') {
                patterns.highPriorityCount++;
            }
            
            // Blocked todos
            if (todo.status === 'blocked' || todo.dependencies.length > 0) {
                patterns.blockedCount++;
            }
            
            // Orphaned todos (no parent, no children)
            const hierarchy = this.todoHierarchy.get(todoId);
            if (hierarchy && hierarchy.parents.length === 0 && hierarchy.children.length === 0) {
                patterns.orphanedCount++;
            }
            
            // Deeply nested todos
            if (todo.depth > 3) {
                patterns.deeplyNestedCount++;
            }
            
            // Recent todos
            if (new Date(todo.createdAt) > oneWeekAgo) {
                patterns.recentCount++;
            }
        }
        
        this.statistics.patterns = patterns;
        
        // Generate recommendations
        this.statistics.recommendations = this.generateRecommendations(patterns);
    }

    generateRecommendations(patterns) {
        const recommendations = [];
        
        if (patterns.highPriorityCount > this.statistics.totalTodos * 0.3) {
            recommendations.push({
                type: 'priority_rebalancing',
                message: 'Consider reviewing priority assignments - many todos are marked as high priority',
                action: 'audit_priorities'
            });
        }
        
        if (patterns.blockedCount > this.statistics.totalTodos * 0.2) {
            recommendations.push({
                type: 'dependency_optimization',
                message: 'Many todos are blocked - consider dependency optimization',
                action: 'resolve_dependencies'
            });
        }
        
        if (patterns.orphanedCount > this.statistics.totalTodos * 0.4) {
            recommendations.push({
                type: 'organization_improvement',
                message: 'Many todos lack clear hierarchy - consider better organization',
                action: 'improve_organization'
            });
        }
        
        if (patterns.deeplyNestedCount > 0) {
            recommendations.push({
                type: 'complexity_reduction',
                message: 'Some todos are deeply nested - consider breaking down complex tasks',
                action: 'simplify_structure'
            });
        }
        
        return recommendations;
    }

    async saveResults() {
        console.log('💾 Saving extraction results...');
        
        const results = {
            metadata: {
                extractedAt: new Date().toISOString(),
                extractionOptions: this.options,
                statistics: this.statistics
            },
            todos: Array.from(this.extractedTodos.values()),
            hierarchy: Object.fromEntries(
                Array.from(this.todoHierarchy.entries()).map(([id, hierarchy]) => [
                    id,
                    {
                        children: hierarchy.children,
                        parents: hierarchy.parents,
                        depth: hierarchy.depth
                    }
                ])
            ),
            dependencies: Object.fromEntries(this.dependencies),
            batching: this.generateBatchingStrategy()
        };
        
        // Ensure output directory exists
        const outputDir = path.dirname(this.options.outputPath);
        await fs.mkdir(outputDir, { recursive: true });
        
        // Save results
        await fs.writeFile(this.options.outputPath, JSON.stringify(results, null, 2));
        
        // Save statistics separately for quick access
        await fs.writeFile(
            path.join(outputDir, 'extraction-stats.json'),
            JSON.stringify(this.statistics, null, 2)
        );
        
        console.log(`✅ Results saved to ${this.options.outputPath}`);
    }

    generateBatchingStrategy() {
        const totalTodos = this.statistics.totalTodos;
        const maxParallel = parseInt(process.env.MAX_PARALLEL_RUNNERS) || 10;
        
        if (totalTodos === 0) {
            return { batches: [], totalBatches: 0, batchSize: 0 };
        }
        
        const batchSize = Math.ceil(totalTodos / maxParallel);
        const batches = [];
        const todoArray = Array.from(this.extractedTodos.values());
        
        for (let i = 0; i < totalTodos; i += batchSize) {
            const batchTodos = todoArray.slice(i, i + batchSize);
            batches.push({
                id: batches.length + 1,
                startIndex: i,
                endIndex: Math.min(i + batchSize - 1, totalTodos - 1),
                todoIds: batchTodos.map(todo => todo.id),
                todoCount: batchTodos.length,
                priority: this.calculateBatchPriority(batchTodos),
                estimatedDuration: this.estimateBatchDuration(batchTodos)
            });
        }
        
        return {
            batches: batches,
            totalBatches: batches.length,
            batchSize: batchSize,
            parallelization: Math.min(batches.length, maxParallel)
        };
    }

    calculateBatchPriority(batchTodos) {
        const priorityWeights = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        };
        
        const totalWeight = batchTodos.reduce((sum, todo) => {
            return sum + (priorityWeights[todo.priority] || 1);
        }, 0);
        
        return totalWeight / batchTodos.length;
    }

    estimateBatchDuration(batchTodos) {
        // Base duration per todo in minutes
        const baseDuration = 5;
        
        // Complexity multipliers
        const complexityMultipliers = {
            'critical': 2.0,
            'high': 1.5,
            'medium': 1.0,
            'low': 0.5
        };
        
        const totalDuration = batchTodos.reduce((sum, todo) => {
            const multiplier = complexityMultipliers[todo.priority] || 1.0;
            const depMultiplier = 1 + (todo.dependencies.length * 0.2);
            const depthMultiplier = 1 + (todo.depth * 0.1);
            
            return sum + (baseDuration * multiplier * depMultiplier * depthMultiplier);
        }, 0);
        
        return Math.round(totalDuration);
    }

    // Helper methods
    categorizeTask(task) {
        const title = (task.title || '').toLowerCase();
        const description = (task.description || '').toLowerCase();
        const text = `${title} ${description}`;
        
        if (text.includes('implement') || text.includes('develop') || text.includes('create')) {
            return 'development';
        } else if (text.includes('test') || text.includes('validation')) {
            return 'testing';
        } else if (text.includes('fix') || text.includes('bug') || text.includes('error')) {
            return 'bugfix';
        } else if (text.includes('optimize') || text.includes('performance')) {
            return 'optimization';
        } else if (text.includes('document') || text.includes('readme')) {
            return 'documentation';
        } else if (text.includes('deploy') || text.includes('release')) {
            return 'deployment';
        } else {
            return 'general';
        }
    }

    extractTags(task) {
        const text = `${task.title || ''} ${task.description || ''}`;
        const tags = [];
        
        // Extract hashtags
        const hashtagMatches = text.match(/#[a-zA-Z0-9_]+/g);
        if (hashtagMatches) {
            tags.push(...hashtagMatches.map(tag => tag.substring(1)));
        }
        
        // Extract common keywords
        const keywords = ['urgent', 'blocking', 'quick', 'complex', 'research', 'experimental'];
        for (const keyword of keywords) {
            if (text.toLowerCase().includes(keyword)) {
                tags.push(keyword);
            }
        }
        
        return tags;
    }

    extractTagsFromMessage(message) {
        const tags = [];
        
        if (message.toLowerCase().includes('todo')) tags.push('todo');
        if (message.toLowerCase().includes('fixme')) tags.push('fixme');
        if (message.toLowerCase().includes('hack')) tags.push('hack');
        if (message.toLowerCase().includes('xxx')) tags.push('urgent');
        
        return tags;
    }

    extractTagsFromComment(comment) {
        const tags = [];
        
        if (comment.includes('TODO')) tags.push('todo');
        if (comment.includes('FIXME')) tags.push('fixme');
        if (comment.includes('HACK')) tags.push('hack');
        if (comment.includes('XXX')) tags.push('urgent');
        if (comment.includes('IMPORTANT')) tags.push('important');
        
        return tags;
    }

    prioritizeFromComment(comment) {
        const text = comment.toLowerCase();
        
        if (text.includes('urgent') || text.includes('critical') || text.includes('xxx')) {
            return 'high';
        } else if (text.includes('important') || text.includes('fixme')) {
            return 'medium';
        } else {
            return 'low';
        }
    }

    detectLanguage(filePath) {
        const ext = path.extname(filePath).toLowerCase();
        const languageMap = {
            '.js': 'javascript',
            '.ts': 'typescript',
            '.py': 'python',
            '.md': 'markdown',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.json': 'json'
        };
        
        return languageMap[ext] || 'unknown';
    }

    getCommitDate(hash) {
        try {
            const dateOutput = execSync(`git show -s --format=%ci ${hash}`, {
                encoding: 'utf8',
                cwd: process.cwd()
            });
            return new Date(dateOutput.trim()).toISOString();
        } catch (error) {
            return new Date().toISOString();
        }
    }

    updateStatistics(todo) {
        this.statistics.totalTodos++;
        
        switch (todo.status) {
            case 'done':
            case 'completed':
                this.statistics.completedTodos++;
                break;
            case 'blocked':
                this.statistics.blockedTodos++;
                break;
            default:
                this.statistics.pendingTodos++;
        }
        
        // Update category counts
        const category = todo.category || 'unknown';
        const currentCount = this.statistics.categories.get(category) || 0;
        this.statistics.categories.set(category, currentCount + 1);
    }
}

// CLI Interface
async function main() {
    const args = process.argv.slice(2);
    const options = {};
    
    for (let i = 0; i < args.length; i += 2) {
        const flag = args[i];
        const value = args[i + 1];
        
        switch (flag) {
            case '--output':
                options.output = value;
                break;
            case '--depth':
                options.depth = value;
                break;
            case '--include-history':
                options.includeHistory = true;
                i--; // No value for this flag
                break;
            case '--include-subtasks':
                options.includeSubtasks = true;
                i--; // No value for this flag
                break;
            case '--include-dependencies':
                options.includeDependencies = true;
                i--; // No value for this flag
                break;
        }
    }
    
    try {
        const extractor = new TodoExtractor(options);
        await extractor.extractAllTodos();
        console.log('🎉 Todo extraction completed successfully!');
        process.exit(0);
    } catch (error) {
        console.error('💥 Todo extraction failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { TodoExtractor };