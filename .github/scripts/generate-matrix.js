#!/usr/bin/env node
/**
 * Generate Processing Matrix for Parallel Todo Processing
 * Creates optimized batch assignments for GitHub Actions parallel execution
 */

const fs = require('fs').promises;
const path = require('path');

class MatrixGenerator {
    constructor(options = {}) {
        this.options = {
            maxParallel: parseInt(options.maxParallel) || 10,
            inputPath: options.input || '.taskmaster/extraction/todos.json',
            outputPath: options.output || '.taskmaster/extraction/matrix.json',
            balanceStrategy: options.balanceStrategy || 'priority_weighted',
            ...options
        };
    }

    async generateMatrix() {
        console.log('📊 Generating processing matrix for parallel execution...');
        
        try {
            // Load extracted todos
            const todosData = await this.loadTodos();
            
            // Generate optimized batches
            const matrix = await this.createOptimizedMatrix(todosData);
            
            // Validate matrix
            this.validateMatrix(matrix, todosData);
            
            // Save matrix
            await this.saveMatrix(matrix);
            
            console.log(`✅ Matrix generated: ${matrix.length} batches for ${todosData.todos.length} todos`);
            
            return matrix;
            
        } catch (error) {
            console.error('❌ Matrix generation failed:', error);
            throw error;
        }
    }

    async loadTodos() {
        console.log(`📂 Loading todos from ${this.options.inputPath}...`);
        
        const data = await fs.readFile(this.options.inputPath, 'utf8');
        const todosData = JSON.parse(data);
        
        if (!todosData.todos || !Array.isArray(todosData.todos)) {
            throw new Error('Invalid todos data structure');
        }
        
        console.log(`📋 Loaded ${todosData.todos.length} todos`);
        return todosData;
    }

    async createOptimizedMatrix(todosData) {
        console.log(`🔄 Creating optimized matrix with strategy: ${this.options.balanceStrategy}`);
        
        const todos = todosData.todos;
        const totalTodos = todos.length;
        
        if (totalTodos === 0) {
            return [];
        }
        
        // Choose batching strategy
        let matrix;
        switch (this.options.balanceStrategy) {
            case 'priority_weighted':
                matrix = this.createPriorityWeightedMatrix(todos);
                break;
            case 'dependency_aware':
                matrix = this.createDependencyAwareMatrix(todos, todosData);
                break;
            case 'complexity_balanced':
                matrix = this.createComplexityBalancedMatrix(todos);
                break;
            case 'simple_round_robin':
                matrix = this.createSimpleRoundRobinMatrix(todos);
                break;
            default:
                matrix = this.createPriorityWeightedMatrix(todos);
        }
        
        // Add metadata to each batch
        matrix = matrix.map((batch, index) => ({
            ...batch,
            id: index + 1,
            metadata: {
                strategy: this.options.balanceStrategy,
                createdAt: new Date().toISOString(),
                estimatedDuration: this.estimateBatchDuration(batch.todos),
                complexity: this.calculateBatchComplexity(batch.todos),
                dependencies: this.analyzeBatchDependencies(batch.todos, todosData)
            }
        }));
        
        return matrix;
    }

    createPriorityWeightedMatrix(todos) {
        console.log('⚖️ Creating priority-weighted matrix...');
        
        // Sort todos by priority and complexity
        const sortedTodos = [...todos].sort((a, b) => {
            const priorityOrder = { 'critical': 4, 'high': 3, 'medium': 2, 'low': 1 };
            const aPriority = priorityOrder[a.priority] || 1;
            const bPriority = priorityOrder[b.priority] || 1;
            
            if (aPriority !== bPriority) {
                return bPriority - aPriority; // Higher priority first
            }
            
            // Secondary sort by depth (complexity)
            return (b.depth || 0) - (a.depth || 0);
        });
        
        // Distribute todos across batches using weighted round-robin
        const batches = Array(this.options.maxParallel).fill(null).map(() => ({
            todos: [],
            totalWeight: 0,
            todo_count: 0
        }));
        
        const priorityWeights = { 'critical': 4, 'high': 3, 'medium': 2, 'low': 1 };
        
        for (const todo of sortedTodos) {
            // Find the batch with the lowest current weight
            const lighterBatch = batches.reduce((min, batch, index) => {
                return batch.totalWeight < batches[min].totalWeight ? index : min;
            }, 0);
            
            const weight = priorityWeights[todo.priority] || 1;
            batches[lighterBatch].todos.push(todo);
            batches[lighterBatch].totalWeight += weight;
            batches[lighterBatch].todo_count++;
        }
        
        // Filter out empty batches
        return batches.filter(batch => batch.todos.length > 0);
    }

    createDependencyAwareMatrix(todos, todosData) {
        console.log('🔗 Creating dependency-aware matrix...');
        
        const dependencies = todosData.dependencies || {};
        const hierarchy = todosData.hierarchy || {};
        
        // Build dependency graph
        const dependencyGraph = new Map();
        const inDegree = new Map();
        
        for (const todo of todos) {
            dependencyGraph.set(todo.id, []);
            inDegree.set(todo.id, 0);
        }
        
        // Populate dependency relationships
        for (const todo of todos) {
            const deps = dependencies[todo.id];
            if (deps && deps.dependsOn) {
                for (const depId of deps.dependsOn) {
                    if (dependencyGraph.has(depId)) {
                        dependencyGraph.get(depId).push(todo.id);
                        inDegree.set(todo.id, inDegree.get(todo.id) + 1);
                    }
                }
            }
        }
        
        // Topological sort to respect dependencies
        const sortedTodos = this.topologicalSort(todos, dependencyGraph, inDegree);
        
        // Distribute respecting dependencies
        const batches = Array(this.options.maxParallel).fill(null).map(() => ({
            todos: [],
            todo_count: 0,
            dependencyLevel: 0
        }));
        
        const todoToBatch = new Map();
        
        for (const todo of sortedTodos) {
            // Find the earliest available batch that satisfies dependencies
            let availableBatch = 0;
            const deps = dependencies[todo.id];
            
            if (deps && deps.dependsOn && deps.dependsOn.length > 0) {
                let maxDepLevel = 0;
                for (const depId of deps.dependsOn) {
                    if (todoToBatch.has(depId)) {
                        const depBatchLevel = todoToBatch.get(depId);
                        maxDepLevel = Math.max(maxDepLevel, depBatchLevel + 1);
                    }
                }
                availableBatch = maxDepLevel % this.options.maxParallel;
            }
            
            // Find the best batch starting from available batch
            let bestBatch = availableBatch;
            for (let i = 0; i < this.options.maxParallel; i++) {
                const batchIndex = (availableBatch + i) % this.options.maxParallel;
                if (batches[batchIndex].todos.length < batches[bestBatch].todos.length) {
                    bestBatch = batchIndex;
                }
            }
            
            batches[bestBatch].todos.push(todo);
            batches[bestBatch].todo_count++;
            todoToBatch.set(todo.id, bestBatch);
        }
        
        return batches.filter(batch => batch.todos.length > 0);
    }

    createComplexityBalancedMatrix(todos) {
        console.log('🧠 Creating complexity-balanced matrix...');
        
        // Calculate complexity score for each todo
        const todosWithComplexity = todos.map(todo => ({
            ...todo,
            complexityScore: this.calculateTodoComplexity(todo)
        }));
        
        // Sort by complexity (descending)
        todosWithComplexity.sort((a, b) => b.complexityScore - a.complexityScore);
        
        // Use bin packing algorithm for balanced distribution
        const batches = Array(this.options.maxParallel).fill(null).map(() => ({
            todos: [],
            totalComplexity: 0,
            todo_count: 0
        }));
        
        for (const todo of todosWithComplexity) {
            // Find the batch with the lowest current complexity
            const lighterBatch = batches.reduce((min, batch, index) => {
                return batch.totalComplexity < batches[min].totalComplexity ? index : min;
            }, 0);
            
            batches[lighterBatch].todos.push(todo);
            batches[lighterBatch].totalComplexity += todo.complexityScore;
            batches[lighterBatch].todo_count++;
        }
        
        return batches.filter(batch => batch.todos.length > 0);
    }

    createSimpleRoundRobinMatrix(todos) {
        console.log('🔄 Creating simple round-robin matrix...');
        
        const batches = Array(this.options.maxParallel).fill(null).map(() => ({
            todos: [],
            todo_count: 0
        }));
        
        todos.forEach((todo, index) => {
            const batchIndex = index % this.options.maxParallel;
            batches[batchIndex].todos.push(todo);
            batches[batchIndex].todo_count++;
        });
        
        return batches.filter(batch => batch.todos.length > 0);
    }

    topologicalSort(todos, dependencyGraph, inDegree) {
        const queue = [];
        const result = [];
        
        // Find all todos with no dependencies
        for (const todo of todos) {
            if (inDegree.get(todo.id) === 0) {
                queue.push(todo);
            }
        }
        
        while (queue.length > 0) {
            const currentTodo = queue.shift();
            result.push(currentTodo);
            
            // Reduce in-degree for all dependents
            const dependents = dependencyGraph.get(currentTodo.id) || [];
            for (const dependentId of dependents) {
                inDegree.set(dependentId, inDegree.get(dependentId) - 1);
                
                if (inDegree.get(dependentId) === 0) {
                    const dependentTodo = todos.find(t => t.id === dependentId);
                    if (dependentTodo) {
                        queue.push(dependentTodo);
                    }
                }
            }
        }
        
        // If we couldn't process all todos, there might be cycles
        if (result.length !== todos.length) {
            console.warn('⚠️ Potential dependency cycles detected, using partial topological order');
            // Add remaining todos at the end
            for (const todo of todos) {
                if (!result.includes(todo)) {
                    result.push(todo);
                }
            }
        }
        
        return result;
    }

    calculateTodoComplexity(todo) {
        let complexity = 1; // Base complexity
        
        // Priority factor
        const priorityFactors = { 'critical': 3, 'high': 2, 'medium': 1, 'low': 0.5 };
        complexity += priorityFactors[todo.priority] || 1;
        
        // Depth factor (nested todos are more complex)
        complexity += (todo.depth || 0) * 0.5;
        
        // Dependencies factor
        complexity += (todo.dependencies || []).length * 0.3;
        
        // Subtasks factor
        complexity += (todo.subtasks || []).length * 0.2;
        
        // Description length factor (longer descriptions suggest complexity)
        const descLength = (todo.description || '').length;
        complexity += Math.min(descLength / 100, 2); // Cap at 2 points
        
        // Source factor
        const sourceFactors = { 'code': 1.5, 'taskmaster': 1.0, 'git': 0.8, 'subtask': 1.2 };
        complexity *= sourceFactors[todo.source] || 1.0;
        
        return Math.round(complexity * 100) / 100; // Round to 2 decimal places
    }

    estimateBatchDuration(todos) {
        const baseDurationPerTodo = 5; // minutes
        
        const totalDuration = todos.reduce((sum, todo) => {
            const complexity = this.calculateTodoComplexity(todo);
            return sum + (baseDurationPerTodo * complexity);
        }, 0);
        
        return Math.round(totalDuration);
    }

    calculateBatchComplexity(todos) {
        const totalComplexity = todos.reduce((sum, todo) => {
            return sum + this.calculateTodoComplexity(todo);
        }, 0);
        
        return Math.round((totalComplexity / todos.length) * 100) / 100;
    }

    analyzeBatchDependencies(todos, todosData) {
        const dependencies = todosData.dependencies || {};
        let internalDeps = 0;
        let externalDeps = 0;
        const todoIds = new Set(todos.map(t => t.id));
        
        for (const todo of todos) {
            const deps = dependencies[todo.id];
            if (deps && deps.dependsOn) {
                for (const depId of deps.dependsOn) {
                    if (todoIds.has(depId)) {
                        internalDeps++;
                    } else {
                        externalDeps++;
                    }
                }
            }
        }
        
        return {
            internal: internalDeps,
            external: externalDeps,
            ratio: internalDeps + externalDeps > 0 ? internalDeps / (internalDeps + externalDeps) : 1
        };
    }

    validateMatrix(matrix, todosData) {
        console.log('✅ Validating matrix...');
        
        const originalTodoCount = todosData.todos.length;
        const matrixTodoCount = matrix.reduce((sum, batch) => sum + batch.todos.length, 0);
        
        if (originalTodoCount !== matrixTodoCount) {
            throw new Error(`Todo count mismatch: original=${originalTodoCount}, matrix=${matrixTodoCount}`);
        }
        
        // Check for duplicate todos across batches
        const allTodoIds = new Set();
        for (const batch of matrix) {
            for (const todo of batch.todos) {
                if (allTodoIds.has(todo.id)) {
                    throw new Error(`Duplicate todo found: ${todo.id}`);
                }
                allTodoIds.add(todo.id);
            }
        }
        
        console.log(`✅ Matrix validation passed: ${matrix.length} batches, ${matrixTodoCount} todos`);
    }

    async saveMatrix(matrix) {
        console.log(`💾 Saving matrix to ${this.options.outputPath}...`);
        
        const matrixData = {
            metadata: {
                generatedAt: new Date().toISOString(),
                strategy: this.options.balanceStrategy,
                maxParallel: this.options.maxParallel,
                totalBatches: matrix.length,
                totalTodos: matrix.reduce((sum, batch) => sum + batch.todos.length, 0)
            },
            batches: matrix.map(batch => ({
                id: batch.id,
                todo_count: batch.todos.length,
                todos: batch.todos.map(todo => todo.id), // Only store IDs to reduce size
                metadata: batch.metadata
            })),
            // Full matrix for GitHub Actions
            matrix: matrix.map(batch => ({
                id: batch.id,
                todo_count: batch.todos.length,
                priority: batch.metadata ? batch.metadata.complexity : 1,
                duration: batch.metadata ? batch.metadata.estimatedDuration : 30
            })),
            statistics: {
                averageBatchSize: Math.round(matrix.reduce((sum, batch) => sum + batch.todos.length, 0) / matrix.length),
                minBatchSize: Math.min(...matrix.map(batch => batch.todos.length)),
                maxBatchSize: Math.max(...matrix.map(batch => batch.todos.length)),
                totalEstimatedDuration: matrix.reduce((sum, batch) => sum + (batch.metadata ? batch.metadata.estimatedDuration : 30), 0)
            }
        };
        
        // Ensure output directory exists
        const outputDir = path.dirname(this.options.outputPath);
        await fs.mkdir(outputDir, { recursive: true });
        
        await fs.writeFile(this.options.outputPath, JSON.stringify(matrixData, null, 2));
        
        console.log(`✅ Matrix saved: ${matrix.length} batches`);
        
        // Also save the GitHub Actions matrix format
        const githubMatrix = matrix.map(batch => ({
            id: batch.id,
            todo_count: batch.todos.length,
            priority: batch.metadata ? batch.metadata.complexity : 1
        }));
        
        await fs.writeFile(
            path.join(path.dirname(this.options.outputPath), 'github-matrix.json'),
            JSON.stringify(githubMatrix, null, 2)
        );
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
            case '--input':
                options.input = value;
                break;
            case '--output':
                options.output = value;
                break;
            case '--max-parallel':
                options.maxParallel = value;
                break;
            case '--strategy':
                options.balanceStrategy = value;
                break;
        }
    }
    
    try {
        const generator = new MatrixGenerator(options);
        await generator.generateMatrix();
        console.log('🎉 Matrix generation completed successfully!');
        process.exit(0);
    } catch (error) {
        console.error('💥 Matrix generation failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { MatrixGenerator };