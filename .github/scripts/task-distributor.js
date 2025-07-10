#!/usr/bin/env node
/**
 * Task Distribution System for GitHub Actions
 * Intelligently distributes tasks across multiple runners based on complexity and dependencies
 */

const fs = require('fs');
const path = require('path');

class TaskDistributor {
    constructor(maxRunners = 10) {
        this.maxRunners = maxRunners;
        this.tasks = [];
        this.runners = [];
        this.dependencyGraph = new Map();
    }

    loadTasks(tasksFilePath = '.taskmaster/tasks/tasks.json') {
        try {
            const tasksData = JSON.parse(fs.readFileSync(tasksFilePath, 'utf8'));
            this.tasks = tasksData.master.tasks || [];
            this.buildDependencyGraph();
            console.log(`📋 Loaded ${this.tasks.length} tasks`);
        } catch (error) {
            console.error(`❌ Failed to load tasks: ${error.message}`);
            this.tasks = [];
        }
    }

    buildDependencyGraph() {
        // Build dependency graph for topological sorting
        this.dependencyGraph.clear();
        
        for (const task of this.tasks) {
            const taskId = String(task.id);
            if (!this.dependencyGraph.has(taskId)) {
                this.dependencyGraph.set(taskId, { dependencies: [], dependents: [] });
            }
            
            // Add dependencies
            if (task.dependencies && Array.isArray(task.dependencies)) {
                for (const depId of task.dependencies) {
                    const depIdStr = String(depId);
                    
                    // Ensure dependency node exists
                    if (!this.dependencyGraph.has(depIdStr)) {
                        this.dependencyGraph.set(depIdStr, { dependencies: [], dependents: [] });
                    }
                    
                    // Add bidirectional relationship
                    this.dependencyGraph.get(taskId).dependencies.push(depIdStr);
                    this.dependencyGraph.get(depIdStr).dependents.push(taskId);
                }
            }
        }
        
        console.log(`🔗 Built dependency graph with ${this.dependencyGraph.size} nodes`);
    }

    calculateTaskComplexity(task) {
        let complexity = 1; // Base complexity
        
        // Increase complexity based on description length
        const descLength = (task.description || '').length + (task.details || '').length;
        complexity += Math.floor(descLength / 200);
        
        // Priority weight
        const priorityWeights = { 'high': 3, 'medium': 2, 'low': 1 };
        complexity += priorityWeights[task.priority] || 1;
        
        // Dependency complexity
        const depCount = (task.dependencies || []).length;
        complexity += depCount * 0.5;
        
        // Test strategy complexity
        if (task.testStrategy && task.testStrategy.length > 100) {
            complexity += 1;
        }
        
        return Math.max(1, Math.floor(complexity));
    }

    getExecutableTasksLevel() {
        // Find tasks that can be executed (no pending dependencies)
        const pendingTasks = this.tasks.filter(task => task.status === 'pending');
        const executable = [];
        
        for (const task of pendingTasks) {
            const taskId = String(task.id);
            const dependencies = this.dependencyGraph.get(taskId)?.dependencies || [];
            
            // Check if all dependencies are completed
            const canExecute = dependencies.every(depId => {
                const depTask = this.tasks.find(t => String(t.id) === depId);
                return depTask && depTask.status === 'done';
            });
            
            if (canExecute) {
                executable.push({
                    ...task,
                    complexity: this.calculateTaskComplexity(task)
                });
            }
        }
        
        // Sort by priority and complexity
        executable.sort((a, b) => {
            const priorityOrder = { 'high': 3, 'medium': 2, 'low': 1 };
            const aPriority = priorityOrder[a.priority] || 1;
            const bPriority = priorityOrder[b.priority] || 1;
            
            if (aPriority !== bPriority) {
                return bPriority - aPriority; // Higher priority first
            }
            
            return b.complexity - a.complexity; // Higher complexity first
        });
        
        return executable;
    }

    distributeTasksAcrossRunners(executableTasks, runnerCount) {
        // Initialize runners
        const runners = Array.from({ length: runnerCount }, (_, i) => ({
            id: i + 1,
            tasks: [],
            totalComplexity: 0,
            estimatedTime: 0
        }));
        
        // Distribute tasks using a greedy algorithm with load balancing
        for (const task of executableTasks) {
            // Find runner with least total complexity
            const targetRunner = runners.reduce((min, runner) => 
                runner.totalComplexity < min.totalComplexity ? runner : min
            );
            
            // Assign task to runner
            targetRunner.tasks.push({
                id: task.id,
                title: task.title,
                priority: task.priority,
                complexity: task.complexity
            });
            
            targetRunner.totalComplexity += task.complexity;
            targetRunner.estimatedTime += task.complexity * 5; // 5 minutes per complexity point
        }
        
        // Remove empty runners
        return runners.filter(runner => runner.tasks.length > 0);
    }

    generateDistributionMatrix(strategy = 'balanced', maxRunners = null) {
        const actualMaxRunners = maxRunners || this.maxRunners;
        const executableTasks = this.getExecutableTasksLevel();
        
        console.log(`🎯 Found ${executableTasks.length} executable tasks`);
        
        if (executableTasks.length === 0) {
            console.log('✅ No executable tasks found');
            return [];
        }
        
        // Determine optimal runner count based on strategy
        let runnerCount;
        
        switch (strategy) {
            case 'aggressive':
                // More runners for faster execution
                runnerCount = Math.min(executableTasks.length, actualMaxRunners);
                break;
                
            case 'conservative':
                // Fewer runners to reduce overhead
                runnerCount = Math.min(Math.ceil(executableTasks.length / 3), actualMaxRunners);
                break;
                
            case 'balanced':
            default:
                // Balanced approach: 1 runner per 2-3 tasks
                runnerCount = Math.min(Math.ceil(executableTasks.length / 2.5), actualMaxRunners);
                break;
        }
        
        runnerCount = Math.max(1, runnerCount);
        
        console.log(`⚖️ Using ${runnerCount} runners for ${executableTasks.length} tasks (${strategy} strategy)`);
        
        // Distribute tasks
        const distribution = this.distributeTasksAcrossRunners(executableTasks, runnerCount);
        
        // Generate GitHub Actions matrix format
        const matrix = distribution.map(runner => ({
            runner_id: runner.id,
            tasks: runner.tasks.map(t => t.id),
            estimated_complexity: runner.totalComplexity,
            estimated_time_minutes: runner.estimatedTime,
            task_count: runner.tasks.length
        }));
        
        return matrix;
    }

    generateExecutionPlan(strategy = 'balanced', maxRunners = null) {
        const matrix = this.generateDistributionMatrix(strategy, maxRunners);
        const executableTasks = this.getExecutableTasksLevel();
        
        const plan = {
            timestamp: new Date().toISOString(),
            strategy: strategy,
            total_executable_tasks: executableTasks.length,
            total_pending_tasks: this.tasks.filter(t => t.status === 'pending').length,
            runner_count: matrix.length,
            max_runners: maxRunners || this.maxRunners,
            distribution_matrix: matrix,
            execution_summary: {
                total_complexity: matrix.reduce((sum, r) => sum + r.estimated_complexity, 0),
                estimated_total_time: Math.max(...matrix.map(r => r.estimated_time_minutes), 0),
                load_balance_score: this.calculateLoadBalanceScore(matrix)
            },
            task_breakdown: {
                high_priority: executableTasks.filter(t => t.priority === 'high').length,
                medium_priority: executableTasks.filter(t => t.priority === 'medium').length,
                low_priority: executableTasks.filter(t => t.priority === 'low').length
            }
        };
        
        return plan;
    }

    calculateLoadBalanceScore(matrix) {
        if (matrix.length === 0) return 0;
        
        const complexities = matrix.map(r => r.estimated_complexity);
        const avgComplexity = complexities.reduce((a, b) => a + b, 0) / complexities.length;
        const variance = complexities.reduce((sum, c) => sum + Math.pow(c - avgComplexity, 2), 0) / complexities.length;
        const stdDev = Math.sqrt(variance);
        
        // Score from 0-1, where 1 is perfectly balanced
        return Math.max(0, 1 - (stdDev / avgComplexity));
    }

    saveExecutionPlan(plan, outputPath = '.github/task-execution-plan.json') {
        try {
            const dir = path.dirname(outputPath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            
            fs.writeFileSync(outputPath, JSON.stringify(plan, null, 2));
            console.log(`💾 Execution plan saved to ${outputPath}`);
        } catch (error) {
            console.error(`❌ Failed to save execution plan: ${error.message}`);
        }
    }

    printExecutionSummary(plan) {
        console.log('\n📊 TASK DISTRIBUTION SUMMARY');
        console.log('============================');
        console.log(`Strategy: ${plan.strategy}`);
        console.log(`Executable Tasks: ${plan.total_executable_tasks}`);
        console.log(`Runners: ${plan.runner_count}`);
        console.log(`Estimated Time: ${plan.execution_summary.estimated_total_time} minutes`);
        console.log(`Load Balance Score: ${(plan.execution_summary.load_balance_score * 100).toFixed(1)}%`);
        
        console.log('\n📋 Runner Distribution:');
        plan.distribution_matrix.forEach(runner => {
            console.log(`  Runner ${runner.runner_id}: ${runner.task_count} tasks (complexity: ${runner.estimated_complexity})`);
            console.log(`    Tasks: ${runner.tasks.join(', ')}`);
        });
        
        console.log('\n🎯 Priority Breakdown:');
        console.log(`  High: ${plan.task_breakdown.high_priority}`);
        console.log(`  Medium: ${plan.task_breakdown.medium_priority}`);
        console.log(`  Low: ${plan.task_breakdown.low_priority}`);
    }
}

// CLI interface
if (require.main === module) {
    const args = process.argv.slice(2);
    const strategy = args[0] || 'balanced';
    const maxRunners = parseInt(args[1]) || 10;
    
    console.log('🚀 Task Distribution System for GitHub Actions');
    console.log('==============================================');
    
    const distributor = new TaskDistributor(maxRunners);
    distributor.loadTasks();
    
    const plan = distributor.generateExecutionPlan(strategy, maxRunners);
    distributor.printExecutionSummary(plan);
    distributor.saveExecutionPlan(plan);
    
    // Output matrix for GitHub Actions
    if (plan.distribution_matrix.length > 0) {
        console.log('\n🔧 GitHub Actions Matrix:');
        console.log(JSON.stringify(plan.distribution_matrix, null, 2));
        
        // Set output for GitHub Actions
        if (process.env.GITHUB_OUTPUT) {
            const output = `task_matrix=${JSON.stringify(plan.distribution_matrix)}\n`;
            fs.appendFileSync(process.env.GITHUB_OUTPUT, output);
        }
    } else {
        console.log('\n✅ No tasks to distribute');
        
        if (process.env.GITHUB_OUTPUT) {
            fs.appendFileSync(process.env.GITHUB_OUTPUT, 'task_matrix=[]\n');
        }
    }
}

module.exports = TaskDistributor;