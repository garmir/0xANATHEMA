#!/usr/bin/env node
/**
 * Recursive Todo Atomization Engine
 * Breaks down complex todos into atomic, actionable tasks
 */

const fs = require('fs').promises;
const path = require('path');

class TodoAtomizer {
    constructor(options = {}) {
        this.options = {
            inputPath: options.input || '.taskmaster/processing/batch-1/results.json',
            outputPath: options.output || '.taskmaster/processing/batch-1/atomized.json',
            maxDepth: parseInt(options.depth) || 5,
            maxAtomsPerTodo: parseInt(options.maxAtoms) || 20,
            atomizationStrategy: options.strategy || 'intelligent',
            ...options
        };
        
        this.atomizedResults = {
            originalTodos: [],
            atomizedTodos: [],
            atomizationStats: {
                totalOriginal: 0,
                totalAtomized: 0,
                averageAtomizationFactor: 0,
                maxDepthReached: 0,
                strategiesUsed: new Set()
            },
            atomizationTree: new Map(),
            warnings: [],
            errors: []
        };
    }

    async atomizeTodos() {
        console.log('⚛️ Starting recursive todo atomization...');
        
        try {
            // Load batch results
            const batchResults = await this.loadBatchResults();
            
            // Atomize each processed todo
            for (const todoResult of batchResults.processedTodos) {
                if (todoResult.success) {
                    await this.atomizeTodo(todoResult, 0);
                }
            }
            
            // Generate atomization analysis
            await this.analyzeAtomization();
            
            // Save atomized results
            await this.saveAtomizedResults();
            
            console.log(`✅ Atomization complete: ${this.atomizedResults.atomizationStats.totalOriginal} → ${this.atomizedResults.atomizationStats.totalAtomized} atomic tasks`);
            
        } catch (error) {
            console.error('❌ Atomization failed:', error);
            throw error;
        }
    }

    async loadBatchResults() {
        console.log(`📂 Loading batch results from ${this.options.inputPath}...`);
        
        const data = await fs.readFile(this.options.inputPath, 'utf8');
        const batchResults = JSON.parse(data);
        
        this.atomizedResults.originalTodos = batchResults.processedTodos || [];
        this.atomizedResults.atomizationStats.totalOriginal = this.atomizedResults.originalTodos.length;
        
        console.log(`📋 Loaded ${this.atomizedResults.atomizationStats.totalOriginal} todos for atomization`);
        
        return batchResults;
    }

    async atomizeTodo(todoResult, currentDepth) {
        if (currentDepth >= this.options.maxDepth) {
            this.atomizedResults.warnings.push({
                type: 'max_depth_reached',
                todoId: todoResult.id,
                depth: currentDepth,
                message: `Maximum atomization depth (${this.options.maxDepth}) reached for todo ${todoResult.id}`
            });
            return [todoResult];
        }
        
        console.log(`⚛️ Atomizing todo: ${todoResult.id} (depth: ${currentDepth})`);
        
        try {
            // Determine if todo is already atomic
            const isAtomic = await this.isAtomicTodo(todoResult);
            
            if (isAtomic) {
                // Todo is already atomic, add to results
                const atomicTodo = this.createAtomicTodoRecord(todoResult, currentDepth);
                this.atomizedResults.atomizedTodos.push(atomicTodo);
                return [atomicTodo];
            }
            
            // Apply atomization strategy
            const atomizationStrategy = this.selectAtomizationStrategy(todoResult);
            this.atomizedResults.atomizationStats.strategiesUsed.add(atomizationStrategy);
            
            const atoms = await this.applyAtomizationStrategy(todoResult, atomizationStrategy, currentDepth);
            
            // Recursively atomize the generated atoms
            const finalAtoms = [];
            for (const atom of atoms) {
                const subAtoms = await this.atomizeTodo(atom, currentDepth + 1);
                finalAtoms.push(...subAtoms);
            }
            
            // Update atomization tree
            this.atomizedResults.atomizationTree.set(todoResult.id, {
                originalTodo: todoResult,
                strategy: atomizationStrategy,
                depth: currentDepth,
                atoms: finalAtoms.map(atom => atom.id),
                atomCount: finalAtoms.length
            });
            
            // Update max depth reached
            this.atomizedResults.atomizationStats.maxDepthReached = Math.max(
                this.atomizedResults.atomizationStats.maxDepthReached,
                currentDepth
            );
            
            return finalAtoms;
            
        } catch (error) {
            console.error(`❌ Failed to atomize todo ${todoResult.id}:`, error);
            this.atomizedResults.errors.push({
                type: 'atomization_error',
                todoId: todoResult.id,
                depth: currentDepth,
                error: error.message
            });
            
            // Return original todo as fallback
            const atomicTodo = this.createAtomicTodoRecord(todoResult, currentDepth);
            this.atomizedResults.atomizedTodos.push(atomicTodo);
            return [atomicTodo];
        }
    }

    async isAtomicTodo(todoResult) {
        // Define atomic criteria
        const atomicCriteria = {
            maxTitleLength: 50,
            maxDescriptionLength: 200,
            maxEstimatedDuration: 30, // minutes
            requiredActionTypes: ['single_step', 'simple_check', 'atomic_update']
        };
        
        // Check title length
        if (todoResult.title && todoResult.title.length > atomicCriteria.maxTitleLength) {
            return false;
        }
        
        // Check description complexity
        const description = todoResult.description || '';
        if (description.length > atomicCriteria.maxDescriptionLength) {
            return false;
        }
        
        // Check for complex keywords that suggest non-atomic nature
        const complexKeywords = [
            'implement', 'develop', 'create system', 'build framework',
            'integrate multiple', 'comprehensive', 'end-to-end',
            'full stack', 'complete solution', 'entire', 'all aspects'
        ];
        
        const text = `${todoResult.title} ${description}`.toLowerCase();
        for (const keyword of complexKeywords) {
            if (text.includes(keyword)) {
                return false;
            }
        }
        
        // Check if todo has multiple distinct actions
        if (todoResult.actions && todoResult.actions.length > 3) {
            return false;
        }
        
        // Check for multiple dependencies or improvements
        if (todoResult.improvements && todoResult.improvements.length > 2) {
            return false;
        }
        
        // Check estimated duration if available
        if (todoResult.estimatedDuration && todoResult.estimatedDuration > atomicCriteria.maxEstimatedDuration) {
            return false;
        }
        
        return true;
    }

    selectAtomizationStrategy(todoResult) {
        const text = `${todoResult.title} ${todoResult.description || ''}`.toLowerCase();
        
        // Strategy selection based on todo characteristics
        if (text.includes('test') || text.includes('validation')) {
            return 'test_decomposition';
        }
        
        if (text.includes('implement') || text.includes('develop')) {
            return 'implementation_breakdown';
        }
        
        if (text.includes('fix') || text.includes('bug') || text.includes('error')) {
            return 'debugging_workflow';
        }
        
        if (text.includes('deploy') || text.includes('release')) {
            return 'deployment_pipeline';
        }
        
        if (text.includes('optimize') || text.includes('performance')) {
            return 'optimization_steps';
        }
        
        if (text.includes('document') || text.includes('readme')) {
            return 'documentation_workflow';
        }
        
        if (todoResult.actions && todoResult.actions.length > 1) {
            return 'action_decomposition';
        }
        
        return 'generic_breakdown';
    }

    async applyAtomizationStrategy(todoResult, strategy, currentDepth) {
        console.log(`🔄 Applying ${strategy} strategy to ${todoResult.id}`);
        
        switch (strategy) {
            case 'test_decomposition':
                return this.decomposeTestingTodo(todoResult, currentDepth);
            case 'implementation_breakdown':
                return this.breakdownImplementationTodo(todoResult, currentDepth);
            case 'debugging_workflow':
                return this.createDebuggingWorkflow(todoResult, currentDepth);
            case 'deployment_pipeline':
                return this.createDeploymentPipeline(todoResult, currentDepth);
            case 'optimization_steps':
                return this.createOptimizationSteps(todoResult, currentDepth);
            case 'documentation_workflow':
                return this.createDocumentationWorkflow(todoResult, currentDepth);
            case 'action_decomposition':
                return this.decomposeActions(todoResult, currentDepth);
            case 'generic_breakdown':
            default:
                return this.genericBreakdown(todoResult, currentDepth);
        }
    }

    decomposeTestingTodo(todoResult, currentDepth) {
        const atoms = [];
        const baseId = todoResult.id;
        
        // Standard testing workflow atoms
        const testingSteps = [
            {
                suffix: 'setup',
                title: 'Set up test environment',
                description: 'Prepare testing environment and dependencies',
                category: 'test_setup'
            },
            {
                suffix: 'unit',
                title: 'Write unit tests',
                description: 'Create unit tests for individual components',
                category: 'unit_testing'
            },
            {
                suffix: 'integration',
                title: 'Write integration tests',
                description: 'Create tests for component interactions',
                category: 'integration_testing'
            },
            {
                suffix: 'validate',
                title: 'Validate test results',
                description: 'Review and validate test outcomes',
                category: 'test_validation'
            },
            {
                suffix: 'cleanup',
                title: 'Clean up test artifacts',
                description: 'Remove temporary test files and reset state',
                category: 'test_cleanup'
            }
        ];
        
        for (let i = 0; i < testingSteps.length; i++) {
            const step = testingSteps[i];
            atoms.push(this.createAtomFromTemplate(todoResult, step, i + 1, currentDepth));
        }
        
        return atoms;
    }

    breakdownImplementationTodo(todoResult, currentDepth) {
        const atoms = [];
        const baseId = todoResult.id;
        
        // Standard implementation workflow atoms
        const implementationSteps = [
            {
                suffix: 'design',
                title: 'Design implementation approach',
                description: 'Plan the implementation strategy and architecture',
                category: 'design'
            },
            {
                suffix: 'setup',
                title: 'Set up development environment',
                description: 'Prepare development tools and dependencies',
                category: 'setup'
            },
            {
                suffix: 'core',
                title: 'Implement core functionality',
                description: 'Build the main features and logic',
                category: 'core_implementation'
            },
            {
                suffix: 'integration',
                title: 'Integrate with existing systems',
                description: 'Connect new implementation with existing codebase',
                category: 'integration'
            },
            {
                suffix: 'testing',
                title: 'Test implementation',
                description: 'Verify implementation works correctly',
                category: 'testing'
            },
            {
                suffix: 'documentation',
                title: 'Document implementation',
                description: 'Create documentation for the new implementation',
                category: 'documentation'
            }
        ];
        
        for (let i = 0; i < implementationSteps.length; i++) {
            const step = implementationSteps[i];
            atoms.push(this.createAtomFromTemplate(todoResult, step, i + 1, currentDepth));
        }
        
        return atoms;
    }

    createDebuggingWorkflow(todoResult, currentDepth) {
        const atoms = [];
        
        const debuggingSteps = [
            {
                suffix: 'reproduce',
                title: 'Reproduce the issue',
                description: 'Consistently reproduce the bug or error',
                category: 'bug_reproduction'
            },
            {
                suffix: 'isolate',
                title: 'Isolate the problem area',
                description: 'Narrow down the source of the issue',
                category: 'problem_isolation'
            },
            {
                suffix: 'analyze',
                title: 'Analyze root cause',
                description: 'Determine the underlying cause of the issue',
                category: 'root_cause_analysis'
            },
            {
                suffix: 'fix',
                title: 'Implement fix',
                description: 'Apply the necessary code changes',
                category: 'bug_fix'
            },
            {
                suffix: 'verify',
                title: 'Verify fix works',
                description: 'Test that the fix resolves the issue',
                category: 'fix_verification'
            },
            {
                suffix: 'regression',
                title: 'Check for regressions',
                description: 'Ensure fix does not break other functionality',
                category: 'regression_testing'
            }
        ];
        
        for (let i = 0; i < debuggingSteps.length; i++) {
            const step = debuggingSteps[i];
            atoms.push(this.createAtomFromTemplate(todoResult, step, i + 1, currentDepth));
        }
        
        return atoms;
    }

    createDeploymentPipeline(todoResult, currentDepth) {
        const atoms = [];
        
        const deploymentSteps = [
            {
                suffix: 'prepare',
                title: 'Prepare deployment package',
                description: 'Build and package application for deployment',
                category: 'deployment_preparation'
            },
            {
                suffix: 'staging',
                title: 'Deploy to staging',
                description: 'Deploy to staging environment for testing',
                category: 'staging_deployment'
            },
            {
                suffix: 'validate',
                title: 'Validate staging deployment',
                description: 'Test deployment in staging environment',
                category: 'deployment_validation'
            },
            {
                suffix: 'production',
                title: 'Deploy to production',
                description: 'Deploy to production environment',
                category: 'production_deployment'
            },
            {
                suffix: 'monitor',
                title: 'Monitor deployment',
                description: 'Monitor application health after deployment',
                category: 'deployment_monitoring'
            }
        ];
        
        for (let i = 0; i < deploymentSteps.length; i++) {
            const step = deploymentSteps[i];
            atoms.push(this.createAtomFromTemplate(todoResult, step, i + 1, currentDepth));
        }
        
        return atoms;
    }

    createOptimizationSteps(todoResult, currentDepth) {
        const atoms = [];
        
        const optimizationSteps = [
            {
                suffix: 'baseline',
                title: 'Establish performance baseline',
                description: 'Measure current performance metrics',
                category: 'performance_baseline'
            },
            {
                suffix: 'profile',
                title: 'Profile performance bottlenecks',
                description: 'Identify areas that need optimization',
                category: 'performance_profiling'
            },
            {
                suffix: 'optimize',
                title: 'Apply optimizations',
                description: 'Implement performance improvements',
                category: 'optimization_implementation'
            },
            {
                suffix: 'measure',
                title: 'Measure improvement',
                description: 'Quantify performance gains',
                category: 'performance_measurement'
            },
            {
                suffix: 'validate',
                title: 'Validate optimization',
                description: 'Ensure optimizations do not break functionality',
                category: 'optimization_validation'
            }
        ];
        
        for (let i = 0; i < optimizationSteps.length; i++) {
            const step = optimizationSteps[i];
            atoms.push(this.createAtomFromTemplate(todoResult, step, i + 1, currentDepth));
        }
        
        return atoms;
    }

    createDocumentationWorkflow(todoResult, currentDepth) {
        const atoms = [];
        
        const documentationSteps = [
            {
                suffix: 'outline',
                title: 'Create documentation outline',
                description: 'Plan documentation structure and content',
                category: 'documentation_planning'
            },
            {
                suffix: 'content',
                title: 'Write documentation content',
                description: 'Create the main documentation text',
                category: 'content_creation'
            },
            {
                suffix: 'examples',
                title: 'Add code examples',
                description: 'Include practical code examples and snippets',
                category: 'example_creation'
            },
            {
                suffix: 'review',
                title: 'Review documentation',
                description: 'Proofread and improve documentation quality',
                category: 'documentation_review'
            },
            {
                suffix: 'publish',
                title: 'Publish documentation',
                description: 'Make documentation available to users',
                category: 'documentation_publishing'
            }
        ];
        
        for (let i = 0; i < documentationSteps.length; i++) {
            const step = documentationSteps[i];
            atoms.push(this.createAtomFromTemplate(todoResult, step, i + 1, currentDepth));
        }
        
        return atoms;
    }

    decomposeActions(todoResult, currentDepth) {
        const atoms = [];
        
        if (!todoResult.actions || todoResult.actions.length === 0) {
            return [this.createAtomicTodoRecord(todoResult, currentDepth)];
        }
        
        // Create an atom for each action
        todoResult.actions.forEach((action, index) => {
            const atom = {
                id: `${todoResult.id}-action-${index + 1}`,
                title: `${action.type}: ${action.description.substring(0, 40)}...`,
                description: action.description,
                status: action.success ? 'completed' : 'pending',
                priority: todoResult.priority || 'medium',
                source: 'action_decomposition',
                parentId: todoResult.id,
                category: `action_${action.type}`,
                atomicLevel: currentDepth + 1,
                originalAction: action,
                metadata: {
                    atomizationStrategy: 'action_decomposition',
                    originalTodoId: todoResult.id,
                    actionIndex: index,
                    actionType: action.type
                }
            };
            
            atoms.push(atom);
        });
        
        return atoms;
    }

    genericBreakdown(todoResult, currentDepth) {
        const atoms = [];
        
        // Break down based on sentence structure or logical components
        const title = todoResult.title || '';
        const description = todoResult.description || '';
        
        // Simple heuristic: break by sentences or logical separators
        const text = `${title}. ${description}`;
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);
        
        if (sentences.length <= 1) {
            // Cannot break down further
            return [this.createAtomicTodoRecord(todoResult, currentDepth)];
        }
        
        sentences.forEach((sentence, index) => {
            if (sentence.trim().length > 0) {
                const atom = {
                    id: `${todoResult.id}-part-${index + 1}`,
                    title: sentence.trim().substring(0, 50),
                    description: sentence.trim(),
                    status: 'pending',
                    priority: todoResult.priority || 'medium',
                    source: 'generic_breakdown',
                    parentId: todoResult.id,
                    category: todoResult.category || 'general',
                    atomicLevel: currentDepth + 1,
                    metadata: {
                        atomizationStrategy: 'generic_breakdown',
                        originalTodoId: todoResult.id,
                        partIndex: index,
                        isGenericAtom: true
                    }
                };
                
                atoms.push(atom);
            }
        });
        
        return atoms.length > 0 ? atoms : [this.createAtomicTodoRecord(todoResult, currentDepth)];
    }

    createAtomFromTemplate(todoResult, template, stepNumber, currentDepth) {
        return {
            id: `${todoResult.id}-${template.suffix}`,
            title: template.title,
            description: template.description,
            status: 'pending',
            priority: todoResult.priority || 'medium',
            source: 'atomization',
            parentId: todoResult.id,
            category: template.category,
            atomicLevel: currentDepth + 1,
            stepNumber: stepNumber,
            metadata: {
                atomizationStrategy: this.selectAtomizationStrategy(todoResult),
                originalTodoId: todoResult.id,
                templateSuffix: template.suffix,
                isAtomicTask: true
            }
        };
    }

    createAtomicTodoRecord(todoResult, currentDepth) {
        return {
            id: todoResult.id,
            title: todoResult.title,
            description: todoResult.description || '',
            status: todoResult.status,
            priority: todoResult.priority || 'medium',
            source: todoResult.source || 'original',
            parentId: todoResult.parentId || null,
            category: todoResult.category || 'general',
            atomicLevel: currentDepth,
            isOriginalTodo: true,
            metadata: {
                alreadyAtomic: true,
                originalTodoId: todoResult.id
            }
        };
    }

    async analyzeAtomization() {
        console.log('📊 Analyzing atomization results...');
        
        this.atomizedResults.atomizationStats.totalAtomized = this.atomizedResults.atomizedTodos.length;
        
        if (this.atomizedResults.atomizationStats.totalOriginal > 0) {
            this.atomizedResults.atomizationStats.averageAtomizationFactor = 
                this.atomizedResults.atomizationStats.totalAtomized / 
                this.atomizedResults.atomizationStats.totalOriginal;
        }
        
        // Analyze strategy effectiveness
        const strategyStats = {};
        for (const strategy of this.atomizedResults.atomizationStats.strategiesUsed) {
            strategyStats[strategy] = {
                usageCount: 0,
                atomsProduced: 0,
                averageAtomsPerTodo: 0
            };
        }
        
        for (const [todoId, atomizationInfo] of this.atomizedResults.atomizationTree) {
            const strategy = atomizationInfo.strategy;
            if (strategyStats[strategy]) {
                strategyStats[strategy].usageCount++;
                strategyStats[strategy].atomsProduced += atomizationInfo.atomCount;
            }
        }
        
        // Calculate averages
        for (const strategy in strategyStats) {
            const stats = strategyStats[strategy];
            if (stats.usageCount > 0) {
                stats.averageAtomsPerTodo = stats.atomsProduced / stats.usageCount;
            }
        }
        
        this.atomizedResults.atomizationStats.strategyStats = strategyStats;
        
        console.log(`📈 Atomization analysis complete:`);
        console.log(`  - Original todos: ${this.atomizedResults.atomizationStats.totalOriginal}`);
        console.log(`  - Atomized todos: ${this.atomizedResults.atomizationStats.totalAtomized}`);
        console.log(`  - Atomization factor: ${this.atomizedResults.atomizationStats.averageAtomizationFactor.toFixed(2)}x`);
        console.log(`  - Max depth reached: ${this.atomizedResults.atomizationStats.maxDepthReached}`);
        console.log(`  - Strategies used: ${Array.from(this.atomizedResults.atomizationStats.strategiesUsed).join(', ')}`);
    }

    async saveAtomizedResults() {
        console.log(`💾 Saving atomized results to ${this.options.outputPath}...`);
        
        // Ensure output directory exists
        const outputDir = path.dirname(this.options.outputPath);
        await fs.mkdir(outputDir, { recursive: true });
        
        // Convert Map to Object for JSON serialization
        const atomizationTreeObj = {};
        for (const [key, value] of this.atomizedResults.atomizationTree) {
            atomizationTreeObj[key] = value;
        }
        
        const results = {
            ...this.atomizedResults,
            atomizationTree: atomizationTreeObj,
            atomizationStats: {
                ...this.atomizedResults.atomizationStats,
                strategiesUsed: Array.from(this.atomizedResults.atomizationStats.strategiesUsed)
            },
            metadata: {
                atomizedAt: new Date().toISOString(),
                atomizationOptions: this.options,
                version: '1.0.0'
            }
        };
        
        await fs.writeFile(this.options.outputPath, JSON.stringify(results, null, 2));
        
        // Save summary for quick access
        const summary = {
            totalOriginal: this.atomizedResults.atomizationStats.totalOriginal,
            totalAtomized: this.atomizedResults.atomizationStats.totalAtomized,
            atomizationFactor: this.atomizedResults.atomizationStats.averageAtomizationFactor,
            maxDepth: this.atomizedResults.atomizationStats.maxDepthReached,
            strategiesUsed: Array.from(this.atomizedResults.atomizationStats.strategiesUsed),
            warningsCount: this.atomizedResults.warnings.length,
            errorsCount: this.atomizedResults.errors.length,
            timestamp: new Date().toISOString()
        };
        
        await fs.writeFile(
            path.join(outputDir, 'atomization-summary.json'),
            JSON.stringify(summary, null, 2)
        );
        
        console.log(`✅ Atomized results saved`);
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
            case '--depth':
                options.depth = value;
                break;
            case '--max-atoms':
                options.maxAtoms = value;
                break;
            case '--strategy':
                options.strategy = value;
                break;
        }
    }
    
    try {
        const atomizer = new TodoAtomizer(options);
        await atomizer.atomizeTodos();
        console.log('🎉 Todo atomization completed successfully!');
        process.exit(0);
    } catch (error) {
        console.error('💥 Todo atomization failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { TodoAtomizer };