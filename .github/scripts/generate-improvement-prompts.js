#!/usr/bin/env node
/**
 * Improvement Prompt Generator
 * Generates recursive improvement prompts from atomized todo outputs
 */

const fs = require('fs').promises;
const path = require('path');

class ImprovementPromptGenerator {
    constructor(options = {}) {
        this.options = {
            inputPath: options.input || '.taskmaster/processing/batch-1/atomized.json',
            outputPath: options.output || '.taskmaster/processing/batch-1/improvement-prompts.json',
            enableRecursive: options.recursive !== false,
            maxPromptDepth: parseInt(options.maxDepth) || 3,
            promptCategories: options.categories || ['optimization', 'automation', 'quality', 'efficiency'],
            ...options
        };
        
        this.improvementPrompts = {
            metadata: {
                generatedAt: new Date().toISOString(),
                inputSource: this.options.inputPath,
                promptCategories: this.options.promptCategories,
                recursiveEnabled: this.options.enableRecursive
            },
            prompts: [],
            promptsByCategory: {},
            recursivePrompts: [],
            statistics: {
                totalPrompts: 0,
                categoryCounts: {},
                recursiveDepth: 0,
                todoSourceAnalysis: {}
            },
            errors: [],
            warnings: []
        };
    }

    async generateImprovementPrompts() {
        console.log('💡 Starting improvement prompt generation...');
        
        try {
            // Load atomized todo data
            const atomizedData = await this.loadAtomizedData();
            
            // Analyze atomized todos for improvement opportunities
            const improvementOpportunities = await this.analyzeImprovementOpportunities(atomizedData);
            
            // Generate base improvement prompts
            for (const opportunity of improvementOpportunities) {
                await this.generatePromptsForOpportunity(opportunity);
            }
            
            // Generate recursive improvement prompts if enabled
            if (this.options.enableRecursive) {
                await this.generateRecursivePrompts();
            }
            
            // Organize prompts by category
            this.organizePromptsByCategory();
            
            // Generate statistics
            this.generateStatistics();
            
            // Save improvement prompts
            await this.saveImprovementPrompts();
            
            console.log(`✅ Generated ${this.improvementPrompts.statistics.totalPrompts} improvement prompts`);
            
        } catch (error) {
            console.error('❌ Improvement prompt generation failed:', error);
            throw error;
        }
    }

    async loadAtomizedData() {
        console.log(`📂 Loading atomized data from ${this.options.inputPath}...`);
        
        const data = await fs.readFile(this.options.inputPath, 'utf8');
        const atomizedData = JSON.parse(data);
        
        if (!atomizedData.atomizedTodos || !Array.isArray(atomizedData.atomizedTodos)) {
            throw new Error('Invalid atomized data structure');
        }
        
        console.log(`📋 Loaded ${atomizedData.atomizedTodos.length} atomized todos for analysis`);
        
        return atomizedData;
    }

    async analyzeImprovementOpportunities(atomizedData) {
        console.log('🔍 Analyzing improvement opportunities...');
        
        const opportunities = [];
        
        // Analyze by todo categories
        const todosByCategory = this.groupTodosByCategory(atomizedData.atomizedTodos);
        
        for (const [category, todos] of Object.entries(todosByCategory)) {
            const categoryOpportunities = await this.analyzeCategoryOpportunities(category, todos);
            opportunities.push(...categoryOpportunities);
        }
        
        // Analyze by completion status
        const statusOpportunities = await this.analyzeStatusOpportunities(atomizedData.atomizedTodos);
        opportunities.push(...statusOpportunities);
        
        // Analyze by atomization patterns
        const atomizationOpportunities = await this.analyzeAtomizationOpportunities(atomizedData);
        opportunities.push(...atomizationOpportunities);
        
        // Analyze by source types
        const sourceOpportunities = await this.analyzeSourceOpportunities(atomizedData.atomizedTodos);
        opportunities.push(...sourceOpportunities);
        
        console.log(`🔍 Found ${opportunities.length} improvement opportunities`);
        
        return opportunities;
    }

    groupTodosByCategory(todos) {
        const grouped = {};
        
        for (const todo of todos) {
            const category = todo.category || 'general';
            if (!grouped[category]) {
                grouped[category] = [];
            }
            grouped[category].push(todo);
        }
        
        return grouped;
    }

    async analyzeCategoryOpportunities(category, todos) {
        const opportunities = [];
        
        // Category-specific analysis
        switch (category) {
            case 'development':
                opportunities.push(...this.analyzeDevelopmentTodos(todos));
                break;
            case 'testing':
                opportunities.push(...this.analyzeTestingTodos(todos));
                break;
            case 'code-maintenance':
                opportunities.push(...this.analyzeMaintenanceTodos(todos));
                break;
            case 'documentation':
                opportunities.push(...this.analyzeDocumentationTodos(todos));
                break;
            case 'deployment':
                opportunities.push(...this.analyzeDeploymentTodos(todos));
                break;
            default:
                opportunities.push(...this.analyzeGenericTodos(category, todos));
        }
        
        return opportunities;
    }

    analyzeDevelopmentTodos(todos) {
        const opportunities = [];
        
        // Look for code quality improvements
        const codeQualityTodos = todos.filter(todo => 
            todo.title.toLowerCase().includes('implement') ||
            todo.title.toLowerCase().includes('develop')
        );
        
        if (codeQualityTodos.length > 0) {
            opportunities.push({
                type: 'code_quality_automation',
                category: 'optimization',
                priority: 'high',
                todos: codeQualityTodos,
                description: 'Automate code quality checks for development todos',
                improvementAreas: ['linting', 'testing', 'code_review', 'documentation']
            });
        }
        
        // Look for repetitive patterns
        const titlePatterns = this.findRepetitivePatterns(todos.map(t => t.title));
        if (titlePatterns.length > 0) {
            opportunities.push({
                type: 'development_pattern_automation',
                category: 'automation',
                priority: 'medium',
                todos: todos,
                description: 'Create templates or automation for repetitive development patterns',
                patterns: titlePatterns,
                improvementAreas: ['templates', 'scaffolding', 'automation']
            });
        }
        
        return opportunities;
    }

    analyzeTestingTodos(todos) {
        const opportunities = [];
        
        // Test automation opportunities
        const manualTestTodos = todos.filter(todo =>
            todo.description && (
                todo.description.toLowerCase().includes('manual') ||
                todo.description.toLowerCase().includes('check')
            )
        );
        
        if (manualTestTodos.length > 0) {
            opportunities.push({
                type: 'test_automation',
                category: 'automation',
                priority: 'high',
                todos: manualTestTodos,
                description: 'Automate manual testing processes',
                improvementAreas: ['automated_testing', 'ci_cd', 'test_frameworks']
            });
        }
        
        // Test coverage improvements
        opportunities.push({
            type: 'test_coverage_improvement',
            category: 'quality',
            priority: 'medium',
            todos: todos,
            description: 'Enhance test coverage and quality',
            improvementAreas: ['coverage_analysis', 'test_quality', 'edge_cases']
        });
        
        return opportunities;
    }

    analyzeMaintenanceTodos(todos) {
        const opportunities = [];
        
        // Technical debt opportunities
        const techDebtTodos = todos.filter(todo =>
            todo.title.toLowerCase().includes('fix') ||
            todo.title.toLowerCase().includes('refactor') ||
            todo.description?.toLowerCase().includes('debt')
        );
        
        if (techDebtTodos.length > 0) {
            opportunities.push({
                type: 'technical_debt_reduction',
                category: 'optimization',
                priority: 'high',
                todos: techDebtTodos,
                description: 'Systematic technical debt reduction',
                improvementAreas: ['refactoring', 'modernization', 'cleanup']
            });
        }
        
        // Maintenance automation
        opportunities.push({
            type: 'maintenance_automation',
            category: 'automation',
            priority: 'medium',
            todos: todos,
            description: 'Automate routine maintenance tasks',
            improvementAreas: ['automated_fixes', 'monitoring', 'alerts']
        });
        
        return opportunities;
    }

    analyzeDocumentationTodos(todos) {
        const opportunities = [];
        
        // Documentation generation
        opportunities.push({
            type: 'documentation_automation',
            category: 'automation',
            priority: 'medium',
            todos: todos,
            description: 'Automate documentation generation and maintenance',
            improvementAreas: ['auto_generation', 'documentation_testing', 'consistency']
        });
        
        // Documentation quality
        const qualityIssues = todos.filter(todo =>
            !todo.description || todo.description.length < 50
        );
        
        if (qualityIssues.length > 0) {
            opportunities.push({
                type: 'documentation_quality',
                category: 'quality',
                priority: 'medium',
                todos: qualityIssues,
                description: 'Improve documentation quality and completeness',
                improvementAreas: ['completeness', 'clarity', 'examples']
            });
        }
        
        return opportunities;
    }

    analyzeDeploymentTodos(todos) {
        const opportunities = [];
        
        // Deployment automation
        opportunities.push({
            type: 'deployment_automation',
            category: 'automation',
            priority: 'high',
            todos: todos,
            description: 'Automate deployment processes',
            improvementAreas: ['ci_cd', 'infrastructure_as_code', 'monitoring']
        });
        
        // Deployment safety
        opportunities.push({
            type: 'deployment_safety',
            category: 'quality',
            priority: 'high',
            todos: todos,
            description: 'Improve deployment safety and reliability',
            improvementAreas: ['rollback_mechanisms', 'health_checks', 'gradual_rollout']
        });
        
        return opportunities;
    }

    analyzeGenericTodos(category, todos) {
        const opportunities = [];
        
        // Generic process improvement
        if (todos.length > 5) {
            opportunities.push({
                type: 'process_standardization',
                category: 'optimization',
                priority: 'medium',
                todos: todos,
                description: `Standardize and optimize ${category} processes`,
                improvementAreas: ['standardization', 'templates', 'best_practices']
            });
        }
        
        return opportunities;
    }

    async analyzeStatusOpportunities(todos) {
        const opportunities = [];
        
        // Analyze completion patterns
        const completedTodos = todos.filter(todo => 
            todo.status === 'completed' || todo.status === 'done'
        );
        const failedTodos = todos.filter(todo => 
            todo.status === 'failed' || todo.status === 'error'
        );
        const pendingTodos = todos.filter(todo => 
            todo.status === 'pending' || todo.status === 'todo'
        );
        
        // High failure rate
        if (failedTodos.length > todos.length * 0.2) {
            opportunities.push({
                type: 'failure_rate_reduction',
                category: 'quality',
                priority: 'high',
                todos: failedTodos,
                description: 'Reduce todo failure rate through better planning and execution',
                improvementAreas: ['error_analysis', 'prevention', 'recovery']
            });
        }
        
        // Stuck pending todos
        if (pendingTodos.length > todos.length * 0.3) {
            opportunities.push({
                type: 'pending_todo_acceleration',
                category: 'efficiency',
                priority: 'medium',
                todos: pendingTodos,
                description: 'Accelerate pending todo completion',
                improvementAreas: ['prioritization', 'resource_allocation', 'blocking_removal']
            });
        }
        
        return opportunities;
    }

    async analyzeAtomizationOpportunities(atomizedData) {
        const opportunities = [];
        
        // Check atomization effectiveness
        const atomizationStats = atomizedData.atomizationStats;
        
        if (atomizationStats.averageAtomizationFactor > 5) {
            opportunities.push({
                type: 'atomization_optimization',
                category: 'optimization',
                priority: 'medium',
                todos: atomizedData.atomizedTodos,
                description: 'Optimize atomization strategy to reduce over-atomization',
                improvementAreas: ['atomization_strategy', 'complexity_analysis', 'balance']
            });
        }
        
        if (atomizationStats.maxDepthReached >= atomizedData.options?.maxDepth) {
            opportunities.push({
                type: 'depth_limit_optimization',
                category: 'optimization',
                priority: 'low',
                todos: atomizedData.atomizedTodos,
                description: 'Optimize atomization depth limits',
                improvementAreas: ['depth_strategy', 'recursive_limits', 'complexity_balance']
            });
        }
        
        return opportunities;
    }

    async analyzeSourceOpportunities(todos) {
        const opportunities = [];
        
        // Group by source
        const todosBySource = {};
        for (const todo of todos) {
            const source = todo.source || 'unknown';
            if (!todosBySource[source]) {
                todosBySource[source] = [];
            }
            todosBySource[source].push(todo);
        }
        
        // Code source todos
        if (todosBySource.code && todosBySource.code.length > 0) {
            opportunities.push({
                type: 'code_todo_automation',
                category: 'automation',
                priority: 'medium',
                todos: todosBySource.code,
                description: 'Automate detection and resolution of code TODOs',
                improvementAreas: ['static_analysis', 'automated_fixes', 'code_quality']
            });
        }
        
        // Git history todos
        if (todosBySource.git && todosBySource.git.length > 0) {
            opportunities.push({
                type: 'git_history_mining',
                category: 'optimization',
                priority: 'low',
                todos: todosBySource.git,
                description: 'Improve git history mining for relevant todos',
                improvementAreas: ['history_analysis', 'relevance_filtering', 'modern_relevance']
            });
        }
        
        return opportunities;
    }

    async generatePromptsForOpportunity(opportunity) {
        const prompts = [];
        
        // Generate base improvement prompt
        const basePrompt = await this.generateBasePrompt(opportunity);
        prompts.push(basePrompt);
        
        // Generate specific prompts for each improvement area
        for (const area of opportunity.improvementAreas) {
            const specificPrompt = await this.generateSpecificPrompt(opportunity, area);
            prompts.push(specificPrompt);
        }
        
        // Add prompts to collection
        this.improvementPrompts.prompts.push(...prompts);
    }

    async generateBasePrompt(opportunity) {
        const prompt = {
            id: `prompt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            type: opportunity.type,
            category: opportunity.category,
            priority: opportunity.priority,
            title: `Improve ${opportunity.type.replace(/_/g, ' ')}`,
            description: opportunity.description,
            context: {
                affectedTodos: opportunity.todos.length,
                todoIds: opportunity.todos.map(t => t.id),
                improvementAreas: opportunity.improvementAreas
            },
            prompt: this.generatePromptText(opportunity),
            expectedOutcomes: this.generateExpectedOutcomes(opportunity),
            successMetrics: this.generateSuccessMetrics(opportunity),
            implementation: {
                estimatedEffort: this.estimateEffort(opportunity),
                requiredSkills: this.identifyRequiredSkills(opportunity),
                dependencies: this.identifyDependencies(opportunity)
            },
            createdAt: new Date().toISOString()
        };
        
        return prompt;
    }

    async generateSpecificPrompt(opportunity, improvementArea) {
        const areaPrompts = {
            linting: 'Implement automated linting rules to catch common issues early',
            testing: 'Create comprehensive automated test suites',
            code_review: 'Establish automated code review processes',
            documentation: 'Generate and maintain up-to-date documentation',
            templates: 'Create reusable templates for common patterns',
            scaffolding: 'Build scaffolding tools for rapid development',
            automation: 'Automate repetitive manual processes',
            ci_cd: 'Implement continuous integration and deployment',
            monitoring: 'Add monitoring and alerting capabilities',
            refactoring: 'Systematically refactor legacy code',
            modernization: 'Modernize outdated technologies and practices',
            cleanup: 'Remove technical debt and unused code'
        };
        
        const promptText = areaPrompts[improvementArea] || `Improve ${improvementArea.replace(/_/g, ' ')}`;
        
        const prompt = {
            id: `prompt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            type: `${opportunity.type}_${improvementArea}`,
            category: opportunity.category,
            priority: this.adjustPriorityForArea(opportunity.priority, improvementArea),
            title: `${promptText} for ${opportunity.type.replace(/_/g, ' ')}`,
            description: `Specific improvement focus: ${promptText}`,
            context: {
                parentOpportunity: opportunity.type,
                focusArea: improvementArea,
                affectedTodos: opportunity.todos.length
            },
            prompt: `${promptText}.\n\nContext: ${opportunity.description}\n\nFocus on implementing ${improvementArea.replace(/_/g, ' ')} improvements that will benefit the ${opportunity.todos.length} related todos.`,
            expectedOutcomes: this.generateAreaSpecificOutcomes(improvementArea),
            successMetrics: this.generateAreaSpecificMetrics(improvementArea),
            implementation: {
                estimatedEffort: this.estimateAreaEffort(improvementArea),
                requiredSkills: this.identifyAreaSkills(improvementArea),
                dependencies: []
            },
            createdAt: new Date().toISOString()
        };
        
        return prompt;
    }

    generatePromptText(opportunity) {
        const baseText = `Analyze and improve the ${opportunity.type.replace(/_/g, ' ')} process.`;
        const contextText = `\n\nContext: ${opportunity.description}`;
        const todosText = `\n\nAffected todos: ${opportunity.todos.length} items`;
        const areasText = `\n\nImprovement areas: ${opportunity.improvementAreas.join(', ')}`;
        
        const actionText = `\n\nActions needed:
1. Analyze current state and identify specific improvement opportunities
2. Design solutions for each improvement area
3. Implement changes with proper testing and validation
4. Measure impact and iterate on improvements
5. Document lessons learned and best practices`;
        
        return baseText + contextText + todosText + areasText + actionText;
    }

    generateExpectedOutcomes(opportunity) {
        const baseOutcomes = [
            'Improved efficiency in handling similar todos',
            'Reduced manual effort and human error',
            'Better quality and consistency',
            'Enhanced maintainability'
        ];
        
        // Add opportunity-specific outcomes
        switch (opportunity.category) {
            case 'automation':
                baseOutcomes.push('Automated processes reduce manual intervention');
                break;
            case 'quality':
                baseOutcomes.push('Higher quality standards and fewer defects');
                break;
            case 'optimization':
                baseOutcomes.push('Better resource utilization and performance');
                break;
            case 'efficiency':
                baseOutcomes.push('Faster completion times and throughput');
                break;
        }
        
        return baseOutcomes;
    }

    generateSuccessMetrics(opportunity) {
        const baseMetrics = [
            'Reduction in todo completion time',
            'Decrease in failure rate',
            'Improvement in quality scores',
            'Increase in automation percentage'
        ];
        
        // Add opportunity-specific metrics
        if (opportunity.todos.length > 0) {
            baseMetrics.push(`Target: Improve ${opportunity.todos.length} related todos`);
        }
        
        return baseMetrics;
    }

    estimateEffort(opportunity) {
        const baseEffort = opportunity.todos.length * 0.5; // Base hours per todo
        const categoryMultipliers = {
            automation: 2.0,
            quality: 1.5,
            optimization: 1.8,
            efficiency: 1.2
        };
        
        const multiplier = categoryMultipliers[opportunity.category] || 1.0;
        const estimatedHours = Math.round(baseEffort * multiplier);
        
        return `${estimatedHours} hours`;
    }

    identifyRequiredSkills(opportunity) {
        const skillsMap = {
            code_quality_automation: ['automation', 'linting', 'ci_cd'],
            test_automation: ['testing', 'automation', 'ci_cd'],
            technical_debt_reduction: ['refactoring', 'architecture', 'code_review'],
            documentation_automation: ['documentation', 'automation', 'templates'],
            deployment_automation: ['devops', 'ci_cd', 'infrastructure'],
            process_standardization: ['process_design', 'templates', 'best_practices']
        };
        
        return skillsMap[opportunity.type] || ['general_development', 'problem_solving'];
    }

    identifyDependencies(opportunity) {
        const dependenciesMap = {
            code_quality_automation: ['linting_tools', 'ci_pipeline'],
            test_automation: ['testing_framework', 'ci_pipeline'],
            deployment_automation: ['infrastructure', 'ci_cd_platform'],
            documentation_automation: ['documentation_tools', 'templates']
        };
        
        return dependenciesMap[opportunity.type] || [];
    }

    adjustPriorityForArea(basePriority, area) {
        const priorityAdjustments = {
            automation: 1, // Increase priority for automation
            ci_cd: 1,
            monitoring: 0, // Keep same priority
            testing: 1,
            documentation: -1, // Decrease priority for documentation
            cleanup: -1
        };
        
        const adjustment = priorityAdjustments[area] || 0;
        const priorities = ['low', 'medium', 'high', 'critical'];
        const currentIndex = priorities.indexOf(basePriority);
        const newIndex = Math.max(0, Math.min(priorities.length - 1, currentIndex + adjustment));
        
        return priorities[newIndex];
    }

    generateAreaSpecificOutcomes(area) {
        const outcomeMap = {
            automation: ['Reduced manual work', 'Consistent execution', 'Faster processing'],
            testing: ['Better test coverage', 'Fewer bugs', 'Confident deployments'],
            ci_cd: ['Automated builds', 'Reliable deployments', 'Fast feedback'],
            monitoring: ['Better visibility', 'Proactive issue detection', 'Improved reliability'],
            refactoring: ['Cleaner code', 'Better maintainability', 'Reduced complexity'],
            documentation: ['Better understanding', 'Easier onboarding', 'Knowledge retention']
        };
        
        return outcomeMap[area] || ['General improvements in ' + area.replace(/_/g, ' ')];
    }

    generateAreaSpecificMetrics(area) {
        const metricsMap = {
            automation: ['Automation percentage', 'Manual effort reduction', 'Error rate'],
            testing: ['Test coverage', 'Bug detection rate', 'Test execution time'],
            ci_cd: ['Build success rate', 'Deployment frequency', 'Lead time'],
            monitoring: ['Detection time', 'False positive rate', 'Coverage'],
            refactoring: ['Code complexity', 'Maintainability index', 'Technical debt'],
            documentation: ['Documentation coverage', 'Freshness', 'Usage metrics']
        };
        
        return metricsMap[area] || ['Improvement metrics for ' + area.replace(/_/g, ' ')];
    }

    estimateAreaEffort(area) {
        const effortMap = {
            automation: '8-16 hours',
            testing: '4-8 hours',
            ci_cd: '12-20 hours',
            monitoring: '6-12 hours',
            refactoring: '4-8 hours per component',
            documentation: '2-4 hours'
        };
        
        return effortMap[area] || '4-8 hours';
    }

    identifyAreaSkills(area) {
        const skillsMap = {
            automation: ['scripting', 'automation_tools', 'workflow_design'],
            testing: ['test_design', 'testing_frameworks', 'automation'],
            ci_cd: ['devops', 'pipeline_design', 'automation'],
            monitoring: ['observability', 'alerting', 'metrics'],
            refactoring: ['code_analysis', 'design_patterns', 'architecture'],
            documentation: ['technical_writing', 'documentation_tools', 'templates']
        };
        
        return skillsMap[area] || ['general_development'];
    }

    async generateRecursivePrompts() {
        console.log('🔄 Generating recursive improvement prompts...');
        
        const recursivePrompts = [];
        let currentDepth = 0;
        let currentPrompts = [...this.improvementPrompts.prompts];
        
        while (currentDepth < this.options.maxPromptDepth && currentPrompts.length > 0) {
            const nextLevelPrompts = [];
            
            for (const prompt of currentPrompts) {
                const recursivePrompt = await this.generateRecursivePrompt(prompt, currentDepth + 1);
                if (recursivePrompt) {
                    nextLevelPrompts.push(recursivePrompt);
                }
            }
            
            recursivePrompts.push(...nextLevelPrompts);
            currentPrompts = nextLevelPrompts;
            currentDepth++;
        }
        
        this.improvementPrompts.recursivePrompts = recursivePrompts;
        this.improvementPrompts.statistics.recursiveDepth = currentDepth;
        
        console.log(`🔄 Generated ${recursivePrompts.length} recursive prompts at depth ${currentDepth}`);
    }

    async generateRecursivePrompt(parentPrompt, depth) {
        // Only generate recursive prompts for high-impact areas
        if (parentPrompt.priority === 'low' && depth > 1) {
            return null;
        }
        
        const recursivePrompt = {
            id: `recursive_${depth}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            type: `recursive_${parentPrompt.type}`,
            category: parentPrompt.category,
            priority: this.adjustPriorityForDepth(parentPrompt.priority, depth),
            title: `Level ${depth}: Enhance ${parentPrompt.title}`,
            description: `Recursive improvement of: ${parentPrompt.description}`,
            context: {
                parentPromptId: parentPrompt.id,
                recursiveDepth: depth,
                baseImprovement: parentPrompt.type
            },
            prompt: this.generateRecursivePromptText(parentPrompt, depth),
            expectedOutcomes: this.generateRecursiveOutcomes(parentPrompt, depth),
            successMetrics: this.generateRecursiveMetrics(parentPrompt, depth),
            implementation: {
                estimatedEffort: this.estimateRecursiveEffort(parentPrompt, depth),
                requiredSkills: [...parentPrompt.implementation.requiredSkills, 'advanced_optimization'],
                dependencies: [parentPrompt.id]
            },
            createdAt: new Date().toISOString()
        };
        
        return recursivePrompt;
    }

    generateRecursivePromptText(parentPrompt, depth) {
        const depthActions = {
            1: 'Optimize and enhance the initial improvement',
            2: 'Create meta-improvements and self-improving systems',
            3: 'Implement autonomous optimization and learning'
        };
        
        const action = depthActions[depth] || 'Further optimize the improvement';
        
        return `${action} from the previous level.

Previous improvement: ${parentPrompt.title}
Previous context: ${parentPrompt.description}

Recursive focus for level ${depth}:
- Analyze the effectiveness of the previous improvement
- Identify meta-patterns and optimization opportunities
- Implement self-improving mechanisms
- Create feedback loops for continuous enhancement
- Design autonomous optimization systems

This recursive improvement should build upon and enhance the previous level's work.`;
    }

    generateRecursiveOutcomes(parentPrompt, depth) {
        const baseOutcomes = [...parentPrompt.expectedOutcomes];
        
        const recursiveOutcomes = [
            `Enhanced effectiveness of ${parentPrompt.title}`,
            'Meta-optimization capabilities',
            'Self-improving systems',
            'Autonomous optimization'
        ];
        
        return [...baseOutcomes, ...recursiveOutcomes.slice(0, depth + 1)];
    }

    generateRecursiveMetrics(parentPrompt, depth) {
        const baseMetrics = [...parentPrompt.successMetrics];
        
        const recursiveMetrics = [
            'Improvement effectiveness increase',
            'Meta-optimization success rate',
            'Self-improvement frequency',
            'Autonomous optimization coverage'
        ];
        
        return [...baseMetrics, ...recursiveMetrics.slice(0, depth + 1)];
    }

    adjustPriorityForDepth(basePriority, depth) {
        // Decrease priority as depth increases
        const priorities = ['critical', 'high', 'medium', 'low'];
        const currentIndex = priorities.indexOf(basePriority);
        const newIndex = Math.min(priorities.length - 1, currentIndex + depth - 1);
        
        return priorities[newIndex];
    }

    estimateRecursiveEffort(parentPrompt, depth) {
        const baseEffortHours = parseInt(parentPrompt.implementation.estimatedEffort.split(' ')[0]) || 8;
        const recursiveMultiplier = 0.5 * depth; // Each level adds 50% of base effort
        const totalHours = Math.round(baseEffortHours * (1 + recursiveMultiplier));
        
        return `${totalHours} hours`;
    }

    organizePromptsByCategory() {
        this.improvementPrompts.promptsByCategory = {};
        
        for (const category of this.options.promptCategories) {
            this.improvementPrompts.promptsByCategory[category] = [];
        }
        
        // Organize base prompts
        for (const prompt of this.improvementPrompts.prompts) {
            const category = prompt.category;
            if (!this.improvementPrompts.promptsByCategory[category]) {
                this.improvementPrompts.promptsByCategory[category] = [];
            }
            this.improvementPrompts.promptsByCategory[category].push(prompt);
        }
        
        // Organize recursive prompts
        for (const prompt of this.improvementPrompts.recursivePrompts) {
            const category = prompt.category;
            if (!this.improvementPrompts.promptsByCategory[category]) {
                this.improvementPrompts.promptsByCategory[category] = [];
            }
            this.improvementPrompts.promptsByCategory[category].push(prompt);
        }
    }

    generateStatistics() {
        this.improvementPrompts.statistics.totalPrompts = 
            this.improvementPrompts.prompts.length + this.improvementPrompts.recursivePrompts.length;
        
        // Count by category
        for (const [category, prompts] of Object.entries(this.improvementPrompts.promptsByCategory)) {
            this.improvementPrompts.statistics.categoryCounts[category] = prompts.length;
        }
        
        // Analyze todo sources
        const sourceAnalysis = {};
        for (const prompt of this.improvementPrompts.prompts) {
            if (prompt.context.todoIds) {
                for (const todoId of prompt.context.todoIds) {
                    const source = this.extractSourceFromTodoId(todoId);
                    sourceAnalysis[source] = (sourceAnalysis[source] || 0) + 1;
                }
            }
        }
        this.improvementPrompts.statistics.todoSourceAnalysis = sourceAnalysis;
    }

    extractSourceFromTodoId(todoId) {
        if (todoId.startsWith('tm-')) return 'taskmaster';
        if (todoId.startsWith('code-')) return 'code';
        if (todoId.startsWith('git-')) return 'git';
        if (todoId.includes('-sub-')) return 'subtask';
        return 'unknown';
    }

    findRepetitivePatterns(titles) {
        const patterns = [];
        const words = new Map();
        
        // Count word occurrences
        for (const title of titles) {
            const titleWords = title.toLowerCase().split(/\s+/);
            for (const word of titleWords) {
                if (word.length > 3) { // Ignore short words
                    words.set(word, (words.get(word) || 0) + 1);
                }
            }
        }
        
        // Find patterns (words appearing in multiple titles)
        for (const [word, count] of words) {
            if (count >= Math.max(2, titles.length * 0.3)) {
                patterns.push(word);
            }
        }
        
        return patterns;
    }

    async saveImprovementPrompts() {
        console.log(`💾 Saving improvement prompts to ${this.options.outputPath}...`);
        
        // Ensure output directory exists
        const outputDir = path.dirname(this.options.outputPath);
        await fs.mkdir(outputDir, { recursive: true });
        
        await fs.writeFile(this.options.outputPath, JSON.stringify(this.improvementPrompts, null, 2));
        
        // Save summary for quick access
        const summary = {
            totalPrompts: this.improvementPrompts.statistics.totalPrompts,
            categoryCounts: this.improvementPrompts.statistics.categoryCounts,
            recursiveDepth: this.improvementPrompts.statistics.recursiveDepth,
            topCategories: Object.entries(this.improvementPrompts.statistics.categoryCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .map(([category, count]) => ({ category, count })),
            timestamp: new Date().toISOString()
        };
        
        await fs.writeFile(
            path.join(outputDir, 'improvement-prompts-summary.json'),
            JSON.stringify(summary, null, 2)
        );
        
        console.log(`✅ Improvement prompts saved: ${this.improvementPrompts.statistics.totalPrompts} total prompts`);
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
            case '--max-depth':
                options.maxDepth = value;
                break;
            case '--categories':
                options.categories = value.split(',');
                break;
            case '--recursive':
                options.recursive = true;
                i--; // No value for this flag
                break;
            case '--no-recursive':
                options.recursive = false;
                i--; // No value for this flag
                break;
        }
    }
    
    try {
        const generator = new ImprovementPromptGenerator(options);
        await generator.generateImprovementPrompts();
        console.log('🎉 Improvement prompt generation completed successfully!');
        process.exit(0);
    } catch (error) {
        console.error('💥 Improvement prompt generation failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { ImprovementPromptGenerator };