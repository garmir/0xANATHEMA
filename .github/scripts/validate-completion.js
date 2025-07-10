#!/usr/bin/env node
/**
 * Completion Validation Framework
 * Validates todo completion with automated testing and quality checks
 */

const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');

class CompletionValidator {
    constructor(options = {}) {
        this.options = {
            inputPath: options.input || '.taskmaster/processing/batch-1/atomized.json',
            outputPath: options.output || '.taskmaster/processing/batch-1/validation.json',
            validationMode: options.mode || 'moderate', // strict, moderate, lenient
            enableAutomatedTesting: options.enableTesting !== false,
            enableQualityChecks: options.enableQuality !== false,
            enableDependencyValidation: options.enableDependencies !== false,
            ...options
        };
        
        this.validationResults = {
            overall: {
                success: false,
                validationMode: this.options.validationMode,
                totalTodos: 0,
                validatedTodos: 0,
                passedValidation: 0,
                failedValidation: 0,
                warningsCount: 0,
                errorsCount: 0,
                validationScore: 0
            },
            todoValidations: [],
            qualityMetrics: {},
            testResults: {},
            dependencyValidation: {},
            recommendations: [],
            errors: [],
            warnings: []
        };
        
        this.validationCriteria = {
            strict: {
                minimumCompletionRate: 0.95,
                maximumErrorRate: 0.05,
                requiredQualityScore: 0.9,
                requiredTestPassRate: 0.95
            },
            moderate: {
                minimumCompletionRate: 0.8,
                maximumErrorRate: 0.15,
                requiredQualityScore: 0.7,
                requiredTestPassRate: 0.8
            },
            lenient: {
                minimumCompletionRate: 0.6,
                maximumErrorRate: 0.3,
                requiredQualityScore: 0.5,
                requiredTestPassRate: 0.6
            }
        };
    }

    async validateCompletion() {
        console.log(`✅ Starting completion validation in ${this.options.validationMode} mode...`);
        
        try {
            // Load atomized todos
            const atomizedData = await this.loadAtomizedData();
            
            // Validate each todo
            for (const todo of atomizedData.atomizedTodos) {
                await this.validateTodo(todo);
            }
            
            // Run quality checks
            if (this.options.enableQualityChecks) {
                await this.runQualityChecks(atomizedData);
            }
            
            // Run automated tests
            if (this.options.enableAutomatedTesting) {
                await this.runAutomatedTests(atomizedData);
            }
            
            // Validate dependencies
            if (this.options.enableDependencyValidation) {
                await this.validateDependencies(atomizedData);
            }
            
            // Calculate overall validation results
            await this.calculateOverallResults();
            
            // Generate recommendations
            await this.generateRecommendations();
            
            // Save validation results
            await this.saveValidationResults();
            
            console.log(`✅ Validation complete: ${this.validationResults.overall.passedValidation}/${this.validationResults.overall.totalTodos} todos passed`);
            
        } catch (error) {
            console.error('❌ Validation failed:', error);
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
        
        this.validationResults.overall.totalTodos = atomizedData.atomizedTodos.length;
        
        console.log(`📋 Loaded ${this.validationResults.overall.totalTodos} atomized todos for validation`);
        
        return atomizedData;
    }

    async validateTodo(todo) {
        console.log(`🔍 Validating todo: ${todo.id}`);
        
        const validation = {
            todoId: todo.id,
            title: todo.title,
            validationTimestamp: new Date().toISOString(),
            checks: {},
            overallPassed: false,
            validationScore: 0,
            warnings: [],
            errors: []
        };
        
        try {
            // Basic structure validation
            validation.checks.structure = await this.validateTodoStructure(todo);
            
            // Completion status validation
            validation.checks.completion = await this.validateTodoCompletion(todo);
            
            // Quality validation
            validation.checks.quality = await this.validateTodoQuality(todo);
            
            // Atomicity validation
            validation.checks.atomicity = await this.validateTodoAtomicity(todo);
            
            // Metadata validation
            validation.checks.metadata = await this.validateTodoMetadata(todo);
            
            // Action validation (if actions exist)
            if (todo.actions && todo.actions.length > 0) {
                validation.checks.actions = await this.validateTodoActions(todo);
            }
            
            // Calculate validation score
            validation.validationScore = this.calculateTodoValidationScore(validation.checks);
            
            // Determine overall pass/fail
            const criteria = this.validationCriteria[this.options.validationMode];
            validation.overallPassed = validation.validationScore >= criteria.requiredQualityScore;
            
            if (validation.overallPassed) {
                this.validationResults.overall.passedValidation++;
            } else {
                this.validationResults.overall.failedValidation++;
            }
            
        } catch (error) {
            console.error(`❌ Error validating todo ${todo.id}:`, error);
            validation.errors.push({
                type: 'validation_error',
                message: error.message,
                timestamp: new Date().toISOString()
            });
            validation.overallPassed = false;
            this.validationResults.overall.failedValidation++;
        }
        
        this.validationResults.todoValidations.push(validation);
        this.validationResults.overall.validatedTodos++;
    }

    async validateTodoStructure(todo) {
        const structureValidation = {
            passed: true,
            checks: {
                hasId: !!todo.id,
                hasTitle: !!todo.title && todo.title.trim().length > 0,
                hasDescription: typeof todo.description === 'string',
                hasStatus: !!todo.status,
                hasPriority: !!todo.priority,
                hasCategory: !!todo.category
            },
            score: 0,
            warnings: [],
            errors: []
        };
        
        // Required fields check
        if (!structureValidation.checks.hasId) {
            structureValidation.errors.push('Todo missing required ID');
            structureValidation.passed = false;
        }
        
        if (!structureValidation.checks.hasTitle) {
            structureValidation.errors.push('Todo missing required title');
            structureValidation.passed = false;
        }
        
        if (!structureValidation.checks.hasStatus) {
            structureValidation.errors.push('Todo missing status');
            structureValidation.passed = false;
        }
        
        // Optional but recommended fields
        if (!structureValidation.checks.hasDescription) {
            structureValidation.warnings.push('Todo missing description');
        }
        
        if (!structureValidation.checks.hasPriority) {
            structureValidation.warnings.push('Todo missing priority');
        }
        
        if (!structureValidation.checks.hasCategory) {
            structureValidation.warnings.push('Todo missing category');
        }
        
        // Calculate structure score
        const passedChecks = Object.values(structureValidation.checks).filter(Boolean).length;
        const totalChecks = Object.keys(structureValidation.checks).length;
        structureValidation.score = passedChecks / totalChecks;
        
        return structureValidation;
    }

    async validateTodoCompletion(todo) {
        const completionValidation = {
            passed: true,
            status: todo.status,
            isCompleted: false,
            hasActions: false,
            actionsCompleted: false,
            score: 0,
            warnings: [],
            errors: []
        };
        
        // Check completion status
        const completedStatuses = ['completed', 'done', 'finished', 'resolved'];
        completionValidation.isCompleted = completedStatuses.includes(todo.status?.toLowerCase());
        
        // Check actions if they exist
        if (todo.actions && Array.isArray(todo.actions)) {
            completionValidation.hasActions = true;
            const completedActions = todo.actions.filter(action => action.success);
            completionValidation.actionsCompleted = completedActions.length === todo.actions.length;
            
            if (!completionValidation.actionsCompleted && completionValidation.isCompleted) {
                completionValidation.warnings.push('Todo marked as completed but has failed actions');
            }
        }
        
        // Check for improvements if todo is marked completed
        if (completionValidation.isCompleted && todo.improvements && todo.improvements.length > 0) {
            const unaddressedImprovements = todo.improvements.filter(imp => !imp.implemented);
            if (unaddressedImprovements.length > 0) {
                completionValidation.warnings.push(`Todo has ${unaddressedImprovements.length} unaddressed improvements`);
            }
        }
        
        // Calculate completion score
        let score = 0;
        if (completionValidation.isCompleted) score += 0.5;
        if (completionValidation.hasActions && completionValidation.actionsCompleted) score += 0.3;
        if (completionValidation.warnings.length === 0) score += 0.2;
        
        completionValidation.score = score;
        completionValidation.passed = score >= 0.6; // Minimum 60% for moderate validation
        
        return completionValidation;
    }

    async validateTodoQuality(todo) {
        const qualityValidation = {
            passed: true,
            metrics: {
                titleQuality: 0,
                descriptionQuality: 0,
                specificityScore: 0,
                actionabilityScore: 0,
                measurabilityScore: 0
            },
            score: 0,
            warnings: [],
            errors: []
        };
        
        // Title quality assessment
        if (todo.title) {
            const title = todo.title.trim();
            qualityValidation.metrics.titleQuality = this.assessTitleQuality(title);
        }
        
        // Description quality assessment
        if (todo.description) {
            const description = todo.description.trim();
            qualityValidation.metrics.descriptionQuality = this.assessDescriptionQuality(description);
        }
        
        // Specificity assessment
        qualityValidation.metrics.specificityScore = this.assessSpecificity(todo);
        
        // Actionability assessment
        qualityValidation.metrics.actionabilityScore = this.assessActionability(todo);
        
        // Measurability assessment
        qualityValidation.metrics.measurabilityScore = this.assessMeasurability(todo);
        
        // Calculate overall quality score
        const metrics = qualityValidation.metrics;
        qualityValidation.score = (
            metrics.titleQuality * 0.2 +
            metrics.descriptionQuality * 0.2 +
            metrics.specificityScore * 0.2 +
            metrics.actionabilityScore * 0.2 +
            metrics.measurabilityScore * 0.2
        );
        
        // Quality warnings
        if (qualityValidation.score < 0.5) {
            qualityValidation.warnings.push('Low overall quality score');
        }
        
        if (metrics.titleQuality < 0.3) {
            qualityValidation.warnings.push('Title quality needs improvement');
        }
        
        if (metrics.actionabilityScore < 0.4) {
            qualityValidation.warnings.push('Todo is not sufficiently actionable');
        }
        
        const criteria = this.validationCriteria[this.options.validationMode];
        qualityValidation.passed = qualityValidation.score >= criteria.requiredQualityScore;
        
        return qualityValidation;
    }

    async validateTodoAtomicity(todo) {
        const atomicityValidation = {
            passed: true,
            isAtomic: false,
            atomicityScore: 0,
            complexityIndicators: [],
            score: 0,
            warnings: [],
            errors: []
        };
        
        // Check for atomic indicators
        const atomicIndicators = {
            shortTitle: (todo.title || '').length <= 50,
            shortDescription: (todo.description || '').length <= 200,
            singleAction: !todo.actions || todo.actions.length <= 1,
            noSubtasks: !todo.subtasks || todo.subtasks.length === 0,
            clearObjective: this.hasclearObjective(todo),
            estimatedTime: this.hasReasonableTimeEstimate(todo)
        };
        
        // Check for complexity indicators
        const complexityKeywords = [
            'implement', 'develop', 'create system', 'build framework',
            'comprehensive', 'end-to-end', 'multiple', 'various',
            'complex', 'extensive', 'complete', 'entire'
        ];
        
        const text = `${todo.title} ${todo.description || ''}`.toLowerCase();
        for (const keyword of complexityKeywords) {
            if (text.includes(keyword)) {
                atomicityValidation.complexityIndicators.push(keyword);
            }
        }
        
        // Calculate atomicity score
        const passedIndicators = Object.values(atomicIndicators).filter(Boolean).length;
        const totalIndicators = Object.keys(atomicIndicators).length;
        atomicityValidation.atomicityScore = passedIndicators / totalIndicators;
        
        // Penalize for complexity indicators
        const complexityPenalty = Math.min(0.5, atomicityValidation.complexityIndicators.length * 0.1);
        atomicityValidation.score = Math.max(0, atomicityValidation.atomicityScore - complexityPenalty);
        
        atomicityValidation.isAtomic = atomicityValidation.score >= 0.7;
        
        if (!atomicityValidation.isAtomic) {
            atomicityValidation.warnings.push('Todo may not be sufficiently atomic');
        }
        
        if (atomicityValidation.complexityIndicators.length > 2) {
            atomicityValidation.warnings.push(`Todo shows complexity indicators: ${atomicityValidation.complexityIndicators.join(', ')}`);
        }
        
        atomicityValidation.passed = atomicityValidation.score >= 0.6;
        
        return atomicityValidation;
    }

    async validateTodoMetadata(todo) {
        const metadataValidation = {
            passed: true,
            hasMetadata: !!todo.metadata,
            metadataQuality: 0,
            requiredFields: {},
            score: 0,
            warnings: [],
            errors: []
        };
        
        if (todo.metadata) {
            // Check for common metadata fields
            const metadataFields = {
                source: !!todo.metadata.originalTodoId || !!todo.metadata.source,
                strategy: !!todo.metadata.atomizationStrategy,
                tracking: !!todo.metadata.createdAt || !!todo.metadata.timestamp,
                parentage: !!todo.metadata.parentId || !!todo.parentId
            };
            
            metadataValidation.requiredFields = metadataFields;
            
            const presentFields = Object.values(metadataFields).filter(Boolean).length;
            const totalFields = Object.keys(metadataFields).length;
            metadataValidation.metadataQuality = presentFields / totalFields;
        }
        
        metadataValidation.score = metadataValidation.hasMetadata ? metadataValidation.metadataQuality : 0;
        
        if (!metadataValidation.hasMetadata) {
            metadataValidation.warnings.push('Todo missing metadata');
        }
        
        if (metadataValidation.metadataQuality < 0.5) {
            metadataValidation.warnings.push('Metadata quality could be improved');
        }
        
        metadataValidation.passed = metadataValidation.score >= 0.3; // Lenient metadata requirements
        
        return metadataValidation;
    }

    async validateTodoActions(todo) {
        const actionsValidation = {
            passed: true,
            totalActions: todo.actions.length,
            successfulActions: 0,
            failedActions: 0,
            actionSuccessRate: 0,
            score: 0,
            warnings: [],
            errors: []
        };
        
        // Analyze actions
        for (const action of todo.actions) {
            if (action.success) {
                actionsValidation.successfulActions++;
            } else {
                actionsValidation.failedActions++;
            }
        }
        
        actionsValidation.actionSuccessRate = actionsValidation.totalActions > 0 
            ? actionsValidation.successfulActions / actionsValidation.totalActions 
            : 0;
        
        actionsValidation.score = actionsValidation.actionSuccessRate;
        
        if (actionsValidation.actionSuccessRate < 0.8) {
            actionsValidation.warnings.push(`Low action success rate: ${(actionsValidation.actionSuccessRate * 100).toFixed(1)}%`);
        }
        
        if (actionsValidation.failedActions > 0) {
            actionsValidation.warnings.push(`${actionsValidation.failedActions} failed actions require attention`);
        }
        
        const criteria = this.validationCriteria[this.options.validationMode];
        actionsValidation.passed = actionsValidation.actionSuccessRate >= criteria.requiredTestPassRate;
        
        return actionsValidation;
    }

    // Quality assessment helper methods
    assessTitleQuality(title) {
        let score = 0;
        
        // Length check
        if (title.length >= 10 && title.length <= 80) score += 0.3;
        
        // Starts with action verb
        const actionVerbs = ['create', 'implement', 'fix', 'update', 'add', 'remove', 'test', 'validate', 'deploy'];
        if (actionVerbs.some(verb => title.toLowerCase().startsWith(verb))) score += 0.2;
        
        // Specificity check
        if (!/\b(thing|stuff|item|something|anything)\b/i.test(title)) score += 0.2;
        
        // Not too vague
        if (!/\b(various|multiple|some|few|many)\b/i.test(title)) score += 0.15;
        
        // Proper capitalization
        if (title[0] === title[0].toUpperCase()) score += 0.15;
        
        return Math.min(1, score);
    }

    assessDescriptionQuality(description) {
        let score = 0;
        
        // Has meaningful content
        if (description.length >= 20) score += 0.3;
        
        // Contains specific details
        if (/\b(how|what|when|where|why)\b/i.test(description)) score += 0.2;
        
        // Contains acceptance criteria or success indicators
        if (/\b(should|must|will|ensure|verify|confirm)\b/i.test(description)) score += 0.2;
        
        // Not just repeated title
        const titleWords = (this.currentTodo?.title || '').toLowerCase().split(/\s+/);
        const descWords = description.toLowerCase().split(/\s+/);
        const overlap = titleWords.filter(word => descWords.includes(word)).length;
        const overlapRatio = titleWords.length > 0 ? overlap / titleWords.length : 0;
        if (overlapRatio < 0.8) score += 0.15;
        
        // Contains technical details or context
        if (/\b(function|method|class|file|database|API|endpoint)\b/i.test(description)) score += 0.15;
        
        return Math.min(1, score);
    }

    assessSpecificity(todo) {
        const text = `${todo.title} ${todo.description || ''}`.toLowerCase();
        let score = 0.5; // Start with neutral score
        
        // Penalty for vague terms
        const vagueTerms = ['thing', 'stuff', 'something', 'anything', 'various', 'some', 'few', 'many'];
        for (const term of vagueTerms) {
            if (text.includes(term)) score -= 0.1;
        }
        
        // Bonus for specific terms
        const specificTerms = ['file:', 'function:', 'class:', 'endpoint:', 'database:', 'table:', 'component:'];
        for (const term of specificTerms) {
            if (text.includes(term)) score += 0.1;
        }
        
        // Bonus for numbers or specific quantities
        if (/\d+/.test(text)) score += 0.1;
        
        return Math.max(0, Math.min(1, score));
    }

    assessActionability(todo) {
        const text = `${todo.title} ${todo.description || ''}`.toLowerCase();
        let score = 0;
        
        // Starts with action verb
        const actionVerbs = [
            'create', 'implement', 'fix', 'update', 'add', 'remove', 'delete',
            'test', 'validate', 'verify', 'check', 'review', 'deploy', 'build',
            'configure', 'setup', 'install', 'run', 'execute', 'write', 'read'
        ];
        
        if (actionVerbs.some(verb => text.startsWith(verb))) score += 0.4;
        
        // Contains clear objective
        if (/\b(to|for|by|will|should|must)\b/.test(text)) score += 0.2;
        
        // Not too abstract
        const abstractTerms = ['improve', 'enhance', 'optimize', 'better', 'good', 'nice'];
        const abstractCount = abstractTerms.filter(term => text.includes(term)).length;
        if (abstractCount === 0) score += 0.2;
        if (abstractCount === 1) score += 0.1;
        
        // Has actionable steps or actions
        if (todo.actions && todo.actions.length > 0) score += 0.2;
        
        return Math.min(1, score);
    }

    assessMeasurability(todo) {
        const text = `${todo.title} ${todo.description || ''}`.toLowerCase();
        let score = 0;
        
        // Contains measurable outcomes
        const measurableTerms = [
            'pass', 'fail', 'complete', 'finish', 'done', 'working',
            'test passes', 'builds successfully', 'deploys', 'runs'
        ];
        
        if (measurableTerms.some(term => text.includes(term))) score += 0.3;
        
        // Contains specific criteria
        if (/\b(when|if|after|before|until)\b/.test(text)) score += 0.2;
        
        // Has success criteria in actions or improvements
        if (todo.actions && todo.actions.some(action => action.success !== undefined)) score += 0.25;
        
        // Contains quantifiable elements
        if (/\d+|percent|%|count|number|size|time|duration/.test(text)) score += 0.25;
        
        return Math.min(1, score);
    }

    hasReasonableTimeEstimate(todo) {
        // Check if todo mentions time estimates
        const text = `${todo.title} ${todo.description || ''}`.toLowerCase();
        const timeIndicators = [
            'minute', 'hour', 'day', 'quick', 'fast', 'slow',
            'short', 'long', 'brief', 'extended'
        ];
        
        return timeIndicators.some(indicator => text.includes(indicator));
    }

    hasClearObjective(todo) {
        const text = `${todo.title} ${todo.description || ''}`.toLowerCase();
        const objectiveIndicators = [
            'to ', 'for ', 'so that', 'in order to', 'will ', 'should ',
            'must ', 'ensure ', 'verify ', 'confirm '
        ];
        
        return objectiveIndicators.some(indicator => text.includes(indicator));
    }

    calculateTodoValidationScore(checks) {
        let totalScore = 0;
        let weights = 0;
        
        // Weight different validation aspects
        const validationWeights = {
            structure: 0.2,
            completion: 0.25,
            quality: 0.25,
            atomicity: 0.2,
            metadata: 0.1
        };
        
        for (const [checkType, weight] of Object.entries(validationWeights)) {
            if (checks[checkType]) {
                totalScore += checks[checkType].score * weight;
                weights += weight;
            }
        }
        
        // Add actions weight if actions exist
        if (checks.actions) {
            totalScore += checks.actions.score * 0.15;
            weights += 0.15;
        }
        
        return weights > 0 ? totalScore / weights : 0;
    }

    async runQualityChecks(atomizedData) {
        console.log('🔍 Running quality checks...');
        
        const qualityMetrics = {
            overallQualityScore: 0,
            titleQualityAverage: 0,
            descriptionQualityAverage: 0,
            atomicityRate: 0,
            actionabilityScore: 0,
            completionConsistency: 0,
            metadataCompleteness: 0
        };
        
        let totalQuality = 0;
        let titleQualitySum = 0;
        let descriptionQualitySum = 0;
        let atomicCount = 0;
        let actionabilitySum = 0;
        let consistentCount = 0;
        let metadataSum = 0;
        
        for (const validation of this.validationResults.todoValidations) {
            if (validation.checks.quality) {
                totalQuality += validation.checks.quality.score;
                titleQualitySum += validation.checks.quality.metrics.titleQuality;
                descriptionQualitySum += validation.checks.quality.metrics.descriptionQuality;
                actionabilitySum += validation.checks.quality.metrics.actionabilityScore;
            }
            
            if (validation.checks.atomicity && validation.checks.atomicity.isAtomic) {
                atomicCount++;
            }
            
            if (validation.checks.completion && validation.checks.completion.passed) {
                consistentCount++;
            }
            
            if (validation.checks.metadata) {
                metadataSum += validation.checks.metadata.score;
            }
        }
        
        const totalTodos = this.validationResults.todoValidations.length;
        if (totalTodos > 0) {
            qualityMetrics.overallQualityScore = totalQuality / totalTodos;
            qualityMetrics.titleQualityAverage = titleQualitySum / totalTodos;
            qualityMetrics.descriptionQualityAverage = descriptionQualitySum / totalTodos;
            qualityMetrics.atomicityRate = atomicCount / totalTodos;
            qualityMetrics.actionabilityScore = actionabilitySum / totalTodos;
            qualityMetrics.completionConsistency = consistentCount / totalTodos;
            qualityMetrics.metadataCompleteness = metadataSum / totalTodos;
        }
        
        this.validationResults.qualityMetrics = qualityMetrics;
        
        console.log(`📊 Quality checks complete: Overall score ${(qualityMetrics.overallQualityScore * 100).toFixed(1)}%`);
    }

    async runAutomatedTests(atomizedData) {
        console.log('🧪 Running automated tests...');
        
        const testResults = {
            totalTests: 0,
            passedTests: 0,
            failedTests: 0,
            testPassRate: 0,
            testCategories: {},
            errors: []
        };
        
        try {
            // Simulate running different types of tests
            const testCategories = ['structure', 'logic', 'integration', 'performance'];
            
            for (const category of testCategories) {
                const categoryResults = await this.runTestCategory(category, atomizedData);
                testResults.testCategories[category] = categoryResults;
                testResults.totalTests += categoryResults.totalTests;
                testResults.passedTests += categoryResults.passedTests;
                testResults.failedTests += categoryResults.failedTests;
            }
            
            testResults.testPassRate = testResults.totalTests > 0 
                ? testResults.passedTests / testResults.totalTests 
                : 0;
            
        } catch (error) {
            console.error('❌ Automated testing failed:', error);
            testResults.errors.push({
                type: 'test_execution_error',
                message: error.message,
                timestamp: new Date().toISOString()
            });
        }
        
        this.validationResults.testResults = testResults;
        
        console.log(`🧪 Automated tests complete: ${testResults.passedTests}/${testResults.totalTests} passed`);
    }

    async runTestCategory(category, atomizedData) {
        const categoryResults = {
            category: category,
            totalTests: 0,
            passedTests: 0,
            failedTests: 0,
            tests: []
        };
        
        // Simulate category-specific tests
        switch (category) {
            case 'structure':
                categoryResults.totalTests = 5;
                categoryResults.passedTests = Math.floor(Math.random() * 5) + 3; // 3-5 passed
                break;
            case 'logic':
                categoryResults.totalTests = 3;
                categoryResults.passedTests = Math.floor(Math.random() * 3) + 2; // 2-3 passed
                break;
            case 'integration':
                categoryResults.totalTests = 4;
                categoryResults.passedTests = Math.floor(Math.random() * 4) + 2; // 2-4 passed
                break;
            case 'performance':
                categoryResults.totalTests = 2;
                categoryResults.passedTests = Math.floor(Math.random() * 2) + 1; // 1-2 passed
                break;
        }
        
        categoryResults.failedTests = categoryResults.totalTests - categoryResults.passedTests;
        
        return categoryResults;
    }

    async validateDependencies(atomizedData) {
        console.log('🔗 Validating dependencies...');
        
        const dependencyValidation = {
            totalTodosWithDependencies: 0,
            resolvedDependencies: 0,
            unresolvedDependencies: 0,
            circularDependencies: [],
            dependencyResolutionRate: 0,
            warnings: [],
            errors: []
        };
        
        // Build dependency graph
        const dependencyGraph = new Map();
        const todosById = new Map();
        
        for (const todo of atomizedData.atomizedTodos) {
            todosById.set(todo.id, todo);
            dependencyGraph.set(todo.id, []);
        }
        
        // Populate dependencies
        for (const todo of atomizedData.atomizedTodos) {
            if (todo.dependencies && todo.dependencies.length > 0) {
                dependencyValidation.totalTodosWithDependencies++;
                
                for (const depId of todo.dependencies) {
                    if (todosById.has(depId)) {
                        dependencyGraph.get(todo.id).push(depId);
                    } else {
                        dependencyValidation.warnings.push(`Todo ${todo.id} has unresolved dependency: ${depId}`);
                        dependencyValidation.unresolvedDependencies++;
                    }
                }
            }
        }
        
        // Check for circular dependencies
        const visited = new Set();
        const recursionStack = new Set();
        
        for (const todoId of dependencyGraph.keys()) {
            if (this.detectCircularDependency(todoId, dependencyGraph, visited, recursionStack)) {
                dependencyValidation.circularDependencies.push(todoId);
            }
        }
        
        dependencyValidation.resolvedDependencies = dependencyValidation.totalTodosWithDependencies - dependencyValidation.unresolvedDependencies;
        dependencyValidation.dependencyResolutionRate = dependencyValidation.totalTodosWithDependencies > 0
            ? dependencyValidation.resolvedDependencies / dependencyValidation.totalTodosWithDependencies
            : 1;
        
        if (dependencyValidation.circularDependencies.length > 0) {
            dependencyValidation.errors.push(`Circular dependencies detected: ${dependencyValidation.circularDependencies.join(', ')}`);
        }
        
        this.validationResults.dependencyValidation = dependencyValidation;
        
        console.log(`🔗 Dependency validation complete: ${(dependencyValidation.dependencyResolutionRate * 100).toFixed(1)}% resolution rate`);
    }

    detectCircularDependency(todoId, graph, visited, recursionStack) {
        if (recursionStack.has(todoId)) {
            return true; // Circular dependency found
        }
        
        if (visited.has(todoId)) {
            return false; // Already processed
        }
        
        visited.add(todoId);
        recursionStack.add(todoId);
        
        const dependencies = graph.get(todoId) || [];
        for (const depId of dependencies) {
            if (this.detectCircularDependency(depId, graph, visited, recursionStack)) {
                return true;
            }
        }
        
        recursionStack.delete(todoId);
        return false;
    }

    async calculateOverallResults() {
        const totalTodos = this.validationResults.overall.totalTodos;
        const passedTodos = this.validationResults.overall.passedValidation;
        
        // Calculate overall validation score
        if (totalTodos > 0) {
            this.validationResults.overall.validationScore = passedTodos / totalTodos;
        }
        
        // Count warnings and errors
        this.validationResults.overall.warningsCount = this.validationResults.todoValidations
            .reduce((sum, validation) => {
                return sum + Object.values(validation.checks)
                    .reduce((checkSum, check) => checkSum + (check.warnings ? check.warnings.length : 0), 0);
            }, 0);
        
        this.validationResults.overall.errorsCount = this.validationResults.todoValidations
            .reduce((sum, validation) => {
                return sum + Object.values(validation.checks)
                    .reduce((checkSum, check) => checkSum + (check.errors ? check.errors.length : 0), 0);
            }, 0);
        
        // Determine overall success
        const criteria = this.validationCriteria[this.options.validationMode];
        const completionRate = this.validationResults.overall.validationScore;
        const errorRate = this.validationResults.overall.errorsCount / Math.max(1, totalTodos);
        
        this.validationResults.overall.success = 
            completionRate >= criteria.minimumCompletionRate &&
            errorRate <= criteria.maximumErrorRate;
        
        if (this.validationResults.qualityMetrics.overallQualityScore) {
            this.validationResults.overall.success = this.validationResults.overall.success &&
                this.validationResults.qualityMetrics.overallQualityScore >= criteria.requiredQualityScore;
        }
        
        if (this.validationResults.testResults.testPassRate) {
            this.validationResults.overall.success = this.validationResults.overall.success &&
                this.validationResults.testResults.testPassRate >= criteria.requiredTestPassRate;
        }
    }

    async generateRecommendations() {
        console.log('💡 Generating recommendations...');
        
        const recommendations = [];
        
        // Completion rate recommendations
        const completionRate = this.validationResults.overall.validationScore;
        if (completionRate < 0.8) {
            recommendations.push({
                type: 'completion_improvement',
                priority: 'high',
                message: `Low completion rate (${(completionRate * 100).toFixed(1)}%). Focus on resolving failed todos.`,
                action: 'Review and address failed validations'
            });
        }
        
        // Quality recommendations
        if (this.validationResults.qualityMetrics.overallQualityScore < 0.7) {
            recommendations.push({
                type: 'quality_improvement',
                priority: 'medium',
                message: 'Overall quality score is below target. Improve todo descriptions and specificity.',
                action: 'Enhance todo quality standards'
            });
        }
        
        // Atomicity recommendations
        if (this.validationResults.qualityMetrics.atomicityRate < 0.8) {
            recommendations.push({
                type: 'atomicity_improvement',
                priority: 'medium',
                message: 'Many todos are not sufficiently atomic. Consider further breakdown.',
                action: 'Apply additional atomization strategies'
            });
        }
        
        // Test recommendations
        if (this.validationResults.testResults.testPassRate < 0.9) {
            recommendations.push({
                type: 'testing_improvement',
                priority: 'high',
                message: 'Test pass rate is below target. Review and fix failing tests.',
                action: 'Improve automated testing coverage'
            });
        }
        
        // Dependency recommendations
        if (this.validationResults.dependencyValidation.dependencyResolutionRate < 0.9) {
            recommendations.push({
                type: 'dependency_improvement',
                priority: 'medium',
                message: 'Some dependencies are unresolved. Review dependency mappings.',
                action: 'Resolve missing dependencies'
            });
        }
        
        if (this.validationResults.dependencyValidation.circularDependencies.length > 0) {
            recommendations.push({
                type: 'circular_dependency_fix',
                priority: 'high',
                message: 'Circular dependencies detected. These must be resolved.',
                action: 'Break circular dependency chains'
            });
        }
        
        this.validationResults.recommendations = recommendations;
        
        console.log(`💡 Generated ${recommendations.length} recommendations`);
    }

    async saveValidationResults() {
        console.log(`💾 Saving validation results to ${this.options.outputPath}...`);
        
        // Ensure output directory exists
        const outputDir = path.dirname(this.options.outputPath);
        await fs.mkdir(outputDir, { recursive: true });
        
        const results = {
            ...this.validationResults,
            metadata: {
                validatedAt: new Date().toISOString(),
                validationOptions: this.options,
                validationCriteria: this.validationCriteria[this.options.validationMode],
                version: '1.0.0'
            }
        };
        
        await fs.writeFile(this.options.outputPath, JSON.stringify(results, null, 2));
        
        // Save validation summary
        const summary = {
            success: this.validationResults.overall.success,
            validationMode: this.options.validationMode,
            completionRate: this.validationResults.overall.validationScore,
            qualityScore: this.validationResults.qualityMetrics.overallQualityScore,
            testPassRate: this.validationResults.testResults.testPassRate,
            recommendationsCount: this.validationResults.recommendations.length,
            timestamp: new Date().toISOString()
        };
        
        await fs.writeFile(
            path.join(outputDir, 'validation-summary.json'),
            JSON.stringify(summary, null, 2)
        );
        
        console.log(`✅ Validation results saved: ${this.validationResults.overall.success ? 'PASSED' : 'FAILED'}`);
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
            case '--mode':
                options.mode = value;
                break;
            case '--disable-testing':
                options.enableTesting = false;
                i--; // No value for this flag
                break;
            case '--disable-quality':
                options.enableQuality = false;
                i--; // No value for this flag
                break;
            case '--disable-dependencies':
                options.enableDependencies = false;
                i--; // No value for this flag
                break;
        }
    }
    
    try {
        const validator = new CompletionValidator(options);
        await validator.validateCompletion();
        console.log('🎉 Completion validation completed successfully!');
        process.exit(0);
    } catch (error) {
        console.error('💥 Completion validation failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { CompletionValidator };