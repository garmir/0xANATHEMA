#!/usr/bin/env node
/**
 * Recursive Enhancement Engine - Discovery Module
 * Discovers enhancement opportunities across the codebase
 */

const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');

class EnhancementDiscovery {
    constructor(options = {}) {
        this.options = {
            enhancementId: options.enhancementId,
            scope: options.scope || 'all',
            baselineMetrics: options.baseline ? JSON.parse(options.baseline) : {},
            enhancementTargets: options.targets ? JSON.parse(options.targets) : [],
            workspaceDir: options.workspace || '.taskmaster/enhancement',
            discoveryDepth: parseInt(options.depth) || 3,
            minImpactThreshold: parseFloat(options.minImpact) || 0.1,
            ...options
        };
        
        this.discoveryResults = {
            enhancementId: this.options.enhancementId,
            metadata: {
                discoveredAt: new Date().toISOString(),
                scope: this.options.scope,
                discoveryConfiguration: this.options
            },
            opportunities: [],
            analysisResults: {},
            processingMatrix: {},
            statistics: {
                totalOpportunities: 0,
                categorizedOpportunities: {},
                priorityDistribution: {},
                complexityAnalysis: {}
            }
        };
    }

    async discoverOpportunities() {
        console.log(`🔍 Starting enhancement opportunity discovery (${this.options.enhancementId})...`);
        
        try {
            // Load baseline metrics and targets
            await this.loadDiscoveryContext();
            
            // Discover opportunities by category
            await this.discoverCodeQualityOpportunities();
            await this.discoverPerformanceOpportunities();
            await this.discoverArchitectureOpportunities();
            await this.discoverDocumentationOpportunities();
            await this.discoverTestingOpportunities();
            await this.discoverSecurityOpportunities();
            await this.discoverMaintainabilityOpportunities();
            
            // Analyze cross-cutting opportunities
            await this.discoverCrossCuttingOpportunities();
            
            // Generate processing matrix for parallel execution
            await this.generateProcessingMatrix();
            
            // Analyze and prioritize opportunities
            await this.analyzeAndPrioritizeOpportunities();
            
            // Save discovery results
            await this.saveDiscoveryResults();
            
            // Output for GitHub Actions
            await this.outputGitHubActions();
            
            console.log(`✅ Discovery complete: ${this.discoveryResults.opportunities.length} opportunities found`);
            
        } catch (error) {
            console.error('❌ Enhancement discovery failed:', error);
            throw error;
        }
    }

    async loadDiscoveryContext() {
        console.log('📚 Loading discovery context...');
        
        // Load enhancement configuration
        const configPath = path.join(this.options.workspaceDir, 'enhancement-config.json');
        try {
            const configData = await fs.readFile(configPath, 'utf8');
            this.enhancementConfig = JSON.parse(configData);
        } catch (error) {
            console.warn('⚠️ Could not load enhancement config, using defaults');
            this.enhancementConfig = {};
        }
        
        // Load baseline metrics if not provided
        if (!this.options.baselineMetrics || Object.keys(this.options.baselineMetrics).length === 0) {
            const metricsPath = path.join(this.options.workspaceDir, 'metrics', 'baseline-metrics.json');
            try {
                const metricsData = await fs.readFile(metricsPath, 'utf8');
                this.options.baselineMetrics = JSON.parse(metricsData);
            } catch (error) {
                console.warn('⚠️ Could not load baseline metrics');
                this.options.baselineMetrics = {};
            }
        }
        
        console.log('📚 Discovery context loaded');
    }

    async discoverCodeQualityOpportunities() {
        if (this.options.scope !== 'all' && this.options.scope !== 'code_quality') return;
        
        console.log('🔍 Discovering code quality opportunities...');
        
        const opportunities = [];
        
        // Complexity reduction opportunities
        const complexityOpportunities = await this.findComplexityReductionOpportunities();
        opportunities.push(...complexityOpportunities);
        
        // Code duplication opportunities
        const duplicationOpportunities = await this.findCodeDuplicationOpportunities();
        opportunities.push(...duplicationOpportunities);
        
        // Code smell elimination opportunities
        const codeSmellOpportunities = await this.findCodeSmellOpportunities();
        opportunities.push(...codeSmellOpportunities);
        
        // Refactoring opportunities
        const refactoringOpportunities = await this.findRefactoringOpportunities();
        opportunities.push(...refactoringOpportunities);
        
        this.addOpportunities(opportunities, 'code_quality');
        console.log(`🔍 Found ${opportunities.length} code quality opportunities`);
    }

    async findComplexityReductionOpportunities() {
        const opportunities = [];
        
        // Analyze JavaScript/TypeScript files for complexity
        const jsFiles = await this.findFiles(['**/*.js', '**/*.ts', '**/*.jsx', '**/*.tsx']);
        
        for (const file of jsFiles.slice(0, 50)) { // Limit for demo
            try {
                const content = await fs.readFile(file, 'utf8');
                const complexity = this.calculateCyclomaticComplexity(content);
                
                if (complexity > 10) {
                    opportunities.push({
                        id: `complexity_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                        type: 'complexity_reduction',
                        category: 'code_quality',
                        priority: complexity > 15 ? 'high' : 'medium',
                        description: `Reduce cyclomatic complexity in ${file}`,
                        location: file,
                        currentValue: complexity,
                        targetValue: Math.max(8, complexity * 0.7),
                        estimatedImpact: Math.min(0.9, complexity / 20),
                        estimatedEffort: Math.min(0.8, complexity / 25),
                        techniques: ['extract_method', 'split_conditionals', 'simplify_expressions'],
                        metadata: {
                            fileSize: content.length,
                            language: this.detectLanguage(file)
                        }
                    });
                }
            } catch (error) {
                console.warn(`⚠️ Could not analyze ${file}: ${error.message}`);
            }
        }
        
        return opportunities;
    }

    async findCodeDuplicationOpportunities() {
        const opportunities = [];
        
        // Simple duplication detection (would use more sophisticated tools in production)
        const codeFiles = await this.findFiles(['**/*.js', '**/*.ts', '**/*.py', '**/*.java']);
        const duplications = await this.detectCodeDuplication(codeFiles.slice(0, 100));
        
        for (const duplication of duplications) {
            if (duplication.similarity > 0.8 && duplication.lines > 10) {
                opportunities.push({
                    id: `duplication_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    type: 'duplication_removal',
                    category: 'code_quality',
                    priority: duplication.lines > 50 ? 'high' : 'medium',
                    description: `Remove code duplication between ${duplication.files.join(' and ')}`,
                    location: duplication.files,
                    currentValue: duplication.similarity,
                    targetValue: 0.3,
                    estimatedImpact: Math.min(0.8, duplication.lines / 100),
                    estimatedEffort: Math.min(0.6, duplication.lines / 150),
                    techniques: ['extract_common_code', 'parameterize_method', 'create_templates'],
                    metadata: {
                        duplicatedLines: duplication.lines,
                        similarity: duplication.similarity,
                        files: duplication.files
                    }
                });
            }
        }
        
        return opportunities;
    }

    async findCodeSmellOpportunities() {
        const opportunities = [];
        
        // Detect various code smells
        const codeFiles = await this.findFiles(['**/*.js', '**/*.ts', '**/*.py']);
        
        for (const file of codeFiles.slice(0, 30)) {
            try {
                const content = await fs.readFile(file, 'utf8');
                const smells = this.detectCodeSmells(content, file);
                
                for (const smell of smells) {
                    opportunities.push({
                        id: `smell_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                        type: 'code_smell_elimination',
                        category: 'code_quality',
                        priority: smell.severity,
                        description: `Eliminate ${smell.type} in ${file}`,
                        location: file,
                        currentValue: smell.severity === 'high' ? 1.0 : 0.5,
                        targetValue: 0.0,
                        estimatedImpact: smell.severity === 'high' ? 0.7 : 0.4,
                        estimatedEffort: smell.severity === 'high' ? 0.6 : 0.3,
                        techniques: [smell.recommendedTechnique],
                        metadata: {
                            smellType: smell.type,
                            line: smell.line,
                            severity: smell.severity
                        }
                    });
                }
            } catch (error) {
                console.warn(`⚠️ Could not analyze ${file}: ${error.message}`);
            }
        }
        
        return opportunities;
    }

    async findRefactoringOpportunities() {
        const opportunities = [];
        
        // Identify refactoring opportunities based on metrics
        const baseline = this.options.baselineMetrics.codeQuality || {};
        
        if (baseline.technicalDebt > 50) {
            opportunities.push({
                id: `refactoring_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                type: 'technical_debt_reduction',
                category: 'code_quality',
                priority: 'high',
                description: 'Reduce overall technical debt',
                location: 'codebase',
                currentValue: baseline.technicalDebt,
                targetValue: baseline.technicalDebt * 0.7,
                estimatedImpact: 0.8,
                estimatedEffort: 0.7,
                techniques: ['systematic_refactoring', 'incremental_improvement', 'automated_refactoring'],
                metadata: {
                    debtCategories: ['complexity', 'duplication', 'design_issues']
                }
            });
        }
        
        return opportunities;
    }

    async discoverPerformanceOpportunities() {
        if (this.options.scope !== 'all' && this.options.scope !== 'performance') return;
        
        console.log('🔍 Discovering performance opportunities...');
        
        const opportunities = [];
        
        // Build performance opportunities
        const buildOpportunities = await this.findBuildPerformanceOpportunities();
        opportunities.push(...buildOpportunities);
        
        // Runtime performance opportunities
        const runtimeOpportunities = await this.findRuntimePerformanceOpportunities();
        opportunities.push(...runtimeOpportunities);
        
        // Bundle optimization opportunities
        const bundleOpportunities = await this.findBundleOptimizationOpportunities();
        opportunities.push(...bundleOpportunities);
        
        // Algorithm optimization opportunities
        const algorithmOpportunities = await this.findAlgorithmOptimizationOpportunities();
        opportunities.push(...algorithmOpportunities);
        
        this.addOpportunities(opportunities, 'performance');
        console.log(`🔍 Found ${opportunities.length} performance opportunities`);
    }

    async findBuildPerformanceOpportunities() {
        const opportunities = [];
        const baseline = this.options.baselineMetrics.performance || {};
        
        if (baseline.buildTime > 60) {
            opportunities.push({
                id: `build_perf_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                type: 'build_optimization',
                category: 'performance',
                priority: baseline.buildTime > 120 ? 'high' : 'medium',
                description: 'Optimize build performance',
                location: 'build_system',
                currentValue: baseline.buildTime,
                targetValue: Math.max(30, baseline.buildTime * 0.6),
                estimatedImpact: 0.8,
                estimatedEffort: 0.6,
                techniques: ['parallel_builds', 'incremental_compilation', 'cache_optimization'],
                metadata: {
                    buildSystem: this.detectBuildSystem(),
                    currentBuildTime: baseline.buildTime
                }
            });
        }
        
        return opportunities;
    }

    async findRuntimePerformanceOpportunities() {
        const opportunities = [];
        
        // Analyze for performance bottlenecks
        const performanceFiles = await this.findFiles(['**/*.js', '**/*.ts']);
        
        for (const file of performanceFiles.slice(0, 20)) {
            try {
                const content = await fs.readFile(file, 'utf8');
                const bottlenecks = this.identifyPerformanceBottlenecks(content, file);
                
                for (const bottleneck of bottlenecks) {
                    opportunities.push({
                        id: `runtime_perf_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                        type: 'runtime_optimization',
                        category: 'performance',
                        priority: bottleneck.severity,
                        description: `Optimize ${bottleneck.type} in ${file}`,
                        location: file,
                        currentValue: bottleneck.impact,
                        targetValue: bottleneck.impact * 0.5,
                        estimatedImpact: bottleneck.impact,
                        estimatedEffort: bottleneck.effort,
                        techniques: bottleneck.techniques,
                        metadata: {
                            bottleneckType: bottleneck.type,
                            line: bottleneck.line
                        }
                    });
                }
            } catch (error) {
                console.warn(`⚠️ Could not analyze ${file}: ${error.message}`);
            }
        }
        
        return opportunities;
    }

    async findBundleOptimizationOpportunities() {
        const opportunities = [];
        const baseline = this.options.baselineMetrics.performance || {};
        
        if (baseline.bundleSize > 2000) {
            opportunities.push({
                id: `bundle_opt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                type: 'bundle_optimization',
                category: 'performance',
                priority: baseline.bundleSize > 4000 ? 'high' : 'medium',
                description: 'Optimize bundle size',
                location: 'bundle_config',
                currentValue: baseline.bundleSize,
                targetValue: Math.max(1500, baseline.bundleSize * 0.7),
                estimatedImpact: 0.7,
                estimatedEffort: 0.5,
                techniques: ['tree_shaking', 'code_splitting', 'compression', 'unused_code_elimination'],
                metadata: {
                    bundler: this.detectBundler(),
                    currentSize: baseline.bundleSize
                }
            });
        }
        
        return opportunities;
    }

    async findAlgorithmOptimizationOpportunities() {
        const opportunities = [];
        
        // Identify algorithmic inefficiencies
        const algorithmFiles = await this.findFiles(['**/*.js', '**/*.ts', '**/*.py']);
        
        for (const file of algorithmFiles.slice(0, 15)) {
            try {
                const content = await fs.readFile(file, 'utf8');
                const inefficiencies = this.identifyAlgorithmicInefficiencies(content, file);
                
                for (const inefficiency of inefficiencies) {
                    opportunities.push({
                        id: `algorithm_opt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                        type: 'algorithm_optimization',
                        category: 'performance',
                        priority: inefficiency.severity,
                        description: `Optimize ${inefficiency.algorithm} in ${file}`,
                        location: file,
                        currentValue: inefficiency.complexity,
                        targetValue: inefficiency.targetComplexity,
                        estimatedImpact: inefficiency.impact,
                        estimatedEffort: inefficiency.effort,
                        techniques: inefficiency.techniques,
                        metadata: {
                            algorithmType: inefficiency.algorithm,
                            currentComplexity: inefficiency.complexity,
                            line: inefficiency.line
                        }
                    });
                }
            } catch (error) {
                console.warn(`⚠️ Could not analyze ${file}: ${error.message}`);
            }
        }
        
        return opportunities;
    }

    async discoverArchitectureOpportunities() {
        if (this.options.scope !== 'all' && this.options.scope !== 'architecture') return;
        
        console.log('🔍 Discovering architecture opportunities...');
        
        const opportunities = [];
        
        // Coupling reduction opportunities
        const couplingOpportunities = await this.findCouplingReductionOpportunities();
        opportunities.push(...couplingOpportunities);
        
        // Modularity improvement opportunities
        const modularityOpportunities = await this.findModularityImprovementOpportunities();
        opportunities.push(...modularityOpportunities);
        
        // Design pattern opportunities
        const designPatternOpportunities = await this.findDesignPatternOpportunities();
        opportunities.push(...designPatternOpportunities);
        
        this.addOpportunities(opportunities, 'architecture');
        console.log(`🔍 Found ${opportunities.length} architecture opportunities`);
    }

    async findCouplingReductionOpportunities() {
        const opportunities = [];
        const baseline = this.options.baselineMetrics.architecture || {};
        
        if (baseline.coupling > 0.4) {
            opportunities.push({
                id: `coupling_reduction_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                type: 'coupling_reduction',
                category: 'architecture',
                priority: 'high',
                description: 'Reduce architectural coupling',
                location: 'architecture',
                currentValue: baseline.coupling,
                targetValue: Math.max(0.2, baseline.coupling * 0.7),
                estimatedImpact: 0.8,
                estimatedEffort: 0.8,
                techniques: ['dependency_injection', 'event_driven_architecture', 'facade_pattern'],
                metadata: {
                    couplingType: 'tight_coupling',
                    affectedModules: this.identifyHighlyCoupledModules()
                }
            });
        }
        
        return opportunities;
    }

    async findModularityImprovementOpportunities() {
        const opportunities = [];
        const baseline = this.options.baselineMetrics.architecture || {};
        
        if (baseline.modularity < 0.7) {
            opportunities.push({
                id: `modularity_improvement_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                type: 'modularity_improvement',
                category: 'architecture',
                priority: 'medium',
                description: 'Improve modular design',
                location: 'architecture',
                currentValue: baseline.modularity,
                targetValue: Math.min(0.9, baseline.modularity + 0.2),
                estimatedImpact: 0.7,
                estimatedEffort: 0.6,
                techniques: ['single_responsibility', 'interface_segregation', 'modular_design'],
                metadata: {
                    currentModularity: baseline.modularity,
                    recommendations: ['split_large_modules', 'define_clear_interfaces']
                }
            });
        }
        
        return opportunities;
    }

    async findDesignPatternOpportunities() {
        const opportunities = [];
        
        // Analyze code for design pattern opportunities
        const codeFiles = await this.findFiles(['**/*.js', '**/*.ts', '**/*.py', '**/*.java']);
        
        for (const file of codeFiles.slice(0, 20)) {
            try {
                const content = await fs.readFile(file, 'utf8');
                const patterns = this.identifyDesignPatternOpportunities(content, file);
                
                for (const pattern of patterns) {
                    opportunities.push({
                        id: `design_pattern_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                        type: 'design_pattern_implementation',
                        category: 'architecture',
                        priority: pattern.priority,
                        description: `Implement ${pattern.pattern} pattern in ${file}`,
                        location: file,
                        currentValue: 0,
                        targetValue: 1,
                        estimatedImpact: pattern.impact,
                        estimatedEffort: pattern.effort,
                        techniques: [pattern.pattern + '_pattern'],
                        metadata: {
                            pattern: pattern.pattern,
                            reason: pattern.reason,
                            line: pattern.line
                        }
                    });
                }
            } catch (error) {
                console.warn(`⚠️ Could not analyze ${file}: ${error.message}`);
            }
        }
        
        return opportunities;
    }

    async discoverDocumentationOpportunities() {
        if (this.options.scope !== 'all' && this.options.scope !== 'documentation') return;
        
        console.log('🔍 Discovering documentation opportunities...');
        
        const opportunities = [];
        const baseline = this.options.baselineMetrics.documentation || {};
        
        if (baseline.coverage < 0.8) {
            opportunities.push({
                id: `doc_coverage_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                type: 'documentation_coverage',
                category: 'documentation',
                priority: 'medium',
                description: 'Improve documentation coverage',
                location: 'documentation',
                currentValue: baseline.coverage,
                targetValue: Math.min(0.9, baseline.coverage + 0.2),
                estimatedImpact: 0.6,
                estimatedEffort: 0.4,
                techniques: ['automated_doc_generation', 'template_creation', 'inline_documentation'],
                metadata: {
                    undocumentedFiles: await this.findUndocumentedFiles()
                }
            });
        }
        
        this.addOpportunities(opportunities, 'documentation');
        console.log(`🔍 Found ${opportunities.length} documentation opportunities`);
    }

    async discoverTestingOpportunities() {
        if (this.options.scope !== 'all' && this.options.scope !== 'testing') return;
        
        console.log('🔍 Discovering testing opportunities...');
        
        const opportunities = [];
        const baseline = this.options.baselineMetrics.testing || {};
        
        if (baseline.coverage < 0.8) {
            opportunities.push({
                id: `test_coverage_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                type: 'test_coverage_improvement',
                category: 'testing',
                priority: 'high',
                description: 'Improve test coverage',
                location: 'tests',
                currentValue: baseline.coverage,
                targetValue: Math.min(0.9, baseline.coverage + 0.2),
                estimatedImpact: 0.9,
                estimatedEffort: 0.6,
                techniques: ['automated_test_generation', 'property_based_testing', 'mutation_testing'],
                metadata: {
                    untestedFiles: await this.findUntestedFiles(),
                    testTypes: baseline.types || {}
                }
            });
        }
        
        this.addOpportunities(opportunities, 'testing');
        console.log(`🔍 Found ${opportunities.length} testing opportunities`);
    }

    async discoverSecurityOpportunities() {
        if (this.options.scope !== 'all' && this.options.scope !== 'security') return;
        
        console.log('🔍 Discovering security opportunities...');
        
        const opportunities = [];
        const baseline = this.options.baselineMetrics.security || {};
        
        if (baseline.vulnerabilities > 0) {
            opportunities.push({
                id: `security_vuln_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                type: 'vulnerability_remediation',
                category: 'security',
                priority: 'critical',
                description: 'Fix security vulnerabilities',
                location: 'security',
                currentValue: baseline.vulnerabilities,
                targetValue: 0,
                estimatedImpact: 1.0,
                estimatedEffort: 0.8,
                techniques: ['vulnerability_patching', 'security_hardening', 'dependency_updates'],
                metadata: {
                    vulnerabilityCount: baseline.vulnerabilities,
                    securityScore: baseline.compliance || 0.8
                }
            });
        }
        
        this.addOpportunities(opportunities, 'security');
        console.log(`🔍 Found ${opportunities.length} security opportunities`);
    }

    async discoverMaintainabilityOpportunities() {
        console.log('🔍 Discovering maintainability opportunities...');
        
        const opportunities = [];
        const baseline = this.options.baselineMetrics.maintainability || {};
        
        if (baseline.readability < 0.8) {
            opportunities.push({
                id: `maintainability_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                type: 'readability_improvement',
                category: 'maintainability',
                priority: 'medium',
                description: 'Improve code readability',
                location: 'codebase',
                currentValue: baseline.readability,
                targetValue: Math.min(0.9, baseline.readability + 0.15),
                estimatedImpact: 0.6,
                estimatedEffort: 0.4,
                techniques: ['naming_improvement', 'code_formatting', 'comment_enhancement'],
                metadata: {
                    readabilityIssues: await this.identifyReadabilityIssues()
                }
            });
        }
        
        this.addOpportunities(opportunities, 'maintainability');
        console.log(`🔍 Found ${opportunities.length} maintainability opportunities`);
    }

    async discoverCrossCuttingOpportunities() {
        console.log('🔍 Discovering cross-cutting opportunities...');
        
        const opportunities = [];
        
        // Find opportunities that span multiple categories
        const crossCuttingOpportunities = await this.identifyCrossCuttingConcerns();
        opportunities.push(...crossCuttingOpportunities);
        
        // Integration opportunities
        const integrationOpportunities = await this.findIntegrationOpportunities();
        opportunities.push(...integrationOpportunities);
        
        this.addOpportunities(opportunities, 'cross_cutting');
        console.log(`🔍 Found ${opportunities.length} cross-cutting opportunities`);
    }

    async identifyCrossCuttingConcerns() {
        return [
            {
                id: `cross_cutting_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                type: 'logging_standardization',
                category: 'cross_cutting',
                priority: 'medium',
                description: 'Standardize logging across the application',
                location: 'codebase',
                currentValue: 0.6,
                targetValue: 0.9,
                estimatedImpact: 0.5,
                estimatedEffort: 0.3,
                techniques: ['centralized_logging', 'structured_logging', 'log_aggregation'],
                metadata: {
                    affectedFiles: await this.findFilesWithLogging()
                }
            }
        ];
    }

    async findIntegrationOpportunities() {
        return [
            {
                id: `integration_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                type: 'ci_cd_optimization',
                category: 'integration',
                priority: 'medium',
                description: 'Optimize CI/CD pipeline',
                location: 'ci_cd',
                currentValue: 0.7,
                targetValue: 0.9,
                estimatedImpact: 0.6,
                estimatedEffort: 0.4,
                techniques: ['pipeline_parallelization', 'cache_optimization', 'artifact_management'],
                metadata: {
                    pipelineFiles: await this.findPipelineFiles()
                }
            }
        ];
    }

    async generateProcessingMatrix() {
        console.log('📊 Generating processing matrix...');
        
        const opportunities = this.discoveryResults.opportunities;
        const maxParallel = parseInt(this.options.maxParallel) || 8;
        
        // Group opportunities by category and priority
        const groupedOpportunities = this.groupOpportunitiesByAttributes(opportunities);
        
        // Create batches for parallel processing
        const batches = this.createProcessingBatches(groupedOpportunities, maxParallel);
        
        // Generate matrix for GitHub Actions
        const matrix = batches.map((batch, index) => ({
            batch_id: index + 1,
            opportunities: batch.map(opp => opp.id),
            enhancement_type: batch[0]?.category || 'mixed',
            priority: this.determineBatchPriority(batch),
            estimated_duration: this.estimateBatchDuration(batch),
            parallel_safe: this.isBatchParallelSafe(batch)
        }));
        
        this.discoveryResults.processingMatrix = {
            batches: batches,
            matrix: matrix,
            totalBatches: batches.length,
            parallelizationStrategy: 'category_priority_based'
        };
        
        console.log(`📊 Generated processing matrix: ${batches.length} batches`);
    }

    groupOpportunitiesByAttributes(opportunities) {
        const grouped = {};
        
        for (const opp of opportunities) {
            const key = `${opp.category}_${opp.priority}`;
            if (!grouped[key]) {
                grouped[key] = [];
            }
            grouped[key].push(opp);
        }
        
        return grouped;
    }

    createProcessingBatches(groupedOpportunities, maxBatches) {
        const batches = [];
        const groups = Object.values(groupedOpportunities);
        
        // Sort groups by priority and size
        groups.sort((a, b) => {
            const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
            const aPriority = priorityOrder[a[0]?.priority] || 1;
            const bPriority = priorityOrder[b[0]?.priority] || 1;
            
            if (aPriority !== bPriority) {
                return bPriority - aPriority;
            }
            
            return b.length - a.length;
        });
        
        // Distribute opportunities across batches
        let currentBatch = 0;
        for (const group of groups) {
            for (const opportunity of group) {
                if (!batches[currentBatch]) {
                    batches[currentBatch] = [];
                }
                
                batches[currentBatch].push(opportunity);
                
                // Move to next batch if current is full or based on balancing logic
                if (batches[currentBatch].length >= Math.ceil(this.discoveryResults.opportunities.length / maxBatches)) {
                    currentBatch = (currentBatch + 1) % maxBatches;
                }
            }
        }
        
        return batches.filter(batch => batch.length > 0);
    }

    determineBatchPriority(batch) {
        const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
        const priorities = batch.map(opp => priorityOrder[opp.priority] || 1);
        const avgPriority = priorities.reduce((sum, p) => sum + p, 0) / priorities.length;
        
        if (avgPriority >= 3.5) return 'high';
        if (avgPriority >= 2.5) return 'medium';
        return 'low';
    }

    estimateBatchDuration(batch) {
        const totalEffort = batch.reduce((sum, opp) => sum + (opp.estimatedEffort || 0.5), 0);
        return Math.ceil(totalEffort * 10); // Estimate in minutes
    }

    isBatchParallelSafe(batch) {
        // Check if opportunities in batch can be processed in parallel
        const categories = new Set(batch.map(opp => opp.category));
        const locations = new Set(batch.map(opp => opp.location));
        
        // If all opportunities are in different categories and locations, they're parallel safe
        return categories.size === batch.length || locations.size === batch.length;
    }

    async analyzeAndPrioritizeOpportunities() {
        console.log('📈 Analyzing and prioritizing opportunities...');
        
        const opportunities = this.discoveryResults.opportunities;
        
        // Calculate priority scores
        for (const opp of opportunities) {
            opp.priorityScore = this.calculatePriorityScore(opp);
        }
        
        // Sort by priority score
        opportunities.sort((a, b) => b.priorityScore - a.priorityScore);
        
        // Update statistics
        this.updateDiscoveryStatistics(opportunities);
        
        console.log('📈 Analysis and prioritization complete');
    }

    calculatePriorityScore(opportunity) {
        const priorityWeight = { critical: 4, high: 3, medium: 2, low: 1 };
        const baseScore = priorityWeight[opportunity.priority] || 1;
        const impactWeight = (opportunity.estimatedImpact || 0.5) * 2;
        const effortPenalty = (opportunity.estimatedEffort || 0.5) * 0.5;
        
        return baseScore + impactWeight - effortPenalty;
    }

    updateDiscoveryStatistics(opportunities) {
        const stats = this.discoveryResults.statistics;
        
        stats.totalOpportunities = opportunities.length;
        
        // Category distribution
        for (const opp of opportunities) {
            stats.categorizedOpportunities[opp.category] = (stats.categorizedOpportunities[opp.category] || 0) + 1;
        }
        
        // Priority distribution
        for (const opp of opportunities) {
            stats.priorityDistribution[opp.priority] = (stats.priorityDistribution[opp.priority] || 0) + 1;
        }
        
        // Complexity analysis
        const impacts = opportunities.map(opp => opp.estimatedImpact || 0.5);
        const efforts = opportunities.map(opp => opp.estimatedEffort || 0.5);
        
        stats.complexityAnalysis = {
            averageImpact: impacts.reduce((sum, val) => sum + val, 0) / impacts.length,
            averageEffort: efforts.reduce((sum, val) => sum + val, 0) / efforts.length,
            highImpactCount: impacts.filter(i => i > 0.7).length,
            lowEffortCount: efforts.filter(e => e < 0.4).length
        };
    }

    addOpportunities(opportunities, category) {
        for (const opp of opportunities) {
            if (opp.estimatedImpact >= this.options.minImpactThreshold) {
                this.discoveryResults.opportunities.push(opp);
            }
        }
    }

    // Utility methods for analysis
    async findFiles(patterns) {
        const files = [];
        for (const pattern of patterns) {
            try {
                const result = execSync(`find . -path "./node_modules" -prune -o -path "./.git" -prune -o -name "${pattern.replace('**/', '')}" -type f -print`, { encoding: 'utf8' });
                files.push(...result.trim().split('\n').filter(f => f && !f.includes('node_modules')));
            } catch (error) {
                // Ignore errors for missing files
            }
        }
        return [...new Set(files)].slice(0, 100); // Limit and deduplicate
    }

    calculateCyclomaticComplexity(content) {
        // Simplified complexity calculation
        const keywords = ['if', 'else', 'for', 'while', 'switch', 'case', 'catch', '&&', '||', '?'];
        let complexity = 1; // Base complexity
        
        for (const keyword of keywords) {
            const matches = content.match(new RegExp(`\\b${keyword}\\b`, 'g'));
            if (matches) {
                complexity += matches.length;
            }
        }
        
        return complexity;
    }

    async detectCodeDuplication(files) {
        // Simplified duplication detection
        const duplications = [];
        
        for (let i = 0; i < files.length && i < 10; i++) {
            for (let j = i + 1; j < files.length && j < 10; j++) {
                try {
                    const content1 = await fs.readFile(files[i], 'utf8');
                    const content2 = await fs.readFile(files[j], 'utf8');
                    
                    const similarity = this.calculateSimilarity(content1, content2);
                    
                    if (similarity > 0.7) {
                        duplications.push({
                            files: [files[i], files[j]],
                            similarity: similarity,
                            lines: Math.min(content1.split('\n').length, content2.split('\n').length)
                        });
                    }
                } catch (error) {
                    // Ignore file read errors
                }
            }
        }
        
        return duplications;
    }

    calculateSimilarity(content1, content2) {
        // Simple similarity calculation based on common lines
        const lines1 = new Set(content1.split('\n').map(line => line.trim()).filter(line => line));
        const lines2 = new Set(content2.split('\n').map(line => line.trim()).filter(line => line));
        
        const intersection = new Set([...lines1].filter(line => lines2.has(line)));
        const union = new Set([...lines1, ...lines2]);
        
        return intersection.size / union.size;
    }

    detectCodeSmells(content, file) {
        const smells = [];
        const lines = content.split('\n');
        
        // Long method detection
        let currentFunctionLines = 0;
        let inFunction = false;
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            if (line.includes('function ') || line.includes(': function') || line.includes('=> {')) {
                inFunction = true;
                currentFunctionLines = 0;
            }
            
            if (inFunction) {
                currentFunctionLines++;
                
                if (line.includes('}') && currentFunctionLines > 50) {
                    smells.push({
                        type: 'long_method',
                        severity: 'medium',
                        line: i + 1,
                        recommendedTechnique: 'extract_method'
                    });
                    inFunction = false;
                }
            }
            
            // Magic number detection
            if (/\b\d{2,}\b/.test(line) && !line.includes('//')) {
                smells.push({
                    type: 'magic_number',
                    severity: 'low',
                    line: i + 1,
                    recommendedTechnique: 'extract_constant'
                });
            }
        }
        
        return smells;
    }

    identifyPerformanceBottlenecks(content, file) {
        const bottlenecks = [];
        const lines = content.split('\n');
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            // Nested loops detection
            if (line.includes('for') && i < lines.length - 10) {
                let nestedLoops = 0;
                for (let j = i + 1; j < Math.min(i + 10, lines.length); j++) {
                    if (lines[j].includes('for')) {
                        nestedLoops++;
                    }
                }
                
                if (nestedLoops > 0) {
                    bottlenecks.push({
                        type: 'nested_loops',
                        severity: nestedLoops > 1 ? 'high' : 'medium',
                        impact: Math.min(0.9, 0.3 + nestedLoops * 0.3),
                        effort: Math.min(0.8, 0.4 + nestedLoops * 0.2),
                        line: i + 1,
                        techniques: ['algorithm_optimization', 'caching', 'early_termination']
                    });
                }
            }
            
            // Synchronous operations that could be async
            if (line.includes('readFileSync') || line.includes('execSync')) {
                bottlenecks.push({
                    type: 'blocking_operation',
                    severity: 'medium',
                    impact: 0.6,
                    effort: 0.3,
                    line: i + 1,
                    techniques: ['async_operations', 'promise_based', 'callback_optimization']
                });
            }
        }
        
        return bottlenecks;
    }

    identifyAlgorithmicInefficiencies(content, file) {
        const inefficiencies = [];
        
        // Look for common algorithmic inefficiencies
        if (content.includes('sort()') && content.includes('indexOf')) {
            inefficiencies.push({
                algorithm: 'inefficient_search',
                complexity: 'O(n²)',
                targetComplexity: 'O(n log n)',
                severity: 'medium',
                impact: 0.7,
                effort: 0.5,
                line: content.split('\n').findIndex(line => line.includes('sort()')) + 1,
                techniques: ['binary_search', 'hash_map', 'pre_sorting']
            });
        }
        
        return inefficiencies;
    }

    identifyDesignPatternOpportunities(content, file) {
        const patterns = [];
        
        // Look for repeated conditional logic that could use strategy pattern
        const conditionalCount = (content.match(/if\s*\(/g) || []).length;
        if (conditionalCount > 10) {
            patterns.push({
                pattern: 'strategy',
                priority: 'medium',
                impact: 0.6,
                effort: 0.7,
                reason: 'Multiple conditional statements detected',
                line: content.split('\n').findIndex(line => line.includes('if')) + 1
            });
        }
        
        return patterns;
    }

    detectLanguage(file) {
        const ext = path.extname(file);
        const languageMap = {
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'react',
            '.tsx': 'react-typescript',
            '.py': 'python',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c'
        };
        return languageMap[ext] || 'unknown';
    }

    detectBuildSystem() {
        // Simple build system detection
        try {
            execSync('ls package.json', { stdio: 'ignore' });
            return 'npm';
        } catch (error) {
            try {
                execSync('ls Makefile', { stdio: 'ignore' });
                return 'make';
            } catch (error) {
                return 'unknown';
            }
        }
    }

    detectBundler() {
        // Simple bundler detection
        try {
            execSync('ls webpack.config.js', { stdio: 'ignore' });
            return 'webpack';
        } catch (error) {
            try {
                execSync('ls vite.config.js', { stdio: 'ignore' });
                return 'vite';
            } catch (error) {
                return 'unknown';
            }
        }
    }

    identifyHighlyCoupledModules() {
        // Simplified coupling analysis
        return ['module_a', 'module_b', 'utils'];
    }

    async findUndocumentedFiles() {
        const codeFiles = await this.findFiles(['**/*.js', '**/*.ts']);
        const undocumented = [];
        
        for (const file of codeFiles.slice(0, 10)) {
            try {
                const content = await fs.readFile(file, 'utf8');
                if (!content.includes('/**') && !content.includes('//')) {
                    undocumented.push(file);
                }
            } catch (error) {
                // Ignore file read errors
            }
        }
        
        return undocumented;
    }

    async findUntestedFiles() {
        const codeFiles = await this.findFiles(['**/*.js', '**/*.ts']);
        const testFiles = await this.findFiles(['**/*.test.js', '**/*.spec.js', '**/*.test.ts', '**/*.spec.ts']);
        
        const testedFiles = new Set();
        for (const testFile of testFiles) {
            const baseName = path.basename(testFile).replace(/\.(test|spec)\.(js|ts)$/, '');
            testedFiles.add(baseName);
        }
        
        return codeFiles.filter(file => {
            const baseName = path.basename(file).replace(/\.(js|ts)$/, '');
            return !testedFiles.has(baseName);
        }).slice(0, 20);
    }

    async identifyReadabilityIssues() {
        return ['long_variable_names', 'unclear_function_names', 'missing_comments'];
    }

    async findFilesWithLogging() {
        const files = await this.findFiles(['**/*.js', '**/*.ts']);
        const loggingFiles = [];
        
        for (const file of files.slice(0, 20)) {
            try {
                const content = await fs.readFile(file, 'utf8');
                if (content.includes('console.log') || content.includes('logger.')) {
                    loggingFiles.push(file);
                }
            } catch (error) {
                // Ignore file read errors
            }
        }
        
        return loggingFiles;
    }

    async findPipelineFiles() {
        return await this.findFiles(['**/.github/workflows/*.yml', '**/.gitlab-ci.yml', '**/Jenkinsfile']);
    }

    async saveDiscoveryResults() {
        const resultsPath = path.join(this.options.workspaceDir, 'discovery', 'discovery-results.json');
        await fs.writeFile(resultsPath, JSON.stringify(this.discoveryResults, null, 2));
        
        // Save processing matrix separately for easier access
        const matrixPath = path.join(this.options.workspaceDir, 'discovery', 'processing-matrix.json');
        await fs.writeFile(matrixPath, JSON.stringify(this.discoveryResults.processingMatrix, null, 2));
    }

    async outputGitHubActions() {
        // Output for GitHub Actions
        console.log(`::set-output name=matrix::${JSON.stringify(this.discoveryResults.processingMatrix.matrix)}`);
        console.log(`::set-output name=total::${this.discoveryResults.statistics.totalOpportunities}`);
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
            case '--enhancement-id':
                options.enhancementId = value;
                break;
            case '--scope':
                options.scope = value;
                break;
            case '--baseline':
                options.baseline = value;
                break;
            case '--targets':
                options.targets = value;
                break;
            case '--workspace':
                options.workspace = value;
                break;
            case '--depth':
                options.depth = value;
                break;
            case '--min-impact':
                options.minImpact = value;
                break;
        }
    }
    
    try {
        const discovery = new EnhancementDiscovery(options);
        await discovery.discoverOpportunities();
        console.log('🎉 Enhancement discovery completed successfully!');
        process.exit(0);
    } catch (error) {
        console.error('💥 Enhancement discovery failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { EnhancementDiscovery };