#!/usr/bin/env node
/**
 * Results Consolidation Engine
 * Consolidates and analyzes results from all parallel batch processing
 */

const fs = require('fs').promises;
const path = require('path');

class ResultsConsolidator {
    constructor(options = {}) {
        this.options = {
            inputDir: options.input || '.taskmaster/consolidated/',
            outputPath: options.output || '.taskmaster/consolidated/final-results.json',
            generateReport: options.generateReport !== false,
            enableAnalytics: options.enableAnalytics !== false,
            ...options
        };
        
        this.consolidatedResults = {
            metadata: {
                consolidatedAt: new Date().toISOString(),
                consolidationOptions: this.options
            },
            batchResults: [],
            summary: {
                totalBatches: 0,
                totalTodos: 0,
                totalAtomizedTodos: 0,
                totalImprovements: 0,
                overallSuccessRate: 0,
                averageProcessingTime: 0,
                validationResults: {},
                qualityMetrics: {}
            },
            analytics: {
                categoryAnalysis: {},
                sourceAnalysis: {},
                performanceAnalysis: {},
                improvementAnalysis: {},
                trendAnalysis: {}
            },
            aggregatedImprovements: [],
            recommendations: [],
            errors: [],
            warnings: []
        };
    }

    async consolidateResults() {
        console.log('🔄 Starting results consolidation...');
        
        try {
            // Discover and load all batch results
            const batchFiles = await this.discoverBatchResults();
            
            // Load and consolidate each batch
            for (const batchFile of batchFiles) {
                await this.loadAndConsolidateBatch(batchFile);
            }
            
            // Generate summary statistics
            await this.generateSummaryStatistics();
            
            // Perform analytics if enabled
            if (this.options.enableAnalytics) {
                await this.performAnalytics();
            }
            
            // Aggregate improvements
            await this.aggregateImprovements();
            
            // Generate recommendations
            await this.generateRecommendations();
            
            // Save consolidated results
            await this.saveConsolidatedResults();
            
            // Generate report if requested
            if (this.options.generateReport) {
                await this.generateReport();
            }
            
            console.log(`✅ Consolidation complete: ${this.consolidatedResults.summary.totalBatches} batches, ${this.consolidatedResults.summary.totalTodos} todos processed`);
            
        } catch (error) {
            console.error('❌ Results consolidation failed:', error);
            throw error;
        }
    }

    async discoverBatchResults() {
        console.log(`🔍 Discovering batch results in ${this.options.inputDir}...`);
        
        const batchFiles = [];
        
        try {
            const files = await fs.readdir(this.options.inputDir, { withFileTypes: true });
            
            // Look for batch result directories
            for (const file of files) {
                if (file.isDirectory() && file.name.startsWith('batch-') && file.name.endsWith('-results')) {
                    const batchDir = path.join(this.options.inputDir, file.name);
                    const resultsFile = path.join(batchDir, 'results.json');
                    
                    try {
                        await fs.access(resultsFile);
                        batchFiles.push({
                            batchId: file.name,
                            batchDir: batchDir,
                            resultsFile: resultsFile,
                            atomizedFile: path.join(batchDir, 'atomized.json'),
                            validationFile: path.join(batchDir, 'validation.json'),
                            improvementsFile: path.join(batchDir, 'improvements.json')
                        });
                    } catch (error) {
                        console.warn(`⚠️ No results.json found for ${file.name}`);
                    }
                }
            }
            
        } catch (error) {
            console.warn(`⚠️ Could not read input directory: ${error.message}`);
        }
        
        console.log(`🔍 Found ${batchFiles.length} batch result sets`);
        
        return batchFiles;
    }

    async loadAndConsolidateBatch(batchFile) {
        console.log(`📂 Loading batch: ${batchFile.batchId}`);
        
        const batchData = {
            batchId: batchFile.batchId,
            files: {},
            summary: {},
            errors: []
        };
        
        try {
            // Load main results
            batchData.files.results = await this.loadJsonFile(batchFile.resultsFile);
            
            // Load atomized results if available
            try {
                batchData.files.atomized = await this.loadJsonFile(batchFile.atomizedFile);
            } catch (error) {
                console.warn(`⚠️ No atomized results for ${batchFile.batchId}`);
            }
            
            // Load validation results if available
            try {
                batchData.files.validation = await this.loadJsonFile(batchFile.validationFile);
            } catch (error) {
                console.warn(`⚠️ No validation results for ${batchFile.batchId}`);
            }
            
            // Load improvements if available
            try {
                batchData.files.improvements = await this.loadJsonFile(batchFile.improvementsFile);
            } catch (error) {
                console.warn(`⚠️ No improvements for ${batchFile.batchId}`);
            }
            
            // Extract summary information
            await this.extractBatchSummary(batchData);
            
            this.consolidatedResults.batchResults.push(batchData);
            
        } catch (error) {
            console.error(`❌ Failed to load batch ${batchFile.batchId}:`, error);
            batchData.errors.push({
                type: 'load_error',
                message: error.message,
                timestamp: new Date().toISOString()
            });
            this.consolidatedResults.errors.push(batchData.errors[0]);
        }
    }

    async loadJsonFile(filePath) {
        const data = await fs.readFile(filePath, 'utf8');
        return JSON.parse(data);
    }

    async extractBatchSummary(batchData) {
        const summary = {
            batchId: batchData.batchId,
            todosProcessed: 0,
            atomizedTodos: 0,
            improvements: 0,
            successRate: 0,
            processingTime: 0,
            validationPassed: false,
            errors: batchData.errors.length
        };
        
        // Extract from results
        if (batchData.files.results) {
            const results = batchData.files.results;
            summary.todosProcessed = results.performance?.totalTodos || 0;
            summary.successRate = results.performance?.successRate || 0;
            summary.processingTime = results.performance?.duration || 0;
        }
        
        // Extract from atomized results
        if (batchData.files.atomized) {
            const atomized = batchData.files.atomized;
            summary.atomizedTodos = atomized.atomizationStats?.totalAtomized || 0;
        }
        
        // Extract from validation results
        if (batchData.files.validation) {
            const validation = batchData.files.validation;
            summary.validationPassed = validation.overall?.success || false;
        }
        
        // Extract from improvements
        if (batchData.files.improvements) {
            const improvements = batchData.files.improvements;
            summary.improvements = improvements.improvements?.length || 0;
        }
        
        batchData.summary = summary;
    }

    async generateSummaryStatistics() {
        console.log('📊 Generating summary statistics...');
        
        const summary = this.consolidatedResults.summary;
        
        summary.totalBatches = this.consolidatedResults.batchResults.length;
        
        // Aggregate totals
        for (const batch of this.consolidatedResults.batchResults) {
            summary.totalTodos += batch.summary.todosProcessed;
            summary.totalAtomizedTodos += batch.summary.atomizedTodos;
            summary.totalImprovements += batch.summary.improvements;
        }
        
        // Calculate averages
        if (summary.totalBatches > 0) {
            const totalSuccessRate = this.consolidatedResults.batchResults
                .reduce((sum, batch) => sum + batch.summary.successRate, 0);
            summary.overallSuccessRate = totalSuccessRate / summary.totalBatches;
            
            const totalProcessingTime = this.consolidatedResults.batchResults
                .reduce((sum, batch) => sum + batch.summary.processingTime, 0);
            summary.averageProcessingTime = totalProcessingTime / summary.totalBatches;
        }
        
        // Validation summary
        const validationPassed = this.consolidatedResults.batchResults
            .filter(batch => batch.summary.validationPassed).length;
        summary.validationResults = {
            totalBatches: summary.totalBatches,
            passedValidation: validationPassed,
            failedValidation: summary.totalBatches - validationPassed,
            validationRate: summary.totalBatches > 0 ? validationPassed / summary.totalBatches : 0
        };
        
        // Quality metrics summary
        summary.qualityMetrics = {
            atomizationFactor: summary.totalTodos > 0 ? summary.totalAtomizedTodos / summary.totalTodos : 0,
            improvementRate: summary.totalTodos > 0 ? summary.totalImprovements / summary.totalTodos : 0,
            errorRate: this.consolidatedResults.errors.length / Math.max(1, summary.totalTodos)
        };
        
        console.log(`📊 Summary generated: ${summary.totalTodos} todos → ${summary.totalAtomizedTodos} atomic tasks → ${summary.totalImprovements} improvements`);
    }

    async performAnalytics() {
        console.log('📈 Performing analytics...');
        
        await this.analyzeCategoryDistribution();
        await this.analyzeSourceDistribution();
        await this.analyzePerformanceMetrics();
        await this.analyzeImprovementPatterns();
        await this.analyzeTrends();
        
        console.log('📈 Analytics complete');
    }

    async analyzeCategoryDistribution() {
        const categoryData = {};
        
        for (const batch of this.consolidatedResults.batchResults) {
            if (batch.files.results && batch.files.results.processedTodos) {
                for (const todo of batch.files.results.processedTodos) {
                    const category = todo.category || 'unknown';
                    if (!categoryData[category]) {
                        categoryData[category] = {
                            count: 0,
                            successRate: 0,
                            successCount: 0,
                            totalCount: 0
                        };
                    }
                    
                    categoryData[category].count++;
                    categoryData[category].totalCount++;
                    
                    if (todo.success) {
                        categoryData[category].successCount++;
                    }
                }
            }
        }
        
        // Calculate success rates
        for (const category in categoryData) {
            const data = categoryData[category];
            data.successRate = data.totalCount > 0 ? data.successCount / data.totalCount : 0;
        }
        
        this.consolidatedResults.analytics.categoryAnalysis = categoryData;
    }

    async analyzeSourceDistribution() {
        const sourceData = {};
        
        for (const batch of this.consolidatedResults.batchResults) {
            if (batch.files.results && batch.files.results.processedTodos) {
                for (const todo of batch.files.results.processedTodos) {
                    const source = todo.source || 'unknown';
                    if (!sourceData[source]) {
                        sourceData[source] = {
                            count: 0,
                            successRate: 0,
                            atomizationRate: 0
                        };
                    }
                    
                    sourceData[source].count++;
                }
            }
        }
        
        this.consolidatedResults.analytics.sourceAnalysis = sourceData;
    }

    async analyzePerformanceMetrics() {
        const performanceData = {
            processingTimeDistribution: {},
            successRateDistribution: {},
            batchSizeImpact: {},
            correlations: {}
        };
        
        const processingTimes = [];
        const successRates = [];
        const batchSizes = [];
        
        for (const batch of this.consolidatedResults.batchResults) {
            const summary = batch.summary;
            
            processingTimes.push(summary.processingTime);
            successRates.push(summary.successRate);
            batchSizes.push(summary.todosProcessed);
        }
        
        // Calculate distributions
        performanceData.processingTimeDistribution = this.calculateDistribution(processingTimes);
        performanceData.successRateDistribution = this.calculateDistribution(successRates);
        performanceData.batchSizeImpact = this.analyzeBatchSizeImpact(batchSizes, successRates);
        
        // Calculate correlations
        performanceData.correlations = {
            sizeVsTime: this.calculateCorrelation(batchSizes, processingTimes),
            sizeVsSuccess: this.calculateCorrelation(batchSizes, successRates),
            timeVsSuccess: this.calculateCorrelation(processingTimes, successRates)
        };
        
        this.consolidatedResults.analytics.performanceAnalysis = performanceData;
    }

    async analyzeImprovementPatterns() {
        const improvementData = {
            typeDistribution: {},
            categoryDistribution: {},
            effectivenessAnalysis: {},
            recursiveAnalysis: {}
        };
        
        let totalImprovements = 0;
        const effectivenessScores = [];
        
        for (const batch of this.consolidatedResults.batchResults) {
            if (batch.files.improvements && batch.files.improvements.improvements) {
                for (const improvement of batch.files.improvements.improvements) {
                    totalImprovements++;
                    
                    // Type distribution
                    const type = improvement.type || 'unknown';
                    improvementData.typeDistribution[type] = (improvementData.typeDistribution[type] || 0) + 1;
                    
                    // Category distribution
                    const category = improvement.category || 'unknown';
                    improvementData.categoryDistribution[category] = (improvementData.categoryDistribution[category] || 0) + 1;
                    
                    // Effectiveness
                    if (improvement.outcomes && improvement.outcomes.metrics && improvement.outcomes.metrics.effectiveness) {
                        effectivenessScores.push(improvement.outcomes.metrics.effectiveness);
                    }
                }
            }
            
            // Recursive improvements
            if (batch.files.improvements && batch.files.improvements.recursiveImprovements) {
                improvementData.recursiveAnalysis.count = batch.files.improvements.recursiveImprovements.length;
            }
        }
        
        // Calculate effectiveness statistics
        if (effectivenessScores.length > 0) {
            improvementData.effectivenessAnalysis = {
                average: effectivenessScores.reduce((sum, score) => sum + score, 0) / effectivenessScores.length,
                min: Math.min(...effectivenessScores),
                max: Math.max(...effectivenessScores),
                distribution: this.calculateDistribution(effectivenessScores)
            };
        }
        
        this.consolidatedResults.analytics.improvementAnalysis = improvementData;
    }

    async analyzeTrends() {
        const trendData = {
            batchPerformanceTrend: [],
            successRateTrend: [],
            improvementTrend: [],
            qualityTrend: []
        };
        
        // Sort batches by processing order (assuming sequential batch IDs)
        const sortedBatches = this.consolidatedResults.batchResults
            .sort((a, b) => a.batchId.localeCompare(b.batchId));
        
        for (let i = 0; i < sortedBatches.length; i++) {
            const batch = sortedBatches[i];
            
            trendData.batchPerformanceTrend.push({
                batchIndex: i,
                processingTime: batch.summary.processingTime,
                todosProcessed: batch.summary.todosProcessed
            });
            
            trendData.successRateTrend.push({
                batchIndex: i,
                successRate: batch.summary.successRate
            });
            
            trendData.improvementTrend.push({
                batchIndex: i,
                improvements: batch.summary.improvements
            });
        }
        
        this.consolidatedResults.analytics.trendAnalysis = trendData;
    }

    calculateDistribution(values) {
        if (values.length === 0) return {};
        
        const sorted = values.sort((a, b) => a - b);
        
        return {
            min: sorted[0],
            max: sorted[sorted.length - 1],
            median: sorted[Math.floor(sorted.length / 2)],
            average: values.reduce((sum, val) => sum + val, 0) / values.length,
            q1: sorted[Math.floor(sorted.length * 0.25)],
            q3: sorted[Math.floor(sorted.length * 0.75)]
        };
    }

    analyzeBatchSizeImpact(sizes, outcomes) {
        const sizeRanges = {
            small: { min: 0, max: 10, sizes: [], outcomes: [] },
            medium: { min: 11, max: 25, sizes: [], outcomes: [] },
            large: { min: 26, max: Infinity, sizes: [], outcomes: [] }
        };
        
        for (let i = 0; i < sizes.length; i++) {
            const size = sizes[i];
            const outcome = outcomes[i];
            
            if (size <= 10) {
                sizeRanges.small.sizes.push(size);
                sizeRanges.small.outcomes.push(outcome);
            } else if (size <= 25) {
                sizeRanges.medium.sizes.push(size);
                sizeRanges.medium.outcomes.push(outcome);
            } else {
                sizeRanges.large.sizes.push(size);
                sizeRanges.large.outcomes.push(outcome);
            }
        }
        
        // Calculate averages for each range
        const impact = {};
        for (const [range, data] of Object.entries(sizeRanges)) {
            if (data.outcomes.length > 0) {
                impact[range] = {
                    count: data.outcomes.length,
                    averageOutcome: data.outcomes.reduce((sum, val) => sum + val, 0) / data.outcomes.length,
                    averageSize: data.sizes.reduce((sum, val) => sum + val, 0) / data.sizes.length
                };
            }
        }
        
        return impact;
    }

    calculateCorrelation(x, y) {
        if (x.length !== y.length || x.length === 0) return 0;
        
        const n = x.length;
        const meanX = x.reduce((sum, val) => sum + val, 0) / n;
        const meanY = y.reduce((sum, val) => sum + val, 0) / n;
        
        let numerator = 0;
        let denomX = 0;
        let denomY = 0;
        
        for (let i = 0; i < n; i++) {
            const diffX = x[i] - meanX;
            const diffY = y[i] - meanY;
            
            numerator += diffX * diffY;
            denomX += diffX * diffX;
            denomY += diffY * diffY;
        }
        
        const denominator = Math.sqrt(denomX * denomY);
        return denominator === 0 ? 0 : numerator / denominator;
    }

    async aggregateImprovements() {
        console.log('🔧 Aggregating improvements...');
        
        const improvementMap = new Map();
        
        for (const batch of this.consolidatedResults.batchResults) {
            if (batch.files.improvements && batch.files.improvements.improvements) {
                for (const improvement of batch.files.improvements.improvements) {
                    const key = `${improvement.type}_${improvement.category}`;
                    
                    if (!improvementMap.has(key)) {
                        improvementMap.set(key, {
                            type: improvement.type,
                            category: improvement.category,
                            title: improvement.title,
                            occurrences: 0,
                            totalEffectiveness: 0,
                            artifacts: new Set(),
                            implementations: []
                        });
                    }
                    
                    const aggregated = improvementMap.get(key);
                    aggregated.occurrences++;
                    
                    if (improvement.outcomes && improvement.outcomes.metrics && improvement.outcomes.metrics.effectiveness) {
                        aggregated.totalEffectiveness += improvement.outcomes.metrics.effectiveness;
                    }
                    
                    if (improvement.implementation && improvement.implementation.artifacts) {
                        improvement.implementation.artifacts.forEach(artifact => 
                            aggregated.artifacts.add(artifact)
                        );
                    }
                    
                    aggregated.implementations.push({
                        batchId: batch.batchId,
                        improvementId: improvement.id,
                        strategy: improvement.implementation?.strategy
                    });
                }
            }
        }
        
        // Convert to array and calculate averages
        this.consolidatedResults.aggregatedImprovements = Array.from(improvementMap.values()).map(improvement => ({
            ...improvement,
            averageEffectiveness: improvement.occurrences > 0 ? improvement.totalEffectiveness / improvement.occurrences : 0,
            artifacts: Array.from(improvement.artifacts),
            priority: this.calculateAggregatedPriority(improvement)
        }));
        
        console.log(`🔧 Aggregated ${this.consolidatedResults.aggregatedImprovements.length} unique improvement types`);
    }

    calculateAggregatedPriority(improvement) {
        let priority = 'low';
        
        if (improvement.occurrences > 3 && improvement.averageEffectiveness > 0.8) {
            priority = 'high';
        } else if (improvement.occurrences > 1 || improvement.averageEffectiveness > 0.6) {
            priority = 'medium';
        }
        
        return priority;
    }

    async generateRecommendations() {
        console.log('💡 Generating recommendations...');
        
        const recommendations = [];
        const summary = this.consolidatedResults.summary;
        
        // Success rate recommendations
        if (summary.overallSuccessRate < 0.8) {
            recommendations.push({
                type: 'success_rate_improvement',
                priority: 'high',
                message: `Overall success rate is ${(summary.overallSuccessRate * 100).toFixed(1)}%. Consider improving batch processing strategies.`,
                targetMetric: 'overallSuccessRate',
                targetValue: 0.85,
                actions: [
                    'Review failed batch processing logs',
                    'Optimize batch size and parallel processing',
                    'Improve error handling and recovery'
                ]
            });
        }
        
        // Validation recommendations
        if (summary.validationResults.validationRate < 0.9) {
            recommendations.push({
                type: 'validation_improvement',
                priority: 'high',
                message: `Validation rate is ${(summary.validationResults.validationRate * 100).toFixed(1)}%. Improve todo quality and validation criteria.`,
                targetMetric: 'validationRate',
                targetValue: 0.95,
                actions: [
                    'Review validation criteria',
                    'Improve todo quality standards',
                    'Enhance atomization strategies'
                ]
            });
        }
        
        // Performance recommendations
        if (summary.averageProcessingTime > 120000) { // 2 minutes
            recommendations.push({
                type: 'performance_optimization',
                priority: 'medium',
                message: `Average processing time is ${(summary.averageProcessingTime / 1000).toFixed(1)}s. Consider performance optimizations.`,
                targetMetric: 'averageProcessingTime',
                targetValue: 60000,
                actions: [
                    'Optimize parallel processing configuration',
                    'Review batch size optimization',
                    'Implement performance monitoring'
                ]
            });
        }
        
        // Improvement recommendations
        if (summary.qualityMetrics.improvementRate < 0.1) {
            recommendations.push({
                type: 'improvement_generation',
                priority: 'medium',
                message: `Improvement rate is ${(summary.qualityMetrics.improvementRate * 100).toFixed(1)}%. Focus on generating more actionable improvements.`,
                targetMetric: 'improvementRate',
                targetValue: 0.2,
                actions: [
                    'Enhance improvement prompt generation',
                    'Implement recursive improvement strategies',
                    'Focus on high-impact improvement categories'
                ]
            });
        }
        
        // Category-specific recommendations
        if (this.consolidatedResults.analytics.categoryAnalysis) {
            const categoryAnalysis = this.consolidatedResults.analytics.categoryAnalysis;
            
            for (const [category, data] of Object.entries(categoryAnalysis)) {
                if (data.successRate < 0.7 && data.count > 5) {
                    recommendations.push({
                        type: 'category_improvement',
                        priority: 'medium',
                        message: `Category '${category}' has low success rate (${(data.successRate * 100).toFixed(1)}%). Focus on improving this category.`,
                        category: category,
                        targetMetric: 'categorySuccessRate',
                        targetValue: 0.8,
                        actions: [
                            `Review ${category} processing strategies`,
                            `Improve ${category} todo quality`,
                            `Add ${category}-specific validation rules`
                        ]
                    });
                }
            }
        }
        
        // High-impact improvement recommendations
        const highImpactImprovements = this.consolidatedResults.aggregatedImprovements
            .filter(imp => imp.priority === 'high')
            .slice(0, 3);
        
        if (highImpactImprovements.length > 0) {
            recommendations.push({
                type: 'high_impact_implementation',
                priority: 'high',
                message: `Implement these high-impact improvements: ${highImpactImprovements.map(imp => imp.type).join(', ')}`,
                improvements: highImpactImprovements.map(imp => imp.type),
                actions: [
                    'Prioritize high-impact improvement implementation',
                    'Allocate resources to top improvement areas',
                    'Measure improvement effectiveness'
                ]
            });
        }
        
        this.consolidatedResults.recommendations = recommendations;
        
        console.log(`💡 Generated ${recommendations.length} recommendations`);
    }

    async saveConsolidatedResults() {
        console.log(`💾 Saving consolidated results to ${this.options.outputPath}...`);
        
        // Ensure output directory exists
        const outputDir = path.dirname(this.options.outputPath);
        await fs.mkdir(outputDir, { recursive: true });
        
        await fs.writeFile(this.options.outputPath, JSON.stringify(this.consolidatedResults, null, 2));
        
        // Save executive summary
        const executiveSummary = {
            processingSummary: {
                totalBatches: this.consolidatedResults.summary.totalBatches,
                totalTodos: this.consolidatedResults.summary.totalTodos,
                totalImprovements: this.consolidatedResults.summary.totalImprovements,
                overallSuccessRate: this.consolidatedResults.summary.overallSuccessRate,
                validationRate: this.consolidatedResults.summary.validationResults.validationRate
            },
            keyMetrics: {
                atomizationFactor: this.consolidatedResults.summary.qualityMetrics.atomizationFactor,
                improvementRate: this.consolidatedResults.summary.qualityMetrics.improvementRate,
                averageProcessingTime: this.consolidatedResults.summary.averageProcessingTime
            },
            topRecommendations: this.consolidatedResults.recommendations
                .filter(rec => rec.priority === 'high')
                .slice(0, 5),
            topImprovements: this.consolidatedResults.aggregatedImprovements
                .filter(imp => imp.priority === 'high')
                .slice(0, 5),
            timestamp: new Date().toISOString()
        };
        
        await fs.writeFile(
            path.join(outputDir, 'executive-summary.json'),
            JSON.stringify(executiveSummary, null, 2)
        );
        
        console.log(`✅ Consolidated results saved`);
    }

    async generateReport() {
        console.log('📊 Generating comprehensive report...');
        
        const reportPath = path.join(path.dirname(this.options.outputPath), 'consolidated-report.md');
        
        const report = `# Recursive Todo Processing - Consolidated Results Report

Generated: ${new Date().toISOString()}

## Executive Summary

- **Total Batches Processed**: ${this.consolidatedResults.summary.totalBatches}
- **Total Todos Processed**: ${this.consolidatedResults.summary.totalTodos}
- **Total Atomized Tasks**: ${this.consolidatedResults.summary.totalAtomizedTodos}
- **Total Improvements Generated**: ${this.consolidatedResults.summary.totalImprovements}
- **Overall Success Rate**: ${(this.consolidatedResults.summary.overallSuccessRate * 100).toFixed(1)}%
- **Validation Pass Rate**: ${(this.consolidatedResults.summary.validationResults.validationRate * 100).toFixed(1)}%

## Key Metrics

### Quality Metrics
- **Atomization Factor**: ${this.consolidatedResults.summary.qualityMetrics.atomizationFactor.toFixed(2)}x
- **Improvement Rate**: ${(this.consolidatedResults.summary.qualityMetrics.improvementRate * 100).toFixed(1)}%
- **Error Rate**: ${(this.consolidatedResults.summary.qualityMetrics.errorRate * 100).toFixed(2)}%

### Performance Metrics
- **Average Processing Time**: ${(this.consolidatedResults.summary.averageProcessingTime / 1000).toFixed(1)}s
- **Parallel Efficiency**: ${this.calculateParallelEfficiency().toFixed(1)}%

## Top Recommendations

${this.consolidatedResults.recommendations.slice(0, 5).map((rec, index) => 
    `${index + 1}. **${rec.type}** (${rec.priority}): ${rec.message}`
).join('\n')}

## High-Impact Improvements

${this.consolidatedResults.aggregatedImprovements
    .filter(imp => imp.priority === 'high')
    .slice(0, 5)
    .map((imp, index) => 
        `${index + 1}. **${imp.type}** (${imp.category}): ${imp.occurrences} occurrences, ${(imp.averageEffectiveness * 100).toFixed(1)}% avg effectiveness`
    ).join('\n')}

## Category Analysis

${Object.entries(this.consolidatedResults.analytics.categoryAnalysis || {})
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, 5)
    .map(([category, data]) => 
        `- **${category}**: ${data.count} todos, ${(data.successRate * 100).toFixed(1)}% success rate`
    ).join('\n')}

## Performance Correlations

${this.consolidatedResults.analytics.performanceAnalysis ? 
    Object.entries(this.consolidatedResults.analytics.performanceAnalysis.correlations || {})
        .map(([metric, correlation]) => 
            `- **${metric}**: ${correlation > 0 ? 'Positive' : 'Negative'} correlation (${correlation.toFixed(3)})`
        ).join('\n')
    : 'No performance correlation data available'}

## Recommendations for Next Steps

1. **Immediate Actions**: Focus on high-priority recommendations
2. **Short-term**: Implement top 3 high-impact improvements
3. **Long-term**: Address category-specific performance issues
4. **Monitoring**: Track key metrics for continuous improvement

---

*Report generated by Recursive Todo Processing System*
`;
        
        await fs.writeFile(reportPath, report);
        
        console.log(`📊 Report generated: ${reportPath}`);
    }

    calculateParallelEfficiency() {
        const totalProcessingTime = this.consolidatedResults.batchResults
            .reduce((sum, batch) => sum + batch.summary.processingTime, 0);
        
        if (totalProcessingTime === 0) return 0;
        
        const serialTime = totalProcessingTime; // Assume this would be serial time
        const parallelTime = Math.max(...this.consolidatedResults.batchResults.map(b => b.summary.processingTime));
        
        return parallelTime > 0 ? (serialTime / parallelTime) / this.consolidatedResults.summary.totalBatches * 100 : 0;
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
            case '--generate-report':
                options.generateReport = true;
                i--; // No value for this flag
                break;
            case '--no-analytics':
                options.enableAnalytics = false;
                i--; // No value for this flag
                break;
        }
    }
    
    try {
        const consolidator = new ResultsConsolidator(options);
        await consolidator.consolidateResults();
        console.log('🎉 Results consolidation completed successfully!');
        process.exit(0);
    } catch (error) {
        console.error('💥 Results consolidation failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { ResultsConsolidator };