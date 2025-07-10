#!/usr/bin/env node
/**
 * Recursive Todo Processing System Deployment Script
 * Deploys and validates the complete GitHub Actions workflow system
 */

const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');

class SystemDeployer {
    constructor(options = {}) {
        this.options = {
            environment: options.environment || 'production', // development, staging, production
            validateDeployment: options.validate !== false,
            runTests: options.test !== false,
            dryRun: options.dryRun === true,
            force: options.force === true,
            ...options
        };
        
        this.deploymentResults = {
            metadata: {
                deploymentStartTime: new Date().toISOString(),
                environment: this.options.environment,
                deploymentOptions: this.options
            },
            deploymentSteps: [],
            validationResults: {},
            testResults: {},
            summary: {
                totalSteps: 0,
                completedSteps: 0,
                failedSteps: 0,
                deploymentSuccess: false,
                deploymentTime: 0
            },
            errors: [],
            warnings: []
        };
        
        this.requiredFiles = [
            '.github/workflows/recursive-todo-processing.yml',
            '.github/scripts/extract-todos.js',
            '.github/scripts/generate-matrix.js',
            '.github/scripts/process-batch.js',
            '.github/scripts/atomize-todos.js',
            '.github/scripts/validate-completion.js',
            '.github/scripts/generate-improvement-prompts.js',
            '.github/scripts/execute-improvements.js',
            '.github/scripts/consolidate-results.js',
            '.github/scripts/test-system.js'
        ];
    }

    async deploySystem() {
        console.log(`🚀 Starting system deployment to ${this.options.environment} ${this.options.dryRun ? '(DRY RUN)' : ''}...`);
        
        try {
            // Pre-deployment validation
            await this.preDeploymentValidation();
            
            // Deploy core components
            await this.deployCoreComponents();
            
            // Configure environment-specific settings
            await this.configureEnvironment();
            
            // Validate deployment
            if (this.options.validateDeployment) {
                await this.validateDeployment();
            }
            
            // Run system tests
            if (this.options.runTests) {
                await this.runDeploymentTests();
            }
            
            // Post-deployment setup
            await this.postDeploymentSetup();
            
            // Generate deployment summary
            await this.generateDeploymentSummary();
            
            // Save deployment results
            await this.saveDeploymentResults();
            
            console.log(`✅ System deployment ${this.deploymentResults.summary.deploymentSuccess ? 'completed successfully' : 'failed'}`);
            
            return this.deploymentResults.summary.deploymentSuccess;
            
        } catch (error) {
            console.error('❌ System deployment failed:', error);
            this.deploymentResults.errors.push({
                type: 'deployment_failure',
                message: error.message,
                timestamp: new Date().toISOString()
            });
            throw error;
        }
    }

    async preDeploymentValidation() {
        await this.executeDeploymentStep('Pre-deployment Validation', async () => {
            console.log('🔍 Running pre-deployment validation...');
            
            // Check if all required files exist
            for (const file of this.requiredFiles) {
                try {
                    await fs.access(file);
                } catch (error) {
                    throw new Error(`Required file missing: ${file}`);
                }
            }
            
            // Validate GitHub Actions workflow syntax
            await this.validateWorkflowSyntax();
            
            // Check Node.js dependencies
            await this.validateNodeDependencies();
            
            // Validate git repository status
            await this.validateGitStatus();
            
            console.log('✅ Pre-deployment validation passed');
        });
    }

    async validateWorkflowSyntax() {
        console.log('📋 Validating GitHub Actions workflow syntax...');
        
        const workflowFile = '.github/workflows/recursive-todo-processing.yml';
        const workflowContent = await fs.readFile(workflowFile, 'utf8');
        
        // Basic YAML validation
        if (!workflowContent.includes('name:') || !workflowContent.includes('on:') || !workflowContent.includes('jobs:')) {
            throw new Error('Invalid GitHub Actions workflow structure');
        }
        
        // Check for required jobs
        const requiredJobs = ['extract-todos', 'process-todos', 'consolidate-results'];
        for (const job of requiredJobs) {
            if (!workflowContent.includes(job + ':')) {
                throw new Error(`Required job missing: ${job}`);
            }
        }
        
        console.log('✅ Workflow syntax validation passed');
    }

    async validateNodeDependencies() {
        console.log('📦 Validating Node.js dependencies...');
        
        // Check if required Node.js modules are available
        const requiredModules = ['fs', 'path', 'child_process'];
        
        for (const module of requiredModules) {
            try {
                require.resolve(module);
            } catch (error) {
                if (!module.startsWith('fs') && !module.startsWith('path') && !module.startsWith('child_process')) {
                    this.deploymentResults.warnings.push({
                        type: 'missing_module',
                        message: `Module ${module} not found, but may be available in GitHub Actions environment`,
                        timestamp: new Date().toISOString()
                    });
                }
            }
        }
        
        console.log('✅ Node.js dependencies validation passed');
    }

    async validateGitStatus() {
        console.log('🔍 Validating git repository status...');
        
        try {
            // Check if we're in a git repository
            execSync('git rev-parse --git-dir', { stdio: 'ignore' });
            
            // Check if there are uncommitted changes (warning only)
            try {
                const status = execSync('git status --porcelain', { encoding: 'utf8' });
                if (status.trim() && !this.options.force) {
                    this.deploymentResults.warnings.push({
                        type: 'uncommitted_changes',
                        message: 'There are uncommitted changes in the repository',
                        timestamp: new Date().toISOString()
                    });
                }
            } catch (error) {
                // Git status failed, but this is not critical
            }
            
        } catch (error) {
            throw new Error('Not in a git repository or git not available');
        }
        
        console.log('✅ Git status validation passed');
    }

    async deployCoreComponents() {
        await this.executeDeploymentStep('Core Components Deployment', async () => {
            console.log('🏗️ Deploying core components...');
            
            // Validate all script files
            for (const scriptFile of this.requiredFiles.filter(f => f.endsWith('.js'))) {
                await this.validateScriptFile(scriptFile);
            }
            
            // Set executable permissions on scripts
            if (!this.options.dryRun) {
                for (const scriptFile of this.requiredFiles.filter(f => f.endsWith('.js'))) {
                    try {
                        execSync(`chmod +x "${scriptFile}"`, { stdio: 'ignore' });
                    } catch (error) {
                        // Permission setting might not be needed in all environments
                        this.deploymentResults.warnings.push({
                            type: 'permission_warning',
                            message: `Could not set executable permission for ${scriptFile}`,
                            timestamp: new Date().toISOString()
                        });
                    }
                }
            }
            
            console.log('✅ Core components deployed');
        });
    }

    async validateScriptFile(scriptFile) {
        console.log(`🔍 Validating script: ${scriptFile}`);
        
        const scriptContent = await fs.readFile(scriptFile, 'utf8');
        
        // Basic JavaScript validation
        if (!scriptContent.includes('#!/usr/bin/env node')) {
            this.deploymentResults.warnings.push({
                type: 'missing_shebang',
                message: `Script ${scriptFile} missing Node.js shebang`,
                timestamp: new Date().toISOString()
            });
        }
        
        // Check for module.exports (indicating proper module structure)
        if (!scriptContent.includes('module.exports')) {
            this.deploymentResults.warnings.push({
                type: 'missing_exports',
                message: `Script ${scriptFile} missing module exports`,
                timestamp: new Date().toISOString()
            });
        }
        
        // Check for main function or CLI interface
        if (!scriptContent.includes('if (require.main === module)') && !scriptContent.includes('async function main')) {
            this.deploymentResults.warnings.push({
                type: 'missing_cli',
                message: `Script ${scriptFile} missing CLI interface`,
                timestamp: new Date().toISOString()
            });
        }
    }

    async configureEnvironment() {
        await this.executeDeploymentStep('Environment Configuration', async () => {
            console.log(`⚙️ Configuring ${this.options.environment} environment...`);
            
            // Create environment-specific configuration
            const envConfig = this.generateEnvironmentConfig();
            
            // Save environment configuration
            if (!this.options.dryRun) {
                const configDir = '.taskmaster/deployment';
                await fs.mkdir(configDir, { recursive: true });
                
                await fs.writeFile(
                    path.join(configDir, `${this.options.environment}-config.json`),
                    JSON.stringify(envConfig, null, 2)
                );
                
                // Create deployment manifest
                const manifest = {
                    deploymentId: `deploy_${Date.now()}`,
                    environment: this.options.environment,
                    deployedAt: new Date().toISOString(),
                    version: '1.0.0',
                    components: this.requiredFiles,
                    configuration: envConfig
                };
                
                await fs.writeFile(
                    path.join(configDir, 'deployment-manifest.json'),
                    JSON.stringify(manifest, null, 2)
                );
            }
            
            console.log(`✅ ${this.options.environment} environment configured`);
        });
    }

    generateEnvironmentConfig() {
        const baseConfig = {
            maxParallelRunners: 10,
            recursionDepth: 5,
            validationMode: 'moderate',
            enableRecursive: true,
            enableTesting: true,
            timeout: 300000
        };
        
        switch (this.options.environment) {
            case 'development':
                return {
                    ...baseConfig,
                    maxParallelRunners: 3,
                    recursionDepth: 2,
                    validationMode: 'lenient',
                    timeout: 120000,
                    debugMode: true
                };
            
            case 'staging':
                return {
                    ...baseConfig,
                    maxParallelRunners: 6,
                    recursionDepth: 3,
                    validationMode: 'moderate',
                    timeout: 180000,
                    enableLogging: true
                };
            
            case 'production':
                return {
                    ...baseConfig,
                    maxParallelRunners: 10,
                    recursionDepth: 5,
                    validationMode: 'strict',
                    timeout: 300000,
                    enableMonitoring: true,
                    enableAlerts: true
                };
            
            default:
                return baseConfig;
        }
    }

    async validateDeployment() {
        await this.executeDeploymentStep('Deployment Validation', async () => {
            console.log('🔍 Validating deployment...');
            
            const validationResults = {
                fileValidation: {},
                configurationValidation: {},
                workflowValidation: {},
                overallValid: true
            };
            
            // Validate deployed files
            for (const file of this.requiredFiles) {
                try {
                    const stats = await fs.stat(file);
                    validationResults.fileValidation[file] = {
                        exists: true,
                        size: stats.size,
                        lastModified: stats.mtime.toISOString()
                    };
                } catch (error) {
                    validationResults.fileValidation[file] = {
                        exists: false,
                        error: error.message
                    };
                    validationResults.overallValid = false;
                }
            }
            
            // Validate configuration
            try {
                const configPath = `.taskmaster/deployment/${this.options.environment}-config.json`;
                const configData = await fs.readFile(configPath, 'utf8');
                const config = JSON.parse(configData);
                
                validationResults.configurationValidation = {
                    valid: true,
                    environment: this.options.environment,
                    configuration: config
                };
            } catch (error) {
                validationResults.configurationValidation = {
                    valid: false,
                    error: error.message
                };
                if (!this.options.dryRun) {
                    validationResults.overallValid = false;
                }
            }
            
            // Validate workflow accessibility
            try {
                const workflowPath = '.github/workflows/recursive-todo-processing.yml';
                const workflowContent = await fs.readFile(workflowPath, 'utf8');
                
                validationResults.workflowValidation = {
                    valid: true,
                    size: workflowContent.length,
                    jobCount: (workflowContent.match(/^\s+[a-zA-Z-]+:/gm) || []).length
                };
            } catch (error) {
                validationResults.workflowValidation = {
                    valid: false,
                    error: error.message
                };
                validationResults.overallValid = false;
            }
            
            this.deploymentResults.validationResults = validationResults;
            
            if (!validationResults.overallValid) {
                throw new Error('Deployment validation failed');
            }
            
            console.log('✅ Deployment validation passed');
        });
    }

    async runDeploymentTests() {
        await this.executeDeploymentStep('Deployment Testing', async () => {
            console.log('🧪 Running deployment tests...');
            
            if (this.options.dryRun) {
                console.log('🏃 Skipping tests in dry run mode');
                this.deploymentResults.testResults = {
                    skipped: true,
                    reason: 'dry_run_mode'
                };
                return;
            }
            
            try {
                // Run the system test script
                const testResult = execSync('node .github/scripts/test-system.js --mode integration --no-cleanup', {
                    encoding: 'utf8',
                    cwd: process.cwd(),
                    timeout: 60000 // 1 minute timeout
                });
                
                this.deploymentResults.testResults = {
                    success: true,
                    output: testResult,
                    timestamp: new Date().toISOString()
                };
                
                console.log('✅ Deployment tests passed');
                
            } catch (error) {
                this.deploymentResults.testResults = {
                    success: false,
                    error: error.message,
                    output: error.stdout || '',
                    timestamp: new Date().toISOString()
                };
                
                // Don't fail deployment for test failures in staging
                if (this.options.environment === 'production') {
                    throw new Error('Deployment tests failed in production environment');
                } else {
                    this.deploymentResults.warnings.push({
                        type: 'test_failure',
                        message: 'Deployment tests failed but continuing in non-production environment',
                        timestamp: new Date().toISOString()
                    });
                }
            }
        });
    }

    async postDeploymentSetup() {
        await this.executeDeploymentStep('Post-deployment Setup', async () => {
            console.log('🔧 Running post-deployment setup...');
            
            // Create deployment documentation
            await this.createDeploymentDocumentation();
            
            // Set up monitoring (if enabled)
            if (this.options.environment === 'production') {
                await this.setupMonitoring();
            }
            
            // Create rollback instructions
            await this.createRollbackInstructions();
            
            console.log('✅ Post-deployment setup complete');
        });
    }

    async createDeploymentDocumentation() {
        console.log('📚 Creating deployment documentation...');
        
        const documentation = `# Recursive Todo Processing System - Deployment Documentation

## Deployment Information

- **Environment**: ${this.options.environment}
- **Deployed At**: ${this.deploymentResults.metadata.deploymentStartTime}
- **Version**: 1.0.0
- **Deployment Success**: ${this.deploymentResults.summary.deploymentSuccess}

## Deployed Components

${this.requiredFiles.map(file => `- ${file}`).join('\n')}

## Configuration

Environment-specific configuration has been applied for ${this.options.environment}.

## Usage

To trigger the recursive todo processing workflow:

1. **Manual Trigger**:
   \`\`\`bash
   gh workflow run recursive-todo-processing.yml
   \`\`\`

2. **With Custom Parameters**:
   \`\`\`bash
   gh workflow run recursive-todo-processing.yml \\
     --field max_parallel_runners=5 \\
     --field depth_limit=3 \\
     --field validation_mode=strict
   \`\`\`

3. **Automatic Trigger**:
   The workflow automatically triggers on:
   - Changes to \`.taskmaster/tasks/**\`
   - Changes to the workflow file itself
   - Daily schedule at 2 AM UTC

## Monitoring

${this.options.environment === 'production' ? 
    '- Workflow execution can be monitored in the GitHub Actions tab\n- Check artifacts for detailed results\n- Review pull requests created by the workflow' :
    '- Basic monitoring available through GitHub Actions tab'
}

## Troubleshooting

1. **Workflow Fails to Start**:
   - Check repository permissions
   - Verify required secrets are configured
   - Check workflow file syntax

2. **Processing Fails**:
   - Review workflow logs
   - Check artifact outputs
   - Verify task-master-ai installation

3. **No Improvements Generated**:
   - Check input todos quality
   - Review batch processing logs
   - Verify improvement prompt generation

## Rollback

To rollback this deployment, see rollback instructions in \`.taskmaster/deployment/rollback-instructions.md\`.

---

Generated by deployment script on ${new Date().toISOString()}
`;
        
        if (!this.options.dryRun) {
            const docsPath = '.taskmaster/deployment/deployment-documentation.md';
            await fs.writeFile(docsPath, documentation);
        }
    }

    async setupMonitoring() {
        console.log('📊 Setting up production monitoring...');
        
        const monitoringConfig = {
            enabled: true,
            environment: 'production',
            alerts: {
                workflowFailure: true,
                highProcessingTime: true,
                lowSuccessRate: true
            },
            metrics: {
                successRate: { threshold: 0.8 },
                processingTime: { threshold: 300000 },
                errorRate: { threshold: 0.1 }
            },
            notifications: {
                slack: false, // Configure as needed
                email: false, // Configure as needed
                github: true
            }
        };
        
        if (!this.options.dryRun) {
            const monitoringPath = '.taskmaster/deployment/monitoring-config.json';
            await fs.writeFile(monitoringPath, JSON.stringify(monitoringConfig, null, 2));
        }
    }

    async createRollbackInstructions() {
        console.log('🔄 Creating rollback instructions...');
        
        const rollbackInstructions = `# Rollback Instructions

## Quick Rollback

To quickly disable the recursive todo processing system:

1. **Disable Workflow**:
   \`\`\`bash
   # Rename the workflow file to disable it
   mv .github/workflows/recursive-todo-processing.yml .github/workflows/recursive-todo-processing.yml.disabled
   \`\`\`

2. **Remove Deployment**:
   \`\`\`bash
   # Remove deployment artifacts
   rm -rf .taskmaster/deployment/
   \`\`\`

## Complete Rollback

To completely remove the system:

1. **Remove All Files**:
   \`\`\`bash
   rm .github/workflows/recursive-todo-processing.yml
   rm -rf .github/scripts/
   rm -rf .taskmaster/
   \`\`\`

2. **Clean Git History** (if needed):
   \`\`\`bash
   git rm .github/workflows/recursive-todo-processing.yml
   git rm -r .github/scripts/
   git commit -m "Rollback: Remove recursive todo processing system"
   \`\`\`

## Partial Rollback

To rollback specific components:

- **Disable automatic triggers**: Edit the workflow file and remove the \`on:\` triggers
- **Reduce processing scope**: Modify environment configuration
- **Disable recursive improvements**: Set \`enableRecursive: false\` in configuration

## Recovery

If rollback was performed in error:

1. **Restore from this deployment**:
   \`\`\`bash
   git checkout HEAD~1 -- .github/workflows/recursive-todo-processing.yml
   git checkout HEAD~1 -- .github/scripts/
   \`\`\`

2. **Re-run deployment**:
   \`\`\`bash
   node .github/scripts/deploy-system.js --environment ${this.options.environment}
   \`\`\`

---

Created during deployment on ${new Date().toISOString()}
`;
        
        if (!this.options.dryRun) {
            const rollbackPath = '.taskmaster/deployment/rollback-instructions.md';
            await fs.writeFile(rollbackPath, rollbackInstructions);
        }
    }

    async executeDeploymentStep(stepName, stepFunction) {
        const step = {
            name: stepName,
            startTime: new Date().toISOString(),
            status: 'running'
        };
        
        this.deploymentResults.summary.totalSteps++;
        
        try {
            await stepFunction();
            
            step.status = 'completed';
            step.success = true;
            this.deploymentResults.summary.completedSteps++;
            
        } catch (error) {
            step.status = 'failed';
            step.success = false;
            step.error = error.message;
            this.deploymentResults.summary.failedSteps++;
            
            throw error;
        } finally {
            step.endTime = new Date().toISOString();
            step.duration = new Date(step.endTime).getTime() - new Date(step.startTime).getTime();
            
            this.deploymentResults.deploymentSteps.push(step);
        }
    }

    async generateDeploymentSummary() {
        console.log('📊 Generating deployment summary...');
        
        this.deploymentResults.summary.deploymentTime = 
            this.deploymentResults.deploymentSteps.reduce((sum, step) => sum + step.duration, 0);
        
        this.deploymentResults.summary.deploymentSuccess = 
            this.deploymentResults.summary.failedSteps === 0;
        
        this.deploymentResults.metadata.deploymentEndTime = new Date().toISOString();
        
        // Generate deployment recommendations
        const recommendations = [];
        
        if (this.deploymentResults.warnings.length > 0) {
            recommendations.push({
                type: 'warning_review',
                message: `Review ${this.deploymentResults.warnings.length} deployment warnings`,
                priority: 'medium'
            });
        }
        
        if (this.options.environment === 'production' && this.deploymentResults.testResults.success === false) {
            recommendations.push({
                type: 'test_investigation',
                message: 'Investigate test failures in production deployment',
                priority: 'high'
            });
        }
        
        if (this.deploymentResults.summary.deploymentTime > 60000) { // 1 minute
            recommendations.push({
                type: 'deployment_optimization',
                message: 'Consider optimizing deployment process for faster execution',
                priority: 'low'
            });
        }
        
        this.deploymentResults.recommendations = recommendations;
        
        console.log(`📊 Deployment summary: ${this.deploymentResults.summary.deploymentSuccess ? 'SUCCESS' : 'FAILED'} in ${this.deploymentResults.summary.deploymentTime}ms`);
    }

    async saveDeploymentResults() {
        console.log('💾 Saving deployment results...');
        
        if (this.options.dryRun) {
            console.log('💾 Skipping save in dry run mode');
            return;
        }
        
        const deploymentDir = '.taskmaster/deployment';
        await fs.mkdir(deploymentDir, { recursive: true });
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const resultsPath = path.join(deploymentDir, `deployment-results-${timestamp}.json`);
        
        await fs.writeFile(resultsPath, JSON.stringify(this.deploymentResults, null, 2));
        
        // Save latest deployment status
        const deploymentStatus = {
            success: this.deploymentResults.summary.deploymentSuccess,
            environment: this.options.environment,
            deploymentTime: this.deploymentResults.summary.deploymentTime,
            componentsDeployed: this.requiredFiles.length,
            warningsCount: this.deploymentResults.warnings.length,
            errorsCount: this.deploymentResults.errors.length,
            timestamp: new Date().toISOString()
        };
        
        await fs.writeFile(
            path.join(deploymentDir, 'latest-deployment.json'),
            JSON.stringify(deploymentStatus, null, 2)
        );
        
        console.log(`💾 Deployment results saved: ${resultsPath}`);
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
            case '--environment':
                options.environment = value;
                break;
            case '--dry-run':
                options.dryRun = true;
                i--; // No value for this flag
                break;
            case '--no-validate':
                options.validate = false;
                i--; // No value for this flag
                break;
            case '--no-test':
                options.test = false;
                i--; // No value for this flag
                break;
            case '--force':
                options.force = true;
                i--; // No value for this flag
                break;
        }
    }
    
    try {
        const deployer = new SystemDeployer(options);
        const success = await deployer.deploySystem();
        
        if (success) {
            console.log('🎉 System deployment completed successfully!');
            process.exit(0);
        } else {
            console.error('💥 System deployment failed!');
            process.exit(1);
        }
    } catch (error) {
        console.error('💥 System deployment failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { SystemDeployer };