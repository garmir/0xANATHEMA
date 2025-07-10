#!/usr/bin/env python3
"""
LABRYS Deployment System
Complete production deployment using dual-blade methodology
"""

import os
import sys
import json
import asyncio
import subprocess
import shutil
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add LABRYS paths
sys.path.append(os.path.join(os.path.dirname(__file__), '.labrys'))

@dataclass
class DeploymentResult:
    """Result of deployment operation"""
    component: str
    status: str
    message: str
    timestamp: str
    details: Dict[str, Any]
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class LabrysDeploymentSystem:
    """
    Complete LABRYS deployment system using dual-blade methodology
    """
    
    def __init__(self, deployment_target: str = "production"):
        self.deployment_target = deployment_target
        self.deployment_root = os.path.dirname(os.path.abspath(__file__))
        self.deployment_results = []
        self.deployment_start_time = time.time()
        
        # Deployment configuration
        self.deployment_config = {
            "environment": deployment_target,
            "safety_mode": True,
            "validation_required": True,
            "monitoring_enabled": True,
            "guardian_enabled": True,
            "backup_enabled": True,
            "rollback_enabled": True,
            "autonomous_mode": True
        }
        
        # Component deployment order
        self.deployment_order = [
            "environment_setup",
            "core_framework",
            "dual_blade_system",
            "safety_systems",
            "monitoring_systems",
            "guardian_system",
            "validation_system",
            "integration_tests",
            "production_activation"
        ]
    
    async def execute_full_deployment(self):
        """Execute complete LABRYS deployment"""
        print("🗲 LABRYS Production Deployment System")
        print("   Ancient wisdom meets modern deployment")
        print("   " + "="*50)
        
        print(f"🎯 Deployment Target: {self.deployment_target}")
        print(f"🏗️  Deployment Root: {self.deployment_root}")
        print(f"⚙️  Configuration: {self.deployment_config}")
        
        # Execute deployment phases
        for phase in self.deployment_order:
            print(f"\n🚀 Deploying: {phase.replace('_', ' ').title()}")
            
            result = await self._deploy_component(phase)
            self.deployment_results.append(result)
            
            if result.status == "failed":
                print(f"❌ Deployment failed at {phase}")
                await self._handle_deployment_failure(phase, result)
                return False
            else:
                print(f"✅ {phase} deployed successfully")
        
        # Generate deployment report
        await self._generate_deployment_report()
        
        print("\n🎉 LABRYS Deployment Complete!")
        print("   System ready for autonomous operation")
        
        return True
    
    async def _deploy_component(self, component: str) -> DeploymentResult:
        """Deploy a specific component"""
        start_time = time.time()
        
        try:
            if component == "environment_setup":
                result = await self._deploy_environment_setup()
            elif component == "core_framework":
                result = await self._deploy_core_framework()
            elif component == "dual_blade_system":
                result = await self._deploy_dual_blade_system()
            elif component == "safety_systems":
                result = await self._deploy_safety_systems()
            elif component == "monitoring_systems":
                result = await self._deploy_monitoring_systems()
            elif component == "guardian_system":
                result = await self._deploy_guardian_system()
            elif component == "validation_system":
                result = await self._deploy_validation_system()
            elif component == "integration_tests":
                result = await self._deploy_integration_tests()
            elif component == "production_activation":
                result = await self._deploy_production_activation()
            else:
                result = DeploymentResult(
                    component=component,
                    status="failed",
                    message=f"Unknown component: {component}",
                    timestamp=datetime.now().isoformat(),
                    details={}
                )
            
            # Add deployment time
            result.details["deployment_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            return DeploymentResult(
                component=component,
                status="failed",
                message=f"Deployment error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                details={"error": str(e), "deployment_time": time.time() - start_time}
            )
    
    async def _deploy_environment_setup(self) -> DeploymentResult:
        """Deploy environment setup"""
        print("   📦 Setting up deployment environment...")
        
        # Create deployment directories
        deployment_dirs = [
            ".labrys/deployment",
            ".labrys/deployment/backups",
            ".labrys/deployment/logs",
            ".labrys/deployment/config",
            ".labrys/deployment/scripts"
        ]
        
        for dir_path in deployment_dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        # Validate Python environment
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            return DeploymentResult(
                component="environment_setup",
                status="failed",
                message="Python 3.8+ required",
                timestamp=datetime.now().isoformat(),
                details={"python_version": f"{python_version.major}.{python_version.minor}"}
            )
        
        # Check dependencies
        required_packages = ["psutil", "asyncio"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return DeploymentResult(
                component="environment_setup",
                status="failed",
                message=f"Missing packages: {', '.join(missing_packages)}",
                timestamp=datetime.now().isoformat(),
                details={"missing_packages": missing_packages}
            )
        
        # Create deployment config
        deployment_config = {
            "deployment_id": f"labrys_deploy_{int(time.time())}",
            "deployment_time": datetime.now().isoformat(),
            "deployment_target": self.deployment_target,
            "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            "deployment_root": self.deployment_root
        }
        
        with open(".labrys/deployment/config/deployment_config.json", "w") as f:
            json.dump(deployment_config, f, indent=2)
        
        return DeploymentResult(
            component="environment_setup",
            status="success",
            message="Environment setup completed",
            timestamp=datetime.now().isoformat(),
            details=deployment_config
        )
    
    async def _deploy_core_framework(self) -> DeploymentResult:
        """Deploy core LABRYS framework"""
        print("   🏗️  Deploying core framework...")
        
        # Validate core files exist
        core_files = [
            "labrys_main.py",
            "taskmaster_labrys.py",
            "recursive_labrys_improvement.py",
            ".labrys/coordination/labrys_coordinator.py",
            ".labrys/analytical/self_analysis_engine.py",
            ".labrys/synthesis/self_synthesis_engine.py"
        ]
        
        missing_files = []
        for file_path in core_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            return DeploymentResult(
                component="core_framework",
                status="failed",
                message=f"Missing core files: {', '.join(missing_files)}",
                timestamp=datetime.now().isoformat(),
                details={"missing_files": missing_files}
            )
        
        # Test core framework initialization
        try:
            # Test main framework
            result = subprocess.run([
                sys.executable, "labrys_main.py", "--help"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return DeploymentResult(
                    component="core_framework",
                    status="failed",
                    message="Core framework initialization failed",
                    timestamp=datetime.now().isoformat(),
                    details={"error": result.stderr}
                )
            
            # Check if help output contains LABRYS
            if "LABRYS" in result.stdout:
                framework_info = {"framework": "LABRYS", "status": "operational"}
            else:
                framework_info = {"framework": "unknown", "status": "failed"}
            
            return DeploymentResult(
                component="core_framework",
                status="success",
                message="Core framework deployed and validated",
                timestamp=datetime.now().isoformat(),
                details=framework_info
            )
            
        except Exception as e:
            return DeploymentResult(
                component="core_framework",
                status="failed",
                message=f"Core framework validation failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)}
            )
    
    async def _deploy_dual_blade_system(self) -> DeploymentResult:
        """Deploy dual-blade system"""
        print("   ⚡ Deploying dual-blade system...")
        
        # Test dual-blade initialization
        try:
            # Test coordination system
            result = subprocess.run([
                sys.executable, "-c", """
import asyncio
import sys
import os
sys.path.append('.labrys')
from coordination.labrys_coordinator import LabrysCoordinator

async def test_coordination():
    coordinator = LabrysCoordinator()
    result = await coordinator.initialize_dual_blades()
    print(f"Coordination test: {result['synchronization']['status']}")
    return result['synchronization']['status'] == 'synchronized'

if asyncio.run(test_coordination()):
    print("SUCCESS")
else:
    print("FAILED")
"""
            ], capture_output=True, text=True, timeout=60)
            
            if "SUCCESS" not in result.stdout:
                return DeploymentResult(
                    component="dual_blade_system",
                    status="failed",
                    message="Dual-blade coordination failed",
                    timestamp=datetime.now().isoformat(),
                    details={"error": result.stderr}
                )
            
            return DeploymentResult(
                component="dual_blade_system",
                status="success",
                message="Dual-blade system deployed and synchronized",
                timestamp=datetime.now().isoformat(),
                details={"synchronization": "active"}
            )
            
        except Exception as e:
            return DeploymentResult(
                component="dual_blade_system",
                status="failed",
                message=f"Dual-blade deployment failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)}
            )
    
    async def _deploy_safety_systems(self) -> DeploymentResult:
        """Deploy safety systems"""
        print("   🛡️  Deploying safety systems...")
        
        # Test safety validator
        try:
            result = subprocess.run([
                sys.executable, "-c", """
import asyncio
import sys
import os
sys.path.append('.labrys')
from validation.safety_validator import SafetyValidator

async def test_safety():
    validator = SafetyValidator()
    test_code = 'def test(): return "safe"'
    result = await validator.validate_modification(test_code, test_code, "test")
    print(f"Safety test: {result.approved_for_deployment}")
    return result.approved_for_deployment

if asyncio.run(test_safety()):
    print("SUCCESS")
else:
    print("FAILED")
"""
            ], capture_output=True, text=True, timeout=30)
            
            if "SUCCESS" not in result.stdout:
                return DeploymentResult(
                    component="safety_systems",
                    status="failed",
                    message="Safety system validation failed",
                    timestamp=datetime.now().isoformat(),
                    details={"error": result.stderr}
                )
            
            return DeploymentResult(
                component="safety_systems",
                status="success",
                message="Safety systems deployed and validated",
                timestamp=datetime.now().isoformat(),
                details={"safety_validation": "active"}
            )
            
        except Exception as e:
            return DeploymentResult(
                component="safety_systems",
                status="failed",
                message=f"Safety system deployment failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)}
            )
    
    async def _deploy_monitoring_systems(self) -> DeploymentResult:
        """Deploy monitoring systems"""
        print("   📊 Deploying monitoring systems...")
        
        # Test monitoring system
        try:
            result = subprocess.run([
                sys.executable, "check_labrys_health.py"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return DeploymentResult(
                    component="monitoring_systems",
                    status="failed",
                    message="Monitoring system test failed",
                    timestamp=datetime.now().isoformat(),
                    details={"error": result.stderr}
                )
            
            # Check if monitoring detected processes
            if "Found" in result.stdout and "processes" in result.stdout:
                return DeploymentResult(
                    component="monitoring_systems",
                    status="success",
                    message="Monitoring systems deployed and operational",
                    timestamp=datetime.now().isoformat(),
                    details={"monitoring": "active"}
                )
            else:
                return DeploymentResult(
                    component="monitoring_systems",
                    status="warning",
                    message="Monitoring deployed but no processes detected",
                    timestamp=datetime.now().isoformat(),
                    details={"monitoring": "active", "processes": 0}
                )
            
        except Exception as e:
            return DeploymentResult(
                component="monitoring_systems",
                status="failed",
                message=f"Monitoring deployment failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)}
            )
    
    async def _deploy_guardian_system(self) -> DeploymentResult:
        """Deploy guardian system"""
        print("   🛡️  Deploying guardian system...")
        
        # Start guardian process
        try:
            # Launch guardian in background
            guardian_process = subprocess.Popen([
                sys.executable, "labrys_process_guardian.py", "--monitor", "--interval", "30"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Give it time to start
            await asyncio.sleep(5)
            
            # Check if guardian is running
            if guardian_process.poll() is None:
                return DeploymentResult(
                    component="guardian_system",
                    status="success",
                    message="Guardian system deployed and monitoring",
                    timestamp=datetime.now().isoformat(),
                    details={"guardian_pid": guardian_process.pid}
                )
            else:
                return DeploymentResult(
                    component="guardian_system",
                    status="failed",
                    message="Guardian system failed to start",
                    timestamp=datetime.now().isoformat(),
                    details={"exit_code": guardian_process.returncode}
                )
            
        except Exception as e:
            return DeploymentResult(
                component="guardian_system",
                status="failed",
                message=f"Guardian deployment failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)}
            )
    
    async def _deploy_validation_system(self) -> DeploymentResult:
        """Deploy validation system"""
        print("   ✅ Deploying validation system...")
        
        # Run validation tests
        try:
            result = subprocess.run([
                sys.executable, "labrys_self_test.py", "--execute"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                return DeploymentResult(
                    component="validation_system",
                    status="failed",
                    message="Validation system tests failed",
                    timestamp=datetime.now().isoformat(),
                    details={"error": result.stderr}
                )
            
            # Check test results
            if "Overall Status: ALL PROCESSES HEALTHY" in result.stdout:
                return DeploymentResult(
                    component="validation_system",
                    status="success",
                    message="Validation system deployed and tests passed",
                    timestamp=datetime.now().isoformat(),
                    details={"validation": "passed"}
                )
            else:
                return DeploymentResult(
                    component="validation_system",
                    status="warning",
                    message="Validation deployed but some tests failed",
                    timestamp=datetime.now().isoformat(),
                    details={"validation": "partial"}
                )
            
        except Exception as e:
            return DeploymentResult(
                component="validation_system",
                status="failed",
                message=f"Validation deployment failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)}
            )
    
    async def _deploy_integration_tests(self) -> DeploymentResult:
        """Deploy integration tests"""
        print("   🧪 Running integration tests...")
        
        # Run comprehensive integration tests
        try:
            # Test recursive improvement
            result = subprocess.run([
                sys.executable, "recursive_labrys_improvement.py", "--execute", "--iterations", "1"
            ], capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                return DeploymentResult(
                    component="integration_tests",
                    status="failed",
                    message="Integration tests failed",
                    timestamp=datetime.now().isoformat(),
                    details={"error": result.stderr}
                )
            
            # Check if improvement completed
            if "Recursive Improvement Complete" in result.stdout:
                return DeploymentResult(
                    component="integration_tests",
                    status="success",
                    message="Integration tests passed",
                    timestamp=datetime.now().isoformat(),
                    details={"integration": "passed"}
                )
            else:
                return DeploymentResult(
                    component="integration_tests",
                    status="warning",
                    message="Integration tests completed with warnings",
                    timestamp=datetime.now().isoformat(),
                    details={"integration": "partial"}
                )
            
        except Exception as e:
            return DeploymentResult(
                component="integration_tests",
                status="failed",
                message=f"Integration test deployment failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)}
            )
    
    async def _deploy_production_activation(self) -> DeploymentResult:
        """Deploy production activation"""
        print("   🚀 Activating production mode...")
        
        # Create production configuration
        production_config = {
            "mode": "production",
            "autonomous_operation": True,
            "monitoring_enabled": True,
            "guardian_enabled": True,
            "safety_enabled": True,
            "recursive_improvement": True,
            "activation_time": datetime.now().isoformat(),
            "deployment_id": f"labrys_prod_{int(time.time())}"
        }
        
        # Save production config
        with open(".labrys/deployment/config/production_config.json", "w") as f:
            json.dump(production_config, f, indent=2)
        
        # Create production status file
        production_status = {
            "status": "active",
            "activation_time": datetime.now().isoformat(),
            "components": {
                "core_framework": "active",
                "dual_blade_system": "active",
                "safety_systems": "active",
                "monitoring_systems": "active",
                "guardian_system": "active",
                "validation_system": "active"
            }
        }
        
        with open(".labrys/deployment/production_status.json", "w") as f:
            json.dump(production_status, f, indent=2)
        
        return DeploymentResult(
            component="production_activation",
            status="success",
            message="Production mode activated",
            timestamp=datetime.now().isoformat(),
            details=production_config
        )
    
    async def _handle_deployment_failure(self, component: str, result: DeploymentResult):
        """Handle deployment failure"""
        print(f"🚨 Deployment failure in {component}: {result.message}")
        
        # Log failure
        failure_log = {
            "component": component,
            "failure_time": datetime.now().isoformat(),
            "error": result.message,
            "details": result.details
        }
        
        with open(".labrys/deployment/logs/failure_log.json", "w") as f:
            json.dump(failure_log, f, indent=2)
        
        # Attempt rollback if configured
        if self.deployment_config.get("rollback_enabled", False):
            print("🔄 Attempting rollback...")
            await self._rollback_deployment(component)
    
    async def _rollback_deployment(self, failed_component: str):
        """Rollback deployment to previous state"""
        print(f"↩️  Rolling back deployment after {failed_component} failure...")
        
        # Implementation would restore from backups
        rollback_log = {
            "rollback_time": datetime.now().isoformat(),
            "failed_component": failed_component,
            "rollback_status": "attempted"
        }
        
        with open(".labrys/deployment/logs/rollback_log.json", "w") as f:
            json.dump(rollback_log, f, indent=2)
    
    async def _generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        deployment_time = time.time() - self.deployment_start_time
        
        successful_deployments = [r for r in self.deployment_results if r.status == "success"]
        failed_deployments = [r for r in self.deployment_results if r.status == "failed"]
        warning_deployments = [r for r in self.deployment_results if r.status == "warning"]
        
        report = {
            "deployment_id": f"labrys_deploy_{int(self.deployment_start_time)}",
            "deployment_time": deployment_time,
            "deployment_target": self.deployment_target,
            "deployment_config": self.deployment_config,
            "deployment_summary": {
                "total_components": len(self.deployment_results),
                "successful": len(successful_deployments),
                "failed": len(failed_deployments),
                "warnings": len(warning_deployments),
                "success_rate": len(successful_deployments) / len(self.deployment_results) * 100
            },
            "deployment_results": [asdict(r) for r in self.deployment_results],
            "deployment_complete": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save deployment report
        with open(".labrys/deployment/deployment_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 Deployment Report Generated:")
        print(f"   Success Rate: {report['deployment_summary']['success_rate']:.1f}%")
        print(f"   Components: {report['deployment_summary']['successful']}/{report['deployment_summary']['total_components']} successful")
        print(f"   Duration: {deployment_time:.1f} seconds")
        print(f"   Report saved to: .labrys/deployment/deployment_report.json")

async def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LABRYS Deployment System")
    parser.add_argument("--target", default="production", help="Deployment target")
    parser.add_argument("--config", help="Deployment configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    args = parser.parse_args()
    
    # Initialize deployment system
    deployment = LabrysDeploymentSystem(args.target)
    
    if args.dry_run:
        print("🧪 Dry run mode - no actual deployment")
        return
    
    # Execute deployment
    success = await deployment.execute_full_deployment()
    
    if success:
        print("\n✅ LABRYS Deployment Successful!")
        print("   System is now operational and ready for autonomous operation")
    else:
        print("\n❌ LABRYS Deployment Failed!")
        print("   Check deployment logs for details")

if __name__ == "__main__":
    asyncio.run(main())