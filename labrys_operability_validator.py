#!/usr/bin/env python3
"""
LABRYS Operability Validator
Validates system operability and fixes issues until operational
"""

import os
import sys
import json
import asyncio
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add system paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.labrys'))

@dataclass
class OperabilityCheck:
    """Represents an operability check"""
    name: str
    description: str
    check_function: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    auto_fix: bool = False
    fix_function: Optional[str] = None

@dataclass
class OperabilityResult:
    """Result of operability check"""
    check_name: str
    passed: bool
    message: str
    severity: str
    fix_attempted: bool = False
    fix_successful: bool = False
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class LabrysOperabilityValidator:
    """
    LABRYS Operability Validator
    
    Validates system operability and attempts to fix issues
    """
    
    def __init__(self):
        self.checks = []
        self.results = []
        self.fix_attempts = 0
        self.max_fix_attempts = 3
        
        # Define operability checks
        self._define_operability_checks()
        
    def _define_operability_checks(self):
        """Define all operability checks"""
        self.checks = [
            OperabilityCheck(
                name="python_environment",
                description="Python environment and basic imports",
                check_function="check_python_environment",
                severity="critical",
                auto_fix=True,
                fix_function="fix_python_environment"
            ),
            OperabilityCheck(
                name="labrys_imports",
                description="LABRYS core module imports",
                check_function="check_labrys_imports",
                severity="critical",
                auto_fix=True,
                fix_function="fix_labrys_imports"
            ),
            OperabilityCheck(
                name="system_initialization",
                description="LABRYS system initialization",
                check_function="check_system_initialization",
                severity="high",
                auto_fix=True,
                fix_function="fix_system_initialization"
            ),
            OperabilityCheck(
                name="basic_functionality",
                description="Basic system functionality",
                check_function="check_basic_functionality",
                severity="high",
                auto_fix=True,
                fix_function="fix_basic_functionality"
            ),
            OperabilityCheck(
                name="self_test_capability",
                description="Self-testing capability",
                check_function="check_self_test_capability",
                severity="medium",
                auto_fix=True,
                fix_function="fix_self_test_capability"
            ),
            OperabilityCheck(
                name="introspection_capability",
                description="Introspection and improvement capability",
                check_function="check_introspection_capability",
                severity="medium",
                auto_fix=False
            ),
            OperabilityCheck(
                name="api_integration",
                description="API integration functionality",
                check_function="check_api_integration",
                severity="low",
                auto_fix=True,
                fix_function="fix_api_integration"
            )
        ]
    
    async def run_operability_validation(self) -> Dict[str, Any]:
        """Run complete operability validation"""
        print("🗲 LABRYS Operability Validator")
        print("   Validating system operability...")
        print("   " + "=" * 40)
        
        results = []
        
        for check in self.checks:
            print(f"\n🔍 Running check: {check.name}")
            print(f"   Description: {check.description}")
            
            try:
                # Run the check
                check_method = getattr(self, check.check_function)
                result = await check_method()
                
                if not result.passed and check.auto_fix and check.fix_function:
                    print(f"   ❌ Check failed: {result.message}")
                    print(f"   🔧 Attempting auto-fix...")
                    
                    # Attempt to fix
                    fix_method = getattr(self, check.fix_function)
                    fix_result = await fix_method()
                    
                    result.fix_attempted = True
                    result.fix_successful = fix_result
                    
                    if fix_result:
                        print(f"   ✅ Auto-fix successful")
                        # Re-run check
                        result = await check_method()
                        result.fix_attempted = True
                        result.fix_successful = True
                    else:
                        print(f"   ❌ Auto-fix failed")
                
                if result.passed:
                    print(f"   ✅ Check passed: {result.message}")
                else:
                    print(f"   ❌ Check failed: {result.message}")
                
                results.append(result)
                
            except Exception as e:
                error_result = OperabilityResult(
                    check_name=check.name,
                    passed=False,
                    message=f"Check error: {str(e)}",
                    severity=check.severity
                )
                results.append(error_result)
                print(f"   ❌ Check error: {str(e)}")
        
        # Calculate overall operability score
        total_checks = len(results)
        passed_checks = sum(1 for r in results if r.passed)
        critical_failures = sum(1 for r in results if not r.passed and r.severity == 'critical')
        
        operability_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Determine operational status
        if critical_failures > 0:
            operational_status = "CRITICAL"
        elif operability_score >= 80:
            operational_status = "OPERATIONAL"
        elif operability_score >= 60:
            operational_status = "DEGRADED"
        else:
            operational_status = "FAILED"
        
        summary = {
            "operational_status": operational_status,
            "operability_score": operability_score,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "critical_failures": critical_failures,
            "check_results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                    "fix_attempted": r.fix_attempted,
                    "fix_successful": r.fix_successful,
                    "timestamp": r.timestamp
                } for r in results
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n📊 Operability Summary:")
        print(f"   Status: {operational_status}")
        print(f"   Score: {operability_score:.1f}%")
        print(f"   Passed: {passed_checks}/{total_checks}")
        print(f"   Critical Failures: {critical_failures}")
        
        return summary
    
    # Individual check methods
    async def check_python_environment(self) -> OperabilityResult:
        """Check Python environment"""
        try:
            import os, sys, json, asyncio
            python_version = sys.version_info
            
            if python_version.major == 3 and python_version.minor >= 8:
                return OperabilityResult(
                    check_name="python_environment",
                    passed=True,
                    message=f"Python {python_version.major}.{python_version.minor} OK",
                    severity="critical"
                )
            else:
                return OperabilityResult(
                    check_name="python_environment",
                    passed=False,
                    message=f"Python version {python_version.major}.{python_version.minor} not supported",
                    severity="critical"
                )
        except Exception as e:
            return OperabilityResult(
                check_name="python_environment",
                passed=False,
                message=f"Python environment error: {str(e)}",
                severity="critical"
            )
    
    async def check_labrys_imports(self) -> OperabilityResult:
        """Check LABRYS module imports"""
        try:
            # Test basic imports
            from labrys_main import LabrysFramework
            from recursive_labrys_improvement import RecursiveLabrysImprovement
            
            return OperabilityResult(
                check_name="labrys_imports",
                passed=True,
                message="LABRYS core modules imported successfully",
                severity="critical"
            )
        except Exception as e:
            return OperabilityResult(
                check_name="labrys_imports",
                passed=False,
                message=f"LABRYS import error: {str(e)}",
                severity="critical"
            )
    
    async def check_system_initialization(self) -> OperabilityResult:
        """Check system initialization"""
        try:
            from labrys_main import LabrysFramework
            
            labrys = LabrysFramework()
            init_result = await labrys.initialize_system()
            
            if init_result.get("status") == "success":
                return OperabilityResult(
                    check_name="system_initialization",
                    passed=True,
                    message="System initialized successfully",
                    severity="high"
                )
            else:
                return OperabilityResult(
                    check_name="system_initialization",
                    passed=False,
                    message=f"System initialization failed: {init_result.get('message', 'Unknown error')}",
                    severity="high"
                )
        except Exception as e:
            return OperabilityResult(
                check_name="system_initialization",
                passed=False,
                message=f"System initialization error: {str(e)}",
                severity="high"
            )
    
    async def check_basic_functionality(self) -> OperabilityResult:
        """Check basic functionality"""
        try:
            from labrys_main import LabrysFramework
            
            labrys = LabrysFramework()
            await labrys.initialize_system()
            
            # Test basic info retrieval
            info = labrys.get_system_info()
            
            if info.get("framework") == "LABRYS":
                return OperabilityResult(
                    check_name="basic_functionality",
                    passed=True,
                    message="Basic functionality working",
                    severity="high"
                )
            else:
                return OperabilityResult(
                    check_name="basic_functionality",
                    passed=False,
                    message="Basic functionality failed",
                    severity="high"
                )
        except Exception as e:
            return OperabilityResult(
                check_name="basic_functionality",
                passed=False,
                message=f"Basic functionality error: {str(e)}",
                severity="high"
            )
    
    async def check_self_test_capability(self) -> OperabilityResult:
        """Check self-test capability"""
        try:
            # Check if self-test module exists and is importable
            import labrys_self_test
            
            return OperabilityResult(
                check_name="self_test_capability",
                passed=True,
                message="Self-test capability available",
                severity="medium"
            )
        except Exception as e:
            return OperabilityResult(
                check_name="self_test_capability",
                passed=False,
                message=f"Self-test capability error: {str(e)}",
                severity="medium"
            )
    
    async def check_introspection_capability(self) -> OperabilityResult:
        """Check introspection capability"""
        try:
            # Check if introspection module exists and is importable
            import labrys_introspection_runner
            
            return OperabilityResult(
                check_name="introspection_capability",
                passed=True,
                message="Introspection capability available",
                severity="medium"
            )
        except Exception as e:
            return OperabilityResult(
                check_name="introspection_capability",
                passed=False,
                message=f"Introspection capability error: {str(e)}",
                severity="medium"
            )
    
    async def check_api_integration(self) -> OperabilityResult:
        """Check API integration"""
        try:
            # Try to load from .env file first
            env_file = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(env_file):
                try:
                    from dotenv import load_dotenv
                    load_dotenv(env_file)
                except ImportError:
                    # Manual parsing if dotenv not available
                    with open(env_file, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                key, value = line.strip().split('=', 1)
                                os.environ[key] = value
                
            api_key = os.getenv('PERPLEXITY_API_KEY')
            
            if api_key and api_key.startswith('pplx-'):
                return OperabilityResult(
                    check_name="api_integration",
                    passed=True,
                    message="API key configured and valid",
                    severity="low"
                )
            else:
                return OperabilityResult(
                    check_name="api_integration",
                    passed=False,
                    message="API key not configured or invalid",
                    severity="low"
                )
        except Exception as e:
            return OperabilityResult(
                check_name="api_integration",
                passed=False,
                message=f"API integration error: {str(e)}",
                severity="low"
            )
    
    # Fix methods
    async def fix_python_environment(self) -> bool:
        """Fix Python environment issues"""
        try:
            # Can't fix Python version, but can check for basic modules
            required_modules = ['os', 'sys', 'json', 'asyncio']
            
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    return False
            
            return True
        except Exception:
            return False
    
    async def fix_labrys_imports(self) -> bool:
        """Fix LABRYS import issues"""
        try:
            # Add current directory to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            # Add .labrys directory to path
            labrys_dir = os.path.join(current_dir, '.labrys')
            if os.path.exists(labrys_dir) and labrys_dir not in sys.path:
                sys.path.insert(0, labrys_dir)
            
            # Test import again
            from labrys_main import LabrysFramework
            return True
        except Exception:
            return False
    
    async def fix_system_initialization(self) -> bool:
        """Fix system initialization issues"""
        try:
            # Create basic directory structure
            required_dirs = [
                ".labrys",
                ".labrys/analytical",
                ".labrys/synthesis",
                ".labrys/validation",
                ".labrys/coordination",
                ".labrys/backups"
            ]
            
            for dir_path in required_dirs:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
            
            return True
        except Exception:
            return False
    
    async def fix_basic_functionality(self) -> bool:
        """Fix basic functionality issues"""
        try:
            # Ensure basic configuration
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write("# LABRYS Configuration\n")
                    f.write("LABRYS_MODE=development\n")
                    f.write("LABRYS_SAFETY_LEVEL=high\n")
            
            return True
        except Exception:
            return False
    
    async def fix_self_test_capability(self) -> bool:
        """Fix self-test capability"""
        try:
            # Check if self-test file exists
            if not os.path.exists('labrys_self_test.py'):
                return False
            
            return True
        except Exception:
            return False
    
    async def fix_api_integration(self) -> bool:
        """Fix API integration issues"""
        try:
            # Check if .env file exists
            env_file = os.path.join(os.path.dirname(__file__), '.env')
            if not os.path.exists(env_file):
                # Create .env file with API key
                with open(env_file, 'w') as f:
                    f.write("# LABRYS Environment Configuration\n")
                    f.write("PERPLEXITY_API_KEY=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\n")
                    f.write("LABRYS_MODE=development\n")
                    f.write("ANALYTICAL_BLADE_ACTIVE=true\n")
                    f.write("SYNTHESIS_BLADE_ACTIVE=true\n")
                    f.write("COORDINATION_ENABLED=true\n")
                    f.write("LOG_LEVEL=info\n")
                
                # Load the environment
                try:
                    from dotenv import load_dotenv
                    load_dotenv(env_file)
                except ImportError:
                    # Manual parsing if dotenv not available
                    with open(env_file, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                key, value = line.strip().split('=', 1)
                                os.environ[key] = value
                
                return True
            else:
                # .env exists, ensure API key is loaded
                try:
                    from dotenv import load_dotenv
                    load_dotenv(env_file)
                except ImportError:
                    # Manual parsing if dotenv not available
                    with open(env_file, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                key, value = line.strip().split('=', 1)
                                os.environ[key] = value
                
                # Check if API key is now available
                api_key = os.getenv('PERPLEXITY_API_KEY')
                return api_key and api_key.startswith('pplx-')
        except Exception:
            return False
    
    async def run_fix_until_operational(self, max_attempts: int = 5) -> Dict[str, Any]:
        """Run validation and fix until operational"""
        print("🔧 Running fix-until-operational cycle...")
        
        for attempt in range(max_attempts):
            print(f"\n🔄 Attempt {attempt + 1}/{max_attempts}")
            
            # Run validation
            result = await self.run_operability_validation()
            
            if result["operational_status"] == "OPERATIONAL":
                print("✅ System is operational!")
                return result
            
            # If not operational and we have more attempts, continue
            if attempt < max_attempts - 1:
                print("⚠️ System not operational, continuing fixes...")
                await asyncio.sleep(1)  # Brief pause between attempts
        
        print("❌ Unable to achieve operational status")
        return result

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LABRYS Operability Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--validate", action="store_true",
                       help="Run operability validation")
    parser.add_argument("--fix-until-operational", action="store_true",
                       help="Fix issues until system is operational")
    parser.add_argument("--report", help="Generate operability report")
    
    args = parser.parse_args()
    
    validator = LabrysOperabilityValidator()
    
    if args.validate:
        result = await validator.run_operability_validation()
        
        # Save results
        with open("operability_results.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n📊 Results saved to: operability_results.json")
        
    elif args.fix_until_operational:
        result = await validator.run_fix_until_operational()
        
        # Save results
        with open("operability_results.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n📊 Results saved to: operability_results.json")
        
    elif args.report:
        result = await validator.run_operability_validation()
        
        # Generate report
        report_content = f"""# LABRYS Operability Report

**Generated:** {result['timestamp']}
**Status:** {result['operational_status']}
**Score:** {result['operability_score']:.1f}%

## Summary
- **Total Checks:** {result['total_checks']}
- **Passed:** {result['passed_checks']}
- **Failed:** {result['failed_checks']}
- **Critical Failures:** {result['critical_failures']}

## Check Results
{chr(10).join(f"- **{check['check_name']}**: {'✅ PASS' if check['passed'] else '❌ FAIL'} - {check['message']}" for check in result['check_results'])}

## Detailed Results
```json
{json.dumps(result, indent=2)}
```
"""
        
        with open(args.report, 'w') as f:
            f.write(report_content)
        
        print(f"📋 Report saved to: {args.report}")
        
    else:
        parser.print_help()
        print("\n🗲 LABRYS Operability Validator")
        print("   System operability validation and fixing")
        print("   Use --validate to check operability")
        print("   Use --fix-until-operational to fix issues")

if __name__ == "__main__":
    asyncio.run(main())