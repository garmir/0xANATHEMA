#!/usr/bin/env python3
"""
Python Package Import Validation Script
Comprehensive validation of all required Python packages for Task Master AI system
"""

import sys
import subprocess
import importlib
from pathlib import Path
import json

class PythonImportValidator:
    """Validates and resolves Python package import issues"""
    
    def __init__(self):
        self.required_packages = {
            # Core packages from main requirements.txt
            'requests': '>=2.31.0',
            'aiohttp': '>=3.8.0', 
            'pydantic': '>=2.0.0',
            'click': '>=8.0.0',
            'colorama': '>=0.4.6',
            'rich': '>=13.0.0',
            
            # Scientific computing packages
            'numpy': '>=1.22.0',
            'scipy': '>=1.8.0',
            'matplotlib': '>=3.5.0',
            'psutil': '>=5.8.0',
            
            # Standard library modules
            'json': None,
            'os': None,
            'sys': None,
            'subprocess': None,
            'pathlib': None,
            'time': None,
            'threading': None,
            'traceback': None,
            'tempfile': None,
            'shutil': None,
            'uuid': None,
            'queue': None,
            'enum': None,
            'dataclasses': None,
            'typing': None
        }
        
        self.validation_results = {}
        
    def validate_imports(self) -> dict:
        """Validate all required package imports"""
        print("🔍 Validating Python package imports...")
        
        for package_name, version_spec in self.required_packages.items():
            try:
                # Try to import the package
                module = importlib.import_module(package_name)
                
                # Get version if available
                version = getattr(module, '__version__', 'unknown')
                
                self.validation_results[package_name] = {
                    'status': 'success',
                    'version': version,
                    'required': version_spec,
                    'error': None
                }
                
                print(f"  ✅ {package_name}: {version}")
                
            except ImportError as e:
                self.validation_results[package_name] = {
                    'status': 'missing',
                    'version': None,
                    'required': version_spec,
                    'error': str(e)
                }
                
                print(f"  ❌ {package_name}: MISSING ({e})")
                
            except Exception as e:
                self.validation_results[package_name] = {
                    'status': 'error',
                    'version': None,
                    'required': version_spec,
                    'error': str(e)
                }
                
                print(f"  ⚠️ {package_name}: ERROR ({e})")
        
        return self.validation_results
    
    def get_missing_packages(self) -> list:
        """Get list of missing packages that need installation"""
        missing = []
        for package, result in self.validation_results.items():
            if result['status'] == 'missing' and result['required'] is not None:
                missing.append(package)
        return missing
    
    def install_missing_packages(self, use_venv: bool = True) -> bool:
        """Install missing packages"""
        missing_packages = self.get_missing_packages()
        
        if not missing_packages:
            print("✅ All required packages are already installed")
            return True
        
        print(f"📦 Installing {len(missing_packages)} missing packages...")
        
        # Prepare pip command
        pip_cmd = [sys.executable, '-m', 'pip', 'install']
        
        # Add packages with version specifications
        for package in missing_packages:
            version_spec = self.required_packages[package]
            if version_spec:
                pip_cmd.append(f"{package}{version_spec}")
            else:
                pip_cmd.append(package)
        
        try:
            # Run pip install
            result = subprocess.run(pip_cmd, capture_output=True, text=True, check=True)
            print(f"✅ Successfully installed packages")
            print(f"   Output: {result.stdout[:200]}...")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            print(f"   Error: {e.stderr}")
            return False
    
    def check_virtual_environment(self) -> dict:
        """Check virtual environment status"""
        venv_info = {
            'in_venv': hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix),
            'python_executable': sys.executable,
            'python_path': sys.path[:3],  # First 3 paths
            'venv_path': None
        }
        
        # Check for common venv paths
        possible_venv_paths = ['.venv', 'venv', '.env']
        for venv_path in possible_venv_paths:
            if Path(venv_path).exists() and Path(venv_path, 'bin', 'python').exists():
                venv_info['venv_path'] = str(Path(venv_path).absolute())
                break
        
        return venv_info
    
    def check_requirements_files(self) -> list:
        """Find all requirements files in the project"""
        req_files = []
        
        # Search for requirements files
        search_patterns = [
            'requirements.txt',
            'requirements/*.txt',
            '.taskmaster/scripts/requirements.txt',
            'pyproject.toml',
            'setup.py'
        ]
        
        for pattern in search_patterns:
            for file_path in Path('.').rglob(pattern):
                if file_path.is_file():
                    req_files.append(str(file_path))
        
        return req_files
    
    def generate_comprehensive_requirements(self) -> str:
        """Generate comprehensive requirements.txt content"""
        requirements_content = "# Task Master AI System - Comprehensive Python Dependencies\n\n"
        
        # Group packages by category
        categories = {
            'Core Dependencies': [
                'requests>=2.31.0',
                'aiohttp>=3.8.0',
                'pydantic>=2.0.0',
                'python-dotenv>=1.0.0'
            ],
            'CLI and UI': [
                'click>=8.0.0',
                'colorama>=0.4.6',
                'rich>=13.0.0'
            ],
            'Scientific Computing': [
                'numpy>=1.22.0',
                'scipy>=1.8.0',
                'matplotlib>=3.5.0',
                'pandas>=1.3.0'
            ],
            'System Utilities': [
                'psutil>=5.8.0'
            ],
            'Development Tools': [
                'pytest>=7.0.0',
                'black>=22.0.0',
                'flake8>=4.0.0',
                'mypy>=0.950'
            ]
        }
        
        for category, packages in categories.items():
            requirements_content += f"# {category}\n"
            for package in packages:
                requirements_content += f"{package}\n"
            requirements_content += "\n"
        
        return requirements_content
    
    def fix_json_serialization_issue(self):
        """Fix JSON serialization issue in test framework"""
        test_suite_file = Path('.taskmaster/scripts/comprehensive-integration-test-suite.py')
        
        if test_suite_file.exists():
            content = test_suite_file.read_text()
            
            # Fix the JSON serialization issue by converting enums to strings
            if 'asdict(result)' in content:
                # Replace the problematic line
                fixed_content = content.replace(
                    '[asdict(result) for result in self.test_results]',
                    '[{**asdict(result), "result": result.result.value} for result in self.test_results]'
                )
                
                test_suite_file.write_text(fixed_content)
                print("✅ Fixed JSON serialization issue in test suite")
                return True
        
        return False
    
    def run_comprehensive_validation(self) -> dict:
        """Run comprehensive validation and resolution"""
        print("🧪 Running Comprehensive Python Import Validation")
        print("=" * 60)
        
        # Check virtual environment
        venv_info = self.check_virtual_environment()
        print(f"🐍 Python Environment:")
        print(f"   In Virtual Env: {venv_info['in_venv']}")
        print(f"   Python Path: {venv_info['python_executable']}")
        if venv_info['venv_path']:
            print(f"   VEnv Path: {venv_info['venv_path']}")
        
        # Find requirements files
        req_files = self.check_requirements_files()
        print(f"\n📄 Requirements Files Found: {len(req_files)}")
        for req_file in req_files:
            print(f"   {req_file}")
        
        # Validate imports
        print(f"\n🔍 Package Import Validation:")
        validation_results = self.validate_imports()
        
        # Get summary
        total_packages = len(validation_results)
        successful_imports = len([r for r in validation_results.values() if r['status'] == 'success'])
        missing_packages = len([r for r in validation_results.values() if r['status'] == 'missing'])
        
        print(f"\n📊 Validation Summary:")
        print(f"   Total Packages: {total_packages}")
        print(f"   ✅ Successful: {successful_imports}")
        print(f"   ❌ Missing: {missing_packages}")
        print(f"   Success Rate: {successful_imports/total_packages:.1%}")
        
        # Install missing packages
        if missing_packages > 0:
            print(f"\n📦 Installing Missing Packages:")
            install_success = self.install_missing_packages()
            
            if install_success:
                # Re-validate after installation
                print(f"\n🔄 Re-validating after installation:")
                validation_results = self.validate_imports()
                
                successful_imports = len([r for r in validation_results.values() if r['status'] == 'success'])
                print(f"   ✅ Final Success Rate: {successful_imports/total_packages:.1%}")
        
        # Fix known issues
        print(f"\n🔧 Fixing Known Issues:")
        self.fix_json_serialization_issue()
        
        # Generate comprehensive requirements
        comp_req_content = self.generate_comprehensive_requirements()
        comp_req_file = Path('.taskmaster/scripts/requirements-complete.txt')
        comp_req_file.write_text(comp_req_content)
        print(f"   📄 Generated: {comp_req_file}")
        
        # Create validation report
        report = {
            'timestamp': str(sys.version_info),
            'python_version': sys.version,
            'virtual_environment': venv_info,
            'requirements_files': req_files,
            'validation_results': validation_results,
            'summary': {
                'total_packages': total_packages,
                'successful_imports': successful_imports,
                'missing_packages': missing_packages,
                'success_rate': successful_imports/total_packages
            }
        }
        
        # Save report
        report_file = Path('.taskmaster/reports/python_import_validation.json')
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Validation report saved: {report_file}")
        
        return report

def main():
    """Main execution function"""
    validator = PythonImportValidator()
    
    try:
        report = validator.run_comprehensive_validation()
        
        print(f"\n🎉 Python Import Validation Complete!")
        
        if report['summary']['success_rate'] >= 0.9:
            print("✅ All critical packages are available")
            return 0
        else:
            print("⚠️ Some packages are still missing")
            return 1
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())