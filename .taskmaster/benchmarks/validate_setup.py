#!/usr/bin/env python3
"""
Setup Validation Script for LLM Benchmarking Framework
======================================================

This script validates the benchmarking framework setup and dependencies.
"""

import sys
import importlib
import json
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("Checking Python version...")
    if sys.version_info < (3.8, 0):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_dependencies():
    """Check required dependencies"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
        'tqdm', 'requests', 'psutil', 'asyncio'
    ]
    
    optional_packages = [
        'torch', 'transformers', 'sentence_transformers',
        'rouge_score', 'nltk', 'plotly', 'dash'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (required)")
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️  {package} (optional)")
            missing_optional.append(package)
    
    return missing_required, missing_optional

def check_configuration():
    """Check configuration file"""
    print("\nChecking configuration...")
    
    config_file = Path("benchmark_config.json")
    if not config_file.exists():
        print("❌ benchmark_config.json not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if "models" not in config:
            print("❌ No models configured")
            return False
        
        print(f"✅ Configuration file valid with {len(config['models'])} models")
        return True
    
    except json.JSONDecodeError:
        print("❌ Invalid JSON in configuration file")
        return False

def check_test_data():
    """Check test data file"""
    print("\nChecking test data...")
    
    test_data_file = Path("test_data.json")
    if not test_data_file.exists():
        print("❌ test_data.json not found")
        return False
    
    try:
        with open(test_data_file, 'r') as f:
            test_data = json.load(f)
        
        expected_capabilities = [
            "recursive_task_breakdown", "multi_step_reasoning", 
            "context_maintenance", "code_generation",
            "research_synthesis", "autonomous_execution", "meta_learning"
        ]
        
        missing_capabilities = []
        for capability in expected_capabilities:
            if capability not in test_data:
                missing_capabilities.append(capability)
        
        if missing_capabilities:
            print(f"❌ Missing test data for: {missing_capabilities}")
            return False
        
        print("✅ Test data file valid with all capabilities")
        return True
    
    except json.JSONDecodeError:
        print("❌ Invalid JSON in test data file")
        return False

def check_framework_import():
    """Check framework import"""
    print("\nChecking framework import...")
    
    try:
        sys.path.insert(0, str(Path(".").absolute()))
        from llm_capability_benchmark import BenchmarkRunner, ModelConfig
        print("✅ Framework imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Framework import failed: {e}")
        return False

def generate_setup_report():
    """Generate setup validation report"""
    print("\n" + "="*50)
    print("SETUP VALIDATION REPORT")
    print("="*50)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Configuration", check_configuration()),
        ("Test Data", check_test_data()),
        ("Framework Import", check_framework_import())
    ]
    
    missing_required, missing_optional = check_dependencies()
    
    all_passed = all(result for _, result in checks) and not missing_required
    
    print(f"\nOverall Status: {'✅ READY' if all_passed else '❌ ISSUES FOUND'}")
    
    if missing_required:
        print(f"\nRequired packages to install:")
        for package in missing_required:
            print(f"  pip install {package}")
    
    if missing_optional:
        print(f"\nOptional packages (recommended):")
        for package in missing_optional:
            print(f"  pip install {package}")
    
    if all_passed:
        print("\n🚀 Your setup is ready for benchmarking!")
        print("\nNext steps:")
        print("1. Configure your models in benchmark_config.json")
        print("2. Run: python example_usage.py")
        print("3. Or run: python llm-capability-benchmark.py --help")
    else:
        print("\n🔧 Please fix the issues above before proceeding.")
    
    return all_passed

def main():
    """Main validation function"""
    print("LLM Capability Benchmarking Framework - Setup Validation")
    print("=" * 60)
    
    result = generate_setup_report()
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()