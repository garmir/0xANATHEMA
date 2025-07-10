# Test Configuration for Robust Testing
import os
import sys
from pathlib import Path

# Add fallback imports at the beginning of test files
def setup_test_environment():
    """Setup robust test environment with fallbacks"""
    
    # Add current directory to Python path
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Import fallback modules
    try:
        exec(open('fallback_imports.py').read())
    except FileNotFoundError:
        pass  # Fallback imports not available
    
    return True

# Test utilities
class RobustTestCase:
    """Base class for robust testing"""
    
    @staticmethod
    def safe_import(module_name, fallback=None):
        """Safely import module with fallback"""
        try:
            return __import__(module_name)
        except ImportError:
            return fallback
    
    @staticmethod
    def skip_if_missing(dependencies):
        """Skip test if dependencies are missing"""
        missing = []
        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        
        if missing:
            return f"Missing dependencies: {missing}"
        return None
