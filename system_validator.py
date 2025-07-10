# System Validator with Fallback Mechanisms
import os
import json
from pathlib import Path
from datetime import datetime

class SystemValidator:
    """Validates system with fallback mechanisms"""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_with_fallbacks(self):
        """Validate system with fallback options"""
        
        validations = [
            ("project_structure", self.validate_project_structure),
            ("python_files", self.validate_python_files),
            ("dependencies", self.validate_dependencies),
            ("configuration", self.validate_configuration)
        ]
        
        results = {}
        
        for validation_name, validation_func in validations:
            try:
                result = validation_func()
                results[validation_name] = {
                    "status": "passed" if result else "failed",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                results[validation_name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Calculate overall health
        passed_validations = len([r for r in results.values() if r["status"] == "passed"])
        total_validations = len(results)
        health_score = (passed_validations / total_validations) * 100
        
        return {
            "validation_results": results,
            "health_score": health_score,
            "overall_status": "healthy" if health_score >= 70 else "degraded",
            "timestamp": datetime.now().isoformat()
        }
    
    def validate_project_structure(self):
        """Validate basic project structure"""
        required_files = ["README.md", "requirements.txt"]
        return all(Path(f).exists() for f in required_files)
    
    def validate_python_files(self):
        """Validate Python files exist"""
        python_files = list(Path('.').glob('*.py'))
        return len(python_files) > 0
    
    def validate_dependencies(self):
        """Validate dependencies"""
        return Path('requirements.txt').exists()
    
    def validate_configuration(self):
        """Validate configuration"""
        return Path('.taskmaster').exists() or Path('.labrys').exists()

if __name__ == "__main__":
    validator = SystemValidator()
    results = validator.validate_with_fallbacks()
    print(json.dumps(results, indent=2))
