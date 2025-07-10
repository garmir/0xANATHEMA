# Enhanced Error Handling Utilities
import traceback
import logging
import json
from datetime import datetime
from pathlib import Path

class ErrorHandler:
    """Enhanced error handling with logging and recovery"""
    
    def __init__(self, log_file: str = "system_errors.log"):
        self.log_file = log_file
        self.setup_logging()
        
    def setup_logging(self):
        """Setup error logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, error: Exception, context: str = "", recovery_action: str = ""):
        """Handle error with logging and optional recovery"""
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "recovery_action": recovery_action
        }
        
        # Log error
        self.logger.error(f"Error in {context}: {error}")
        
        # Save error details
        error_file = f"error_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w') as f:
            json.dump(error_info, f, indent=2)
        
        return error_info
    
    def safe_execute(self, func, *args, **kwargs):
        """Safely execute function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_info = self.handle_error(e, f"Function: {func.__name__}")
            return {"error": error_info, "success": False}

# Global error handler instance
error_handler = ErrorHandler()

def safe_import(module_name, fallback_value=None):
    """Safely import module with fallback"""
    try:
        return __import__(module_name)
    except ImportError as e:
        error_handler.handle_error(e, f"Importing {module_name}", f"Using fallback: {fallback_value}")
        return fallback_value

def safe_execute_with_fallback(primary_func, fallback_func, *args, **kwargs):
    """Execute primary function with fallback"""
    try:
        return primary_func(*args, **kwargs)
    except Exception as e:
        error_handler.handle_error(e, f"Primary function: {primary_func.__name__}", "Using fallback function")
        return fallback_func(*args, **kwargs)
