#!/usr/bin/env python3
"""
TouchID Sudo Integration - Seamless authentication for autonomous execution on macOS

This module provides TouchID integration for sudo operations during autonomous task execution,
with secure fallback to password authentication and comprehensive security validation.
"""

import os
import sys
import subprocess
import time
import platform
import tempfile
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging


class AuthMethod(Enum):
    """Available authentication methods"""
    TOUCHID = "touchid"
    PASSWORD = "password"
    CACHED = "cached"
    FAILED = "failed"


@dataclass
class AuthResult:
    """Authentication result information"""
    method: AuthMethod
    success: bool
    duration: float
    error_message: Optional[str] = None
    timestamp: str = ""
    session_id: Optional[str] = None


class TouchIDIntegration:
    """
    TouchID integration for seamless sudo authentication during autonomous execution
    """
    
    def __init__(self):
        """Initialize TouchID integration"""
        self.logger = self._setup_logging()
        self.is_macos = platform.system() == "Darwin"
        self.touchid_available = self._check_touchid_availability()
        self.sudo_config_path = "/etc/pam.d/sudo"
        self.auth_cache = {}
        self.session_timeout = 300  # 5 minutes
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for authentication operations"""
        logger = logging.getLogger("touchid_integration")
        logger.setLevel(logging.INFO)
        
        # Create console handler if no handlers exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _check_touchid_availability(self) -> bool:
        """Check if TouchID is available on the system"""
        if not self.is_macos:
            self.logger.info("TouchID not available: Not running on macOS")
            return False
        
        try:
            # Check for biometric authentication capability
            result = subprocess.run(
                ["bioutil", "-r"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                self.logger.info("TouchID hardware detected")
                return True
            else:
                self.logger.warning("TouchID hardware not available")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # bioutil might not be available, try alternative check
            try:
                # Check for Touch ID in system_profiler
                result = subprocess.run(
                    ["system_profiler", "SPiBridgeDataType"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if "Touch ID" in result.stdout:
                    self.logger.info("TouchID detected via system_profiler")
                    return True
                else:
                    self.logger.warning("TouchID not found in system profile")
                    return False
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.logger.warning("Unable to detect TouchID availability")
                return False
    
    def check_pam_configuration(self) -> Dict[str, Any]:
        """Check current PAM configuration for TouchID support"""
        config_status = {
            "touchid_configured": False,
            "pam_tid_present": False,
            "config_readable": False,
            "backup_exists": False,
            "configuration_lines": []
        }
        
        try:
            # Check if sudo PAM config is readable
            with open(self.sudo_config_path, 'r') as f:
                lines = f.readlines()
                config_status["config_readable"] = True
                config_status["configuration_lines"] = [line.strip() for line in lines]
                
                # Check for pam_tid.so configuration
                for line in lines:
                    if "pam_tid.so" in line:
                        config_status["pam_tid_present"] = True
                        if not line.strip().startswith('#'):
                            config_status["touchid_configured"] = True
                        break
                        
        except PermissionError:
            self.logger.warning("Permission denied reading PAM configuration")
        except FileNotFoundError:
            self.logger.error("PAM configuration file not found")
        
        # Check for backup
        backup_path = f"{self.sudo_config_path}.backup"
        if os.path.exists(backup_path):
            config_status["backup_exists"] = True
        
        return config_status
    
    def configure_touchid_sudo(self, dry_run: bool = False) -> bool:
        """
        Configure TouchID for sudo authentication
        
        Args:
            dry_run: If True, only show what would be done without making changes
            
        Returns:
            True if configuration successful or already configured
        """
        if not self.is_macos:
            self.logger.error("TouchID configuration only available on macOS")
            return False
        
        if not self.touchid_available:
            self.logger.error("TouchID hardware not available")
            return False
        
        config_status = self.check_pam_configuration()
        
        if config_status["touchid_configured"]:
            self.logger.info("TouchID already configured for sudo")
            return True
        
        if not config_status["config_readable"]:
            self.logger.error("Cannot read PAM configuration - permission denied")
            return False
        
        if dry_run:
            self.logger.info("DRY RUN: Would configure TouchID for sudo")
            self.logger.info("DRY RUN: Would add 'auth sufficient pam_tid.so' to /etc/pam.d/sudo")
            return True
        
        try:
            # Create backup of original configuration
            backup_path = f"{self.sudo_config_path}.backup"
            if not os.path.exists(backup_path):
                subprocess.run(
                    ["sudo", "cp", self.sudo_config_path, backup_path],
                    check=True
                )
                self.logger.info(f"Created backup: {backup_path}")
            
            # Add TouchID configuration
            touchid_line = "auth       sufficient     pam_tid.so\\n"
            
            # Use sed to insert TouchID configuration after the first line
            cmd = [
                "sudo", "sed", "-i", "", 
                "2i\\",
                touchid_line,
                self.sudo_config_path
            ]
            
            subprocess.run(cmd, check=True)
            self.logger.info("TouchID configuration added to sudo PAM")
            
            # Verify configuration
            new_config = self.check_pam_configuration()
            if new_config["touchid_configured"]:
                self.logger.info("TouchID sudo configuration verified")
                return True
            else:
                self.logger.error("TouchID configuration verification failed")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to configure TouchID: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error configuring TouchID: {e}")
            return False
    
    def restore_sudo_config(self) -> bool:
        """Restore original sudo PAM configuration from backup"""
        if not self.is_macos:
            return False
        
        backup_path = f"{self.sudo_config_path}.backup"
        
        if not os.path.exists(backup_path):
            self.logger.error("No backup configuration found")
            return False
        
        try:
            subprocess.run(
                ["sudo", "cp", backup_path, self.sudo_config_path],
                check=True
            )
            self.logger.info("Sudo PAM configuration restored from backup")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to restore configuration: {e}")
            return False
    
    def test_touchid_authentication(self, timeout: int = 30) -> AuthResult:
        """
        Test TouchID authentication with sudo
        
        Args:
            timeout: Maximum time to wait for authentication
            
        Returns:
            AuthResult with test results
        """
        start_time = time.time()
        
        if not self.is_macos:
            return AuthResult(
                method=AuthMethod.FAILED,
                success=False,
                duration=0,
                error_message="TouchID not available on non-macOS systems",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        
        if not self.touchid_available:
            return AuthResult(
                method=AuthMethod.FAILED,
                success=False,
                duration=0,
                error_message="TouchID hardware not available",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        
        config_status = self.check_pam_configuration()
        if not config_status["touchid_configured"]:
            return AuthResult(
                method=AuthMethod.FAILED,
                success=False,
                duration=0,
                error_message="TouchID not configured for sudo",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        
        try:
            # Test sudo with a harmless command
            process = subprocess.Popen(
                ["sudo", "-n", "whoami"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            duration = time.time() - start_time
            
            if process.returncode == 0:
                # Check if this was cached authentication or fresh TouchID
                if duration < 1.0:
                    method = AuthMethod.CACHED
                else:
                    method = AuthMethod.TOUCHID
                
                return AuthResult(
                    method=method,
                    success=True,
                    duration=duration,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
            else:
                return AuthResult(
                    method=AuthMethod.FAILED,
                    success=False,
                    duration=duration,
                    error_message=stderr.strip(),
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
        except subprocess.TimeoutExpired:
            return AuthResult(
                method=AuthMethod.FAILED,
                success=False,
                duration=timeout,
                error_message="Authentication timeout",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        except Exception as e:
            return AuthResult(
                method=AuthMethod.FAILED,
                success=False,
                duration=time.time() - start_time,
                error_message=str(e),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
    
    def authenticate_sudo(self, command: List[str], timeout: int = 30) -> AuthResult:
        """
        Authenticate and execute sudo command with TouchID fallback
        
        Args:
            command: Command to execute with sudo (without 'sudo' prefix)
            timeout: Maximum time to wait for authentication
            
        Returns:
            AuthResult with execution results
        """
        start_time = time.time()
        
        # Build full sudo command
        full_command = ["sudo"] + command
        
        try:
            # First try: TouchID authentication (if available)
            if self.touchid_available and self.check_pam_configuration()["touchid_configured"]:
                self.logger.info("Attempting TouchID authentication...")
                
                process = subprocess.Popen(
                    full_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    duration = time.time() - start_time
                    
                    if process.returncode == 0:
                        self.logger.info("TouchID authentication successful")
                        return AuthResult(
                            method=AuthMethod.TOUCHID,
                            success=True,
                            duration=duration,
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                        )
                    else:
                        self.logger.warning(f"TouchID authentication failed: {stderr}")
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    self.logger.warning("TouchID authentication timeout")
            
            # Fallback: Password authentication
            self.logger.info("Falling back to password authentication...")
            
            # For autonomous execution, we can't prompt for password interactively
            # Instead, check if sudo is already authenticated or fail gracefully
            process = subprocess.Popen(
                ["sudo", "-n"] + command,  # -n flag prevents password prompt
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            duration = time.time() - start_time
            
            if process.returncode == 0:
                self.logger.info("Cached sudo authentication successful")
                return AuthResult(
                    method=AuthMethod.CACHED,
                    success=True,
                    duration=duration,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
            else:
                self.logger.error("No valid authentication available")
                return AuthResult(
                    method=AuthMethod.FAILED,
                    success=False,
                    duration=duration,
                    error_message="Authentication required but not available in autonomous mode",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
        except Exception as e:
            return AuthResult(
                method=AuthMethod.FAILED,
                success=False,
                duration=time.time() - start_time,
                error_message=str(e),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
    
    def generate_integration_script(self, output_path: str = "touchid-setup.sh") -> str:
        """Generate setup script for TouchID integration"""
        
        script_content = f"""#!/bin/bash
# TouchID Sudo Integration Setup Script
# Generated by Task Master AI

set -e

echo "TouchID Sudo Integration Setup"
echo "=============================="

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: TouchID integration only available on macOS"
    exit 1
fi

# Check for TouchID hardware
echo "Checking for TouchID hardware..."
if bioutil -r &>/dev/null || system_profiler SPiBridgeDataType | grep -q "Touch ID"; then
    echo "✓ TouchID hardware detected"
else
    echo "⚠ TouchID hardware not detected"
    echo "This script will configure PAM but TouchID may not work"
fi

# Check current configuration
echo "Checking current sudo PAM configuration..."
if grep -q "pam_tid.so" /etc/pam.d/sudo; then
    if grep -q "^auth.*sufficient.*pam_tid.so" /etc/pam.d/sudo; then
        echo "✓ TouchID already configured for sudo"
        exit 0
    else
        echo "⚠ pam_tid.so found but not properly configured"
    fi
else
    echo "○ TouchID not yet configured for sudo"
fi

# Create backup
echo "Creating backup of sudo PAM configuration..."
if [ ! -f /etc/pam.d/sudo.backup ]; then
    sudo cp /etc/pam.d/sudo /etc/pam.d/sudo.backup
    echo "✓ Backup created: /etc/pam.d/sudo.backup"
else
    echo "✓ Backup already exists"
fi

# Configure TouchID
echo "Configuring TouchID for sudo..."
sudo sed -i '' '2i\\
auth       sufficient     pam_tid.so
' /etc/pam.d/sudo

# Verify configuration
if grep -q "^auth.*sufficient.*pam_tid.so" /etc/pam.d/sudo; then
    echo "✓ TouchID configuration added successfully"
else
    echo "✗ TouchID configuration failed"
    exit 1
fi

# Test configuration
echo "Testing TouchID configuration..."
echo "Please authenticate with TouchID when prompted..."
if sudo -k && sudo whoami &>/dev/null; then
    echo "✓ TouchID authentication test successful"
else
    echo "⚠ TouchID authentication test failed"
    echo "You may need to authenticate manually the first time"
fi

echo ""
echo "TouchID Sudo Integration Setup Complete!"
echo ""
echo "Usage:"
echo "  - TouchID will now prompt for authentication on sudo commands"
echo "  - Fallback to password authentication if TouchID fails"
echo "  - To restore original configuration: sudo cp /etc/pam.d/sudo.backup /etc/pam.d/sudo"
"""
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(output_path, 0o755)
        self.logger.info(f"TouchID integration script generated: {output_path}")
        
        return output_path
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report for TouchID integration"""
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": {
                "platform": platform.system(),
                "is_macos": self.is_macos,
                "macos_version": platform.mac_ver()[0] if self.is_macos else None
            },
            "touchid": {
                "hardware_available": self.touchid_available,
                "detection_method": "bioutil" if self.touchid_available else "not_detected"
            },
            "configuration": self.check_pam_configuration() if self.is_macos else {},
            "recommendations": []
        }
        
        # Generate recommendations
        if not self.is_macos:
            report["recommendations"].append("TouchID integration only available on macOS")
        elif not self.touchid_available:
            report["recommendations"].append("TouchID hardware not detected - verify device compatibility")
        elif not report["configuration"].get("touchid_configured", False):
            report["recommendations"].append("Run TouchID configuration to enable seamless authentication")
        else:
            report["recommendations"].append("TouchID integration ready for autonomous execution")
        
        return report
    
    def cleanup_integration(self) -> bool:
        """Clean up TouchID integration and restore original configuration"""
        
        if not self.is_macos:
            return True
        
        try:
            # Restore original configuration
            if self.restore_sudo_config():
                self.logger.info("TouchID integration cleanup completed")
                return True
            else:
                self.logger.error("Failed to restore original configuration")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return False


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TouchID Sudo Integration for Task Master AI")
    parser.add_argument("--configure", action="store_true", help="Configure TouchID for sudo")
    parser.add_argument("--test", action="store_true", help="Test TouchID authentication")
    parser.add_argument("--status", action="store_true", help="Show integration status")
    parser.add_argument("--generate-script", action="store_true", help="Generate setup script")
    parser.add_argument("--restore", action="store_true", help="Restore original configuration")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without changes")
    
    args = parser.parse_args()
    
    integration = TouchIDIntegration()
    
    if args.status:
        print("TouchID Integration Status")
        print("=" * 40)
        report = integration.generate_status_report()
        print(json.dumps(report, indent=2))
        
    elif args.configure:
        print("Configuring TouchID for sudo...")
        success = integration.configure_touchid_sudo(dry_run=args.dry_run)
        if success:
            print("✓ TouchID configuration completed successfully")
        else:
            print("✗ TouchID configuration failed")
            sys.exit(1)
            
    elif args.test:
        print("Testing TouchID authentication...")
        result = integration.test_touchid_authentication()
        print(f"Method: {result.method.value}")
        print(f"Success: {result.success}")
        print(f"Duration: {result.duration:.2f}s")
        if result.error_message:
            print(f"Error: {result.error_message}")
            
    elif args.generate_script:
        script_path = integration.generate_integration_script()
        print(f"✓ Setup script generated: {script_path}")
        
    elif args.restore:
        print("Restoring original sudo configuration...")
        success = integration.restore_sudo_config()
        if success:
            print("✓ Original configuration restored")
        else:
            print("✗ Failed to restore configuration")
            sys.exit(1)
            
    else:
        # Default: show status and recommendations
        report = integration.generate_status_report()
        
        print("TouchID Integration for Task Master AI")
        print("=" * 40)
        print(f"macOS System: {report['system']['is_macos']}")
        print(f"TouchID Hardware: {report['touchid']['hardware_available']}")
        
        if report['system']['is_macos']:
            print(f"TouchID Configured: {report['configuration'].get('touchid_configured', False)}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        if integration.touchid_available and not report['configuration'].get('touchid_configured', False):
            print(f"\nTo configure TouchID: python3 {sys.argv[0]} --configure")


if __name__ == "__main__":
    main()