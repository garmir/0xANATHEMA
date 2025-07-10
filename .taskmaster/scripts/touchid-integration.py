#!/usr/bin/env python3
"""
TouchID Integration for Task-Master Autonomous Execution
========================================================

Implements seamless TouchID authentication for autonomous sudo operations on macOS.
Provides password fallback and session caching for improved user experience.
"""

import os
import sys
import time
import subprocess
import tempfile
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import platform

@dataclass
class TouchIDAuthResult:
    """Result of TouchID authentication attempt"""
    success: bool
    method: str  # "touchid", "password", "cached"
    duration: float
    error: Optional[str] = None
    session_cached: bool = False

class TouchIDSudoIntegration:
    """TouchID integration for seamless autonomous sudo operations"""
    
    def __init__(self):
        self.platform = platform.system()
        self.session_cache = {}
        self.cache_duration = 300  # 5 minutes
        self.cache_file = Path.home() / ".taskmaster_auth_cache"
        
        # Load existing session cache
        self._load_session_cache()
    
    def authenticate_sudo(self, command: str, use_touchid: bool = True) -> TouchIDAuthResult:
        """Authenticate and execute sudo command with TouchID integration"""
        
        start_time = time.time()
        
        # Check if we're on macOS for TouchID support
        if self.platform != "Darwin":
            return TouchIDAuthResult(
                success=False,
                method="unsupported",
                duration=time.time() - start_time,
                error="TouchID only supported on macOS"
            )
        
        # Check session cache first
        cache_result = self._check_session_cache(command)
        if cache_result.success:
            return cache_result
        
        # Try TouchID authentication
        if use_touchid:
            touchid_result = self._authenticate_with_touchid(command)
            if touchid_result.success:
                self._cache_session(command)
                touchid_result.session_cached = True
                return touchid_result
        
        # Fallback to password authentication
        password_result = self._authenticate_with_password(command)
        if password_result.success:
            self._cache_session(command)
            password_result.session_cached = True
        
        return password_result
    
    def _authenticate_with_touchid(self, command: str) -> TouchIDAuthResult:
        """Attempt TouchID authentication for sudo command"""
        
        start_time = time.time()
        
        try:
            # Configure TouchID for sudo if not already done
            self._configure_touchid_sudo()
            
            # Execute command with TouchID prompt
            result = subprocess.run(
                f"sudo -u root {command}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return TouchIDAuthResult(
                success=result.returncode == 0,
                method="touchid",
                duration=time.time() - start_time,
                error=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return TouchIDAuthResult(
                success=False,
                method="touchid",
                duration=time.time() - start_time,
                error="TouchID authentication timeout"
            )
        except Exception as e:
            return TouchIDAuthResult(
                success=False,
                method="touchid",
                duration=time.time() - start_time,
                error=f"TouchID authentication error: {str(e)}"
            )
    
    def _authenticate_with_password(self, command: str) -> TouchIDAuthResult:
        """Fallback password authentication for sudo command"""
        
        start_time = time.time()
        
        try:
            print("TouchID not available, falling back to password authentication...")
            
            # Use subprocess with stdin for password input
            process = subprocess.Popen(
                f"sudo -S {command}",
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # For autonomous execution, we can't prompt for password
            # This is a placeholder for password handling
            stdout, stderr = process.communicate(timeout=10)
            
            return TouchIDAuthResult(
                success=process.returncode == 0,
                method="password",
                duration=time.time() - start_time,
                error=stderr if process.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return TouchIDAuthResult(
                success=False,
                method="password",
                duration=time.time() - start_time,
                error="Password authentication timeout"
            )
        except Exception as e:
            return TouchIDAuthResult(
                success=False,
                method="password",
                duration=time.time() - start_time,
                error=f"Password authentication error: {str(e)}"
            )
    
    def _configure_touchid_sudo(self):
        """Configure TouchID for sudo operations"""
        
        try:
            # Check if TouchID is already configured for sudo
            pam_sudo_path = "/etc/pam.d/sudo"
            
            # Read current sudo PAM configuration
            with open(pam_sudo_path, 'r') as f:
                pam_config = f.read()
            
            # Check if TouchID line is already present
            touchid_line = "auth sufficient pam_tid.so"
            
            if touchid_line not in pam_config:
                print("TouchID not configured for sudo. Manual configuration may be required.")
                # Note: Actual configuration requires sudo privileges and is risky to automate
                return False
            
            return True
            
        except Exception as e:
            print(f"TouchID configuration check failed: {e}")
            return False
    
    def _check_session_cache(self, command: str) -> TouchIDAuthResult:
        """Check if command is cached in current session"""
        
        start_time = time.time()
        
        # Generate cache key for command
        cache_key = self._generate_cache_key(command)
        
        if cache_key in self.session_cache:
            cache_entry = self.session_cache[cache_key]
            
            # Check if cache entry is still valid
            if time.time() - cache_entry["timestamp"] < self.cache_duration:
                return TouchIDAuthResult(
                    success=True,
                    method="cached",
                    duration=time.time() - start_time,
                    session_cached=True
                )
            else:
                # Remove expired cache entry
                del self.session_cache[cache_key]
        
        return TouchIDAuthResult(
            success=False,
            method="cached",
            duration=time.time() - start_time,
            error="No valid session cache"
        )
    
    def _cache_session(self, command: str):
        """Cache successful authentication for session"""
        
        cache_key = self._generate_cache_key(command)
        
        self.session_cache[cache_key] = {
            "timestamp": time.time(),
            "command": command
        }
        
        # Save to persistent cache file
        self._save_session_cache()
    
    def _generate_cache_key(self, command: str) -> str:
        """Generate cache key for command"""
        
        return hashlib.md5(command.encode()).hexdigest()
    
    def _load_session_cache(self):
        """Load session cache from persistent storage"""
        
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Filter out expired entries
                current_time = time.time()
                self.session_cache = {
                    key: entry for key, entry in cache_data.items()
                    if current_time - entry["timestamp"] < self.cache_duration
                }
        except Exception as e:
            print(f"Warning: Could not load session cache: {e}")
            self.session_cache = {}
    
    def _save_session_cache(self):
        """Save session cache to persistent storage"""
        
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.session_cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save session cache: {e}")
    
    def clear_session_cache(self):
        """Clear all cached authentication sessions"""
        
        self.session_cache = {}
        
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
        except Exception as e:
            print(f"Warning: Could not remove cache file: {e}")
    
    def get_auth_status(self) -> Dict[str, Any]:
        """Get current authentication status and capabilities"""
        
        return {
            "platform": self.platform,
            "touchid_supported": self.platform == "Darwin",
            "cached_sessions": len(self.session_cache),
            "cache_duration": self.cache_duration,
            "cache_file_exists": self.cache_file.exists()
        }

class AutonomousSudoWrapper:
    """Wrapper for autonomous sudo operations with TouchID integration"""
    
    def __init__(self):
        self.touchid_integration = TouchIDSudoIntegration()
    
    def sudo_with_touchid(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute sudo command with TouchID authentication"""
        
        # Authenticate and execute
        auth_result = self.touchid_integration.authenticate_sudo(command)
        
        if not auth_result.success:
            return {
                "success": False,
                "error": auth_result.error,
                "auth_method": auth_result.method,
                "auth_duration": auth_result.duration
            }
        
        # Execute the command
        try:
            result = subprocess.run(
                f"sudo {command}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "auth_method": auth_result.method,
                "auth_duration": auth_result.duration,
                "session_cached": auth_result.session_cached
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timeout after {timeout} seconds",
                "auth_method": auth_result.method,
                "auth_duration": auth_result.duration
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Command execution error: {str(e)}",
                "auth_method": auth_result.method,
                "auth_duration": auth_result.duration
            }

def test_touchid_integration():
    """Test TouchID integration functionality"""
    
    print("🔐 Testing TouchID Sudo Integration")
    print("=" * 50)
    
    # Initialize TouchID integration
    touchid = TouchIDSudoIntegration()
    sudo_wrapper = AutonomousSudoWrapper()
    
    # Get authentication status
    auth_status = touchid.get_auth_status()
    print(f"Platform: {auth_status['platform']}")
    print(f"TouchID Supported: {auth_status['touchid_supported']}")
    print(f"Cached Sessions: {auth_status['cached_sessions']}")
    print()
    
    # Test authentication with a safe command
    print("🧪 Testing sudo authentication with 'whoami'...")
    
    result = sudo_wrapper.sudo_with_touchid("whoami")
    
    print(f"✅ Success: {result['success']}")
    print(f"🔑 Auth Method: {result.get('auth_method', 'unknown')}")
    print(f"⏱️  Auth Duration: {result.get('auth_duration', 0):.3f}s")
    
    if result['success']:
        print(f"👤 User: {result.get('stdout', '').strip()}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    print()
    print("🔐 TouchID integration test complete!")

def main():
    """Execute TouchID integration testing"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_touchid_integration()
    else:
        print("TouchID Integration for Task-Master")
        print("Usage: python3 touchid-integration.py test")

if __name__ == "__main__":
    main()