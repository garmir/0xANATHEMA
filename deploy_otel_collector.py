#!/usr/bin/env python3
"""
OpenTelemetry Collector Deployment Script for Task-Master
Implements atomic task: Deploy and configure OpenTelemetry Collector

Based on research-driven breakdown:
- Deploy collector as agent/gateway based on architecture
- Configure for MELT data routing to backends
- Set up batching and compression policies
- Ensure proper collector deployment and data routing
"""

import os
import sys
import subprocess
import yaml
import logging
import shutil
import platform
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import tempfile
import tarfile
import zipfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OtelCollectorDeployer:
    """OpenTelemetry Collector deployment and configuration manager"""
    
    def __init__(self, config_path: str = "otel_collector_config.yaml"):
        self.config_path = Path(config_path)
        self.collector_dir = Path(".taskmaster/otel")
        self.logs_dir = Path(".taskmaster/logs")
        
        # Collector download info
        self.otel_version = "0.91.0"
        self.base_url = f"https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v{self.otel_version}"
        
        # Platform-specific binary names
        self.platform_info = self._get_platform_info()
        
    def _get_platform_info(self) -> Dict[str, str]:
        """Get platform-specific information for binary download"""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Map Python platform names to OpenTelemetry naming
        if system == "darwin":
            os_name = "darwin"
            if machine in ["x86_64", "amd64"]:
                arch = "amd64"
            elif machine in ["arm64", "aarch64"]:
                arch = "arm64"
            else:
                arch = "amd64"  # Default fallback
        elif system == "linux":
            os_name = "linux"
            if machine in ["x86_64", "amd64"]:
                arch = "amd64"
            elif machine in ["arm64", "aarch64"]:
                arch = "arm64"
            else:
                arch = "amd64"  # Default fallback
        elif system == "windows":
            os_name = "windows"
            arch = "amd64"
        else:
            os_name = "linux"  # Default fallback
            arch = "amd64"
        
        return {
            "os": os_name,
            "arch": arch,
            "extension": ".exe" if os_name == "windows" else ""
        }
    
    def setup_directories(self):
        """Create necessary directories for collector deployment"""
        logger.info("Setting up directories for OpenTelemetry Collector...")
        
        directories = [
            self.collector_dir,
            self.logs_dir,
            self.collector_dir / "config",
            self.collector_dir / "data",
            Path(".taskmaster/logs/otel")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def download_collector(self) -> Path:
        """Download OpenTelemetry Collector binary"""
        logger.info(f"Downloading OpenTelemetry Collector v{self.otel_version}...")
        
        # Construct download URL
        filename = f"otelcol_{self.otel_version}_{self.platform_info['os']}_{self.platform_info['arch']}"
        if self.platform_info['os'] == "windows":
            archive_name = f"{filename}.zip"
        else:
            archive_name = f"{filename}.tar.gz"
        
        download_url = f"{self.base_url}/{archive_name}"
        
        # Download to temporary location
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / archive_name
            
            logger.info(f"Downloading from: {download_url}")
            
            try:
                response = requests.get(download_url, stream=True)
                response.raise_for_status()
                
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded {temp_path.stat().st_size} bytes")
                
                # Extract binary
                binary_path = self._extract_collector(temp_path)
                return binary_path
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download collector: {e}")
                # Fallback: try to use system-installed otelcol
                return self._find_system_collector()
    
    def _extract_collector(self, archive_path: Path) -> Path:
        """Extract collector binary from downloaded archive"""
        logger.info(f"Extracting collector from {archive_path}...")
        
        binary_name = f"otelcol{self.platform_info['extension']}"
        target_path = self.collector_dir / binary_name
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    # Extract binary
                    for member in zip_ref.namelist():
                        if member.endswith(binary_name):
                            with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                            break
            else:
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    # Extract binary
                    for member in tar_ref.getmembers():
                        if member.name.endswith(binary_name):
                            with tar_ref.extractfile(member) as source, open(target_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                            break
            
            # Make binary executable
            if self.platform_info['os'] != "windows":
                os.chmod(target_path, 0o755)
            
            logger.info(f"Extracted collector binary to: {target_path}")
            return target_path
            
        except Exception as e:
            logger.error(f"Failed to extract collector: {e}")
            return self._find_system_collector()
    
    def _find_system_collector(self) -> Optional[Path]:
        """Try to find system-installed OpenTelemetry Collector"""
        logger.info("Attempting to find system-installed OpenTelemetry Collector...")
        
        possible_names = ["otelcol", "otelcol-contrib", "opentelemetry-collector"]
        
        for name in possible_names:
            try:
                result = subprocess.run(["which", name], capture_output=True, text=True)
                if result.returncode == 0:
                    path = Path(result.stdout.strip())
                    logger.info(f"Found system collector at: {path}")
                    return path
            except Exception:
                continue
        
        logger.warning("No system OpenTelemetry Collector found")
        return None
    
    def validate_config(self) -> bool:
        """Validate OpenTelemetry Collector configuration"""
        logger.info("Validating OpenTelemetry Collector configuration...")
        
        if not self.config_path.exists():
            logger.error(f"Configuration file not found: {self.config_path}")
            return False
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['receivers', 'processors', 'exporters', 'service']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required section in config: {section}")
                    return False
            
            # Validate service pipelines
            if 'pipelines' not in config['service']:
                logger.error("Missing pipelines in service configuration")
                return False
            
            # Check for required pipelines
            pipelines = config['service']['pipelines']
            required_pipelines = ['traces', 'metrics', 'logs']
            for pipeline in required_pipelines:
                if pipeline not in pipelines:
                    logger.warning(f"Missing recommended pipeline: {pipeline}")
            
            logger.info("Configuration validation passed")
            return True
            
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML configuration: {e}")
            return False
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def install_config(self):
        """Install configuration file to collector directory"""
        logger.info("Installing OpenTelemetry Collector configuration...")
        
        config_target = self.collector_dir / "config" / "otel-config.yaml"
        
        # Copy and potentially modify configuration
        with open(self.config_path, 'r') as source:
            config_content = source.read()
        
        # Replace placeholder environment variables
        config_content = config_content.replace(
            "${OBSERVABILITY_API_KEY}", 
            os.getenv("OBSERVABILITY_API_KEY", "your-api-key-here")
        )
        config_content = config_content.replace(
            "${HONEYCOMB_API_KEY}", 
            os.getenv("HONEYCOMB_API_KEY", "your-honeycomb-key-here")
        )
        
        with open(config_target, 'w') as target:
            target.write(config_content)
        
        logger.info(f"Configuration installed to: {config_target}")
        return config_target
    
    def create_systemd_service(self, binary_path: Path, config_path: Path):
        """Create systemd service file for Linux systems"""
        if self.platform_info['os'] != "linux":
            return
        
        logger.info("Creating systemd service for OpenTelemetry Collector...")
        
        service_content = f"""[Unit]
Description=OpenTelemetry Collector for Task-Master
After=network.target

[Service]
Type=simple
User=nobody
Group=nobody
ExecStart={binary_path.absolute()} --config={config_path.absolute()}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=otel-collector

[Install]
WantedBy=multi-user.target
"""
        
        service_path = Path("/tmp/otel-collector.service")
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        logger.info(f"Systemd service file created at: {service_path}")
        logger.info("To install: sudo cp /tmp/otel-collector.service /etc/systemd/system/")
        logger.info("To enable: sudo systemctl enable otel-collector.service")
        logger.info("To start: sudo systemctl start otel-collector.service")
    
    def create_launch_script(self, binary_path: Path, config_path: Path):
        """Create launch script for easy collector startup"""
        logger.info("Creating launch script for OpenTelemetry Collector...")
        
        script_path = self.collector_dir / "start-collector.sh"
        
        script_content = f"""#!/bin/bash
# OpenTelemetry Collector Launch Script for Task-Master
# Generated by deploy_otel_collector.py

set -e

COLLECTOR_BINARY="{binary_path.absolute()}"
CONFIG_PATH="{config_path.absolute()}"
LOG_FILE="{self.logs_dir.absolute()}/otel-collector.log"

echo "Starting OpenTelemetry Collector for Task-Master..."
echo "Binary: $COLLECTOR_BINARY"
echo "Config: $CONFIG_PATH"
echo "Logs: $LOG_FILE"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Start collector
exec "$COLLECTOR_BINARY" \\
    --config="$CONFIG_PATH" \\
    --log-level=info \\
    --log-format=json \\
    2>&1 | tee "$LOG_FILE"
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Launch script created: {script_path}")
        return script_path
    
    def test_collector(self, binary_path: Path, config_path: Path) -> bool:
        """Test collector configuration and basic functionality"""
        logger.info("Testing OpenTelemetry Collector configuration...")
        
        try:
            # Test configuration validation
            result = subprocess.run([
                str(binary_path),
                "--config", str(config_path),
                "--dry-run"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✅ Collector configuration test passed")
                return True
            else:
                logger.error(f"❌ Collector configuration test failed:")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Collector test timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Collector test failed: {e}")
            return False
    
    def deploy(self) -> bool:
        """Execute complete OpenTelemetry Collector deployment"""
        logger.info("🚀 Starting OpenTelemetry Collector deployment for Task-Master")
        logger.info("=" * 60)
        
        try:
            # Step 1: Setup directories
            self.setup_directories()
            
            # Step 2: Validate configuration
            if not self.validate_config():
                logger.error("❌ Configuration validation failed")
                return False
            
            # Step 3: Download collector
            binary_path = self.download_collector()
            if not binary_path or not binary_path.exists():
                logger.error("❌ Failed to obtain OpenTelemetry Collector binary")
                return False
            
            # Step 4: Install configuration
            config_path = self.install_config()
            
            # Step 5: Test collector
            if not self.test_collector(binary_path, config_path):
                logger.error("❌ Collector testing failed")
                return False
            
            # Step 6: Create deployment artifacts
            script_path = self.create_launch_script(binary_path, config_path)
            self.create_systemd_service(binary_path, config_path)
            
            logger.info("\n✅ OpenTelemetry Collector deployment completed successfully!")
            logger.info("📋 Deployment Summary:")
            logger.info(f"   • Binary: {binary_path}")
            logger.info(f"   • Config: {config_path}")
            logger.info(f"   • Launch Script: {script_path}")
            logger.info(f"   • Logs Directory: {self.logs_dir}")
            
            logger.info("\n🚀 Next Steps:")
            logger.info(f"   1. Start collector: {script_path}")
            logger.info("   2. Verify health: curl http://localhost:13133/health")
            logger.info("   3. Check metrics: curl http://localhost:8889/metrics")
            logger.info("   4. View zpages: http://localhost:55679")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False

def main():
    """Main deployment execution"""
    deployer = OtelCollectorDeployer()
    
    try:
        success = deployer.deploy()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()