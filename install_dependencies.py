#!/usr/bin/env python3
import subprocess
import sys
import os

def install_requirements():
    requirements = [
        'requests>=2.31.0',
        'python-dotenv>=1.0.0', 
        'aiohttp>=3.8.0',
        'psutil>=5.8.0',
        'gitpython>=3.1.0',
        'pytest>=7.0.0'
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', req])
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"⚠️ Could not install {req}")
        except Exception as e:
            print(f"❌ Error installing {req}: {e}")

if __name__ == "__main__":
    install_requirements()
