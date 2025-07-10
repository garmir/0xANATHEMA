#!/usr/bin/env python3
"""
LABRYS Environment Setup Script
Configures the environment for LABRYS framework
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup LABRYS environment"""
    print("🗲 LABRYS Environment Setup")
    print("   Configuring Double-Edged AI Development Framework")
    print("   " + "=" * 45)
    
    # Check if .env file exists
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("\n📋 Creating .env file from template...")
            
            # Read template
            with open(env_example, 'r') as f:
                template_content = f.read()
            
            # Get API key from user
            api_key = input("\n🔑 Enter your Perplexity API key: ").strip()
            
            if api_key:
                # Replace placeholder with actual key
                env_content = template_content.replace("your_perplexity_api_key_here", api_key)
                
                # Write .env file
                with open(env_file, 'w') as f:
                    f.write(env_content)
                
                print("✅ .env file created successfully!")
            else:
                print("❌ No API key provided. Please add it to .env manually.")
        else:
            print("❌ .env.example not found. Creating basic .env file...")
            
            # Create basic .env file
            api_key = input("\n🔑 Enter your Perplexity API key: ").strip()
            
            basic_env = f"""# LABRYS Environment Configuration
PERPLEXITY_API_KEY={api_key}
LABRYS_MODE=development
ANALYTICAL_BLADE_ACTIVE=true
SYNTHESIS_BLADE_ACTIVE=true
COORDINATION_ENABLED=true
LOG_LEVEL=info
"""
            
            with open(env_file, 'w') as f:
                f.write(basic_env)
            
            print("✅ Basic .env file created!")
    else:
        print("✅ .env file already exists")
    
    # Check directory structure
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        ".labrys",
        ".labrys/analytical",
        ".labrys/synthesis",
        ".labrys/validation",
        ".labrys/coordination",
        ".labrys/backups",
        ".labrys/synthesis/generated"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created {dir_path}")
        else:
            print(f"   ✅ {dir_path} exists")
    
    # Check Python dependencies
    print("\n📦 Checking Python dependencies...")
    
    try:
        import dotenv
        print("   ✅ python-dotenv available")
    except ImportError:
        print("   ❌ python-dotenv not installed")
        print("   💡 Run: pip install python-dotenv")
    
    # Check for optional dependencies
    optional_deps = [
        ("requests", "HTTP requests"),
        ("aiohttp", "Async HTTP"),
        ("pydantic", "Data validation"),
        ("click", "CLI framework"),
        ("rich", "Rich text and beautiful formatting")
    ]
    
    for dep, desc in optional_deps:
        try:
            __import__(dep)
            print(f"   ✅ {dep} available ({desc})")
        except ImportError:
            print(f"   ⚠️  {dep} not installed ({desc})")
    
    print("\n🎯 Environment setup complete!")
    print("\nNext steps:")
    print("1. Add your Perplexity API key to GitHub Secrets as 'PERPLEXITY_API_KEY'")
    print("2. Run: python3 labrys_main.py --initialize")
    print("3. Test: python3 labrys_self_test.py --execute")
    print("4. Explore: python3 labrys_main.py --interactive")

if __name__ == "__main__":
    setup_environment()