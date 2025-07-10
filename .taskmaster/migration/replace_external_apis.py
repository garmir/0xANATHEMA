#!/usr/bin/env python3
"""
Migration Script: Replace External API Dependencies with Local LLM Infrastructure
Automatically updates existing research and planning modules to use local LLMs
"""

import os
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import json

class ExternalAPIReplacer:
    """
    Automated migration tool to replace external API dependencies
    with local LLM infrastructure
    """
    
    def __init__(self, project_root: str = "/Users/anam/archive"):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / ".taskmaster" / "migration" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Files that need migration
        self.files_to_migrate = [
            "hardcoded_research_workflow.py",
            "autonomous_research_integration.py", 
            "autonomous_workflow_loop.py",
            "perplexity_client.py.old"
        ]
        
        # API replacement patterns
        self.replacement_patterns = {
            # Perplexity API imports
            r"import\s+requests": "# import requests  # Replaced with local LLM",
            r"from\s+perplexity.*": "# from perplexity  # Replaced with local LLM adapter",
            
            # Task-master research calls
            r'"task-master",\s*"add-task",\s*"--prompt",.*"--research"': '"task-master", "add-task", "--prompt", prompt_text',
            r'"task-master",\s*"expand",.*"--research"': '"task-master", "expand", "--id", task_id',
            r'"task-master",\s*"update-task",.*"--research"': '"task-master", "update-task", "--id", task_id, "--prompt", update_text',
            
            # Perplexity API calls
            r"PERPLEXITY_API_KEY": "# PERPLEXITY_API_KEY  # No longer needed with local LLM",
            r"https://api\.perplexity\.ai.*": '"http://localhost:11434"  # Local LLM endpoint',
            r"llama-3\.1-sonar-.*": '"llama2"  # Local model name',
            
            # Research function calls
            r"research_with_perplexity\(": "await local_research_replacement(",
            r"autonomous_stuck_handler\(": "await local_autonomous_stuck_handler(",
        }
        
        # Import additions for local LLM support
        self.local_imports = '''
# Local LLM imports for autonomous operation
import asyncio
from pathlib import Path
import sys

# Add local LLM modules to path
sys.path.append(str(Path(__file__).parent / ".taskmaster"))
try:
    from adapters.local_api_adapter import (
        LocalAPIAdapter, 
        replace_perplexity_call,
        replace_task_master_research,
        replace_autonomous_stuck_handler
    )
    from research.local_research_workflow import (
        LocalResearchWorkflow,
        local_autonomous_stuck_handler
    )
    LOCAL_LLM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Local LLM modules not available: {e}")
    LOCAL_LLM_AVAILABLE = False
'''
    
    def backup_file(self, file_path: Path) -> Path:
        """Create backup of original file before migration"""
        if not file_path.exists():
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"{file_path.name}.{timestamp}.backup"
        
        shutil.copy2(file_path, backup_file)
        print(f"📁 Backed up {file_path.name} to {backup_file}")
        return backup_file
    
    def migrate_file(self, file_path: Path) -> bool:
        """Migrate a single file to use local LLM infrastructure"""
        if not file_path.exists():
            print(f"⚠️ File not found: {file_path}")
            return False
        
        print(f"🔄 Migrating {file_path.name}...")
        
        # Create backup
        backup_path = self.backup_file(file_path)
        if not backup_path:
            print(f"❌ Failed to backup {file_path}")
            return False
        
        try:
            # Read original file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply replacements
            modified_content = self.apply_replacements(content, file_path.name)
            
            # Write migrated file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            print(f"✅ Successfully migrated {file_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to migrate {file_path}: {e}")
            # Restore from backup
            if backup_path and backup_path.exists():
                shutil.copy2(backup_path, file_path)
                print(f"🔄 Restored {file_path.name} from backup")
            return False
    
    def apply_replacements(self, content: str, filename: str) -> str:
        """Apply specific replacements based on file content and type"""
        modified_content = content
        
        # Add local imports at the top (after existing imports)
        if "import" in modified_content and "local_api_adapter" not in modified_content:
            # Find last import statement
            import_lines = []
            other_lines = []
            in_imports = True
            
            for line in modified_content.split('\n'):
                if line.strip().startswith(('import ', 'from ')) and in_imports:
                    import_lines.append(line)
                elif line.strip() == '' and in_imports:
                    import_lines.append(line)
                else:
                    in_imports = False
                    other_lines.append(line)
            
            # Insert local imports
            import_lines.append(self.local_imports)
            modified_content = '\n'.join(import_lines + other_lines)
        
        # Apply pattern replacements
        for pattern, replacement in self.replacement_patterns.items():
            modified_content = re.sub(pattern, replacement, modified_content, flags=re.MULTILINE)
        
        # File-specific modifications
        if "research_workflow" in filename:
            modified_content = self.modify_research_workflow(modified_content)
        elif "autonomous" in filename:
            modified_content = self.modify_autonomous_workflow(modified_content)
        elif "perplexity" in filename:
            modified_content = self.modify_perplexity_client(modified_content)
        
        return modified_content
    
    def modify_research_workflow(self, content: str) -> str:
        """Specific modifications for research workflow files"""
        # Replace research function with local version
        research_function_replacement = '''
async def local_research_replacement(query: str, context: str = "") -> str:
    """Local LLM replacement for Perplexity research"""
    if LOCAL_LLM_AVAILABLE:
        try:
            return await replace_perplexity_call(query, context)
        except Exception as e:
            print(f"⚠️ Local research failed: {e}")
            return f"Local research unavailable. Manual research needed for: {query}"
    else:
        return f"Local LLM not available. Manual research needed for: {query}"

async def local_autonomous_stuck_handler(problem: str, context: str = "") -> dict:
    """Local LLM replacement for autonomous stuck handler"""
    if LOCAL_LLM_AVAILABLE:
        try:
            return await replace_autonomous_stuck_handler(problem, context)
        except Exception as e:
            print(f"⚠️ Local stuck handler failed: {e}")
            return {
                "problem": problem,
                "context": context,
                "todo_steps": [f"Manual investigation needed: {problem}"],
                "error": str(e)
            }
    else:
        return {
            "problem": problem,
            "context": context,
            "todo_steps": [f"Local LLM unavailable. Manual research needed: {problem}"]
        }
'''
        
        # Add the replacement functions
        if "local_research_replacement" not in content:
            content = content + "\n" + research_function_replacement
        
        return content
    
    def modify_autonomous_workflow(self, content: str) -> str:
        """Specific modifications for autonomous workflow files"""
        # Make async functions where needed
        content = content.replace(
            "def autonomous_stuck_handler(",
            "async def autonomous_stuck_handler("
        )
        content = content.replace(
            "def research_with_perplexity(",
            "async def research_with_perplexity("
        )
        
        # Add async execution wrapper
        async_wrapper = '''
def run_async_research(coro):
    """Helper to run async research functions"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)
'''
        
        if "run_async_research" not in content:
            content = content + "\n" + async_wrapper
        
        return content
    
    def modify_perplexity_client(self, content: str) -> str:
        """Specific modifications for Perplexity client files"""
        # Replace entire client with local adapter
        local_client_code = '''
# Local LLM Client - Replacement for Perplexity API
import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / ".taskmaster"))
from adapters.local_api_adapter import LocalAPIAdapter

class LocalPerplexityClient:
    """Drop-in replacement for Perplexity client using local LLMs"""
    
    def __init__(self, api_key: str = None):
        # API key not needed for local LLMs
        self.adapter = LocalAPIAdapter()
    
    async def chat_completion(self, messages, model=None, **kwargs):
        """Chat completion using local LLM"""
        return await self.adapter.perplexity_chat_completion(messages, model, **kwargs)
    
    async def research_query(self, query: str, context: str = ""):
        """Research query using local LLM"""
        return await self.adapter.perplexity_research_query(query, context)

# Global client instance
perplexity_client = LocalPerplexityClient()
'''
        
        return local_client_code
    
    def create_local_research_module(self):
        """Create a local research module for backwards compatibility"""
        local_module_path = self.project_root / "local_research_module.py"
        
        module_content = '''
#!/usr/bin/env python3
"""
Local Research Module - Backwards Compatibility Interface
Provides drop-in replacements for external research functions
"""

import asyncio
from pathlib import Path
import sys

# Add taskmaster modules to path
sys.path.append(str(Path(__file__).parent / ".taskmaster"))

try:
    from adapters.local_api_adapter import (
        LocalAPIAdapter,
        replace_perplexity_call,
        replace_autonomous_stuck_handler
    )
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False

class LocalResearchModule:
    """Local research module for backwards compatibility"""
    
    def __init__(self):
        if LOCAL_LLM_AVAILABLE:
            self.adapter = LocalAPIAdapter()
        else:
            self.adapter = None
    
    async def research_query(self, query: str, context: str = "") -> str:
        """Research query using local LLM"""
        if self.adapter:
            return await self.adapter.perplexity_research_query(query, context)
        else:
            return f"Local research unavailable. Manual research needed: {query}"
    
    async def autonomous_stuck_handler(self, problem: str, context: str = "") -> dict:
        """Autonomous stuck handler using local LLM"""
        if self.adapter:
            return await self.adapter.autonomous_stuck_handler_replacement(problem, context)
        else:
            return {
                "problem": problem,
                "todo_steps": [f"Manual investigation needed: {problem}"]
            }

def create_local_perplexity_replacement():
    """Create local Perplexity replacement"""
    return LocalResearchModule()

# Global instance
local_research = LocalResearchModule()

# Backwards compatibility functions
async def research_with_local_llm(query: str, context: str = "") -> str:
    """Backwards compatible research function"""
    return await local_research.research_query(query, context)

async def autonomous_stuck_handler_local(problem: str, context: str = "") -> dict:
    """Backwards compatible stuck handler"""
    return await local_research.autonomous_stuck_handler(problem, context)

def run_sync_research(query: str, context: str = "") -> str:
    """Synchronous wrapper for async research"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(research_with_local_llm(query, context))
'''
        
        with open(local_module_path, 'w') as f:
            f.write(module_content)
        
        print(f"✅ Created local research module: {local_module_path}")
    
    def migrate_all_files(self):
        """Migrate all identified files to use local LLM infrastructure"""
        print("🚀 Starting migration to local LLM infrastructure...")
        
        # Create local research module first
        self.create_local_research_module()
        
        migration_results = []
        
        for filename in self.files_to_migrate:
            file_path = self.project_root / filename
            success = self.migrate_file(file_path)
            migration_results.append((filename, success))
        
        # Summary
        successful = sum(1 for _, success in migration_results if success)
        total = len(migration_results)
        
        print(f"\n📊 Migration Summary:")
        print(f"✅ Successful: {successful}/{total}")
        print(f"❌ Failed: {total - successful}/{total}")
        
        if successful > 0:
            print(f"\n🎯 Next steps:")
            print(f"1. Start local LLM server (e.g., ollama serve)")
            print(f"2. Pull required models (e.g., ollama pull llama2)")
            print(f"3. Test migrated functionality")
            print(f"4. Remove external API keys from environment")
        
        return migration_results
    
    def test_migration(self):
        """Test the migrated functionality"""
        print("🧪 Testing migrated functionality...")
        
        test_script = self.project_root / ".taskmaster" / "migration" / "test_migration.py"
        
        test_content = '''
#!/usr/bin/env python3
"""Test script for migration validation"""

import asyncio
import sys
from pathlib import Path

# Add local modules to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from local_research_module import LocalResearchModule
    print("✅ Local research module import successful")
    
    async def test_research():
        research = LocalResearchModule()
        
        # Test research query
        result = await research.research_query(
            "Test local LLM research functionality", 
            "Migration testing"
        )
        print(f"📝 Research result: {result[:100]}...")
        
        # Test stuck handler
        stuck_result = await research.autonomous_stuck_handler(
            "Test stuck handler migration",
            "Testing local LLM integration"
        )
        print(f"🔧 Stuck handler todos: {len(stuck_result.get('todo_steps', []))}")
        
        return True
    
    # Run test
    asyncio.run(test_research())
    print("✅ Migration test completed successfully")
    
except Exception as e:
    print(f"❌ Migration test failed: {e}")
    sys.exit(1)
'''
        
        test_script.parent.mkdir(parents=True, exist_ok=True)
        with open(test_script, 'w') as f:
            f.write(test_content)
        
        # Run test
        try:
            result = subprocess.run([
                sys.executable, str(test_script)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ Migration test passed")
                print(result.stdout)
            else:
                print("❌ Migration test failed")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("⏰ Migration test timed out")
        except Exception as e:
            print(f"❌ Migration test error: {e}")

def main():
    """Main migration execution"""
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode only
        replacer = ExternalAPIReplacer()
        replacer.test_migration()
    else:
        # Full migration
        replacer = ExternalAPIReplacer()
        results = replacer.migrate_all_files()
        
        # Test if requested
        if "--with-test" in sys.argv:
            replacer.test_migration()

if __name__ == "__main__":
    main()