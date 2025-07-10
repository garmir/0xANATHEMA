
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
