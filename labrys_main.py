#!/usr/bin/env python3
"""
LABRYS - Main Entry Point
Double-Edged AI Development Framework
"""

import os
import sys
import json
import asyncio
import argparse
from typing import Dict, Any, Optional
from datetime import datetime

# Add system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from taskmaster_labrys import TaskMasterLabrys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.labrys'))
from coordination.labrys_coordinator import LabrysCoordinator
from validation.system_validator import SystemValidator

class LabrysFramework:
    """
    Main LABRYS Framework Controller
    """
    
    def __init__(self):
        self.taskmaster = TaskMasterLabrys()
        self.coordinator = LabrysCoordinator()
        self.validator = SystemValidator()
        self.system_initialized = False
        
    async def initialize_system(self) -> Dict[str, Any]:
        """
        Initialize the complete LABRYS system
        """
        print("🗲 Initializing LABRYS Framework...")
        print("   Double-Edged AI Development System")
        print("   " + "="*40)
        
        # Initialize TaskMaster with LABRYS
        init_result = await self.taskmaster.initialize_labrys_system()
        
        if init_result["status"] == "success":
            self.system_initialized = True
            print("✅ LABRYS system initialized successfully")
            return init_result
        else:
            print("❌ LABRYS system initialization failed")
            return init_result
    
    async def run_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive system validation
        """
        print("\n🔍 Running LABRYS System Validation...")
        validation_report = await self.validator.run_comprehensive_validation()
        
        # Display summary
        print(f"\n📊 Validation Results:")
        print(f"   Overall Status: {validation_report['overall_status']}")
        print(f"   Success Rate: {validation_report['success_rate']:.1f}%")
        print(f"   Tests Passed: {validation_report['test_summary']['passed']}")
        print(f"   Tests Failed: {validation_report['test_summary']['failed']}")
        
        return validation_report
    
    async def execute_labrys_tasks(self, tasks_file: str) -> Dict[str, Any]:
        """
        Execute tasks using LABRYS methodology
        """
        if not self.system_initialized:
            await self.initialize_system()
        
        # Load tasks from file
        with open(tasks_file, 'r') as f:
            tasks_json = json.load(f)
        
        # Load and execute tasks
        tasks = self.taskmaster.load_tasks_from_json(tasks_json)
        
        print(f"\n🎯 Executing {len(tasks)} LABRYS tasks...")
        
        result = await self.taskmaster.execute_task_sequence(tasks)
        
        # Display results
        print(f"\n📋 Execution Results:")
        print(f"   Total Tasks: {result['total_tasks']}")
        print(f"   Completed: {result['completed_tasks']}")
        print(f"   Failed: {result['failed_tasks']}")
        
        return result
    
    async def run_interactive_mode(self):
        """
        Run LABRYS in interactive mode
        """
        if not self.system_initialized:
            await self.initialize_system()
        
        print("\n🎮 LABRYS Interactive Mode")
        print("   Type 'help' for available commands")
        print("   Type 'quit' to exit")
        
        while True:
            try:
                command = input("\nlabrys> ").strip()
                
                if command.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif command.lower() == 'help':
                    self._show_help()
                
                elif command.lower() == 'status':
                    status = self.taskmaster.get_system_status()
                    print(json.dumps(status, indent=2))
                
                elif command.lower() == 'validate':
                    await self.run_validation()
                
                elif command.startswith('research'):
                    query = command[8:].strip()
                    if query:
                        await self._research_query(query)
                    else:
                        print("Please provide a research query")
                
                elif command.startswith('generate'):
                    spec = command[8:].strip()
                    if spec:
                        await self._generate_code(spec)
                    else:
                        print("Please provide generation specifications")
                
                elif command.startswith('analyze'):
                    target = command[7:].strip()
                    if target:
                        await self._analyze_target(target)
                    else:
                        print("Please provide analysis target")
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_help(self):
        """Show available commands"""
        print("\n📖 Available Commands:")
        print("   help        - Show this help message")
        print("   status      - Show system status")
        print("   validate    - Run system validation")
        print("   research    - Perform research query")
        print("   generate    - Generate code")
        print("   analyze     - Analyze code/project")
        print("   quit/exit   - Exit interactive mode")
    
    async def _research_query(self, query: str):
        """Execute research query"""
        print(f"🔍 Researching: {query}")
        
        analytical_blade = self.coordinator.analytical_blade
        result = await analytical_blade.computational_research(query)
        
        if "error" in result:
            print(f"❌ Research failed: {result['error']}")
        else:
            print("✅ Research completed")
            # Display simplified result
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                print(f"Result: {content[:500]}...")
    
    async def _generate_code(self, spec: str):
        """Generate code using synthesis blade"""
        print(f"🛠️  Generating code: {spec}")
        
        # Parse specification
        specs = {
            "type": "function",
            "name": spec.replace(" ", "_").lower(),
            "language": "python",
            "requirements": [spec]
        }
        
        synthesis_blade = self.coordinator.synthesis_blade
        result = await synthesis_blade.claude_sparc_generation(specs)
        
        print("✅ Code generation completed")
        print(f"Generated code:\n{result.code_content}")
    
    async def _analyze_target(self, target: str):
        """Analyze target using analytical blade"""
        print(f"🔬 Analyzing: {target}")
        
        analytical_blade = self.coordinator.analytical_blade
        
        if os.path.exists(target):
            # Analyze file
            with open(target, 'r') as f:
                content = f.read()
            
            result = await analytical_blade.static_analysis(content)
            
            print("✅ Analysis completed")
            print(f"Findings: {len(result.findings)} issues found")
            print(f"Risk Level: {result.risk_level}")
            
            for finding in result.findings[:3]:  # Show first 3 findings
                print(f"  • {finding}")
        else:
            # Analyze as project description
            result = await analytical_blade.constraint_identification(target)
            
            print("✅ Constraint analysis completed")
            print(f"Findings: {len(result.findings)} constraints identified")
            
            for finding in result.findings:
                print(f"  • {finding}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "framework": "LABRYS",
            "version": "1.0.0",
            "description": "Double-Edged AI Development Framework",
            "components": {
                "taskmaster": "TaskMaster Integration Layer",
                "coordinator": "Dual-Blade Coordination System",
                "analytical_blade": "Left Blade - Analysis Engine",
                "synthesis_blade": "Right Blade - Synthesis Engine",
                "validator": "System Validation Framework"
            },
            "status": {
                "initialized": self.system_initialized,
                "timestamp": datetime.now().isoformat()
            }
        }

async def main():
    """
    Main entry point for LABRYS framework
    """
    parser = argparse.ArgumentParser(
        description="LABRYS - Double-Edged AI Development Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python labrys_main.py --initialize                    # Initialize system
  python labrys_main.py --validate                      # Run validation
  python labrys_main.py --execute tasks.json            # Execute tasks
  python labrys_main.py --interactive                   # Interactive mode
  python labrys_main.py --info                          # Show system info
        """
    )
    
    parser.add_argument("--initialize", action="store_true", help="Initialize LABRYS system")
    parser.add_argument("--validate", action="store_true", help="Run system validation")
    parser.add_argument("--execute", help="Execute tasks from JSON file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--info", action="store_true", help="Show system information")
    parser.add_argument("--status", action="store_true", help="Show system status")
    
    args = parser.parse_args()
    
    # Create LABRYS framework instance
    labrys = LabrysFramework()
    
    if args.initialize:
        result = await labrys.initialize_system()
        print(json.dumps(result, indent=2))
    
    elif args.validate:
        result = await labrys.run_validation()
        print(f"\nValidation report saved to: .labrys/validation_report.json")
    
    elif args.execute:
        if os.path.exists(args.execute):
            result = await labrys.execute_labrys_tasks(args.execute)
            print(json.dumps(result, indent=2))
        else:
            print(f"Task file not found: {args.execute}")
    
    elif args.interactive:
        await labrys.run_interactive_mode()
    
    elif args.info:
        info = labrys.get_system_info()
        print(json.dumps(info, indent=2))
    
    elif args.status:
        if not labrys.system_initialized:
            await labrys.initialize_system()
        status = labrys.taskmaster.get_system_status()
        print(json.dumps(status, indent=2))
    
    else:
        # Default: show help and basic info
        parser.print_help()
        print("\n" + "="*50)
        print("LABRYS - Double-Edged AI Development Framework")
        print("="*50)
        print("The ancient labrys symbol represents dual-aspect")
        print("processing: analytical precision and creative synthesis")
        print("working in perfect harmony.")
        print("\nUse --help for detailed usage information.")

if __name__ == "__main__":
    asyncio.run(main())