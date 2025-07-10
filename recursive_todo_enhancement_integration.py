#!/usr/bin/env python3
"""
Recursive Todo Enhancement Engine - System Integration Module
Task 51.4: Integrate Enhancement Engine with Todo System

This module provides integration interfaces for connecting the recursive
enhancement engine with existing todo management systems including TaskMaster,
Claude Code TodoWrite, and other external systems.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import argparse

# Import our enhancement engine
from recursive_todo_enhancement_architecture import (
    RecursiveEnhancementEngine, TaskMasterIntegration, TodoItem, 
    RecursiveEnhancementContext, EnhancementResult
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for system integration"""
    taskmaster_enabled: bool = True
    claude_code_enabled: bool = True
    auto_enhancement: bool = False
    enhancement_interval: int = 300  # seconds
    max_recursion_depth: int = 3
    quality_threshold: float = 0.8
    api_enabled: bool = True
    api_port: int = 8080
    log_level: str = "INFO"
    backup_enabled: bool = True


class TaskMasterCLIIntegration:
    """Integration with TaskMaster CLI commands"""
    
    def __init__(self, enhancement_engine: RecursiveEnhancementEngine, config: IntegrationConfig):
        self.enhancement_engine = enhancement_engine
        self.config = config
        self.taskmaster_integration = TaskMasterIntegration(enhancement_engine)
        self.logger = logging.getLogger("TaskMasterCLIIntegration")
    
    async def enhance_task_command(self, task_id: str = None, recursive: bool = True, 
                                 force: bool = False) -> Dict[str, Any]:
        """Enhanced task-master command for todo enhancement"""
        try:
            # Read current tasks
            tasks_file = Path(".taskmaster/tasks/tasks.json")
            if not tasks_file.exists():
                return {"error": "TaskMaster tasks.json not found", "success": False}
            
            with open(tasks_file, 'r') as f:
                task_data = json.load(f)
            
            # Create backup if enabled
            if self.config.backup_enabled:
                backup_file = tasks_file.with_suffix(f".backup.{int(time.time())}.json")
                with open(backup_file, 'w') as f:
                    json.dump(task_data, f, indent=2)
                self.logger.info(f"Created backup: {backup_file}")
            
            # Apply enhancements
            enhanced_data = await self.taskmaster_integration.enhance_task_json(task_data)
            
            # Calculate enhancement statistics
            stats = self._calculate_enhancement_stats(task_data, enhanced_data)
            
            # Write enhanced data back if improvements found
            if stats['total_improvements'] > 0 or force:
                with open(tasks_file, 'w') as f:
                    json.dump(enhanced_data, f, indent=2)
                
                # Update tasks.md files if they exist
                self._update_task_markdown_files(enhanced_data)
                
                self.logger.info(f"Enhanced {stats['total_improvements']} tasks")
                return {
                    "success": True,
                    "enhanced_tasks": stats['total_improvements'],
                    "statistics": stats,
                    "backup_created": self.config.backup_enabled
                }
            else:
                self.logger.info("No improvements found or forced enhancement disabled")
                return {
                    "success": True,
                    "enhanced_tasks": 0,
                    "message": "No enhancements needed",
                    "statistics": stats
                }
        
        except Exception as e:
            self.logger.error(f"Enhancement command failed: {e}")
            return {"error": str(e), "success": False}
    
    def _calculate_enhancement_stats(self, original: Dict[str, Any], 
                                   enhanced: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhancement statistics"""
        original_tasks = original.get("master", {}).get("tasks", [])
        enhanced_tasks = enhanced.get("master", {}).get("tasks", [])
        
        improvements = 0
        total_tasks = len(original_tasks)
        
        for orig_task, enh_task in zip(original_tasks, enhanced_tasks):
            if orig_task.get("title") != enh_task.get("title"):
                improvements += 1
            
            # Check subtasks
            orig_subtasks = orig_task.get("subtasks", [])
            enh_subtasks = enh_task.get("subtasks", [])
            
            for orig_sub, enh_sub in zip(orig_subtasks, enh_subtasks):
                total_tasks += 1
                if orig_sub.get("title") != enh_sub.get("title"):
                    improvements += 1
        
        return {
            "total_tasks": total_tasks,
            "total_improvements": improvements,
            "improvement_rate": improvements / total_tasks if total_tasks > 0 else 0.0,
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_task_markdown_files(self, enhanced_data: Dict[str, Any]):
        """Update task markdown files if they exist"""
        tasks_dir = Path(".taskmaster/tasks")
        if not tasks_dir.exists():
            return
        
        for task in enhanced_data.get("master", {}).get("tasks", []):
            task_id = task.get("id")
            if task_id:
                md_file = tasks_dir / f"task-{task_id}.md"
                if md_file.exists():
                    # Update markdown with enhanced title
                    content = md_file.read_text()
                    # Simple title update - in production would use proper markdown parsing
                    enhanced_title = task.get("title", "")
                    if enhanced_title:
                        lines = content.split('\n')
                        if lines and lines[0].startswith('#'):
                            lines[0] = f"# {enhanced_title}"
                            md_file.write_text('\n'.join(lines))
    
    def generate_enhancement_report(self) -> Dict[str, Any]:
        """Generate comprehensive enhancement report"""
        stats = self.enhancement_engine.get_enhancement_statistics()
        meta_status = self.enhancement_engine.get_meta_learning_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "engine_statistics": stats,
            "meta_learning_status": meta_status,
            "integration_status": {
                "taskmaster_enabled": self.config.taskmaster_enabled,
                "auto_enhancement": self.config.auto_enhancement,
                "enhancement_interval": self.config.enhancement_interval
            }
        }


class ClaudeCodeTodoIntegration:
    """Integration with Claude Code TodoWrite functionality"""
    
    def __init__(self, enhancement_engine: RecursiveEnhancementEngine, config: IntegrationConfig):
        self.enhancement_engine = enhancement_engine
        self.config = config
        self.logger = logging.getLogger("ClaudeCodeTodoIntegration")
    
    async def enhance_todowrite_todos(self, todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance todos from Claude Code TodoWrite format"""
        try:
            # Convert TodoWrite format to our TodoItem format
            todo_items = []
            for todo in todos:
                todo_item = TodoItem(
                    id=todo.get("id", str(uuid.uuid4())),
                    content=todo.get("content", ""),
                    priority=self._map_claude_priority(todo.get("priority", "medium")),
                    metadata={
                        "source": "claude_code_todowrite",
                        "original_status": todo.get("status", "pending")
                    }
                )
                todo_items.append(todo_item)
            
            # Create enhancement context
            context = RecursiveEnhancementContext(
                max_depth=self.config.max_recursion_depth,
                quality_threshold=self.config.quality_threshold
            )
            
            # Enhance todos
            enhancement_results = await self.enhancement_engine.enhance_todo_batch(todo_items, context)
            
            # Convert back to TodoWrite format
            enhanced_todos = []
            for result in enhancement_results:
                enhanced_todo = {
                    "id": result.enhanced_todo.id,
                    "content": result.enhanced_todo.content,
                    "status": todos[enhancement_results.index(result)].get("status", "pending"),
                    "priority": self._map_priority_back(result.enhanced_todo.priority),
                    "enhancement_metadata": {
                        "improved": result.quality_improvement > 0.05,
                        "improvement_score": result.quality_improvement,
                        "processing_time": result.processing_time,
                        "strategies_applied": result.applied_strategies,
                        "enhanced_at": result.timestamp.isoformat()
                    }
                }
                enhanced_todos.append(enhanced_todo)
            
            self.logger.info(f"Enhanced {len(enhanced_todos)} Claude Code todos")
            return enhanced_todos
            
        except Exception as e:
            self.logger.error(f"Claude Code todo enhancement failed: {e}")
            return todos  # Return original todos on error
    
    def _map_claude_priority(self, claude_priority: str) -> int:
        """Map Claude priority string to numeric value"""
        priority_map = {
            "low": 3,
            "medium": 5,
            "high": 8
        }
        return priority_map.get(claude_priority.lower(), 5)
    
    def _map_priority_back(self, numeric_priority: int) -> str:
        """Map numeric priority back to Claude format"""
        if numeric_priority <= 3:
            return "low"
        elif numeric_priority <= 6:
            return "medium"
        else:
            return "high"


class RealTimeEnhancementWatcher:
    """Real-time file watcher for automatic enhancement"""
    
    def __init__(self, enhancement_engine: RecursiveEnhancementEngine, 
                 cli_integration: TaskMasterCLIIntegration, config: IntegrationConfig):
        self.enhancement_engine = enhancement_engine
        self.cli_integration = cli_integration
        self.config = config
        self.logger = logging.getLogger("RealTimeWatcher")
        self.watching = False
        self.last_enhancement = datetime.now()
    
    async def start_watching(self):
        """Start watching for file changes"""
        if not self.config.auto_enhancement:
            self.logger.info("Auto-enhancement disabled")
            return
        
        self.watching = True
        self.logger.info("Started real-time enhancement watching")
        
        while self.watching:
            try:
                # Check if enough time has passed since last enhancement
                time_since_last = (datetime.now() - self.last_enhancement).total_seconds()
                
                if time_since_last >= self.config.enhancement_interval:
                    # Check if tasks.json has been modified
                    tasks_file = Path(".taskmaster/tasks/tasks.json")
                    if tasks_file.exists():
                        file_mtime = datetime.fromtimestamp(tasks_file.stat().st_mtime)
                        
                        if file_mtime > self.last_enhancement:
                            self.logger.info("Tasks file modified, triggering enhancement")
                            result = await self.cli_integration.enhance_task_command(force=False)
                            
                            if result.get("success"):
                                self.logger.info(f"Auto-enhanced {result.get('enhanced_tasks', 0)} tasks")
                            
                            self.last_enhancement = datetime.now()
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Watcher error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def stop_watching(self):
        """Stop watching for file changes"""
        self.watching = False
        self.logger.info("Stopped real-time enhancement watching")


class EnhancementAPIServer:
    """Simple HTTP API server for enhancement operations"""
    
    def __init__(self, enhancement_engine: RecursiveEnhancementEngine, 
                 cli_integration: TaskMasterCLIIntegration, config: IntegrationConfig):
        self.enhancement_engine = enhancement_engine
        self.cli_integration = cli_integration
        self.config = config
        self.logger = logging.getLogger("EnhancementAPI")
    
    async def start_server(self):
        """Start the API server"""
        if not self.config.api_enabled:
            self.logger.info("API server disabled")
            return
        
        try:
            # Simple HTTP server implementation
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import threading
            
            class EnhancementHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == "/health":
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(b'{"status": "healthy"}')
                    elif self.path == "/stats":
                        stats = self.server.enhancement_engine.get_enhancement_statistics()
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(stats).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                def do_POST(self):
                    if self.path == "/enhance":
                        content_length = int(self.headers['Content-Length'])
                        post_data = self.rfile.read(content_length)
                        
                        try:
                            request_data = json.loads(post_data.decode('utf-8'))
                            result = asyncio.run(self.server.cli_integration.enhance_task_command(
                                force=request_data.get("force", False)
                            ))
                            
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps(result).encode())
                        
                        except Exception as e:
                            self.send_response(500)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps({"error": str(e)}).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                def log_message(self, format, *args):
                    pass  # Suppress default logging
            
            server = HTTPServer(('localhost', self.config.api_port), EnhancementHandler)
            server.enhancement_engine = self.enhancement_engine
            server.cli_integration = self.cli_integration
            
            self.logger.info(f"API server started on port {self.config.api_port}")
            
            # Start server in separate thread
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            return server
            
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            return None


class RecursiveTodoEnhancementIntegrator:
    """Main integration orchestrator"""
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        self.enhancement_engine = RecursiveEnhancementEngine()
        self.cli_integration = TaskMasterCLIIntegration(self.enhancement_engine, self.config)
        self.claude_integration = ClaudeCodeTodoIntegration(self.enhancement_engine, self.config)
        self.watcher = RealTimeEnhancementWatcher(
            self.enhancement_engine, self.cli_integration, self.config
        )
        self.api_server = EnhancementAPIServer(
            self.enhancement_engine, self.cli_integration, self.config
        )
        self.logger = logging.getLogger("RecursiveTodoEnhancementIntegrator")
    
    async def start_all_services(self):
        """Start all integration services"""
        self.logger.info("Starting Recursive Todo Enhancement Integration Services")
        
        services = []
        
        # Start API server
        if self.config.api_enabled:
            api_server = await self.api_server.start_server()
            if api_server:
                services.append("API Server")
        
        # Start file watcher
        if self.config.auto_enhancement:
            watcher_task = asyncio.create_task(self.watcher.start_watching())
            services.append("File Watcher")
        
        self.logger.info(f"Started services: {', '.join(services)}")
        
        # Keep services running
        try:
            if self.config.auto_enhancement:
                await watcher_task
        except KeyboardInterrupt:
            self.logger.info("Shutting down services...")
            self.watcher.stop_watching()
    
    async def enhance_current_tasks(self, force: bool = False) -> Dict[str, Any]:
        """Enhance current TaskMaster tasks"""
        return await self.cli_integration.enhance_task_command(force=force)
    
    async def enhance_claude_todos(self, todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance Claude Code todos"""
        return await self.claude_integration.enhance_todowrite_todos(todos)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "integration_config": asdict(self.config),
            "enhancement_report": self.cli_integration.generate_enhancement_report(),
            "services": {
                "api_enabled": self.config.api_enabled,
                "watcher_enabled": self.config.auto_enhancement,
                "taskmaster_enabled": self.config.taskmaster_enabled,
                "claude_code_enabled": self.config.claude_code_enabled
            }
        }


def create_cli():
    """Create command-line interface for the integration system"""
    parser = argparse.ArgumentParser(description="Recursive Todo Enhancement Integration")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Enhance command
    enhance_parser = subparsers.add_parser('enhance', help='Enhance current tasks')
    enhance_parser.add_argument('--force', action='store_true', help='Force enhancement even if no improvements')
    enhance_parser.add_argument('--recursive', action='store_true', default=True, help='Use recursive enhancement')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Start services command
    services_parser = subparsers.add_parser('start', help='Start all integration services')
    services_parser.add_argument('--config', help='Configuration file path')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate enhancement report')
    
    return parser


async def main():
    """Main function for CLI usage"""
    parser = create_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create integrator
    config = IntegrationConfig()
    integrator = RecursiveTodoEnhancementIntegrator(config)
    
    if args.command == 'enhance':
        print("= Enhancing current tasks...")
        result = await integrator.enhance_current_tasks(force=args.force)
        
        if result.get("success"):
            print(f" Enhanced {result.get('enhanced_tasks', 0)} tasks")
            if result.get('statistics'):
                stats = result['statistics']
                print(f"   Total tasks: {stats['total_tasks']}")
                print(f"   Improvement rate: {stats['improvement_rate']:.2%}")
        else:
            print(f"Enhancement failed: {result.get('error')}")
    
    elif args.command == 'status':
        print("System Status:")
        status = integrator.get_system_status()
        
        print(f"  Enhancement Engine: Operational")
        print(f"  API Server: {'Enabled' if status['integration_config']['api_enabled'] else 'Disabled'}")
        print(f"  Auto Enhancement: {'Enabled' if status['integration_config']['auto_enhancement'] else 'Disabled'}")
        print(f"  TaskMaster Integration: {'Enabled' if status['integration_config']['taskmaster_enabled'] else 'Disabled'}")
        
        if 'enhancement_report' in status:
            report = status['enhancement_report']
            if 'engine_statistics' in report:
                stats = report['engine_statistics']
                if stats.get('total_enhancements', 0) > 0:
                    print(f"  Total Enhancements: {stats['total_enhancements']}")
                    print(f"  Average Improvement: {stats['average_improvement']:.3f}")
    
    elif args.command == 'start':
        print("Starting integration services...")
        await integrator.start_all_services()
    
    elif args.command == 'report':
        print("Generating enhancement report...")
        report = integrator.cli_integration.generate_enhancement_report()
        
        print(f"\nEnhancement Report - {report['timestamp']}")
        print("=" * 50)
        
        if 'engine_statistics' in report:
            stats = report['engine_statistics']
            if stats.get('total_enhancements', 0) > 0:
                print(f"Total Enhancements: {stats['total_enhancements']}")
                print(f"Average Improvement: {stats['average_improvement']:.3f}")
                print(f"Average Processing Time: {stats['average_processing_time']:.3f}s")
                
                if 'strategy_performance' in stats:
                    print("\nStrategy Performance:")
                    for strategy, perf in stats['strategy_performance'].items():
                        print(f"  {strategy}: {perf['average_improvement']:.3f} avg improvement")
            else:
                print("No enhancements recorded yet")
        
        if 'meta_learning_status' in report:
            meta = report['meta_learning_status']
            print(f"\nMeta-Learning: {'Enabled' if meta['enabled'] else 'Disabled'}")
            if meta['enabled']:
                print(f"Parameter Stability: {meta['parameter_stability']:.3f}")


if __name__ == "__main__":
    # Demo integration
    async def demo():
        print("= Recursive Todo Enhancement Integration Demo")
        print("=" * 55)
        
        # Create integrator
        config = IntegrationConfig(
            auto_enhancement=False,  # Disabled for demo
            api_enabled=False       # Disabled for demo
        )
        integrator = RecursiveTodoEnhancementIntegrator(config)
        
        # Test enhancement of current tasks
        print("\nTesting TaskMaster integration...")
        result = await integrator.enhance_current_tasks(force=True)
        
        if result.get("success"):
            print(f"   Enhanced {result.get('enhanced_tasks', 0)} tasks")
            if result.get('statistics'):
                stats = result['statistics']
                print(f"  Total tasks: {stats['total_tasks']}")
                print(f"  Improvement rate: {stats['improvement_rate']:.2%}")
        else:
            print(f"  Enhancement failed: {result.get('error')}")
        
        # Test Claude Code integration
        print("\n> Testing Claude Code integration...")
        sample_todos = [
            {
                "id": "demo-1",
                "content": "fix the bug",
                "status": "pending",
                "priority": "high"
            },
            {
                "id": "demo-2", 
                "content": "maybe add some tests",
                "status": "pending",
                "priority": "medium"
            }
        ]
        
        enhanced_todos = await integrator.enhance_claude_todos(sample_todos)
        
        for i, (orig, enhanced) in enumerate(zip(sample_todos, enhanced_todos)):
            print(f"  Todo {i+1}:")
            print(f"    Original: '{orig['content']}'")
            print(f"    Enhanced: '{enhanced['content']}'")
            if enhanced.get('enhancement_metadata', {}).get('improved'):
                score = enhanced['enhancement_metadata']['improvement_score']
                print(f"    Improvement: {score:.3f}")
        
        # Show system status
        print("\nSystem Status:")
        status = integrator.get_system_status()
        
        services = status['services']
        print(f"  API Server: {'Enabled' if services['api_enabled'] else 'Disabled'}")
        print(f"  Auto Enhancement: {'Enabled' if services['watcher_enabled'] else 'Disabled'}")
        print(f"  TaskMaster: {'Enabled' if services['taskmaster_enabled'] else 'Disabled'}")
        print(f"  Claude Code: {'Enabled' if services['claude_code_enabled'] else 'Disabled'}")
        
        print("\n Integration system operational!")
    
    # Run demo or CLI
    import sys
    if len(sys.argv) == 1:
        asyncio.run(demo())
    else:
        asyncio.run(main())