#!/usr/bin/env python3
"""
Todo Enhancement Integration Layer
Atomic Task 51.4: Integrate Enhancement Engine with Todo System

This module provides seamless integration between the recursive todo
enhancement engine and existing todo management systems, including
Task-Master AI integration and CLI interface.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import argparse
import sys

# Import our enhancement engine
from recursive_todo_enhancer import RecursiveTodoEnhancer, EnhancementQualityMetrics


@dataclass
class TodoEnhancementRequest:
    """Request for todo enhancement"""
    todo_text: str
    context: Dict[str, Any] = field(default_factory=dict)
    max_depth: int = 3
    timeout_seconds: float = 30.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TodoEnhancementResponse:
    """Response from todo enhancement"""
    request_id: str
    original_text: str
    enhanced_text: str
    improvement_score: float
    strategy_used: str
    processing_time: float
    quality_metrics: Dict[str, float]
    success: bool = True
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class TaskMasterTodoExtractor:
    """Extract and convert todos from Task-Master format"""
    
    def __init__(self):
        self.logger = logging.getLogger("TaskMasterTodoExtractor")
    
    def extract_todos_from_file(self, tasks_file: str) -> List[Dict[str, Any]]:
        """Extract todos from Task-Master tasks.json file"""
        try:
            with open(tasks_file, 'r') as f:
                task_data = json.load(f)
            
            todos = []
            self._extract_recursive(task_data, todos)
            
            self.logger.info(f"Extracted {len(todos)} todos from {tasks_file}")
            return todos
            
        except Exception as e:
            self.logger.error(f"Error extracting todos from {tasks_file}: {e}")
            return []
    
    def _extract_recursive(self, data: Any, todos: List[Dict[str, Any]], parent_path: str = ""):
        """Recursively extract todos from nested data structure"""
        if isinstance(data, dict):
            # Check if this is a task object
            if self._is_task_object(data):
                todo = self._convert_task_to_todo(data, parent_path)
                todos.append(todo)
                
                # Process subtasks
                if "subtasks" in data:
                    for subtask in data["subtasks"]:
                        subtask_path = f"{parent_path}.{data.get('id', 'unknown')}" if parent_path else str(data.get('id', 'unknown'))
                        self._extract_recursive(subtask, todos, subtask_path)
            
            # Recurse through other dictionary values
            for key, value in data.items():
                if key != "subtasks":  # Avoid double-processing subtasks
                    new_path = f"{parent_path}.{key}" if parent_path else key
                    self._extract_recursive(value, todos, new_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_path = f"{parent_path}[{i}]" if parent_path else f"[{i}]"
                self._extract_recursive(item, todos, item_path)
    
    def _is_task_object(self, obj: Dict[str, Any]) -> bool:
        """Check if object is a task"""
        task_indicators = ["title", "description", "status"]
        return isinstance(obj, dict) and any(key in obj for key in task_indicators)
    
    def _convert_task_to_todo(self, task: Dict[str, Any], parent_path: str) -> Dict[str, Any]:
        """Convert task object to todo format"""
        return {
            "id": str(task.get("id", f"task_{uuid.uuid4()}")),
            "text": task.get("title", task.get("description", "No title")),
            "context": {
                "source": "taskmaster",
                "parent_path": parent_path,
                "priority": task.get("priority", "medium"),
                "status": task.get("status", "pending"),
                "original_task": task
            },
            "metadata": {
                "dependencies": task.get("dependencies", []),
                "test_strategy": task.get("testStrategy", ""),
                "details": task.get("details", "")
            }
        }
    
    def update_tasks_with_enhancements(self, tasks_file: str, 
                                     enhancement_results: List[TodoEnhancementResponse]) -> bool:
        """Update Task-Master file with enhancement results"""
        try:
            # Create enhancement lookup
            enhancements_by_id = {
                result.request_id: result for result in enhancement_results 
                if result.success
            }
            
            # Load current task data
            with open(tasks_file, 'r') as f:
                task_data = json.load(f)
            
            # Apply enhancements
            updated_count = self._apply_enhancements_recursive(task_data, enhancements_by_id)
            
            # Create backup
            backup_file = f"{tasks_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(backup_file, 'w') as f:
                json.dump(task_data, f, indent=2)
            
            # Write updated data
            with open(tasks_file, 'w') as f:
                json.dump(task_data, f, indent=2)
            
            self.logger.info(f"Updated {updated_count} tasks in {tasks_file}, backup created: {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating tasks file {tasks_file}: {e}")
            return False
    
    def _apply_enhancements_recursive(self, data: Any, 
                                    enhancements: Dict[str, TodoEnhancementResponse]) -> int:
        """Recursively apply enhancements to task data"""
        updated_count = 0
        
        if isinstance(data, dict):
            # Check if this is a task that can be enhanced
            if self._is_task_object(data):
                task_id = str(data.get("id", ""))
                if task_id in enhancements:
                    enhancement = enhancements[task_id]
                    data["title"] = enhancement.enhanced_text
                    data["enhancement_metadata"] = {
                        "enhanced": True,
                        "improvement_score": enhancement.improvement_score,
                        "strategy_used": enhancement.strategy_used,
                        "enhanced_at": enhancement.timestamp.isoformat(),
                        "original_text": enhancement.original_text
                    }
                    updated_count += 1
            
            # Recurse through dictionary values
            for value in data.values():
                updated_count += self._apply_enhancements_recursive(value, enhancements)
        
        elif isinstance(data, list):
            for item in data:
                updated_count += self._apply_enhancements_recursive(item, enhancements)
        
        return updated_count


class TodoEnhancementService:
    """Main service for todo enhancement operations"""
    
    def __init__(self, max_depth: int = 3, timeout_seconds: float = 30.0):
        self.enhancer = RecursiveTodoEnhancer(max_depth, timeout_seconds)
        self.extractor = TaskMasterTodoExtractor()
        self.logger = logging.getLogger("TodoEnhancementService")
        
        # Service statistics
        self.requests_processed = 0
        self.successful_enhancements = 0
        self.total_processing_time = 0.0
        self.service_start_time = datetime.now()
    
    async def enhance_single_todo(self, request: TodoEnhancementRequest) -> TodoEnhancementResponse:
        """Enhance a single todo item"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing enhancement request {request.request_id}: {request.todo_text[:50]}...")
            
            # Run enhancement
            result = await self.enhancer.enhance_recursive(
                request.todo_text, 
                request.context
            )
            
            processing_time = time.time() - start_time
            
            # Create response
            response = TodoEnhancementResponse(
                request_id=request.request_id,
                original_text=request.todo_text,
                enhanced_text=result["enhanced_text"],
                improvement_score=result["improvement_score"],
                strategy_used=result.get("strategy_used", "unknown"),
                processing_time=processing_time,
                quality_metrics=result["quality_metrics"],
                success=True
            )
            
            # Update statistics
            self.requests_processed += 1
            if result["improvement_score"] > 0:
                self.successful_enhancements += 1
            self.total_processing_time += processing_time
            
            self.logger.info(f"Enhancement completed: {result['improvement_score']:.3f} improvement in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Enhancement failed for request {request.request_id}: {e}")
            
            # Create error response
            response = TodoEnhancementResponse(
                request_id=request.request_id,
                original_text=request.todo_text,
                enhanced_text=request.todo_text,  # Return original on error
                improvement_score=0.0,
                strategy_used="error",
                processing_time=processing_time,
                quality_metrics={},
                success=False,
                error_message=str(e)
            )
            
            self.requests_processed += 1
            self.total_processing_time += processing_time
            
            return response
    
    async def enhance_todo_batch(self, requests: List[TodoEnhancementRequest]) -> List[TodoEnhancementResponse]:
        """Enhance multiple todos in batch"""
        self.logger.info(f"Processing batch enhancement of {len(requests)} todos")
        
        responses = []
        for request in requests:
            response = await self.enhance_single_todo(request)
            responses.append(response)
        
        self.logger.info(f"Batch enhancement completed: {len(responses)} responses generated")
        return responses
    
    async def enhance_taskmaster_file(self, tasks_file: str, 
                                    update_file: bool = True) -> Dict[str, Any]:
        """Enhance todos from Task-Master file"""
        self.logger.info(f"Enhancing todos from Task-Master file: {tasks_file}")
        
        # Extract todos
        todos = self.extractor.extract_todos_from_file(tasks_file)
        if not todos:
            return {"success": False, "error": "No todos found in file"}
        
        # Create enhancement requests
        requests = []
        for todo in todos:
            request = TodoEnhancementRequest(
                todo_text=todo["text"],
                context=todo.get("context", {}),
                request_id=todo["id"]
            )
            requests.append(request)
        
        # Process enhancements
        responses = await self.enhance_todo_batch(requests)
        
        # Update file if requested
        if update_file:
            success = self.extractor.update_tasks_with_enhancements(tasks_file, responses)
            if not success:
                return {"success": False, "error": "Failed to update tasks file"}
        
        # Generate summary
        successful_responses = [r for r in responses if r.success]
        summary = {
            "success": True,
            "todos_processed": len(todos),
            "enhancements_successful": len(successful_responses),
            "average_improvement": sum(r.improvement_score for r in successful_responses) / max(1, len(successful_responses)),
            "total_processing_time": sum(r.processing_time for r in responses),
            "file_updated": update_file
        }
        
        self.logger.info(f"Task-Master file enhancement completed: {summary}")
        return summary
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        uptime = (datetime.now() - self.service_start_time).total_seconds()
        
        stats = {
            "service_uptime_seconds": uptime,
            "requests_processed": self.requests_processed,
            "successful_enhancements": self.successful_enhancements,
            "success_rate": self.successful_enhancements / max(1, self.requests_processed),
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / max(1, self.requests_processed),
            "requests_per_second": self.requests_processed / max(1, uptime)
        }
        
        # Add enhancer statistics
        enhancer_stats = self.enhancer.get_performance_statistics()
        stats["enhancer_statistics"] = enhancer_stats
        
        return stats


class TodoEnhancementCLI:
    """Command-line interface for todo enhancement"""
    
    def __init__(self):
        self.service = TodoEnhancementService()
        self.logger = logging.getLogger("TodoEnhancementCLI")
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser"""
        parser = argparse.ArgumentParser(
            description="Recursive Todo Enhancement Engine CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Enhance a single todo
  python todo_enhancement_integration.py enhance "fix authentication bug"
  
  # Enhance from Task-Master file
  python todo_enhancement_integration.py file .taskmaster/tasks/tasks.json
  
  # Enhance with custom parameters
  python todo_enhancement_integration.py enhance "implement api" --max-depth 2 --timeout 15
  
  # Show service statistics
  python todo_enhancement_integration.py stats
            """
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Single enhancement command
        enhance_parser = subparsers.add_parser("enhance", help="Enhance a single todo")
        enhance_parser.add_argument("text", help="Todo text to enhance")
        enhance_parser.add_argument("--max-depth", type=int, default=3, help="Maximum recursion depth")
        enhance_parser.add_argument("--timeout", type=float, default=30.0, help="Timeout in seconds")
        enhance_parser.add_argument("--context", type=str, help="JSON context string")
        enhance_parser.add_argument("--output", choices=["text", "json"], default="text", help="Output format")
        
        # File enhancement command
        file_parser = subparsers.add_parser("file", help="Enhance todos from Task-Master file")
        file_parser.add_argument("file_path", help="Path to Task-Master tasks.json file")
        file_parser.add_argument("--no-update", action="store_true", help="Don't update the file")
        file_parser.add_argument("--output", choices=["summary", "json"], default="summary", help="Output format")
        
        # Statistics command
        stats_parser = subparsers.add_parser("stats", help="Show service statistics")
        stats_parser.add_argument("--output", choices=["text", "json"], default="text", help="Output format")
        
        # Batch command
        batch_parser = subparsers.add_parser("batch", help="Enhance multiple todos from stdin")
        batch_parser.add_argument("--max-depth", type=int, default=3, help="Maximum recursion depth")
        batch_parser.add_argument("--timeout", type=float, default=30.0, help="Timeout in seconds")
        batch_parser.add_argument("--output", choices=["text", "json"], default="text", help="Output format")
        
        return parser
    
    async def handle_enhance_command(self, args) -> None:
        """Handle single todo enhancement"""
        context = {}
        if args.context:
            try:
                context = json.loads(args.context)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON context: {args.context}")
                return
        
        request = TodoEnhancementRequest(
            todo_text=args.text,
            context=context,
            max_depth=args.max_depth,
            timeout_seconds=args.timeout
        )
        
        response = await self.service.enhance_single_todo(request)
        
        if args.output == "json":
            print(json.dumps(response.to_dict(), indent=2))
        else:
            print(f"Original:  {response.original_text}")
            print(f"Enhanced:  {response.enhanced_text}")
            print(f"Improvement: {response.improvement_score:.3f}")
            print(f"Strategy: {response.strategy_used}")
            print(f"Time: {response.processing_time:.3f}s")
            if not response.success:
                print(f"Error: {response.error_message}")
    
    async def handle_file_command(self, args) -> None:
        """Handle Task-Master file enhancement"""
        if not Path(args.file_path).exists():
            print(f"Error: File not found: {args.file_path}")
            return
        
        result = await self.service.enhance_taskmaster_file(
            args.file_path, 
            update_file=not args.no_update
        )
        
        if args.output == "json":
            print(json.dumps(result, indent=2))
        else:
            if result["success"]:
                print(f"✅ Enhancement completed successfully")
                print(f"   Todos processed: {result['todos_processed']}")
                print(f"   Successful enhancements: {result['enhancements_successful']}")
                print(f"   Average improvement: {result['average_improvement']:.3f}")
                print(f"   Total processing time: {result['total_processing_time']:.3f}s")
                print(f"   File updated: {result['file_updated']}")
            else:
                print(f"❌ Enhancement failed: {result.get('error', 'Unknown error')}")
    
    async def handle_batch_command(self, args) -> None:
        """Handle batch enhancement from stdin"""
        todos = []
        for line in sys.stdin:
            line = line.strip()
            if line:
                todos.append(line)
        
        if not todos:
            print("Error: No todos provided via stdin")
            return
        
        requests = [
            TodoEnhancementRequest(
                todo_text=todo,
                max_depth=args.max_depth,
                timeout_seconds=args.timeout
            ) for todo in todos
        ]
        
        responses = await self.service.enhance_todo_batch(requests)
        
        if args.output == "json":
            print(json.dumps([r.to_dict() for r in responses], indent=2))
        else:
            for i, response in enumerate(responses, 1):
                print(f"\n{i}. Original:  {response.original_text}")
                print(f"   Enhanced:  {response.enhanced_text}")
                print(f"   Improvement: {response.improvement_score:.3f}")
                print(f"   Strategy: {response.strategy_used}")
    
    def handle_stats_command(self, args) -> None:
        """Handle statistics display"""
        stats = self.service.get_service_statistics()
        
        if args.output == "json":
            print(json.dumps(stats, indent=2))
        else:
            print("📊 Todo Enhancement Service Statistics")
            print(f"   Uptime: {stats['service_uptime_seconds']:.1f}s")
            print(f"   Requests processed: {stats['requests_processed']}")
            print(f"   Successful enhancements: {stats['successful_enhancements']}")
            print(f"   Success rate: {stats['success_rate']:.1%}")
            print(f"   Average processing time: {stats['average_processing_time']:.3f}s")
            print(f"   Requests per second: {stats['requests_per_second']:.2f}")
            
            if "enhancer_statistics" in stats and stats["enhancer_statistics"].get("status") != "no_enhancements_performed":
                enhancer_stats = stats["enhancer_statistics"]
                print(f"\n🔧 Enhancement Engine Statistics")
                print(f"   Total enhancements: {enhancer_stats['total_enhancements']}")
                print(f"   Cache hit rate: {enhancer_stats['cache_hit_rate']:.1%}")
                print(f"   Cache size: {enhancer_stats['cache_size']}")
    
    async def run(self) -> None:
        """Run the CLI application"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        try:
            if args.command == "enhance":
                await self.handle_enhance_command(args)
            elif args.command == "file":
                await self.handle_file_command(args)
            elif args.command == "batch":
                await self.handle_batch_command(args)
            elif args.command == "stats":
                self.handle_stats_command(args)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
        except Exception as e:
            print(f"Error: {e}")
            self.logger.error(f"CLI error: {e}")


# Export key classes
__all__ = [
    "TodoEnhancementRequest", "TodoEnhancementResponse", "TaskMasterTodoExtractor",
    "TodoEnhancementService", "TodoEnhancementCLI"
]


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run CLI
    cli = TodoEnhancementCLI()
    asyncio.run(cli.run())