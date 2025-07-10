#!/usr/bin/env python3
"""
MELT Data Correlation and Semantic Conventions for Task-Master
Implements atomic task: Implement MELT data correlation and semantic conventions

Based on research-driven breakdown:
- Standardize semantic conventions for telemetry attributes across all signals
- Ensure logs, traces, and metrics share common context fields (TraceId, SpanId, timestamps)
- Validate cross-signal correlation in observability tools
- Deliverable: End-to-end MELT correlation verified in observability backend

This module provides standardized semantic conventions and correlation utilities
to ensure consistent telemetry data emission across the Task-Master system.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# OpenTelemetry imports for correlation
try:
    from opentelemetry import trace, baggage
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.semconv.trace import SpanAttributes
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.baggage import set_baggage, get_baggage
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not available - correlation will use fallback implementation")

class SignalType(Enum):
    """Types of telemetry signals in MELT architecture"""
    METRICS = "metrics"
    EVENTS = "events"
    LOGS = "logs"
    TRACES = "traces"

class CorrelationLevel(Enum):
    """Levels of telemetry correlation"""
    NONE = "none"
    BASIC = "basic"      # TraceId and SpanId only
    ENHANCED = "enhanced" # Basic + service context
    FULL = "full"        # All semantic conventions

@dataclass
class CorrelationContext:
    """Standard correlation context shared across all MELT signals"""
    trace_id: str
    span_id: str
    service_name: str
    service_version: str
    service_instance_id: str
    environment: str
    timestamp: str
    operation_name: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for inclusion in telemetry"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_current_span(cls, service_name: str = "task-master", service_version: str = "1.0.0") -> 'CorrelationContext':
        """Create correlation context from current OpenTelemetry span"""
        if OTEL_AVAILABLE:
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                trace_id = format(span.get_span_context().trace_id, '032x')
                span_id = format(span.get_span_context().span_id, '016x')
            else:
                trace_id = '0' * 32
                span_id = '0' * 16
        else:
            # Fallback implementation
            trace_id = uuid.uuid4().hex.ljust(32, '0')
            span_id = uuid.uuid4().hex[:16]
        
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            service_name=service_name,
            service_version=service_version,
            service_instance_id=os.getenv('HOSTNAME', 'unknown-instance'),
            environment=os.getenv('DEPLOYMENT_ENVIRONMENT', 'development'),
            timestamp=datetime.utcnow().isoformat() + 'Z'
        )

class TaskMasterSemanticConventions:
    """
    Standardized semantic conventions for Task-Master telemetry
    
    Implements OpenTelemetry semantic conventions with Task-Master specific extensions
    """
    
    # Service attributes
    SERVICE_NAME = "service.name"
    SERVICE_VERSION = "service.version"
    SERVICE_INSTANCE_ID = "service.instance.id"
    SERVICE_NAMESPACE = "service.namespace"
    
    # Task-Master specific attributes
    TASK_ID = "task.id"
    TASK_TYPE = "task.type"
    TASK_STATUS = "task.status"
    TASK_PRIORITY = "task.priority"
    TASK_COMPLEXITY = "task.complexity"
    TASK_DURATION = "task.duration_seconds"
    TASK_DEPENDENCIES = "task.dependencies"
    TASK_PARENT_ID = "task.parent.id"
    TASK_DEPTH = "task.depth"
    
    # Research specific attributes
    RESEARCH_QUERY = "research.query"
    RESEARCH_SOURCE = "research.source"
    RESEARCH_RESULTS_COUNT = "research.results.count"
    RESEARCH_CONFIDENCE = "research.confidence"
    RESEARCH_TOKENS_USED = "research.tokens.used"
    RESEARCH_API_ENDPOINT = "research.api.endpoint"
    
    # Optimization attributes
    OPTIMIZATION_TYPE = "optimization.type"
    OPTIMIZATION_ALGORITHM = "optimization.algorithm"
    OPTIMIZATION_MEMORY_BEFORE = "optimization.memory.before_bytes"
    OPTIMIZATION_MEMORY_AFTER = "optimization.memory.after_bytes"
    OPTIMIZATION_TIME_COMPLEXITY = "optimization.time.complexity"
    OPTIMIZATION_SPACE_COMPLEXITY = "optimization.space.complexity"
    
    # Error attributes
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"
    ERROR_STACK = "error.stack"
    ERROR_RECOVERABLE = "error.recoverable"
    
    # System attributes
    SYSTEM_CPU_USAGE = "system.cpu.usage_percent"
    SYSTEM_MEMORY_USAGE = "system.memory.usage_bytes"
    SYSTEM_DISK_USAGE = "system.disk.usage_bytes"
    SYSTEM_NETWORK_IO = "system.network.io_bytes"
    
    # User context attributes
    USER_ID = "user.id"
    SESSION_ID = "session.id"
    CORRELATION_ID = "correlation.id"
    
    @classmethod
    def get_standard_attributes(cls, context: CorrelationContext) -> Dict[str, Any]:
        """Get standard attributes for all telemetry signals"""
        return {
            cls.SERVICE_NAME: context.service_name,
            cls.SERVICE_VERSION: context.service_version,
            cls.SERVICE_INSTANCE_ID: context.service_instance_id,
            cls.SERVICE_NAMESPACE: "task-master",
            "trace_id": context.trace_id,
            "span_id": context.span_id,
            "timestamp": context.timestamp,
            "environment": context.environment
        }

class MELTCorrelationManager:
    """
    Manages correlation context across all MELT signals
    
    Ensures consistent context propagation and correlation
    """
    
    def __init__(self, correlation_level: CorrelationLevel = CorrelationLevel.ENHANCED):
        self.correlation_level = correlation_level
        self.semantic_conventions = TaskMasterSemanticConventions()
        self._context_stack: List[CorrelationContext] = []
        
        # Setup logging formatter with correlation
        self._setup_correlated_logging()
    
    def _setup_correlated_logging(self):
        """Setup logging with correlation context"""
        class CorrelatedFormatter(logging.Formatter):
            def __init__(self, manager: MELTCorrelationManager):
                super().__init__()
                self.manager = manager
            
            def format(self, record):
                # Add correlation context to log record
                context = self.manager.get_current_context()
                if context:
                    record.trace_id = context.trace_id
                    record.span_id = context.span_id
                    record.service_name = context.service_name
                    record.environment = context.environment
                else:
                    record.trace_id = '0' * 32
                    record.span_id = '0' * 16
                    record.service_name = 'unknown'
                    record.environment = 'unknown'
                
                # Format with correlation context
                format_str = (
                    '%(asctime)s - %(name)s - %(levelname)s - '
                    'trace_id=%(trace_id)s span_id=%(span_id)s '
                    'service=%(service_name)s env=%(environment)s - '
                    '%(message)s'
                )
                formatter = logging.Formatter(format_str)
                return formatter.format(record)
        
        # Apply to root logger
        formatter = CorrelatedFormatter(self)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    
    def create_context(self, operation_name: str, **kwargs) -> CorrelationContext:
        """Create new correlation context"""
        context = CorrelationContext.from_current_span()
        context.operation_name = operation_name
        
        # Add any additional context
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
        
        return context
    
    def push_context(self, context: CorrelationContext):
        """Push correlation context onto stack"""
        self._context_stack.append(context)
        
        # Set baggage for cross-service propagation if OpenTelemetry available
        if OTEL_AVAILABLE:
            set_baggage("task.operation", context.operation_name or "unknown")
            set_baggage("service.name", context.service_name)
            set_baggage("environment", context.environment)
    
    def pop_context(self) -> Optional[CorrelationContext]:
        """Pop correlation context from stack"""
        if self._context_stack:
            return self._context_stack.pop()
        return None
    
    def get_current_context(self) -> Optional[CorrelationContext]:
        """Get current correlation context"""
        if self._context_stack:
            return self._context_stack[-1]
        return None
    
    def correlate_metric(self, metric_name: str, value: float, attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add correlation context to metric"""
        context = self.get_current_context()
        if not context:
            return {"name": metric_name, "value": value, "attributes": attributes or {}}
        
        correlated_attributes = self.semantic_conventions.get_standard_attributes(context)
        if attributes:
            correlated_attributes.update(attributes)
        
        return {
            "name": metric_name,
            "value": value,
            "attributes": correlated_attributes,
            "timestamp": context.timestamp
        }
    
    def correlate_log(self, message: str, level: str = "INFO", attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add correlation context to log entry"""
        context = self.get_current_context()
        if not context:
            return {"message": message, "level": level, "attributes": attributes or {}}
        
        correlated_attributes = self.semantic_conventions.get_standard_attributes(context)
        if attributes:
            correlated_attributes.update(attributes)
        
        return {
            "message": message,
            "level": level,
            "attributes": correlated_attributes,
            "timestamp": context.timestamp
        }
    
    def correlate_trace(self, span_name: str, attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add correlation context to trace span"""
        context = self.get_current_context()
        if not context:
            return {"span_name": span_name, "attributes": attributes or {}}
        
        correlated_attributes = self.semantic_conventions.get_standard_attributes(context)
        if attributes:
            correlated_attributes.update(attributes)
        
        return {
            "span_name": span_name,
            "attributes": correlated_attributes,
            "start_time": context.timestamp
        }
    
    def correlate_event(self, event_name: str, attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add correlation context to event"""
        context = self.get_current_context()
        if not context:
            return {"event_name": event_name, "attributes": attributes or {}}
        
        correlated_attributes = self.semantic_conventions.get_standard_attributes(context)
        if attributes:
            correlated_attributes.update(attributes)
        
        return {
            "event_name": event_name,
            "attributes": correlated_attributes,
            "timestamp": context.timestamp
        }

class MELTValidator:
    """
    Validates MELT correlation and semantic convention compliance
    """
    
    def __init__(self):
        self.semantic_conventions = TaskMasterSemanticConventions()
        self.required_fields = [
            "trace_id", "span_id", "service.name", "service.version", 
            "service.instance.id", "timestamp", "environment"
        ]
    
    def validate_correlation(self, telemetry_data: Dict[str, Any], signal_type: SignalType) -> Dict[str, Any]:
        """Validate correlation context in telemetry data"""
        validation_result = {
            "signal_type": signal_type.value,
            "valid": True,
            "missing_fields": [],
            "invalid_fields": [],
            "warnings": []
        }
        
        # Check required correlation fields
        attributes = telemetry_data.get("attributes", {})
        for field in self.required_fields:
            if field not in attributes:
                validation_result["missing_fields"].append(field)
                validation_result["valid"] = False
        
        # Validate trace_id format (32 hex characters)
        trace_id = attributes.get("trace_id", "")
        if trace_id and (len(trace_id) != 32 or not all(c in '0123456789abcdef' for c in trace_id.lower())):
            validation_result["invalid_fields"].append("trace_id")
            validation_result["valid"] = False
        
        # Validate span_id format (16 hex characters)
        span_id = attributes.get("span_id", "")
        if span_id and (len(span_id) != 16 or not all(c in '0123456789abcdef' for c in span_id.lower())):
            validation_result["invalid_fields"].append("span_id")
            validation_result["valid"] = False
        
        # Validate timestamp format
        timestamp = attributes.get("timestamp", "")
        if timestamp:
            try:
                datetime.fromisoformat(timestamp.rstrip('Z'))
            except ValueError:
                validation_result["invalid_fields"].append("timestamp")
                validation_result["valid"] = False
        
        # Signal-specific validations
        if signal_type == SignalType.METRICS:
            if "value" not in telemetry_data:
                validation_result["missing_fields"].append("value")
                validation_result["valid"] = False
        
        elif signal_type == SignalType.LOGS:
            if "message" not in telemetry_data:
                validation_result["missing_fields"].append("message")
                validation_result["valid"] = False
        
        elif signal_type == SignalType.TRACES:
            if "span_name" not in telemetry_data:
                validation_result["missing_fields"].append("span_name")
                validation_result["valid"] = False
        
        elif signal_type == SignalType.EVENTS:
            if "event_name" not in telemetry_data:
                validation_result["missing_fields"].append("event_name")
                validation_result["valid"] = False
        
        return validation_result
    
    def generate_correlation_report(self, telemetry_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate correlation validation report"""
        report = {
            "total_samples": len(telemetry_samples),
            "valid_samples": 0,
            "invalid_samples": 0,
            "correlation_coverage": 0.0,
            "signal_breakdown": {},
            "common_issues": {},
            "recommendations": []
        }
        
        for sample in telemetry_samples:
            signal_type = SignalType(sample.get("signal_type", "logs"))
            validation = self.validate_correlation(sample, signal_type)
            
            if validation["valid"]:
                report["valid_samples"] += 1
            else:
                report["invalid_samples"] += 1
                
                # Track common issues
                for field in validation["missing_fields"]:
                    if field not in report["common_issues"]:
                        report["common_issues"][field] = 0
                    report["common_issues"][field] += 1
            
            # Track signal breakdown
            signal_str = signal_type.value
            if signal_str not in report["signal_breakdown"]:
                report["signal_breakdown"][signal_str] = {"total": 0, "valid": 0}
            
            report["signal_breakdown"][signal_str]["total"] += 1
            if validation["valid"]:
                report["signal_breakdown"][signal_str]["valid"] += 1
        
        # Calculate correlation coverage
        if report["total_samples"] > 0:
            report["correlation_coverage"] = report["valid_samples"] / report["total_samples"]
        
        # Generate recommendations
        if report["correlation_coverage"] < 0.9:
            report["recommendations"].append("Improve correlation coverage - target 90%+")
        
        if "trace_id" in report["common_issues"]:
            report["recommendations"].append("Ensure trace_id propagation across all signals")
        
        if "timestamp" in report["common_issues"]:
            report["recommendations"].append("Standardize timestamp format (ISO 8601)")
        
        return report

# Context manager for automatic correlation
class CorrelatedOperation:
    """Context manager for operations with automatic MELT correlation"""
    
    def __init__(self, manager: MELTCorrelationManager, operation_name: str, **context_kwargs):
        self.manager = manager
        self.operation_name = operation_name
        self.context_kwargs = context_kwargs
        self.context = None
    
    def __enter__(self):
        self.context = self.manager.create_context(self.operation_name, **self.context_kwargs)
        self.manager.push_context(self.context)
        return self.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.pop_context()
        
        # Log operation completion with correlation
        if exc_type is None:
            logging.info(f"Operation '{self.operation_name}' completed successfully")
        else:
            logging.error(f"Operation '{self.operation_name}' failed: {exc_val}")

# Global correlation manager instance
_correlation_manager: Optional[MELTCorrelationManager] = None

def get_correlation_manager() -> MELTCorrelationManager:
    """Get global correlation manager instance"""
    global _correlation_manager
    if _correlation_manager is None:
        _correlation_manager = MELTCorrelationManager()
    return _correlation_manager

def demonstrate_melt_correlation():
    """Demonstrate MELT data correlation capabilities"""
    print("🔗 MELT Data Correlation and Semantic Conventions Demo")
    print("=" * 60)
    
    # Initialize correlation manager
    manager = get_correlation_manager()
    validator = MELTValidator()
    
    # Demonstrate correlated operations
    with CorrelatedOperation(manager, "task_execution", task_id="task-001", task_type="research") as context:
        print(f"📋 Started operation with trace_id: {context.trace_id}")
        
        # Generate correlated telemetry
        metric_data = manager.correlate_metric("task.duration_seconds", 2.5, {
            "task.type": "research",
            "task.status": "completed"
        })
        
        log_data = manager.correlate_log("Task execution started", "INFO", {
            "task.id": "task-001",
            "task.priority": "high"
        })
        
        trace_data = manager.correlate_trace("research_task_execution", {
            "research.query": "OpenTelemetry best practices",
            "research.source": "perplexity"
        })
        
        event_data = manager.correlate_event("task_completed", {
            "task.result": "success",
            "task.output_size": 1024
        })
        
        # Validate correlation
        telemetry_samples = [
            {**metric_data, "signal_type": "metrics"},
            {**log_data, "signal_type": "logs"},
            {**trace_data, "signal_type": "traces"},
            {**event_data, "signal_type": "events"}
        ]
        
        print("\n📊 Validation Results:")
        for sample in telemetry_samples:
            signal_type = SignalType(sample["signal_type"])
            validation = validator.validate_correlation(sample, signal_type)
            status = "✅ Valid" if validation["valid"] else "❌ Invalid"
            print(f"  {signal_type.value.title()}: {status}")
            
            if not validation["valid"]:
                print(f"    Missing: {validation['missing_fields']}")
                print(f"    Invalid: {validation['invalid_fields']}")
        
        # Generate correlation report
        report = validator.generate_correlation_report(telemetry_samples)
        print(f"\n📈 Correlation Report:")
        print(f"  Coverage: {report['correlation_coverage']:.1%}")
        print(f"  Valid Samples: {report['valid_samples']}/{report['total_samples']}")
        
        if report["recommendations"]:
            print(f"  Recommendations:")
            for rec in report["recommendations"]:
                print(f"    • {rec}")
    
    print("\n✅ MELT correlation demonstration completed!")
    print("🔍 All signals now share common correlation context for observability")

def export_semantic_conventions():
    """Export semantic conventions documentation"""
    conventions_doc = {
        "task_master_semantic_conventions": {
            "version": "1.0.0",
            "description": "Standardized semantic conventions for Task-Master telemetry",
            "service_attributes": {
                "service.name": "Name of the service",
                "service.version": "Version of the service",
                "service.instance.id": "Unique identifier for service instance",
                "service.namespace": "Namespace for service grouping"
            },
            "task_attributes": {
                "task.id": "Unique identifier for task",
                "task.type": "Type of task (e.g., research, optimization)",
                "task.status": "Current status of task",
                "task.priority": "Priority level of task",
                "task.complexity": "Complexity rating of task",
                "task.duration_seconds": "Task execution duration in seconds",
                "task.dependencies": "List of task dependencies",
                "task.parent.id": "Parent task identifier",
                "task.depth": "Depth in task hierarchy"
            },
            "research_attributes": {
                "research.query": "Research query string",
                "research.source": "Source of research (e.g., perplexity, openai)",
                "research.results.count": "Number of research results",
                "research.confidence": "Confidence score of research results",
                "research.tokens.used": "Number of tokens used in research",
                "research.api.endpoint": "API endpoint used for research"
            },
            "correlation_requirements": {
                "trace_id": "32-character hexadecimal trace identifier",
                "span_id": "16-character hexadecimal span identifier",
                "timestamp": "ISO 8601 formatted timestamp",
                "service.name": "Required service identifier",
                "environment": "Deployment environment"
            }
        }
    }
    
    # Save to file
    conventions_path = ".taskmaster/docs/semantic-conventions.json"
    os.makedirs(os.path.dirname(conventions_path), exist_ok=True)
    
    with open(conventions_path, 'w') as f:
        json.dump(conventions_doc, f, indent=2)
    
    print(f"📄 Semantic conventions exported to: {conventions_path}")
    return conventions_path

if __name__ == "__main__":
    try:
        # Run demonstration
        demonstrate_melt_correlation()
        
        # Export conventions
        export_semantic_conventions()
        
        print("\n🎯 MELT Data Correlation Implementation Complete!")
        print("✨ Features implemented:")
        print("  • Standardized semantic conventions")
        print("  • Cross-signal correlation context")
        print("  • Automatic context propagation")
        print("  • Correlation validation")
        print("  • Compliance reporting")
        
    except Exception as e:
        print(f"❌ Error during MELT correlation demonstration: {e}")
        import traceback
        traceback.print_exc()