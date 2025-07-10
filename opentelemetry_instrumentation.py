#!/usr/bin/env python3
"""
OpenTelemetry Instrumentation for Task-Master System
Implements MELT observability with trace context propagation

Based on research findings from atomic task breakdown:
- Instrument all critical services and components
- Propagate trace context through service boundaries
- Include trace context in logs for correlation
- Ensure OpenTelemetry-compliant telemetry data emission
"""

import os
import logging
import time
import functools
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# OpenTelemetry core imports
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource

# Semantic conventions for standardized attributes
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.semconv.trace import SpanAttributes

# Instrumentation libraries for automatic instrumentation
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

# Propagation for trace context
from opentelemetry import propagate
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.propagators.textmap import DefaultTextMapPropagator

class TaskMasterTelemetry:
    """
    Central telemetry configuration for Task-Master system
    Implements OpenTelemetry instrumentation with MELT correlation
    """
    
    def __init__(
        self,
        service_name: str = "task-master",
        service_version: str = "1.0.0",
        environment: str = "development",
        otlp_endpoint: str = "http://localhost:4317"
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.otlp_endpoint = otlp_endpoint
        
        # Initialize telemetry components
        self.resource = self._create_resource()
        self.tracer_provider = None
        self.meter_provider = None
        self.tracer = None
        self.meter = None
        
        # Setup instrumentation
        self._setup_telemetry()
        self._setup_automatic_instrumentation()
        self._setup_logging_integration()
        
        logging.info(f"OpenTelemetry instrumentation initialized for {service_name}")
    
    def _create_resource(self) -> Resource:
        """Create resource with semantic conventions"""
        return Resource.create({
            SERVICE_NAME: self.service_name,
            SERVICE_VERSION: self.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.environment,
            ResourceAttributes.SERVICE_NAMESPACE: "task-master",
            "task.master.component": "core",
        })
    
    def _setup_telemetry(self):
        """Setup OpenTelemetry providers and exporters"""
        
        # Setup tracing
        self.tracer_provider = TracerProvider(resource=self.resource)
        
        # OTLP exporter for traces
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.otlp_endpoint,
            insecure=True  # Use secure=False for development
        )
        
        # Batch processor for performance
        span_processor = BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            export_timeout_millis=30000,
            schedule_delay_millis=5000
        )
        
        self.tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(self.tracer_provider)
        
        self.tracer = trace.get_tracer(__name__)
        
        # Setup metrics
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=self.otlp_endpoint,
                insecure=True
            ),
            export_interval_millis=10000
        )
        
        self.meter_provider = MeterProvider(
            resource=self.resource,
            metric_readers=[metric_reader]
        )
        
        metrics.set_meter_provider(self.meter_provider)
        self.meter = metrics.get_meter(__name__)
        
        # Setup propagation
        propagate.set_global_textmap(B3MultiFormat())
    
    def _setup_automatic_instrumentation(self):
        """Setup automatic instrumentation for common libraries"""
        
        # Instrument HTTP requests
        RequestsInstrumentor().instrument()
        URLLib3Instrumentor().instrument()
        
        # Add more instrumentations as needed
        # For example: FlaskInstrumentor, FastAPIInstrumentor, etc.
        
        logging.info("Automatic instrumentation enabled for HTTP libraries")
    
    def _setup_logging_integration(self):
        """Setup logging integration with trace context"""
        
        # Instrument logging to include trace context
        LoggingInstrumentor().instrument(set_logging_format=True)
        
        # Custom formatter to include trace context
        class TraceContextFormatter(logging.Formatter):
            def format(self, record):
                # Get current span context
                span = trace.get_current_span()
                if span and span.get_span_context().is_valid:
                    trace_id = format(span.get_span_context().trace_id, '032x')
                    span_id = format(span.get_span_context().span_id, '016x')
                    record.trace_id = trace_id
                    record.span_id = span_id
                else:
                    record.trace_id = '0' * 32
                    record.span_id = '0' * 16
                
                return super().format(record)
        
        # Configure root logger with trace context
        formatter = TraceContextFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            'trace_id=%(trace_id)s span_id=%(span_id)s - %(message)s'
        )
        
        # Apply to console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)
        
        logging.info("Logging integration with trace context enabled")

class TaskMasterInstrumentation:
    """
    Task-Master specific instrumentation decorators and utilities
    """
    
    def __init__(self, telemetry: TaskMasterTelemetry):
        self.telemetry = telemetry
        self.tracer = telemetry.tracer
        self.meter = telemetry.meter
        
        # Create custom metrics
        self.task_counter = self.meter.create_counter(
            "task_master_tasks_total",
            description="Total number of tasks processed",
            unit="1"
        )
        
        self.task_duration = self.meter.create_histogram(
            "task_master_task_duration_seconds",
            description="Task execution duration",
            unit="s"
        )
        
        self.error_counter = self.meter.create_counter(
            "task_master_errors_total",
            description="Total number of errors",
            unit="1"
        )
    
    def trace_task_execution(
        self,
        task_type: str,
        task_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Decorator to trace task execution with custom attributes"""
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                
                # Create span with semantic attributes
                span_attributes = {
                    "task.type": task_type,
                    "task.function": func.__name__,
                    "task.module": func.__module__,
                }
                
                if task_id:
                    span_attributes["task.id"] = task_id
                
                if attributes:
                    span_attributes.update(attributes)
                
                with self.tracer.start_as_current_span(
                    f"task_execution.{task_type}",
                    attributes=span_attributes
                ) as span:
                    
                    start_time = time.time()
                    
                    try:
                        # Log task start with trace context
                        logging.info(f"Starting task execution: {task_type}")
                        
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        # Record success metrics
                        duration = time.time() - start_time
                        
                        self.task_counter.add(1, {
                            "task_type": task_type,
                            "status": "success"
                        })
                        
                        self.task_duration.record(duration, {
                            "task_type": task_type,
                            "status": "success"
                        })
                        
                        span.set_attribute("task.status", "success")
                        span.set_attribute("task.duration_seconds", duration)
                        
                        logging.info(f"Task execution completed: {task_type} in {duration:.3f}s")
                        
                        return result
                        
                    except Exception as e:
                        # Record error metrics
                        duration = time.time() - start_time
                        
                        self.error_counter.add(1, {
                            "task_type": task_type,
                            "error_type": type(e).__name__
                        })
                        
                        self.task_duration.record(duration, {
                            "task_type": task_type,
                            "status": "error"
                        })
                        
                        # Add error details to span
                        span.record_exception(e)
                        span.set_attribute("task.status", "error")
                        span.set_attribute("task.error", str(e))
                        span.set_attribute("task.duration_seconds", duration)
                        
                        logging.error(f"Task execution failed: {task_type} after {duration:.3f}s - {e}")
                        
                        raise
            
            return wrapper
        return decorator
    
    def trace_api_call(self, api_name: str, operation: str):
        """Decorator to trace external API calls"""
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                
                with self.tracer.start_as_current_span(
                    f"api_call.{api_name}.{operation}",
                    attributes={
                        "api.name": api_name,
                        "api.operation": operation,
                        "api.function": func.__name__,
                        SpanAttributes.HTTP_METHOD: "POST"  # Assume POST for research calls
                    }
                ) as span:
                    
                    start_time = time.time()
                    
                    try:
                        result = func(*args, **kwargs)
                        
                        duration = time.time() - start_time
                        span.set_attribute("api.status", "success")
                        span.set_attribute("api.duration_seconds", duration)
                        
                        logging.info(f"API call successful: {api_name}.{operation} in {duration:.3f}s")
                        
                        return result
                        
                    except Exception as e:
                        duration = time.time() - start_time
                        span.record_exception(e)
                        span.set_attribute("api.status", "error")
                        span.set_attribute("api.duration_seconds", duration)
                        
                        logging.error(f"API call failed: {api_name}.{operation} after {duration:.3f}s - {e}")
                        
                        raise
            
            return wrapper
        return decorator
    
    def create_custom_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for creating custom spans"""
        return self.tracer.start_as_current_span(name, attributes=attributes)

# Global telemetry instance
_telemetry_instance: Optional[TaskMasterTelemetry] = None
_instrumentation_instance: Optional[TaskMasterInstrumentation] = None

def initialize_telemetry(
    service_name: str = "task-master",
    service_version: str = "1.0.0",
    environment: str = "development",
    otlp_endpoint: str = None
) -> TaskMasterTelemetry:
    """Initialize global telemetry instance"""
    global _telemetry_instance, _instrumentation_instance
    
    # Use environment variable for OTLP endpoint if not provided
    if otlp_endpoint is None:
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    
    _telemetry_instance = TaskMasterTelemetry(
        service_name=service_name,
        service_version=service_version,
        environment=environment,
        otlp_endpoint=otlp_endpoint
    )
    
    _instrumentation_instance = TaskMasterInstrumentation(_telemetry_instance)
    
    return _telemetry_instance

def get_instrumentation() -> TaskMasterInstrumentation:
    """Get global instrumentation instance"""
    if _instrumentation_instance is None:
        raise RuntimeError("Telemetry not initialized. Call initialize_telemetry() first.")
    return _instrumentation_instance

def get_tracer():
    """Get global tracer instance"""
    if _telemetry_instance is None:
        raise RuntimeError("Telemetry not initialized. Call initialize_telemetry() first.")
    return _telemetry_instance.tracer

def shutdown_telemetry():
    """Shutdown telemetry providers"""
    global _telemetry_instance
    
    if _telemetry_instance:
        if _telemetry_instance.tracer_provider:
            _telemetry_instance.tracer_provider.shutdown()
        if _telemetry_instance.meter_provider:
            _telemetry_instance.meter_provider.shutdown()
        
        logging.info("OpenTelemetry telemetry shut down")

# Example usage and demonstrations
def demonstrate_instrumentation():
    """Demonstrate OpenTelemetry instrumentation capabilities"""
    
    # Initialize telemetry
    telemetry = initialize_telemetry(
        service_name="task-master-demo",
        environment="development"
    )
    
    instrumentation = get_instrumentation()
    
    # Example: Instrumented task execution
    @instrumentation.trace_task_execution(
        task_type="research",
        attributes={"research.source": "perplexity", "research.topic": "opentelemetry"}
    )
    def perform_research_task(query: str) -> Dict[str, Any]:
        """Example research task with instrumentation"""
        
        # Simulate research API call
        @instrumentation.trace_api_call("perplexity", "research")
        def call_research_api(query: str) -> str:
            import time
            time.sleep(0.1)  # Simulate API latency
            return f"Research results for: {query}"
        
        # Perform research with automatic tracing
        results = call_research_api(query)
        
        # Add custom span for processing
        with instrumentation.create_custom_span(
            "process_research_results",
            attributes={"results.length": len(results)}
        ):
            # Simulate processing
            time.sleep(0.05)
            processed_results = {
                "query": query,
                "results": results,
                "processed_at": datetime.now().isoformat(),
                "status": "completed"
            }
        
        return processed_results
    
    # Example: Instrumented task management
    @instrumentation.trace_task_execution(
        task_type="task_management",
        task_id="task-001"
    )
    def manage_task(task_id: str, action: str) -> bool:
        """Example task management with instrumentation"""
        
        logging.info(f"Managing task {task_id} with action {action}")
        
        # Simulate task operations
        with instrumentation.create_custom_span(
            f"task_action.{action}",
            attributes={"task.id": task_id, "action.type": action}
        ):
            time.sleep(0.02)  # Simulate work
            return True
    
    try:
        # Run demonstration
        print("🔍 OpenTelemetry Instrumentation Demonstration")
        print("=" * 50)
        
        # Execute instrumented functions
        research_result = perform_research_task("OpenTelemetry MELT correlation")
        print(f"Research completed: {research_result['status']}")
        
        task_result = manage_task("task-001", "update_status")
        print(f"Task management completed: {task_result}")
        
        # Demonstrate error handling
        @instrumentation.trace_task_execution(task_type="error_demo")
        def failing_task():
            raise ValueError("Demonstration error for tracing")
        
        try:
            failing_task()
        except ValueError as e:
            print(f"Error traced: {e}")
        
        print("\n✅ OpenTelemetry instrumentation demonstration completed")
        print("📊 Check your observability backend for traces, metrics, and correlated logs")
        
    finally:
        # Allow time for telemetry export
        time.sleep(2)
        shutdown_telemetry()

if __name__ == "__main__":
    # Check if OpenTelemetry libraries are available
    try:
        demonstrate_instrumentation()
    except ImportError as e:
        print(f"⚠️ OpenTelemetry libraries not installed: {e}")
        print("Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
        print("Auto-instrumentation: pip install opentelemetry-instrumentation-requests opentelemetry-instrumentation-urllib3 opentelemetry-instrumentation-logging")